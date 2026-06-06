"""Context Parallel pass: CPKind-aware shape splitting with structural
dimension inference, scope-based weight detection, and op-type filtering.

Design
------
Three orthogonal mechanisms determine which tensor dimensions to split:

1. **Scope classification** (from ``tensor_parallel.py`` pattern):
   Nodes are classified as attention-internal or general by scope keywords.

2. **Structural dimension rules** (from tensor rank):
   4D ``(B, H, S, D)`` → dim 1=heads, dims 2,3=seq
   3D ``(B, S, D)``    → dim 1=seq
   3D ``(H, Sq, Sk)``  → dim 0=heads, dims 1,2=seq  (bmm attention)
   2D ``(S, D)``       → dim 0=seq

3. **Weight detection** (from ``adapter.py::_is_param_node`` pattern):
   Scope-suffix matching + op-type position + shared-tensor heuristic
   + GroupedMatMul/mega_moe detection.

4. **Op-type filtering**:
   Memory-movement ops (index/gather/select/scatter), reduction ops
   (sum/mean/var), and shape ops (view/reshape/permute) are skipped.
"""
from __future__ import annotations

import logging
from math import prod
from typing import Dict, List, Set

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


# ── Scope classification (attention vs general) ─────────────────────────

_ATTN_SCOPE_KW = (
    "self_attn", "attention", "attn",
    "q_proj", "k_proj", "v_proj", "o_proj", "out_proj",
    "q_a_proj", "q_b_proj", "kv_a_proj", "kv_b_proj",
    "q_a_layernorm", "kv_a_layernorm",
    "rotary", "rope",
)

_KV_PROJ_SCOPE_KW = ("k_proj", "v_proj", "kv_a_proj", "kv_b_proj")


def _is_attn_scope(scope: str) -> bool:
    s = scope.lower()
    if "layernorm" in s or "rmsnorm" in s or "rms_norm" in s or "layer_norm" in s:
        return False
    return any(kw in s for kw in _ATTN_SCOPE_KW)


def _is_kv_proj_scope(scope: str) -> bool:
    return any(kw in scope.lower() for kw in _KV_PROJ_SCOPE_KW)


def _infer_layer_type_from_scope(scope: str) -> str:
    """Infer CP layer type (csa / hca / swa) from module scope.

    Used as a fallback when no ``TrainingConfig`` is available.  Mirrors
    the layer-type taxonomy of ``ModelSpec.get_layer_cp_type``:

    - scope contains ``swa`` / ``sliding`` → ``swa``
    - scope contains ``hca`` → ``hca``
    - scope contains ``sparse`` / ``csa`` / ``indexer`` → ``csa``
    - otherwise → ``csa`` (DeepSeek-V4 default)
    """
    s = scope.lower()
    if "swa" in s or "sliding" in s:
        return "swa"
    if "hca" in s:
        return "hca"
    if "sparse" in s or "csa" in s or "indexer" in s:
        return "csa"
    return "csa"


# ── Weight detection (scope suffix + op-type position + shared tensors) ─

_PARAM_SCOPE_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
    "q_a_proj", "kv_a_proj", "q_b_proj", "kv_b_proj",
    "shared_expert.gate_proj", "shared_expert.up_proj", "shared_expert.down_proj",
    "wq_a", "wkv", "wq_b", "wo_a", "wo_b",
    "w1", "w2", "w3",
    "weights_proj",
    "wgate", "comp_wgate",
    "fc1", "fc2",
)

_MATMUL_OPS = frozenset({
    "aten.mm.default", "aten.mm",
    "aten.addmm.default", "aten.addmm",
    "aten.linear.default", "aten.linear",
    "aten.bmm.default", "aten.bmm",
    "aten.matmul.default", "aten.matmul",
    "aten.baddbmm.default", "aten.baddbmm",
})

_EMBED_OPS = frozenset({
    "aten.embedding.default", "aten.embedding",
    "aten._convolution.default", "aten._convolution",
})

_GROUPED_OPS = frozenset({
    "GroupedMatMul", "grouped_matmul", "mega_moe",
})

_POSTPROCESS_EXCLUDE = frozenset({
    "aten.embedding.default", "aten.embedding",
    "aten.index.Tensor", "aten.index",
    "aten.index_select.default", "aten.index_select",
    "aten.gather.default", "aten.gather",
    "aten.scatter.src", "aten.scatter.default", "aten.scatter",
    "aten._convolution.default", "aten._convolution",
    "aten.cumsum.default", "aten.cumsum",
    "aten.cumprod.default", "aten.cumprod",
    "parallel_embedding", "embedding", "embedding_backward",
    "moe_dispatch", "npu_moe_dispatch",
})


def _is_weight_input(node: OpNode, inp_idx: int, shared_ids: Set[str]) -> bool:
    """Determine if the tensor at ``node.inputs[inp_idx]`` is a weight.

    Detection mechanisms (in priority order):
    1. Shared tensor: appears as input to ≥2 nodes → weight
    2. Embedding/conv: input[0] is always weight table
    3. GroupedMatMul/mega_moe: input[1] is weight
    4. Matmul + scope suffix + position (from adapter.py)
    5. is_param annotation (set by stitch_fwd_bwd)
    """
    if inp_idx >= len(node.inputs):
        return False
    tensor_id = node.inputs[inp_idx].id

    if tensor_id in shared_ids:
        return True

    if node.op_type in _EMBED_OPS:
        return inp_idx == 0

    if node.op_type in _GROUPED_OPS:
        return inp_idx == 1

    if node.op_type in _MATMUL_OPS:
        scope = (node.scope or "").lower().rstrip(".")
        if any(scope.endswith(s) for s in _PARAM_SCOPE_SUFFIXES):
            op = node.op_type.split(".")[1] if "." in node.op_type else node.op_type
            if op in ("mm", "matmul", "linear"):
                return inp_idx == 1
            if op in ("addmm", "baddbmm"):
                return inp_idx == 2

    if node.annotations.get("is_param") and inp_idx > 0:
        return True

    return False


# ── Op-type filtering (skip non-splittable ops) ─────────────────────────

_SKIP_OPS = frozenset({
    "aten.index.Tensor", "aten.index",
    "aten.gather.default", "aten.gather",
    "aten.scatter.src", "aten.scatter.default", "aten.scatter",
    "aten.scatter_add.default", "aten.scatter_add",
    "aten.index_select.default", "aten.index_select",
    "aten.select.int", "aten.select",
    "aten.embedding.default", "aten.embedding",
    "aten._convolution.default", "aten._convolution",
    "aten.sum.dim_IntList", "aten.sum.default", "aten.sum",
    "aten.mean.dim", "aten.mean.default", "aten.mean",
    "aten.var.correction", "aten.var.default", "aten.var",
    "aten.std.correction", "aten.std.default", "aten.std",
    "aten.argmax.default", "aten.argmax",
    "aten.argmin.default", "aten.argmin",
    "aten.max.dim", "aten.max.default", "aten.max",
    "aten.min.dim", "aten.min.default", "aten.min",
    "aten.amax.default", "aten.amax",
    "aten.amin.default", "aten.amin",
    "aten.nonzero.default", "aten.nonzero",
    "aten.topk.default", "aten.topk",
    "aten.sort.default", "aten.sort",
    "aten.argsort.default", "aten.argsort",
    "aten.cumsum.default", "aten.cumsum",
    "aten.cumprod.default", "aten.cumprod",
    "moe_dispatch", "npu_moe_dispatch",
    "parallel_embedding",
    "embedding", "embedding_backward",
})


def _should_skip_node(node: OpNode) -> bool:
    """Check if node should be skipped entirely (no splitting)."""
    if node.op_type in _SKIP_OPS:
        return True
    if node.category in ("communication", "memory"):
        return True
    return False


def _build_shared_tensor_ids(graph: OpGraph) -> Set[str]:
    """Build set of tensor IDs that are likely shared weights.

    Very conservative heuristic: only mark tensors as shared weights if they:
    1. Are NOT outputs of any node (external parameters)
    2. Appear as inputs to ≥3 nodes (highly shared, unlikely for activations)
    3. Have 2D shape with both dims > 100 (typical weight matrix shape)

    This avoids false positives on:
    - Activations used by 2 nodes (common in residual/fork patterns)
    - Initial input tensors (created externally, used by first layer ops)
    - 3D+ tensors (handled by structural rules)
    """
    output_ids: Set[str] = set()
    for node in graph.nodes.values():
        for t in node.outputs:
            output_ids.add(t.id)

    tensor_use_count: Dict[str, int] = {}
    tensor_shapes: Dict[str, tuple] = {}
    for node in graph.nodes.values():
        for t in node.inputs:
            tensor_use_count[t.id] = tensor_use_count.get(t.id, 0) + 1
            tensor_shapes[t.id] = t.shape

    shared_ids: Set[str] = set()
    for tid, count in tensor_use_count.items():
        if tid in output_ids:
            continue
        if count < 3:
            continue
        shape = tensor_shapes.get(tid, ())
        if len(shape) != 2:
            continue
        if shape[0] <= 100 or shape[1] <= 100:
            continue
        shared_ids.add(tid)

    return shared_ids


def _has_splittable_tensor(
    node: OpNode, shared_ids: Set[str],
) -> bool:
    """Check if node has any non-weight tensor eligible for CP splitting."""
    for i, t in enumerate(node.inputs):
        if not _is_weight_input(node, i, shared_ids) and len(t.shape) >= 2:
            return True
    for t in node.outputs:
        if len(t.shape) >= 2:
            return True
    return False


class ContextParallelPass(GraphPass):
    """CPKind-aware Context Parallel pass with structural dimension inference.

    Dimensions are determined by tensor rank + scope semantics + weight
    detection + op-type filtering.  No hardcoded ``seq_len`` value is
    used for dimension identification or validation.
    """

    name = "context_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        """Run context parallel transformation."""
        if ctx.parallel.cp <= 1:
            return graph

        g = graph.clone()
        cp = ctx.parallel.cp
        cp_kind = ctx.training.resolve_cp_kind(ctx.model_id, cp) if ctx.training else "none"
        num_heads = ctx.training.num_heads if ctx.training else 0
        num_kv_heads = ctx.training.num_kv_heads if ctx.training else 0
        
        # Calculate TP-adjusted head dimensions
        tp = ctx.parallel.tp if ctx.parallel else 1
        num_heads_tp = num_heads // tp if num_heads > 0 else 0
        num_kv_heads_tp = num_kv_heads // tp if num_kv_heads > 0 else 0
        
        # Get CP factors for hybrid
        cp_ulysses = ctx.parallel.cp_ulysses if ctx.parallel.cp_ulysses else cp
        cp_ring = ctx.parallel.cp_ring if ctx.parallel.cp_ring else cp

        # Build shared tensor IDs
        shared_ids = _build_shared_tensor_ids(g)

        # Classify nodes
        attn_nodes, general_nodes = self._classify_nodes(g, shared_ids)

        tensor_map: Dict[str, TensorMeta] = {}
        split_nodes: List[OpNode] = []

        for node in general_nodes:
            changed = self._apply_seq_split(g, node, cp, shared_ids, tensor_map)
            if changed:
                split_nodes.append(node)

        for node in attn_nodes:
            changed = self._apply_attn_split(
                g, node, cp_kind, cp,
                cp_ulysses, cp_ring,
                num_heads_tp, num_kv_heads_tp,
                _is_kv_proj_scope(node.scope or ""),
                shared_ids, tensor_map,
            )
            if changed:
                split_nodes.append(node)

        for node in split_nodes:
            ann = {"kind": cp_kind, "cp": cp}
            if cp_kind == "hybrid":
                ann["cp_ulysses"] = cp_ulysses
                ann["cp_ring"] = cp_ring
            if cp_kind == "compressed":
                try:
                    layer_id = int(node.layer) if node.layer else -1
                except (ValueError, TypeError):
                    layer_id = -1
                if layer_id >= 0 and ctx.training is not None:
                    ann["layer_type"] = ctx.training.resolve_layer_cp_type(layer_id)
                else:
                    ann["layer_type"] = _infer_layer_type_from_scope(node.scope or "")
            node.annotations["cp_split"] = ann

        for edge in g.edges:
            if edge.tensor and edge.tensor.id in tensor_map:
                edge.tensor = tensor_map[edge.tensor.id]

        for node in g.nodes.values():
            for i, t in enumerate(node.inputs):
                if t.id in tensor_map:
                    node.inputs[i] = tensor_map[t.id]
            for i, t in enumerate(node.outputs):
                if t.id in tensor_map:
                    node.outputs[i] = tensor_map[t.id]

        seq_len = ctx.training.seq_len if ctx.training else 0
        if seq_len > 0:
            self._postprocess_remaining_seq_tensors(g, seq_len, cp, tensor_map)

        logger.info(
            "ContextParallelPass: cp=%s kind=%s | "
            "attn_nodes=%d general_nodes=%d split_nodes=%d",
            cp, cp_kind, len(attn_nodes), len(general_nodes), len(split_nodes),
        )

        return g

    # ── node classification ──────────────────────────────────────────────

    def _classify_nodes(
        self, graph: OpGraph, shared_ids: Set[str],
    ) -> tuple[List[OpNode], List[OpNode]]:
        """Classify nodes by scope + splittable tensor presence.

        Skips:
        - Communication/memory nodes
        - Index/gather/select/scatter ops
        - Reduction ops (sum/mean/var)
        - Embedding/conv ops (weight table should not be split)
        - Nodes with no splittable tensors
        """
        attn_nodes: List[OpNode] = []
        general_nodes: List[OpNode] = []

        for node in graph.topo_sort():
            if _should_skip_node(node):
                continue
            if not _has_splittable_tensor(node, shared_ids):
                continue
            if node.scope and _is_attn_scope(node.scope):
                attn_nodes.append(node)
            else:
                general_nodes.append(node)

        return attn_nodes, general_nodes

    # ── seq-split (non-attention nodes) ──────────────────────────────────

    def _apply_seq_split(
        self,
        graph: OpGraph,
        node: OpNode,
        cp: int,
        shared_ids: Set[str],
        tensor_map: Dict[str, TensorMeta],
    ) -> bool:
        changed = False

        new_inputs = []
        for i, t in enumerate(node.inputs):
            if _is_weight_input(node, i, shared_ids):
                new_inputs.append(t)
                continue
            nt = self._split_seq_structural(t, cp, tensor_map)
            new_inputs.append(nt)
            if nt is not t:
                changed = True

        new_outputs = []
        for t in node.outputs:
            nt = self._split_seq_structural(t, cp, tensor_map)
            new_outputs.append(nt)
            if nt is not t:
                changed = True

        if changed:
            graph.nodes[node.id] = self._rebuild_node(
                node, new_inputs, new_outputs,
            )
        return changed

    # ── attention-internal split ─────────────────────────────────────────

    def _apply_attn_split(
        self,
        graph: OpGraph,
        node: OpNode,
        cp_kind: str,
        cp: int,
        cp_ulysses: int,
        cp_ring: int,
        num_heads_tp: int,
        num_kv_heads_tp: int,
        is_kv_proj: bool,
        shared_ids: Set[str],
        tensor_map: Dict[str, TensorMeta],
    ) -> bool:
        if cp_kind == "ulysses":
            heads_factor = cp
            seq_factor = None
        elif cp_kind == "hybrid":
            heads_factor = cp_ulysses
            seq_factor = cp_ring
        elif cp_kind in ("ring", "compressed"):
            heads_factor = None
            seq_factor = cp
        else:
            heads_factor = None
            seq_factor = cp

        heads_dim = num_kv_heads_tp if is_kv_proj else num_heads_tp

        if heads_factor and heads_dim == 0:
            seq_factor = heads_factor
            heads_factor = None

        changed = False
        new_inputs = []
        for i, t in enumerate(node.inputs):
            if _is_weight_input(node, i, shared_ids):
                new_inputs.append(t)
                continue
            nt = self._split_attn_tensor(
                t, seq_factor, heads_dim, heads_factor, tensor_map,
            )
            new_inputs.append(nt)
            if nt is not t:
                changed = True

        new_outputs = []
        for t in node.outputs:
            nt = self._split_attn_tensor(
                t, seq_factor, heads_dim, heads_factor, tensor_map,
            )
            new_outputs.append(nt)
            if nt is not t:
                changed = True

        if changed:
            graph.nodes[node.id] = self._rebuild_node(
                node, new_inputs, new_outputs,
            )
        return changed

    # ── tensor-level splitting helpers ───────────────────────────────────

    @staticmethod
    def _split_seq_structural(
        tensor: TensorMeta,
        cp: int,
        tensor_map: Dict[str, TensorMeta],
    ) -> TensorMeta:
        """Split seq dimension using structural rules.

        3D+ ``(B, S, ...)``: dim 1 is seq → split.
        2D ``(S, D)``:       dim 0 is seq → split.
        1D / scalar:         skip.
        """
        if tensor.id in tensor_map:
            return tensor_map[tensor.id]

        shape = tensor.shape
        rank = len(shape)
        if rank < 2:
            return tensor

        new_shape = list(shape)
        if rank == 2:
            # For 2D tensors, always split dim 0 (standard layout: (S, D))
            new_shape[0] = shape[0] // cp
        else:
            new_shape[1] = shape[1] // cp

        return _make_tensor(tensor, tuple(new_shape), tensor_map)

    @staticmethod
    def _split_attn_tensor(
        tensor: TensorMeta,
        seq_factor: int | None,
        heads_dim: int,
        heads_factor: int | None,
        tensor_map: Dict[str, TensorMeta],
    ) -> TensorMeta:
        """Split attention tensor using structural rules.

        4D ``(B, H, S, D)``:
          heads_factor → split dim 1 by heads_factor.
          seq_factor   → split dims 2,3 by seq_factor.
        3D ``(H, Sq, Sk)`` (bmm):
          heads_factor → skip (heads at dim 0, protected).
          seq_factor   → split dims 1,2 by seq_factor.
        3D ``(B, S, D)`` (projection):
          heads_factor → match heads_dim in any position.
          seq_factor   → split dim 1 by seq_factor.
        2D ``(S, D)``:
          seq_factor   → split dim 0.
          heads_factor → match heads_dim in any position.
        """
        if tensor.id in tensor_map:
            return tensor_map[tensor.id]

        shape = tensor.shape
        rank = len(shape)
        new_shape = list(shape)
        found = False

        if rank == 4:
            # Check if dim 1 is heads dimension
            is_heads_dim = heads_factor and heads_dim > 0 and shape[1] == heads_dim
            if is_heads_dim:
                new_shape[1] = max(1, heads_dim // heads_factor)
                found = True
            
            if seq_factor:
                # Split dim 1 if it's not a heads dimension
                if not is_heads_dim and shape[1] > 1:
                    new_shape[1] = shape[1] // seq_factor
                    found = True
                # Also check dims 2 and 3
                for i in (2, 3):
                    if shape[i] > 1:
                        new_shape[i] = shape[i] // seq_factor
                        found = True

        elif rank == 3:
            if heads_factor and heads_dim > 0:
                for i in range(rank):
                    if shape[i] == heads_dim:
                        new_shape[i] = max(1, shape[i] // heads_factor)
                        found = True
                        break
            if seq_factor:
                protect = bool(heads_factor and heads_dim > 0)
                for i in (1, 2):
                    if shape[i] > 1 and (not protect or shape[i] != heads_dim):
                        new_shape[i] = shape[i] // seq_factor
                        found = True

        elif rank == 2:
            if seq_factor:
                new_shape[0] = shape[0] // seq_factor
                found = True
            elif heads_factor and heads_dim > 0:
                for i in range(rank):
                    if shape[i] == heads_dim:
                        new_shape[i] = max(1, shape[i] // heads_factor)
                        found = True
                        break

        if not found:
            return tensor

        return _make_tensor(tensor, tuple(new_shape), tensor_map)

    # ── post-processing for skipped nodes ─────────────────────────────────

    @staticmethod
    def _postprocess_remaining_seq_tensors(
        graph: OpGraph,
        seq_len: int,
        cp: int,
        tensor_map: Dict[str, TensorMeta],
    ) -> None:
        """Fallback: split remaining tensors that still have seq_len.

        This catches tensors that were not processed by the main CP pass, including:
        - Tensors from skipped nodes (sum, zeros, scatter_add, clone, comm)
        - Tensors from nodes that were processed but whose tensors weren't split
          (e.g., due to classification issues or edge cases)

        Only applies to:
        - Tensors NOT already in tensor_map (i.e., not already split)
        - Nodes NOT in _POSTPROCESS_EXCLUDE (embedding/index/gather where the
          seq_len-sized dimension is actually vocab/index, not sequence)
        - Nodes WITHOUT cp_split annotation (nodes with annotation were correctly
          processed by the main pass and their tensor shapes are intentional)
        """
        seq_local = seq_len // cp

        for node in graph.nodes.values():
            if node.op_type in _POSTPROCESS_EXCLUDE:
                continue
            if node.annotations.get("cp_split"):
                continue

            for i, t in enumerate(node.inputs):
                if t.id in tensor_map:
                    continue
                if seq_len not in t.shape:
                    continue
                new_shape = tuple(
                    d // cp if d == seq_len else d for d in t.shape
                )
                tensor_map[t.id] = _make_tensor(t, new_shape, {})
                node.inputs[i] = tensor_map[t.id]

            for i, t in enumerate(node.outputs):
                if t.id in tensor_map:
                    continue
                if seq_len not in t.shape:
                    continue
                
                # Special handling for backward sum operations with seq_len in dim 1
                # Pattern: 2D output tensor [hidden, seq_len] from sum operations
                # Gate on phase annotation (set by adapter.stitch_fwd_bwd), not
                # the bwd_ id prefix — id-based gating is brittle and coupled
                # to adapter.py:665.
                if (len(t.shape) == 2 and t.shape[1] == seq_len and
                    "sum" in node.op_type.lower() and
                    node.annotations.get("phase") == "bwd"):
                    # Split dim 1 instead of dim 0
                    new_shape = (t.shape[0], seq_local)
                else:
                    new_shape = tuple(
                        d // cp if d == seq_len else d for d in t.shape
                    )
                
                tensor_map[t.id] = _make_tensor(t, new_shape, {})
                node.outputs[i] = tensor_map[t.id]

        for edge in graph.edges:
            if edge.tensor and edge.tensor.id in tensor_map:
                edge.tensor = tensor_map[edge.tensor.id]

    # ── shared helpers ───────────────────────────────────────────────────

    @staticmethod
    def _rebuild_node(
        node: OpNode,
        new_inputs: list[TensorMeta],
        new_outputs: list[TensorMeta],
    ) -> OpNode:
        return OpNode(
            id=node.id,
            op_type=node.op_type,
            inputs=new_inputs,
            outputs=new_outputs,
            attrs=node.attrs,
            scope=node.scope,
            layer=node.layer,
            category=node.category,
            annotations=node.annotations,
            op_short=node.op_short,
            module_class=node.module_class,
            component=node.component,
            name=node.name,
            provenance=node.provenance,
            src_file=node.src_file,
            src_line=node.src_line,
            src_code=node.src_code,
            call_id=node.call_id,
            fused_from=node.fused_from,
            num_sub_ops=node.num_sub_ops,
            fusion_level=node.fusion_level,
        )


def _make_tensor(
    tensor: TensorMeta,
    new_shape: tuple[int, ...],
    tensor_map: Dict[str, TensorMeta],
) -> TensorMeta:
    new_numel = prod(new_shape) if new_shape else 1
    new_bytes = int(new_numel * tensor.dtype.itemsize)
    new_tensor = TensorMeta(
        id=tensor.id,
        shape=new_shape,
        dtype=tensor.dtype,
        mem_bytes=new_bytes,
    )
    tensor_map[tensor.id] = new_tensor
    return new_tensor
