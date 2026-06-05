"""Context Parallel pass: CPKind-aware shape splitting.

Follows the training-side sharding semantics (``shard.py::_apply_cp_sharding``)
and the scope-classification pattern from ``tensor_parallel.py``.

Splitting rules by CPKind
--------------------------
Ulysses:
  - Attention-internal nodes: heads ÷ cp, seq unchanged (A2A gathers seq).
  - Other nodes: seq ÷ cp.
Ring / Compressed:
  - All nodes: seq ÷ cp.
Hybrid:
  - Attention-internal nodes: heads ÷ cp_ulysses, seq ÷ cp_ring.
  - Other nodes: seq ÷ cp.

Attention-internal scopes
--------------------------
Any node whose ``scope`` places it inside the self-attention region:
QKV projections, attention score / softmax, RoPE, output projection,
and MLA-specific projections (q_a_proj, kv_a_proj, etc.).
"""
from __future__ import annotations

import logging
from math import prod
from typing import Dict, List

from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode
from python.zrt.ir.types import TensorMeta
from python.zrt.transform.base import GraphPass
from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


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


class ContextParallelPass(GraphPass):
    """CPKind-aware Context Parallel pass.

    Mirrors the training-side ``_apply_cp_sharding`` logic:
    - Ulysses splits heads for attention, seq for everything else.
    - Ring / Compressed split seq everywhere.
    - Hybrid splits heads by ``cp_ulysses`` and seq by ``cp_ring``.
    """

    name = "context_parallel"

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        if ctx.parallel.cp <= 1:
            return graph

        g = graph.clone()
        cp = ctx.parallel.cp

        cp_kind = (
            ctx.training.resolve_cp_kind(ctx.model_id, cp)
            if ctx.training
            else "ulysses"
        )

        seq_len = ctx.training.seq_len if ctx.training else 2048
        num_heads = ctx.training.num_heads if ctx.training else 0
        num_kv_heads = (
            ctx.training.num_kv_heads
            if ctx.training and ctx.training.num_kv_heads > 0
            else num_heads
        )
        tp = ctx.parallel.tp
        num_heads_tp = num_heads // tp if num_heads > 0 and tp > 0 else 0
        num_kv_heads_tp = num_kv_heads // tp if num_kv_heads > 0 and tp > 0 else 0

        cp_ulysses, cp_ring = ctx.parallel.hybrid_cp_factors()

        attn_nodes, general_nodes = self._classify_nodes(g, seq_len)

        tensor_map: Dict[str, TensorMeta] = {}
        split_nodes: List[OpNode] = []

        for node in general_nodes:
            changed = self._apply_seq_split(
                g, node, seq_len, cp, tensor_map,
            )
            if changed:
                split_nodes.append(node)

        for node in attn_nodes:
            changed = self._apply_attn_split(
                g, node, seq_len, cp_kind, cp,
                cp_ulysses, cp_ring,
                num_heads_tp, num_kv_heads_tp,
                _is_kv_proj_scope(node.scope or ""),
                tensor_map,
            )
            if changed:
                split_nodes.append(node)

        for node in split_nodes:
            ann = {"kind": cp_kind, "cp": cp}
            if cp_kind == "hybrid":
                ann["cp_ulysses"] = cp_ulysses
                ann["cp_ring"] = cp_ring
            node.annotations["cp_split"] = ann

        for edge in g.edges:
            if edge.tensor and edge.tensor.id in tensor_map:
                edge.tensor = tensor_map[edge.tensor.id]

        logger.info(
            "ContextParallelPass: cp=%s kind=%s | "
            "attn_nodes=%d general_nodes=%d split_nodes=%d",
            cp, cp_kind, len(attn_nodes), len(general_nodes), len(split_nodes),
        )

        return g

    # ── node classification ──────────────────────────────────────────────

    def _classify_nodes(
        self, graph: OpGraph, seq_len: int,
    ) -> tuple[List[OpNode], List[OpNode]]:
        attn_nodes: List[OpNode] = []
        general_nodes: List[OpNode] = []

        for node in graph.topo_sort():
            if node.category == "communication":
                continue
            has_seq = any(
                seq_len in t.shape for t in node.inputs + node.outputs
            )
            if not has_seq:
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
        seq_len: int,
        cp: int,
        tensor_map: Dict[str, TensorMeta],
    ) -> bool:
        seq_local = seq_len // cp
        changed = False

        new_inputs = []
        for t in node.inputs:
            nt = self._split_seq_dim(t, seq_len, seq_local, tensor_map)
            new_inputs.append(nt)
            if nt is not t:
                changed = True

        new_outputs = []
        for t in node.outputs:
            nt = self._split_seq_dim(t, seq_len, seq_local, tensor_map)
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
        seq_len: int,
        cp_kind: str,
        cp: int,
        cp_ulysses: int,
        cp_ring: int,
        num_heads_tp: int,
        num_kv_heads_tp: int,
        is_kv_proj: bool,
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
        for t in node.inputs:
            nt = self._split_attn_tensor(
                t, seq_len, seq_factor, heads_dim, heads_factor, tensor_map,
            )
            new_inputs.append(nt)
            if nt is not t:
                changed = True

        new_outputs = []
        for t in node.outputs:
            nt = self._split_attn_tensor(
                t, seq_len, seq_factor, heads_dim, heads_factor, tensor_map,
            )
            new_outputs.append(nt)
            if nt is not t:
                changed = True

        if changed:
            graph.nodes[node.id] = self._rebuild_node(
                node, new_inputs, new_outputs,
            )
        return changed

    # ── tensor-level helpers ─────────────────────────────────────────────

    def _split_attn_tensor(
        self,
        tensor: TensorMeta,
        seq_len: int,
        seq_factor: int | None,
        heads_dim: int,
        heads_factor: int | None,
        tensor_map: Dict[str, TensorMeta],
    ) -> TensorMeta:
        if tensor.id in tensor_map:
            return tensor_map[tensor.id]

        shape = tensor.shape
        new_shape = list(shape)
        found = False

        if heads_factor and heads_dim > 0:
            if len(shape) == 4:
                if shape[1] == heads_dim:
                    new_shape[1] = max(1, heads_dim // heads_factor)
                    found = True
                for i in range(2, len(shape)):
                    if seq_factor and shape[i] == seq_len:
                        new_shape[i] = seq_len // seq_factor
                        found = True
            else:
                for i in range(len(shape)):
                    if shape[i] == heads_dim:
                        new_shape[i] = max(1, shape[i] // heads_factor)
                        found = True
                        break

        if seq_factor:
            protect_attn_seq = bool(heads_factor and heads_dim > 0) and len(shape) == 4
            if not found or not protect_attn_seq:
                if len(shape) == 2 and shape[0] == seq_len:
                    new_shape[0] = seq_len // seq_factor
                    found = True
                for i in range(1, len(shape)):
                    if shape[i] == seq_len and (
                        not protect_attn_seq or i not in (2, 3)
                    ):
                        new_shape[i] = seq_len // seq_factor
                        found = True

        if not found:
            return tensor

        return self._make_tensor(tensor, tuple(new_shape), tensor_map)

    def _split_seq_dim(
        self,
        tensor: TensorMeta,
        seq_len: int,
        seq_local: int,
        tensor_map: Dict[str, TensorMeta],
    ) -> TensorMeta:
        if tensor.id in tensor_map:
            return tensor_map[tensor.id]

        shape = tensor.shape
        new_shape = list(shape)
        found_seq = False

        if len(shape) == 2 and shape[0] == seq_len:
            new_shape[0] = seq_local
            found_seq = True

        for i in range(1, len(shape)):
            if shape[i] == seq_len:
                new_shape[i] = seq_local
                found_seq = True

        if not found_seq:
            return tensor

        return self._make_tensor(tensor, tuple(new_shape), tensor_map)

    # ── shared helpers ───────────────────────────────────────────────────

    @staticmethod
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
