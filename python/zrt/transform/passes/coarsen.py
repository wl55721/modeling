"""GraphCoarsenPass: aggregate aten-level OpGraph nodes into block-level OpNodes.

Groups aten ops by module scope (``OpNode.scope``) and replaces each group
with a single block-level OpNode carrying ``spec_kind`` and ``source``
attributes compatible with the downstream Transform Pipeline.

Supports both standard HF naming (``self_attn``/``mlp``/``input_layernorm``)
and V4 naming (``attn``/``ffn``/``attn_norm``/``ffn_norm``/``wq_a``/``wkv``).

For already block-level inputs (all ``op_type`` start with ``"spec."`` or
``attrs["source"] == "model_spec"``), the pass is a no-op.

Pipeline position: runs at the end of the ``split`` stage, after parallel
passes and comm insertion, so that split-stage passes can still read raw
aten ``op_type`` values.  After coarsening, the ``fuse`` stage becomes a
structural no-op (graph is already at module granularity).
"""
from __future__ import annotations

import fnmatch
import math
import os
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

import yaml

from python.zrt.ir.edge import Edge
from python.zrt.ir.node import OpNode, infer_category
from python.zrt.ir.types import TensorMeta
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext


# ---------------------------------------------------------------------------
# YAML rule loader
# ---------------------------------------------------------------------------

_COARSEN_RULES: list[dict[str, Any]] = []
_RULES_LOADED = False


def _load_coarsen_rules() -> list[dict[str, Any]]:
    """Load coarsen rules from coarsen_rules.yaml (cached at module level)."""
    global _COARSEN_RULES, _RULES_LOADED
    if _RULES_LOADED:
        return _COARSEN_RULES

    rules_path = os.path.join(os.path.dirname(__file__), "coarsen_rules.yaml")
    if os.path.exists(rules_path):
        with open(rules_path, "r", encoding="utf-8") as f:
            _COARSEN_RULES = yaml.safe_load(f) or []
    else:
        _COARSEN_RULES = []
    _RULES_LOADED = True
    return _COARSEN_RULES


def _match_rule(scope: str, module_class: str, rules: list[dict]) -> dict | None:
    """Find the highest-priority rule matching the given scope and module_class."""
    base_scope = scope.split("@")[0] if "@" in scope else scope
    candidates = []
    for rule in rules:
        pattern = rule.get("scope_pattern", "")
        if not fnmatch.fnmatch(base_scope, pattern):
            continue
        rule_classes = rule.get("module_classes", [])
        if rule_classes and module_class and module_class not in rule_classes:
            continue
        candidates.append(rule)

    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("priority", 0))


def _extract_shape_vars(inputs: list, outputs: list, graph: "OpGraph") -> dict[str, float]:
    """Extract shape variables (B, S, H, Ho, Hi, T) from tensor lists."""
    vars_: dict[str, float] = {}

    all_tensors = list(inputs or []) + list(outputs or [])

    for t in all_tensors:
        s = t.shape if hasattr(t, "shape") else t
        if len(s) >= 4:
            if "B" not in vars_:
                vars_["B"] = s[0]
                vars_["S"] = s[1]
            H_candidate = 1
            for d in s[2:]:
                H_candidate *= d
            if H_candidate > vars_.get("H", 0):
                vars_["H"] = H_candidate
        elif len(s) >= 3:
            if "B" not in vars_:
                vars_["B"] = s[0]
                vars_["S"] = s[1]
            if s[-1] > vars_.get("H", 0):
                vars_["H"] = s[-1]
        elif len(s) == 2:
            if "B" not in vars_:
                vars_["B"] = 1
                vars_["S"] = s[0]
            if s[-1] > vars_.get("H", 0):
                vars_["H"] = s[-1]
            if s[0] > s[1]:
                vars_["Hi"] = max(vars_.get("Hi", 0), s[0])

    if outputs:
        last_out = outputs[-1]
        shape = last_out.shape if hasattr(last_out, "shape") else last_out
        if shape:
            if len(shape) >= 4:
                vars_["Ho"] = 1
                for d in shape[2:]:
                    vars_["Ho"] *= d
            else:
                vars_["Ho"] = shape[-1]
            if len(shape) >= 2:
                vars_["Hc"] = shape[-1]

    vars_["T"] = vars_.get("B", 1) * vars_.get("S", 1)

    meta = graph.metadata
    H = vars_.get("H", 7168)
    vars_["E"] = meta.get("moe_total_experts", meta.get("num_experts", 384))
    vars_["K"] = meta.get("moe_active_experts", meta.get("top_k", 6))
    vars_["V"] = meta.get("vocab_size", 129280)
    vars_["D"] = meta.get("head_dim", 128)
    vars_["N"] = meta.get("num_heads", max(1, int(H // vars_["D"])))
    vars_["R"] = meta.get("compress_ratio", 4)
    vars_["Hi"] = vars_.get("Hi", meta.get("ffn_hidden", meta.get("moe_ffn", H * 4)))

    for key in ("B", "S", "H", "Ho", "Hi", "T", "E", "K", "V", "N", "D", "R", "Hc"):
        vars_.setdefault(key, 1)

    return vars_


def _eval_formula(formula: str, vars_: dict[str, float]) -> float:
    """Safely evaluate a formula string with the given variables."""
    if not formula:
        return 0.0
    try:
        safe_dict = {
            "max": max, "min": min, "abs": abs,
            "log2": lambda x: math.log2(max(x, 1)),
            "sqrt": math.sqrt,
            **vars_,
        }
        result = eval(formula, {"__builtins__": {}}, safe_dict)
        return float(result) if result is not None else 0.0
    except Exception:
        return 0.0


class GraphCoarsenPass(GraphPass):
    """Aggregate aten-level OpGraph nodes into block-level OpNodes by module scope.

    For each module scope that has leaf ops, all non-communication OpNodes
    sharing that scope are merged into a single block-level OpNode.
    Communication nodes (``category == "communication"``) are preserved as
    singletons so that the pipeline's comm-latency modeling remains intact.

    Handles both leaf scopes (no child modules) and intermediate scopes
    (e.g. V4 ``attn`` module where attention kernels execute at the parent
    scope while projections are child modules).

    Block-level inputs (``op_type`` prefix ``"spec."``) pass through unchanged.
    """

    name = "graph_coarsen"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        if _is_block_level(graph):
            return graph

        # if _has_backward_nodes(graph):
        #     return graph

        g = graph.clone()

        scope_groups = _build_scope_groups(g)

        nid_map: dict[str, str] = {}
        new_nodes: list[OpNode] = []

        for group_key, node_ids in scope_groups.items():
            if group_key.startswith("__comm__"):
                nid = node_ids[0]
                nid_map[nid] = nid
                new_nodes.append(g.nodes[nid])
            else:
                agg = _build_aggregated_node(g, node_ids, group_key)
                for nid in node_ids:
                    nid_map[nid] = agg.id
                new_nodes.append(agg)

        orig_order = {nid: idx for idx, nid in enumerate(g.nodes)}

        seen_edges: set[tuple[str, str]] = set()
        new_edges: list[Edge] = []
        for e in g.edges:
            new_src = nid_map.get(e.src, e.src)
            new_dst = nid_map.get(e.dst, e.dst)
            if new_src == new_dst:
                continue
            key = (new_src, new_dst)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            new_edges.append(Edge(
                src=new_src, src_idx=0,
                dst=new_dst, dst_idx=0,
                tensor=e.tensor, tensor_id=e.tensor_id,
            ))

        new_edges = _break_cycles(new_edges, orig_order, nid_map)

        from python.zrt.ir.graph import OpGraph as _OpGraph

        coarsened = _OpGraph(
            name=g.name,
            phase=g.phase,
            nodes={n.id: n for n in new_nodes},
            edges=new_edges,
            metadata=dict(g.metadata),
        )
        coarsened.metadata["coarsened"] = True
        return coarsened


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def _is_block_level(graph: "OpGraph") -> bool:
    """Return True if the graph is already at block-level granularity.

    A graph is block-level when every non-communication node either has
    ``op_type`` starting with ``"spec."`` or carries ``attrs["source"] ==
    "model_spec"``.
    """
    for node in graph.nodes.values():
        if node.category == "communication":
            continue
        if node.op_type.startswith("spec."):
            continue
        if node.attrs.get("source") == "model_spec":
            continue
        return False
    return True


def _has_backward_nodes(graph: "OpGraph") -> bool:
    """Return True if the graph contains backward-phase nodes.

    Stitched fwd+bwd graphs have reversed gradient-flow edges that create
    cycles when coarsened by scope (forward: A→B, backward: B→A → cycle
    A→B→A after merging fwd/bwd nodes of the same scope).  Coarsening is
    therefore skipped for stitched graphs.
    """
    for node in graph.nodes.values():
        if node.annotations.get("phase") == "bwd":
            return True
    return False


def _break_cycles(
    edges: list[Edge],
    orig_order: dict[str, int],
    nid_map: dict[str, str],
) -> list[Edge]:
    """Remove back-edges that create cycles in the coarsened graph.

    Uses DFS coloring to find all back-edges (edges pointing to a GRAY
    node in the DFS tree).  The original aten-level graph is a DAG, so
    any cycle in the coarsened graph is an artifact of coarsening.

    To minimize information loss, we process nodes in original execution
    order (earliest first) so that back-edges tend to be the "reverse"
    direction of the original dataflow.
    """
    from collections import defaultdict

    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    for idx, e in enumerate(edges):
        adj[e.src].append((e.dst, idx))

    node_order: dict[str, int] = {}
    for orig_nid, coarsened_nid in nid_map.items():
        idx = orig_order.get(orig_nid, 0)
        if coarsened_nid not in node_order or idx < node_order[coarsened_nid]:
            node_order[coarsened_nid] = idx

    all_nodes = set()
    for e in edges:
        all_nodes.add(e.src)
        all_nodes.add(e.dst)

    sorted_nodes = sorted(all_nodes, key=lambda n: node_order.get(n, 0))

    WHITE, GRAY, BLACK = 0, 1, 2
    color: dict[str, int] = {n: WHITE for n in all_nodes}
    back_edge_indices: set[int] = set()

    for start in sorted_nodes:
        if color[start] != WHITE:
            continue
        stack: list[tuple[str, int]] = [(start, 0)]
        color[start] = GRAY
        while stack:
            node, ei = stack[-1]
            neighbors = adj.get(node, [])
            if ei < len(neighbors):
                stack[-1] = (node, ei + 1)
                dst, edge_idx = neighbors[ei]
                if color[dst] == GRAY:
                    back_edge_indices.add(edge_idx)
                elif color[dst] == WHITE:
                    color[dst] = GRAY
                    stack.append((dst, 0))
            else:
                color[node] = BLACK
                stack.pop()

    if not back_edge_indices:
        return edges
    return [e for idx, e in enumerate(edges) if idx not in back_edge_indices]


# ---------------------------------------------------------------------------
# Scope grouping
# ---------------------------------------------------------------------------

def _build_scope_groups(graph: "OpGraph") -> dict[str, list[str]]:
    """Group node IDs by module scope and phase.

    Communication nodes are placed in singleton groups keyed by
    ``__comm__<node_id>`` so they are never merged with compute nodes.

    Non-communication nodes are grouped by ``(scope, phase)`` — forward
    and backward nodes for the same scope are kept in separate groups
    to avoid merging them into a single node (which would create
    self-edges and destroy graph connectivity).

    Scopes that are intermediate (have child modules) get a ``@parent_ops``
    suffix to distinguish them from leaf scopes.
    """
    all_scopes = _all_scopes(graph)
    scopes_with_children = _scopes_with_children(all_scopes)

    groups: dict[str, list[str]] = defaultdict(list)

    for node in graph.nodes.values():
        if node.category == "communication":
            groups[f"__comm__{node.id}"].append(node.id)
            continue

        scope = node.scope or ""
        phase = node.annotations.get("phase", "")
        phase_suffix = f"@{phase}" if phase else ""

        if scope in scopes_with_children:
            groups[f"{scope}@parent_ops{phase_suffix}"].append(node.id)
        else:
            groups[f"{scope}{phase_suffix}"].append(node.id)

    return dict(groups)


def _all_scopes(graph: "OpGraph") -> set[str]:
    scopes: set[str] = set()
    for node in graph.nodes.values():
        s = node.scope or ""
        if s:
            scopes.add(s)
    return scopes


def _scopes_with_children(scopes: set[str]) -> set[str]:
    """Return scopes that are proper prefixes of at least one other scope."""
    result: set[str] = set()
    sorted_scopes = sorted(scopes, key=len)
    for i, s in enumerate(sorted_scopes):
        for j in range(i + 1, len(sorted_scopes)):
            if sorted_scopes[j].startswith(s + "."):
                result.add(s)
                break
    return result


# ---------------------------------------------------------------------------
# Module → spec_kind mapping
# ---------------------------------------------------------------------------

_MODULE_TO_SPEC_KIND: dict[str, str] = {
    # Standard HF: LayerNorm / RMSNorm
    "input_layernorm": "rmsnorm",
    "post_attention_layernorm": "rmsnorm",
    "final_layernorm": "rmsnorm",
    "norm": "rmsnorm",
    # V4: LayerNorm / RMSNorm
    "attn_norm": "rmsnorm",
    "ffn_norm": "rmsnorm",
    "enorm": "rmsnorm",
    "hnorm": "rmsnorm",
    # V4: attention projections
    "wq_a": "matmul",
    "wq_b": "matmul",
    "wkv": "matmul",
    "wo_a": "matmul",
    "wo_b": "matmul",
    "q_norm": "rmsnorm",
    "kv_norm": "rmsnorm",
    "q_a_proj": "matmul",
    "q_a_layernorm": "rmsnorm",
    "q_b_proj": "matmul",
    "kv_a_proj": "matmul",
    "kv_a_layernorm": "rmsnorm",
    "kv_b_proj": "matmul",
    # Standard HF: attention projections
    "q_proj": "matmul",
    "k_proj": "matmul",
    "v_proj": "matmul",
    "o_proj": "matmul",
    "out_proj": "matmul",
    "qkv_proj": "matmul",
    # Standard HF: FFN
    "gate_proj": "matmul",
    "up_proj": "matmul",
    "down_proj": "matmul",
    "act_fn": "swiglu",
    # V4: MoE
    "gate": "router",
    "experts": "expert",
    "shared_experts": "shared_expert",
    # V4: Compressor / Indexer
    "compressor": "compressor",
    "indexer": "indexer",
    "weights_proj": "matmul",
    # Embedding / LM head
    "embed_tokens": "embed",
    "embed": "embed",
    "lm_head": "lm_head",
    "head": "lm_head",
    # Rotary
    "rotary_emb": "rope",
    # V4: Hyper-Connections (injected by patches.py)
    "hc_pre_attn": "mhc_pre",
    "hc_post_attn": "mhc_post",
    "hc_pre_ffn": "mhc_pre",
    "hc_post_ffn": "mhc_post",
    "hc_head_module": "mhc_head",
    # V4: MTP
    "e_proj": "matmul",
    "h_proj": "matmul",
}

_MODULE_CLASS_TO_KIND: dict[str, str] = {
    "RMSNorm": "rmsnorm",
    "LayerNorm": "ln",
    "LlamaRMSNorm": "rmsnorm",
    "Qwen2RMSNorm": "rmsnorm",
    "DeepseekV3RMSNorm": "rmsnorm",
    "Linear": "matmul",
    "ColumnParallelLinear": "matmul",
    "RowParallelLinear": "matmul",
    "Embedding": "embed",
    "ParallelEmbedding": "embed",
    "RotaryEmbedding": "rope",
    "LlamaRotaryEmbedding": "rope",
    "SiLU": "swiglu",
    "SwiGLU": "swiglu",
    "Attention": "attn_core",
    "Compressor": "compressor",
    "Indexer": "indexer",
    "Gate": "router",
    "Expert": "expert",
    "ParallelHead": "lm_head",
    "HCPreAttn": "mhc_pre",
    "HCPostAttn": "mhc_post",
    "HCPreFfn": "mhc_pre",
    "HCPostFfn": "mhc_post",
    "HCHead": "mhc_head",
}


def _infer_spec_kind(scope: str, module_class: str, component: str) -> str:
    """Infer block-level ``spec_kind`` from scope, module class, and component."""
    base_scope = scope.split("@")[0] if "@" in scope else scope
    leaf_name = base_scope.rsplit(".", 1)[-1] if base_scope else ""

    kind = _MODULE_TO_SPEC_KIND.get(leaf_name)
    if kind:
        if leaf_name == "gate_proj" and _is_moe_context(base_scope, component):
            return "router"
        return kind

    parent_name = ""
    if base_scope and "." in base_scope:
        parts = base_scope.rsplit(".", 2)
        if len(parts) >= 2:
            parent_name = parts[-2]

    if parent_name == "self_attn" and leaf_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        return "matmul"

    if parent_name in ("mlp", "moe") and leaf_name in ("gate_proj", "up_proj", "down_proj"):
        return "matmul"

    if parent_name == "mlp" and leaf_name == "act_fn":
        return "swiglu"

    if ("self_attn" in base_scope or ".attn" in base_scope) and module_class in (
        "Linear", "ColumnParallelLinear", "RowParallelLinear",
    ):
        return "matmul"

    if ("mlp" in base_scope or ".ffn" in base_scope) and module_class in (
        "Linear", "ColumnParallelLinear", "RowParallelLinear",
    ):
        return "matmul"

    kind = _MODULE_CLASS_TO_KIND.get(module_class)
    if kind:
        return kind

    if component:
        return component

    return leaf_name or "unknown"


def _is_moe_context(scope: str, component: str) -> bool:
    """Return True if the scope/component indicates a MoE context."""
    if component in ("moe", "routed_expert"):
        return True
    if ".moe." in scope or ".ffn." in scope:
        return True
    return False


_MODULE_CLASS_TO_OP_TYPE: dict[str, str] = {
    "Linear": "linear",
    "ColumnParallelLinear": "column_parallel_linear",
    "RowParallelLinear": "row_parallel_linear",
    "RMSNorm": "rms_norm",
    "LlamaRMSNorm": "rms_norm",
    "Qwen2RMSNorm": "rms_norm",
    "DeepseekV3RMSNorm": "rms_norm",
    "LayerNorm": "rms_norm",
    "ParallelEmbedding": "parallel_embedding",
    "Embedding": "parallel_embedding",
    "RotaryEmbedding": "rotary_emb",
    "LlamaRotaryEmbedding": "rotary_emb",
    "Compressor": "kv_compressor",
    "Indexer": "sparse_indexer",
    "Attention": "mla_sparse_attn",
    "Gate": "moe_gate",
    "Expert": "moe_expert_swiglu",
    "HCPreAttn": "hc_pre",
    "HCPreFfn": "hc_pre",
    "HCPostAttn": "hc_post",
    "HCPostFfn": "hc_post",
    "HCHead": "hc_head",
    "SiLU": "swiglu",
    "SwiGLU": "swiglu",
    "ParallelHead": "linear",
}


def _module_class_to_op_type(module_class: str, spec_kind: str) -> str:
    """Map module_class to FusionPass-compatible op_type.

    Falls back to spec_kind-based mapping for unknown module classes.
    """
    if module_class and module_class in _MODULE_CLASS_TO_OP_TYPE:
        return _MODULE_CLASS_TO_OP_TYPE[module_class]
    _KIND_TO_OP = {
        "matmul": "linear",
        "attn_core": "mla_sparse_attn",
        "sparse_attn": "mla_sparse_attn",
        "hca_attn": "mla_sparse_attn",
        "swa_attn": "mla_sparse_attn",
        "rmsnorm": "rms_norm",
        "ln": "rms_norm",
        "rope": "rotary_emb",
        "swiglu": "swiglu",
        "embed": "parallel_embedding",
        "lm_head": "linear",
        "router": "moe_gate",
        "expert": "moe_expert_swiglu",
        "shared_expert": "moe_expert_swiglu",
        "indexer": "sparse_indexer",
        "compressor": "kv_compressor",
        "mhc_pre": "hc_pre",
        "mhc_post": "hc_post",
        "mhc_head": "hc_head",
    }
    return _KIND_TO_OP.get(spec_kind, spec_kind)


# ---------------------------------------------------------------------------
# Scope → short name helpers
# ---------------------------------------------------------------------------

_LAYER_RE = re.compile(r"layers\.(\d+)")


def _extract_layer_id(scope: str) -> str:
    base_scope = scope.split("@")[0] if "@" in scope else scope
    m = _LAYER_RE.search(base_scope)
    return m.group(1) if m else "-1"


def _extract_short_name(scope: str) -> str:
    """Extract a human-readable short name from a full scope path.

    Strips wrapper prefixes (``model``, ``transformer``, ``layers.N``)
    and keeps the remaining path segments, including instance indices
    (e.g. expert index ``0`` in ``experts.0.w1``).

    ``"model.transformer.layers.3.attn.wq_a"`` → ``"attn.wq_a"``
    ``"model.transformer.layers.0.attn@parent_ops"`` → ``"attn"``
    ``"model.transformer.embed"`` → ``"embed"``
    ``"model.transformer.layers.0.ffn.experts.0.w1"`` → ``"ffn.experts.0.w1"``
    """
    base_scope = scope.split("@")[0] if "@" in scope else scope
    parts = base_scope.split(".")

    skip_prefixes = {"model", "transformer", "h"}
    start = 0
    for i, p in enumerate(parts):
        if p in skip_prefixes:
            start = i + 1
        elif p == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
            start = i + 2
        elif p.isdigit() and i == start:
            start = i + 1
        else:
            break

    remaining = parts[start:]
    return ".".join(remaining) if remaining else (parts[-1] if parts else "unknown")


def _extract_component(scope: str, nodes: list[OpNode]) -> str:
    """Determine the semantic component for a group of nodes."""
    for n in nodes:
        if n.component:
            return n.component
    base_scope = scope.split("@")[0] if "@" in scope else scope
    if "self_attn" in base_scope or ".attn" in base_scope:
        return "attention"
    if "mlp" in base_scope or ".ffn" in base_scope or "moe" in base_scope:
        if "expert" in base_scope:
            return "routed_expert"
        if "shared" in base_scope:
            return "shared_expert"
        return "ffn"
    if "embed" in base_scope:
        return "embedding"
    if "norm" in base_scope or "layernorm" in base_scope:
        return "norm"
    if "hc_pre" in base_scope or "hc_post" in base_scope:
        return "hc"
    return ""


# ---------------------------------------------------------------------------
# Node aggregation
# ---------------------------------------------------------------------------

def _build_aggregated_node(
    graph: "OpGraph",
    node_ids: list[str],
    scope: str,
) -> OpNode:
    """Build one block-level OpNode from a group of aten-level nodes."""
    nodes = [graph.nodes[nid] for nid in node_ids]
    if not nodes:
        raise ValueError(f"Empty node group for scope {scope!r}")

    first = nodes[0]
    last = nodes[-1]

    layer_id = _extract_layer_id(scope)
    short_name = _extract_short_name(scope)
    base_id = f"L{layer_id}.{short_name}" if layer_id != "-1" else short_name

    phase = first.annotations.get("phase", "")
    if phase == "bwd":
        node_id = f"{base_id}_bwd"
    else:
        node_id = base_id

    module_class = first.module_class or ""
    component = _extract_component(scope, nodes)
    spec_kind = _infer_spec_kind(scope, module_class, component)

    attrs: dict[str, Any] = {}
    for n in nodes:
        attrs.update(n.attrs)
    attrs["spec_kind"] = spec_kind
    attrs["source"] = "coarsened"
    attrs["layer_kind"] = attrs.get("layer_kind", "dense")
    attrs["layer_id"] = int(layer_id) if layer_id != "-1" else -1

    annotations: dict[str, Any] = {}
    for n in nodes:
        for k, v in n.annotations.items():
            if isinstance(v, (int, float)):
                annotations[k] = annotations.get(k, 0) + v
            elif k not in annotations:
                annotations[k] = v

    rules = _load_coarsen_rules()
    rule = _match_rule(scope, module_class, rules)
    if rule and not annotations.get("flops"):
        coarsened_inputs = list(first.inputs)
        coarsened_outputs = list(last.outputs)
        shape_vars = _extract_shape_vars(coarsened_inputs, coarsened_outputs, graph)
        fwd_flops = _eval_formula(rule.get("flops_formula", ""), shape_vars)
        mem_bytes = _eval_formula(rule.get("memory_formula", ""), shape_vars)
        grad_ratio = rule.get("grad_ratio", {})
        dx_ratio = grad_ratio.get("dx", 0)
        dw_ratio = grad_ratio.get("dw", 0)

        annotations["flops"] = int(fwd_flops)
        annotations["read_bytes"] = int(mem_bytes * 0.6)
        annotations["write_bytes"] = int(mem_bytes * 0.4)
        annotations["flops_fwd"] = int(fwd_flops)
        annotations["flops_dx"] = int(fwd_flops * dx_ratio)
        annotations["flops_dw"] = int(fwd_flops * dw_ratio)

        rule_annots = rule.get("annotations", {})
        for k, v in rule_annots.items():
            annotations.setdefault(f"coarsen.{k}", v)

    op_type = _module_class_to_op_type(module_class, spec_kind)
    category = infer_category(op_type, component)

    fused_from = [n.op_type for n in nodes]

    return OpNode(
        id=node_id,
        op_type=op_type,
        inputs=list(first.inputs),
        outputs=list(last.outputs),
        attrs=attrs,
        scope=scope.split("@")[0] if "@" in scope else scope,
        category=category,
        annotations=annotations,
        op_short=short_name,
        module_class=module_class,
        layer=layer_id,
        component=component,
        fused_from=fused_from,
        num_sub_ops=len(nodes),
        name=short_name,
    )
