"""FusionPass: graph transform that applies MRO-based operator fusion."""
from __future__ import annotations

import fnmatch
import re
from collections import defaultdict
from typing import TYPE_CHECKING, Any

from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.ir.node import OpNode
    from python.zrt.transform.context import TransformContext


# ---------------------------------------------------------------------------
# Coarsen helper functions
# ---------------------------------------------------------------------------

def _match_coarsen_rule(scope: str, module_class: str, model_id: str = "") -> dict | None:
    """Find the highest-priority coarsen rule matching the given scope and module_class.
    
    Coarsen rules are loaded as plain dicts from the YAML file.
    They are matched by scope_pattern and module_classes.
    """
    from python.zrt.transform.fusion.loading.rule_set_initializer import get_coarsen_rules
    
    base_scope = scope.split("@")[0] if "@" in scope else scope
    candidates = []
    
    for rule in get_coarsen_rules(model_id):
        # Extract scope_pattern
        scope_pattern = rule.get("scope_pattern", "")
        if not scope_pattern:
            continue
        
        # Match scope against pattern (supports | for multiple patterns)
        patterns = [p.strip() for p in scope_pattern.split("|")]
        if not any(fnmatch.fnmatch(base_scope, p) for p in patterns):
            continue
        
        # Match module_class against module_classes
        module_classes = rule.get("module_classes", [])
        if module_classes and module_class and module_class not in module_classes:
            continue
        
        candidates.append(rule)
    
    if not candidates:
        return None
    return max(candidates, key=lambda r: r.get("priority", 0))


def _build_scope_groups_for_nodes(graph: "OpGraph", nodes: list["OpNode"]) -> dict[str, list[str]]:
    """Group a subset of node IDs by module scope, layer, and phase."""
    scopes: set[str] = set()
    for node in nodes:
        s = node.scope or ""
        if s:
            scopes.add(s)
    
    scopes_with_children = _scopes_with_children(scopes)
    
    groups: dict[str, list[str]] = defaultdict(list)
    
    for node in nodes:
        if node.category == "communication":
            groups[f"__comm__{node.id}"].append(node.id)
            continue
        
        scope = node.scope or ""
        phase = node.annotations.get("phase", "")
        phase_suffix = f"@{phase}" if phase else ""
        layer = node.layer or ""
        layer_suffix = f"@layer{layer}" if layer else ""
        
        if scope in scopes_with_children:
            groups[f"{scope}@parent_ops{phase_suffix}{layer_suffix}"].append(node.id)
        else:
            groups[f"{scope}{phase_suffix}{layer_suffix}"].append(node.id)
    
    return dict(groups)


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


def _break_cycles(
    edges: list,
    orig_order: dict[str, int],
    nid_map: dict[str, str],
) -> list:
    """Remove back-edges that create cycles in the coarsened graph."""
    from python.zrt.ir.edge import Edge
    
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
# Module → spec_kind mapping
# ---------------------------------------------------------------------------

_MODULE_TO_SPEC_KIND: dict[str, str] = {
    "input_layernorm": "rmsnorm", "post_attention_layernorm": "rmsnorm",
    "final_layernorm": "rmsnorm", "norm": "rmsnorm",
    "attn_norm": "rmsnorm", "ffn_norm": "rmsnorm", "enorm": "rmsnorm", "hnorm": "rmsnorm",
    "wq_a": "matmul", "wq_b": "matmul", "wkv": "matmul", "wo_a": "matmul", "wo_b": "matmul",
    "q_norm": "rmsnorm", "kv_norm": "rmsnorm",
    "q_a_proj": "matmul", "q_a_layernorm": "rmsnorm", "q_b_proj": "matmul",
    "kv_a_proj": "matmul", "kv_a_layernorm": "rmsnorm", "kv_b_proj": "matmul",
    "q_proj": "matmul", "k_proj": "matmul", "v_proj": "matmul", "o_proj": "matmul",
    "out_proj": "matmul", "qkv_proj": "matmul",
    "gate_proj": "matmul", "up_proj": "matmul", "down_proj": "matmul", "act_fn": "swiglu",
    "gate": "router", "experts": "expert", "shared_experts": "shared_expert",
    "compressor": "compressor", "indexer": "indexer", "weights_proj": "matmul",
    "embed_tokens": "embed", "embed": "embed", "lm_head": "lm_head", "head": "lm_head",
    "rotary_emb": "rope",
    "hc_pre_attn": "mhc_pre", "hc_post_attn": "mhc_post",
    "hc_pre_ffn": "mhc_pre", "hc_post_ffn": "mhc_post", "hc_head_module": "mhc_head",
    "e_proj": "matmul", "h_proj": "matmul",
}

_MODULE_CLASS_TO_KIND: dict[str, str] = {
    "RMSNorm": "rmsnorm", "LayerNorm": "ln",
    "LlamaRMSNorm": "rmsnorm", "Qwen2RMSNorm": "rmsnorm", "DeepseekV3RMSNorm": "rmsnorm",
    "Linear": "matmul", "ColumnParallelLinear": "matmul", "RowParallelLinear": "matmul",
    "Embedding": "embed", "ParallelEmbedding": "embed",
    "RotaryEmbedding": "rope", "LlamaRotaryEmbedding": "rope",
    "SiLU": "swiglu", "SwiGLU": "swiglu",
    "Attention": "attn_core", "Compressor": "compressor", "Indexer": "indexer",
    "Gate": "router", "Expert": "expert", "ParallelHead": "lm_head",
    "HCPreAttn": "mhc_pre", "HCPostAttn": "mhc_post",
    "HCPreFfn": "mhc_pre", "HCPostFfn": "mhc_post", "HCHead": "mhc_head",
}


def _infer_spec_kind(scope: str, module_class: str, component: str) -> str:
    """Infer block-level spec_kind from scope, module class, and component."""
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
    "Linear": "linear", "ColumnParallelLinear": "column_parallel_linear",
    "RowParallelLinear": "row_parallel_linear",
    "RMSNorm": "rms_norm", "LlamaRMSNorm": "rms_norm", "Qwen2RMSNorm": "rms_norm",
    "DeepseekV3RMSNorm": "rms_norm", "LayerNorm": "rms_norm",
    "ParallelEmbedding": "parallel_embedding", "Embedding": "parallel_embedding",
    "RotaryEmbedding": "rotary_emb", "LlamaRotaryEmbedding": "rotary_emb",
    "Compressor": "kv_compressor", "Indexer": "sparse_indexer",
    "Attention": "mla_sparse_attn", "Gate": "moe_gate", "Expert": "moe_expert_swiglu",
    "HCPreAttn": "hc_pre", "HCPreFfn": "hc_pre",
    "HCPostAttn": "hc_post", "HCPostFfn": "hc_post", "HCHead": "hc_head",
    "SiLU": "swiglu", "SwiGLU": "swiglu", "ParallelHead": "linear",
}


def _module_class_to_op_type(module_class: str, spec_kind: str) -> str:
    """Map module_class to FusionPass-compatible op_type."""
    if module_class and module_class in _MODULE_CLASS_TO_OP_TYPE:
        return _MODULE_CLASS_TO_OP_TYPE[module_class]
    _KIND_TO_OP = {
        "matmul": "linear", "attn_core": "mla_sparse_attn",
        "sparse_attn": "mla_sparse_attn", "hca_attn": "mla_sparse_attn",
        "swa_attn": "mla_sparse_attn", "rmsnorm": "rms_norm", "ln": "rms_norm",
        "rope": "rotary_emb", "swiglu": "swiglu", "embed": "parallel_embedding",
        "lm_head": "linear", "router": "moe_gate", "expert": "moe_expert_swiglu",
        "shared_expert": "moe_expert_swiglu", "indexer": "sparse_indexer",
        "compressor": "kv_compressor", "mhc_pre": "hc_pre", "mhc_post": "hc_post",
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
    """Extract a human-readable short name from a full scope path."""
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


def _extract_component(scope: str, nodes: list["OpNode"]) -> str:
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


def _build_aggregated_node(
    graph: "OpGraph",
    node_ids: list[str],
    scope: str,
    model_id: str = "",
) -> "OpNode":
    """Build one block-level OpNode from a group of aten-level nodes."""
    from python.zrt.ir.node import OpNode, infer_category
    
    nodes = [graph.nodes[nid] for nid in node_ids]
    if not nodes:
        raise ValueError(f"Empty node group for scope {scope!r}")
    
    first = nodes[0]
    last = nodes[-1]
    
    # Use the node's layer attribute instead of extracting from scope
    # This ensures nodes from different layers are not merged
    layer_id = first.layer or _extract_layer_id(scope)
    short_name = _extract_short_name(scope)
    base_id = f"L{layer_id}.{short_name}" if layer_id and layer_id != "-1" else short_name
    
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
    attrs["layer_id"] = int(layer_id) if layer_id and layer_id != "-1" else -1
    
    annotations: dict[str, Any] = {}
    for n in nodes:
        for k, v in n.annotations.items():
            if isinstance(v, (int, float)):
                annotations[k] = annotations.get(k, 0) + v
            elif k not in annotations:
                annotations[k] = v
    
    rule = _match_coarsen_rule(scope, module_class, model_id)
    if rule and not annotations.get("flops"):
        # Skip FLOPs computation here - let FlopsPass handle it later
        # The shape extraction is too simplistic for complex tensor shapes
        # and would produce incorrect FLOPs values
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


# ---------------------------------------------------------------------------
# FusionPass
# ---------------------------------------------------------------------------

class FusionPass(GraphPass):
    """Apply MRO-based fusion to an OpGraph.

    Rules are loaded from ``registry/platforms/`` based on the model
    classes found in the OpGraph's node metadata.
    
    After rich fusion rules, remaining aten ops are grouped by scope
    into block-level nodes using FusionPass-compatible op_type naming.
    """

    name = "fusion"

    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        from python.zrt.transform.fusion.loading import initialize_rules
        from python.zrt.transform.fusion.loading.fusion_config import (
            resolve_fusion_config,
        )
        from python.zrt.transform.fusion.pipeline.fuser import MultiPassFuser
        from python.zrt.transform.fusion.registry import default_registry

        initialize_rules(getattr(ctx, "model_id", "") or "")

        # Resolve fusion config from disk only when the caller hasn't
        # explicitly set one — CLI / tests that pre-populate ctx.fusion
        # win over auto-discovery.
        existing = getattr(ctx, "fusion", None)
        is_default = (
            existing is None
            or (
                existing.enabled_rules is None
                and not existing.disabled_rules
                and not existing.allow_structural_collapse
                and not existing.merge_sibling_classes
            )
        )
        if is_default:
            phase = ctx.phase_for_fusion() if ctx is not None else "inference"
            ctx.fusion = resolve_fusion_config(
                getattr(ctx, "model_id", "") or "", phase, explicit_path=None,
            )

        fuser = MultiPassFuser(registry=default_registry())
        fused_graph = fuser.fuse(graph, ctx)
        
        # Post-processing: coarsen remaining aten ops by scope
        return self._coarsen_remaining_ops(fused_graph, ctx)
    
    def _coarsen_remaining_ops(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        """Group remaining aten.* ops by scope into block-level nodes.
        
        This runs after rich fusion rules to handle ops that weren't matched
        by ordered_regex or dag_signature patterns.
        """
        from python.zrt.ir.graph import OpGraph
        from python.zrt.ir.edge import Edge
        
        # Skip coarsening if fusion config has enabled_rules or disabled_rules set
        # (coarsening should only run when fusion is doing full fusion)
        fusion_cfg = getattr(ctx, "fusion", None)
        if fusion_cfg and (fusion_cfg.enabled_rules is not None or fusion_cfg.disabled_rules):
            return graph
        
        # Skip coarsening if structural collapse is not enabled
        if fusion_cfg and not fusion_cfg.allow_structural_collapse:
            return graph
        
        model_id = getattr(ctx, "model_id", "") or ""
        
        # Identify remaining aten ops (not yet fused)
        aten_nodes = [
            n for n in graph.nodes.values()
            if n.op_type.startswith("aten.") and not n.is_fused
        ]
        
        if not aten_nodes:
            return graph  # Nothing to coarsen
        
        # Skip coarsening for single-node graphs or graphs with no scope
        if len(aten_nodes) == 1:
            return graph
        
        # Check if all aten nodes have no scope
        has_scope = any(n.scope for n in aten_nodes)
        if not has_scope:
            return graph
        
        g = graph.clone()
        
        # Build scope groups for aten ops only
        scope_groups = _build_scope_groups_for_nodes(g, aten_nodes)
        
        # Create new coarsened nodes
        nid_map: dict[str, str] = {}
        new_nodes = []
        
        # Keep all non-aten nodes (fused, comm, etc.)
        for node in g.nodes.values():
            if not node.op_type.startswith("aten.") or node.is_fused:
                nid_map[node.id] = node.id
                new_nodes.append(node)
        
        # Coarsen aten ops by scope
        for group_key, node_ids in scope_groups.items():
            if group_key.startswith("__comm__"):
                # Keep comm nodes as-is
                nid = node_ids[0]
                nid_map[nid] = nid
                new_nodes.append(g.nodes[nid])
            else:
                # Aggregate aten ops into block-level node
                agg = _build_aggregated_node(g, node_ids, group_key, model_id)
                for nid in node_ids:
                    nid_map[nid] = agg.id
                new_nodes.append(agg)
        
        # Rebuild edges with new node IDs
        orig_order = {nid: idx for idx, nid in enumerate(g.nodes)}
        seen_edges: set[tuple[str, str]] = set()
        new_edges = []
        
        for e in g.edges:
            new_src = nid_map.get(e.src, e.src)
            new_dst = nid_map.get(e.dst, e.dst)
            
            if new_src == new_dst:
                continue  # Skip self-edges
            
            key = (new_src, new_dst)
            if key in seen_edges:
                continue
            seen_edges.add(key)
            
            new_edges.append(Edge(
                src=new_src, src_idx=0,
                dst=new_dst, dst_idx=0,
                tensor=e.tensor, tensor_id=e.tensor_id,
            ))
        
        # Break cycles if needed
        new_edges = _break_cycles(new_edges, orig_order, nid_map)
        
        # Build new graph
        coarsened = OpGraph(
            name=g.name,
            phase=g.phase,
            nodes={n.id: n for n in new_nodes},
            edges=new_edges,
            metadata=dict(g.metadata),
        )
        coarsened.metadata["coarsened"] = True
        
        return coarsened
