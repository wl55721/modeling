"""Parameter counting utilities for OpGraph IR.

Shared by inference reports and training analysis passes.
Three-tier strategy: metadata → name heuristic → structural fallback.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph


def compute_layer_scale(graph: "OpGraph") -> float:
    """Compute layer scaling factor from typical_indices or num_layers ratio.
    
    Priority:
    1. graph.metadata["layer_scale"] — if already set by LayerScalingPass
    2. num_layers / num_layers_traced — accurate when traced layer count is known
    3. num_layers / (max(typical_indices) + 1) — fallback when typical_indices present
    
    Note: len(typical_indices) is NOT used because it doesn't reflect actual traced layers.
    Example: DeepSeek-V3 typical_indices=[0, 56] (2 indices) but num_layers_traced=4.
    Using len would give 61/2=30.5 (2x overestimate), correct is 61/4=15.25.
    
    Returns 1.0 if no scaling needed or information unavailable.
    """
    layer_scale = graph.metadata.get("layer_scale", 0.0)
    if layer_scale > 0.0:
        return layer_scale
    
    num_layers = graph.metadata.get("num_layers", 0)
    if num_layers == 0:
        return 1.0
    
    # Priority 2: Use num_layers_traced (most accurate)
    num_layers_traced = graph.metadata.get("num_layers_traced", None)
    if num_layers_traced is not None and num_layers_traced > 0:
        if num_layers != num_layers_traced:
            return num_layers / num_layers_traced
        else:
            return 1.0  # No scaling needed
    
    # Priority 3: Fallback to typical_indices estimate
    # Use max(typical_indices) + 1 as traced layer count estimate
    typical_indices = graph.metadata.get("typical_indices", None)
    if typical_indices is not None and len(typical_indices) > 0:
        estimated_traced = max(typical_indices) + 1
        if estimated_traced > 0 and num_layers != estimated_traced:
            return num_layers / estimated_traced
    
    return 1.0


@dataclass
class ComponentParams:
    """Parameter counts split by component category."""
    routed_expert: int = 0
    shared_expert: int = 0
    other: int = 0
    non_layer: int = 0  # embedding, lm_head, final_norm — not layer-scaled

    @property
    def total(self) -> int:
        return self.routed_expert + self.shared_expert + self.other + self.non_layer

# Short op-name tokens that consume weight matrices.
# Mapped to the first input index that is a weight (inputs before it are activations/bias).
_MATMUL_WEIGHT_START: dict[str, int] = {
    "mm": 1, "matmul": 1, "linear": 1, "bmm": 1, "baddbmm": 2, "addmm": 2,
}
# Embedding ops: input[0] is the weight table, input[1] is the index tensor
_EMBED_OPS: frozenset[str] = frozenset({"embedding"})


def op_short(op_type: str) -> str:
    """Extract the short token from a qualified op name like 'aten.mm.default' -> 'mm'."""
    parts = op_type.split(".")
    return parts[1] if len(parts) >= 2 else parts[0]


def count_params(graph: OpGraph, apply_layer_scale: bool = False) -> int:
    """Count model parameters from an OpGraph.

    Three-tier strategy, tried in order:
    1. graph.metadata["total_params"] — authoritative when set by a model loader
       (or LayerScalingPass for pre-scaled params)
    2. Name heuristic — tensor IDs containing "weight" or "param" (synthetic graphs)
    3. Structural fallback — external 2-D inputs to matmul/embedding ops, skipping
       activation positions that differ per op type (captured graphs use opaque IDs)

    If ``apply_layer_scale=True`` and metadata["total_params"] is 0, applies
    layer_scale to the counted params (useful for training FLOPs calculation).
    """
    if graph.metadata.get("total_params", 0) > 0:
        return int(graph.metadata["total_params"])

    counted_ids: set[str] = set()
    name_total = 0
    for node in graph.nodes.values():
        if node.category == "compute":
            for inp in node.inputs:
                if inp.id in counted_ids:
                    continue
                if ("weight" in inp.id or "param" in inp.id) and inp.shape:
                    counted_ids.add(inp.id)
                    name_total += math.prod(inp.shape)
    if name_total > 0:
        return name_total

    produced_ids: set[str] = set()
    for node in graph.nodes.values():
        for out in node.outputs:
            produced_ids.add(out.id)

    counted_ids = set()
    struct_total = 0
    for node in graph.nodes.values():
        if node.category != "compute":
            continue
        short = op_short(node.op_type)
        weight_start = _MATMUL_WEIGHT_START.get(short)
        is_embed = short in _EMBED_OPS
        if weight_start is None and not is_embed:
            continue

        for i, inp in enumerate(node.inputs):
            if inp.id in produced_ids or inp.id in counted_ids:
                continue
            if weight_start is not None and i < weight_start:
                continue
            if is_embed and i > 0:
                continue  # only input[0] is the embedding table
            if inp.shape and len(inp.shape) == 2:
                counted_ids.add(inp.id)
                struct_total += math.prod(inp.shape)
    
    if apply_layer_scale and struct_total > 0:
        layer_scale = compute_layer_scale(graph)
        if layer_scale != 1.0:
            struct_total = int(struct_total * layer_scale)
    
    return struct_total


def _classify_node_component(node) -> str:
    """Classify a node using the shared component classifier."""
    from python.zrt.ir.component_classifier import classify
    return classify(node)


def count_params_by_component(graph: "OpGraph") -> ComponentParams:
    """Count parameters split by component: routed_expert, shared_expert, other, non_layer.

    Uses the same three-tier strategy as ``count_params`` but classifies each
    parameter tensor by the node's component/scope.

    Tier-1 metadata override: when ``graph.metadata["total_params"]`` is set
    (authoritative from model loader), the total is split by the fraction of
    per-node parameters in each component bucket.
    """
    # Tier 0: metadata override — total_params is authoritative
    meta_total = graph.metadata.get("total_params", 0)
    if meta_total > 0:
        return _component_split_from_meta(graph, meta_total)

    produced_ids: set[str] = set()
    for node in graph.nodes.values():
        for out in node.outputs:
            produced_ids.add(out.id)

    counted_ids: set[str] = set()
    result = ComponentParams()

    # Tier 1 & 2: name heuristic
    name_counts = ComponentParams()
    for node in graph.nodes.values():
        if node.category != "compute":
            continue
        for inp in node.inputs:
            if inp.id in counted_ids:
                continue
            if ("weight" in inp.id or "param" in inp.id) and inp.shape:
                counted_ids.add(inp.id)
                n = math.prod(inp.shape)
                _accumulate_by_component(node, n, name_counts)

    if name_counts.total > 0:
        return name_counts

    # Tier 3: structural fallback
    counted_ids.clear()
    for node in graph.nodes.values():
        if node.category != "compute":
            continue
        short = op_short(node.op_type)
        weight_start = _MATMUL_WEIGHT_START.get(short)
        is_embed = short in _EMBED_OPS
        if weight_start is None and not is_embed:
            continue
        for i, inp in enumerate(node.inputs):
            if inp.id in produced_ids or inp.id in counted_ids:
                continue
            if weight_start is not None and i < weight_start:
                continue
            if is_embed and i > 0:
                continue
            if inp.shape and len(inp.shape) == 2:
                counted_ids.add(inp.id)
                n = math.prod(inp.shape)
                _accumulate_by_component(node, n, result)
    return result


def _accumulate_by_component(node, n: int, result: ComponentParams) -> None:
    """Add ``n`` params to the correct bucket in ``result``."""
    comp = _classify_node_component(node)
    if comp == "routed_expert":
        result.routed_expert += n
    elif comp == "shared_expert":
        result.shared_expert += n
    elif comp in ("embedding", "norm"):
        result.non_layer += n
    else:
        result.other += n


def _component_split_from_meta(graph: "OpGraph", total: int) -> ComponentParams:
    """When total_params metadata is authoritative, split by per-node fraction."""
    # Count per-node params using structural/name heuristics to get fractions,
    # then scale to the authoritative total.
    raw = ComponentParams()
    produced_ids: set[str] = set()
    for node in graph.nodes.values():
        for out in node.outputs:
            produced_ids.add(out.id)

    counted_ids: set[str] = set()
    struct_total = 0

    # Name heuristic pass
    for node in graph.nodes.values():
        if node.category != "compute":
            continue
        for inp in node.inputs:
            if inp.id in counted_ids:
                continue
            if ("weight" in inp.id or "param" in inp.id) and inp.shape:
                counted_ids.add(inp.id)
                n = math.prod(inp.shape)
                _accumulate_by_component(node, n, raw)
                struct_total += n

    # Structural fallback if name heuristic found nothing
    if struct_total == 0:
        counted_ids.clear()
        for node in graph.nodes.values():
            if node.category != "compute":
                continue
            short = op_short(node.op_type)
            weight_start = _MATMUL_WEIGHT_START.get(short)
            is_embed = short in _EMBED_OPS
            if weight_start is None and not is_embed:
                continue
            for i, inp in enumerate(node.inputs):
                if inp.id in produced_ids or inp.id in counted_ids:
                    continue
                if weight_start is not None and i < weight_start:
                    continue
                if is_embed and i > 0:
                    continue
                if inp.shape and len(inp.shape) == 2:
                    counted_ids.add(inp.id)
                    n = math.prod(inp.shape)
                    _accumulate_by_component(node, n, raw)
                    struct_total += n

    if struct_total == 0:
        return ComponentParams(other=total)

    # Scale raw counts to authoritative total
    scale = total / struct_total
    return ComponentParams(
        routed_expert=round(raw.routed_expert * scale),
        shared_expert=round(raw.shared_expert * scale),
        other=round(raw.other * scale),
        non_layer=round(raw.non_layer * scale),
    )
