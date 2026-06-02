"""LayerScalingPass: scale typical layers to full model BEFORE TP/EP sharding.

This pass addresses the pipeline ordering bug where TP/EP sharding happens
BEFORE layer scaling, causing incorrect FLOPs/memory calculations.

Strategy: Pre-compute scaled metadata instead of cloning nodes (expensive).

Design rationale:
- Cloning 5-layer graph to 61 layers would create 12x more nodes/edges
- Analyze passes only need metadata (total_params, training_flops) to be scaled
- TP/EP/PP passes shard based on annotation and metadata
- By setting metadata BEFORE split stage, downstream passes get correct values

Workflow with this pass:
1. Capture typical layers (5 layers)
2. LayerScalingPass computes scaled metadata (params, FLOPs)
3. TP/EP/PP passes shard the 5-layer graph BUT use scaled metadata
4. Analyze passes use metadata (correct) + node annotations (correct layer-type scaling)

Key insight:
- Node annotations (latency_us, flops_fwd) are computed per-node
- Layer-type scaling (training.py:1195) uses layer_profile correctly
- Only count_params() and FLOPs aggregation need pre-scaling
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from python.zrt.ir.param_count import count_params_by_component, compute_layer_scale
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


class LayerScalingPass(GraphPass):
    """Scale typical layer metadata to full model BEFORE TP/EP sharding.
    
    This pass:
    1. Checks if metadata["total_params"] already set (authoritative value)
    2. Reads layer_profile and typical_indices from graph metadata
    3. Computes layer_scale = num_layers / len(typical_indices)
    4. Pre-computes scaled total_params by component (per-layer params only)
    5. Stores in metadata for downstream passes
    
    Critical: Must run BEFORE TensorParallelPass/ExpertParallelPass
    """
    
    @property
    def name(self) -> str:
        return "layer_scaling"
    
    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        
        # ── Guard: Skip if authoritative total_params already set ───────────────
        # If model_loader or CLI --total-params already set metadata["total_params"],
        # that value is authoritative (e.g., DSV4 = 168B). Do not overwrite it.
        # Old code had has_param_override guard, restore it here.
        if g.metadata.get("total_params", 0) > 0:
            logger.debug(
                "LayerScalingPass: skipping (metadata['total_params'] already set = %d)",
                g.metadata["total_params"]
            )
            return g
        
        # ── Compute layer_scale using unified helper ──────────────────────────
        layer_scale = compute_layer_scale(g)
        
        if layer_scale == 1.0:
            logger.debug(
                "LayerScalingPass: no scaling needed (layer_scale=1.0)"
            )
            return g
        
        logger.info(
            "LayerScalingPass: scaling typical layers (scale=%.2f)",
            layer_scale
        )
        
        # ── Pre-compute scaled params BEFORE TP/EP sharding ───────────────────
        # Use count_params_by_component() to separate per-layer vs non-layer params.
        # Per-layer params (routed_expert, shared_expert, other) should be scaled.
        # Non-layer params (embedding, lm_head, norm) should NOT be scaled.
        # This fixes Known Bug #68: embedding params overestimated.
        comp_params = count_params_by_component(g)
        
        # Scale only per-layer components
        scaled_routed = int(comp_params.routed_expert * layer_scale)
        scaled_shared = int(comp_params.shared_expert * layer_scale)
        scaled_other = int(comp_params.other * layer_scale)
        # non_layer (embedding, lm_head) stays as-is
        
        total_scaled = scaled_routed + scaled_shared + scaled_other + comp_params.non_layer
        typical_total = comp_params.total
        
        g.metadata["total_params"] = total_scaled
        g.metadata["typical_params"] = typical_total  # for diagnostics
        g.metadata["layer_scale"] = layer_scale
        
        # Store component breakdown for downstream passes
        g.metadata["params_by_component"] = {
            "routed_expert": scaled_routed,
            "shared_expert": scaled_shared,
            "other": scaled_other,
            "non_layer": comp_params.non_layer,
        }
        
        # ── Mark that scaling is complete ─────────────────────────────────────
        g.metadata["layer_scaling_complete"] = True
        
        logger.info(
            "LayerScalingPass: params scaled from %d to %d "
            "(per-layer: %d->%d, non-layer: %d unchanged)",
            typical_total, total_scaled,
            (comp_params.routed_expert + comp_params.shared_expert + comp_params.other),
            (scaled_routed + scaled_shared + scaled_other),
            comp_params.non_layer
        )
        
        return g