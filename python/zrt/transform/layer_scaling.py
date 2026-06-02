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

from python.zrt.ir.param_count import count_params, compute_layer_scale
from python.zrt.transform.base import GraphPass

if TYPE_CHECKING:
    from python.zrt.ir.graph import OpGraph
    from python.zrt.transform.context import TransformContext

logger = logging.getLogger(__name__)


class LayerScalingPass(GraphPass):
    """Scale typical layer metadata to full model BEFORE TP/EP sharding.
    
    This pass:
    1. Reads layer_profile and typical_indices from graph metadata
    2. Computes layer_scale = num_layers / len(typical_indices)
    3. Pre-computes scaled total_params
    4. Stores in metadata for downstream passes
    
    Critical: Must run BEFORE TensorParallelPass/ExpertParallelPass
    """
    
    @property
    def name(self) -> str:
        return "layer_scaling"
    
    def run(self, graph: "OpGraph", ctx: "TransformContext") -> "OpGraph":
        g = graph.clone()
        
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
        # count_params() on the current graph (typical layers, not yet sharded)
        typical_params = count_params(g)
        scaled_params = int(typical_params * layer_scale)
        
        g.metadata["total_params"] = scaled_params
        g.metadata["typical_params"] = typical_params  # for diagnostics
        g.metadata["layer_scale"] = layer_scale
        
        # ── Mark that scaling is complete ─────────────────────────────────────
        g.metadata["layer_scaling_complete"] = True
        
        logger.info(
            "LayerScalingPass: params scaled from %d to %d",
            typical_params, scaled_params
        )
        
        return g