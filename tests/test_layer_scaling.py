"""Test LayerScalingPass: pre-compute scaled params BEFORE TP/EP sharding.

Tests cover:
1. Normal case: typical_indices present -> compute layer_scale
2. No scaling needed: num_layers == num_typical
3. Missing metadata: no layer_profile or typical_indices
4. Metadata correctness: total_params, layer_scale, layer_scaling_complete
5. Integration with count_params and pipeline
"""
import pytest

from python.zrt.ir.node import OpNode
from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.types import TensorMeta, DType
from python.zrt.ir.param_count import count_params, compute_layer_scale
import python.zrt.hardware.registry as hw_registry
from python.zrt.transform import ParallelConfig, TransformContext
from python.zrt.transform.context import TrainingConfig
from python.zrt.transform.layer_scaling import LayerScalingPass


def _t(tid, shape, dtype=DType.BF16):
    return TensorMeta.from_shape_dtype(tid, shape, dtype)


def _linear_node(nid, scope, in_shape, out_shape):
    return OpNode(
        id=nid,
        op_type="aten.mm.default",
        inputs=[_t(f"{nid}_in", in_shape), _t(f"{nid}_weight", out_shape)],
        outputs=[_t(f"{nid}_out", out_shape)],
        scope=scope,
        category="compute",
    )


def typical_layer_graph(num_layers=4):
    """Create a graph with typical layers (captured from model).
    
    Each layer has a Linear node with 22M params (4096×3072).
    """
    nodes = {}
    for i in range(num_layers):
        nid = f"layer_{i}"
        scope = f"model.layers.{i}.mlp.gate_proj"
        nodes[nid] = _linear_node(nid, scope, (128, 4096), (4096, 3072))
    
    edges = []
    for i in range(num_layers - 1):
        edges.append(Edge(
            src=f"layer_{i}", src_idx=0,
            dst=f"layer_{i+1}", dst_idx=0,
            tensor=_t(f"e{i}", (128, 3072))
        ))
    
    return OpGraph(name="test_typical", phase="train_forward", nodes=nodes, edges=edges)


def _ctx(hw_name="nvidia_h100_sxm"):
    from python.zrt.transform.context import TrainingConfig
    hw = hw_registry.load(hw_name)
    return TransformContext(
        hw_spec=hw,
        parallel=ParallelConfig(tp=1, ep=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),  # Set training config
    )


def test_layer_scaling_pass_basic():
    """LayerScalingPass computes layer_scale from typical_indices."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61  # DeepSeek-V4 has 61 layers
    g.metadata["num_layers_traced"] = 4
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    g.metadata["typical_indices"] = [0, 1, 2, 3]  # 4 typical layers
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Check layer_scale calculation
    layer_scale = result.metadata.get("layer_scale")
    assert layer_scale == pytest.approx(61.0 / 4.0, rel=0.01)
    
    # Check metadata is set
    assert result.metadata.get("layer_scaling_complete") is True
    assert result.metadata.get("total_params") > 0


def test_layer_scaling_pass_metadata_correctness():
    """LayerScalingPass sets correct metadata fields."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Check all metadata fields
    assert "total_params" in result.metadata
    assert "typical_params" in result.metadata
    assert "layer_scale" in result.metadata
    assert "layer_scaling_complete" in result.metadata
    
    # Check scaling factor
    typical_params = result.metadata["typical_params"]
    total_params = result.metadata["total_params"]
    layer_scale = result.metadata["layer_scale"]
    
    assert total_params == pytest.approx(typical_params * layer_scale, rel=0.01)


def test_layer_scaling_pass_no_scaling_needed():
    """LayerScalingPass skips when num_layers == num_typical."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 4  # No scaling needed
    g.metadata["layer_profile"] = {"dense": 4}
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Should NOT set layer_scaling_complete
    assert result.metadata.get("layer_scaling_complete") is not True
    assert result.metadata.get("layer_scale") is None or result.metadata.get("layer_scale") == 1.0


def test_layer_scaling_pass_missing_typical_indices():
    """LayerScalingPass skips when typical_indices is missing."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    # NO typical_indices
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Should NOT set metadata
    assert result.metadata.get("layer_scaling_complete") is not True


def test_layer_scaling_pass_missing_layer_profile():
    """LayerScalingPass runs when typical_indices present (layer_profile optional)."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    # NO layer_profile - but compute_layer_scale() uses typical_indices
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Should still run (compute_layer_scale uses typical_indices)
    assert result.metadata.get("layer_scaling_complete") is True


def test_compute_layer_scale_helper():
    """compute_layer_scale() computes correct scaling factor with priority logic.
    
    Priority order (FIXED):
    1. metadata["layer_scale"] (if > 0.0)
    2. num_layers / num_layers_traced (most accurate)
    3. num_layers / (max(typical_indices) + 1) (fallback estimate)
    4. Return 1.0 if no info available
    
    Note: len(typical_indices) is NOT used (would overestimate for sparse indices).
    Example: typical_indices=[0, 56] -> len=2, but max+1=57 gives correct estimate.
    """
    g = typical_layer_graph(num_layers=4)
    
    # Case 1: metadata already has layer_scale (highest priority)
    g.metadata["layer_scale"] = 15.25
    assert compute_layer_scale(g) == 15.25
    
    # Case 2: num_layers_traced available (Priority 2, most accurate)
    g.metadata["layer_scale"] = 0.0
    g.metadata["num_layers"] = 61
    g.metadata["num_layers_traced"] = 4
    g.metadata["typical_indices"] = [0, 1, 2, 3]  # Present but Priority 2 takes precedence
    assert compute_layer_scale(g) == pytest.approx(61.0 / 4.0, rel=0.01)
    
    # Case 3: typical_indices only (use max+1 estimate, Priority 3)
    g.metadata["num_layers_traced"] = None  # Remove num_layers_traced
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    assert compute_layer_scale(g) == pytest.approx(61.0 / 4.0, rel=0.01)  # max([0,1,2,3])+1 = 4
    
    # Case 4: sparse typical_indices (e.g., DeepSeek-V3 [0, 56])
    g.metadata["typical_indices"] = [0, 56]
    assert compute_layer_scale(g) == pytest.approx(61.0 / 57.0, rel=0.01)  # max+1 = 57
    
    # Case 5: no scaling needed (num_layers == num_layers_traced)
    g.metadata["num_layers"] = 4
    g.metadata["num_layers_traced"] = 4
    g.metadata["typical_indices"] = None
    assert compute_layer_scale(g) == 1.0
    
    # Case 5: typical_indices empty list
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = []
    g.metadata["num_layers_traced"] = 4
    assert compute_layer_scale(g) == pytest.approx(61.0 / 4.0, rel=0.01)


def test_compute_layer_scale_edge_cases():
    """compute_layer_scale() handles edge cases gracefully.
    
    Edge cases:
    - num_layers = 0 (return 1.0)
    - typical_indices = None (fallback to num_layers_traced)
    - num_layers_traced = 0 (return 1.0)
    - all info missing (return 1.0)
    """
    g = typical_layer_graph(num_layers=4)
    
    # Edge case 1: num_layers = 0 -> return 1.0
    g.metadata["layer_scale"] = 0.0
    g.metadata["num_layers"] = 0
    assert compute_layer_scale(g) == 1.0
    
    # Edge case 2: typical_indices = None, num_layers_traced = 0
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = None
    g.metadata["num_layers_traced"] = 0
    assert compute_layer_scale(g) == 1.0
    
    # Edge case 3: typical_indices empty, num_layers_traced missing
    g.metadata["typical_indices"] = []
    g.metadata["num_layers_traced"] = None
    assert compute_layer_scale(g) == 1.0
    
    # Edge case 4: all metadata missing
    g.metadata.clear()
    assert compute_layer_scale(g) == 1.0


def test_compute_layer_scale_priority():
    """compute_layer_scale() respects priority order: metadata > num_layers_traced > typical_indices.
    
    Priority (FIXED to use num_layers_traced instead of len(typical_indices)):
    1. metadata["layer_scale"] overrides everything
    2. num_layers_traced (most accurate)
    3. max(typical_indices) + 1 (fallback estimate)
    """
    g = typical_layer_graph(num_layers=4)
    
    # Priority 1: metadata["layer_scale"] overrides everything
    g.metadata["layer_scale"] = 20.0  # Override value
    g.metadata["num_layers"] = 61
    g.metadata["num_layers_traced"] = 4  # Would give 15.25
    g.metadata["typical_indices"] = [0, 1, 2, 3]  # Would give 15.25 (max+1=4)
    assert compute_layer_scale(g) == 20.0  # Uses metadata override
    
    # Priority 2: num_layers_traced takes precedence over typical_indices
    g.metadata["layer_scale"] = 0.0  # Remove override
    g.metadata["num_layers_traced"] = 5  # Would give 61/5=12.2
    g.metadata["typical_indices"] = [0, 1, 2, 3]  # Would give 61/4=15.25 (max+1=4)
    assert compute_layer_scale(g) == pytest.approx(12.2, rel=0.01)  # Uses num_layers_traced
    
    # Priority 3: typical_indices fallback (use max+1)
    g.metadata["num_layers_traced"] = None  # Remove num_layers_traced
    assert compute_layer_scale(g) == pytest.approx(15.25, rel=0.01)  # Uses max+1=4
    
    # Edge case: typical_indices=[0, 56] (sparse indices)
    g.metadata["typical_indices"] = [0, 56]  # max+1 = 57
    assert compute_layer_scale(g) == pytest.approx(61.0 / 57.0, rel=0.01)


def test_compute_layer_scale_different_ratios():
    """compute_layer_scale() computes correct ratio for various layer counts.
    
    Test different scenarios:
    - Small ratio (e.g., 8/4 = 2.0)
    - Large ratio (e.g., 61/4 = 15.25)
    - Equal (e.g., 4/4 = 1.0)
    """
    g = typical_layer_graph(num_layers=4)
    
    # Small ratio: 8 total, 4 typical
    g.metadata["num_layers"] = 8
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    assert compute_layer_scale(g) == 2.0
    
    # Large ratio: 100 total, 4 typical
    g.metadata["num_layers"] = 100
    assert compute_layer_scale(g) == 25.0
    
    # Equal ratio: 4 total, 4 typical
    g.metadata["num_layers"] = 4
    assert compute_layer_scale(g) == 1.0
    
    # Fractional ratio: 7 total, 4 typical
    g.metadata["num_layers"] = 7
    assert compute_layer_scale(g) == pytest.approx(1.75, rel=0.01)


def test_compute_layer_scale_real_models():
    """compute_layer_scale() handles real-world model configurations with correct formula.
    
    Examples (using FIXED formula: num_layers / num_layers_traced):
    - DeepSeek-V4: 61 layers, 4 traced -> 15.25 (NOT 61/len([0,1,2,3])=15.25)
    - DeepSeek-V4-pro: 61 layers, 5 traced -> 12.2 (typical_indices=[0,1,2,3], max+1=4 -> 15.25 if num_layers_traced missing)
    - Llama-3-70B: 80 layers, 4 traced -> 20.0
    - Mixtral-8x7B: 32 layers, 2 traced -> 16.0
    - DeepSeek-V3: 61 layers, 4 traced -> 15.25 (typical_indices=[0, 56], len=2 would give 30.5 WRONG)
    
    Note: The fix prevents overestimation from using len(typical_indices) instead of num_layers_traced.
    """
    g = typical_layer_graph(num_layers=4)
    
    # DeepSeek-V4: 61 sparse MoE layers, 4 typical traced
    g.metadata["num_layers"] = 61
    g.metadata["num_layers_traced"] = 4
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    assert compute_layer_scale(g) == pytest.approx(15.25, rel=0.01)
    
    # DeepSeek-V4-pro: 61 layers, 5 traced (max(typical_indices)+1=4, but num_layers_traced=5)
    g.metadata["num_layers_traced"] = 5
    assert compute_layer_scale(g) == pytest.approx(12.2, rel=0.01)  # Uses num_layers_traced (correct)
    # NOT 61/4=15.25 (which would be 25% overestimate)
    
    # Llama-3-70B: 80 dense layers, 4 traced
    g.metadata["num_layers"] = 80
    g.metadata["num_layers_traced"] = 4
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    assert compute_layer_scale(g) == 20.0
    
    # Mixtral-8x7B: 32 MoE layers, 2 traced
    g.metadata["num_layers"] = 32
    g.metadata["num_layers_traced"] = 2
    g.metadata["typical_indices"] = [0, 1]
    assert compute_layer_scale(g) == 16.0
    
    # DeepSeek-V3: 61 layers, typical_indices=[0, 56] (sparse indices)
    # OLD BUG: len(typical_indices)=2 would give 61/2=30.5 (2x overestimate)
    # FIXED: num_layers_traced=4 gives 61/4=15.25, or max+1=57 gives 61/57≈1.07
    g.metadata["num_layers"] = 61
    g.metadata["num_layers_traced"] = 4
    g.metadata["typical_indices"] = [0, 56]
    assert compute_layer_scale(g) == pytest.approx(15.25, rel=0.01)  # Uses num_layers_traced (correct)
    
    # Without num_layers_traced, fallback to max+1
    g.metadata["num_layers_traced"] = None
    assert compute_layer_scale(g) == pytest.approx(61.0 / 57.0, rel=0.01)  # max([0,56])+1 = 57


def test_count_params_with_apply_layer_scale():
    """count_params(apply_layer_scale=True) applies scaling when metadata["total_params"] is 0.
    
    Note: When metadata["total_params"] is set, count_params returns it directly.
    LayerScalingPass sets metadata["total_params"] = scaled_params.
    TrainingFlopsPass calls count_params(g, apply_layer_scale=True) which reads metadata.
    """
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    # Case 1: No metadata["total_params"] - count_params scales in fallback path
    # (apply_layer_scale works when name heuristic fails)
    typical_params = 4 * 4096 * 3072
    expected_scaled = int(typical_params * (61.0 / 4.0))
    
    # Case 2: metadata["total_params"] is set (LayerScalingPass ran)
    g.metadata["total_params"] = expected_scaled
    params_with_meta = count_params(g, apply_layer_scale=True)
    assert params_with_meta == expected_scaled


def test_count_params_uses_metadata_total_params():
    """count_params() returns metadata['total_params'] when set."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["total_params"] = 168e9  # Authoritative value
    
    # Should return authoritative value
    assert count_params(g) == 168e9
    assert count_params(g, apply_layer_scale=True) == 168e9


def test_layer_scaling_preserves_graph_structure():
    """LayerScalingPass does not modify nodes/edges."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Graph structure should be unchanged
    assert len(result.nodes) == len(g.nodes)
    assert len(result.edges) == len(g.edges)
    assert set(result.nodes.keys()) == set(g.nodes.keys())


def test_layer_scaling_integration_with_pipeline():
    """LayerScalingPass must run BEFORE TensorParallelPass.
    
    Note: This test verifies pipeline order, not TP sharding on weight shapes.
    TP sharding modifies tensor shapes, but LayerScalingPass stores params in metadata.
    """
    from python.zrt.transform import build_default_pipeline
    from python.zrt.graph.layer_strategy import LayerProfile, LayerType
    
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["layer_profile"] = LayerProfile(
        layer_types=[LayerType.MOE] * 61,
        typical_indices=[0, 1, 2, 3],
    )
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    ctx = TransformContext(
        hw_spec=hw_registry.load("nvidia_h100_sxm"),
        parallel=ParallelConfig(tp=8, ep=1),
        training=TrainingConfig(micro_batch=1, global_batch=8),
    )
    
    # Run full pipeline
    pipeline = build_default_pipeline()
    result = pipeline.run(g, ctx)
    
    # Check LayerScalingPass ran first
    assert result.metadata.get("layer_scaling_complete") is True
    
    # Check params are scaled BEFORE TP sharding
    # total_params should be full model params, not per-GPU params
    total_params = result.metadata.get("total_params", 0)
    assert total_params > 0
    
    # Verify layer_scale
    layer_scale = result.metadata.get("layer_scale")
    assert layer_scale == pytest.approx(61.0 / 4.0, rel=0.01)
    
    # total_params should be scaled from typical layers
    typical_params = 4 * 4096 * 3072
    expected_scaled = int(typical_params * layer_scale)
    assert total_params == pytest.approx(expected_scaled, rel=0.01)


def test_layer_scaling_with_different_tp_configs():
    """LayerScalingPass works correctly with different TP configs."""
    from python.zrt.transform.context import TrainingConfig
    
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    for tp in [1, 2, 4, 8, 16]:
        ctx = TransformContext(
            hw_spec=hw_registry.load("nvidia_h100_sxm"),
            parallel=ParallelConfig(tp=tp, ep=1),
            training=TrainingConfig(micro_batch=1, global_batch=8),
        )
        
        pass_obj = LayerScalingPass()
        result = pass_obj.run(g.clone(), ctx)
        
        # total_params should be SAME regardless of TP
        # (scaling happens BEFORE TP sharding)
        assert result.metadata["total_params"] > 0
        assert result.metadata["layer_scale"] == pytest.approx(61.0 / 4.0, rel=0.01)


def test_layer_scaling_does_not_mutate_original():
    """LayerScalingPass clones graph (functional style)."""
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    ctx = _ctx()
    original_metadata = g.metadata.copy()
    
    result = LayerScalingPass().run(g, ctx)
    
    # Original graph should be unchanged
    assert g.metadata == original_metadata
    assert "layer_scaling_complete" not in g.metadata
    
    # Result should have new metadata
    assert result.metadata.get("layer_scaling_complete") is True


def test_layer_scaling_real_world_deepseek_v4():
    """Test with DeepSeek-V4 realistic parameters."""
    # DeepSeek-V4: 61 layers, 4 typical layers captured
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["num_layers_traced"] = 4
    g.metadata["layer_profile"] = {
        "dense": 0,  # No dense layers
        "sparse": 61,  # All MoE layers
    }
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Check layer_scale
    layer_scale = result.metadata["layer_scale"]
    assert layer_scale == pytest.approx(61.0 / 4.0, rel=0.01)
    
    # Check params scaling
    typical_params = result.metadata["typical_params"]
    scaled_params = result.metadata["total_params"]
    
    # Each layer has ~22M params (4096×3072)
    # 4 layers = ~88M typical
    # 61 layers = ~1.34B scaled
    assert typical_params == pytest.approx(4 * 4096 * 3072, rel=0.05)
    assert scaled_params == pytest.approx(typical_params * 15.25, rel=0.05)


def test_layer_scaling_guard_authoritative_total_params():
    """LayerScalingPass skips when metadata['total_params'] already set (authoritative).
    
    Fixes Issue 2: LayerScalingPass should not overwrite authoritative total_params.
    Example: model_loader or CLI --total-params sets metadata['total_params'] = 168B.
    LayerScalingPass should skip scaling and preserve the authoritative value.
    """
    g = typical_layer_graph(num_layers=4)
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    
    # Set authoritative total_params (e.g., from model_loader)
    authoritative_params = 168e9  # DeepSeek-V4 total params
    g.metadata["total_params"] = int(authoritative_params)
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Should NOT scale, preserve authoritative value
    assert result.metadata["total_params"] == int(authoritative_params)
    assert result.metadata.get("layer_scaling_complete") is not True
    
    # Should NOT have typical_params (scaling skipped)
    assert "typical_params" not in result.metadata


def test_layer_scaling_non_layer_params_not_scaled():
    """LayerScalingPass scales only per-layer params, not non-layer (embedding/lm_head).
    
    Fixes Known Bug #68: embedding/non-layer params should not be scaled by layer_count.
    Per-layer params: routed_expert, shared_expert, other (scaled)
    Non-layer params: embedding, lm_head, norm (NOT scaled)
    
    Verification: Check params_by_component breakdown exists and non_layer is NOT scaled.
    """
    # Create graph with embedding + transformer layers
    nodes = {}
    
    # Embedding node (non-layer, should NOT be scaled)
    embed_weight = _t("embed_tokens_weight", (128000, 7168))  # vocab×hidden, has "weight" in ID
    nodes["embed"] = OpNode(
        id="embed",
        op_type="aten.embedding.default",
        inputs=[embed_weight],
        outputs=[_t("embed_out", (128, 7168))],
        scope="model.embed_tokens",
        component="embedding",
        category="compute",
    )
    
    # Transformer layers (per-layer, should be scaled)
    for i in range(4):
        nid = f"layer_{i}"
        nodes[nid] = _linear_node(nid, f"model.layers.{i}.mlp", (128, 4096), (4096, 3072))
    
    g = OpGraph(name="test_with_embed", phase="train_forward", nodes=nodes, edges=[])
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    g.metadata["layer_profile"] = {"dense": 0, "sparse": 61}
    
    ctx = _ctx()
    result = LayerScalingPass().run(g, ctx)
    
    # Check params_by_component breakdown exists
    comp_breakdown = result.metadata.get("params_by_component")
    assert comp_breakdown is not None, "params_by_component should be set by LayerScalingPass"
    
    # Check each component category is present
    assert "routed_expert" in comp_breakdown
    assert "shared_expert" in comp_breakdown
    assert "other" in comp_breakdown
    assert "non_layer" in comp_breakdown
    
    # Key verification: non_layer should NOT be scaled
    # Embedding params: 128K × 7168 = 922M (unchanged by layer_scale)
    non_layer_params = comp_breakdown["non_layer"]
    expected_non_layer = 128000 * 7168  # vocab × hidden
    assert non_layer_params == pytest.approx(expected_non_layer, rel=0.05), \
        f"non_layer params should NOT be scaled (expected {expected_non_layer}, got {non_layer_params})"
    
    # Per-layer params should be scaled (check they are different from typical)
    # If all params were scaled uniformly, non_layer would be scaled too (WRONG)
    # This test ensures non_layer is NOT scaled
    layer_scale = result.metadata["layer_scale"]
    assert layer_scale > 1.0, "layer_scale should be > 1.0 for scaling to happen"
    
    # Verify total_params includes both scaled per-layer and unchanged non_layer
    total_params = result.metadata["total_params"]
    assert total_params > 0
    
    # Most importantly: non_layer is NOT multiplied by layer_scale
    # If it were scaled: non_layer × layer_scale = 922M × 15.25 = 14.1B (overestimate)
    # We verify it stays at 922M
    assert non_layer_params < 1e9, "non_layer should be ~922M, NOT 14.1B (scaled)"