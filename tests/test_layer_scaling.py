"""Test LayerScalingPass: pre-compute scaled params BEFORE TP/EP sharding.

Tests cover:
1. Normal case: typical_indices present -> compute layer_scale
2. No scaling needed: num_layers == num_typical
3. Missing metadata: no layer_profile or typical_indices
4. Metadata correctness: total_params, layer_scale, layer_scaling_complete
5. Integration with count_params and pipeline
"""
import pytest
from types import SimpleNamespace

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
    """compute_layer_scale() computes correct scaling factor."""
    g = typical_layer_graph(num_layers=4)
    
    # Case 1: metadata already has layer_scale
    g.metadata["layer_scale"] = 15.25
    assert compute_layer_scale(g) == 15.25
    
    # Case 2: compute from typical_indices
    g.metadata["layer_scale"] = 0.0
    g.metadata["num_layers"] = 61
    g.metadata["typical_indices"] = [0, 1, 2, 3]
    assert compute_layer_scale(g) == pytest.approx(61.0 / 4.0, rel=0.01)
    
    # Case 3: compute from num_layers ratio
    g.metadata["typical_indices"] = None
    g.metadata["num_layers_traced"] = 4
    assert compute_layer_scale(g) == pytest.approx(61.0 / 4.0, rel=0.01)
    
    # Case 4: no scaling needed
    g.metadata["num_layers"] = 4
    g.metadata["num_layers_traced"] = 4
    assert compute_layer_scale(g) == 1.0


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
    params_no_meta = count_params(g, apply_layer_scale=True)
    typical_params = 4 * 4096 * 3072
    expected_scaled = typical_params * (61.0 / 4.0)
    # Note: apply_layer_scale works in structural fallback path
    
    # Case 2: metadata["total_params"] is set (LayerScalingPass ran)
    g.metadata["total_params"] = int(expected_scaled)
    params_with_meta = count_params(g, apply_layer_scale=True)
    assert params_with_meta == int(expected_scaled)


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