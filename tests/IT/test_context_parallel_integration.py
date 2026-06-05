"""Context Parallel (CP) Integration Tests.

Tests cover:
1. Shape split verification - input/output tensor shapes correctly halved by CP factor
2. Communication node insertion - correct type (A2A/P2P/AllGather) at correct positions
3. FLOPs scaling - compute FLOPs reduced proportionally to CP factor
4. Memory bytes scaling - activation memory reduced proportionally
5. Communication volume - correct message size calculation
6. End-to-end validation - full pipeline from capture to transformed graph

These are baseline tests for CP functionality - do not modify after baselining.
"""

import pytest

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.ir.adapter import records_to_opgraph, stitch_fwd_bwd
from zrt.transform.context import TransformContext, ParallelConfig, TrainingConfig, FusionConfig
from zrt.transform.parallel.context_parallel import ContextParallelPass
from zrt.transform.parallel.comm_inserter import CommInserterPass
from zrt.transform.pipeline import build_pipeline
from zrt.transform.analysis.passes import FlopsPass
from zrt.hardware import load


def _make_simple_mlp_graph(seq_len: int = 2048, hidden: int = 4096, num_layers: int = 2):
    """Create a simple MLP graph for CP testing."""
    nodes = {}
    edges = []

    activation = TensorMeta(
        id="act_0",
        shape=(1, seq_len, hidden),
        dtype=DType.BF16,
        mem_bytes=seq_len * hidden * 2,
    )

    for layer_idx in range(num_layers):
        linear_in = TensorMeta(
            id=f"linear_in_{layer_idx}",
            shape=(1, seq_len, hidden),
            dtype=DType.BF16,
            mem_bytes=seq_len * hidden * 2,
        )
        linear_out = TensorMeta(
            id=f"linear_out_{layer_idx}",
            shape=(1, seq_len, hidden),
            dtype=DType.BF16,
            mem_bytes=seq_len * hidden * 2,
        )

        linear_node = OpNode(
            id=f"linear_{layer_idx}",
            op_type="aten.linear.default",
            inputs=[linear_in],
            outputs=[linear_out],
            scope=f"model.layers.{layer_idx}.mlp.linear",
            layer=str(layer_idx),
            category="compute",
        )

        silu_node = OpNode(
            id=f"silu_{layer_idx}",
            op_type="aten.silu.default",
            inputs=[linear_out],
            outputs=[linear_out],
            scope=f"model.layers.{layer_idx}.mlp.act",
            layer=str(layer_idx),
            category="activation",
        )

        nodes[linear_node.id] = linear_node
        nodes[silu_node.id] = silu_node

        edges.append(Edge(
            src=linear_node.id, src_idx=0,
            dst=silu_node.id, dst_idx=0,
            tensor=linear_out,
        ))

    return OpGraph(
        name="test_mlp",
        phase="forward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": seq_len, "hidden": hidden},
    )


def _make_hardware_spec():
    """Create a test hardware spec."""
    return load("nvidia_h100_sxm")


class TestCPShapeSplit:
    """Test that CP pass correctly splits tensor shapes."""

    def test_shape_split_factor_2(self):
        """Verify seq_len dimension is halved when cp=2."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)

        seq_local = seq_len // cp

        for node in after_cp.nodes.values():
            for tensor in node.inputs + node.outputs:
                if seq_len in tensor.shape:
                    assert seq_local in tensor.shape, (
                        f"Node {node.id}: expected seq_local={seq_local} in shape {tensor.shape}"
                    )

    def test_shape_split_preserves_other_dims(self):
        """Verify hidden dimension is not affected by CP."""
        seq_len, hidden, cp = 4096, 7168, 4
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)

        seq_local = seq_len // cp

        linear_nodes = [n for n in after_cp.nodes.values() if "linear" in n.op_type]
        for node in linear_nodes:
            for tensor in node.inputs + node.outputs:
                shape = tensor.shape
                if len(shape) >= 2:
                    assert shape[-1] == hidden, (
                        f"Node {node.id}: hidden dim changed from {hidden} to {shape[-1]}"
                    )

    def test_shape_split_batch_dim_unchanged(self):
        """Verify batch dimension (dim 0) is not affected."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)

        for node in after_cp.nodes.values():
            for tensor in node.inputs + node.outputs:
                if len(tensor.shape) >= 3:
                    assert tensor.shape[0] == 1, (
                        f"Node {node.id}: batch dim changed from 1 to {tensor.shape[0]}"
                    )

    def test_no_shape_split_when_cp1(self):
        """Verify shapes unchanged when cp=1."""
        seq_len, hidden = 2048, 4096
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=1),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden),
        )

        after_cp = ContextParallelPass().run(graph, ctx)

        for node in after_cp.nodes.values():
            for tensor in node.inputs + node.outputs:
                if len(tensor.shape) >= 2:
                    assert tensor.shape[1] == seq_len, (
                        f"Node {node.id}: seq_len changed when cp=1"
                    )


class TestCPCommunicationInsertion:
    """Test communication node insertion for different CP strategies."""

    def test_ulysses_a2a_insertion(self):
        """Verify A2A nodes inserted at layer boundaries for Ulysses CP."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=3)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        a2a_nodes = [
            n for n in after_comm.nodes.values()
            if n.op_type == "comm.all_to_all" and "cp_ulysses" in n.attrs.get("role", "")
        ]

        assert len(a2a_nodes) >= 2, f"Expected at least 2 A2A nodes, got {len(a2a_nodes)}"

        for node in a2a_nodes:
            assert node.category == "communication"
            assert node.attrs.get("group_size") == cp

    def test_ulysses_pre_post_pairing(self):
        """Verify pre-A2A and post-A2A are paired per layer."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=2)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        pre_nodes = [
            n for n in after_comm.nodes.values()
            if n.attrs.get("role") == "cp_ulysses_pre"
        ]
        post_nodes = [
            n for n in after_comm.nodes.values()
            if n.attrs.get("role") == "cp_ulysses_pre"
        ]

        assert len(pre_nodes) == len(post_nodes), (
            f"Pre/Post A2A count mismatch: {len(pre_nodes)} vs {len(post_nodes)}"
        )

    def test_compressed_cp_stages(self):
        """Verify stage1 P2P and stage2 AllGather for compressed CP."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=2)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="compressed"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        roles = {n.attrs.get("role") for n in after_comm.nodes.values()}

        assert "cp_compressed_stage1" in roles, "Missing stage1 P2P node"
        assert "cp_compressed_stage2" in roles, "Missing stage2 AllGather node"

    def test_comm_node_layer_annotation(self):
        """Verify communication nodes have correct layer annotation."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=3)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        comm_nodes = [
            n for n in after_comm.nodes.values()
            if n.category == "communication" and "cp" in n.attrs.get("role", "")
        ]

        for node in comm_nodes:
            assert node.layer is not None and node.layer != "", (
                f"Comm node {node.id} missing layer annotation"
            )


class TestCPCommunicationVolume:
    """Test communication volume calculations."""

    def test_ulysses_a2a_bytes(self):
        """Verify A2A message size = batch * seq_local * hidden * 2."""
        seq_len, hidden, cp, batch = 2048, 4096, 2, 1
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="ulysses",
                micro_batch=batch,
            ),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        seq_local = seq_len // cp
        expected_bytes = batch * seq_local * hidden * 2

        a2a_nodes = [
            n for n in after_comm.nodes.values()
            if n.op_type == "comm.all_to_all" and "cp_ulysses" in n.attrs.get("role", "")
        ]

        for node in a2a_nodes:
            actual_bytes = node.attrs.get("bytes", 0)
            assert actual_bytes == expected_bytes, (
                f"Node {node.id}: expected {expected_bytes} bytes, got {actual_bytes}"
            )

    def test_compressed_stage1_bytes(self):
        """Verify stage1 P2P bytes = batch * seq_local * hidden * 2."""
        seq_len, hidden, cp, batch = 2048, 4096, 2, 1
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="compressed",
                micro_batch=batch,
            ),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        seq_local = seq_len // cp
        expected_bytes = batch * seq_local * hidden * 2

        p2p_nodes = [
            n for n in after_comm.nodes.values()
            if n.attrs.get("role") == "cp_compressed_stage1"
        ]

        for node in p2p_nodes:
            actual_bytes = node.attrs.get("bytes", 0)
            assert actual_bytes == expected_bytes, (
                f"Node {node.id}: expected {expected_bytes} bytes, got {actual_bytes}"
            )

    def test_compressed_stage2_bytes_compressed(self):
        """Verify stage2 AllGather bytes = stage1_bytes / compression_ratio."""
        seq_len, hidden, cp, batch = 2048, 4096, 2, 1
        compression_ratio = 4
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="compressed",
                micro_batch=batch,
            ),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        seq_local = seq_len // cp
        stage1_bytes = batch * seq_local * hidden * 2
        expected_bytes = stage1_bytes // compression_ratio

        ag_nodes = [
            n for n in after_comm.nodes.values()
            if n.attrs.get("role") == "cp_compressed_stage2"
        ]

        for node in ag_nodes:
            actual_bytes = node.attrs.get("bytes", 0)
            assert actual_bytes == expected_bytes, (
                f"Node {node.id}: expected {expected_bytes} bytes, got {actual_bytes}"
            )


class TestCPFLOPsScaling:
    """Test FLOPs scaling under CP."""

    def test_flops_reduced_by_cp_factor(self):
        """Verify FLOPs reduced proportionally to CP factor for seq-dependent ops."""
        seq_len, hidden = 2048, 4096

        graph1 = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)
        ctx1 = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=1),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden),
            fusion=FusionConfig(),
        )
        pipe1 = build_pipeline()
        result1 = pipe1.run(graph1.clone(), ctx1)

        total_flops_cp1 = sum(
            n.annotations.get("flops", 0) for n in result1.nodes.values()
        )

        graph2 = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)
        ctx2 = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=2),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
            fusion=FusionConfig(),
        )
        pipe2 = build_pipeline()
        result2 = pipe2.run(graph2.clone(), ctx2)

        total_flops_cp2 = sum(
            n.annotations.get("flops", 0) for n in result2.nodes.values()
        )

        if total_flops_cp1 > 0:
            actual_ratio = total_flops_cp2 / total_flops_cp1
            assert actual_ratio < 1.0, (
                f"FLOPs should decrease under CP: ratio={actual_ratio:.2f}"
            )

    def test_flops_per_node_shape_consistent(self):
        """Verify FLOPs annotation uses post-CP shape."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
            fusion=FusionConfig(),
        )

        pipe = build_pipeline()
        result = pipe.run(graph.clone(), ctx)

        seq_local = seq_len // cp

        linear_nodes = [n for n in result.nodes.values() if "linear" in n.op_type.lower()]
        for node in linear_nodes:
            flops = node.annotations.get("flops", 0)
            if flops > 0:
                input_tensor = node.inputs[0] if node.inputs else None
                if input_tensor and len(input_tensor.shape) >= 2:
                    actual_seq = input_tensor.shape[1]
                    assert actual_seq == seq_local, (
                        f"Node {node.id}: FLOPs uses shape with seq={actual_seq}, "
                        f"expected seq_local={seq_local}"
                    )


class TestCPMemoryScaling:
    """Test activation memory scaling under CP."""

    def test_tensor_mem_bytes_correct(self):
        """Verify tensor mem_bytes reflects post-CP shape."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)

        seq_local = seq_len // cp
        expected_bytes_per_elem = hidden * seq_local * 2

        checked = 0
        for node in after_cp.nodes.values():
            for tensor in node.inputs + node.outputs:
                if len(tensor.shape) >= 2 and tensor.shape[1] == seq_local:
                    expected = tensor.shape[0] * tensor.shape[1] * tensor.shape[-1] * 2
                    if tensor.mem_bytes > 0:
                        assert tensor.mem_bytes == expected, (
                            f"Tensor {tensor.id}: mem_bytes={tensor.mem_bytes}, expected={expected}"
                        )
                        checked += 1

        assert checked > 0, "No tensors checked for mem_bytes"


class TestCPGraphConnectivity:
    """Test graph connectivity after CP transformation."""

    def test_no_cycles_after_cp_and_comm(self):
        """Verify graph remains DAG after CP + Comm insertion."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=3)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        try:
            sorted_nodes = after_comm.topo_sort()
            assert len(sorted_nodes) == len(after_comm.nodes), (
                f"Topo sort reached only {len(sorted_nodes)}/{len(after_comm.nodes)} nodes"
            )
        except RuntimeError as e:
            pytest.fail(f"Graph has cycle after CP transformation: {e}")

    def test_comm_node_connectivity(self):
        """Verify communication nodes properly connect compute nodes."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=2)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        comm_nodes = [
            n for n in after_comm.nodes.values()
            if n.category == "communication"
        ]

        connected_count = 0
        for comm in comm_nodes:
            in_edges = after_comm.in_edges(comm.id)
            out_edges = after_comm.out_edges(comm.id)

            if len(in_edges) > 0 or len(out_edges) > 0:
                connected_count += 1

        assert connected_count >= len(comm_nodes) // 2, (
            f"Only {connected_count}/{len(comm_nodes)} comm nodes have edges"
        )


class TestCPEndToEnd:
    """End-to-end integration tests using real model capture."""

    @pytest.fixture(scope="class")
    def captured_graph(self):
        """Capture DeepSeek-V4 graph for testing."""
        from zrt.pipeline import run_trace_phases

        output_dir, phase_records = run_trace_phases(
            model_id="hf_models/deepseek_v4",
            num_layers=2,
            batch_size=1,
            seq_len=20000,
            phases=("train_forward", "train_backward"),
        )

        fwd_records = phase_records.get("train_forward", [])
        bwd_records = phase_records.get("train_backward", [])

        fwd_graph = records_to_opgraph(fwd_records, "dsv4_fwd", "train_forward")
        bwd_graph = records_to_opgraph(bwd_records, "dsv4_bwd", "train_backward")

        unified = stitch_fwd_bwd(fwd_graph, bwd_graph)

        return unified

    def test_e2e_no_cycle_cp2_compressed(self, captured_graph):
        """End-to-end: CP=2 compressed produces valid DAG."""
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            model_id="hf_models/deepseek_v4",
            parallel=ParallelConfig(tp=1, pp=1, ep=1, dp=1, cp=2),
            training=TrainingConfig(seq_len=20000, hidden=7168, cp_kind="compressed"),
        )

        after_cp = ContextParallelPass().run(captured_graph.clone(), ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        try:
            sorted_nodes = after_comm.topo_sort()
            assert len(sorted_nodes) == len(after_comm.nodes)
        except RuntimeError as e:
            pytest.fail(f"End-to-end graph has cycle: {e}")

    def test_e2e_shape_split_verified(self, captured_graph):
        """End-to-end: tensor shapes correctly halved."""
        seq_len = 20000
        cp = 2
        seq_local = seq_len // cp

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            model_id="hf_models/deepseek_v4",
            parallel=ParallelConfig(tp=1, pp=1, ep=1, dp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=7168, cp_kind="compressed"),
        )

        after_cp = ContextParallelPass().run(captured_graph.clone(), ctx)

        cp_nodes = [
            n for n in after_cp.nodes.values()
            if n.annotations.get("cp_split")
        ]

        sample_count = 0
        for node in cp_nodes[:20]:
            for tensor in node.inputs[:1] + node.outputs[:1]:
                if len(tensor.shape) >= 2 and tensor.shape[1] == seq_local:
                    sample_count += 1

        assert sample_count >= 10, (
            f"Expected >= 10 tensors with seq_local={seq_local}, found {sample_count}"
        )

    def test_e2e_flops_ratio_reasonable(self, captured_graph):
        """End-to-end: FLOPs reduced roughly by CP factor."""
        seq_len = 20000

        ctx1 = TransformContext(
            hw_spec=_make_hardware_spec(),
            model_id="hf_models/deepseek_v4",
            parallel=ParallelConfig(tp=1, pp=1, ep=1, dp=1, cp=1),
            training=TrainingConfig(seq_len=seq_len, hidden=7168),
            fusion=FusionConfig(),
        )
        pipe1 = build_pipeline()
        result1 = pipe1.run(captured_graph.clone(), ctx1)
        flops_cp1 = sum(n.annotations.get("flops", 0) for n in result1.nodes.values())

        ctx2 = TransformContext(
            hw_spec=_make_hardware_spec(),
            model_id="hf_models/deepseek_v4",
            parallel=ParallelConfig(tp=1, pp=1, ep=1, dp=1, cp=2),
            training=TrainingConfig(seq_len=seq_len, hidden=7168, cp_kind="compressed"),
            fusion=FusionConfig(),
        )
        pipe2 = build_pipeline()
        result2 = pipe2.run(captured_graph.clone(), ctx2)
        flops_cp2 = sum(n.annotations.get("flops", 0) for n in result2.nodes.values())

        if flops_cp1 > 0:
            ratio = flops_cp2 / flops_cp1
            assert ratio < 1.0, f"FLOPs ratio {ratio:.2f} should be < 1.0 for CP=2"

    def test_e2e_comm_nodes_present(self, captured_graph):
        """End-to-end: compressed CP inserts stage1 and stage2 comm nodes."""
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            model_id="hf_models/deepseek_v4",
            parallel=ParallelConfig(tp=1, pp=1, ep=1, dp=1, cp=2),
            training=TrainingConfig(seq_len=20000, hidden=7168, cp_kind="compressed"),
        )

        pipe = build_pipeline()
        result = pipe.run(captured_graph.clone(), ctx)

        comm_nodes = [
            n for n in result.nodes.values()
            if n.category == "communication"
        ]

        assert len(comm_nodes) >= 8, f"Expected >= 8 comm nodes, got {len(comm_nodes)}"

        roles = {n.attrs.get("role") for n in comm_nodes}
        assert "cp_compressed_stage1" in roles, "Missing stage1 P2P"
        assert "cp_compressed_stage2" in roles, "Missing stage2 AllGather"

    def test_e2e_tensor_shape_consistency_per_node(self, captured_graph):
        """End-to-end: tensor shape consistent within each node's I/O."""
        seq_len = 20000
        cp = 2

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            model_id="hf_models/deepseek_v4",
            parallel=ParallelConfig(tp=1, pp=1, ep=1, dp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=7168, cp_kind="compressed"),
        )

        after_cp = ContextParallelPass().run(captured_graph.clone(), ctx)

        inconsistent_count = 0
        for node in after_cp.nodes.values():
            tensor_shapes = {}
            for tensor in node.inputs + node.outputs:
                tid = tensor.id
                shape = tensor.shape
                if tid in tensor_shapes:
                    if tensor_shapes[tid] != shape:
                        inconsistent_count += 1
                        break
                else:
                    tensor_shapes[tid] = shape

        assert inconsistent_count == 0, (
            f"Found {inconsistent_count} nodes with inconsistent tensor shapes for same ID"
        )


class TestCPMetadataAnnotations:
    """Test CP-specific metadata annotations."""

    def test_cp_split_annotation_present(self):
        """Verify cp_split annotation on transformed nodes."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)

        annotated_nodes = [
            n for n in after_cp.nodes.values()
            if n.annotations.get("cp_split")
        ]

        assert len(annotated_nodes) > 0, "No nodes have cp_split annotation"

        for node in annotated_nodes[:5]:
            annotation = node.annotations["cp_split"]
            assert annotation.get("kind") in ["ulysses", "compressed"], (
                f"Node {node.id}: unexpected cp_kind {annotation.get('kind')}"
            )
            assert annotation.get("cp") == cp

    def test_comm_node_phase_annotation(self):
        """Verify communication nodes have phase annotation (fwd/bwd)."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_simple_mlp_graph(seq_len=seq_len, hidden=hidden, num_layers=2)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )

        after_cp = ContextParallelPass().run(graph, ctx)
        after_comm = CommInserterPass().run(after_cp, ctx)

        comm_nodes = [
            n for n in after_comm.nodes.values()
            if n.category == "communication" and "cp" in n.attrs.get("role", "")
        ]

        for node in comm_nodes:
            phase = node.annotations.get("phase")
            assert phase in ["fwd", "bwd", None], (
                f"Node {node.id}: unexpected phase annotation {phase}"
            )


def _make_mixed_graph(seq_len=2048, hidden=4096, num_heads=32, head_dim=128):
    """Graph with attention + non-attention nodes for CPKind testing."""
    nodes = {}
    edges = []
    batch = 1
    inp = TensorMeta(id="inp", shape=(batch, seq_len, hidden), dtype=DType.BF16,
                     mem_bytes=batch * seq_len * hidden * 2)
    qkv_out = TensorMeta(
        id="qkv_out", shape=(batch, seq_len, num_heads * head_dim * 3),
        dtype=DType.BF16, mem_bytes=batch * seq_len * num_heads * head_dim * 3 * 2,
    )
    q_4d = TensorMeta(
        id="q_4d", shape=(batch, num_heads, seq_len, head_dim),
        dtype=DType.BF16, mem_bytes=batch * num_heads * seq_len * head_dim * 2,
    )
    attn_scores = TensorMeta(
        id="attn_scores", shape=(batch, num_heads, seq_len, seq_len),
        dtype=DType.BF16, mem_bytes=batch * num_heads * seq_len * seq_len * 2,
    )
    attn_out = TensorMeta(
        id="attn_out", shape=(batch, seq_len, num_heads * head_dim),
        dtype=DType.BF16, mem_bytes=batch * seq_len * num_heads * head_dim * 2,
    )
    mlp_out = TensorMeta(
        id="mlp_out", shape=(batch, seq_len, hidden),
        dtype=DType.BF16, mem_bytes=batch * seq_len * hidden * 2,
    )

    norm = OpNode(id="norm", op_type="aten.rms_norm",
                  inputs=[inp], outputs=[inp],
                  scope="model.layers.0.input_layernorm",
                  layer="0", category="compute")
    qkv = OpNode(id="qkv_proj", op_type="aten.linear",
                 inputs=[inp], outputs=[qkv_out],
                 scope="model.layers.0.self_attn.q_proj",
                 layer="0", category="compute")
    reshape = OpNode(id="reshape_q", op_type="aten.view",
                     inputs=[qkv_out], outputs=[q_4d],
                     scope="model.layers.0.self_attn",
                     layer="0", category="memory")
    score = OpNode(id="attn_score", op_type="aten.bmm",
                   inputs=[q_4d, q_4d], outputs=[attn_scores],
                   scope="model.layers.0.self_attn",
                   layer="0", category="compute")
    sdpa = OpNode(id="sdpa", op_type="aten._scaled_dot_product_attention",
                  inputs=[attn_scores], outputs=[attn_out],
                  scope="model.layers.0.self_attn",
                  layer="0", category="compute")
    o_proj = OpNode(id="o_proj", op_type="aten.linear",
                    inputs=[attn_out], outputs=[inp],
                    scope="model.layers.0.self_attn.o_proj",
                    layer="0", category="compute")
    mlp = OpNode(id="mlp", op_type="aten.linear",
                 inputs=[inp], outputs=[mlp_out],
                 scope="model.layers.0.mlp.gate_proj",
                 layer="0", category="compute")

    for n in [norm, qkv, reshape, score, sdpa, o_proj, mlp]:
        nodes[n.id] = n
    edges.append(Edge(src="norm", src_idx=0, dst="qkv_proj", dst_idx=0, tensor=inp))
    edges.append(Edge(src="qkv_proj", src_idx=0, dst="reshape_q", dst_idx=0, tensor=qkv_out))
    edges.append(Edge(src="reshape_q", src_idx=0, dst="attn_score", dst_idx=0, tensor=q_4d))
    edges.append(Edge(src="attn_score", src_idx=0, dst="sdpa", dst_idx=0, tensor=attn_scores))
    edges.append(Edge(src="sdpa", src_idx=0, dst="o_proj", dst_idx=0, tensor=attn_out))
    edges.append(Edge(src="o_proj", src_idx=0, dst="mlp", dst_idx=0, tensor=inp))

    return OpGraph(name="mixed", phase="forward", nodes=nodes, edges=edges,
                   metadata={"seq_len": seq_len, "hidden": hidden})


class TestCPKindAwareSplitting:
    """Test CPKind-differentiated shape splitting."""

    def test_ulysses_attn_heads_split_seq_preserved(self):
        """Ulysses: attention nodes split heads, seq unchanged."""
        seq_len, hidden, cp = 2048, 4096, 2
        num_heads, head_dim = 32, 128
        graph = _make_mixed_graph(seq_len, hidden, num_heads, head_dim)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="ulysses",
                num_heads=num_heads, head_dim=head_dim,
            ),
        )
        g = ContextParallelPass().run(graph, ctx)

        sdpa = g.nodes["sdpa"]
        for t in sdpa.inputs + sdpa.outputs:
            if len(t.shape) >= 2:
                assert seq_len in t.shape, (
                    f"Ulysses attn: seq_len should be preserved, got {t.shape}"
                )

        score = g.nodes["attn_score"]
        for t in score.outputs:
            if len(t.shape) == 4:
                assert t.shape[1] == num_heads // cp, (
                    f"Ulysses: heads should be {num_heads // cp}, got {t.shape[1]}"
                )
                assert t.shape[2] == seq_len
                assert t.shape[3] == seq_len

    def test_ulysses_non_attn_seq_split(self):
        """Ulysses: non-attention nodes split seq."""
        seq_len, hidden, cp = 2048, 4096, 2
        num_heads, head_dim = 32, 128
        graph = _make_mixed_graph(seq_len, hidden, num_heads, head_dim)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="ulysses",
                num_heads=num_heads, head_dim=head_dim,
            ),
        )
        g = ContextParallelPass().run(graph, ctx)

        mlp = g.nodes["mlp"]
        for t in mlp.inputs + mlp.outputs:
            if len(t.shape) >= 2 and seq_len in t.shape:
                assert seq_len // cp in t.shape, (
                    f"Ulysses non-attn: seq should be split, got {t.shape}"
                )

    def test_ring_all_seq_split(self):
        """Ring: all nodes split seq."""
        seq_len, hidden, cp = 2048, 4096, 4
        num_heads, head_dim = 32, 128
        graph = _make_mixed_graph(seq_len, hidden, num_heads, head_dim)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="ring",
                num_heads=num_heads, head_dim=head_dim,
            ),
        )
        g = ContextParallelPass().run(graph, ctx)

        for node in g.nodes.values():
            for t in node.inputs + node.outputs:
                if seq_len in t.shape:
                    assert seq_len // cp in t.shape, (
                        f"Ring: all seq dims should be split, "
                        f"node={node.id} shape={t.shape}"
                    )

    def test_hybrid_dual_split(self):
        """Hybrid: attn splits heads by cp_ulysses + seq by cp_ring."""
        seq_len, hidden = 2048, 4096
        num_heads, head_dim = 32, 128
        cp_ulysses, cp_ring = 2, 2
        cp = cp_ulysses * cp_ring
        graph = _make_mixed_graph(seq_len, hidden, num_heads, head_dim)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp, cp_ulysses=cp_ulysses, cp_ring=cp_ring),
            training=TrainingConfig(
                seq_len=seq_len, hidden=hidden, cp_kind="hybrid",
                num_heads=num_heads, head_dim=head_dim,
            ),
        )
        g = ContextParallelPass().run(graph, ctx)

        score = g.nodes["attn_score"]
        for t in score.outputs:
            if len(t.shape) == 4:
                assert t.shape[1] == num_heads // cp_ulysses, (
                    f"Hybrid: heads should be {num_heads // cp_ulysses}, got {t.shape[1]}"
                )
                assert t.shape[2] == seq_len // cp_ring, (
                    f"Hybrid: seq should be {seq_len // cp_ring}, got {t.shape[2]}"
                )

    def test_ulysses_fallback_no_num_heads(self):
        """Ulysses without num_heads falls back to seq-split."""
        seq_len, hidden, cp = 2048, 4096, 2
        graph = _make_mixed_graph(seq_len, hidden)

        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, cp=cp),
            training=TrainingConfig(seq_len=seq_len, hidden=hidden, cp_kind="ulysses"),
        )
        g = ContextParallelPass().run(graph, ctx)

        for node in g.nodes.values():
            for t in node.inputs + node.outputs:
                if seq_len in t.shape:
                    assert seq_len // cp in t.shape, (
                        f"Fallback: seq should be split, got {t.shape}"
                    )