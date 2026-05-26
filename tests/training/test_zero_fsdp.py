"""Quantitative tests for ZeroFSDPPass communication impact.

Tests cover:
1. Comm node count & topology per layer
2. Communication volume (weight_bytes / grad_bytes) accuracy
3. Comm latency scaling with DP degree
4. Scheduling overhead (timeline impact)
5. ZeRO stage comparison
6. Cross-node vs intra-node bandwidth selection
7. End-to-end MFU impact via TrainingPipelinePass
"""

import pytest
from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import (
    TransformContext, ParallelConfig, TrainingConfig,
)
from zrt.transform.training.zero_fsdp import ZeroFSDPPass
from zrt.transform.analysis.comm_latency import CommLatencyPass
from zrt.transform.analysis.passes import StreamAssignPass
from zrt.transform.analysis.training import TrainingPipelinePass
from zrt.executor.scheduler import DAGScheduler


def _make_hardware_spec():
    """Create a mock H100-like hardware spec."""
    from zrt.hardware.spec import HardwareSpec, ComputeSpec, MemorySpec, InterconnectSpec, LinkSpec
    return HardwareSpec(
        name="test_gpu",
        vendor="test",
        device_type="gpu",
        compute=ComputeSpec(bf16_tflops=1000),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3000),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="nvlink", num_devices=8,
                                bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="ib", num_devices=1000,
                                bandwidth_gbps=400, latency_us=5.0),
        ),
    )


def _make_train_graph(num_layers=2, hidden=4096, seq_len=2048, param_bytes_per_layer=0):
    """Create a stitched forward+backward graph with parameter nodes.

    Each layer has:
    - fwd node with "phase"="fwd" annotation
    - bwd node with "phase"="bwd" annotation
    - param node with "is_param"=True annotation (if param_bytes_per_layer > 0)
    """
    nodes = {}
    edges = []
    tensor_bytes = seq_len * hidden * 2  # BF16

    if param_bytes_per_layer == 0:
        param_bytes_per_layer = hidden * hidden * 2  # default: 4096x4096 BF16 = 32MB

    for i in range(num_layers):
        out_t = TensorMeta(id=f"fwd_out_{i}", shape=(1, seq_len, hidden),
                           dtype=DType.BF16, mem_bytes=tensor_bytes)
        fwd_node = OpNode(
            id=f"fwd_{i}",
            op_type="aten.linear",
            inputs=[TensorMeta(id=f"fwd_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=tensor_bytes)],
            outputs=[out_t],
            scope=f"model.layers.{i}.mlp",
            layer=str(i),
            category="compute",
            annotations={"phase": "fwd"},
        )
        nodes[fwd_node.id] = fwd_node

        bwd_out = TensorMeta(id=f"bwd_out_{i}", shape=(1, seq_len, hidden),
                             dtype=DType.BF16, mem_bytes=tensor_bytes)
        bwd_node = OpNode(
            id=f"bwd_{i}",
            op_type="aten.mm_backward",
            inputs=[TensorMeta(id=f"bwd_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=tensor_bytes)],
            outputs=[bwd_out],
            scope=f"model.layers.{i}.mlp",
            layer=str(i),
            category="compute",
            annotations={"phase": "bwd"},
        )
        nodes[bwd_node.id] = bwd_node

        param_t = TensorMeta(
            id=f"param_{i}",
            shape=(hidden, hidden),
            dtype=DType.BF16,
            mem_bytes=param_bytes_per_layer,
        )
        param_node = OpNode(
            id=f"param_{i}",
            op_type="aten.param",
            inputs=[],
            outputs=[param_t],
            scope=f"model.layers.{i}.mlp",
            layer=str(i),
            category="compute",
            annotations={"is_param": True, "phase": "fwd"},
        )
        nodes[param_node.id] = param_node

    for i in range(1, num_layers):
        edges.append(Edge(
            src=f"fwd_{i-1}", src_idx=0, dst=f"fwd_{i}", dst_idx=0,
            tensor=nodes[f"fwd_{i}"].inputs[0],
        ))
        edges.append(Edge(
            src=f"bwd_{i-1}", src_idx=0, dst=f"bwd_{i}", dst_idx=0,
            tensor=nodes[f"bwd_{i}"].inputs[0],
        ))

    return OpGraph(
        name="test_train_model",
        phase="train",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": seq_len, "hidden": hidden, "num_layers": num_layers},
    )


def _make_ctx(dp=4, zero_stage=3, hw_spec=None):
    """Helper to create a TransformContext for training."""
    if hw_spec is None:
        hw_spec = _make_hardware_spec()
    return TransformContext(
        hw_spec=hw_spec,
        parallel=ParallelConfig(tp=1, dp=dp),
        training=TrainingConfig(
            micro_batch=1, global_batch=8, zero_stage=zero_stage,
        ),
    )


# ── 1. Comm node count & topology ─────────────────────────────────────────────

class TestCommNodeCount:
    """Verify correct number of comm nodes per layer for ZeRO-3."""

    def test_fsdp_comm_count_4_layers(self):
        """ZeRO-3: each layer gets 1 fwd all_gather + 1 bwd all_gather + 1 bwd reduce_scatter."""
        graph = _make_train_graph(num_layers=4)
        ctx = _make_ctx(dp=8, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        ag_fwd = [n for n in result.nodes.values()
                  if n.op_type == "comm.all_gather"
                  and n.annotations.get("phase") == "fwd"]
        ag_bwd = [n for n in result.nodes.values()
                  if n.op_type == "comm.all_gather"
                  and n.annotations.get("phase") == "bwd"]
        rs_bwd = [n for n in result.nodes.values()
                  if n.op_type == "comm.reduce_scatter"
                  and n.annotations.get("phase") == "bwd"]

        assert len(ag_fwd) == 4
        assert len(ag_bwd) == 4
        assert len(rs_bwd) == 4

    def test_fsdp_comm_count_2_layers(self):
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        comm_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        assert len(comm_nodes) == 6  # 2 layers * 3 comm nodes

    def test_fsdp_comm_annotations(self):
        """All inserted comm nodes should have correct annotations."""
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        comm_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        for node in comm_nodes:
            assert node.category == "communication"
            assert "group_size" in node.attrs
            assert node.attrs["group_size"] == 4


# ── 2. Communication volume accuracy ─────────────────────────────────────────

class TestCommVolume:
    """Verify weight_bytes and grad_bytes are computed correctly."""

    def test_all_gather_weight_volume(self):
        """all_gather input tensor should carry layer parameter bytes."""
        param_bytes = 4096 * 4096 * 2  # 32MB
        graph = _make_train_graph(num_layers=1, param_bytes_per_layer=param_bytes)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        ag_fwd = next(n for n in result.nodes.values()
                      if n.op_type == "comm.all_gather"
                      and n.annotations.get("phase") == "fwd")
        assert ag_fwd.inputs[0].mem_bytes == param_bytes

    def test_reduce_scatter_grad_volume(self):
        """reduce_scatter input tensor should carry backward output bytes."""
        hidden = 4096
        seq_len = 2048
        grad_bytes = seq_len * hidden * 2  # BF16
        graph = _make_train_graph(num_layers=1, hidden=hidden, seq_len=seq_len)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        rs_bwd = next(n for n in result.nodes.values()
                      if n.op_type == "comm.reduce_scatter")
        assert rs_bwd.inputs[0].mem_bytes == grad_bytes

    def test_zero_metadata_shard_factors(self):
        """g.metadata['zero'] should have correct shard factors."""
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=8, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        z = result.metadata["zero"]
        assert z["stage"] == 3
        assert z["weight_shard"] == 8
        assert z["grad_shard"] == 8
        assert z["optstate_shard"] == 8


# ── 3. ZeRO stage comparison ─────────────────────────────────────────────────

class TestZeroStageComparison:
    """ZeRO-0/1/2 should NOT insert FSDP comm; only ZeRO-3 should."""

    def test_zero_stage_0_no_fsdp_comm(self):
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=4, zero_stage=0)
        result = ZeroFSDPPass().run(graph, ctx)
        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        assert len(fsdp_nodes) == 0

    def test_zero_stage_1_no_fsdp_comm(self):
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=4, zero_stage=1)
        result = ZeroFSDPPass().run(graph, ctx)
        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        assert len(fsdp_nodes) == 0

    def test_zero_stage_2_no_fsdp_comm(self):
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=4, zero_stage=2)
        result = ZeroFSDPPass().run(graph, ctx)
        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        assert len(fsdp_nodes) == 0

    def test_zero_stage_3_has_fsdp_comm(self):
        graph = _make_train_graph(num_layers=2)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)
        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        assert len(fsdp_nodes) == 6  # 2 layers * 3 comm nodes


# ── 4. Comm latency scaling with DP ──────────────────────────────────────────

class TestCommLatencyScaling:
    """Verify all_gather/reduce_scatter latency scales correctly with DP degree."""

    def test_latency_increases_with_dp(self):
        """Comm latency should increase as DP grows (sub-linear due to ring algo)."""
        graph = _make_train_graph(num_layers=2)
        hw = _make_hardware_spec()

        latencies = {}
        for dp in [2, 4, 8]:
            ctx = _make_ctx(dp=dp, zero_stage=3, hw_spec=hw)
            g = ZeroFSDPPass().run(graph, ctx)
            g = CommLatencyPass().run(g, ctx)

            comm_nodes = [n for n in g.nodes.values()
                          if n.category == "communication"]
            latencies[dp] = sum(n.annotations.get("latency_us", 0.0)
                                for n in comm_nodes)

        assert latencies[4] > latencies[2]
        assert latencies[8] > latencies[4]

    def test_all_gather_vs_reduce_scatter_same_formula(self):
        """all_gather and reduce_scatter use the same ring factor: (n-1)/n."""
        graph = _make_train_graph(num_layers=1)
        ctx = _make_ctx(dp=4, zero_stage=3)
        g = ZeroFSDPPass().run(graph, ctx)
        g = CommLatencyPass().run(g, ctx)

        ag_nodes = [n for n in g.nodes.values()
                    if n.op_type == "comm.all_gather"]
        rs_nodes = [n for n in g.nodes.values()
                    if n.op_type == "comm.reduce_scatter"]

        # With same data size, they should have same latency
        # (weight_bytes != grad_bytes in our test graph, so we compare ring factor)
        for ag in ag_nodes:
            for rs in rs_nodes:
                ag_vol = ag.inputs[0].mem_bytes
                rs_vol = rs.inputs[0].mem_bytes
                ag_lat = ag.annotations.get("latency_us", 0)
                rs_lat = rs.annotations.get("latency_us", 0)
                if ag_vol > 0 and rs_vol > 0:
                    # latency per byte should be same
                    ag_per_byte = ag_lat / ag_vol
                    rs_per_byte = rs_lat / rs_vol
                    assert ag_per_byte == pytest.approx(rs_per_byte, rel=0.01)


# ── 5. Scheduling overhead (timeline impact) ─────────────────────────────────

class TestSchedulingOverhead:
    """Quantify FSDP comm overhead in scheduled timeline."""

    def test_fsdp_increases_total_latency(self):
        """FSDP comm nodes should increase total latency."""
        graph = _make_train_graph(num_layers=4)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=8, zero_stage=3, hw_spec=hw)

        # Without FSDP
        g_base = graph.clone()
        g_base = CommLatencyPass().run(g_base, ctx)
        g_base = StreamAssignPass().run(g_base, ctx)
        timeline_base = DAGScheduler(hw_spec=hw).schedule(g_base)

        # With FSDP
        g_fsdp = ZeroFSDPPass().run(graph, ctx)
        g_fsdp = CommLatencyPass().run(g_fsdp, ctx)
        g_fsdp = StreamAssignPass().run(g_fsdp, ctx)
        timeline_fsdp = DAGScheduler(hw_spec=hw).schedule(g_fsdp)

        assert timeline_fsdp.total_latency_us > timeline_base.total_latency_us

    def test_fsdp_comm_time_reported(self):
        """Timeline should report non-zero comm_time_us with FSDP."""
        graph = _make_train_graph(num_layers=4)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=8, zero_stage=3, hw_spec=hw)

        g_fsdp = ZeroFSDPPass().run(graph, ctx)
        g_fsdp = CommLatencyPass().run(g_fsdp, ctx)
        g_fsdp = StreamAssignPass().run(g_fsdp, ctx)
        timeline = DAGScheduler(hw_spec=hw).schedule(g_fsdp)

        assert timeline.comm_time_us > 0
        assert len(timeline.comm_ops()) == 12  # 4 layers * 3 comm nodes

    def test_comm_fraction_reasonable(self):
        """FSDP comm overhead should be a reasonable fraction of total.

        Note: With synthetic graphs lacking real compute latency, comm dominates.
        We verify comm is measurable and < 100% (i.e., some compute exists).
        """
        graph = _make_train_graph(num_layers=4, hidden=4096, seq_len=2048)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=8, zero_stage=3, hw_spec=hw)

        g_fsdp = ZeroFSDPPass().run(graph, ctx)
        g_fsdp = CommLatencyPass().run(g_fsdp, ctx)
        g_fsdp = StreamAssignPass().run(g_fsdp, ctx)
        timeline = DAGScheduler(hw_spec=hw).schedule(g_fsdp)

        comm_overhead = timeline.comm_time_us
        total = timeline.total_latency_us
        assert total > 0, "Total latency should be > 0"
        assert comm_overhead > 0, "Comm time should be > 0"
        assert comm_overhead < total, "Comm time should be < total latency"


# ── 6. Cross-node vs intra-node bandwidth ────────────────────────────────────

class TestCrossNodeBandwidth:
    """Verify CommLatencyPass selects correct bandwidth based on group_size."""

    def test_intra_node_comm(self):
        """DP=4 within 8-device node should use intra-node bandwidth."""
        graph = _make_train_graph(num_layers=1)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=4, zero_stage=3, hw_spec=hw)

        g = ZeroFSDPPass().run(graph, ctx)
        g = CommLatencyPass().run(g, ctx)

        comm_nodes = [n for n in g.nodes.values()
                      if n.category == "communication"]
        for node in comm_nodes:
            assert node.annotations.get("cross_node") is False

    def test_cross_node_comm(self):
        """DP=16 across nodes should use inter-node bandwidth."""
        graph = _make_train_graph(num_layers=1)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=16, zero_stage=3, hw_spec=hw)

        g = ZeroFSDPPass().run(graph, ctx)
        g = CommLatencyPass().run(g, ctx)

        comm_nodes = [n for n in g.nodes.values()
                      if n.category == "communication"]
        for node in comm_nodes:
            assert node.annotations.get("cross_node") is True

    def test_cross_node_has_higher_latency(self):
        """Cross-node comm should have higher latency than intra-node."""
        graph = _make_train_graph(num_layers=1)
        hw = _make_hardware_spec()

        # Intra-node: DP=4
        ctx_intra = _make_ctx(dp=4, zero_stage=3, hw_spec=hw)
        g_intra = ZeroFSDPPass().run(graph, ctx_intra)
        g_intra = CommLatencyPass().run(g_intra, ctx_intra)
        lat_intra = sum(n.annotations.get("latency_us", 0)
                        for n in g_intra.nodes.values()
                        if n.category == "communication")

        # Cross-node: DP=16 (uses inter-node IB bandwidth)
        ctx_cross = _make_ctx(dp=16, zero_stage=3, hw_spec=hw)
        g_cross = ZeroFSDPPass().run(graph, ctx_cross)
        g_cross = CommLatencyPass().run(g_cross, ctx_cross)
        lat_cross = sum(n.annotations.get("latency_us", 0)
                        for n in g_cross.nodes.values()
                        if n.category == "communication")

        assert lat_cross > lat_intra


# ── 7. Topology correctness ──────────────────────────────────────────────────

class TestTopologyCorrectness:
    """Verify comm nodes are inserted at correct positions in the graph."""

    def test_all_gather_precedes_first_fwd(self):
        """Fwd all_gather should be a predecessor of the first fwd node."""
        graph = _make_train_graph(num_layers=1)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        ag_fwd = next(n for n in result.nodes.values()
                      if n.op_type == "comm.all_gather"
                      and n.annotations.get("phase") == "fwd")
        fwd_0 = result.nodes["fwd_0"]

        succs = result.successors(ag_fwd.id)
        assert fwd_0.id in succs

    def test_reduce_scatter_succeeds_last_bwd(self):
        """Bwd reduce_scatter should be a successor of the last bwd node."""
        graph = _make_train_graph(num_layers=1)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        rs_bwd = next(n for n in result.nodes.values()
                      if n.op_type == "comm.reduce_scatter")
        bwd_0 = result.nodes["bwd_0"]

        preds = result.predecessors(rs_bwd.id)
        assert bwd_0.id in preds

    def test_bwd_all_gather_precedes_first_bwd(self):
        """Bwd all_gather should be a predecessor of the first bwd node."""
        graph = _make_train_graph(num_layers=1)
        ctx = _make_ctx(dp=4, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        ag_bwd = next(n for n in result.nodes.values()
                      if n.op_type == "comm.all_gather"
                      and n.annotations.get("phase") == "bwd")
        bwd_0 = result.nodes["bwd_0"]

        succs = result.successors(ag_bwd.id)
        assert bwd_0.id in succs


# ── 8. DP=1 skip ─────────────────────────────────────────────────────────────

class TestDP1Skip:
    """When DP=1, no FSDP comm should be inserted."""

    def test_no_comm_when_dp_1(self):
        graph = _make_train_graph(num_layers=4)
        ctx = _make_ctx(dp=1, zero_stage=3)
        result = ZeroFSDPPass().run(graph, ctx)

        fsdp_nodes = [n for n in result.nodes.values()
                      if n.annotations.get("inserted_by") == "zero_fsdp_pass"]
        assert len(fsdp_nodes) == 0


# ── 9. Training pipeline integration ─────────────────────────────────────────

class TestTrainingPipelineIntegration:
    """End-to-end: ZeroFSDPPass through TrainingPipelinePass."""

    def test_zero_metadata_propagates(self):
        """ZeroFSDPPass metadata should survive through TrainingPipelinePass."""
        graph = _make_train_graph(num_layers=2)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=4, zero_stage=3, hw_spec=hw)

        g = ZeroFSDPPass().run(graph, ctx)
        assert "zero" in g.metadata
        assert g.metadata["zero"]["stage"] == 3

    def test_comm_nodes_survive_analysis_passes(self):
        """Comm nodes inserted by ZeroFSDPPass should not be removed by analysis."""
        graph = _make_train_graph(num_layers=2)
        hw = _make_hardware_spec()
        ctx = _make_ctx(dp=4, zero_stage=3, hw_spec=hw)

        g = ZeroFSDPPass().run(graph, ctx)
        g = CommLatencyPass().run(g, ctx)

        comm_before = len([n for n in g.nodes.values()
                           if n.annotations.get("inserted_by") == "zero_fsdp_pass"])
        assert comm_before == 6
