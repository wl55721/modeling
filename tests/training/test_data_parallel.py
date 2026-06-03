"""Test data parallel pass: per-group comm insertion, ZeRO staging, overlap."""

import json
from types import SimpleNamespace

import pytest

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import TensorMeta, DType
from zrt.ir.edge import Edge
from zrt.transform.context import (
    TransformContext, ParallelConfig, TrainingConfig, OffloadConfig,
)
from zrt.transform.parallel.data_parallel import DataParallelPass
from zrt.transform.analysis import TrainingPipelinePass
from zrt.transform.analysis.passes import StreamAssignPass
from zrt.transform.analysis.comm_latency import CommLatencyPass
from zrt.executor.scheduler import DAGScheduler, ScheduledOp, Timeline
from zrt.executor.chrome_trace import ChromeTraceExporter
from zrt.executor.pp_stitcher import GridTask, PPStitchedTimeline
from zrt.transform.training.offload import OffloadPass
from zrt.transform.training.optimizer import OptimizerPass


def _make_backward_graph(num_layers=2, hidden=4096, seq_len=2048):
    """Create a backward-phase graph with gradient-producing nodes."""
    nodes = {}
    edges = []

    for i in range(num_layers):
        grad_out = TensorMeta(
            id=f"grad_out_{i}",
            shape=(1, seq_len, hidden),
            dtype=DType.BF16,
            mem_bytes=seq_len * hidden * 2,
        )
        grad_node = OpNode(
            id=f"grad_node_{i}",
            op_type=f"aten.mm_backward",
            inputs=[TensorMeta(id=f"grad_in_{i}", shape=(1, seq_len, hidden),
                               dtype=DType.BF16, mem_bytes=seq_len * hidden * 2)],
            outputs=[grad_out],
            scope=f"model.layers.{i}.self_attn.q_proj",
            layer=str(i),
            category="compute",
        )
        nodes[grad_node.id] = grad_node

        if i > 0:
            edges.append(Edge(
                src=f"grad_node_{i-1}", src_idx=0,
                dst=f"grad_node_{i}", dst_idx=0,
                tensor=grad_out,
            ))

    return OpGraph(
        name="test_dp_model",
        phase="train_backward",
        nodes=nodes,
        edges=edges,
        metadata={"seq_len": seq_len, "hidden": hidden, "num_layers": num_layers},
    )


def _make_hardware_spec():
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


def _make_ddp_tail_timeline():
    return Timeline(scheduled_ops=[
        ScheduledOp(
            node_id="bwd_0",
            stream_id=0,
            stream_type="compute",
            start_us=0.0,
            end_us=100.0,
            latency_us=100.0,
            op_type="aten.mm_backward",
            category="compute",
            phase="bwd",
        ),
        ScheduledOp(
            node_id="comm_grad_reduce_bucket_0",
            stream_id=1,
            stream_type="comm",
            start_us=100.0,
            end_us=180.0,
            latency_us=80.0,
            op_type="comm.all_reduce",
            category="communication",
            phase="bwd",
            parallelism_tag="dp",
            attrs={"role": "dp_grad_reduce"},
        ),
        ScheduledOp(
            node_id="bwd_1",
            stream_id=0,
            stream_type="compute",
            start_us=100.0,
            end_us=200.0,
            latency_us=100.0,
            op_type="aten.mm_backward",
            category="compute",
            phase="bwd",
        ),
        ScheduledOp(
            node_id="comm_grad_reduce_bucket_1",
            stream_id=1,
            stream_type="comm",
            start_us=200.0,
            end_us=320.0,
            latency_us=120.0,
            op_type="comm.all_reduce",
            category="communication",
            phase="bwd",
            parallelism_tag="dp",
            attrs={"role": "dp_grad_reduce", "bucket_ready_node": "bwd_1"},
        ),
        ScheduledOp(
            node_id="ddp_wait_all_buckets",
            stream_id=1,
            stream_type="comm",
            start_us=320.0,
            end_us=320.0,
            latency_us=0.0,
            op_type="comm.wait",
            category="communication",
            phase="bwd",
            parallelism_tag="dp",
            attrs={"role": "ddp_wait_all_buckets"},
        ),
    ])


class TestDPZero0:
    """ZeRO-0: all_reduce comm nodes."""

    def test_all_reduce_created_per_layer_by_default(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=False,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 3

        for node in dp_nodes:
            assert node.op_type == "comm.all_reduce"
            assert node.attrs["group_size"] == 4
            assert node.attrs["collective"] == "all_reduce"
            assert node.attrs["bucket_bytes"] > 0
            assert node.id.startswith("comm_grad_reduce_layer_")

    def test_dp_comm_annotation_present(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.annotations["dp_comm"] is True
            assert node.annotations["inserted_by"] == "data_parallel_pass"


class TestDPZero2:
    """ZeRO-2/3: reduce_scatter comm nodes."""

    def test_reduce_scatter_for_zero2(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.op_type == "comm.reduce_scatter"
            assert node.attrs["collective"] == "reduce_scatter"

    def test_reduce_scatter_for_zero3(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=3),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.op_type == "comm.reduce_scatter"

    def test_reduce_scatter_has_lower_modeled_time_than_all_reduce(self):
        graph = _make_backward_graph(num_layers=1)
        ctx_z0 = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )
        ctx_z2 = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        ar_graph = DataParallelPass().run(graph, ctx_z0)
        rs_graph = DataParallelPass().run(graph, ctx_z2)

        ar_time = TrainingPipelinePass._compute_dp_ar_time(ar_graph, _make_hardware_spec(), ctx_z0)
        rs_time = TrainingPipelinePass._compute_dp_ar_time(rs_graph, _make_hardware_spec(), ctx_z2)

        assert rs_time == pytest.approx(ar_time / 2)

    def test_dp_ar_time_treats_interconnect_bandwidth_as_gb_per_second(self):
        dp = 4
        bucket_bytes = 900_000
        comm_node = OpNode(
            id="comm_grad_reduce",
            op_type="comm.all_reduce",
            inputs=[], outputs=[],
            attrs={
                "group_size": dp,
                "collective": "all_reduce",
                "bucket_bytes": bucket_bytes,
                "role": "dp_grad_reduce",
            },
            scope="data_parallel.grad_reduce",
            category="communication",
        )
        comm_node.annotations["dp_comm"] = True

        graph = OpGraph(
            name="test_dp_bw_units",
            phase="train_backward",
            nodes={"comm_grad_reduce": comm_node},
            metadata={"seq_len": 4096, "hidden": 4096},
        )
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=dp),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        ar_time = TrainingPipelinePass._compute_dp_ar_time(
            graph, _make_hardware_spec(), ctx,
        )

        expected = 2.0 * (dp - 1) / dp * bucket_bytes / (900e9 / 1e6)
        assert ar_time == pytest.approx(expected)

    def test_comm_latency_uses_dp_bucket_bytes(self):
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        result = CommLatencyPass().run(dp_graph, ctx)
        comm_node = next(n for n in result.nodes.values() if n.annotations.get("dp_comm"))

        bucket_bytes = comm_node.attrs["bucket_bytes"]
        link = ctx.hw_spec.interconnect.intra_node
        expected = ((4 - 1) / 4 * bucket_bytes) / (link.effective_bw_bps(4) / 1e6)
        expected += (4 - 1) * link.latency_us
        assert comm_node.annotations["latency_us"] == pytest.approx(expected)


class TestDPOverlap:
    """Tests for DP overlap-in-bubble behavior."""

    def test_overlap_annotation_set(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=True,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert node.annotations.get("overlap_in_bubble") is True

    def test_no_overlap_annotation_when_disabled(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=False,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        for node in dp_nodes:
            assert "overlap_in_bubble" not in node.annotations

    def test_dp_skip_when_dp1(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values()
                    if n.annotations.get("dp_comm")]
        assert len(dp_nodes) == 0

    def test_default_layer_bucket_preserves_original_rewire_behavior(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        assert "grad_node_1" not in result.successors("grad_node_0")
        assert "comm_grad_reduce_layer_0" in result.successors("grad_node_0")
        assert "grad_scale_layer_0" in result.successors("comm_grad_reduce_layer_0")
        assert "grad_scale_layer_0" in result.predecessors("grad_node_1")
        assert "grad_node_1" in result.successors("grad_scale_layer_0")
        assert result.nodes["comm_grad_reduce_layer_0"].attrs["bucket_ready_node"] == "grad_node_0"

    def test_default_layer_bucket_keeps_original_overlap_annotation(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = [n for n in result.nodes.values() if n.annotations.get("dp_comm")]
        assert dp_nodes
        assert all(n.annotations.get("overlap_in_bubble") is True for n in dp_nodes)

    def test_no_dp_overlap_layer_bucket_blocks_later_backward_compute(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=False,
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        for node_id, latency_us in {
            "grad_node_0": 100.0,
            "comm_grad_reduce_layer_0": 80.0,
            "grad_scale_layer_0": 1.0,
            "grad_node_1": 100.0,
        }.items():
            dp_graph.nodes[node_id].annotations["latency_us"] = latency_us

        scheduled_graph = StreamAssignPass().run(dp_graph, ctx)
        timeline = DAGScheduler().schedule(scheduled_graph)
        by_id = {op.node_id: op for op in timeline.scheduled_ops}

        first_comm = by_id["comm_grad_reduce_layer_0"]
        scale = by_id["grad_scale_layer_0"]
        later_bwd = by_id["grad_node_1"]

        assert first_comm.start_us == pytest.approx(by_id["grad_node_0"].end_us)
        assert scale.start_us == pytest.approx(first_comm.end_us)
        assert later_bwd.start_us == pytest.approx(scale.end_us)
        assert first_comm.end_us <= later_bwd.start_us

    def test_ddp_bucket_comm_is_side_branch(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=True,
                dp_bucket_mode="ddp",
                dp_bucket_cap_mb=1.0,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        assert "grad_node_1" in result.successors("grad_node_0")
        assert "comm_grad_reduce_bucket_0" in result.successors("grad_node_0")
        assert "comm_grad_reduce_bucket_0" not in result.predecessors("grad_node_1")
        assert "grad_node_1" not in result.successors("comm_grad_reduce_bucket_0")
        assert result.nodes["comm_grad_reduce_bucket_0"].attrs["bucket_ready_node"] == "grad_node_0"
        assert "ddp_wait_all_buckets" in result.nodes
        assert "ddp_wait_all_buckets" in result.successors("grad_scale_bucket_0")
        assert "ddp_wait_all_buckets" in result.successors("grad_scale_bucket_1")
        assert result.nodes["ddp_wait_all_buckets"].attrs["bucket_count"] == 2

    def test_ddp_bucket_comm_overlaps_later_backward_compute(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_overlap_in_bubble=True,
                dp_bucket_mode="ddp",
                dp_bucket_cap_mb=1.0,
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        for node_id, latency_us in {
            "grad_node_0": 100.0,
            "grad_node_1": 100.0,
            "comm_grad_reduce_bucket_0": 80.0,
            "comm_grad_reduce_bucket_1": 80.0,
            "grad_scale_bucket_0": 1.0,
            "grad_scale_bucket_1": 1.0,
            "ddp_wait_all_buckets": 0.0,
        }.items():
            dp_graph.nodes[node_id].annotations["latency_us"] = latency_us

        scheduled_graph = StreamAssignPass().run(dp_graph, ctx)
        timeline = DAGScheduler().schedule(scheduled_graph)
        by_id = {op.node_id: op for op in timeline.scheduled_ops}

        first_comm = by_id["comm_grad_reduce_bucket_0"]
        later_bwd = by_id["grad_node_1"]
        scale = by_id["grad_scale_bucket_0"]
        wait = by_id["ddp_wait_all_buckets"]

        assert first_comm.stream_type == "comm"
        assert first_comm.parallelism_tag == "dp"
        assert first_comm.start_us == pytest.approx(by_id["grad_node_0"].end_us)
        assert later_bwd.start_us == pytest.approx(by_id["grad_node_0"].end_us)
        assert scale.stream_type == "comm"
        assert scale.start_us == pytest.approx(first_comm.end_us)
        assert later_bwd.start_us < scale.start_us
        assert min(first_comm.end_us, later_bwd.end_us) > max(first_comm.start_us, later_bwd.start_us)
        assert wait.start_us == pytest.approx(by_id["grad_scale_bucket_1"].end_us)

    def test_no_dp_overlap_ddp_bucket_blocks_later_backward_compute(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1,
                global_batch=8,
                zero_stage=0,
                dp_overlap_in_bubble=False,
                dp_bucket_mode="ddp",
                dp_bucket_cap_mb=1.0,
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        assert dp_graph.nodes["comm_grad_reduce_bucket_0"].annotations["blocking_comm"] is True
        for node_id, latency_us in {
            "grad_node_0": 100.0,
            "comm_grad_reduce_bucket_0": 80.0,
            "grad_scale_bucket_0": 1.0,
            "grad_node_1": 100.0,
        }.items():
            dp_graph.nodes[node_id].annotations["latency_us"] = latency_us

        scheduled_graph = StreamAssignPass().run(dp_graph, ctx)
        timeline = DAGScheduler().schedule(scheduled_graph)
        by_id = {op.node_id: op for op in timeline.scheduled_ops}

        first_comm = by_id["comm_grad_reduce_bucket_0"]
        scale = by_id["grad_scale_bucket_0"]
        later_bwd = by_id["grad_node_1"]

        assert first_comm.start_us == pytest.approx(by_id["grad_node_0"].end_us)
        assert scale.start_us == pytest.approx(first_comm.end_us)
        assert later_bwd.start_us == pytest.approx(scale.end_us)
        assert first_comm.end_us <= later_bwd.start_us


class TestDPGroupIdx:
    """Tests for per-group index assignment."""

    def test_group_idx_sequential_for_default_layer_buckets(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = sorted(
            [n for n in result.nodes.values() if n.annotations.get("dp_comm")],
            key=lambda n: n.attrs["dp_grad_group_idx"],
        )
        indices = [n.attrs["dp_grad_group_idx"] for n in dp_nodes]
        assert indices == [0, 1, 2]

    def test_ddp_mode_accumulates_ready_grads_into_cap_buckets(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_bucket_mode="ddp",
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = sorted(
            [n for n in result.nodes.values() if n.annotations.get("dp_comm")],
            key=lambda n: n.attrs["bucket_index"],
        )
        assert len(dp_nodes) == 2
        assert dp_nodes[0].attrs["bucket_param_count"] == 2
        assert dp_nodes[0].attrs["bucket_ready_node"] == "grad_node_1"
        assert dp_nodes[1].attrs["bucket_param_count"] == 1
        assert dp_nodes[1].attrs["bucket_ready_node"] == "grad_node_2"

    def test_ddp_mode_multi_layer_bucket_attrs_export_to_trace_args(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_bucket_mode="ddp",
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        bucket = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm") and n.attrs["bucket_index"] == 0
        )
        assert bucket.attrs["bucket_layers"] == ["0", "1"]

        args = ChromeTraceExporter._bucket_args(SimpleNamespace(attrs=bucket.attrs))
        assert args["bucket_layers"] == ["0", "1"]
        assert args["bucket_source_ids"] == ["grad_node_0", "grad_node_1"]

    def test_ddp_bucket_tail_exposes_only_comm_after_backward_end(self):
        timeline = _make_ddp_tail_timeline()

        has_buckets, exposed_tail = (
            TrainingPipelinePass._ddp_bucket_exposed_tail_us({0: timeline})
        )

        assert has_buckets is True
        assert exposed_tail == pytest.approx(120.0)
        assert TrainingPipelinePass._phase_latency(
            timeline, "bwd", exclude_dp=True,
        ) == pytest.approx(200.0)

    def test_ddp_bucket_mode_step_result_uses_timeline_tail(self):
        timeline = _make_ddp_tail_timeline()
        graph = OpGraph(name="ddp_tail", phase="train_backward")

        result = TrainingPipelinePass._build_trace_step_result(
            graph,
            pp=1,
            M=1,
            pp_schedule="1f1b",
            vpp_chunks=1,
            stage_fwd={0: 0.0},
            stage_bwd={0: 200.0},
            stage_bwd_dw={0: 0.0},
            strategy_proxy=SimpleNamespace(dp_overlap_in_bubble=True),
            dp_ar_time_s=200.0 / 1e6,
            stage_timelines={0: timeline},
            dp_bucket_mode="ddp",
        )

        assert result.dp_exposed == pytest.approx(120.0 / 1e6)
        assert result.dp_hidden == pytest.approx(80.0 / 1e6)
        assert result.step_time == pytest.approx((200.0 + 120.0) / 1e6)
        assert graph.metadata["dp_bucket_exposed_tail_us"] == pytest.approx(120.0)

    def test_ddp_tail_ignores_later_non_bucket_ready_bwd_bookkeeping(self):
        timeline = _make_ddp_tail_timeline()
        timeline.scheduled_ops.append(
            ScheduledOp(
                node_id="bwd_bookkeeping_after_last_bucket_ready",
                stream_id=0,
                stream_type="compute",
                start_us=200.0,
                end_us=300.0,
                latency_us=100.0,
                op_type="aten.new_empty_strided.default",
                category="compute",
                phase="bwd",
            )
        )

        has_buckets, exposed_tail = (
            TrainingPipelinePass._ddp_bucket_exposed_tail_us({0: timeline})
        )

        assert has_buckets is True
        assert exposed_tail == pytest.approx(120.0)

    def test_chrome_trace_keeps_raw_lanes_and_marks_reduce_tail_args(self):
        timeline = _make_ddp_tail_timeline()
        timeline.scheduled_ops.append(
            ScheduledOp(
                node_id="bwd_bookkeeping_after_last_bucket_ready",
                stream_id=0,
                stream_type="compute",
                start_us=200.0,
                end_us=300.0,
                latency_us=100.0,
                op_type="aten.new_empty_strided.default",
                category="compute",
                phase="bwd",
            )
        )
        stitched = PPStitchedTimeline(
            tasks=[
                GridTask(
                    task_id="s0_m0_bwd",
                    stage_id=0,
                    mb_id=0,
                    phase="bwd",
                    latency_us=200.0,
                    stream_id=0,
                    start_us=1000.0,
                    end_us=1200.0,
                ),
            ],
            pp=1,
            M=1,
            schedule_name="1f1b",
        )

        doc = json.loads(ChromeTraceExporter().export_stitched_detailed(stitched, [timeline]))
        bookkeeping_events = [
            event for event in doc["traceEvents"]
            if event.get("args", {}).get("node_id") == "bwd_bookkeeping_after_last_bucket_ready"
        ]
        reduce_events = [
            event for event in doc["traceEvents"]
            if event.get("args", {}).get("node_id") == "comm_grad_reduce_bucket_1"
        ]
        thread_names = {
            event["args"]["name"]
            for event in doc["traceEvents"]
            if event.get("name") == "thread_name" and event.get("pid") == 0
        }

        assert "Bookkeeping Ops" not in thread_names
        assert len(bookkeeping_events) == 1
        bookkeeping = bookkeeping_events[0]
        assert bookkeeping["cat"] == "compute"
        assert bookkeeping["tid"] == 2
        assert bookkeeping["ts"] == pytest.approx(1200.0)
        assert bookkeeping["dur"] == pytest.approx(100.0)
        assert bookkeeping["args"]["view"] == "detail"

        assert len(reduce_events) == 1
        reduce = reduce_events[0]
        assert reduce["cat"] == "communication"
        assert reduce["tid"] == 3
        assert reduce["args"]["ddp_exposed_tail_us"] == pytest.approx(120.0)
        assert reduce["args"]["bucket_ready_ts_us"] == pytest.approx(200.0)
        assert reduce["args"]["ddp_complete_ts_us"] == pytest.approx(320.0)

    def test_ddp_wait_barrier_precedes_optimizer_step(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_bucket_mode="ddp",
                dp_bucket_cap_mb=1.0,
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        result = OptimizerPass().run(dp_graph, ctx)

        assert "ddp_wait_all_buckets" in result.predecessors("optimizer_step")

    def test_ddp_mode_small_cap_flushes_each_ready_grad(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_bucket_mode="ddp",
                dp_bucket_cap_mb=1.0,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = sorted(
            [n for n in result.nodes.values() if n.annotations.get("dp_comm")],
            key=lambda n: n.attrs["bucket_index"],
        )
        assert len(dp_nodes) == 3
        assert [n.attrs["bucket_ready_node"] for n in dp_nodes] == [
            "grad_node_0", "grad_node_1", "grad_node_2",
        ]

    def test_ddp_mode_pp_does_not_bucket_across_layers(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4, pp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                dp_bucket_mode="ddp",
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        dp_nodes = sorted(
            [n for n in result.nodes.values() if n.annotations.get("dp_comm")],
            key=lambda n: n.attrs["bucket_index"],
        )
        assert len(dp_nodes) == 3
        assert [n.attrs["bucket_layers"] for n in dp_nodes] == [["0"], ["1"], ["2"]]
        assert [n.layer for n in dp_nodes] == ["0", "1", "2"]


class TestDPDivScale:
    """Tests for aten.div.Scalar gradient averaging nodes.

    DataParallelPass inserts a div/scale node after every DP comm node
    so that all_reduce / reduce_scatter SUM is averaged by dp.
    """

    def test_div_scale_node_created_per_layer_by_default(self):
        graph = _make_backward_graph(num_layers=3)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
            ),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.annotations.get("inserted_by") == "data_parallel_pass"
            and n.op_type == "aten.div.Scalar"
        ]
        assert len(scale_nodes) == 3

    def test_div_scale_divisor_equals_dp(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=8),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        ]
        for node in scale_nodes:
            assert node.attrs["divisor"] == 8
            assert node.attrs["role"] == "dp_grad_average"

    def test_div_scale_inserted_after_comm_node(self):
        """Verify div/scale is a successor of the corresponding comm node."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        comm_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm")
        )
        scale_node = next(
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        )

        comm_succs = result.successors(comm_node.id)
        assert scale_node.id in comm_succs, (
            f"div/scale {scale_node.id} not a successor of "
            f"comm node {comm_node.id}"
        )

    def test_div_scale_has_correct_annotations(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        ]
        for node in scale_nodes:
            assert node.annotations["inserted_by"] == "data_parallel_pass"
            assert node.annotations["phase"] == "bwd"
            assert node.category == "compute"

    def test_div_scale_not_inserted_when_dp_1(self):
        graph = _make_backward_graph(num_layers=2)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=1),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=0),
        )

        result = DataParallelPass().run(graph, ctx)

        scale_nodes = [
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        ]
        assert len(scale_nodes) == 0

    def test_div_scale_divisor_matches_reduce_scatter(self):
        """div/scale divisor equals dp regardless of collective type."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(micro_batch=1, global_batch=8, zero_stage=2),
        )

        result = DataParallelPass().run(graph, ctx)

        comm_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm")
        )
        assert comm_node.op_type == "comm.reduce_scatter"

        scale_node = next(
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        )
        assert scale_node.attrs["divisor"] == 4


class TestDPOffloadChain:
    """Tests for OffloadPass _find_dp_chain_end with div/scale nodes.

    When DataParallelPass inserts a div/scale node after the DP comm node,
    OffloadPass must walk past it so that D2H is inserted after the averaged
    gradients, not before.
    """

    def test_find_dp_chain_end_skips_div_scale(self):
        """_find_dp_chain_end from comm node should return scale_node."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                offload=OffloadConfig(pct=1.0, grads=True),
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)

        comm_node = next(
            n for n in dp_graph.nodes.values()
            if n.annotations.get("dp_comm")
        )

        offload_pass = OffloadPass()
        chain_end = offload_pass._find_dp_chain_end(dp_graph, comm_node)

        assert chain_end is not comm_node, (
            "_find_dp_chain_end should skip past div/scale to the chain end"
        )
        assert chain_end.op_type == "aten.div.Scalar", (
            f"expected div/scale at chain end, got {chain_end.op_type}"
        )
        assert chain_end.annotations["inserted_by"] == "data_parallel_pass"

    def test_d2h_inserted_after_div_scale(self):
        """After OffloadPass runs, D2H should be a successor of div/scale,
        not of the comm node directly."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                offload=OffloadConfig(pct=1.0, grads=True),
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        result = OffloadPass().run(dp_graph, ctx)

        comm_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("dp_comm")
        )
        scale_node = next(
            n for n in result.nodes.values()
            if n.op_type == "aten.div.Scalar"
        )
        d2h_node = next(
            n for n in result.nodes.values()
            if n.annotations.get("inserted_by") == "offload_pass"
        )

        scale_succs = result.successors(scale_node.id)
        assert d2h_node.id in scale_succs, (
            f"D2H {d2h_node.id} should be a successor of "
            f"div/scale {scale_node.id}, got succs={scale_succs}"
        )

        comm_succs = result.successors(comm_node.id)
        assert scale_node.id in comm_succs, (
            "comm node should still point to div/scale"
        )

    def test_no_offload_when_grads_disabled(self):
        """D2H should not be inserted when offload.grads is False."""
        graph = _make_backward_graph(num_layers=1)
        ctx = TransformContext(
            hw_spec=_make_hardware_spec(),
            parallel=ParallelConfig(tp=1, dp=4),
            training=TrainingConfig(
                micro_batch=1, global_batch=8, zero_stage=0,
                offload=OffloadConfig(pct=1.0, grads=False),
            ),
        )

        dp_graph = DataParallelPass().run(graph, ctx)
        result = OffloadPass().run(dp_graph, ctx)

        d2h_nodes = [
            n for n in result.nodes.values()
            if n.annotations.get("inserted_by") == "offload_pass"
        ]
        assert len(d2h_nodes) == 0

    def test_find_dp_chain_end_stops_at_non_dp_node(self):
        """When div/scale is NOT present (legacy graph), chain end is the
        comm node itself."""
        graph = _make_backward_graph(num_layers=1)

        dp = 4
        comm_node = OpNode(
            id="comm_grad_reduce",
            op_type="comm.all_reduce",
            inputs=[], outputs=[],
            attrs={"group_size": dp, "collective": "all_reduce",
                   "bucket_bytes": 1000, "role": "dp_grad_reduce"},
            scope="data_parallel.grad_reduce",
            category="communication",
        )
        comm_node.annotations["dp_comm"] = True
        comm_node.annotations["inserted_by"] = "data_parallel_pass"
        comm_node.annotations["phase"] = "bwd"

        g = OpGraph(
            name="test_no_scale",
            phase="train_backward",
            nodes={"comm_grad_reduce": comm_node},
            metadata={"seq_len": 4096, "hidden": 4096},
        )

        offload_pass = OffloadPass()
        chain_end = offload_pass._find_dp_chain_end(g, comm_node)

        assert chain_end is comm_node, (
            "without div/scale, chain end should be the comm node itself"
        )
