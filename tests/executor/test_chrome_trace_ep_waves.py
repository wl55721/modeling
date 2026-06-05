import json

from python.zrt.executor.chrome_trace import ChromeTraceExporter
from python.zrt.executor.pp_stitcher import GridTask, PPStitchedTimeline
from python.zrt.executor.scheduler import ScheduledOp, Timeline


def test_ep_wave_scope_match_requires_non_empty_scopes():
    same_scope = ChromeTraceExporter._same_ep_region_scope(
        "transformer.layers.0.ffn.moe",
        "transformer.layers.0.ffn.experts",
    )

    assert same_scope is True
    assert ChromeTraceExporter._same_ep_region_scope("", "") is False
    assert ChromeTraceExporter._same_ep_region_scope("transformer.layers.0.ffn.moe", "") is False
    assert ChromeTraceExporter._same_ep_region_scope("", "transformer.layers.0.ffn.experts") is False


def test_ep_wave_trace_segments_dispatch_compute_and_combine_on_original_streams():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=100.0,
                stream_id=0,
                start_us=0.0,
                end_us=100.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=100.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=20.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="expert",
                stream_id=0,
                stream_type="compute",
                start_us=20.0,
                end_us=80.0,
                latency_us=60.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=80.0,
                end_us=100.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
        ]
    )

    doc = ChromeTraceExporter(trace_ep_waves=True).export_stitched_detailed(
        stitched, [timeline]
    )
    events = json.loads(doc)["traceEvents"]
    names = [e.get("name", "") for e in events]

    assert "m0:fwd:wave0-dispatch" in names
    assert "m0:fwd:wave1-dispatch" in names
    assert "m0:fwd:wave0-expert-GroupedMatMul" in names
    assert "m0:fwd:wave1-expert-GroupedMatMul" in names
    assert "m0:fwd:wave0-combine" in names
    assert "m0:fwd:wave1-combine" in names
    assert "m0:fwd:comm.all_to_all" not in names

    by_name = {e["name"]: e for e in events}
    assert by_name["m0:fwd:wave0-dispatch"]["tid"] == 3
    assert by_name["m0:fwd:wave0-combine"]["tid"] == 3
    assert by_name["m0:fwd:wave0-expert-GroupedMatMul"]["tid"] == 2
    assert by_name["m0:fwd:wave0-dispatch"]["args"]["role"] == "dispatch"
    assert by_name["m0:fwd:wave0-combine"]["args"]["role"] == "combine"
    assert by_name["m0:fwd:wave1-expert-GroupedMatMul"]["args"]["wave"] == 1

    assert by_name["m0:fwd:wave0-dispatch"]["ts"] == 0.0
    assert by_name["m0:fwd:wave1-dispatch"]["ts"] == 10.0
    assert by_name["m0:fwd:wave0-expert-GroupedMatMul"]["ts"] == 20.0
    assert by_name["m0:fwd:wave0-combine"]["ts"] == 50.0
    assert by_name["m0:fwd:wave1-expert-GroupedMatMul"]["ts"] == 50.0
    assert by_name["m0:fwd:wave1-combine"]["ts"] == 80.0

    assert by_name["m0:fwd:wave1-dispatch"]["ts"] < (
        by_name["m0:fwd:wave0-expert-GroupedMatMul"]["ts"]
        + by_name["m0:fwd:wave0-expert-GroupedMatMul"]["dur"]
    )
    assert by_name["m0:fwd:wave0-combine"]["ts"] == (
        by_name["m0:fwd:wave0-expert-GroupedMatMul"]["ts"]
        + by_name["m0:fwd:wave0-expert-GroupedMatMul"]["dur"]
    )


def test_moe_fb_trace_marks_unannotated_all_to_all_as_exposed_without_overlay():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=100.0,
                stream_id=0,
                start_us=0.0,
                end_us=100.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=100.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=10.0,
                end_us=30.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=70.0,
                end_us=90.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
        ]
    )

    doc = ChromeTraceExporter(
        trace_ep_waves=False,
        trace_moe_fb_overlap=True,
    ).export_stitched_detailed(stitched, [timeline])
    events = json.loads(doc)["traceEvents"]
    names = [e.get("name", "") for e in events]
    fb_events = [
        e for e in events
        if e.get("args", {}).get("overlap") == "moe_fb"
    ]

    assert not any("wave" in name for name in names)
    assert len(fb_events) == 2
    assert {e["cat"] for e in fb_events} == {"communication.ep.moe_fb.exposed"}
    assert {e["tid"] for e in fb_events} == {3}
    assert {e["args"]["op_type"] for e in fb_events} == {"comm.all_to_all"}
    assert {e["args"]["role"] for e in fb_events} == {"dispatch", "combine"}
    assert all("comm.all_to_all" in e["name"] for e in fb_events)
    assert all("moe_fb exposed" in e["name"] for e in fb_events)
    assert all(e["args"]["hidden"] is False for e in fb_events)


def test_moe_fb_trace_marks_annotated_hidden_all_to_all():
    op = ScheduledOp(
        node_id="dispatch",
        stream_id=1,
        stream_type="comm",
        start_us=0.0,
        end_us=20.0,
        latency_us=20.0,
        op_type="comm.all_to_all",
        category="communication",
        phase="fwd",
        parallelism_tag="ep",
        comm_role="dispatch",
        overlap_hidden_us=12.0,
        overlap_exposed_us=8.0,
    )
    exporter = ChromeTraceExporter(trace_moe_fb_overlap=True)

    args = exporter._moe_fb_event_args(op, {"view": "detail"})

    assert exporter._moe_fb_event_cat(op, "communication") == "communication.ep.moe_fb.hidden"
    assert "moe_fb hidden dispatch" in exporter._moe_fb_event_name(op, "dispatch")
    assert args["hidden"] is True
    assert args["hidden_us"] == 12.0
    assert args["exposed_us"] == 8.0


def test_ep_wave_trace_starts_region_at_grouped_matmul():
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=20.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="activation",
                stream_id=0,
                stream_type="compute",
                start_us=20.0,
                end_us=22.0,
                latency_us=2.0,
                op_type="aten.silu.default",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.activation",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="grouped",
                stream_id=0,
                stream_type="compute",
                start_us=22.0,
                end_us=82.0,
                latency_us=60.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=82.0,
                end_us=102.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
        ]
    )

    group = ChromeTraceExporter(trace_ep_waves=True)._find_ep_wave_group(
        timeline.scheduled_ops, 0
    )

    assert group is not None
    assert [op.node_id for op in group[1]] == ["grouped"]


def test_ep_wave_trace_splits_whole_expert_region_before_combine():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=210.0,
                stream_id=0,
                start_us=0.0,
                end_us=210.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=210.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=20.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="gate",
                stream_id=0,
                stream_type="compute",
                start_us=20.0,
                end_us=40.0,
                latency_us=20.0,
                op_type="moe_gate",
                category="compute",
                phase="fwd",
                scope="transformer.layers.0.ffn.gate",
            ),
            ScheduledOp(
                node_id="gate_up",
                stream_id=0,
                stream_type="compute",
                start_us=40.0,
                end_us=100.0,
                latency_us=60.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="silu",
                stream_id=0,
                stream_type="compute",
                start_us=120.0,
                end_us=130.0,
                latency_us=10.0,
                op_type="aten.silu",
                category="compute",
                phase="fwd",
                scope="transformer.layers.0.ffn.moe",
            ),
            ScheduledOp(
                node_id="inner_gate",
                stream_id=0,
                stream_type="compute",
                start_us=100.0,
                end_us=120.0,
                latency_us=20.0,
                op_type="moe_gate",
                category="compute",
                phase="fwd",
                module_class="Gate",
                scope="transformer.layers.0.ffn.gate",
            ),
            ScheduledOp(
                node_id="grouped_down",
                stream_id=0,
                stream_type="compute",
                start_us=130.0,
                end_us=190.0,
                latency_us=60.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
            ScheduledOp(
                node_id="post_down",
                stream_id=0,
                stream_type="compute",
                start_us=195.0,
                end_us=200.0,
                latency_us=5.0,
                op_type="linear",
                category="compute",
                phase="fwd",
                scope="transformer.layers.0.ffn.other",
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=190.0,
                end_us=210.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=2,
            ),
        ]
    )

    doc = ChromeTraceExporter(trace_ep_waves=True).export_stitched_detailed(
        stitched, [timeline]
    )
    events = json.loads(doc)["traceEvents"]
    by_name = {e["name"]: e for e in events if e.get("ph") == "X"}

    assert "m0:fwd:moe_gate" in by_name
    assert "m0:fwd:wave0-expert-moe_gate" not in by_name
    assert "m0:fwd:wave0-expert-linear" not in by_name
    assert "m0:fwd:wave0-expert-GroupedMatMul" in by_name
    assert "m0:fwd:wave0-expert-aten.silu" in by_name
    assert "m0:fwd:wave0-expert-GroupedMatMul:down" in by_name

    raw_gates = sorted(
        [e for e in events if e.get("name") == "m0:fwd:moe_gate"],
        key=lambda e: e["ts"],
    )
    gate_end = raw_gates[0]["ts"] + raw_gates[0]["dur"]
    wave0_dispatch = by_name["m0:fwd:wave0-dispatch"]
    wave0_gate_up = by_name["m0:fwd:wave0-expert-GroupedMatMul"]
    wave0_silu = by_name["m0:fwd:wave0-expert-aten.silu"]
    wave0_down = by_name["m0:fwd:wave0-expert-GroupedMatMul:down"]
    wave0_combine = by_name["m0:fwd:wave0-combine"]
    blocker_end = raw_gates[1]["ts"] + raw_gates[1]["dur"]

    assert wave0_dispatch["ts"] >= gate_end
    assert wave0_gate_up["ts"] >= gate_end
    assert wave0_silu["ts"] == wave0_gate_up["ts"] + wave0_gate_up["dur"]
    assert wave0_down["ts"] == blocker_end
    assert wave0_combine["ts"] == wave0_down["ts"] + wave0_down["dur"]


def test_ep_wave_trace_pipelines_later_dispatch_under_first_compute():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=200.0,
                stream_id=0,
                start_us=0.0,
                end_us=200.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=200.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=40.0,
                latency_us=40.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_gate_up",
                stream_id=0,
                stream_type="compute",
                start_us=40.0,
                end_us=80.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_down",
                stream_id=0,
                stream_type="compute",
                start_us=120.0,
                end_us=120.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=120.0,
                end_us=160.0,
                latency_us=40.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=4,
            ),
        ]
    )

    doc = ChromeTraceExporter(trace_ep_waves=True).export_stitched_detailed(
        stitched, [timeline]
    )
    events = json.loads(doc)["traceEvents"]
    by_name = {e["name"]: e for e in events if e.get("ph") == "X"}

    wave0_compute = by_name["m0:fwd:wave0-expert-GroupedMatMul"]
    wave1_dispatch = by_name["m0:fwd:wave1-dispatch"]
    wave2_dispatch = by_name["m0:fwd:wave2-dispatch"]
    wave3_dispatch = by_name["m0:fwd:wave3-dispatch"]
    wave0_compute_end = wave0_compute["ts"] + wave0_compute["dur"]

    assert wave1_dispatch["ts"] < wave0_compute_end
    assert wave2_dispatch["ts"] < wave0_compute_end
    assert wave3_dispatch["ts"] < wave0_compute_end


def test_ep_wave_trace_exposes_long_dispatch_when_compute_cannot_hide_it():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=260.0,
                stream_id=0,
                start_us=0.0,
                end_us=260.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=260.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=160.0,
                latency_us=160.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_gate_up",
                stream_id=0,
                stream_type="compute",
                start_us=40.0,
                end_us=80.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_down",
                stream_id=0,
                stream_type="compute",
                start_us=80.0,
                end_us=120.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=120.0,
                end_us=160.0,
                latency_us=40.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=4,
            ),
        ]
    )

    doc = ChromeTraceExporter(trace_ep_waves=True).export_stitched_detailed(
        stitched, [timeline]
    )
    events = json.loads(doc)["traceEvents"]
    by_name = {e["name"]: e for e in events if e.get("ph") == "X"}

    wave0_compute = by_name["m0:fwd:wave0-expert-GroupedMatMul"]
    wave1_dispatch = by_name["m0:fwd:wave1-dispatch"]
    wave1_compute = by_name["m0:fwd:wave1-expert-GroupedMatMul"]

    wave0_compute_end = wave0_compute["ts"] + wave0_compute["dur"]
    wave1_dispatch_end = wave1_dispatch["ts"] + wave1_dispatch["dur"]

    assert wave1_dispatch["ts"] < wave0_compute_end
    assert wave1_dispatch_end > wave0_compute_end
    assert wave1_compute["ts"] == wave1_dispatch_end


def test_ep_wave_trace_excludes_shared_expert_compute():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=180.0,
                stream_id=0,
                start_us=0.0,
                end_us=180.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=180.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=20.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                scope="transformer.layers.0.ffn.moe",
                parallelism_tag="ep",
                comm_role="dispatch",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_gate_up",
                stream_id=0,
                stream_type="compute",
                start_us=20.0,
                end_us=60.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_down",
                stream_id=0,
                stream_type="compute",
                start_us=100.0,
                end_us=140.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="shared_expert_linear",
                stream_id=0,
                stream_type="compute",
                start_us=60.0,
                end_us=100.0,
                latency_us=40.0,
                op_type="linear",
                category="compute",
                phase="fwd",
                scope="transformer.layers.0.ffn.shared_experts.w2",
                component="shared_expert",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=140.0,
                end_us=160.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                scope="transformer.layers.0.ffn.moe",
                parallelism_tag="ep",
                comm_role="combine",
                ep_wave_k=4,
            ),
        ]
    )

    doc = ChromeTraceExporter(trace_ep_waves=True).export_stitched_detailed(
        stitched, [timeline]
    )
    events = json.loads(doc)["traceEvents"]

    ep_wave_events = [
        e for e in events
        if e.get("args", {}).get("parallelism") == "ep"
    ]
    assert ep_wave_events
    originals = {e["args"]["original_node"] for e in ep_wave_events}
    assert "shared_expert_linear" not in originals


def test_ep_wave_trace_delays_following_compute_after_synthesized_waves():
    stitched = PPStitchedTimeline(
        tasks=[
            GridTask(
                task_id="s0_m0_fwd",
                stage_id=0,
                mb_id=0,
                phase="fwd",
                latency_us=150.0,
                stream_id=0,
                start_us=0.0,
                end_us=150.0,
            )
        ],
        pp=1,
        M=1,
        step_time_us=150.0,
    )
    timeline = Timeline(
        scheduled_ops=[
            ScheduledOp(
                node_id="dispatch",
                stream_id=1,
                stream_type="comm",
                start_us=0.0,
                end_us=20.0,
                latency_us=20.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="dispatch",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_gate_up",
                stream_id=0,
                stream_type="compute",
                start_us=40.0,
                end_us=80.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="grouped_down",
                stream_id=0,
                stream_type="compute",
                start_us=80.0,
                end_us=120.0,
                latency_us=40.0,
                op_type="GroupedMatMul",
                category="compute",
                phase="fwd",
                component="routed_expert",
                scope="transformer.layers.0.ffn.experts",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="combine",
                stream_id=1,
                stream_type="comm",
                start_us=120.0,
                end_us=160.0,
                latency_us=40.0,
                op_type="comm.all_to_all",
                category="communication",
                phase="fwd",
                parallelism_tag="ep",
                comm_role="combine",
                scope="transformer.layers.0.ffn.moe",
                ep_wave_k=4,
            ),
            ScheduledOp(
                node_id="post_sum",
                stream_id=0,
                stream_type="compute",
                start_us=160.0,
                end_us=170.0,
                latency_us=10.0,
                op_type="aten.sum.dim_IntList",
                category="compute",
                phase="fwd",
            ),
        ]
    )

    doc = ChromeTraceExporter(trace_ep_waves=True).export_stitched_detailed(
        stitched, [timeline]
    )
    events = json.loads(doc)["traceEvents"]
    by_name = {e["name"]: e for e in events if e.get("ph") == "X"}

    post_sum = by_name["m0:fwd:aten.sum.dim_IntList"]
    wave_experts = [
        e for e in events
        if e.get("ph") == "X" and "wave" in e["name"] and e.get("cat") == "compute.ep.expert"
    ]
    last_expert_end = max(e["ts"] + e["dur"] for e in wave_experts)

    assert post_sum["ts"] >= last_expert_end
