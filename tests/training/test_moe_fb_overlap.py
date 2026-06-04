from __future__ import annotations

import pytest

from zrt.hardware.spec import InterconnectSpec, LinkSpec
from zrt.training.compose.schedules import pipeline_step_time
from zrt.training.compose.stage import stage_time
from zrt.training.io.config_loader import _parse_strategy
from zrt.training.ir.builders import build_graph
from zrt.training.spec.model import LayerKind, ModelSpec
from zrt.training.spec.report import TrainingReport
from zrt.training.spec.strategy import PPSched, Strategy
from zrt.training.spec.system import GPU, SystemSpec
from zrt.transform.analysis.training import (
    _apply_moe_fb_disabled_ep_accounting,
    _graph_moe_fb_overlap_us,
)


def _system(*, ep_overlap_waves: int = 4) -> SystemSpec:
    return SystemSpec(
        gpu=GPU(
            name="h100",
            flops_bf16=989,
            flops_fp8=1979,
            hbm_gb=80,
            hbm_bw_gbps=3350,
            ep_overlap_waves=ep_overlap_waves,
        ),
        host_mem_gb=256,
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(
                type="NVLink",
                bandwidth_gbps=900,
                latency_us=1.0,
                topology="all_to_all",
                num_devices=8,
            ),
            inter_node=LinkSpec(
                type="IB",
                bandwidth_gbps=400,
                latency_us=2.0,
                topology="fat_tree",
            ),
        ),
        nodes=1,
        gpus_per_node=8,
    )


def _moe_model(**kwargs) -> ModelSpec:
    base = dict(
        hidden=4096,
        ffn=8192,
        num_heads=32,
        num_kv_heads=32,
        head_dim=128,
        vocab=32000,
        seq_len=512,
        layers=[LayerKind.MOE],
        num_experts=16,
        moe_ffn=8192,
        top_k=2,
        n_shared_experts=0,
    )
    base.update(kwargs)
    return ModelSpec(**base)


def test_moe_fb_overlap_defaults_off_and_can_be_enabled():
    assert Strategy().moe_fb_overlap is False
    assert _parse_strategy({}).moe_fb_overlap is False
    assert _parse_strategy({"moe_fb_overlap": True}).moe_fb_overlap is True
    assert _parse_strategy({"moe_fb_overlap": False}).moe_fb_overlap is False


def test_training_report_exports_moe_fb_overlap_breakdown():
    report = TrainingReport(
        ep_fb_total_ms=7.0,
        ep_fb_hidden_ms=2.0,
        ep_fb_exposed_ms=5.0,
        ep_fb_steady_hidden_ms=1.5,
        ep_fb_boundary_hidden_ms=0.5,
        mega_moe_hidden_ms=3.0,
    )

    data = report.to_dict()

    assert data["ep_fb_total_ms"] == pytest.approx(7.0)
    assert data["ep_fb_hidden_ms"] == pytest.approx(2.0)
    assert data["ep_fb_exposed_ms"] == pytest.approx(5.0)
    assert data["ep_fb_steady_hidden_ms"] == pytest.approx(1.5)
    assert data["ep_fb_boundary_hidden_ms"] == pytest.approx(0.5)
    assert data["mega_moe_hidden_ms"] == pytest.approx(3.0)


def test_normal_ep_does_not_consume_ep_overlap_waves():
    model = _moe_model(seq_len=1024)
    strategy = Strategy(
        ep=4,
        dp=4,
        ep_overlap=True,
        moe_fb_overlap=False,
        mega_moe=False,
        micro_batch=2,
    )
    graph = build_graph(model, strategy)

    no_waves = stage_time(graph.ops, graph.collectives, model, _system(ep_overlap_waves=0), strategy)
    four_waves = stage_time(graph.ops, graph.collectives, model, _system(ep_overlap_waves=4), strategy)

    assert no_waves.ep_hidden == pytest.approx(0.0)
    assert four_waves.ep_hidden == pytest.approx(0.0)
    assert four_waves.ep_exposed == pytest.approx(no_waves.ep_exposed)


def test_moe_fb_overlap_hides_normal_ep_at_pipeline_level():
    model = _moe_model(seq_len=1024)
    system = _system(ep_overlap_waves=0)
    base = dict(
        pp=2,
        ep=4,
        dp=4,
        micro_batch=2,
        global_batch=64,
        pp_schedule=PPSched.ONE_F_ONE_B,
        ep_overlap=False,
        mega_moe=False,
    )
    disabled = Strategy(**base, moe_fb_overlap=False)
    enabled = Strategy(**base, moe_fb_overlap=True)
    graph_disabled = build_graph(model, disabled)
    graph_enabled = build_graph(model, enabled)

    step_disabled = pipeline_step_time(graph_disabled, model, system, disabled)
    step_enabled = pipeline_step_time(graph_enabled, model, system, enabled)

    assert step_disabled.ep_hidden == pytest.approx(0.0)
    assert step_disabled.ep_fb_hidden == pytest.approx(0.0)
    assert step_enabled.ep_fb_total > 0.0
    assert step_enabled.ep_fb_hidden > 0.0
    assert step_enabled.ep_hidden == pytest.approx(step_enabled.ep_fb_hidden)
    assert step_enabled.ep_fb_exposed == pytest.approx(step_enabled.ep_exposed)
    assert step_enabled.ep_fb_total == pytest.approx(
        step_enabled.ep_fb_exposed + step_enabled.ep_fb_hidden
    )
    assert step_enabled.ep_exposed < step_disabled.ep_exposed


def test_graph_path_moe_fb_overlap_uses_pipeline_compute_window():
    strategy = Strategy(
        pp=2,
        ep=8,
        dp=1,
        micro_batch=1,
        global_batch=32,
        pp_schedule=PPSched.ONE_F_ONE_B,
        moe_fb_overlap=True,
    )

    hidden, exposed = _graph_moe_fb_overlap_us(
        ep_total_us=260.0,
        fwd_compute_us=80_000.0,
        bwd_compute_us=120_000.0,
        strategy=strategy,
    )

    assert hidden == pytest.approx(260.0)
    assert exposed == pytest.approx(0.0)


def test_graph_path_moe_fb_overlap_can_be_disabled():
    strategy = Strategy(
        pp=2,
        ep=8,
        dp=1,
        micro_batch=1,
        global_batch=32,
        pp_schedule=PPSched.ONE_F_ONE_B,
        moe_fb_overlap=False,
    )

    hidden, exposed = _graph_moe_fb_overlap_us(
        ep_total_us=260.0,
        fwd_compute_us=80_000.0,
        bwd_compute_us=120_000.0,
        strategy=strategy,
    )

    assert hidden == pytest.approx(0.0)
    assert exposed == pytest.approx(260.0)


def test_training_context_defaults_moe_fb_overlap_off():
    from zrt.training.ir.context_builder import build_context

    model = _moe_model()
    system = _system()
    strategy = Strategy(ep=4, dp=4)

    ctx = build_context(model, system, strategy)

    assert ctx.training.moe_fb_overlap is False


def test_graph_path_disabling_moe_fb_overlap_preserves_existing_ep_hidden():
    from zrt.executor.overlap import PerStrategyOverlapReport

    per_strat = PerStrategyOverlapReport(
        ep_total_us=100.0,
        ep_exposed_us=70.0,
        ep_hidden_us=30.0,
    )

    fb_total, fb_hidden, fb_exposed, fb_steady, fb_boundary = (
        _apply_moe_fb_disabled_ep_accounting(per_strat)
    )

    assert per_strat.ep_hidden_us == pytest.approx(30.0)
    assert per_strat.ep_exposed_us == pytest.approx(70.0)
    assert fb_total == pytest.approx(100.0)
    assert fb_hidden == pytest.approx(0.0)
    assert fb_exposed == pytest.approx(100.0)
    assert fb_steady == pytest.approx(0.0)
    assert fb_boundary == pytest.approx(0.0)
