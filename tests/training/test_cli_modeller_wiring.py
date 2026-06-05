from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from python.zrt.ir.edge import Edge
from python.zrt.ir.graph import OpGraph
from python.zrt.ir.node import OpNode


def _check_torch_available():
    """Check if torch is available."""
    try:
        import torch
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _check_torch_available(), reason="torch not installed")
class _Report:
    def summary(self) -> str:
        return "graph-native report"

    def to_dict(self) -> dict:
        return {"summary": "graph-native report"}


class _FakeOutputPath:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.writes: list[tuple[Path, str]] = []

    def __truediv__(self, other: str):
        child = _FakeOutputPath(self.path / other)
        child.writes = self.writes
        return child

    def __eq__(self, other):
        return self.path == Path(other)

    def __str__(self) -> str:
        return str(self.path)

    def __fspath__(self) -> str:
        return str(self.path)

    def mkdir(self, *args, **kwargs) -> None:
        return None

    def write_text(self, text: str, *args, **kwargs) -> int:
        self.writes.append((self.path, text))
        return len(text)


def test_train_hw_cli_delegates_to_graph_native_modeller(monkeypatch, capsys, tmp_path):
    if not _check_torch_available():
        pytest.skip("torch not installed")
    
    from python.zrt import cli

    calls = []
    onnx_calls = []

    transformed_unified = OpGraph(name="transformed_unified", phase="train")
    transformed_unified.add_node(OpNode(
        id="comm_a2a_dispatch",
        op_type="comm.all_to_all",
        annotations={"phase": "fwd"},
    ))
    transformed_unified.add_node(OpNode(
        id="grouped_gate_up",
        op_type="GroupedMatMul",
        annotations={"phase": "fwd"},
    ))
    transformed_unified.add_node(OpNode(
        id="grouped_down_bwd",
        op_type="GroupedMatMul",
        annotations={"phase": "bwd"},
    ))
    transformed_unified.add_edge(Edge(
        src="comm_a2a_dispatch", src_idx=0,
        dst="grouped_gate_up", dst_idx=0,
    ))

    def fake_estimate_training_from_graphs(**kwargs):
        calls.append(kwargs)
        # cli.py unpacks (report, ctx, transformed); return matching 3-tuple
        return _Report(), None, {"unified": transformed_unified}

    def fake_export_transformed_graph_onnx(graph, output_path):
        onnx_calls.append({"graph": graph, "output_path": output_path})
        return output_path

    monkeypatch.setattr(
        "python.zrt.transform.analysis.estimate_training_from_graphs",
        fake_estimate_training_from_graphs,
    )
    monkeypatch.setattr(
        "python.zrt.transform.exporter.export_transformed_graph_onnx",
        fake_export_transformed_graph_onnx,
    )

    args = SimpleNamespace(
        hw="test_gpu",
        layers=4,
        batch_size=2,
        seq_len=128,
        tp=2,
        pp=3,
        ep=1,
        dp=4,
        cp=5,
        cp_kind="ring",
        zero_stage=2,
        optimizer="adamw",
        muon_rotation=True,
        muon_ns_steps=None,
        micro_batch=1,
        global_batch=16,
        dp_overlap=True,
        dp_ddp_buckets=True,
        dp_bucket_cap_mb=12.5,
        recompute_policy=None,
        gradient_checkpointing=False,
        total_params=123e9,
        hidden=4096,
        num_layers_full=32,
        quant=None,
        pp_schedule="dualpipev",
        vpp_chunks=2,
        tp_coc=False,
        pp_mode="formula",
        mega_moe=True,
        mega_moe_waves=4,
    )
    fwd_graph = SimpleNamespace(metadata={})
    bwd_graph = SimpleNamespace(metadata={})
    result = SimpleNamespace(
        graphs={
            "train_forward": fwd_graph,
            "train_backward": bwd_graph,
        },
        output_dir=tmp_path,
        phase_records={},
    )
    hw = object()

    cli._run_training_modelling(args, "hf_models/llama3_8b", hw, result)

    assert len(calls) == 1
    assert calls[0]["forward_graph"] is fwd_graph
    assert calls[0]["backward_graph"] is bwd_graph
    assert calls[0]["hw_spec"] is hw
    assert calls[0]["tp"] == 2
    assert calls[0]["pp"] == 3
    assert calls[0]["dp"] == 4
    assert calls[0]["cp"] == 5
    assert calls[0]["cp_kind"] == "ring"
    assert calls[0]["zero_stage"] == 2
    assert calls[0]["dp_overlap_in_bubble"] is True
    assert calls[0]["dp_bucket_mode"] == "ddp"
    assert calls[0]["dp_bucket_cap_mb"] == 12.5
    assert calls[0]["pp_schedule"] == "dualpipev"
    assert calls[0]["vpp_chunks"] == 2
    assert calls[0]["pp_mode"] == "formula"
    assert calls[0]["mega_moe"] is True
    assert calls[0]["mega_moe_waves"] == 4
    assert "graph-native report" in capsys.readouterr().out
    assert {call["output_path"].name for call in onnx_calls} == {
        "llama3_8b_train_forward_graph.onnx",
        "llama3_8b_train_backward_graph.onnx",
        "llama3_8b_unified_graph.onnx",
    }
    onnx_by_name = {call["output_path"].name: call["graph"] for call in onnx_calls}
    assert onnx_by_name["llama3_8b_unified_graph.onnx"].name == "transformed_unified"
    assert onnx_by_name["llama3_8b_train_forward_graph.onnx"].name == "transformed_train_forward"
    assert onnx_by_name["llama3_8b_train_backward_graph.onnx"].name == "transformed_train_backward"
    assert set(onnx_by_name["llama3_8b_unified_graph.onnx"].nodes) == set(transformed_unified.nodes)
    assert {n.op_type for n in onnx_by_name["llama3_8b_train_forward_graph.onnx"].nodes.values()} == {
        "comm.all_to_all",
        "GroupedMatMul",
    }
    assert {n.op_type for n in onnx_by_name["llama3_8b_train_backward_graph.onnx"].nodes.values()} == {
        "GroupedMatMul",
    }


def test_phase_subgraph_for_training_export_unknown_phase_returns_none():
    from python.zrt import cli

    graph = OpGraph(name="transformed_unified", phase="train")
    graph.add_node(OpNode(
        id="comm_a2a_dispatch",
        op_type="comm.all_to_all",
        annotations={"phase": "fwd"},
    ))

    assert cli._phase_subgraph_for_training_export(
        graph,
        phase="optimizer",
        name_suffix="optimizer",
    ) is None


def test_dp_ddp_bucket_cli_command_wires_end_to_end(monkeypatch, capsys):
    if not _check_torch_available():
        pytest.skip("torch not installed")

    from python.zrt import cli
    import python.zrt.hardware.registry as hw_registry
    import python.zrt.transform.exporter as exporter

    trace_calls = []
    estimate_calls = []

    fwd_graph = SimpleNamespace(metadata={})
    bwd_graph = SimpleNamespace(metadata={})
    fake_output_dir = _FakeOutputPath("output/dp_test_pp4_dp4")

    def fake_run_trace_phases(**kwargs):
        trace_calls.append(kwargs)
        return SimpleNamespace(
            graphs={
                "train_forward": fwd_graph,
                "train_backward": bwd_graph,
            },
            phase_records={
                "train_forward": [],
                "train_backward": [],
            },
            output_dir=fake_output_dir,
        )

    def fake_hw_load(name):
        assert name == "nvidia_h100_sxm"
        return SimpleNamespace(vendor="nvidia", device_type="gpu")

    def fake_estimate_training_from_graphs(**kwargs):
        estimate_calls.append(kwargs)
        return _Report(), None, {}

    monkeypatch.setattr(cli, "_run_trace_phases", fake_run_trace_phases)
    monkeypatch.setattr(hw_registry, "load", fake_hw_load)
    monkeypatch.setattr(
        "python.zrt.transform.analysis.estimate_training_from_graphs",
        fake_estimate_training_from_graphs,
    )
    monkeypatch.setattr(exporter, "export_training_graphs", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python.zrt",
            "--model-id", "hf_models/deepseek_v4",
            "--train",
            "--hw", "nvidia_h100_sxm",
            "--hidden", "7168",
            "--layers", "4",
            "--seq-len", "128",
            "--global-batch", "32",
            "--micro-batch", "8",
            "--dp", "4",
            "--pp", "4",
            "--tp", "1",
            "--pp-schedule", "1f1b",
            "--recompute-policy", "full",
            "--optimizer", "adam",
            "--zero-stage", "1",
            "--dp-ddp-buckets",
            "--dp-bucket-cap-mb", "25",
            "--output-dir", "output/dp_test_pp4_dp4",
        ],
    )

    cli.main()

    assert len(trace_calls) == 1
    assert trace_calls[0]["model_id"] == "hf_models/deepseek_v4"
    assert trace_calls[0]["num_layers"] == 4
    assert trace_calls[0]["seq_len"] == 128
    assert trace_calls[0]["phases"] == ("train_forward", "train_backward")
    assert trace_calls[0]["platform"] == "cuda"
    assert trace_calls[0]["output_dir"] == Path("output/dp_test_pp4_dp4")

    assert len(estimate_calls) == 1
    call = estimate_calls[0]
    assert call["forward_graph"] is fwd_graph
    assert call["backward_graph"] is bwd_graph
    assert call["output_dir"] == Path("output/dp_test_pp4_dp4")
    assert call["hidden"] == 7168
    assert call["num_layers"] == 4
    assert call["seq_len"] == 128
    assert call["global_batch"] == 32
    assert call["micro_batch"] == 8
    assert call["dp"] == 4
    assert call["pp"] == 4
    assert call["tp"] == 1
    assert call["pp_schedule"] == "1f1b"
    assert call["recompute_policy"] == "full"
    assert call["optimizer"] == "adam"
    assert call["zero_stage"] == 1
    assert call["dp_overlap_in_bubble"] is True
    assert call["dp_bucket_mode"] == "ddp"
    assert call["dp_bucket_cap_mb"] == 25
    assert "graph-native report" in capsys.readouterr().out


def test_train_cli_without_ddp_buckets_uses_original_dp_overlap(monkeypatch, capsys):
    if not _check_torch_available():
        pytest.skip("torch not installed")

    from python.zrt import cli
    import python.zrt.hardware.registry as hw_registry
    import python.zrt.transform.exporter as exporter

    estimate_calls = []
    fwd_graph = SimpleNamespace(metadata={})
    bwd_graph = SimpleNamespace(metadata={})

    monkeypatch.setattr(
        cli,
        "_run_trace_phases",
        lambda **kwargs: SimpleNamespace(
            graphs={
                "train_forward": fwd_graph,
                "train_backward": bwd_graph,
            },
            phase_records={},
            output_dir=_FakeOutputPath("output/pure_dp"),
        ),
    )
    monkeypatch.setattr(
        hw_registry,
        "load",
        lambda name: SimpleNamespace(vendor="nvidia", device_type="gpu"),
    )
    monkeypatch.setattr(
        "python.zrt.transform.analysis.estimate_training_from_graphs",
        lambda **kwargs: estimate_calls.append(kwargs) or (_Report(), None, {}),
    )
    monkeypatch.setattr(exporter, "export_training_graphs", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python.zrt",
            "--model-id", "hf_models/deepseek_v4",
            "--train",
            "--hw", "nvidia_h100_sxm",
            "--layers", "4",
            "--seq-len", "128",
            "--global-batch", "32",
            "--micro-batch", "8",
            "--dp", "4",
            "--pp", "4",
            "--tp", "1",
            "--output-dir", "output/pure_dp",
        ],
    )

    cli.main()

    assert len(estimate_calls) == 1
    call = estimate_calls[0]
    assert call["dp_overlap_in_bubble"] is True
    assert call["dp_bucket_mode"] == "layer"
    assert "graph-native report" in capsys.readouterr().out


def test_training_modelling_normalizes_optional_dp_cli_defaults(monkeypatch, capsys):
    if not _check_torch_available():
        pytest.skip("torch not installed")

    from python.zrt import cli

    calls = []

    monkeypatch.setattr(
        "python.zrt.transform.analysis.estimate_training_from_graphs",
        lambda **kwargs: calls.append(kwargs) or (_Report(), None, {}),
    )

    args = SimpleNamespace(
        hw="test_gpu",
        layers=4,
        batch_size=1,
        seq_len=128,
        tp=1,
        pp=1,
        ep=1,
        dp=1,
        cp=1,
        cp_kind="ulysses",
        zero_stage=0,
        optimizer="adam",
        muon_rotation=True,
        muon_ns_steps=None,
        micro_batch=1,
        global_batch=1,
        dp_overlap=None,
        dp_ddp_buckets=False,
        dp_bucket_cap_mb=None,
        recompute_policy=None,
        gradient_checkpointing=False,
        total_params=None,
        hidden=4096,
        num_layers_full=None,
        quant=None,
        pp_schedule="1f1b",
        vpp_chunks=1,
        tp_coc=False,
        pp_mode="trace",
        mega_moe=False,
        mega_moe_waves=0,
    )
    fwd_graph = SimpleNamespace(metadata={})
    bwd_graph = SimpleNamespace(metadata={})
    result = SimpleNamespace(
        graphs={"train_forward": fwd_graph, "train_backward": bwd_graph},
        phase_records={},
        output_dir=None,
    )

    cli._run_training_modelling(args, "hf_models/llama3_8b", object(), result)

    assert len(calls) == 1
    assert calls[0]["dp_overlap_in_bubble"] is True
    assert calls[0]["dp_bucket_cap_mb"] == 25.0
    assert "graph-native report" in capsys.readouterr().out


def test_train_cli_no_dp_overlap_uses_pure_dp(monkeypatch, capsys):
    if not _check_torch_available():
        pytest.skip("torch not installed")

    from python.zrt import cli
    import python.zrt.hardware.registry as hw_registry
    import python.zrt.transform.exporter as exporter

    estimate_calls = []
    fwd_graph = SimpleNamespace(metadata={})
    bwd_graph = SimpleNamespace(metadata={})

    monkeypatch.setattr(
        cli,
        "_run_trace_phases",
        lambda **kwargs: SimpleNamespace(
            graphs={
                "train_forward": fwd_graph,
                "train_backward": bwd_graph,
            },
            phase_records={},
            output_dir=_FakeOutputPath("output/pure_dp"),
        ),
    )
    monkeypatch.setattr(
        hw_registry,
        "load",
        lambda name: SimpleNamespace(vendor="nvidia", device_type="gpu"),
    )
    monkeypatch.setattr(
        "python.zrt.transform.analysis.estimate_training_from_graphs",
        lambda **kwargs: estimate_calls.append(kwargs) or (_Report(), None, {}),
    )
    monkeypatch.setattr(exporter, "export_training_graphs", lambda **kwargs: None)
    monkeypatch.setattr(
        "sys.argv",
        [
            "python.zrt",
            "--model-id", "hf_models/deepseek_v4",
            "--train",
            "--hw", "nvidia_h100_sxm",
            "--layers", "4",
            "--seq-len", "128",
            "--global-batch", "32",
            "--micro-batch", "8",
            "--dp", "4",
            "--pp", "4",
            "--tp", "1",
            "--no-dp-overlap",
            "--output-dir", "output/pure_dp",
        ],
    )

    cli.main()

    assert len(estimate_calls) == 1
    call = estimate_calls[0]
    assert call["dp_overlap_in_bubble"] is False
    assert call["dp_bucket_mode"] == "layer"
    assert "graph-native report" in capsys.readouterr().out


def test_train_cli_rejects_ddp_buckets_when_dp_overlap_disabled(monkeypatch):
    if not _check_torch_available():
        pytest.skip("torch not installed")

    from python.zrt import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "python.zrt",
            "--model-id", "hf_models/deepseek_v4",
            "--train",
            "--no-dp-overlap",
            "--dp-ddp-buckets",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2


def test_pp_mode_cli_argument_is_not_supported(monkeypatch):
    if not _check_torch_available():
        pytest.skip("torch not installed")

    from python.zrt import cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "python.zrt",
            "--model-id", "hf_models/deepseek_v4",
            "--train",
            "--pp-mode", "trace",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 2
