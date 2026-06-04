from __future__ import annotations

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
        tp_coc=False,
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
        recompute_policy=None,
        gradient_checkpointing=False,
        total_params=123e9,
        hidden=4096,
        num_layers_full=32,
        quant=None,
        pp_schedule="dualpipev",
        vpp_chunks=2,
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
    assert onnx_by_name["llama3_8b_unified_graph.onnx"] is transformed_unified
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
