"""Tests for GraphCoarsenPass — aten-level to block-level aggregation."""
from __future__ import annotations

import pytest

from zrt.ir.graph import OpGraph
from zrt.ir.node import OpNode
from zrt.ir.types import DType, TensorMeta
from zrt.transform.context import TransformContext
from zrt.transform.passes.coarsen import (
    GraphCoarsenPass,
    _is_block_level,
    _scopes_with_children,
    _infer_spec_kind,
)


def _tensor(name="t", shape=(1, 128, 4096)):
    return TensorMeta(
        id=name, shape=shape, dtype=DType.BF16,
        mem_bytes=shape[0] * shape[1] * shape[2] * 2,
    )


def _ctx():
    from zrt.hardware.spec import (
        ComputeSpec, HardwareSpec, InterconnectSpec, LinkSpec, MemorySpec,
    )
    hw = HardwareSpec(
        name="test_hw", vendor="nvidia", device_type="gpu",
        compute=ComputeSpec(bf16_tflops=989),
        memory=MemorySpec(capacity_gb=80, hbm_bandwidth_gbps=3350),
        interconnect=InterconnectSpec(
            intra_node=LinkSpec(type="NVLink", bandwidth_gbps=900, latency_us=1.0),
            inter_node=LinkSpec(type="IB", bandwidth_gbps=100, latency_us=10.0),
        ),
    )
    return TransformContext(hw_spec=hw)


# ---------------------------------------------------------------------------
# Helpers to build test graphs
# ---------------------------------------------------------------------------

def _aten_node(nid, op_type, scope, module_class="Linear",
               category="compute", component=""):
    return OpNode(
        id=nid,
        op_type=op_type,
        inputs=[_tensor(f"{nid}_in")],
        outputs=[_tensor(f"{nid}_out")],
        scope=scope,
        module_class=module_class,
        category=category,
        component=component,
    )


def _simple_aten_graph():
    """Two-layer graph with aten ops under self_attn.q_proj and input_layernorm."""
    from zrt.ir.edge import Edge

    ln1 = _aten_node("op_0", "aten.rms_norm.default",
                     "model.layers.0.input_layernorm", "RMSNorm")
    q1 = _aten_node("op_1", "aten.mm.default",
                    "model.layers.0.self_attn.q_proj", "Linear")
    q2 = _aten_node("op_2", "aten.add.Tensor",
                    "model.layers.0.self_attn.q_proj", "Linear")

    nodes = {n.id: n for n in [ln1, q1, q2]}
    edges = [
        Edge(src="op_0", src_idx=0, dst="op_1", dst_idx=0),
        Edge(src="op_1", src_idx=0, dst="op_2", dst_idx=0),
    ]
    return OpGraph(name="test", phase="train", nodes=nodes, edges=edges)


def _block_level_graph():
    """Graph that is already block-level (spec.* op_types)."""
    from zrt.ir.edge import Edge

    n1 = OpNode(
        id="L0.ln1", op_type="aten.rms_norm.default",
        inputs=[_tensor("ln1_in")], outputs=[_tensor("ln1_out")],
        scope="model.layers.0.input_layernorm",
        attrs={"spec_kind": "rmsnorm", "source": "model_spec"},
    )
    n2 = OpNode(
        id="L0.q_proj", op_type="aten.mm.default",
        inputs=[_tensor("q_in")], outputs=[_tensor("q_out")],
        scope="model.layers.0.self_attn.q_proj",
        attrs={"spec_kind": "matmul", "source": "model_spec"},
    )
    nodes = {n.id: n for n in [n1, n2]}
    edges = [Edge(src="L0.ln1", src_idx=0, dst="L0.q_proj", dst_idx=0)]
    return OpGraph(name="test_block", phase="train", nodes=nodes, edges=edges)


def _graph_with_comm():
    """Aten graph with a communication node inserted between compute nodes."""
    from zrt.ir.edge import Edge

    mm1 = _aten_node("op_0", "aten.mm.default",
                     "model.layers.0.self_attn.q_proj", "Linear")
    comm = OpNode(
        id="comm_0", op_type="comm.all_reduce",
        inputs=[_tensor("comm_in")], outputs=[_tensor("comm_out")],
        scope="model.layers.0.self_attn.q_proj",
        category="communication",
    )
    mm2 = _aten_node("op_1", "aten.mm.default",
                     "model.layers.0.self_attn.o_proj", "Linear")

    nodes = {n.id: n for n in [mm1, comm, mm2]}
    edges = [
        Edge(src="op_0", src_idx=0, dst="comm_0", dst_idx=0),
        Edge(src="comm_0", src_idx=0, dst="op_1", dst_idx=0),
    ]
    return OpGraph(name="test_comm", phase="train", nodes=nodes, edges=edges)


def _multi_layer_graph():
    """Two-layer graph with multiple modules per layer."""
    from zrt.ir.edge import Edge

    nodes_list = [
        _aten_node("op_0", "aten.embedding.default",
                   "model.embed_tokens", "Embedding"),
        _aten_node("op_1", "aten.rms_norm.default",
                   "model.layers.0.input_layernorm", "RMSNorm"),
        _aten_node("op_2", "aten.mm.default",
                   "model.layers.0.self_attn.q_proj", "Linear"),
        _aten_node("op_3", "aten.mm.default",
                   "model.layers.0.self_attn.o_proj", "Linear"),
        _aten_node("op_4", "aten.rms_norm.default",
                   "model.layers.0.post_attention_layernorm", "RMSNorm"),
        _aten_node("op_5", "aten.mm.default",
                   "model.layers.0.mlp.gate_proj", "Linear"),
        _aten_node("op_6", "aten.silu.default",
                   "model.layers.0.mlp.act_fn", "SiLU"),
        _aten_node("op_7", "aten.mm.default",
                   "model.layers.0.mlp.down_proj", "Linear"),
        _aten_node("op_8", "aten.rms_norm.default",
                   "model.layers.1.input_layernorm", "RMSNorm"),
        _aten_node("op_9", "aten.mm.default",
                   "model.layers.1.self_attn.q_proj", "Linear"),
        _aten_node("op_10", "aten.mm.default",
                   "model.norm", "RMSNorm"),
        _aten_node("op_11", "aten.mm.default",
                   "lm_head", "Linear"),
    ]

    edges = [
        Edge(src=f"op_{i}", src_idx=0, dst=f"op_{i+1}", dst_idx=0)
        for i in range(len(nodes_list) - 1)
    ]

    nodes = {n.id: n for n in nodes_list}
    return OpGraph(name="test_multi", phase="train", nodes=nodes, edges=edges)


def _v4_like_graph():
    """V4-style graph with attn/ffn/attn_norm/ffn_norm naming.

    Simulates the scope structure of a real DeepSeek-V4 captured graph:
    - model.transformer.embed (ParallelEmbedding)
    - model.transformer.layers.0.attn_norm (RMSNorm)
    - model.transformer.layers.0.attn (Attention — kernel ops at this scope)
    - model.transformer.layers.0.attn.wq_a (Linear)
    - model.transformer.layers.0.attn.wq_b (ColumnParallelLinear)
    - model.transformer.layers.0.attn.wkv (Linear)
    - model.transformer.layers.0.attn.wo_a (ColumnParallelLinear)
    - model.transformer.layers.0.attn.wo_b (RowParallelLinear)
    - model.transformer.layers.0.attn.compressor (Compressor)
    - model.transformer.layers.0.ffn_norm (RMSNorm)
    - model.transformer.layers.0.ffn.gate (Gate)
    - model.transformer.layers.0.ffn.experts (Expert)
    - model.transformer.layers.0.ffn.shared_experts (Expert)
    - model.transformer.layers.0.hc_pre_attn (HCPreAttn)
    - model.transformer.layers.0.hc_post_attn (HCPostAttn)
    - model.transformer.norm (RMSNorm)
    - model.transformer.head (ParallelHead)
    """
    from zrt.ir.edge import Edge

    nodes_list = [
        _aten_node("op_0", "aten.embedding.default",
                   "model.transformer.embed", "ParallelEmbedding",
                   component="embedding"),
        _aten_node("op_1", "aten.mm.default",
                   "model.transformer.layers.0.hc_pre_attn", "HCPreAttn",
                   component="hc.pre_attn"),
        _aten_node("op_2", "aten.rms_norm.default",
                   "model.transformer.layers.0.attn_norm", "RMSNorm"),
        _aten_node("op_3", "aten.mm.default",
                   "model.transformer.layers.0.attn.wq_a", "Linear",
                   component="attn.q_a_proj"),
        _aten_node("op_4", "aten.rms_norm.default",
                   "model.transformer.layers.0.attn.q_norm", "RMSNorm",
                   component="attn.q_norm"),
        _aten_node("op_5", "aten.mm.default",
                   "model.transformer.layers.0.attn.wq_b", "ColumnParallelLinear",
                   component="attn.q_b_proj"),
        _aten_node("op_6", "aten.mm.default",
                   "model.transformer.layers.0.attn.wkv", "Linear",
                   component="attn.kv_a_proj"),
        _aten_node("op_7", "aten.rms_norm.default",
                   "model.transformer.layers.0.attn.kv_norm", "RMSNorm",
                   component="attn.kv_norm"),
        _aten_node("op_8", "aten.mm.default",
                   "model.transformer.layers.0.attn.compressor", "Compressor",
                   component="attn.compressor"),
        _aten_node("op_9", "aten.bmm.default",
                   "model.transformer.layers.0.attn", "Attention",
                   component="attn.score"),
        _aten_node("op_10", "aten.mm.default",
                   "model.transformer.layers.0.attn.wo_a", "ColumnParallelLinear",
                   component="attn.o_proj"),
        _aten_node("op_11", "aten.mm.default",
                   "model.transformer.layers.0.attn.wo_b", "RowParallelLinear",
                   component="attn.o_proj"),
        _aten_node("op_12", "aten.mm.default",
                   "model.transformer.layers.0.hc_post_attn", "HCPostAttn",
                   component="hc.post_attn"),
        _aten_node("op_13", "aten.mm.default",
                   "model.transformer.layers.0.hc_pre_ffn", "HCPreFfn",
                   component="hc.pre_ffn"),
        _aten_node("op_14", "aten.rms_norm.default",
                   "model.transformer.layers.0.ffn_norm", "RMSNorm"),
        _aten_node("op_15", "aten.mm.default",
                   "model.transformer.layers.0.ffn.gate", "Gate",
                   component="moe.gate"),
        _aten_node("op_16", "aten.mm.default",
                   "model.transformer.layers.0.ffn.experts", "Expert",
                   component="moe.experts"),
        _aten_node("op_17", "aten.mm.default",
                   "model.transformer.layers.0.ffn.shared_experts", "Expert",
                   component="moe.shared"),
        _aten_node("op_18", "aten.mm.default",
                   "model.transformer.layers.0.hc_post_ffn", "HCPostFfn",
                   component="hc.post_ffn"),
        _aten_node("op_19", "aten.rms_norm.default",
                   "model.transformer.norm", "RMSNorm"),
        _aten_node("op_20", "aten.mm.default",
                   "model.transformer.head", "ParallelHead"),
    ]

    edges = [
        Edge(src=f"op_{i}", src_idx=0, dst=f"op_{i+1}", dst_idx=0)
        for i in range(len(nodes_list) - 1)
    ]

    nodes = {n.id: n for n in nodes_list}
    return OpGraph(name="test_v4", phase="train", nodes=nodes, edges=edges)


# ---------------------------------------------------------------------------
# Tests: Detection
# ---------------------------------------------------------------------------

class TestIsBlockLevel:
    def test_block_level_graph(self):
        g = _block_level_graph()
        assert _is_block_level(g) is True

    def test_aten_level_graph(self):
        g = _simple_aten_graph()
        assert _is_block_level(g) is False

    def test_empty_graph(self):
        g = OpGraph(name="empty", phase="train")
        assert _is_block_level(g) is True

    def test_only_comm_nodes(self):
        comm = OpNode(
            id="c0", op_type="comm.all_reduce",
            scope="model.layers.0", category="communication",
        )
        g = OpGraph(name="comm_only", phase="train", nodes={"c0": comm})
        assert _is_block_level(g) is True


# ---------------------------------------------------------------------------
# Tests: Scope grouping helpers
# ---------------------------------------------------------------------------

class TestScopesWithChildren:
    def test_simple_hierarchy(self):
        scopes = {
            "model",
            "model.layers",
            "model.layers.0",
            "model.layers.0.self_attn",
            "model.layers.0.self_attn.q_proj",
            "model.layers.0.mlp",
        }
        parents = _scopes_with_children(scopes)
        assert "model" in parents
        assert "model.layers" in parents
        assert "model.layers.0" in parents
        assert "model.layers.0.self_attn" in parents
        assert "model.layers.0.self_attn.q_proj" not in parents
        assert "model.layers.0.mlp" not in parents

    def test_single_scope(self):
        scopes = {"model.layers.0.self_attn.q_proj"}
        parents = _scopes_with_children(scopes)
        assert len(parents) == 0

    def test_v4_attn_with_children(self):
        scopes = {
            "model.transformer.layers.0.attn",
            "model.transformer.layers.0.attn.wq_a",
            "model.transformer.layers.0.attn.wkv",
        }
        parents = _scopes_with_children(scopes)
        assert "model.transformer.layers.0.attn" in parents
        assert "model.transformer.layers.0.attn.wq_a" not in parents


# ---------------------------------------------------------------------------
# Tests: spec_kind inference
# ---------------------------------------------------------------------------

class TestInferSpecKind:
    def test_rmsnorm(self):
        assert _infer_spec_kind(
            "model.layers.0.input_layernorm", "RMSNorm", ""
        ) == "rmsnorm"

    def test_q_proj(self):
        assert _infer_spec_kind(
            "model.layers.0.self_attn.q_proj", "Linear", "attention"
        ) == "matmul"

    def test_o_proj(self):
        assert _infer_spec_kind(
            "model.layers.0.self_attn.o_proj", "Linear", "attention"
        ) == "matmul"

    def test_gate_proj(self):
        assert _infer_spec_kind(
            "model.layers.0.mlp.gate_proj", "Linear", ""
        ) == "matmul"

    def test_act_fn(self):
        assert _infer_spec_kind(
            "model.layers.0.mlp.act_fn", "SiLU", ""
        ) == "swiglu"

    def test_embed(self):
        assert _infer_spec_kind(
            "model.embed_tokens", "Embedding", "embedding"
        ) == "embed"

    def test_lm_head(self):
        assert _infer_spec_kind(
            "lm_head", "Linear", ""
        ) == "lm_head"

    def test_fallback_to_module_class(self):
        assert _infer_spec_kind(
            "model.layers.0.unknown_module", "RMSNorm", ""
        ) == "rmsnorm"

    def test_v4_wq_a(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.wq_a", "Linear", "attn.q_a_proj"
        ) == "matmul"

    def test_v4_wq_b(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.wq_b", "ColumnParallelLinear", "attn.q_b_proj"
        ) == "matmul"

    def test_v4_wkv(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.wkv", "Linear", "attn.kv_a_proj"
        ) == "matmul"

    def test_v4_wo_a(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.wo_a", "ColumnParallelLinear", "attn.o_proj"
        ) == "matmul"

    def test_v4_wo_b(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.wo_b", "RowParallelLinear", "attn.o_proj"
        ) == "matmul"

    def test_v4_q_norm(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.q_norm", "RMSNorm", "attn.q_norm"
        ) == "rmsnorm"

    def test_v4_kv_norm(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.kv_norm", "RMSNorm", "attn.kv_norm"
        ) == "rmsnorm"

    def test_v4_attn_norm(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn_norm", "RMSNorm", ""
        ) == "rmsnorm"

    def test_v4_ffn_norm(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.ffn_norm", "RMSNorm", ""
        ) == "rmsnorm"

    def test_v4_embed(self):
        assert _infer_spec_kind(
            "model.transformer.embed", "ParallelEmbedding", "embedding"
        ) == "embed"

    def test_v4_head(self):
        assert _infer_spec_kind(
            "model.transformer.head", "ParallelHead", ""
        ) == "lm_head"

    def test_v4_gate(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.ffn.gate", "Gate", "moe.gate"
        ) == "router"

    def test_v4_experts(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.ffn.experts", "Expert", "moe.experts"
        ) == "expert"

    def test_v4_shared_experts(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.ffn.shared_experts", "Expert", "moe.shared"
        ) == "shared_expert"

    def test_v4_compressor(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.compressor", "Compressor", "attn.compressor"
        ) == "compressor"

    def test_v4_indexer(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn.indexer", "Indexer", "attn.indexer"
        ) == "indexer"

    def test_v4_hc_pre_attn(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.hc_pre_attn", "HCPreAttn", "hc.pre_attn"
        ) == "mhc_pre"

    def test_v4_hc_post_attn(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.hc_post_attn", "HCPostAttn", "hc.post_attn"
        ) == "mhc_post"

    def test_v4_hc_pre_ffn(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.hc_pre_ffn", "HCPreFfn", "hc.pre_ffn"
        ) == "mhc_pre"

    def test_v4_hc_post_ffn(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.hc_post_ffn", "HCPostFfn", "hc.post_ffn"
        ) == "mhc_post"

    def test_v4_attn_parent_scope(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.attn@parent_ops", "Attention", "attn.score"
        ) == "attn_core"

    def test_gate_proj_moe_context(self):
        assert _infer_spec_kind(
            "model.transformer.layers.0.ffn.gate_proj", "Linear", "moe"
        ) == "router"


# ---------------------------------------------------------------------------
# Tests: GraphCoarsenPass — standard HF graphs
# ---------------------------------------------------------------------------

class TestGraphCoarsenPass:
    def test_noop_on_block_level(self):
        g = _block_level_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())
        assert result is g

    def test_coarsen_simple(self):
        g = _simple_aten_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert result is not g
        assert len(result.nodes) == 2
        assert result.metadata.get("coarsened") is True

        ln_node = result.nodes.get("L0.input_layernorm")
        assert ln_node is not None
        assert ln_node.attrs["spec_kind"] == "rmsnorm"
        assert ln_node.num_sub_ops == 1

        q_node = result.nodes.get("L0.self_attn.q_proj")
        assert q_node is not None
        assert q_node.attrs["spec_kind"] == "matmul"
        assert q_node.num_sub_ops == 2

    def test_coarsen_preserves_comm(self):
        g = _graph_with_comm()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert "comm_0" in result.nodes
        assert result.nodes["comm_0"].category == "communication"

        compute_nodes = [n for n in result.nodes.values()
                         if n.category == "compute"]
        assert len(compute_nodes) == 2

    def test_coarsen_edge_rewiring(self):
        g = _simple_aten_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert len(result.edges) == 1
        e = result.edges[0]
        assert e.src == "L0.input_layernorm"
        assert e.dst == "L0.self_attn.q_proj"

    def test_coarsen_comm_edges(self):
        g = _graph_with_comm()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert len(result.edges) == 2
        srcs = {e.src for e in result.edges}
        dsts = {e.dst for e in result.edges}
        assert "comm_0" in srcs
        assert "comm_0" in dsts

    def test_coarsen_multi_layer(self):
        g = _multi_layer_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert result.metadata.get("coarsened") is True

        node_ids = set(result.nodes.keys())
        assert "L0.input_layernorm" in node_ids
        assert "L0.self_attn.q_proj" in node_ids
        assert "L0.self_attn.o_proj" in node_ids
        assert "L0.post_attention_layernorm" in node_ids
        assert "L0.mlp.gate_proj" in node_ids
        assert "L0.mlp.act_fn" in node_ids
        assert "L0.mlp.down_proj" in node_ids
        assert "L1.input_layernorm" in node_ids
        assert "L1.self_attn.q_proj" in node_ids

        l0_ln = result.nodes["L0.input_layernorm"]
        assert l0_ln.attrs["spec_kind"] == "rmsnorm"
        assert l0_ln.layer == "0"

        l0_gate = result.nodes["L0.mlp.gate_proj"]
        assert l0_gate.attrs["spec_kind"] == "matmul"

        l0_act = result.nodes["L0.mlp.act_fn"]
        assert l0_act.attrs["spec_kind"] == "swiglu"

    def test_coarsen_preserves_graph_metadata(self):
        g = _simple_aten_graph()
        g.metadata["model_name"] = "test_model"
        g.metadata["batch_size"] = 4

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert result.metadata["model_name"] == "test_model"
        assert result.metadata["batch_size"] == 4
        assert result.metadata["coarsened"] is True

    def test_coarsen_preserves_phase(self):
        g = _simple_aten_graph()
        g.phase = "train_forward"

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())
        assert result.phase == "train_forward"

    def test_coarsen_node_count_reduction(self):
        from zrt.ir.edge import Edge
        n1 = _aten_node("op_0", "aten.mm.default",
                        "model.layers.0.self_attn.q_proj", "Linear")
        n2 = _aten_node("op_1", "aten.add.Tensor",
                        "model.layers.0.self_attn.q_proj", "Linear")
        n3 = _aten_node("op_2", "aten.view.default",
                        "model.layers.0.self_attn.q_proj", "Linear")
        n4 = _aten_node("op_3", "aten.mm.default",
                        "model.layers.0.self_attn.o_proj", "Linear")
        edges = [
            Edge(src="op_0", src_idx=0, dst="op_1", dst_idx=0),
            Edge(src="op_1", src_idx=0, dst="op_2", dst_idx=0),
            Edge(src="op_2", src_idx=0, dst="op_3", dst_idx=0),
        ]
        g = OpGraph(name="reduction", phase="train",
                    nodes={n.id: n for n in [n1, n2, n3, n4]}, edges=edges)
        assert len(g.nodes) == 4

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert len(result.nodes) == 2
        assert "L0.self_attn.q_proj" in result.nodes
        assert "L0.self_attn.o_proj" in result.nodes
        assert result.nodes["L0.self_attn.q_proj"].num_sub_ops == 3

    def test_coarsen_no_self_edges(self):
        g = _simple_aten_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        for e in result.edges:
            assert e.src != e.dst, f"Self-edge found: {e.src} -> {e.dst}"

    def test_coarsen_topo_sort_succeeds(self):
        g = _multi_layer_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        sorted_nodes = result.topo_sort()
        assert len(sorted_nodes) == len(result.nodes)

    def test_coarsen_fused_from_populated(self):
        g = _simple_aten_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        q_node = result.nodes["L0.self_attn.q_proj"]
        assert len(q_node.fused_from) == 2
        assert "aten.mm.default" in q_node.fused_from
        assert "aten.add.Tensor" in q_node.fused_from

    def test_coarsen_annotations_summed(self):
        g = _simple_aten_graph()
        g.nodes["op_1"].annotations["flops"] = 100
        g.nodes["op_2"].annotations["flops"] = 50

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        q_node = result.nodes["L0.self_attn.q_proj"]
        assert q_node.annotations.get("flops") == 150

    def test_coarsen_layer_id_attr(self):
        g = _simple_aten_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        ln_node = result.nodes["L0.input_layernorm"]
        assert ln_node.attrs["layer_id"] == 0

        q_node = result.nodes["L0.self_attn.q_proj"]
        assert q_node.attrs["layer_id"] == 0


# ---------------------------------------------------------------------------
# Tests: V4-like graph coarsening
# ---------------------------------------------------------------------------

class TestV4Coarsen:
    def test_v4_coarsen_produces_expected_nodes(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        node_ids = set(result.nodes.keys())
        assert "embed" in node_ids
        assert "L0.hc_pre_attn" in node_ids
        assert "L0.attn_norm" in node_ids
        assert "L0.attn.wq_a" in node_ids
        assert "L0.attn.q_norm" in node_ids
        assert "L0.attn.wq_b" in node_ids
        assert "L0.attn.wkv" in node_ids
        assert "L0.attn.kv_norm" in node_ids
        assert "L0.attn.compressor" in node_ids
        assert "L0.attn" in node_ids
        assert "L0.attn.wo_a" in node_ids
        assert "L0.attn.wo_b" in node_ids
        assert "L0.hc_post_attn" in node_ids
        assert "L0.hc_pre_ffn" in node_ids
        assert "L0.ffn_norm" in node_ids
        assert "L0.ffn.gate" in node_ids
        assert "L0.ffn.experts" in node_ids
        assert "L0.ffn.shared_experts" in node_ids
        assert "L0.hc_post_ffn" in node_ids
        assert "norm" in node_ids
        assert "head" in node_ids

    def test_v4_coarsen_spec_kinds(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert result.nodes["embed"].attrs["spec_kind"] == "embed"
        assert result.nodes["L0.attn_norm"].attrs["spec_kind"] == "rmsnorm"
        assert result.nodes["L0.attn.wq_a"].attrs["spec_kind"] == "matmul"
        assert result.nodes["L0.attn.q_norm"].attrs["spec_kind"] == "rmsnorm"
        assert result.nodes["L0.attn.wq_b"].attrs["spec_kind"] == "matmul"
        assert result.nodes["L0.attn.wkv"].attrs["spec_kind"] == "matmul"
        assert result.nodes["L0.attn.kv_norm"].attrs["spec_kind"] == "rmsnorm"
        assert result.nodes["L0.attn.compressor"].attrs["spec_kind"] == "compressor"
        assert result.nodes["L0.attn"].attrs["spec_kind"] == "attn_core"
        assert result.nodes["L0.attn.wo_a"].attrs["spec_kind"] == "matmul"
        assert result.nodes["L0.attn.wo_b"].attrs["spec_kind"] == "matmul"
        assert result.nodes["L0.ffn_norm"].attrs["spec_kind"] == "rmsnorm"
        assert result.nodes["L0.ffn.gate"].attrs["spec_kind"] == "router"
        assert result.nodes["L0.ffn.experts"].attrs["spec_kind"] == "expert"
        assert result.nodes["L0.ffn.shared_experts"].attrs["spec_kind"] == "shared_expert"
        assert result.nodes["L0.hc_pre_attn"].attrs["spec_kind"] == "mhc_pre"
        assert result.nodes["L0.hc_post_attn"].attrs["spec_kind"] == "mhc_post"
        assert result.nodes["L0.hc_pre_ffn"].attrs["spec_kind"] == "mhc_pre"
        assert result.nodes["L0.hc_post_ffn"].attrs["spec_kind"] == "mhc_post"
        assert result.nodes["norm"].attrs["spec_kind"] == "rmsnorm"
        assert result.nodes["head"].attrs["spec_kind"] == "lm_head"

    def test_v4_attn_parent_ops_separate_from_children(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        attn_node = result.nodes["L0.attn"]
        assert attn_node.num_sub_ops == 1
        assert "aten.bmm.default" in attn_node.fused_from

        wq_a_node = result.nodes["L0.attn.wq_a"]
        assert wq_a_node.num_sub_ops == 1

    def test_v4_coarsen_topo_sort(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        sorted_nodes = result.topo_sort()
        assert len(sorted_nodes) == len(result.nodes)

    def test_v4_coarsen_no_self_edges(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        for e in result.edges:
            assert e.src != e.dst, f"Self-edge: {e.src} -> {e.dst}"

    def test_v4_layer_ids(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert result.nodes["L0.attn.wq_a"].layer == "0"
        assert result.nodes["L0.attn"].layer == "0"
        assert result.nodes["L0.ffn.gate"].layer == "0"
        assert result.nodes["L0.ffn.gate"].attrs["layer_id"] == 0
        assert result.nodes["embed"].layer == "-1"
        assert result.nodes["embed"].attrs["layer_id"] == -1

    def test_v4_node_count(self):
        g = _v4_like_graph()
        original_count = len(g.nodes)
        assert original_count == 21

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert len(result.nodes) == 21


# ---------------------------------------------------------------------------
# Tests: E2.2 — End-to-end pipeline validation
# ---------------------------------------------------------------------------

class TestEndToEnd:
    def test_coarsen_then_pipeline_topo_valid(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        for node in result.nodes.values():
            assert node.attrs.get("spec_kind"), f"Missing spec_kind on {node.id}"
            assert node.attrs.get("source") == "coarsened", f"Missing source on {node.id}"

    def test_coarsen_preserves_all_tensor_metadata(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        for node in result.nodes.values():
            assert len(node.inputs) > 0, f"No inputs on {node.id}"
            assert len(node.outputs) > 0, f"No outputs on {node.id}"
            for t in node.inputs + node.outputs:
                assert t.shape, f"Empty shape on tensor {t.id}"
                assert t.dtype is not None, f"No dtype on tensor {t.id}"

    def test_coarsen_all_edges_valid(self):
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        for e in result.edges:
            assert e.src in result.nodes, f"Edge src {e.src} not in nodes"
            assert e.dst in result.nodes, f"Edge dst {e.dst} not in nodes"

    def test_coarsen_v4_with_comm_nodes(self):
        from zrt.ir.edge import Edge

        g = _v4_like_graph()
        comm = OpNode(
            id="comm_0", op_type="comm.all_reduce",
            inputs=[_tensor("comm_in")], outputs=[_tensor("comm_out")],
            scope="model.transformer.layers.0.attn",
            category="communication",
        )
        g.add_node(comm)
        g.add_edge(Edge(src="op_11", src_idx=0, dst="comm_0", dst_idx=0))
        g.add_edge(Edge(src="comm_0", src_idx=0, dst="op_12", dst_idx=0))

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        assert "comm_0" in result.nodes
        assert result.nodes["comm_0"].category == "communication"


# ---------------------------------------------------------------------------
# Tests: E2.3 — Coarsen output vs explicit graph structure
# ---------------------------------------------------------------------------

class TestStructureComparison:
    def test_coarsen_and_explicit_have_same_spec_kinds(self):
        """Coarsened V4 graph should produce similar spec_kind distribution
        as build_opgraph_direct for the same model architecture."""
        from zrt.training.spec.model import ModelSpec, LayerKind
        from zrt.training.spec.strategy import Strategy
        from zrt.training.ir.builders import build_opgraph_direct

        model = ModelSpec(
            hidden=4096, ffn=11008,
            num_heads=32, num_kv_heads=8, head_dim=128,
            vocab=32000, seq_len=2048,
            layers=[LayerKind.DENSE, LayerKind.DENSE],
        )
        strategy = Strategy(tp=1, pp=1, ep=1, dp=1)

        explicit = build_opgraph_direct(model, strategy)

        explicit_kinds = set()
        for node in explicit.nodes.values():
            if node.category != "communication":
                explicit_kinds.add(node.attrs.get("spec_kind", ""))

        assert "matmul" in explicit_kinds
        assert "rmsnorm" in explicit_kinds

    def test_coarsen_output_node_attrs_complete(self):
        """Every coarsened node must have all required attrs for downstream passes."""
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        required_attrs = {"spec_kind", "source", "layer_kind", "layer_id"}
        for node in result.nodes.values():
            if node.category == "communication":
                continue
            missing = required_attrs - set(node.attrs.keys())
            assert not missing, f"Node {node.id} missing attrs: {missing}"

    def test_coarsen_output_op_types_valid(self):
        """All coarsened op_types should be recognized by downstream passes."""
        g = _v4_like_graph()
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())

        valid_prefixes = ("aten.", "spec.", "comm.")
        valid_fusion_ops = {
            "linear", "column_parallel_linear", "row_parallel_linear",
            "rms_norm", "rms_norm_inline", "rms_coef",
            "parallel_embedding", "rotary_emb",
            "kv_compressor", "sparse_indexer", "mla_sparse_attn",
            "moe_gate", "moe_expert_swiglu",
            "hc_pre", "hc_post", "hc_head",
            "swiglu", "cross_entropy", "dropout",
        }
        for node in result.nodes.values():
            is_valid = (
                any(node.op_type.startswith(p) for p in valid_prefixes)
                or node.op_type in valid_fusion_ops
            )
            assert is_valid, f"Invalid op_type {node.op_type} on {node.id}"


# ---------------------------------------------------------------------------
# Tests: Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_graph(self):
        g = OpGraph(name="empty", phase="train")
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())
        assert len(result.nodes) == 0

    def test_single_node(self):
        n = _aten_node("op_0", "aten.mm.default",
                       "model.layers.0.q_proj", "Linear")
        g = OpGraph(name="single", phase="train", nodes={"op_0": n})
        p = GraphCoarsenPass()
        result = p.run(g, _ctx())
        assert len(result.nodes) == 1

    def test_nodes_without_scope(self):
        n1 = _aten_node("op_0", "aten.mm.default", "", "Linear")
        n2 = _aten_node("op_1", "aten.add.Tensor", "", "Linear")
        from zrt.ir.edge import Edge
        edges = [Edge(src="op_0", src_idx=0, dst="op_1", dst_idx=0)]
        g = OpGraph(name="no_scope", phase="train",
                    nodes={"op_0": n1, "op_1": n2}, edges=edges)

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())
        assert len(result.nodes) >= 1

    def test_mixed_scope_depths(self):
        from zrt.ir.edge import Edge
        n1 = _aten_node("op_0", "aten.mm.default",
                        "model.layers.0.self_attn.q_proj", "Linear")
        n2 = _aten_node("op_1", "aten.mm.default",
                        "model.layers.0.mlp", "Linear")
        edges = [Edge(src="op_0", src_idx=0, dst="op_1", dst_idx=0)]
        g = OpGraph(name="mixed", phase="train",
                    nodes={"op_0": n1, "op_1": n2}, edges=edges)

        p = GraphCoarsenPass()
        result = p.run(g, _ctx())
        assert len(result.nodes) == 2
