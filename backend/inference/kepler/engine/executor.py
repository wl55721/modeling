"""
Kepler 模型执行引擎。

将模型配置转化为 NetworkX 有向图，
按拓扑顺序执行各算子，计算每层的 Flops、显存与时延成本。
"""

from __future__ import annotations

from dataclasses import dataclass

import networkx as nx

from .common.tensor_base import TensorBase, DType
from .chips.config import AIChipConfig
from .layers.base import OperatorExecuteResult
from .model_config import ModelConfig, OperatorConfig
from . import layers as _layers
from ....utils.log import logger


# 非张量配置键（在 fill_operator 中提取为 attrs）
_CONFIG_KEYS = {"bias", "eps", "causal", "tp_size", "ep_size", "top_k"}


def _coerce_param(val: str) -> int | float | str:
    if not isinstance(val, str):
        return val
    stripped = val.strip()
    if not stripped:
        return val
    try:
        return int(stripped)
    except ValueError:
        pass
    try:
        return float(stripped)
    except ValueError:
        pass
    return val


# ── 图节点 ────────────────────────────────────────────

@dataclass
class Node:
    """计算图中的一个算子节点。"""
    op_id: int
    op_name: str
    layer_idx: int
    rank_idx: int = 0
    op_module: str = ""
    inputs: list[dict] = None  # type: ignore
    weights: list[dict] = None  # type: ignore
    outputs: list[dict] = None  # type: ignore
    compute_flops: str = ""
    op: _layers.OperatorBase | None = None
    # 执行后填充：已解析的具体 TensorBase 列表
    input_tensors: list[TensorBase] = None  # type: ignore
    weight_tensors: list[TensorBase] = None  # type: ignore
    output_tensors: list[TensorBase] = None  # type: ignore

    def __post_init__(self):
        if self.inputs is None:
            self.inputs = []
        if self.weights is None:
            self.weights = []
        if self.outputs is None:
            self.outputs = []
        if self.input_tensors is None:
            self.input_tensors = []
        if self.weight_tensors is None:
            self.weight_tensors = []
        if self.output_tensors is None:
            self.output_tensors = []

    def parse_tensors(self, context: dict) -> None:
        ratios = context.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        context["compress_ratio"] = cmp_ratio

        self.input_tensors = [TensorBase.from_spec(t, context=context) for t in self.inputs]
        self.weight_tensors = [TensorBase.from_spec(t, context=context) for t in self.weights]
        self.output_tensors = [TensorBase.from_spec(t, context=context) for t in self.outputs]

    def fill_operator(self, context: dict) -> None:
        op_cls = getattr(_layers, self.op_name, None)
        if op_cls is not None:
            try:
                self.op = op_cls(weights=self.weight_tensors, **context)
                self.op.op_id = self.op_id
                self.op.op_name = self.op_name
                self.op.layer_idx = self.layer_idx
                self.op.rank_idx = self.rank_idx
                self.op.op_module = self.op_module
                self.op.compute_flops_str = self.compute_flops
            except Exception:
                logger.error("failed to create operator %s", self.op_name, exc_info=True)
        else:
            logger.error("unknown operator type: %s", self.op_name)


# ── 执行器 ────────────────────────────────────────────

class GraphExecutor:

    def __init__(self, model: ModelConfig, context: dict, chip: AIChipConfig | None = None):
        self.model = model
        self.context = context
        self.chip = chip or AIChipConfig()
        self.graph = nx.DiGraph()

    def run(self) -> dict[int, OperatorExecuteResult]:
        self._build_graph()
        return self._execute_topological()

    # ── 图构建 ────────────────────────────────────────

    def _build_graph(self) -> None:
        for op in self.model.operators:
            self._add_node(
                op_id=op.op_id,
                layer_idx=op.layer_idx,
                rank_idx=op.rank_idx,
                op_name=op.op_name,
                op_module=op.op_module,
                inputs=op.inputs,
                weights=op.params,
                outputs=op.outputs,
                compute_flops=op.compute_flops,
            )

        if self.model.edges:
            for edge in self.model.edges:
                if edge.source != edge.target:
                    self.graph.add_edge(edge.source, edge.target)
        else:
            op_ids = sorted(op.op_id for op in self.model.operators)
            for i in range(1, len(op_ids)):
                self.graph.add_edge(op_ids[i - 1], op_ids[i])

        logger.info("graph built: %d nodes, %d edges",
                    self.graph.number_of_nodes(), self.graph.number_of_edges())

    def _add_node(
            self,
            op_id: int,
            layer_idx: int,
            rank_idx: int = 0,
            op_name: str = "",
            op_module: str = "",
            inputs: list[dict] | None = None,
            weights: list[dict] | None = None,
            outputs: list[dict] | None = None,
            compute_flops: str = "",
    ) -> None:
        node = Node(
            op_id=op_id,
            op_name=op_name,
            layer_idx=layer_idx,
            rank_idx=rank_idx,
            op_module=op_module,
            inputs=inputs or [],
            weights=weights or [],
            outputs=outputs or [],
            compute_flops=compute_flops,
        )
        node.parse_tensors(self.context)
        node.fill_operator(self.context)

        self.graph.add_node(op_id, node=node)

    # ── 拓扑执行 ───────────────────────────────────────

    def _execute_topological(self) -> dict[int, OperatorExecuteResult]:
        results: dict[int, OperatorExecuteResult] = {}
        finish_time: dict[int, int] = {}
        logger.info("topological execution start — %d nodes", self.graph.number_of_nodes())

        for op_id in nx.topological_sort(self.graph):
            node: Node = self.graph.nodes[op_id]["node"]

            if node.op is None:
                results[op_id] = OperatorExecuteResult(op_id=op_id, op_name=node.op_name, layer_idx=node.layer_idx, rank_idx=node.rank_idx)
                finish_time[op_id] = 0
                continue

            try:
                output = node.op(node.input_tensors, node.output_tensors)
            except Exception:
                logger.error("operator %s (idx=%s) execution failed", node.op_name, node.layer_idx, exc_info=True)
                results[op_id] = OperatorExecuteResult(op_id=op_id, op_name=node.op_name, layer_idx=node.layer_idx, rank_idx=node.rank_idx)
                finish_time[op_id] = 0
                continue

            node.op.calc_cost_us(self.chip)
            r = node.op.execute_result

            # earliest start = max finish time of all predecessors
            start_time: int = 0
            for pred in self.graph.predecessors(op_id):
                start_time = max(start_time, finish_time.get(pred, 0))

            r.op_id = op_id
            r.rank_idx = node.rank_idx
            r.start_time_ns = start_time
            r.end_time_ns = int(start_time + node.op.execute_result.total_cost_us * 1000)
            results[op_id] = r

            finish_time[op_id] = r.end_time_ns

        logger.info("execution done — %d results", len(results))
        return results



def execute_model(model: ModelConfig, context: dict,
                  chip: AIChipConfig | None = None) -> dict[int, OperatorExecuteResult]:
    return GraphExecutor(model=model, context=context, chip=chip).run()
