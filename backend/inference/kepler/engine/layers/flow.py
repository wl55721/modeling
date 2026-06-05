from typing import TYPE_CHECKING

from .base import OperatorExecuteResult, OperatorBase, TensorBase

if TYPE_CHECKING:
    from ..chips.config import AIChipConfig


class _FlowNode(OperatorBase):
    """图入口/出口基类 —— 零计算、零带宽的透传节点。"""

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        super().__init__(weights, **attrs)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []
        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def calc_bw_bytes(self):
        self.bw_bytes = 0

    def calc_comm_bytes(self):
        self.comm_bytes = 0

    def calc_cost_us(self, chip: AIChipConfig) -> float:
        self.execute_result = OperatorExecuteResult(
            op_id=self.op_id,
            op_name=self.op_name,
            layer_idx=self.layer_idx,
            rank_idx=self.rank_idx,
            compute_flops=0,
            input_bytes=0,
            param_bytes=0,
            cache_bytes=0,
            output_bytes=0,
            bw_bytes=0,
            comm_bytes=0,
            compute_cost_us=0.0,
            bw_cost_us=0,
            comm_cost_us=0,
            total_cost_us=0,
            static_cost_us=0,
            bound_type="communication"
        )
        return 0


class START(_FlowNode):
    """模型入口 —— 产生初始输入张量。"""


class END(_FlowNode):
    """模型出口 —— 透传最终输出张量。"""
