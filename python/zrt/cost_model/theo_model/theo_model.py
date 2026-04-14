
from typing import List
from zrt.cost_model.base_model import BaseModel, SimulateResult
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

class TheoreticalModel(BaseModel):
    def __init__(self, ai_chip_config, rt_config):
        super().__init__(ai_chip_config, rt_config)
        self.model_name = "TheoreticalModel"

    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        op(inputs, **kwargs)
        # 假设op.get_overlap_cost方法返回两个数值
        # 这里添加一个默认实现，以防该方法不存在
        try:
            cost1, cost2 = op.get_overlap_cost(self.ai_chip_config)
        except AttributeError:
            cost1, cost2 = 0.0, 0.0

        return SimulateResult(max(cost1, cost2))