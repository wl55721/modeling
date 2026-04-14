from typing import List
from zrt.cost_model.base_model import BaseModel, SimulateResult
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

class LookupTableModel(BaseModel):
    """查表模型"""
    def __init__(self, ai_chip_config, rt_config):
        super().__init__(ai_chip_config, rt_config)
        self.model_name = "LookupTableModel"
        # 初始化查找表
        self.lookup_table = {}

    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        """根据算子和输入预测执行时间"""
        # 生成查找键
        key = self._generate_key(op, inputs)
        # 从查找表中获取结果
        if key in self.lookup_table:
            return SimulateResult(self.lookup_table[key])
        else:
            # 如果查找表中没有，返回默认值
            return SimulateResult(0.0)

    def _generate_key(self, op: OperatorBase, inputs: List[TensorBase]) -> str:
        """生成查找键"""
        # 基于算子类型和输入形状生成键
        op_type = op.__class__.__name__
        input_shapes = [str(input_tensor.get_shape()) for input_tensor in inputs]
        return f"{op_type}_{'_'.join(input_shapes)}"
