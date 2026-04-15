from typing import List
from zrt.cost_model.base_model import BaseModel, SimulateResult
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

class TilesimEngModel(BaseModel):
    """Tilesim工程模型"""
    def __init__(self, ai_chip_config, rt_config):
        super().__init__(ai_chip_config, rt_config)
        self.model_name = "TilesimEngModel"

    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        """根据算子和输入预测执行时间"""
        # 这里是Tilesim工程模型的实现
        # 暂时返回默认值
        return SimulateResult(0.0)

class TilesimTheoModel(BaseModel):
    """Tilesim理论模型"""
    def __init__(self, ai_chip_config, rt_config):
        super().__init__(ai_chip_config, rt_config)
        self.model_name = "TilesimTheoModel"

    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        """根据算子和输入预测执行时间"""
        # 这里是Tilesim理论模型的实现
        # 暂时返回默认值
        return SimulateResult(0.0)

class TilesimEngDSLModel(BaseModel):
    """Tilesim工程DSL模型"""
    def __init__(self, ai_chip_config, rt_config):
        super().__init__(ai_chip_config, rt_config)
        self.model_name = "TilesimEngDSLModel"

    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        """根据算子和输入预测执行时间"""
        # 这里是Tilesim工程DSL模型的实现
        # 暂时返回默认值
        return SimulateResult(0.0)
