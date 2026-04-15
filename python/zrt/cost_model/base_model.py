from abc import ABC, abstractmethod
from typing import List
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

# 假设SimulateResult类型在其他模块中定义
class SimulateResult:
    def __init__(self, value):
        self.value = value

class BaseModel(ABC):
    def __init__(self, ai_chip_config, rt_config):
        self.ai_chip_config = ai_chip_config
        self.rt_config = rt_config
        self.model_name = "BaseModel"
    
    def __call__(self, op: OperatorBase, inputs: List[TensorBase], **kwargs):
        op_time = self.predict(op, inputs, **kwargs)
        return op_time
    
    @abstractmethod
    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        pass
