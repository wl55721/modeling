
from abc import ABC, abstractmethod
from typing import List
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase
from zrt.runtime_config import RuntimeConfig

class PolicyBaseModel(ABC):
    def __init__(self, rt_config: RuntimeConfig):
        self.rt_config = rt_config
        self.ai_chip_config = rt_config.ai_chip_config

    def __call__(self, op: OperatorBase, input_tensor: List[TensorBase], **kwargs):
        return self.predict(op, input_tensor, **kwargs)
    
    @abstractmethod
    def predict(self, op: OperatorBase, input_tensor: List[TensorBase], **kwargs) -> float:
        raise NotImplementedError("predict method not implemented")
