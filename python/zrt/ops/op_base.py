from enum import Enum, auto
from dataclasses import dataclass
from typing import List

from zrt.common.chip_spec import ChipSpec
from zrt.common.tensor_base import TensorBase


class OpType(Enum):
    VECTOR = auto()
    CUBE = auto()
    MIX = auto()
    COMMUNICATION = auto()

@dataclass
class OpResult():
    # static cost for launching kernel
    static_cost:float #in us
    # compute cost
    total_compute_flops:float
    total_compute_time:float # in us
    compute_formula:str
    # memory cost
    total_memory_bytes:float
    total_memory_time:float # in us
    memory_formula:str
    # TODO communication cost

    def duration(self) -> float:
        """Roofline duration: kernel launch + max(compute, memory), in us."""
        return self.static_cost + max(self.total_compute_time, self.total_memory_time)

    def peak_memory(self) -> float:
        """Peak live memory for this op. in bytes"""
        return self.total_memory_bytes


class OperatorBase():
    input_tensors:List[TensorBase]

    def __init__(self, op_type:OpType, op_name:str, chip_spec:ChipSpec):
        self.op_type = op_type
        self.name = op_name
        self.chip_spec = chip_spec

    def get_memory_cost(self):
        pass

    def get_compute_cost(self): 
        pass

class OpVectorBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.VECTOR, op_name, chip_spec)


class OpCubeBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.CUBE, op_name, chip_spec)


class OpMixBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.MIX, op_name, chip_spec)


class OpCommBase(OperatorBase):
    def __init__(self, op_name: str, chip_spec: ChipSpec):
        super().__init__(OpType.COMMUNICATION, op_name, chip_spec)


OP_CLASS_REGISTRY = {}

def op_register(names: str | List[str]):
    def decorator(cls: type[OperatorBase]):
        if isinstance(names, str):
            op_names = [names]
        elif isinstance(names, list) and all(isinstance(name, str) for name in names):
            op_names = names
        else:
            raise ValueError("names must be a string or a list of strings")
        for name in op_names:
            OP_CLASS_REGISTRY[name] = cls
        return cls
    return decorator

def get_class_by_name(name: str) -> type[OperatorBase]:
    
    if name not in OP_CLASS_REGISTRY:
        raise ValueError(f"Operator class not found for name: {name}")
    return OP_CLASS_REGISTRY[name]