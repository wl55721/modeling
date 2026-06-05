from ..common.tensor_base import TensorBase
from .base import OpCubeBase


class MatMul(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("bias", False)
        super().__init__(weights, **attrs)


class Linear(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("bias", False)
        super().__init__(weights, **attrs)


class ColumnParallelLinear(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("bias", False)
        super().__init__(weights, **attrs)


class RowParallelLinear(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("bias", False)
        super().__init__(weights, **attrs)


class GroupMatMul(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("top_k", 2)
        super().__init__(weights, **attrs)


class ColumnParallelLinearQuant(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("bias", False)
        super().__init__(weights, **attrs)

