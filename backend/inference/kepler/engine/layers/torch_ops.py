from .op_base import OpVectorBase, OpCubeBase, TensorBase


class TorchMul(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchAdd(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchSoftmax(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchSum(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchSin(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchCos(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchCumsum(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchMm(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class TorchSort(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)
