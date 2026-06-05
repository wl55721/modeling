from backend.kepler.engine.common.tensor_base import DType

from .base import OpVectorBase, TensorBase


class Sigmoid(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def calc_compute_flops(self) -> float:
        # [B, S, 2 * intermediate_size]
        B, S, D = self.inputs[0].shape

        # 2: exp + 1/exp 
        self.sfu_compute_flops = 2 * B * S * D // 2 * DType.FP16.bytes

        self.compute_flops = 2 * B * S * D // 2 * DType.FP16.bytes


class SwiGlu(OpVectorBase):
    SFU_RATIO = 5.0 / 9.0

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def calc_compute_flops(self) -> float:
        # [B, S, 2 * intermediate_size]
        B, S, D = self.inputs[0].shape

        # 2: exp + 1/exp 
        self.sfu_compute_flops = 2 * B * S * D // 2 * DType.FP16.bytes

        # outputs=swiglu(x,dim=−1)=swish(A)∗B=A∗sigmoid(A)∗B
        self.compute_flops = 4 * B * S * D // 2 * DType.FP16.bytes


class SwiGluQuant(OpVectorBase):
    SFU_RATIO = 5.0 / 9.0

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        out_shape = self.outputs[1].shape
        out_shape[0] = bsz
        out_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen

    def calc_compute_flops(self) -> float:
        # [B, S, 2 * intermediate_size]
        B, S, D = self.inputs[0].shape

        # 2: exp + 1/exp 
        self.sfu_compute_flops = 2 * B * S * D // 2 * DType.FP16.bytes

        # outputs=swiglu(x,dim=−1)=swish(A)∗B=A∗sigmoid(A)∗B
        self.compute_flops = 7 * B * S * D // 2 * DType.FP16.bytes
