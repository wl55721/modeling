from .op_base import OpVectorBase, TensorBase


class RopeComplex(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


class RopeInterLeave(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        in_shape = self.inputs[1].shape
        in_shape[0] = bsz
        in_shape[1] = qlen
        
        out_shape = self.outputs[1].shape
        out_shape[0] = bsz
        out_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen
