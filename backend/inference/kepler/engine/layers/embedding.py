from .op_base import OpVectorBase, TensorBase


class Embedding(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        x_shape[0] = bsz
        x_shape[1] = qlen

        o_shape = self.outputs[0].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen