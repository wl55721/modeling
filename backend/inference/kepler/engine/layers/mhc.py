from .op_base import OpMixBase, TensorBase


class MHCPre(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        o_shape = self.outputs[1].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        o_shape = self.outputs[2].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class MHCPost(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)


    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        in_shape = self.inputs[1].shape
        in_shape[0] = bsz
        in_shape[1] = qlen
        
        in_shape = self.inputs[2].shape
        in_shape[0] = bsz
        in_shape[1] = qlen

        in_shape = self.inputs[3].shape
        in_shape[0] = bsz
        in_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class MHCHead(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)
