from .op_base import OpMixBase, TensorBase


class FlashAttention(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("causal", True)
        super().__init__(weights, **attrs)
        self.static_cost_us = 10


class SparseAttentionSharedKV(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("causal", True)
        super().__init__(weights, **attrs)
        self.static_cost_us = 10

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []

        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()
        
        _, kv_len, _, _ = self.inputs[1].shape
        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 0
        if cmp_ratio == 0 or kv_len < 128:
            if kv_len > 128:
                kv_len = 128
                kv_shape = list(self.inputs[1].shape)
                kv_shape[1] = kv_len
                self.inputs[1] = TensorBase(name=self.inputs[1].name, shape=kv_shape, dtype=self.inputs[1].dtype)

        if cmp_ratio == 4:
            kv_len = (kv_len + 4 - 1) // 4 + 128
            kv_shape = list(self.inputs[1].shape)
            kv_shape[1] = kv_len
            self.inputs[1] = TensorBase(name=self.inputs[1].name, shape=kv_shape, dtype=self.inputs[1].dtype)

        if cmp_ratio == 128:
            kv_len = (kv_len + 128 - 1) // 128 + 128
            kv_shape = list(self.inputs[1].shape)
            kv_shape[1] = kv_len
            self.inputs[1] = TensorBase(name=self.inputs[1].name, shape=kv_shape, dtype=self.inputs[1].dtype)

        self.cfg["seq_len"] = kv_len

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)
        
        in_shape = self.inputs[1].shape
        in_shape[0] = bsz
        in_shape[1] = kvlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class ScaledDotProductAttn(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("causal", True)
        super().__init__(weights, **attrs)
        self.static_cost_us = 10


class PageAttention(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)
        self.static_cost_us = 10


class SparseFlashAttention(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("causal", True)
        super().__init__(weights, **attrs)
        self.static_cost_us = 10

