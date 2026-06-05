from ..common.tensor_base import DType

from .base import OpCubeBase, OpMixBase, OpVectorBase, TensorBase
from .communication import OpCommBase


class MoEDispatch(OpCommBase):

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        attrs.setdefault("top_k", 2)
        attrs.setdefault("ep_size", 1)
        super().__init__(weights, **attrs)

        self.top_k = self.cfg.get('top_k', 6)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []
        
        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()
        
        if self.cfg.get('external_shared_expert_rank_size', 0) > 0:
            n_shared_experts = self.cfg.get('n_shared_experts', 0)
            if n_shared_experts > 0:
                self.top_k += n_shared_experts
        self.rank_size = self.cfg.get('ep_size', 1)

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        x_shape[0] = bsz
        x_shape[1] = qlen

        expert_ids = self.inputs[1].shape
        expert_ids[0] = bsz
        expert_ids[1] = qlen

        o_shape = self.outputs[0].shape
        o_shape[1] = bsz
        o_shape[2] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen

    def calc_comm_bytes(self):
        quant_tensor = self.outputs[0].clone().to(DType.INT8)
        # 2: 发送和接收
        self.comm_bytes = 2 * quant_tensor.nbytes

class MoECombine(OpCommBase):

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        attrs.setdefault("top_k", 2)
        attrs.setdefault("ep_size", 1)
        super().__init__(weights, **attrs)

        self.top_k = self.cfg.get('top_k', 6)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []
        
        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()

        if self.cfg.get('external_shared_expert_rank_size', 0) > 0:
            n_shared_experts = self.cfg.get('n_shared_experts', 0)
            if n_shared_experts > 0:
                self.top_k += n_shared_experts
        self.rank_size = self.cfg.get('ep_size', 1)

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        x_shape[1] = bsz
        x_shape[2] = qlen

        expert_ids = self.inputs[1].shape
        expert_ids[0] = bsz
        expert_ids[1] = qlen

        o_shape = self.outputs[0].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen

    def calc_comm_bytes(self):
        # 2: 发送和接收
        self.comm_bytes = 2 * self.input_bytes


class MoEGate(OpCubeBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def calc_compute_flops(self) -> float:
        # [B, S, hidden_size]
        B, S, D = self.inputs[0].shape

        num_experts = self.cfg.get('num_experts', 1)

        # softplus(x)=ln(1+e^x).sqrt(x)
        self.sfu_compute_flops = 3 * B * S * num_experts * DType.FP32.bytes

        # outputs=swiglu(x,dim=−1)=swish(A)∗B=A∗sigmoid(A)∗B
        self.compute_flops = 2 * B * S * D * num_experts * DType.FP32.bytes + 2 * B * S * num_experts * DType.FP32.bytes

class MoETopK(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("top_k", 2)
        super().__init__(weights, **attrs)

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        o_shape = self.outputs[1].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class MoEGateTopK(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("top_k", 2)
        super().__init__(weights, **attrs)

    def calc_compute_flops(self) -> float:
        # [B, S, hidden_size]
        B, S, D = self.inputs[0].shape

        num_experts = self.cfg.get('num_experts', 1)

        # 2: exp + 1/exp 
        self.sfu_compute_flops = 3 * B * S * num_experts * DType.FP32.bytes

        # outputs=swiglu(x,dim=−1)=swish(A)∗B=A∗sigmoid(A)∗B
        self.compute_flops = 2 * B * S * D * num_experts * DType.FP32.bytes + 4 * B * S * num_experts * DType.FP32.bytes

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        o_shape = self.outputs[1].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class MoEGateHashTopK(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("top_k", 2)
        super().__init__(weights, **attrs)

    def calc_compute_flops(self) -> float:
        # [B, S, hidden_size]
        B, S, D = self.inputs[0].shape

        num_experts = self.cfg.get('num_experts', 1)

        # 2: exp + 1/exp 
        self.sfu_compute_flops = 3 * B * S * num_experts * DType.FP32.bytes

        # outputs=swiglu(x,dim=−1)=swish(A)∗B=A∗sigmoid(A)∗B
        self.compute_flops = 2 * B * S * D * num_experts * DType.FP32.bytes + 4 * B * S * num_experts * DType.FP32.bytes

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        token_ids = self.inputs[1].shape
        token_ids[0] = bsz
        token_ids[1] = qlen

        o_shape = self.outputs[1].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class LightningIndexer(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        attrs.setdefault("top_k", 2)
        super().__init__(weights, **attrs)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []

        if not self.inputs:
            return self.outputs

        self.dynamic_update_b_s()

        # TODO
        self.caches = []

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        qr = self.inputs[0].shape
        key = self.inputs[1].shape
        weights = self.inputs[2].shape

        qr[0] = bsz
        key[0] = bsz
        weights[0] = bsz

        qr[1] = qlen
        weights[1] = qlen

        key_len = kvlen + 1
        if 'compress_ratios' in self.cfg:
            compress_ratio = self.cfg['compress_ratios'][self.layer_idx]
            key_len = key_len // compress_ratio
            key[1] = key_len

        self.cfg["seq_len"] = key_len * compress_ratio

        index_topk = self.cfg.get('index_topk', 1024)
        key_len = index_topk if key_len > index_topk else key_len
       
        if self.outputs:
            self.outputs[0].shape[-1] = key_len
            
        self.cfg["B"] = bsz
        self.cfg["S"] = qlen

        


class IndexPrologV4(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        o_shape = self.outputs[1].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class MLAPrologV4(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)
    
    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        super().update_bsz_qlen_kvlen(bsz, qlen, kvlen)

        o_shape = self.outputs[1].shape
        o_shape[0] = bsz
        o_shape[1] = qlen

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class MLAEpilogV4(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)
    