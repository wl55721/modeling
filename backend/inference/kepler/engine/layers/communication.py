from ..common.tensor_base import DType, TensorBase
from .op_base import OpCommBase


class AllReduce(OpCommBase):

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        super().__init__(weights, **attrs)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []

        if not self.inputs:
            return self.outputs

        self.dynamic_update_b_s()

        if 'embed' in self.op_module:
            embed_tp = self.cfg.get('embed_tp_size', 1)
            self.rank_size = embed_tp
        elif 'o_proj' in self.op_module:
            o_tp = self.cfg.get('o_tp_size', 1)
            self.rank_size = o_tp
        else:
            self.rank_size = 8

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def calc_comm_bytes(self):
        self.comm_bytes = 2 * (self.rank_size - 1) / self.rank_size * self.inputs[0].nbytes


class AllGather(OpCommBase):

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        super().__init__(weights, **attrs)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []

        if not self.inputs:
            return self.outputs

        self.dynamic_update_b_s()

        if 'lm_head' in self.op_module:
            lmhead_tp = self.cfg.get('lmhead_tp_size', 1)
            self.rank_size = lmhead_tp
        else:
            self.rank_size = 8

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def calc_comm_bytes(self):
        self.comm_bytes = self.rank_size * self.inputs[0].nbytes
        

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
        # 2: send and receive
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
        # 2: send and receive
        self.comm_bytes = 2 * self.input_bytes
