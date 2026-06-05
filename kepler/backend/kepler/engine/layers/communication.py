from .base import OpCommBase
from ..common.tensor_base import TensorBase


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
