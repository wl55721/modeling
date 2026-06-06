import math

from .op_base import OpMixBase, OpVectorBase, TensorBase


class Compressor(OpMixBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []
        
        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()
        
        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        self.cfg["compress_ratio"] = cmp_ratio

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        x_shape[0] = bsz
        x_shape[1] = qlen

        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        o_shape = self.outputs[0].shape
        o_shape[0] = bsz
        o_shape[1] = math.ceil(qlen / cmp_ratio)

        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class IndexCompressorEpilog(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)

    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []
        
        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()
        
        phase = self.cfg.get('phase', 'decode')
        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        if phase == 'prefill':
            factor = 1.0 if cmp_ratio != 1 else 0.0
        else:
            if cmp_ratio == 1:
                factor = 0.0
            elif cmp_ratio == 4:
                factor = 1 / 4
            elif cmp_ratio == 128:
                factor = 1 / 128
            else:
                factor = 1.0
        

        self.cfg["factor"] = factor
        self.cfg["compress_ratio"] = cmp_ratio

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        x_shape[0] = bsz
        x_shape[1] = qlen

        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        o_shape = self.outputs[0].shape
        o_shape[0] = bsz
        o_shape[1] = math.ceil(qlen / cmp_ratio)
        
        self.cfg["B"] = bsz
        self.cfg["S"] = qlen


class KVCompressorEpilog(OpVectorBase):

    def __init__(self, weights: list[TensorBase], **attrs):
        super().__init__(weights, **attrs)
    
    def __call__(self, input_tensors, out_tensors):
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []
        
        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()
        
        phase = self.cfg.get('phase', 'decode')
        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        if phase == 'prefill':
            factor = 1.0 if cmp_ratio != 1 else 0.0
        else:
            if cmp_ratio == 1:
                factor = 0.0
            elif cmp_ratio == 4:
                factor = 1 / 4
            elif cmp_ratio == 128:
                factor = 1 / 128
            else:
                factor = 1.0
        

        self.cfg["factor"] = factor
        self.cfg["compress_ratio"] = cmp_ratio

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs

    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        x_shape[0] = bsz
        x_shape[1] = qlen

        ratios = self.cfg.get("compress_ratios")
        cmp_ratio = ratios[self.layer_idx] if ratios and 0 <= self.layer_idx < len(ratios) else 1
        o_shape = self.outputs[0].shape
        o_shape[0] = bsz
        o_shape[1] = math.ceil(qlen / cmp_ratio)

