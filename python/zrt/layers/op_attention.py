from zrt.layers.op_base import OpVectorBase, op_register
from zrt.tensor_base import TensorBase
from zrt.input_param import InputParam

from typing import List

@op_register("ScaledDotProductAttention")
class ScaledDotProductAttention(OpVectorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, "ScaledDotProductAttention")

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        # 通常输入包括query、key、value
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        # 输出形状与query相同
        if input_tensor:
            output_shape = input_tensor[0].get_shape()
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        # 假设存在这些方法
        if input_tensor:
            self.compute_formula = input_tensor[0].get_string()
            self.compute_flops = input_tensor[0].get_flops()

        # 假设存在这个方法
        self.set_memory_bytes()

        return self.outputs[0] if self.outputs else None

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = tensor.get_shape()
            if len(in_shape) == 3:
                # 假设形状为 [batch, seq_len, hidden_dim]
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs
    
    def get_overlap_cost(self, ai_chip_config):
        """计算算子的重叠成本"""
        # 假设硬件参数
        flops_per_second = 1e12  # 1 TFLOPS
        memory_bandwidth = 1e12  # 1 TB/s
        
        # 计算FLOPS
        flops = 0.0
        if len(self.inputs) >= 3:
            # 注意力计算: Q[B,S,D] * K^T[B,D,S] = [B,S,S], FLOPS = 2*B*S*S*D
            shape = self.inputs[0].get_shape()
            if len(shape) == 3:
                B, S, D = shape
                flops = 2 * B * S * S * D
        
        # 计算内存访问
        memory_access = 0.0
        for tensor in self.inputs:
            shape = tensor.get_shape()
            elements = 1
            for dim in shape:
                elements *= dim
            memory_access += elements * 2  # 假设FP16
        for tensor in self.outputs:
            shape = tensor.get_shape()
            elements = 1
            for dim in shape:
                elements *= dim
            memory_access += elements * 2  # 假设FP16
        
        # 计算时间
        compute_time = flops / flops_per_second * 1000  # 毫秒
        memory_time = memory_access / memory_bandwidth * 1000  # 毫秒
        
        # 返回计算时间和内存时间
        return compute_time, memory_time
