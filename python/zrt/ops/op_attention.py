from zrt.ops.op_base import OpVectorBase, op_register, OpResult
from zrt.common.tensor_base import TensorBase
from zrt.common.chip_spec import ChipSpec
from zrt.input_param import InputParam

from typing import List

@op_register("ScaledDotProductAttention")
class ScaledDotProductAttention(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("ScaledDotProductAttention", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        # 通常输入包括query、key、value
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        # 输出形状与query相同
        if input_tensor:
            output_shape = input_tensor[0].shape
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        return self.outputs[0] if self.outputs else None

    def get_memory_cost(self) -> OpResult:
        # 计算内存成本
        total_memory_bytes = 0
        for tensor in self.inputs:
            total_memory_bytes += tensor.nbytes
        for tensor in self.outputs:
            total_memory_bytes += tensor.nbytes
        
        # 计算内存访问时间
        memory_bandwidth = self.chip_spec.hbm_bandwidth_gbps  # 假设芯片规格中有内存带宽
        total_memory_time = (total_memory_bytes * 2) / (memory_bandwidth * 1e9) * 1e6  # 转换为微秒
        
        # 计算计算成本
        total_compute_flops = 0
        if len(self.inputs) >= 3:
            # 注意力计算: Q[B,S,D] * K^T[B,D,S] = [B,S,S], FLOPS = 2*B*S*S*D
            shape = self.inputs[0].shape
            if len(shape) == 3:
                B, S, D = shape
                total_compute_flops = 2 * B * S * S * D
        
        # 计算计算时间
        from zrt.common.tensor_base import DType
        compute_flops = self.chip_spec.peak_tflops(DType.from_str("float16"))  # 获取 float16 精度的计算能力，单位为 TFLOPS
        total_compute_time = total_compute_flops / (compute_flops * 1e12) * 1e6  # 转换为微秒
        
        # 静态成本（内核启动时间）
        static_cost = 1.0  # 假设为1微秒
        
        # 构建并返回OpResult
        return OpResult(
            static_cost=static_cost,
            total_compute_flops=total_compute_flops,
            total_compute_time=total_compute_time,
            compute_formula="2 * B * S * S * D",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于注意力计算，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
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
            shape = self.inputs[0].shape
            if len(shape) == 3:
                B, S, D = shape
                flops = 2 * B * S * S * D
        
        # 计算内存访问
        memory_access = 0.0
        for tensor in self.inputs:
            shape = tensor.shape
            elements = 1
            for dim in shape:
                elements *= dim
            memory_access += elements * 2  # 假设FP16
        for tensor in self.outputs:
            shape = tensor.shape
            elements = 1
            for dim in shape:
                elements *= dim
            memory_access += elements * 2  # 假设FP16
        
        # 计算时间
        compute_time = flops / flops_per_second * 1000  # 毫秒
        memory_time = memory_access / memory_bandwidth * 1000  # 毫秒
        
        # 返回计算时间和内存时间
        return compute_time, memory_time
