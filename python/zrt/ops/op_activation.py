from zrt.ops.op_base import OpVectorBase, op_register, OpResult
from zrt.common.tensor_base import TensorBase
from zrt.common.chip_spec import ChipSpec
from zrt.input_param import InputParam

from typing import List

@op_register("SwiGlu")
class SwiGlu(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("SwiGlu", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # SwiGlu通常有两个输入，输出形状与输入相同
            output_shape = tensor.shape
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
        for tensor in self.inputs:
            # SwiGlu的计算量：对于每个元素，需要一次sigmoid和一次乘法
            elements = tensor.numel
            total_compute_flops += elements * 2  # 假设sigmoid和乘法各算一次操作
        
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
            compute_formula="elements * 2",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于SwiGlu，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            if len(in_shape) == 2:
                in_shape[0] = b * s
            elif len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs
