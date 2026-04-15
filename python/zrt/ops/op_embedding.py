from zrt.ops.op_base import OpVectorBase, op_register, OpResult
from zrt.common.tensor_base import TensorBase
from zrt.common.chip_spec import ChipSpec
from zrt.input_param import InputParam

from typing import List

@op_register("Embedding")
class Embedding(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Embedding", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # Embedding的输出形状通常是 [batch, seq_len, embedding_dim]
            input_shape = input_tensor[0].shape
            # 假设最后一维是词索引，输出将增加embedding_dim维度
            output_shape = list(input_shape)
            output_shape.append(kwargs.get('embedding_dim', 768))  # 默认embedding维度
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
        
        # 计算计算成本（Embedding主要是内存访问，计算成本较低）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements  # 假设每个元素需要一次查表操作
        
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
            compute_formula="elements",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于Embedding，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            if len(in_shape) == 1:
                # 假设形状为 [seq_len]
                in_shape = [b * s]
            elif len(in_shape) == 2:
                # 假设形状为 [batch, seq_len]
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs
