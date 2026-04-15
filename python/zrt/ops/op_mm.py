from zrt.ops.op_base import OpVectorBase, OpCubeBase, op_register, OpResult
from zrt.common.tensor_base import TensorBase
from zrt.common.chip_spec import ChipSpec
from zrt.input_param import InputParam

from typing import List

@op_register("Bmm")
class Bmm(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Bmm", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # Bmm的输出形状取决于输入形状
            # 假设输入是 [batch, m, k] 和 [batch, k, n]
            # 输出是 [batch, m, n]
            input_shape1 = input_tensor[0].shape
            input_shape2 = input_tensor[1].shape
            output_shape = [input_shape1[0], input_shape1[1], input_shape2[2]]
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
        
        # 计算计算成本（Bmm的计算量）
        total_compute_flops = 0
        if len(self.inputs) >= 2:
            shape1 = self.inputs[0].shape
            shape2 = self.inputs[1].shape
            if len(shape1) == 3 and len(shape2) == 3:
                # 批量矩阵乘法: [B,M,K] * [B,K,N] = [B,M,N]
                B, M, K = shape1
                N = shape2[2]
                total_compute_flops = 2 * B * M * K * N
        
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
            compute_formula="2 * B * M * K * N",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于Bmm，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            if len(in_shape) == 3:
                in_shape[0] = b
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("MatMul")
class MatMul(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("MatMul", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # MatMul的输出形状取决于输入形状
            # 假设输入是 [m, k] 和 [k, n]
            # 输出是 [m, n]
            input_shape1 = input_tensor[0].shape
            input_shape2 = input_tensor[1].shape
            output_shape = [input_shape1[0], input_shape2[1]]
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
        
        # 计算计算成本（MatMul的计算量）
        total_compute_flops = 0
        if len(self.inputs) >= 2:
            shape1 = self.inputs[0].shape
            shape2 = self.inputs[1].shape
            if len(shape1) == 3 and len(shape2) == 2:
                # 批量矩阵乘法: [B,M,K] * [K,N] = [B,M,N]
                B, M, K = shape1
                N = shape2[1]
                total_compute_flops = 2 * B * M * K * N
            elif len(shape1) == 2 and len(shape2) == 2:
                # 普通矩阵乘法: [M,K] * [K,N] = [M,N]
                M, K = shape1
                N = shape2[1]
                total_compute_flops = 2 * M * K * N
        
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
            compute_formula="2 * M * K * N",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于MatMul，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs
    
    def get_overlap_cost(self, ai_chip_config):
        """计算算子的重叠成本"""
        # 假设硬件参数
        flops_per_second = 1e12  # 1 TFLOPS
        memory_bandwidth = 1e12  # 1 TB/s
        
        # 计算FLOPS
        flops = 0.0
        if len(self.inputs) >= 2:
            shape1 = self.inputs[0].shape
            shape2 = self.inputs[1].shape
            if len(shape1) == 3 and len(shape2) == 2:
                # 批量矩阵乘法: [B,M,K] * [K,N] = [B,M,N]
                B, M, K = shape1
                N = shape2[1]
                flops = 2 * B * M * K * N
            elif len(shape1) == 2 and len(shape2) == 2:
                # 普通矩阵乘法: [M,K] * [K,N] = [M,N]
                M, K = shape1
                N = shape2[1]
                flops = 2 * M * K * N
        
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

@op_register("GroupedMatMul")
class GroupedMatMul(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("GroupedMatMul", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # GroupedMatMul的输出形状取决于输入形状
            input_shape1 = input_tensor[0].shape
            input_shape2 = input_tensor[1].shape
            output_shape = list(input_shape1)
            output_shape[-1] = input_shape2[-1]
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
        
        # 计算计算成本（GroupedMatMul的计算量）
        total_compute_flops = 0
        if len(self.inputs) >= 2:
            shape1 = self.inputs[0].shape
            shape2 = self.inputs[1].shape
            if len(shape1) == 4 and len(shape2) == 3:
                # 分组矩阵乘法: [B, G, M, K] * [G, K, N] = [B, G, M, N]
                B, G, M, K = shape1
                N = shape2[2]
                total_compute_flops = 2 * B * G * M * K * N
        
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
            compute_formula="2 * B * G * M * K * N",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于GroupedMatMul，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            if len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("Mm")
class Mm(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Mm", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if len(input_tensor) >= 2:
            # Mm的输出形状取决于输入形状
            input_shape1 = input_tensor[0].shape
            input_shape2 = input_tensor[1].shape
            output_shape = [input_shape1[0], input_shape2[1]]
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
        
        # 计算计算成本（Mm的计算量）
        total_compute_flops = 0
        if len(self.inputs) >= 2:
            shape1 = self.inputs[0].shape
            shape2 = self.inputs[1].shape
            if len(shape1) == 2 and len(shape2) == 2:
                # 普通矩阵乘法: [M,K] * [K,N] = [M,N]
                M, K = shape1
                N = shape2[1]
                total_compute_flops = 2 * M * K * N
        
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
            compute_formula="2 * M * K * N",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于Mm，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("Linear")
class Linear(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Linear", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # Linear的输出形状取决于输入形状和输出维度
            input_shape = input_tensor[0].shape
            output_dim = kwargs.get('output_dim', 768)  # 默认输出维度
            output_shape = list(input_shape)
            output_shape[-1] = output_dim
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
        
        # 计算计算成本（Linear的计算量）
        total_compute_flops = 0
        if self.inputs and self.outputs:
            input_shape = self.inputs[0].shape
            output_shape = self.outputs[0].shape
            if len(input_shape) == 2:
                # 线性层: [B, in_dim] * [in_dim, out_dim] = [B, out_dim]
                B, in_dim = input_shape
                out_dim = output_shape[1]
                total_compute_flops = 2 * B * in_dim * out_dim
            elif len(input_shape) == 3:
                # 线性层: [B, S, in_dim] * [in_dim, out_dim] = [B, S, out_dim]
                B, S, in_dim = input_shape
                out_dim = output_shape[2]
                total_compute_flops = 2 * B * S * in_dim * out_dim
        
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
            compute_formula="2 * B * in_dim * out_dim",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于Linear，计算成本和内存成本可以使用相同的方法
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

@op_register("Einsum")
class Einsum(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Einsum", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # Einsum的输出形状取决于爱因斯坦求和约定
            # 这里简化处理，使用第一个输入的形状
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
        
        # 计算计算成本（Einsum的计算量，这里简化处理）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements * 2  # 假设每个元素需要两次操作
        
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
        # 对于Einsum，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        dynamic_inputs = []
        for tensor in input_tensor:
            in_shape = list(tensor.shape)
            if len(in_shape) == 3:
                in_shape[0] = b
                in_shape[1] = s
            dynamic_inputs.append(TensorBase(shape=in_shape, dtype=tensor.dtype))
        
        return dynamic_inputs

@op_register("ColumnParallelLinear")
class ColumnParallelLinear(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("ColumnParallelLinear", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs = []
        for tensor in input_tensor:
            self.inputs.append(tensor)
        
        self.outputs = []
        if input_tensor:
            # ColumnParallelLinear的输出形状取决于输入形状和输出维度
            input_shape = input_tensor[0].shape
            output_dim = kwargs.get('output_dim', 768)  # 默认输出维度
            output_shape = list(input_shape)
            output_shape[-1] = output_dim
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
        
        # 计算计算成本（ColumnParallelLinear的计算量）
        total_compute_flops = 0
        if self.inputs and self.outputs:
            input_shape = self.inputs[0].shape
            output_shape = self.outputs[0].shape
            if len(input_shape) == 2:
                # 线性层: [B, in_dim] * [in_dim, out_dim] = [B, out_dim]
                B, in_dim = input_shape
                out_dim = output_shape[1]
                total_compute_flops = 2 * B * in_dim * out_dim
            elif len(input_shape) == 3:
                # 线性层: [B, S, in_dim] * [in_dim, out_dim] = [B, S, out_dim]
                B, S, in_dim = input_shape
                out_dim = output_shape[2]
                total_compute_flops = 2 * B * S * in_dim * out_dim
        
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
            compute_formula="2 * B * in_dim * out_dim",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于ColumnParallelLinear，计算成本和内存成本可以使用相同的方法
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
