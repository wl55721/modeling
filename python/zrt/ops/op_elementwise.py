from typing import List

from zrt.ops.op_base import OpVectorBase, OpCubeBase, op_register, OpResult
from zrt.common.tensor_base import TensorBase
from zrt.common.chip_spec import ChipSpec
from zrt.input_param import InputParam

@op_register(["Add", "AddInplace"])
class Add(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Add", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        if len(input_tensor) > 1:
            y_tensor = input_tensor[1]
            self.inputs.append(y_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements  # 假设每个元素需要一次加法操作
        
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
        # 对于Add，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)

        if len(input_tensor) == 1:
            return [in0]
        
        in_shape_y = list(input_tensor[1].shape)
        if len(in_shape_y) == 2:
            in_shape_y[0] = b * s
        else:
            in_shape_y[0] = b
            in_shape_y[1] = s

        in1 = TensorBase(shape=in_shape_y, dtype=input_tensor[1].dtype)
        return [in0, in1]

@op_register(["Mul", "MulInplace"])
class Mul(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Mul", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        if len(input_tensor) > 1:
            y_tensor = input_tensor[1]
            self.inputs.append(y_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements  # 假设每个元素需要一次乘法操作
        
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
        # 对于Mul，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)

        if len(input_tensor) == 1:
            return [in0]
        
        in_shape_y = list(input_tensor[1].shape)
        if len(in_shape_y) == 2:
            in_shape_y[0] = b * s
        else:
            in_shape_y[0] = b
            in_shape_y[1] = s

        in1 = TensorBase(shape=in_shape_y, dtype=input_tensor[1].dtype)
        return [in0, in1]

@op_register("Softmax")
class Softmax(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Softmax", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Softmax需要指数、求和和除法操作）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements * 3  # 假设每个元素需要三次操作
        
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
            compute_formula="elements * 3",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于Softmax，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Sin")
class Sin(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Sin", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Sin需要三角函数计算）
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
        # 对于Sin，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Cos")
class Cos(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Cos", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Cos需要三角函数计算）
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
        # 对于Cos，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Histc")
class Histc(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Histc", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        # Histc的输出形状取决于bins参数
        bins = kwargs.get('bins', 10)
        self.outputs.append(TensorBase(shape=[bins], dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Histc需要统计每个bin的计数）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements  # 假设每个元素需要一次操作
        
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
        # 对于Histc，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Sort")
class Sort(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Sort", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Sort需要排序操作，时间复杂度为O(n log n)）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements * 10  # 假设每个元素需要10次操作（近似O(n log n)）
        
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
            compute_formula="elements * 10",
            total_memory_bytes=total_memory_bytes,
            total_memory_time=total_memory_time,
            memory_formula="sum(inputs) + sum(outputs)"
        )

    def get_compute_cost(self) -> OpResult:
        # 对于Sort，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Sum")
class Sum(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Sum", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        # Sum的输出形状取决于dim参数
        dim = kwargs.get('dim', None)
        if dim is None:
            # 全局求和，输出标量
            self.outputs.append(TensorBase(shape=[], dtype=self.op_dtype))
        else:
            # 沿指定维度求和，输出形状为原形状去掉该维度
            output_shape = list(x_tensor.shape)
            if dim < len(output_shape):
                output_shape.pop(dim)
            self.outputs.append(TensorBase(shape=output_shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Sum需要累加操作）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements  # 假设每个元素需要一次加法操作
        
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
        # 对于Sum，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]

@op_register("Cumsum")
class Cumsum(OpVectorBase):
    def __init__(self, chip_spec: ChipSpec):
        super().__init__("Cumsum", chip_spec)
        self.inputs = []
        self.outputs = []
        self.op_dtype = "float16"  # 默认数据类型

    def __call__(self, input_tensor: List[TensorBase], **kwargs):
        self.inputs  = []
        x_tensor = input_tensor[0]
        self.inputs.append(x_tensor)
        
        self.outputs = []
        self.outputs.append(TensorBase(shape=x_tensor.shape, dtype=self.op_dtype))

        return self.outputs[0]

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
        
        # 计算计算成本（Cumsum需要累加操作）
        total_compute_flops = 0
        if self.inputs:
            elements = self.inputs[0].numel
            total_compute_flops = elements  # 假设每个元素需要一次加法操作
        
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
        # 对于Cumsum，计算成本和内存成本可以使用相同的方法
        return self.get_memory_cost()

    @classmethod
    def build_dynamic_input(cls, input_tensor: List[TensorBase], params: InputParam) -> List[TensorBase]:
        b = params.batch_size
        s = params.seq_len
        
        in_shape_x = list(input_tensor[0].shape)
        if len(in_shape_x) == 2:
            in_shape_x[0] = b * s
        else:
            in_shape_x[0] = b
            in_shape_x[1] = s
        
        in0 = TensorBase(shape=in_shape_x, dtype=input_tensor[0].dtype)
        return [in0]
