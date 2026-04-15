
from typing import List
from zrt.cost_model.base_model import BaseModel, SimulateResult
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

class TheoreticalModel(BaseModel):
    def __init__(self, ai_chip_config, rt_config):
        super().__init__(ai_chip_config, rt_config)
        self.model_name = "TheoreticalModel"
        
        # 硬件参数（假设值，实际应从ai_chip_config获取）
        self.flops_per_second = 1e12  # 1 TFLOPS
        self.memory_bandwidth = 1e12  # 1 TB/s

    def calculate_flops(self, op: OperatorBase, inputs: List[TensorBase]) -> float:
        """计算算子的FLOPS"""
        # 根据算子类型计算FLOPS
        if op.name == "MatMul":
            # 矩阵乘法: A[M,K] * B[K,N] = C[M,N], FLOPS = 2*M*K*N
            if len(inputs) >= 2:
                shape1 = inputs[0].get_shape()
                shape2 = inputs[1].get_shape()
                if len(shape1) == 3 and len(shape2) == 2:
                    # 批量矩阵乘法: [B,M,K] * [K,N] = [B,M,N]
                    B, M, K = shape1
                    N = shape2[1]
                    return 2 * B * M * K * N
                elif len(shape1) == 2 and len(shape2) == 2:
                    # 普通矩阵乘法: [M,K] * [K,N] = [M,N]
                    M, K = shape1
                    N = shape2[1]
                    return 2 * M * K * N
        elif op.name == "ScaledDotProductAttention":
            # 注意力计算: Q[B,S,D] * K^T[B,D,S] = [B,S,S], FLOPS = 2*B*S*S*D
            if len(inputs) >= 3:
                shape = inputs[0].get_shape()
                if len(shape) == 3:
                    B, S, D = shape
                    return 2 * B * S * S * D
        elif op.name == "Embedding":
            # 嵌入层: 查找操作，FLOPS相对较小
            return 0.0
        elif op.name == "RMSNorm":
            # 层归一化: 每个元素的计算，FLOPS = 元素数量 * 操作数
            if len(inputs) >= 1:
                shape = inputs[0].get_shape()
                elements = 1
                for dim in shape:
                    elements *= dim
                return elements * 5  # 假设每个元素需要5次操作
        elif op.name == "SwiGlu":
            # SwiGLU激活函数: 每个元素的计算，FLOPS = 元素数量 * 操作数
            if len(inputs) >= 1:
                shape = inputs[0].get_shape()
                elements = 1
                for dim in shape:
                    elements *= dim
                return elements * 10  # 假设每个元素需要10次操作
        
        # 默认返回0
        return 0.0

    def calculate_memory_access(self, op: OperatorBase, inputs: List[TensorBase]) -> float:
        """计算内存访问量（字节）"""
        # 计算输入和输出的内存访问
        memory_access = 0.0
        
        # 输入内存访问
        for tensor in inputs:
            shape = tensor.get_shape()
            elements = 1
            for dim in shape:
                elements *= dim
            # 假设FP16数据类型，2字节 per element
            memory_access += elements * 2
        
        # 输出内存访问
        for tensor in op.outputs:
            shape = tensor.get_shape()
            elements = 1
            for dim in shape:
                elements *= dim
            memory_access += elements * 2
        
        return memory_access

    def predict(self, op: OperatorBase, inputs: List[TensorBase], **kwargs) -> SimulateResult:
        # 执行算子，设置输入输出
        op(inputs, **kwargs)
        
        # 计算计算时间（基于FLOPS）
        flops = self.calculate_flops(op, inputs)
        compute_time = flops / self.flops_per_second * 1000  # 转换为毫秒
        
        # 计算内存访问时间
        memory_access = self.calculate_memory_access(op, inputs)
        memory_time = memory_access / self.memory_bandwidth * 1000  # 转换为毫秒
        
        # 考虑算子的静态成本
        static_cost = getattr(op, 'static_cost', 0.0)
        
        # 总执行时间 = 计算时间 + 内存访问时间 + 静态成本
        total_time = compute_time + memory_time + static_cost
        
        # 尝试使用算子的get_overlap_cost方法（如果有）
        try:
            cost1, cost2 = op.get_overlap_cost(self.ai_chip_config)
            # 如果算子有自己的成本计算方法，使用它
            if cost1 > 0 or cost2 > 0:
                total_time = max(cost1, cost2)
        except AttributeError:
            pass
        
        # 确保时间不为负数
        total_time = max(total_time, 0.0)
        
        return SimulateResult(total_time)