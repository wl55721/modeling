#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试Python路径下所有模块的集成和交互
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase
from zrt.runtime_config import RuntimeConfig, AIChipConfig
from zrt.policy_model.policy_register import register_model as register_policy_model
from zrt.policy_model.policy_model_manager import PolicyModelManager
from zrt.policy_model.policy_register import PolicyType
from zrt.cost_model.model_register import register_model as register_cost_model, ModelType

# 创建测试用的算子类
class TestOp(OperatorBase):
    def __init__(self, op_model, op_name):
        super().__init__(op_model, op_name, is_vector=True, op_dtype=None)
    
    def __call__(self, inputs, **kwargs):
        pass
    
    def get_overlap_cost(self, ai_chip_config):
        return 1.0, 2.0

# 创建测试用的张量类
class TestTensor(TensorBase):
    def __init__(self, shape):
        self.shape = shape
    
    def get_shape(self):
        return self.shape
    
    def get_string(self):
        return str(self.shape)
    
    def get_flops(self):
        return 0.0

def test_integration():
    """测试所有模块的集成和交互"""
    print("开始测试Python路径下的代码集成...")
    
    # 1. 初始化配置
    print("\n1. 测试配置模块...")
    rt_config = RuntimeConfig()
    ai_chip_config = AIChipConfig()
    rt_config.ai_chip_config = ai_chip_config
    print("配置模块测试成功")
    
    # 2. 注册策略模型
    print("\n2. 测试策略模型模块...")
    register_policy_model()
    print("策略模型注册成功")
    
    # 创建策略模型管理器
    policy_manager = PolicyModelManager(rt_config)
    print("策略模型管理器创建成功")
    
    # 3. 注册成本模型
    print("\n3. 测试成本模型模块...")
    register_cost_model()
    print("成本模型注册成功")
    
    # 4. 创建测试用的算子和输入
    print("\n4. 测试算子模块...")
    op = TestOp(None, "TestOp")
    inputs = [TestTensor((1, 1024)), TestTensor((1024, 1024))]
    print("算子和输入创建成功")
    
    # 5. 测试策略模型预测
    print("\n5. 测试策略模型预测...")
    policy_types = [
        PolicyType.PRIORITY,
        PolicyType.OOTB_PERFORMANCE,
        PolicyType.OPERATOR_OPTIMIZATION,
        PolicyType.SYSTEM_DESIGN
    ]
    
    for policy_type in policy_types:
        try:
            result = policy_manager.predict(policy_type, op, inputs)
            print(f"  测试{policy_type.value}策略成功，结果: {result}")
        except Exception as e:
            print(f"  测试{policy_type.value}策略失败: {e}")
    
    # 6. 测试算子调用
    print("\n6. 测试算子调用...")
    try:
        op(inputs)
        print("  算子调用成功")
    except Exception as e:
        print(f"  算子调用失败: {e}")
    
    print("\n所有集成测试完成！")

if __name__ == "__main__":
    test_integration()
