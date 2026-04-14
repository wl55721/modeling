#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试policy_model模块的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from zrt.policy_model.policy_register import register_model, PolicyType
from zrt.policy_model.policy_model_manager import PolicyModelManager
from zrt.runtime_config import RuntimeConfig, AIChipConfig
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

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

def test_policy_model():
    """测试policy_model模块的功能"""
    print("开始测试policy_model模块...")
    
    # 注册模型
    register_model()
    print("模型注册成功")
    
    # 创建RuntimeConfig实例
    rt_config = RuntimeConfig()
    
    # 创建PolicyModelManager实例
    try:
        manager = PolicyModelManager(rt_config)
        print("PolicyModelManager创建成功")
    except Exception as e:
        print(f"PolicyModelManager创建失败: {e}")
        return False
    
    # 测试预测功能
    op = TestOp(None, "TestOp")
    input_tensor = [TestTensor((1, 1024)), TestTensor((1024, 1024))]
    
    # 测试所有策略类型
    policy_types = [
        PolicyType.PRIORITY,
        PolicyType.OOTB_PERFORMANCE,
        PolicyType.OPERATOR_OPTIMIZATION,
        PolicyType.SYSTEM_DESIGN
    ]
    
    for policy_type in policy_types:
        try:
            result = manager.predict(policy_type, op, input_tensor)
            print(f"测试{policy_type.value}成功，结果: {result}")
        except Exception as e:
            print(f"测试{policy_type.value}失败: {e}")
            return False
    
    print("所有测试通过！")
    return True

if __name__ == "__main__":
    test_policy_model()
