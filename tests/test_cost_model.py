#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
测试cost_model模块的功能
"""

import sys
import os

# 添加项目根目录到Python路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python')))

from zrt.cost_model.model_register import register_model, ModelType, COST_MODEL_MAP
from zrt.runtime_config import RuntimeConfig, AIChipConfig
from zrt.layers.op_base import OperatorBase
from zrt.tensor_base import TensorBase

# 创建测试用的算子类
class TestOp(OperatorBase):
    def __init__(self, op_model, op_name):
        # 添加缺少的参数
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

def test_cost_model():
    """测试cost_model模块的功能"""
    print("开始测试cost_model模块...")
    
    # 注册模型
    register_model()
    print(f"模型注册成功，注册了{len(COST_MODEL_MAP)}个模型")
    
    # 创建配置
    rt_config = RuntimeConfig()
    ai_chip_config = AIChipConfig()
    rt_config.ai_chip_config = ai_chip_config
    
    # 创建测试用的算子和输入
    op = TestOp(None, "TestOp")
    inputs = [TestTensor((1, 1024)), TestTensor((1024, 1024))]
    
    # 测试每个注册的模型
    for model_type, model_class in COST_MODEL_MAP.items():
        try:
            # 创建模型实例
            model = model_class(ai_chip_config, rt_config)
            print(f"创建{model.model_name}成功")
            
            # 测试预测功能
            result = model(op, inputs)
            print(f"测试{model.model_name}成功，结果: {result.value}")
        except Exception as e:
            print(f"测试{model_type.name}失败: {e}")
    
    print("所有测试完成！")

if __name__ == "__main__":
    test_cost_model()
