#!/usr/bin/env python3
"""
项目入口文件

此文件是ZRT项目的主入口，用于启动成本模型和策略模型的预测功能。
"""

import argparse
import os
import sys

# 添加python目录到路径，确保zrt模块能被正确导入
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from typing import List
from zrt.cost_model.cost_model_manager import CostModelManager
from zrt.cost_model.model_register import ModelType, register_model
from zrt.policy_model.policy_model_manager import PolicyModelManager
from zrt.policy_model.policy_register import PolicyType, register_model as register_policy_model
from zrt.layers.op_base import OperatorBase, get_class_by_name
from zrt.tensor_base import TensorBase
from zrt.runtime_config import RuntimeConfig, AIChipConfig
from zrt.input_param import InputParam

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='ZRT Model Runner')
    
    # 基本配置
    parser.add_argument('--model-target', nargs='+', type=str, default=['THEO_MODEL'],
                      help='成本模型目标，可选值: LOOKUP, TILESIM_ENGI, TILESIM_THEO, TILESIM_ENGI_DSL, THEO_MODEL')
    
    # 算子配置
    parser.add_argument('--operator', type=str, default='MatMul',
                      help='要测试的算子名称')
    
    # 张量配置
    parser.add_argument('--batch-size', type=int, default=32,
                      help='批处理大小')
    parser.add_argument('--seq-len', type=int, default=1024,
                      help='序列长度')
    parser.add_argument('--hidden-size', type=int, default=768,
                      help='隐藏层大小')
    
    # 策略模型配置
    parser.add_argument('--policy-type', type=str, default='priority',
                      help='策略模型类型，可选值: priority, ootb_performance, operator_optimization, system_design')
    
    return parser.parse_args()

def initialize_models(model_targets: List[str], rt_config: RuntimeConfig):
    """初始化模型"""
    # 注册成本模型
    register_model()
    
    # 注册策略模型
    register_policy_model()
    
    # 转换模型目标为枚举类型
    model_target_enums = []
    for target in model_targets:
        if target in ModelType.__members__:
            model_target_enums.append(ModelType[target])
        else:
            print(f"警告: 未知的模型目标 {target}，将被忽略")
    
    # 创建成本模型管理器
    cost_model_manager = CostModelManager(rt_config.ai_chip_config, rt_config, model_target_enums)
    
    # 创建策略模型管理器
    policy_model_manager = PolicyModelManager(rt_config)
    
    return cost_model_manager, policy_model_manager

def create_operator(operator_name: str, input_param: InputParam, hidden_size: int) -> OperatorBase:
    """创建算子实例"""
    try:
        # 获取算子类
        op_class = get_class_by_name(operator_name)
        
        # 创建算子实例
        op = op_class(None, operator_name)
        
        # 设置输入张量
        input_tensor = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        weight_tensor = TensorBase([hidden_size, hidden_size])
        op.inputs = [input_tensor, weight_tensor]
        
        # 设置输出张量
        output_tensor = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        op.outputs = [output_tensor]
        
        # 设置内存大小
        op.set_memory_bytes()
        
        return op
    except ValueError as e:
        print(f"创建算子失败: {e}")
        return None

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 初始化配置
    rt_config = RuntimeConfig()
    rt_config.ai_chip_config = AIChipConfig()
    
    # 初始化输入参数
    input_param = InputParam(batch_size=args.batch_size, seq_len=args.seq_len)
    
    # 初始化模型
    cost_model_manager, policy_model_manager = initialize_models(args.model_target, rt_config)
    
    # 创建算子
    op = create_operator(args.operator, input_param, args.hidden_size)
    if not op:
        return
    
    # 准备输入张量
    input_tensors = op.inputs
    
    # 使用成本模型预测
    print("\n===== 成本模型预测 =====")
    cost = cost_model_manager.predict(op, input_tensors)
    print(f"算子 {args.operator} 的预测成本: {cost}")
    
    # 使用策略模型预测
    print("\n===== 策略模型预测 =====")
    try:
        policy_type = PolicyType(args.policy_type)
        policy_score = policy_model_manager.predict(policy_type, op, input_tensors)
        print(f"策略 {args.policy_type} 的预测分数: {policy_score}")
    except ValueError as e:
        print(f"策略模型预测失败: {e}")

if __name__ == "__main__":
    main()
