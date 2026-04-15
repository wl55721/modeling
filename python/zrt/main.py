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
    
    # 模型配置
    parser.add_argument('--num-layers', type=int, default=12,
                      help='Transformer层数')
    parser.add_argument('--num-heads', type=int, default=12,
                      help='注意力头数')
    parser.add_argument('--vocab-size', type=int, default=50257,
                      help='词汇表大小')
    
    # 策略模型配置
    parser.add_argument('--policy-type', type=str, default='priority',
                      help='策略模型类型，可选值: priority, ootb_performance, operator_optimization, system_design')
    
    # 输出配置
    parser.add_argument('--output-file', type=str, default=None,
                      help='输出结果到文件')
    
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

def setup_runtime_config(args) -> RuntimeConfig:
    """设置运行时配置"""
    # 初始化配置
    rt_config = RuntimeConfig()
    rt_config.ai_chip_config = AIChipConfig()
    
    # 将命令行参数写入RuntimeConfig
    rt_config.model_target = args.model_target
    rt_config.operator = args.operator
    rt_config.batch_size = args.batch_size
    rt_config.seq_len = args.seq_len
    rt_config.hidden_size = args.hidden_size
    rt_config.num_layers = args.num_layers
    rt_config.num_heads = args.num_heads
    rt_config.vocab_size = args.vocab_size
    rt_config.policy_type = args.policy_type
    rt_config.output_file = args.output_file
    
    return rt_config

def create_deepseekv3_2_operators(input_param: InputParam, hidden_size: int, num_layers: int, num_heads: int, vocab_size: int) -> List[OperatorBase]:
    """创建DeepSeek V3.2模型的算子序列"""
    operators = []
    
    # 1. 嵌入层
    embedding_op = get_class_by_name("Embedding")(None, "Embedding")
    embedding_input = TensorBase([input_param.batch_size, input_param.seq_len])
    embedding_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
    embedding_op.inputs = [embedding_input]
    embedding_op.outputs = [embedding_output]
    embedding_op.set_memory_bytes()
    operators.append(embedding_op)
    
    # 2. Transformer层
    for layer_idx in range(num_layers):
        # 2.1 层归一化 (注意力前)
        norm1_op = get_class_by_name("RMSNorm")(None, f"RMSNorm_Layer{layer_idx}_1")
        norm1_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        norm1_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        norm1_op.inputs = [norm1_input]
        norm1_op.outputs = [norm1_output]
        norm1_op.set_memory_bytes()
        operators.append(norm1_op)
        
        # 2.2 多头注意力层
        # 2.2.1 QKV线性变换
        qkv_op = get_class_by_name("MatMul")(None, f"MatMul_QKV_Layer{layer_idx}")
        qkv_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        qkv_weight = TensorBase([hidden_size, hidden_size * 3])  # Q, K, V
        qkv_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size * 3])
        qkv_op.inputs = [qkv_input, qkv_weight]
        qkv_op.outputs = [qkv_output]
        qkv_op.set_memory_bytes()
        operators.append(qkv_op)
        
        # 2.2.2 注意力计算
        attention_op = get_class_by_name("ScaledDotProductAttention")(None, f"ScaledDotProductAttention_Layer{layer_idx}")
        attention_input1 = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])  # Q
        attention_input2 = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])  # K
        attention_input3 = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])  # V
        attention_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        attention_op.inputs = [attention_input1, attention_input2, attention_input3]
        attention_op.outputs = [attention_output]
        attention_op.set_memory_bytes()
        operators.append(attention_op)
        
        # 2.2.3 注意力输出线性变换
        attention_output_op = get_class_by_name("MatMul")(None, f"MatMul_Attention_Output_Layer{layer_idx}")
        attn_out_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        attn_out_weight = TensorBase([hidden_size, hidden_size])
        attn_out_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        attention_output_op.inputs = [attn_out_input, attn_out_weight]
        attention_output_op.outputs = [attn_out_output]
        attention_output_op.set_memory_bytes()
        operators.append(attention_output_op)
        
        # 2.3 层归一化 (前馈网络前)
        norm2_op = get_class_by_name("RMSNorm")(None, f"RMSNorm_Layer{layer_idx}_2")
        norm2_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        norm2_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        norm2_op.inputs = [norm2_input]
        norm2_op.outputs = [norm2_output]
        norm2_op.set_memory_bytes()
        operators.append(norm2_op)
        
        # 2.4 前馈网络
        # 2.4.1 第一层线性变换
        ff1_op = get_class_by_name("MatMul")(None, f"MatMul_FFN1_Layer{layer_idx}")
        ff1_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        ff1_weight = TensorBase([hidden_size, hidden_size * 4])  # 通常是hidden_size的4倍
        ff1_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size * 4])
        ff1_op.inputs = [ff1_input, ff1_weight]
        ff1_op.outputs = [ff1_output]
        ff1_op.set_memory_bytes()
        operators.append(ff1_op)
        
        # 2.4.2 SwiGLU激活函数
        swiglu_op = get_class_by_name("SwiGlu")(None, f"SwiGlu_Layer{layer_idx}")
        swiglu_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size * 4])
        swiglu_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size * 2])
        swiglu_op.inputs = [swiglu_input]
        swiglu_op.outputs = [swiglu_output]
        swiglu_op.set_memory_bytes()
        operators.append(swiglu_op)
        
        # 2.4.3 第二层线性变换
        ff2_op = get_class_by_name("MatMul")(None, f"MatMul_FFN2_Layer{layer_idx}")
        ff2_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size * 2])
        ff2_weight = TensorBase([hidden_size * 2, hidden_size])
        ff2_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
        ff2_op.inputs = [ff2_input, ff2_weight]
        ff2_op.outputs = [ff2_output]
        ff2_op.set_memory_bytes()
        operators.append(ff2_op)
    
    # 3. 最终层归一化
    final_norm_op = get_class_by_name("RMSNorm")(None, "RMSNorm_Final")
    final_norm_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
    final_norm_output = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
    final_norm_op.inputs = [final_norm_input]
    final_norm_op.outputs = [final_norm_output]
    final_norm_op.set_memory_bytes()
    operators.append(final_norm_op)
    
    # 4. 输出层
    output_op = get_class_by_name("MatMul")(None, "MatMul_Output")
    output_input = TensorBase([input_param.batch_size, input_param.seq_len, hidden_size])
    output_weight = TensorBase([hidden_size, vocab_size])
    output_output = TensorBase([input_param.batch_size, input_param.seq_len, vocab_size])
    output_op.inputs = [output_input, output_weight]
    output_op.outputs = [output_output]
    output_op.set_memory_bytes()
    operators.append(output_op)
    
    return operators

def simulate_deepseekv3_2_inference(cost_model_manager: CostModelManager, input_param: InputParam, hidden_size: int, num_layers: int, num_heads: int, vocab_size: int):
    """模拟DeepSeek V3.2模型的推理过程"""
    # 创建算子序列
    operators = create_deepseekv3_2_operators(input_param, hidden_size, num_layers, num_heads, vocab_size)
    
    # 统计信息
    total_time = 0.0
    total_memory = 0.0
    operator_stats = []
    
    print("\n===== DeepSeek V3.2 推理模拟 =====")
    print(f"批处理大小: {input_param.batch_size}")
    print(f"序列长度: {input_param.seq_len}")
    print(f"隐藏层大小: {hidden_size}")
    print(f"Transformer层数: {num_layers}")
    print(f"注意力头数: {num_heads}")
    print(f"词汇表大小: {vocab_size}")
    print("\n算子执行详情:")
    print("-" * 160)
    print(f"{'算子名称':<25} {'执行时间(ms)':<15} {'输入张量':<50} {'输出张量':<50} {'内存占用(MB)':<15}")
    print("-" * 160)
    
    for i, op in enumerate(operators):
        # 预测执行时间
        time = cost_model_manager.predict(op, op.inputs)
        
        # 计算内存占用 (bytes -> MB)
        memory_mb = op.memory_bytes / (1024 * 1024)
        
        # 构建输入输出张量描述
        input_desc = ", ".join([f"{t.get_shape()}" for t in op.inputs])
        output_desc = ", ".join([f"{t.get_shape()}" for t in op.outputs])
        
        # 输出详情
        print(f"{op.name:<25} {time:<15.6f} {input_desc:<50} {output_desc:<50} {memory_mb:<15.6f}")
        
        # 累计统计
        total_time += time
        total_memory += memory_mb
        operator_stats.append({
            "name": op.name,
            "time": time,
            "input_shape": [t.get_shape() for t in op.inputs],
            "output_shape": [t.get_shape() for t in op.outputs],
            "memory": memory_mb
        })
    
    print("-" * 160)
    print(f"{'总计':<25} {total_time:<15.6f} {'':<50} {'':<50} {total_memory:<15.6f}")
    print("-" * 160)
    
    return total_time, total_memory, operator_stats

def export_results(output_file, total_time, total_memory, operator_stats, rt_config):
    """导出结果到文件"""
    if not output_file:
        return
    
    import json
    
    results = {
        "config": {
            "batch_size": rt_config.batch_size,
            "seq_len": rt_config.seq_len,
            "hidden_size": rt_config.hidden_size,
            "num_layers": rt_config.num_layers,
            "num_heads": rt_config.num_heads,
            "vocab_size": rt_config.vocab_size,
            "model_target": rt_config.model_target
        },
        "summary": {
            "total_time": total_time,
            "total_memory": total_memory,
            "average_time_per_operator": total_time / len(operator_stats),
            "average_memory_per_operator": total_memory / len(operator_stats),
            "num_operators": len(operator_stats)
        },
        "operators": operator_stats
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n结果已导出到: {output_file}")

def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置运行时配置
    rt_config = setup_runtime_config(args)
    
    # 初始化输入参数
    input_param = InputParam(batch_size=rt_config.batch_size, seq_len=rt_config.seq_len)
    
    # 初始化模型
    cost_model_manager, policy_model_manager = initialize_models(rt_config.model_target, rt_config)
    
    # 模拟DeepSeek V3.2推理
    total_time, total_memory, operator_stats = simulate_deepseekv3_2_inference(
        cost_model_manager, 
        input_param, 
        rt_config.hidden_size, 
        rt_config.num_layers, 
        rt_config.num_heads, 
        rt_config.vocab_size
    )
    
    # 输出总结
    print("\n===== 推理总结 =====")
    print(f"总执行时间: {total_time:.6f} ms")
    print(f"总内存占用: {total_memory:.6f} MB")
    print(f"平均每个算子时间: {total_time / len(operator_stats):.6f} ms")
    print(f"平均每个算子内存: {total_memory / len(operator_stats):.6f} MB")
    print(f"总算子数量: {len(operator_stats)}")
    
    # 导出结果
    export_results(rt_config.output_file, total_time, total_memory, operator_stats, rt_config)

if __name__ == "__main__":
    main()
