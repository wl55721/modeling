# ZRT 项目入口说明

## 项目概述

ZRT (Zero Runtime) 是一个用于预测AI算子执行成本和制定优化策略的模型系统。该系统包含成本模型和策略模型两大部分，可以帮助开发者评估和优化AI模型的性能。

## 入口文件

项目的主入口文件是 `main.py`，位于 `python/zrt/` 目录下。

## 启动参数

`main.py` 支持以下命令行参数：

### 1. 基本配置

- `--model-target`：成本模型目标，可指定多个模型，可选值包括：
  - `LOOKUP`：查找表模型
  - `TILESIM_ENGI`：Tilesim工程模型
  - `TILESIM_THEO`：Tilesim理论模型
  - `TILESIM_ENGI_DSL`：Tilesim工程DSL模型
  - `THEO_MODEL`：理论模型
  - 默认值：`THEO_MODEL`

### 2. 算子配置

- `--operator`：要测试的算子名称，默认值：`MatMul`

### 3. 张量配置

- `--batch-size`：批处理大小，默认值：`32`
- `--seq-len`：序列长度，默认值：`1024`
- `--hidden-size`：隐藏层大小，默认值：`768`

### 4. 策略模型配置

- `--policy-type`：策略模型类型，可选值包括：
  - `priority`：优先级模型
  - `ootb_performance`：开箱即用性能模型
  - `operator_optimization`：算子优化模型
  - `system_design`：系统设计模型
  - 默认值：`priority`

## 使用示例

### 基本用法

```bash
# 使用默认参数运行
python main.py

# 指定成本模型和算子
python main.py --model-target THEO_MODEL TILESIM_THEO --operator MatMul

# 自定义张量配置
python main.py --batch-size 64 --seq-len 2048 --hidden-size 1024

# 使用不同的策略模型
python main.py --policy-type operator_optimization
```

### 输出示例

```
===== 成本模型预测 =====
算子 MatMul 的预测成本: 0.123456

===== 策略模型预测 =====
策略 priority 的预测分数: 0.987654
```

## 系统架构

1. **成本模型系统**：
   - `CostModelManager`：管理不同类型的成本模型
   - 支持多种成本模型，包括查找表模型、Tilesim模型和理论模型

2. **策略模型系统**：
   - `PolicyModelManager`：管理不同类型的策略模型
   - 支持优先级模型、开箱即用性能模型、算子优化模型和系统设计模型

3. **算子系统**：
   - `OperatorBase`：所有算子的基类
   - 支持多种算子类型，如矩阵乘法、激活函数、注意力机制等

4. **基础数据结构**：
   - `TensorBase`：表示张量，提供形状和类型信息
   - `RuntimeConfig`：管理运行时配置
   - `AIChipConfig`：管理AI芯片配置
   - `InputParam`：管理输入参数

## 依赖关系

- Python 3.7+
- 无外部依赖，纯Python实现

## 扩展方法

1. **添加新的成本模型**：
   - 在 `cost_model` 目录下创建新的模型文件
   - 在 `model_register.py` 中注册新模型

2. **添加新的策略模型**：
   - 在 `policy_model` 目录下创建新的模型文件
   - 在 `policy_register.py` 中注册新模型

3. **添加新的算子**：
   - 在 `layers` 目录下创建新的算子文件
   - 使用 `@op_register` 装饰器注册新算子
