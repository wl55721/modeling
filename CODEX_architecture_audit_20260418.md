# ARCHITECTURE.md 设计目标盘点

基于 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md) 对当前项目代码进行盘点，目标是回答三个问题：

1. 文档中的架构目标是什么。
2. 当前仓库已经实现到什么程度。
3. 主要缺口和后续优先级是什么。

盘点时间：2026-04-18  
盘点范围：`python/zrt/*`、`tests/*`、`README.md`、`ARCHITECTURE.md`

---

## 总体结论

当前项目已经完成了从“LLM 算子抓图工具”向“LLM 性能建模框架”迁移的大半主干，但还没有达到 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md) 描述的完整 V2 状态。

更准确地说：

- `Foundation Layer` 和 `Core Layer` 的主脊梁已经搭起来了。
- `Application Layer` 基本停留在设计层，代码里尚未形成统一编排。
- `Extension Layer` 基本未开始。
- 当前最成熟、最可信的能力是：
  `graph capture -> OpGraph IR -> transform -> roofline simulate -> DAG schedule -> summary/report`

按文档路线图判断，当前大致处于 `Phase 1.5 ~ Phase 2.0`，而不是完整 V2。

---

## 分层盘点

### 1. 设计理念对齐情况

文档 1.1 中的主要设计理念，对齐情况如下：

| 设计理念 | 当前状态 | 判断 |
|---|---|---|
| 图驱动 | 已有 `OpGraph` 作为新主线中枢 | 基本实现 |
| 硬件与软件栈正交 | 硬件注册表已实现，软件栈体系未落地 | 部分实现 |
| 先切分再融合 | `TransformPipeline` 已固定 `split -> fuse -> optim -> analyze` | 已实现 |
| 显存一等公民 | 文档要求独立 `MemoryModel`，当前缺失 | 未实现 |
| 仿真器可插拔 | `SimulatorHub` 已有，但只有 roofline 后端 | 部分实现 |
| 无卡运行 | `graph` 抓图已基于 fake/meta tensor 主线 | 已实现 |
| 模块独立可服务化 | 代码接口有这个方向，但未形成服务化应用层 | 部分实现 |

结论：

- 架构方向是对的。
- 目前最弱的不是 IR 或抓图，而是 `MemoryModel`、`SoftwareStack`、`Application Layer`。

---

### 2. Foundation Layer

#### 2.1 计算图 IR

对应目标：

- `TensorMeta / OpNode / Edge / OpGraph`
- 层次化视图 `GraphHierarchy`
- JSON 序列化
- 平台无关，通过 annotations 承载平台信息

当前实现：

- [`python/zrt/ir/types.py`](D:/workspace/claude/modeling/python/zrt/ir/types.py)
- [`python/zrt/ir/node.py`](D:/workspace/claude/modeling/python/zrt/ir/node.py)
- [`python/zrt/ir/edge.py`](D:/workspace/claude/modeling/python/zrt/ir/edge.py)
- [`python/zrt/ir/graph.py`](D:/workspace/claude/modeling/python/zrt/ir/graph.py)
- [`python/zrt/ir/hierarchy.py`](D:/workspace/claude/modeling/python/zrt/ir/hierarchy.py)
- [`python/zrt/ir/serde.py`](D:/workspace/claude/modeling/python/zrt/ir/serde.py)
- [`python/zrt/ir/adapter.py`](D:/workspace/claude/modeling/python/zrt/ir/adapter.py)

完成度：`已实现`

判断：

- 这是当前最接近架构文档的一层。
- `OpGraph` 已经是新链路里的真实中枢。
- `GraphHierarchy` 也已经进入 summary 计算链路。

缺口：

- 文档中的 `ir/utils.py` 未独立落地。
- 旧 `graph/*` 表示和新 IR 仍并存，迁移尚未完全收口。

#### 2.2 硬件注册表

当前实现：

- [`python/zrt/hardware/spec.py`](D:/workspace/claude/modeling/python/zrt/hardware/spec.py)
- [`python/zrt/hardware/registry.py`](D:/workspace/claude/modeling/python/zrt/hardware/registry.py)
- [`python/zrt/hardware/configs`](D:/workspace/claude/modeling/python/zrt/hardware/configs)

完成度：`已实现`

判断：

- 代码结构和文档目标基本一致。
- YAML 加载、`HardwareSpec`、带宽/算力接口都具备。

缺口：

- 目前是“机制完整、覆盖面有限”。

#### 2.3 软件栈注册表

文档目标：

- `stacks/`
- `SoftwareStack`
- `mindie / vllm / tensorrt_llm`
- `fusion_rules / op_mapping / optim_caps`

当前实现：

- 无 `python/zrt/stacks`
- 仅 [`python/zrt/transform/context.py`](D:/workspace/claude/modeling/python/zrt/transform/context.py) 中有 `stack` 预留字段

完成度：`缺失`

判断：

- 这是当前最明显的“文档有、代码无”的模块之一。
- 也因此文档中“硬件 × 软件栈正交”的设计还没有真正成立。

#### 2.4 通信模型

文档目标：

- 独立 `CommModel`
- 负责 all-reduce / all-to-all 等通信开销估算
- 供 `CommInserter`、调度与报告使用

当前实现：

- 代码里有通信节点插入
- 但没有独立 `CommModel`

完成度：`缺失`

判断：

- 当前是“有通信节点，没有通信建模”。
- 多卡链路还缺一块关键基础设施。

---

### 3. Core Layer

#### 3.1 模型管理器

当前最接近的实现：

- [`python/zrt/graph/model_loader.py`](D:/workspace/claude/modeling/python/zrt/graph/model_loader.py)
- [`python/zrt/graph/compat.py`](D:/workspace/claude/modeling/python/zrt/graph/compat.py)
- [`python/zrt/graph/patches.py`](D:/workspace/claude/modeling/python/zrt/graph/patches.py)

完成度：`部分实现`

判断：

- 模型加载、compat shim、patch 注入、fake/meta mode 生命周期都已经具备。
- 但它仍是“抓图链路的模型加载器”，不是文档中的统一 `ModelManager`。

缺口：

- 缺统一 `ModelProfile`
- 缺面向 orchestrator 的统一抽象

#### 3.2 图抓取

当前实现：

- [`python/zrt/graph/main.py`](D:/workspace/claude/modeling/python/zrt/graph/main.py)
- [`python/zrt/graph/dispatch.py`](D:/workspace/claude/modeling/python/zrt/graph/dispatch.py)
- [`python/zrt/graph/tracker.py`](D:/workspace/claude/modeling/python/zrt/graph/tracker.py)

完成度：`已实现`

判断：

- 这是项目当前最成熟的能力之一。
- `prefill/decode` 双阶段抓图已经落地。
- 抓图结果还能进一步转成 `OpGraph`。

#### 3.3 显存模型

文档目标：

- 独立 `MemoryModel`
- 输出 `MemoryBudget`
- 在搜索和端到端运行前做快速可行性剪枝

当前实现：

- 无 `python/zrt/memory`
- 无 `MemoryModel`
- 无 `MemoryBudget`

完成度：`缺失`

判断：

- 这是当前最关键的架构缺口之一。
- 它直接阻塞 `Application Layer` 中的 `predict/search` 形态。

#### 3.4 图变换

当前实现：

- [`python/zrt/transform/pipeline.py`](D:/workspace/claude/modeling/python/zrt/transform/pipeline.py)
- [`python/zrt/transform/parallel/tensor_parallel.py`](D:/workspace/claude/modeling/python/zrt/transform/parallel/tensor_parallel.py)
- [`python/zrt/transform/parallel/expert_parallel.py`](D:/workspace/claude/modeling/python/zrt/transform/parallel/expert_parallel.py)
- [`python/zrt/transform/parallel/comm_inserter.py`](D:/workspace/claude/modeling/python/zrt/transform/parallel/comm_inserter.py)
- [`python/zrt/transform/fusion/pass_.py`](D:/workspace/claude/modeling/python/zrt/transform/fusion/pass_.py)
- [`python/zrt/transform/optim/passes.py`](D:/workspace/claude/modeling/python/zrt/transform/optim/passes.py)
- [`python/zrt/transform/analysis/passes.py`](D:/workspace/claude/modeling/python/zrt/transform/analysis/passes.py)

完成度：`部分实现`

已具备：

- `TransformPipeline`
- `TensorParallelPass`
- `ExpertParallelPass`
- `CommInserterPass`
- `FusionPass`
- `FlopsPass`
- `RooflinePass`
- `StreamAssignPass`

缺口：

- `SequenceParallelPass` 未实现
- `PipelineParallelPass` 未实现
- `EPLBPass` 是空实现
- `MTPPass` 是空实现
- `SharedExpertPass` 只是轻量标注
- 融合规则没有真正从 `SoftwareStack` 驱动

判断：

- Stage 顺序和主线是对的。
- Stage 1 与 Stage 3 还没有达到文档完整状态。

#### 3.5 算子仿真器 Hub

当前实现：

- [`python/zrt/simulator/hub.py`](D:/workspace/claude/modeling/python/zrt/simulator/hub.py)
- [`python/zrt/simulator/base.py`](D:/workspace/claude/modeling/python/zrt/simulator/base.py)
- [`python/zrt/simulator/cache.py`](D:/workspace/claude/modeling/python/zrt/simulator/cache.py)
- [`python/zrt/simulator/result.py`](D:/workspace/claude/modeling/python/zrt/simulator/result.py)
- [`python/zrt/simulator/backends/roofline.py`](D:/workspace/claude/modeling/python/zrt/simulator/backends/roofline.py)

完成度：`部分实现`

已具备：

- `SimulatorHub`
- 内容哈希缓存
- `SimResult`
- `RooflineSimulator`

缺口：

- `profile_db` 后端缺失
- `regression` 后端缺失
- `tiling_sim` 后端缺失
- `custom backend` 注册能力未形成独立对外接口

判断：

- 已达到文档 Phase 1 水平。
- 还没有达到文档描述的“多层级后端生态”。

#### 3.6 图执行器

当前实现：

- [`python/zrt/executor/scheduler.py`](D:/workspace/claude/modeling/python/zrt/executor/scheduler.py)

完成度：`部分实现`

已具备：

- `DAGScheduler`
- `Timeline`
- 基本多流调度
- overlap 指标计算所需的核心数据

缺口：

- 没有文档中的 `stream.py`
- 没有独立 `timeline.py`
- 没有独立 `overlap.py`
- `OverlapAnalyzer` 未单独成型

判断：

- 能力上已经可用。
- 结构上仍是简化实现，不是文档中的完整模块拆分。

#### 3.7 报表生成器

当前实现分成两条线：

旧链路：

- [`python/zrt/graph/excel_writer.py`](D:/workspace/claude/modeling/python/zrt/graph/excel_writer.py)
- [`python/zrt/graph/graph_exporter.py`](D:/workspace/claude/modeling/python/zrt/graph/graph_exporter.py)

新链路：

- [`python/zrt/report/summary.py`](D:/workspace/claude/modeling/python/zrt/report/summary.py)

完成度：`部分实现`

已具备：

- Excel 输出
- JSON/ONNX 导出
- `E2ESummary`

缺口：

- 无 `report/engine.py`
- 无 `html_writer.py`
- 无 `onnx_writer.py`
- 无 `json_writer.py`
- 报表体系仍未统一到 `report/*`

判断：

- 当前是“旧 graph 导出能力 + 新 summary 能力并存”。
- 文档要求的统一报告引擎还未形成。

---

### 4. Application Layer

文档目标包含：

- `Orchestrator`
- `RunConfig`
- `ConfigSearchEngine`
- 硬件对比
- 瓶颈分析
- 统一 CLI：`zrt predict/search/compare/capture/simulate/memory/calibrate/serve`

当前实现：

- 仓库中未发现 `Orchestrator`
- 未发现 `ConfigSearchEngine`
- 未发现统一 `app/` 层
- 未发现统一 CLI 主程序

完成度：`缺失`

判断：

- 这是当前最大的产品化缺口。
- 现有仓库更像“若干可用引擎模块”，而不是一个完整应用系统。

影响：

- 用户无法通过统一入口完成“加载 -> 剪枝 -> 抓图 -> 变换 -> 仿真 -> 调度 -> 报表”的端到端流程。
- 配置搜索、硬件对比、瓶颈分析都无法以文档方式成立。

---

### 5. Extension Layer

文档目标包含：

- Serving 仿真
- 训练估算
- 校准模块
- REST API
- 更高精度仿真后端

当前实现：

- 无 serving simulator
- 无 training estimator
- 无 calibration 模块
- 无 API 服务
- 无 profile/regression/tiling 级后端

完成度：`基本缺失`

判断：

- 这一层基本尚未开始。
- 当前代码仍集中在“单次图级分析和预测”。

---

## 模块级详细盘点

| 架构目标 | 当前实现位置 | 完成度 | 现状判断 | 主要缺口 | 优先级 |
|---|---|---|---|---|---|
| `OpGraph` 统一 IR | [`python/zrt/ir`](D:/workspace/claude/modeling/python/zrt/ir) | 已实现 | 新主线中枢已成型 | 旧 `graph/*` 未完全收口 | 高 |
| `GraphHierarchy` | [`python/zrt/ir/hierarchy.py`](D:/workspace/claude/modeling/python/zrt/ir/hierarchy.py) | 已实现 | 已进入 summary 链路 | 仍依赖 scope 规范 | 中 |
| 硬件注册表 | [`python/zrt/hardware`](D:/workspace/claude/modeling/python/zrt/hardware) | 已实现 | 与文档目标基本一致 | 硬件覆盖面可继续扩展 | 中 |
| 软件栈注册表 | 无 | 缺失 | 文档有设计，代码无 | 无 `stacks/`、无 registry | 高 |
| `CommModel` | 无 | 缺失 | 有通信节点，无通信建模 | all-reduce/A2A 估算缺失 | 高 |
| `ModelManager + ModelProfile` | 最接近 [`python/zrt/graph/model_loader.py`](D:/workspace/claude/modeling/python/zrt/graph/model_loader.py) | 部分实现 | 仍偏抓图链路内部实现 | 缺统一 profile 抽象 | 高 |
| 图抓取 | [`python/zrt/graph/main.py`](D:/workspace/claude/modeling/python/zrt/graph/main.py) | 已实现 | 现阶段最成熟能力之一 | 尚未纳入统一 orchestrator | 高 |
| `MemoryModel` | 无 | 缺失 | 文档核心能力未落地 | 直接阻塞 search/predict | 很高 |
| 4-stage pipeline | [`python/zrt/transform/pipeline.py`](D:/workspace/claude/modeling/python/zrt/transform/pipeline.py) | 已实现 | 顺序与设计一致 | 与 stack/profile 联动不足 | 高 |
| TP 切分 | [`python/zrt/transform/parallel/tensor_parallel.py`](D:/workspace/claude/modeling/python/zrt/transform/parallel/tensor_parallel.py) | 部分实现 | 可用，但仍偏启发式 | 尚未深度依赖 profile/stack | 高 |
| EP 切分 | [`python/zrt/transform/parallel/expert_parallel.py`](D:/workspace/claude/modeling/python/zrt/transform/parallel/expert_parallel.py) | 部分实现 | 有基本语义 | token dispatch/combine 不完整 | 高 |
| SP | 无 | 缺失 | 文档有，代码无 | 未开始 | 中 |
| PP | 无 | 缺失 | 文档有，代码无 | 未开始 | 中 |
| CommInserter | [`python/zrt/transform/parallel/comm_inserter.py`](D:/workspace/claude/modeling/python/zrt/transform/parallel/comm_inserter.py) | 部分实现 | 多卡主线关键一环已出现 | 缺通信模型支撑 | 高 |
| FusionPass | [`python/zrt/transform/fusion/pass_.py`](D:/workspace/claude/modeling/python/zrt/transform/fusion/pass_.py) | 部分实现 | 能力较强，已有测试 | 还没改成 stack 驱动 | 很高 |
| `Quant/EPLB/SharedExpert/MTP` | [`python/zrt/transform/optim/passes.py`](D:/workspace/claude/modeling/python/zrt/transform/optim/passes.py) | 部分实现 | quant/shared_expert 只有轻量标注 | EPLB/MTP 为空实现 | 高 |
| `FLOPs/Roofline` 分析 | [`python/zrt/transform/analysis/passes.py`](D:/workspace/claude/modeling/python/zrt/transform/analysis/passes.py), [`python/zrt/simulator/backends/roofline.py`](D:/workspace/claude/modeling/python/zrt/simulator/backends/roofline.py) | 已实现 | 新主链条最完整的一段 | 覆盖面还能扩展 | 高 |
| `SimulatorHub + cache` | [`python/zrt/simulator/hub.py`](D:/workspace/claude/modeling/python/zrt/simulator/hub.py) | 已实现 | 机制已具备 | 多后端生态未形成 | 高 |
| `ProfileDB` 后端 | 无 | 缺失 | 文档后续阶段能力 | 无真实 profile 接入 | 中 |
| `Regression` 后端 | 无 | 缺失 | 文档后续阶段能力 | 未开始 | 中 |
| `TilingSim` 后端 | 无 | 缺失 | 文档后续阶段能力 | 未开始 | 低 |
| `DAGScheduler` | [`python/zrt/executor/scheduler.py`](D:/workspace/claude/modeling/python/zrt/executor/scheduler.py) | 部分实现 | 已能调度并产出时间指标 | 结构未按文档拆分 | 高 |
| 独立 overlap 分析 | 无独立模块 | 部分实现 | 指标有，但模块无 | 缺 `OverlapAnalyzer` | 中 |
| `E2ESummary` | [`python/zrt/report/summary.py`](D:/workspace/claude/modeling/python/zrt/report/summary.py) | 已实现 | 汇总口径已成型 | 尚未统一整合 memory/stack/report engine | 高 |
| 统一报表引擎 | 无 | 缺失 | 仍分裂在 `graph/*` 和 `report/*` | 无统一 `ReportEngine` | 高 |
| `Orchestrator` | 无 | 缺失 | 当前最大产品化缺口 | 没有统一编排主线 | 很高 |
| `RunConfig` | 无统一实现 | 缺失 | 入口参数体系不统一 | 无统一配置对象 | 很高 |
| `ConfigSearchEngine` | 无 | 缺失 | 文档 Phase 3 关键能力 | 受 `MemoryModel` 和 `Orchestrator` 阻塞 | 很高 |
| 硬件对比 | 无 | 缺失 | 文档 Application 层功能 | 无实现 | 中 |
| 瓶颈分析模块 | 无独立 app 层实现 | 缺失 | 目前只有 summary 局部指标 | 无面向用户的分析层 | 中 |
| 统一 CLI | 无 | 缺失 | 仍以 `python -m python.zrt.graph.main` 为主入口 | 无 `zrt predict/search/...` | 很高 |
| Serving 仿真 | 无 | 缺失 | Extension 层未开始 | 无实现 | 低 |
| 训练估算 | 无 | 缺失 | Extension 层未开始 | 无实现 | 低 |
| 校准模块 | 无 | 缺失 | Phase 4 未开始 | 无精度闭环 | 中 |
| API 服务化 | 无 | 缺失 | 文档末端能力未开始 | 无服务入口 | 低 |

---

## 与路线图的对应关系

### Phase 1：单卡理论建模

文档目标：

- IR
- capture
- hardware
- roofline
- executor
- memory
- report
- `zrt predict`

当前状态：

- `IR` 已实现
- `capture` 已实现
- `hardware` 已实现
- `roofline` 已实现
- `executor` 已实现基础版
- `report` 已部分实现
- `memory` 缺失
- 统一 `predict` 缺失

结论：

- `Phase 1` 大体完成
- 但少了 `MemoryModel` 和应用层封装

### Phase 2：融合 + 多卡并行

文档目标：

- TP/EP
- 平台融合
- 通信模型
- 多流调度
- overlap 分析
- HTML timeline

当前状态：

- TP/EP 已部分实现
- 融合已部分实现
- 通信模型缺失
- 多流调度已有基础版
- overlap 指标已有，但无独立 analyzer
- HTML timeline 缺失

结论：

- `Phase 2` 已做一半左右

### Phase 3：优化项 + 搜索

文档目标：

- quant / EPLB / shared expert / MTP
- SP / PP
- config search
- hw compare
- bottleneck analysis

当前状态：

- quant/shared expert 只有轻量实现
- EPLB/MTP 基本未实现
- SP/PP 未实现
- search / compare / bottleneck app 层未实现

结论：

- `Phase 3` 仅刚起步

### Phase 4：精度提升

文档目标：

- profile_db
- regression
- calibration

当前状态：

- 基本未开始

### Phase 5：扩展场景

文档目标：

- serving
- training
- api
- tiling simulator

当前状态：

- 基本未开始

---

## 最关键的 6 个缺口

### 1. `MemoryModel` 缺失

这是最关键的架构缺口之一。  
没有它，文档中的：

- 可行性判定
- 快速剪枝
- 搜索引擎
- `predict` 的前置检查

都无法按设计成立。

### 2. `Orchestrator` 缺失

当前仓库里有不少可用模块，但没有统一编排层。  
结果就是：

- 能力存在
- 产品形态不存在

### 3. `SoftwareStack` 缺失

文档强调“硬件 × 软件栈正交”，但当前只有硬件这一半真正成型。  
这会直接影响：

- 融合规则
- 平台算子映射
- 优化能力声明

### 4. `CommModel` 缺失

现在的多卡链路只有“通信节点”，没有“通信性能模型”。  
这会限制：

- 多卡 latency 预测可信度
- overlap 分析质量
- TP/EP 调优能力

### 5. 新旧链路尚未收口

当前同时存在：

- 旧 `graph/*` 导出与融合链
- 新 `ir/transform/simulator/report/*` 建模链

这不是坏事，但说明迁移还在中途，维护成本会持续上升。

### 6. `Application Layer` 未形成

这是从“研究原型”走向“完整系统”的最后一层。  
不补这一层，项目始终更像内部引擎集合，而不是面向使用者的完整工具。

---

## 建议的补齐顺序

如果目标是尽快对齐 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md)，建议按这个顺序推进：

1. 补 `MemoryModel`
2. 补 `Orchestrator + RunConfig`
3. 补 `SoftwareStack + generic/mindie/vllm`
4. 补 `CommModel`
5. 收口旧 `graph` 导出链到统一 `report` 体系
6. 再做 `search / compare / bottleneck`
7. 最后补 `profile_db / regression / calibration`

原因：

- 前四项能把现有“模块能力”变成真正的端到端系统。
- 后三项更偏精度提升和产品增强，应该建立在主干成型之后。

---

## 最终判断

一句话总结：

当前项目已经具备了 V2 架构的核心骨架，尤其是 `OpGraph + Transform + Roofline + Schedule + Summary` 这条主线；但它还不是 [`ARCHITECTURE.md`](D:/workspace/claude/modeling/ARCHITECTURE.md) 定义的完整系统，离“统一编排、可搜索、可对比、可校准”的目标仍有明显距离。

从工程状态看，它更像：

- 不是早期 PoC
- 也还不是完整产品
- 而是一个“主干已成型、应用层待补齐”的中间阶段架构

