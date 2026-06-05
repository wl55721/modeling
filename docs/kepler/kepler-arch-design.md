# Kepler 架构设计文档

## 1. 系统总览

Kepler 是一个 LLM 推理资源建模工具，采用前后端分离的单体架构，通过 Docker 单容器部署。

```
┌─────────────────────────────────────────────────────────────┐
│                        Browser                              │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────────┐  │
│  │ Step 1   │ │ Step 2   │ │ Step 3   │ │ Step 4       │  │
│  │ 模型编辑器│ │ 负载配置  │ │ 硬件配置  │ │ 仿真结果      │  │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘ └──────┬───────┘  │
│       │             │            │               │          │
│       └─────────────┴────────────┴───────────────┘          │
│                          │  REST API                        │
└──────────────────────────┼──────────────────────────────────┘
                           │
┌──────────────────────────┼──────────────────────────────────┐
│                    Docker Container                          │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────────────┐  │
│  │                 FastAPI (uvicorn)                       │  │
│  │                                                        │  │
│  │  ┌──────────┐  ┌──────────┐  ┌──────────────────┐    │  │
│  │  │ Routes   │  │ Routes   │  │ Routes           │    │  │
│  │  │ /simulate│  │ /optimize│  │ /library/*       │    │  │
│  │  └────┬─────┘  └────┬─────┘  └────────┬─────────┘    │  │
│  │       │              │                 │               │  │
│  │  ┌────┴──────────────┴─────────────────┴──────────┐   │  │
│  │  │              Services Layer                     │   │  │
│  │  │  SimulationService  │  OptimizerService         │   │  │
│  │  └──────────────────────┬──────────────────────────┘   │  │
│  │                         │                               │  │
│  │  ┌──────────────────────┴──────────────────────────┐   │  │
│  │  │              Engine Layer                         │   │  │
│  │  │  Executor (NetworkX DAG)  │  Operator Classes     │   │  │
│  │  │  Chip Configs             │  Model Config         │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  │                         │                               │  │
│  │  ┌──────────────────────┴──────────────────────────┐   │  │
│  │  │              Data Layer                           │   │  │
│  │  │  operators/*.json  │  modules/*.json              │   │  │
│  │  │  hardwares/*.json  │  models/*.json               │   │  │
│  │  └──────────────────────────────────────────────────┘   │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
```

---

## 2. 前端架构

### 2.1 组件树

```
App
├── Header (步骤导航 1-2-3-4)
├── Step 1: 模型编辑器
│   ├── ResizablePanel (左)
│   │   └── VerticalSplit
│   │       ├── OperatorPanel        ← 算子库
│   │       └── ModulePanel          ← 模块库
│   ├── ReactFlowProvider
│   │   └── ModelCanvas              ← 画布
│   └── ResizablePanel (右)
│       └── VerticalSplit
│           ├── ModelConfig          ← 模型配置
│           └── OperatorDetail       ← 算子详情
├── Step 2: 负载配置
│   └── WorkloadConfigPanel
├── Step 3: 硬件配置
│   └── HardwarePanel
└── Step 4: 结果
    ├── ResultsPanel                  ← 手动仿真结果
    └── OptimizeResults               ← 自动寻优结果
```

### 2.2 状态管理架构

```
┌─────────────────────────────────────────────────┐
│                   Zustand Stores                 │
│                                                  │
│  ┌─────────────────────┐  ┌───────────────────┐ │
│  │ useModelStore        │  │ useInferenceStore  │ │
│  │ (stores/model.ts)   │  │ (WorkloadConfig)   │ │
│  │                     │  │                    │ │
│  │ • nodes[]           │  │ • phase            │ │
│  │ • layers[]          │  │ • batch_size       │ │
│  │ • edges[]           │  │ • input_length     │ │
│  │ • ranks[]           │  │ • output_length    │ │
│  │ • operators[]       │  │ • num_mtp_tokens   │ │
│  │ • customOperators[] │  │ • tp/dp/pp/ep/...  │ │
│  │ • customModules[]   │  │ • quant_*          │ │
│  │ • selectedOperator  │  │ • optimizeMode     │ │
│  │ • selectedModule    │  │ • targetTpotMs     │ │
│  │                     │  │                    │ │
│  │ persist: localStorage│  │ no persist         │ │
│  └─────────┬───────────┘  └─────────┬─────────┘ │
│            │                        │            │
│  ┌─────────┴───────────┐            │            │
│  │ getHardwareConfigs()│            │            │
│  │ (stores/hardware.ts)│            │            │
│  │ • chip_name         │            │            │
│  │ • spec_memory_size  │            │            │
│  │ • spec_cube_fp16    │            │            │
│  │ • ...               │            │            │
│  │ no persist          │            │            │
│  └─────────────────────┘            │            │
│            │                        │            │
│            └────────┬───────────────┘            │
│                     │                            │
│                     ▼                            │
│              App.handleRun()                     │
│           exportModel() + getHardwareConfigs()   │
│           + useInferenceStore.getState()         │
│                     │                            │
│                     ▼                            │
│           runSimulate() | runOptimize()          │
└─────────────────────────────────────────────────┘
```

三个 Store 的职责边界：

| Store | 职责 | 持久化 |
|-------|------|--------|
| `useModelStore` | 模型结构（节点/层/连线/算子定义/自定义算子/自定义模块/多Rank） | localStorage（自定义算子/模块） |
| `useInferenceStore` | 工作负载参数（推理/并行/量化/寻优模式） | 无 |
| `hardware.ts` | 硬件配置列表（多芯片） | 无 |

### 2.3 数据流：手动仿真

```
Step 1                     Step 2                  Step 3            Step 4
───────                    ───────                 ───────           ───────
useModelStore              useInferenceStore        hardware.ts
    │                           │                       │
    │ exportModel()             │ getState()            │ getHardwareConfigs()
    │ → model_json              │ → params              │ → hardwares[]
    │                           │                       │
    └───────────────┬───────────┴───────────────────────┘
                    │
                    ▼
              POST /api/simulate
              {
                model_json,
                hf_config_json?,
                workloads: [{ request, parallel, quant }],
                hardwares: [{ name, config }]
              }
                    │
                    ▼
              SimulationService
                    │
              ┌─────┴─────┐
              │ per hw loop│
              │  deep copy │
              │  context   │
              │  execute   │
              │  collect   │
              └─────┬─────┘
                    │
                    ▼
              SimulateMultiResponse
              { results: [{ hardware_name, result: {...} }] }
                    │
                    ▼
              setResults() → ResultsPanel
```

### 2.4 数据流：自动寻优

```
Step 2 (auto mode)                 Step 3                Step 4
─────────────────                  ───────               ───────
useInferenceStore                  hardware.ts
    │                                  │
    │ targetTpotMs,                    │
    │ minWorldSize, maxWorldSize        │
    │                                  │
    └──────────────┬───────────────────┘
                   │
                   ▼
             POST /api/optimize
             {
               model_json,
               workload: { request, optimize: {...}, quant },
               hardwares: [...]
             }
                   │
                   ▼
             OptimizerService
                   │
             ┌─────┴──────────────────────┐
             │ for ws in 1,2,4,8,...:     │
             │   for tp in factors(ws):   │
             │     for sub_tp in 8 combos:│
             │       simulate() ──────────┤
             │       if meets_target:     │
             │         early stop         │
             └─────┬──────────────────────┘
                   │
                   ▼
             OptimizeResponse
             { optimal, candidates[], search_summary }
                   │
                   ▼
             setOptimizeResult() → OptimizeResults
```

---

## 3. 后端架构

### 3.1 分层结构

```
┌─────────────────────────────────────────┐
│          Routes Layer (薄层)              │
│  参数校验 → 委托 Service → 返回响应        │
│  simulate.py / optimize.py / library.py  │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│          Services Layer                  │
│  SimulationService: 仿真编排 + 结果聚合   │
│  OptimizerService: 策略搜索 + 早停        │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│          Engine Layer                    │
│  Executor: NetworkX 拓扑排序 → 逐算子执行 │
│  Layers/: 20+ 算子类（__call__）         │
│  Chips/: NVIDIA / Ascend 芯片参数        │
│  model_config.py: JSON → 内部配置解析    │
└────────────────┬────────────────────────┘
                 │
┌────────────────┴────────────────────────┐
│          Data Layer                      │
│  JSON 文件: operators / modules          │
│           hardwares / models             │
└─────────────────────────────────────────┘
```

### 3.2 Engine 核心：算子执行模型

```
             ┌──────────────────┐
             │   Model JSON      │
             │  operators[]      │
             │  edges[]          │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  NetworkX DAG     │
             │  拓扑排序          │
             └────────┬─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  Context Dict     │  ← 共享状态，包含所有张量/参数
             │  {                │
             │    "bsz": 1,      │
             │    "seq_len": 2048│
             │    "hidden_dim":  │
             │      4096,        │
             │    "hidden_states":│
             │      TensorSpec,  │
             │    ...            │
             │  }                │
             └────────┬─────────┘
                      │
         ┌────────────┼────────────┐
         ▼            ▼            ▼
    ┌─────────┐ ┌─────────┐ ┌─────────┐
    │ Op 1    │ │ Op 2    │ │ Op 3    │
    │ __call__│ │ __call__│ │ __call__│
    │         │ │         │ │         │
    │ 读 cfg  │ │ 读 cfg  │ │ 读 cfg  │
    │ 计算     │ │ 计算     │ │ 计算     │
    │ 写 cfg  │ │ 写 cfg  │ │ 写 cfg  │
    └─────────┘ └─────────┘ └─────────┘
                      │
                      ▼
             ┌──────────────────┐
             │  OpExecuteResult │
             │  • compute_us    │
             │  • mem_us        │
             │  • comm_us       │
             │  • noise_us      │
             └──────────────────┘
```

关键设计点：
- **Context dict 是算子间唯一的通信机制**：每个算子从 `self.cfg` 读取输入，计算结果写回 `self.cfg`
- **Deep copy 隔离**：每个硬件迭代前 `copy.deepcopy(context)`，防止不同硬件配置的执行互相污染
- **算子无状态**：算子本身不保存执行结果，所有状态在 context dict 中

### 3.3 算子基类体系

```
OperatorBase                    # 定义 __call__ 接口 + cfg 引用
├── OpCubeBase                 # 矩阵乘法等 Cube 单元密集算子
│   ├── MatMul
│   ├── Linear
│   ├── FlashAttention
│   └── ...
├── OpVectorBase               # 逐元素 Vector 单元算子
│   ├── RMSNorm
│   ├── SwiGlu
│   ├── TorchAdd / TorchMul
│   └── ...
├── OpMixBase                  # 同时使用 Cube + Vector
│   └── ...
└── OpCommBase                 # 通信算子（使用带宽模型而非计算单元）
    ├── AllReduce
    └── AllGather
```

每个基类封装了对应计算单元的成本模型（Cube TFLOPs / Vector throughput / Memory bandwidth），子类只需描述计算量和数据搬运量。

### 3.4 芯片参数模型

```
ChipConfig (chips/config.py)
├── NvidiaConfig (nvidia.py)    # NVIDIA GPU 参数
│   • cube_core_cnt             # Tensor Core 数量
│   • vector_core_cnt           # CUDA Core 数量
│   • cube_freq                 # Tensor Core 频率
│   • spec_bw_memory            # 显存带宽
│   • spec_comm_intra/inter     # 通信带宽
│   └── ...
└── AscendConfig (ascend.py)    # Ascend NPU 参数
    • 类似字段，不同架构参数
```

每种芯片定义计算单元的吞吐能力和通信带宽，算子执行时查表获取硬件能力。

---

## 4. API 设计

### 4.1 端点总览

| 方法 | 路径 | 请求体 | 响应体 | 用途 |
|------|------|--------|--------|------|
| POST | `/api/simulate` | `SimulateRequest` | `SimulateMultiResponse` | 手动仿真 |
| POST | `/api/optimize` | `OptimizeRequest` | `OptimizeResponse` | 自动寻优 |
| GET | `/api/library/operators` | — | `string[]` | 算子名称列表 |
| GET | `/api/library/operators/{name}` | — | `OperatorDef` | 算子定义 |
| GET | `/api/library/modules` | — | `string[]` | 模块名称列表 |
| GET | `/api/library/modules/{name}` | — | `OperatorDef[]` | 模块算子列表 |
| GET | `/api/library/models` | — | `string[]` | 模型名称列表 |
| GET | `/api/library/models/{name}` | — | `ModelStructure` | 模型结构 |
| POST | `/api/library/models` | body | — | 保存模型 |
| DELETE | `/api/library/models/{name}` | — | — | 删除模型 |
| GET | `/api/library/hardwares` | — | `HardwareListItem[]` | 硬件名称列表 |
| GET | `/api/library/hardwares/{name}` | — | `HardwareSpec` | 硬件规格 |
| POST | `/api/library/hardwares` | body | — | 保存硬件 |
| DELETE | `/api/library/hardwares/{name}` | — | — | 删除硬件 |
| GET | `/api/library/hf_configs` | — | `string[]` | HF 配置列表 |
| GET | `/api/library/hf_configs/{name}` | — | `object` | HF 配置 |

### 4.2 核心 Schema（SimulateRequest）

```
SimulateRequest
├── model_name?: string            # 内置模型名称
├── model_json?: object            # 自定义模型 JSON（前端 exportModel 输出）
├── hf_config_json?: object        # HuggingFace 模型配置
├── workloads: WorkloadEntry[]
│   ├── request: RequestConfig     # phase, batch_size, input/output_length, mtp, prefix_hit_ratio
│   ├── parallel: ParallelConfig   # world_size, tp/dp/pp/ep/cp, 专用 TP
│   └── quant: QuantConfig         # 6 种量化配置
└── hardwares: HardwareEntry[]
    ├── name: string
    └── config: string | object    # 硬件配置（名称引用或内联）
```

### 4.3 核心 Schema（SimulateMultiResponse）

```
SimulateMultiResponse
└── results: SimulateSingleResult[]
    └── hardware_name: string
    └── result: SimulateResponse
        ├── ttot_ms, tpot_ms, tps, qps            # 高层指标
        ├── prefill_latency_ms, decode_latency_per_token_ms
        ├── peak_mem_gb, oom, strategy
        ├── operators: OperatorResult[]             # 每个算子的执行详情
        ├── op_statistics: OperatorStatistics[]     # 按算子聚合的统计
        └── ranks: RankResult[]                     # 每个 rank 的汇总
            ├── total_cost_ms, peak_mem_gb, oom
            ├── param_bytes, io_bytes, num_ops
            └── layers: LayerResultPerRank[]        # rank 内每层详情
                ├── layer_cost_ns, repeat
                ├── param_bytes, io_bytes
                ├── start_time_ns, end_time_ns
                └── op_ids[]
```

---

## 5. 模型编辑器：多 Rank 架构

### 5.1 概念模型

```
Model
├── Ranks: [0, 1, 2, ...]
│   └── 每个 Rank 拥有独立的算子副本
│
├── Layers: [{ id, kind, layerIdx, repeat, rankOps: { rank → opIdx[] } }]
│   ├── kind: 'regular'  → 常规层，layerIdx 0..N
│   └── kind: 'mtp'      → MTP 层，layerIdx 980+，最多 1 个
│
├── Top Globals: 层前全局算子（layerIdx = -1），per-rank
├── Bottom Globals: 层后全局算子（layerIdx = 900），per-rank
│
└── Nodes: 按 rank → (top globals → regular layers → bottom globals → mtp layers) 排列
```

### 5.2 画布节点排序规则

```
start
  ├── Rank 0: top globals → Layer_0 ops → Layer_1 ops → ... → bottom globals → mtp ops
  ├── Rank 1: top globals → Layer_0 ops → Layer_1 ops → ... → bottom globals → mtp ops
  └── ...
end
```

### 5.3 自动连线算法

`computeAutoEdgePairs()` 位于 `stores/model.ts`，在以下两种情况下生成隐式边：

1. **同 rank 相邻节点**：基于前驱集合判断串行/并行
   - 两者前驱集合相同 → 并行，不连线
   - prev 在 curr 的前驱集合中 → 串行，自动连线
2. **跨 rank**：start → 每个 rank 第一个节点；每个 rank 最后一个节点 → end

手动边优先于自动边；`disconnected` Set 记录被用户显式断开的自动边。

---

## 6. 仿真结果：层时间计算

### 6.1 MTP 层特殊处理（`_fill_mtp_results`）

```
对每个 MTP layer:
  layer_cost_ns = end_time_ns - start_time_ns
  - 减去 Compressor/Indexer 算子的耗时
  + attention 算子的额外计算（基于 compress_ratio）
  param_bytes / io_bytes = 继承自原始层

MTP 层总体成本:
  mtp_cost_ns = Σ layer_cost_ns (repeat 次，每次对 num_mtp_tokens 个 token)
```

### 6.2 Rank 总耗时计算

```
rank.total_cost_ms = Σ(layer.layer_cost_ns × layer.repeat) / 1_000_000
```

每个层的 `layer_cost_ns` 是单次调用的原始耗时，前端展示层详情时也显示原始值，summary 行使用 × repeat 的总和。

### 6.3 层内算子时间对齐

`_adjust_layer_operator_start_and_end_time`：将层内所有算子的开始/结束时间整体平移，保持算子间的相对时间关系（包括多流并发和跨 rank 依赖）不变。

---

## 7. 自动寻优算法

### 7.1 搜索空间

```
变量:
  world_size ∈ {1, 2, 4, 8, 16, 32, 64, 128, 256, 512}
  tp_size ∈ factors(world_size)
  dp_size = world_size / tp_size
  embed_tp_size  ∈ {1, tp_size}            # 粗粒度
  o_tp_size      ∈ {1, tp_size}
  lmhead_tp_size ∈ {1, tp_size}

约束:
  pp_size = 1, cp_size = 1
  ep_size = world_size
  external_shared_expert_rank_size = 0
```

### 7.2 搜索策略

```
for world_size in [1, 2, 4, ...] (递增):
  for tp in factors(world_size):
    for (embed_tp, o_tp, lmhead_tp) in {1,tp}^3 (8 combos):
      result = simulate(ws, tp, dp, eTP, oTP, lmTP)
      if result.meets_target && already found one at this ws:
        → 同 ws 内剪枝（子候选），不 break 整个 ws 循环
  
  if ∃ c.meets_target && c.world_size == world_size:
    break  # 早停：最少 GPU 原则
```

### 7.3 排序规则

```
1. is_oom        → OOM 排最后
2. meets_target  → 满足目标的优先
3. world_size    → GPU 少优先
4. -dp_size      → DP 大优先（同一 GPU 下通信更少）
5. tpot_ms       → 延迟低优先
```

---

## 8. 关键架构决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 状态管理 | Zustand（非 Redux） | 简单 API，无 boilerplate，内置 selector 模式 |
| 画布 | React Flow | 成熟的节点编辑器框架，支持 DnD/缩放/自定义节点 |
| 样式方案 | 纯 CSS Variables（非 Tailwind/CSS-in-JS） | 零运行时开销，设计系统通过 CSS 变量贯彻 |
| 数据格式 | JSON 文件（非 SQLite） | 算子/模块/硬件数据量小，文件编辑友好，Git 可追踪 |
| 后端图计算 | NetworkX | 拓扑排序能力，Python 标准生态 |
| 自动寻优 | 同步调用（非 SSE 流式） | Phase 1 搜索量可控（几十到上百次 simulate），简化实现 |
| Context 隔离 | deep copy | 防止硬件迭代间算子执行污染共享 context dict |
| 自定义算子/模块 | localStorage 持久化 | 纯前端能力，无需后端存储 |
| MTP 层限制 | 最多 1 个 | 模型架构合理性约束，且满足当前搜索空间需求 |
| 多 Rank 支持 | per-layer rankOps 结构 | 每层可独立管理不同 rank 的算子列表，灵活且向后兼容 |

---

## 9. 扩展点

- **SSE 流式寻优**：当前同步返回，搜索量大时可改为 SSE 逐候选推送
- **多目标帕累托**：在时延目标外增加吞吐量目标，展示帕累托前沿
- **算子缓存**：对相同模型+硬件的算子评估结果做缓存，避免重复计算
- **HuggingFace 模型导入**：通过 `hf_config_json` 字段已支持，可进一步自动化
- **更多芯片**：扩展 `chips/` 目录添加新芯片架构参数
