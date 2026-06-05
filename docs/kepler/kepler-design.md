# Kepler — LLM Inference Resource Modeling Tool

## 项目概述

Kepler 是一个 LLM 推理资源建模工具。命名寓意：正如开普勒用数学公式预测行星运动、替代了每次实际观测，Kepler 用数学公式替代昂贵的真实推理，快速预测 LLM 的 GPU 资源消耗。

### 核心定位

| 维度 | 决策 |
|------|------|
| **目标用户** | 内部研发团队 |
| **核心方法** | 算子级 cost model + 拓扑图执行仿真 |
| **交付形式** | Web 服务（FastAPI + React 前端），单容器部署 |
| **推理引擎依赖** | 无（Framework-agnostic，纯理论建模） |

---

## 技术栈

| 层 | 技术 |
|---|------|
| 前端框架 | React 19 + TypeScript |
| 流程图 | @xyflow/react（React Flow） |
| 状态管理 | Zustand（含 localStorage 持久化） |
| 样式方案 | CSS 自定义属性（CSS Variables） |
| 构建工具 | Vite |
| 后端框架 | FastAPI (async) |
| 图计算 | NetworkX（拓扑排序） |
| 数据存储 | JSON 文件（算子/模块/硬件/模型） |
| 部署 | Docker Compose 单容器 |

---

## 项目目录结构

```
kepler/
├── README.md
├── docker-compose.yml
├── Dockerfile
│
├── backend/
│   ├── pyproject.toml
│   ├── kepler/
│   │   ├── __init__.py
│   │   ├── engine/                         # 核心计算引擎
│   │   │   ├── executor.py                 # 拓扑图执行器（NetworkX）
│   │   │   ├── model_config.py             # 模型配置解析
│   │   │   ├── layers/                     # 算子实现（20+ 类）
│   │   │   │   ├── base.py                 # OperatorBase / OpCubeBase / OpVectorBase / OpMixBase
│   │   │   │   ├── attention.py            # FlashAttention / PageAttention / SparseAttention 等
│   │   │   │   ├── embedding.py            # Embedding
│   │   │   │   ├── linear.py               # MatMul / Linear / ColumnParallelLinear / RowParallelLinear / GroupMatMul
│   │   │   │   ├── swiglu.py               # SwiGlu / SwiGluQuant / Sigmoid
│   │   │   │   ├── moe.py                  # MoEGate / MoEDispatch / MoECombine / LightningIndexer
│   │   │   │   ├── rms_norm.py             # RMSNorm / AddRMSNorm / GemmaRMSNorm / RMSNormGated / RMSNormQuant
│   │   │   │   ├── compressor.py           # Compressor / IndexCompressorEpilog / KVCompressorEpilog
│   │   │   │   ├── mhc.py                  # MHCPre / MHCPost / MHCHead（Multi-Head Compressor）
│   │   │   │   ├── communication.py        # AllReduce / AllGather
│   │   │   │   ├── position.py             # RopeComplex / RopeInterLeave
│   │   │   │   ├── quant.py                # DynamicQuant
│   │   │   │   ├── flow.py                 # START / END
│   │   │   │   └── torch_ops.py            # TorchMul / TorchAdd / TorchSoftmax 等逐元素算子
│   │   │   └── chips/                      # 芯片特定参数
│   │   │       ├── config.py               # 芯片配置基类
│   │   │       ├── nvidia.py               # NVIDIA GPU 参数
│   │   │       └── ascend.py               # Ascend NPU 参数
│   │   ├── web/                            # FastAPI Web 层
│   │   │   ├── __init__.py
│   │   │   ├── __main__.py
│   │   │   ├── app.py                      # FastAPI 实例、CORS、路由注册
│   │   │   ├── schemas.py                  # Pydantic request/response 模型
│   │   │   ├── routes/
│   │   │   │   ├── simulate.py             # POST /api/simulate
│   │   │   │   ├── optimize.py             # POST /api/optimize
│   │   │   │   └── library.py              # 算子/模块/模型/硬件/HF配置 CRUD API
│   │   │   └── services/
│   │   │       ├── simulation.py           # SimulationService：仿真执行 + MTP 结果填充
│   │   │       └── optimizer.py            # OptimizerService：并行策略自动寻优
│   │   └── utils/
│   │       └── log.py
│   ├── data/                               # JSON 数据文件
│   │   ├── operators/                      # 31 个内置算子定义（name/inputs/params/outputs/compute_flops/module）
│   │   ├── modules/                        # 内置模块定义（算子分组）
│   │   ├── hardwares/                      # 内置芯片规格
│   │   └── models/                         # 内置模型结构
│   └── tests/
│
├── frontend/
│   ├── package.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.tsx                          # 应用入口，四步式工作流导航 + 布局
│       ├── main.tsx                         # React DOM 挂载
│       ├── index.css                        # 全局样式（CSS 自定义属性体系）
│       ├── components/
│       │   ├── ModelEditor/                 # 手搓模型编辑器（Step 1）
│       │   │   ├── OperatorPanel.tsx        # 算子库面板（搜索/分类/拖拽/自定义算子 CRUD）
│       │   │   ├── ModulePanel.tsx          # 模块库面板（内置模块/自定义模块/拖拽批量添加）
│       │   │   ├── ModelCanvas.tsx          # React Flow 画布（DnD/自动布局/连线管理/ArrowPad 导航）
│       │   │   ├── LayerNode.tsx            # 自定义层节点渲染
│       │   │   ├── ModelConfig.tsx          # 模型配置面板（层管理/tab切换/导入导出/Rank管理）
│       │   │   ├── OperatorDetail.tsx       # 算子详情编辑（Module/Inputs/Params/Outputs/FLOPs）
│       │   │   ├── ModuleSelect.tsx         # Module 下拉选择器
│       │   │   ├── CustomOperatorDialog.tsx # 自定义算子创建/编辑弹窗
│       │   │   └── CustomModuleDialog.tsx   # 自定义模块创建/编辑弹窗
│       │   ├── ConfigForm/                  # 配置表单（Step 2-3）
│       │   │   ├── WorkloadConfigPanel.tsx  # 推理参数 + 并行策略 + 量化配置 + 自动寻优模式
│       │   │   └── HardwarePanel.tsx        # 硬件规格配置
│       │   ├── Results/                     # 仿真结果展示（Step 4）
│       │   │   ├── ResultsPanel.tsx         # 结果表格 + 图表 + 算子统计 + Rank 详情 + 层详情
│       │   │   └── OptimizeResults.tsx      # 自动寻优结果展示（最优策略 + 候选对比表）
│       │   └── ResizeHandle.tsx             # 可拖拽调整面板宽度
│       ├── stores/                          # Zustand 状态管理
│       │   ├── model.ts                     # 模型结构/节点/层/连线/自定义算子/自定义模块/多 Rank
│       │   └── hardware.ts                  # 硬件配置
│       ├── api/                             # API 调用层
│       │   └── library.ts                   # 算子/模块/模型/硬件/仿真/寻优 API
│       ├── types/                           # TypeScript 类型定义
│       │   ├── model.ts                     # OperatorDef / OpNodeData / LayerConfig / ModuleDef / HardwareSpec
│       │   └── results.ts                   # 仿真结果类型（含 Rank/Layer 详情）
│       ├── constants/
│       │   └── operators.ts                 # DTYPES / MODULE_GROUPS 共享常量
│       └── utils/
│           ├── classnames.ts                # cn() class 名合并
│           └── file.ts                      # openFileDialog() 文件选择
│
└── docs/
    ├── kepler-design.md                     # 本文档
    ├── kepler-auto-optimizing-design.md     # 自动寻优设计文档
    └── kepler-cost-calc-desgin.md           # 成本计算设计文档
```

---

## 核心引擎设计

### 算子架构

所有算子继承自基类体系：

```
OperatorBase              # 最底层基类
├── OpCubeBase            # Cube 类算子（矩阵乘等密集计算）
├── OpVectorBase          # Vector 类算子（逐元素操作）
├── OpMixBase             # 混合类算子（同时包含 Cube + Vector）
└── OpCommBase            # 通信类算子（AllReduce / AllGather）
```

每个算子实现 `__call__` 方法，通过 `self.cfg`（上下文字典）获取输入/参数，写入输出。算子通过 `dynamic_update_b_s` 方法动态更新 batch size 和 sequence length 相关的张量形状。

核心执行流程（`executor.py`）：
1. 接收模型 JSON（operators + edges）
2. 通过 NetworkX 拓扑排序确定执行顺序
3. 逐算子实例化并执行，传递 context dict
4. 收集每个算子的执行结果（耗时、显存、FLOPs 等）

### 算子清单（20+ 类）

| 分类 | 算子 | 说明 |
|------|------|------|
| Attention | FlashAttention, PageAttention, ScaledDotProductAttn, SparseAttentionSharedKV, SparseFlashAttention | 各类注意力计算 |
| Linear | MatMul, Linear, ColumnParallelLinear, RowParallelLinear, GroupMatMul, ColumnParallelLinearQuant | 矩阵乘法及并行变体 |
| Normalization | RMSNorm, AddRMSNorm, AddRMSNormQuant, GemmaRMSNorm, RMSNormGated, RMSNormQuant | 各类归一化 |
| Activation | SwiGlu, SwiGluQuant, Sigmoid | 激活函数 |
| MoE | MoEGateTopK, MoEDispatch, MoECombine, MoEGate, MoETopK, MoEGateHashTopK, LightningIndexer | 混合专家 |
| Embedding | Embedding | 词嵌入 |
| Position | RopeComplex, RopeInterLeave | 位置编码 |
| Communication | AllReduce, AllGather | 分布式通信 |
| Compressor | Compressor, IndexCompressorEpilog, KVCompressorEpilog | KV Cache 压缩 |
| MHC | MHCPre, MHCPost, MHCHead | Multi-Head Compressor |
| MLA | MLAPrologV4, MLAEpilogV4, IndexPrologV4 | MLA 架构专用 |
| Quant | DynamicQuant | 动态量化 |
| Flow | START, END | 模型出入口 |
| TorchOps | TorchMul, TorchAdd, TorchSoftmax, TorchSum, TorchSin, TorchCos, TorchCumsum, TorchMm, TorchSort | 逐元素算子 |

### 仿真服务（SimulationService）

位于 `backend/kepler/web/services/simulation.py`：

- **输入**：模型结构 JSON + 工作负载配置 + 硬件配置
- **执行**：对每个硬件配置，deep copy context 后运行一次完整仿真（循环内 copy 防止硬件间污染）
- **输出**：每硬件一组仿真结果（时延、吞吐、显存、算子统计、Rank 统计、层统计）

关键方法：
- `simulate()` — 主入口，遍历硬件列表调用 `_simulate_single`
- `_fill_rank_results()` — 按 rank 聚合算子结果，计算 total_cost_ms（sum of layer_cost_ns × repeat）
- `_fill_mtp_results()` — 计算 MTP 层成本（含 compressor/indexer 扣除、attention 压缩比调整）
- `_adjust_layer_operator_start_and_end_time()` — 层内算子时间整体平移

### 自动寻优（OptimizerService）

位于 `backend/kepler/web/services/optimizer.py`：

- **搜索空间**：world_size（2 的幂） × tp_size（world_size 因子） × 专用 TP（embed/o/lmhead）
- **评估函数**：复用 SimulationService.simulate()
- **策略**：world_size 从 1 递增搜索，找到满足 TPOT 目标的策略后早停
- **排序**：OOM 淘汰 → 满足目标优先 → GPU 最少 → Dp 大优先 → 延迟最低
- **输出**：最优策略 + 候选列表 + 搜索摘要

---

## Web API 设计

### 仿真运行

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/api/simulate` | 提交模型 + 工作负载 + 硬件，返回仿真结果 |

### 自动寻优

| 方法 | 路径 | 功能 |
|------|------|------|
| `POST` | `/api/optimize` | 提交模型 + 工作负载 + 硬件 + 目标 TPOT，返回最优并行策略 |

### 算子库管理

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/library/operators` | 列出所有内置算子名称 |
| `GET` | `/api/library/operators/{name}` | 获取指定算子定义 JSON |

### 模块库管理

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/library/modules` | 列出所有内置模块名称 |
| `GET` | `/api/library/modules/{name}` | 获取指定模块的算子列表 |

### 模型结构管理

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/library/models` | 列出所有内置模型名称 |
| `GET` | `/api/library/models/{name}` | 获取指定模型结构 JSON |
| `POST` | `/api/library/models` | 添加/导入新模型结构 |
| `DELETE` | `/api/library/models/{name}` | 删除模型结构 |

### 硬件规格管理

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/library/hardwares` | 列出所有内置芯片名称 |
| `GET` | `/api/library/hardwares/{name}` | 获取指定硬件规格 JSON |
| `POST` | `/api/library/hardwares` | 添加/导入新硬件规格 |
| `DELETE` | `/api/library/hardwares/{name}` | 删除硬件规格 |

### HF 配置管理

| 方法 | 路径 | 功能 |
|------|------|------|
| `GET` | `/api/library/hf_configs` | 列出所有 HF 配置名称 |
| `GET` | `/api/library/hf_configs/{name}` | 获取指定 HF 配置 JSON |

---

## 数据格式

### 硬件规格 JSON (`data/hardwares/A100-80GB.json`)

```json
{
  "name": "NVIDIA A100 80GB SXM",
  "chip": "GA100",
  "memory_gb": 80,
  "memory_bandwidth_gb_s": 2039,
  "fp16_tflops": 312,
  "bf16_tflops": 312,
  "fp8_tflops": 624,
  "int8_tops": 624,
  "nvlink_bandwidth_gb_s": 600,
  "pcie_bandwidth_gb_s": 64,
  "tdp_w": 400,
  "interconnect": "NVLink 3.0",
  "gpus_per_node": 8,
  "inter_node_bandwidth_gb_s": 400
}
```

### 算子定义 JSON (`data/operators/FlashAttention.json`)

```json
{
  "name": "FlashAttention",
  "description": "Flash Attention 加速注意力计算",
  "inputs": [
    {"name": "hidden_states", "shape": "[bsz, seq_len, hidden_dim]", "dtype": "bf16"}
  ],
  "params": [
    {"name": "num_heads", "shape": "32", "dtype": ""},
    {"name": "head_dim", "shape": "128", "dtype": ""}
  ],
  "outputs": [
    {"name": "attn_output", "shape": "[bsz, seq_len, hidden_dim]", "dtype": "bf16"}
  ],
  "compute_flops": "4 * bsz * seq_len * hidden_dim^2",
  "module": "attn.flash_attention",
  "category": "Attention"
}
```

> 字段顺序固定为：`name → description → inputs → params → outputs → compute_flops → module → category`

### 模型结构导出 JSON（当前格式）

```json
{
  "name": "custom-model",
  "num_ops": 10,
  "num_layers": 3,
  "num_edges": 9,
  "operators": [
    {
      "op_id": 0,
      "op_name": "Embedding",
      "layer_idx": -1,
      "rank_idx": 0,
      "op_module": "embed",
      "inputs": [],
      "params": [],
      "outputs": [],
      "compute_flops": "0"
    }
  ],
  "edges": [
    {"from": 0, "to": 1}
  ],
  "ranks": [
    {
      "rank_idx": 0,
      "ops": [0, 1, 2],
      "layers": [
        {"layer_idx": 0, "repeat": 1, "kind": "regular", "ops": [1, 2]}
      ]
    }
  ]
}
```

> **layer_idx 语义**：`-2` = start, `-1` = 层前全局算子, `0..N` = 第 N 层（regular）, `900` = 层后全局算子, `980-989` = MTP 层, `1000` = end

---

## 前端设计

### 四步式工作流

应用使用步骤式导航，引导用户完成建模全流程：

- **Step 1 — 手搓模型**：拖拽算子/模块构建模型结构
- **Step 2 — 负载配置**：设置推理参数、并行策略或自动寻优目标、量化配置
- **Step 3 — 硬件配置**：选择或自定义芯片规格
- **Step 4 — 仿真运行**：查看结果（手动仿真结果或自动寻优结果）

### 手搓模型编辑器（Step 1）— 四面板布局

```
┌──────────────────────────────────────────────────────────────────┐
│  [算子库]        │  [React Flow 画布]  │  [模型配置]              │
│  [模块库]        │                    │  [算子详情]              │
│  (垂直分割)      │  节点 + 连线        │  (垂直分割，可拖拽调整)    │
│                  │  ArrowPad 导航      │                          │
└──────────────────────────────────────────────────────────────────┘
```

#### 算子库面板（OperatorPanel）

- **7 个内置分类**：Flow、Embedding、Normalization、Attention、FFN & Activation、Parallel Linear、MoE
- **搜索过滤**：按算子名称或描述搜索，搜索时自动展开所有分类
- **拖拽到画布**：HTML5 原生 DnD，`dataTransfer` 传递 `application/kepler-operator` MIME 数据
- **点击选中**：单击选中算子，双击取消选中
- **分类折叠**：点击分类标题展开/折叠，记住折叠状态
- **自定义算子**：localStorage 持久化（`kepler-custom-operators`），支持创建/编辑/复制/删除/导入/导出

#### 模块库面板（ModulePanel）

- **内置模块**：从后端 `/api/library/modules` 加载，按模块分组算子
- **自定义模块**：localStorage 持久化（`kepler-custom-modules`），用户可组合算子为模块
- **拖拽批量添加**：拖拽模块到画布，一次性添加所有算子
- **展开查看**：点击模块展开/折叠查看内部算子列表
- **搜索过滤**：按模块名称搜索

#### 模型画布（ModelCanvas）

- **React Flow**：节点、连线、背景网格、缩放控件
- **拖拽放置**：支持算子拖放和模块拖放，在鼠标位置创建节点
- **自动布局**：节点数量变化或层排序变化时，拓扑排序自动重排（120ms 防抖）
- **连线**：
  - 自动连线：相邻节点间默认有隐式顺序边
  - 手动连线：用户可额外连接任意两节点
  - 手动断开：双击已存在的边移除
  - 循环检测：DFS 检查，防止创建环
  - 自动连线规则：同 rank 节点根据前驱关系判断串行/并行
- **层颜色编码**：不同层使用不同颜色区分
- **ArrowPad 导航**：画布右上角 D-pad，支持点击和按住连续平移，中心按钮可拖拽重新定位，双击居中有图
- **键盘导航**：方向键平移画布

#### 模型配置面板（ModelConfig）

- **双 Tab**：
  - **手搓模型 Tab**：
    - 模型名称输入 + 导入/导出 JSON
    - 多 Rank 管理（添加/复制/删除 Rank，编辑 Rank 索引）
    - 层前全局算子 / 层后全局算子 添加按钮
    - 层管理：新建层/新建 MTP 层（最多 1 个）
    - 层列表：折叠/展开、▲▼ 排序、命名、repeat 乘数、复制（MTP 层不可复制）、删除
    - 层内算子列表：编号、名称、描述，可删除和排序
    - 未分配算子可视化分配
  - **模型config.json Tab**：
    - 导入 HuggingFace config.json
    - 独立 JSON 文本域编辑

#### 算子详情面板（OperatorDetail）

- **Module 下拉**：按分类 optgroup 组织（Attention / MLP / MoE / Embedding / MHC / MTP / Norm / Output）
- **Inputs / Params / Outputs**：三个可折叠区域，每行编辑 name/shape/dtype
- **Compute FLOPs**：多行文本域
- **空状态**：未选中算子时显示引导文案

### 负载配置（Step 2）

- **模式切换**：手动仿真 / 自动寻优 双按钮切换
- **请求配置**：Phase、Batch Size、Input Length、Output Length、Prefix Hit Ratio
- **MTP 配置**：
  - MTP Tokens：有 MTP 层时可编辑（≥1），无 MTP 层时自动为 0 只读
  - MTP Ratio（0-1）：控制平均接受 tokens 数量
  - 平均接受 tokens：自动计算（prefill 固定为 1，decode 根据 ratio 公式计算）
- **并行策略**：
  - 手动模式：TP × DP × PP × EP = World Size 流水线，专用 TP（Embed/O/LMHead/ExtSE）
  - 自动模式：目标 TPOT + 最小/最大 GPU 数
  - PP 锁定为 1，CP 锁定为 1，EP 自动跟随 World Size
- **量化配置**：全局/MLP/共享专家/路由专家/KV Cache/激活值，全局量化可一键同步所有权重量化

### 硬件配置（Step 3）

- **内置芯片选择**：从后端硬件库加载预置规格
- **自定义输入**：手动填写所有硬件参数（显存/带宽/算力/互联/功耗等）
- **开始仿真按钮**：根据模式显示不同文案（手动="开始仿真"，自动="开始自动寻优"）

### 仿真结果（Step 4）

#### 手动模式结果（ResultsPanel）

- **模型统计 Tab**：多硬件对比表（TTOT/TPOT/Prefill/Decode/TPS/QPS/显存/OOM/并行策略），最优值高亮
- **Rank 统计 Tab**：
  - Summary 卡片（总 Ranks/最慢耗时/最大显存/OOM 统计）
  - Rank 网格快速预览
  - Rank 详情表（耗时/显存/权重/激活/噪声/容量/算子数/OOM），支持全部展开
  - 层详情表（per-layer breakdown）：每层显示 layer_idx/repeat/权重/激活/开始时间/结束时间/耗时，按 rank 分组折叠展开
- **算子统计 Tab**：环形占比图 + 排序表格（按总耗时/平均耗时排序），支持全部展开
- **算子详情 Tab**：按 Rank 筛选 + 列选择器（compute_cost/mem_cost/comm_cost/noise 等），支持全部展开

#### 自动模式结果（OptimizeResults）

- **搜索摘要**：候选数 / OOM 数 / 耗时 / 达标数
- **最优策略卡片**：策略标签 + 并行流水线 + TPOT/TPS/显存指标 + "应用到负载配置"按钮
- **候选策略对比表**：所有候选的 WS/TP/DP/专用TP/TPOT/TPS/显存/OOM，最优行高亮，达标/超标/OOM 行颜色区分

---

## 前端状态管理（Zustand）

### 模型 Store (`stores/model.ts`)

```typescript
interface ModelState {
  // 数据
  operators: OperatorDef[]          // 内置算子定义
  operatorList: string[]            // 后端算子名列表
  customOperators: OperatorDef[]    // 自定义算子（localStorage: kepler-custom-operators）
  moduleList: string[]              // 后端模块名列表
  moduleDefs: Record<string, OperatorDef[]>  // 模块名 → 算子列表
  customModules: ModuleDef[]        // 自定义模块（localStorage: kepler-custom-modules）
  nodes: OpNodeData[]              // 所有算子节点
  layers: LayerConfig[]            // 层配置列表（每层 per-rank 算子）
  edges: EdgeData[]                // 手动连线列表
  disconnected: Set<string>        // 已断开的自动边
  topGlobalIndices: number[]       // 层前全局算子索引
  bottomGlobalIndices: number[]    // 层后全局算子索引

  // 多 Rank 支持
  ranks: number[]                  // 所有 rank 索引
  activeRank: number               // 当前活跃 rank

  // 选中状态
  selectedNodeId: string | null
  selectedOperator: OperatorDef | null
  selectedModule: ModuleDef | null
  modelName: string
  hfConfigText: string             // HF config.json 文本

  // 算子 CRUD
  addNode, removeNode, selectNode, updateNodeSection, moveNode, addNodeToLayer

  // 层操作（支持 regular 和 MTP 两种）
  addLayer, addMtpLayer, duplicateLayer, removeLayer, setLayerRepeat, renameLayer
  moveLayerUp, moveLayerDown, setLayerIndex
  addOpToLayer, removeOpFromLayer, reorderLayerOps

  // 全局算子
  addTopGlobal, addBottomGlobal, removeTopGlobal, removeBottomGlobal
  reorderTopGlobals, reorderBottomGlobals

  // 连线
  addEdge, removeEdge, wouldCreateCycle

  // 自定义算子 CRUD
  addCustomOperator, removeCustomOperator, updateCustomOperator, importCustomOperators

  // 自定义模块 CRUD
  addCustomModule, removeCustomModule, updateCustomModule

  // 多 Rank 管理
  addRank, duplicateRank, removeRank, setActiveRank, updateRankIndex

  // 序列化
  exportModel, importFromJSON, reorderNodes
  loadOperatorList, loadModuleList
}
```

### 工作负载 Store（`WorkloadConfigPanel.tsx` 内 `useInferenceStore`）

```typescript
interface InferenceParams {
  phase: string                 // "prefill" | "decode"
  batch_size: number
  input_length: number
  output_length: number
  num_mtp_tokens: number
  ratio_mtp_tokens: number
  prefix_hit_ratio: number
  // 并行策略
  world_size, tp_size, dp_size, pp_size, ep_size, cp_size: number
  embed_tp_size, o_tp_size, lmhead_tp_size: number
  external_shared_expert_rank_size: number
  // 量化
  quant_global, quant_mlp, quant_shared_expert, quant_routed_expert: string
  quant_kv_cache, quant_activation: string
  // 自动寻优
  optimizeMode: 'manual' | 'auto'
  targetTpotMs: number
  minWorldSize: number
  maxWorldSize: number
}
```

### 硬件 Store (`stores/hardware.ts`)

管理硬件配置列表，支持内置加载和手动添加。

---

## 部署方案

### Dockerfile（多阶段构建）

- **Stage 1 (Node)**：`npm install && npm run build`，产出 `dist/`
- **Stage 2 (Python)**：COPY 前端 `dist/` → FastAPI `static/`，`uvicorn kepler.web.__main__:app`

### Docker Compose

```yaml
services:
  kepler:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./backend/data:/app/data
```

---

## 功能清单

### 一、手搓模型编辑器（Step 1）

| # | 功能 | 说明 |
|---|------|------|
| 1 | 算子库面板 | 7 个内置分类 + 自定义分类，搜索过滤 |
| 2 | 模块库面板 | 内置模块 + 自定义模块，拖拽批量添加算子 |
| 3 | 拖拽建模 | HTML5 原生 DnD，支持算子拖放和模块拖放 |
| 4 | 画布自动布局 | 拓扑排序，120ms 防抖，节点变化自动触发 |
| 5 | 连线管理 | 自动顺序连线 + 手动连线 + 跨 rank 连线 + 循环检测 |
| 6 | 层管理 | 新建/删除/复制/折叠/排序/命名/乘数，支持 regular + MTP（最多 1 个） |
| 7 | 全局算子 | 层前/层后全局算子，per-rank 管理 |
| 8 | 算子详情编辑 | Module/Inputs/Params/Outputs/FLOPs 可编辑 |
| 9 | 模型导入/导出 | JSON 文件导入/导出完整模型结构（含 rank 信息） |
| 10 | 模型config.json Tab | 可编辑 JSON 文本域，导入 HuggingFace config.json |
| 11 | 自定义算子 CRUD | 创建/编辑/复制/删除，localStorage 持久化 |
| 12 | 自定义算子导入/导出 | JSON 文件导入/导出 |
| 13 | 自定义模块 CRUD | 创建/编辑/删除，localStorage 持久化 |
| 14 | 多 Rank 管理 | 添加/复制/删除 Rank，编辑索引，per-rank 层内算子 |
| 15 | 面板拖拽调整 | 左/右/上/下面板均可拖拽调整 |
| 16 | ArrowPad 导航 | 画布 D-pad，点击/按住平移，拖拽重定位，双击居中 |

### 二、负载配置（Step 2）

| # | 功能 | 说明 |
|---|------|------|
| 17 | 推理参数 | phase, batch_size, input_length, output_length |
| 18 | Prefix Hit Ratio | prefix cache 命中率配置 |
| 19 | MTP 配置 | 有 MTP 层时 num_mtp_tokens 可编辑，否则自动 0；ratio_mtp_tokens；自动计算平均接受 tokens |
| 20 | 并行策略（手动模式） | TP × DP × PP = EP = WS 流水线 + 专用 TP 配置 |
| 21 | 自动寻优（自动模式） | 目标 TPOT + GPU 范围，一键搜索最优并行策略 |
| 22 | 量化配置 | global/mlp/shared_expert/routed_expert/kv_cache/activation，全局同步 |

### 三、硬件配置（Step 3）

| # | 功能 | 说明 |
|---|------|------|
| 23 | 内置芯片库 | 从后端 `/api/library/hardwares` 加载 |
| 24 | 自定义硬件 | 手动填写完整硬件参数 |
| 25 | 双模式开始按钮 | 手动→"开始仿真"，自动→"开始自动寻优" |

### 四、仿真结果（Step 4）

| # | 功能 | 说明 |
|---|------|------|
| 26 | 模型统计 Tab | 多硬件对比表，最优值高亮 |
| 27 | Rank 统计 Tab | 汇总卡片 + 网格预览 + 详情表 + 层详情表 |
| 28 | 算子统计 Tab | 环形占比图 + 排序表格，按总/平均耗时排序 |
| 29 | 算子详情 Tab | Rank 筛选 + 列选择器 + 张量信息展示 |
| 30 | 自动寻优结果 | 搜索摘要 + 最优策略卡片 + 候选策略对比表 + 应用策略 |
| 31 | 重新运行 | 返回配置或重新执行 |

### 五、部署

| # | 功能 | 说明 |
|---|------|------|
| 32 | Docker Compose 单容器 | 前端构建产物 + FastAPI 统一打包 |
