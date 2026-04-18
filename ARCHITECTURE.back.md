# ZRT-Sim: LLM Performance Modeling & Simulation Architecture

## 1. 总体架构

### 1.1 设计理念

- **声明式建模**：用户只需声明"什么模型 + 什么硬件 + 什么优化策略"，系统自动完成图抓取→图变换→仿真执行→报表生成全流程
- **图驱动**：所有分析都基于计算图（Op Graph），图是贯穿全流程的核心数据结构
- **平台无关的图 + 平台相关的规则**：计算图本身不绑定平台，平台差异通过可插拔的规则包和仿真器注入
- **无卡运行**：全流程基于 FakeTensor，不需要真实硬件和模型权重

### 1.2 系统架构总览

```
┌─────────────────────────────────────────────────────────────────────┐
│                        Application Layer                            │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌────────────┐ │
│  │ Config Search │ │ HW Compare   │ │ Bottleneck   │ │ What-If    │ │
│  │ (寻优)        │ │ (硬件对比)    │ │ Analysis     │ │ Analysis   │ │
│  └──────┬───────┘ └──────┬───────┘ └──────┬───────┘ └─────┬──────┘ │
├─────────┼────────────────┼────────────────┼────────────────┼────────┤
│         └────────────────┴────────┬───────┴────────────────┘        │
│                        Orchestrator (Pipeline)                      │
├─────────────────────────────────────────────────────────────────────┤
│                         Core Layer                                  │
│  ┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌───────────┐ │
│  │ Model   │ │ Graph    │ │ Graph    │ │ Graph    │ │ Report    │ │
│  │ Manager │→│ Capture  │→│Transform │→│ Executor │→│ Generator │ │
│  └─────────┘ └──────────┘ └──────────┘ └──────────┘ └───────────┘ │
│                                           ↕                         │
│                               ┌───────────────────┐                │
│                               │  Op Simulator Hub  │                │
│                               │ ┌───┐ ┌───┐ ┌───┐ │                │
│                               │ │公式│ │拟合│ │tiling│                │
│                               │ └───┘ └───┘ └───┘ │                │
│                               └───────────────────┘                │
├─────────────────────────────────────────────────────────────────────┤
│                       Foundation Layer                              │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐               │
│  │ Hardware     │ │ Platform     │ │ Common       │               │
│  │ Registry     │ │ Rule Packs   │ │ (IR/types)   │               │
│  └──────────────┘ └──────────────┘ └──────────────┘               │
└─────────────────────────────────────────────────────────────────────┘
```

### 1.3 数据流全景

```
                    ┌──────────┐
                    │ HF ID /  │
                    │ 本地路径  │
                    └────┬─────┘
                         ▼
              ┌──────────────────┐    ┌──────────────┐
              │   Model Manager  │◄───│ patches.py   │
              │  (FakeTensor加载) │    │ compat.py    │
              └────────┬─────────┘    └──────────────┘
                       ▼
              ┌──────────────────┐
              │  Graph Capture   │  dispatch + tracker
              │  (aten op序列)   │  prefill / decode / train
              └────────┬─────────┘
                       ▼
                 ┌───────────┐          Raw Op Graph (IR)
                 │ OpGraph   │─────────────────────────────┐
                 └─────┬─────┘                             │
                       ▼                                   │
    ┌─────────────────────────────────┐                    │
    │       Graph Transform Pipeline  │                    │
    │  ┌────────┐ ┌────────┐ ┌──────┐│                    │
    │  │Fusion  │→│Parallel│→│Optim ││                    │
    │  │Engine  │ │Splitter│ │Passes││                    │
    │  └────────┘ └────────┘ └──────┘│                    │
    └───────────────┬─────────────────┘                    │
                    ▼                                      │
              Transformed Graph(s)                         │
                    │  (可能有多个：不同 rank 的子图)         │
                    ▼                                      ▼
           ┌──────────────────┐                   ┌──────────────┐
           │  Graph Executor  │                   │ Report Gen   │
           │  (多流调度+仿真)  │──────────────────→│ (xlsx/html/  │
           │                  │   perf data        │  onnx/json)  │
           └────────┬─────────┘                   └──────────────┘
                    │
                    ▼
           ┌──────────────────┐
           │ Op Simulator Hub │
           │  (算子耗时查询)   │
           └──────────────────┘
```

---

## 2. Foundation Layer — 基础层

### 2.1 计算图 IR（Intermediate Representation）

计算图 IR 是全系统的核心数据结构，所有模块围绕它工作。

#### 2.1.1 设计原则

- **与框架解耦**：IR 不依赖 PyTorch/ONNX 的内部类型
- **层次化**：OpNode → SubGraph → OpGraph，支持逐级聚合
- **可序列化**：JSON 序列化/反序列化，支持持久化和跨进程传递
- **平台无关**：IR 本身不含平台信息，平台特性通过 annotation 附加

```
python/zrt/
├── ir/
│   ├── __init__.py
│   ├── types.py            # 基础类型定义 (DType, Shape, TensorMeta, DeviceType)
│   ├── op_node.py          # OpNode: 单个算子节点
│   ├── edge.py             # Edge: tensor 数据边 + 控制依赖边
│   ├── sub_graph.py        # SubGraph: 子图（融合后的算子组、一个 pipeline stage）
│   ├── op_graph.py         # OpGraph: 完整计算图
│   ├── graph_utils.py      # 拓扑排序、依赖分析、子图提取等通用图算法
│   └── serialization.py    # JSON/protobuf 序列化
```

#### 2.1.2 核心数据结构

```python
@dataclass
class TensorMeta:
    """tensor 的元信息，不持有真实数据"""
    id: str                     # 全局唯一 tensor ID
    shape: tuple[int, ...]
    dtype: DType
    device: DeviceType          # "cpu" | "npu" | "gpu"
    memory_bytes: int           # 估算显存占用
    is_contiguous: bool = True

@dataclass
class OpNode:
    """计算图中的一个算子节点"""
    id: str                     # 全局唯一
    op_type: str                # aten op name, e.g. "aten.mm", "aten.layer_norm"
    inputs: list[TensorMeta]    # 输入 tensor 元信息
    outputs: list[TensorMeta]   # 输出 tensor 元信息
    attrs: dict[str, Any]       # 算子属性 (kernel_size, groups, ...)
    scope: str                  # 模块层次路径 "model.layers.0.self_attn.q_proj"
    component: str              # 组件分类 "attention" | "mlp" | "norm" | ...

    # 以下为可选，由 Graph Transform 阶段填充
    annotations: dict[str, Any] = field(default_factory=dict)
    # annotations 示例:
    #   "parallel_strategy": "tensor_parallel",
    #   "parallel_dim": 1,
    #   "fused_into": "fused_op_123",
    #   "platform_op": "flash_attention_v2",
    #   "estimated_flops": 1234567,

@dataclass
class Edge:
    """图中的数据边或控制依赖边"""
    src_node_id: str
    src_output_idx: int
    dst_node_id: str
    dst_input_idx: int
    tensor_meta: TensorMeta | None  # 控制边为 None
    edge_type: str = "data"         # "data" | "control"

@dataclass
class OpGraph:
    """完整的计算图"""
    name: str
    nodes: dict[str, OpNode]        # id -> OpNode
    edges: list[Edge]
    metadata: GraphMetadata         # 模型名、phase、batch_size、seq_len 等

    # 层次化支持
    subgraphs: dict[str, SubGraph] = field(default_factory=dict)

    # 图操作 API
    def topo_sort(self) -> list[OpNode]: ...
    def predecessors(self, node_id: str) -> list[OpNode]: ...
    def successors(self, node_id: str) -> list[OpNode]: ...
    def subgraph_by_scope(self, prefix: str) -> OpGraph: ...
    def clone(self) -> OpGraph: ...
```

### 2.2 硬件注册表（Hardware Registry）

```
python/zrt/
├── hardware/
│   ├── __init__.py
│   ├── registry.py          # HardwareRegistry: 加载+查询
│   ├── spec.py              # HardwareSpec 数据类
│   └── configs/             # 内置硬件配置 YAML
│       ├── nvidia_a100_80g.yaml
│       ├── nvidia_h100_sxm.yaml
│       ├── nvidia_h800.yaml
│       ├── ascend_910b.yaml
│       ├── ascend_910c.yaml
│       └── kunlunxin_r480.yaml
```

#### 硬件配置 YAML 示例

```yaml
# hardware/configs/ascend_910b.yaml
name: "Ascend 910B"
vendor: "huawei"
device_type: "npu"

compute:
  fp16_tflops: 320
  bf16_tflops: 320
  fp32_tflops: 160
  int8_tops: 640
  int4_tops: 1280
  # 不同算子类型的实际效率系数（可选，用于精细建模）
  efficiency:
    matmul: 0.85        # 大矩阵可达 85% 利用率
    vector: 0.60        # element-wise 类算子
    reduce: 0.50        # reduction 类

memory:
  capacity_gb: 64
  bandwidth_gbps: 1600     # HBM 带宽
  l2_cache_mb: 48
  # 分级带宽（可选）
  bandwidth_tiers:
    - name: "L2"
      bandwidth_gbps: 6400
      capacity_mb: 48
    - name: "HBM"
      bandwidth_gbps: 1600
      capacity_gb: 64

interconnect:
  intra_node:
    type: "HCCS"            # NVLink / HCCS / PCIe / ...
    bandwidth_gbps: 392     # 单向带宽
    latency_us: 3
    topology: "full_mesh"   # full_mesh / ring / tree
    num_devices: 8
  inter_node:
    type: "RoCE"
    bandwidth_gbps: 200
    latency_us: 5

software:
  # 平台能力声明，影响 Graph Transform 的融合/优化规则选择
  supported_fusions:
    - "flash_attention"
    - "add_rmsnorm"
    - "swiglu"
    - "moe_token_dispatch"
  supported_dtypes: ["fp16", "bf16", "fp32", "int8", "int4"]
  supported_parallel:
    - "tensor_parallel"
    - "pipeline_parallel"
    - "expert_parallel"
    - "sequence_parallel"
  max_streams: 4            # 最大并行流数
```

### 2.3 平台规则包（Platform Rule Packs）

将平台相关的所有策略封装为独立的规则包，一个平台一个目录：

```
python/zrt/
├── platforms/
│   ├── __init__.py
│   ├── base.py              # PlatformRulePack 抽象基类
│   ├── ascend/
│   │   ├── __init__.py
│   │   ├── fusion_rules.py  # Ascend 特有融合规则
│   │   ├── parallel_rules.py
│   │   ├── optim_passes.py  # EPLB、共享专家外置等
│   │   └── op_mapping.py    # aten op → Ascend 算子映射
│   ├── nvidia/
│   │   ├── __init__.py
│   │   ├── fusion_rules.py  # CUDA graph、flash_attn_v2 等
│   │   ├── parallel_rules.py
│   │   ├── optim_passes.py
│   │   └── op_mapping.py
│   └── generic/             # 通用/理想化平台（基线对比用）
│       └── ...
```

```python
class PlatformRulePack(ABC):
    """平台规则包接口"""

    @abstractmethod
    def get_fusion_rules(self) -> list[FusionRule]: ...

    @abstractmethod
    def get_op_mapping(self, op_type: str, tensor_metas: list[TensorMeta]) -> MappedOp: ...

    @abstractmethod
    def get_parallel_strategies(self) -> list[ParallelStrategy]: ...

    @abstractmethod
    def get_optim_passes(self) -> list[OptimPass]: ...

    @abstractmethod
    def get_communication_cost(self, collective: str, data_bytes: int,
                                num_devices: int, hw: HardwareSpec) -> float: ...
```

---

## 3. Core Layer — 核心模块

### 3.1 模型管理器（Model Manager）

**职责**：加载模型为 FakeTensor，返回 `(model, config, fake_mode)`，模型源码不修改。

```
python/zrt/
├── model/
│   ├── __init__.py
│   ├── loader.py            # 统一加载入口（沿用现有 model_loader.py 逻辑）
│   ├── compat.py            # transformers 版本 shim（沿用现有）
│   ├── patches.py           # 运行时 monkey-patch（沿用现有）
│   ├── registry.py          # 本地模型注册表
│   └── config_parser.py     # 从 config.json 提取模型结构参数
│                            # (num_layers, hidden_size, num_heads, ...)
```

> 现有 `python/zrt/graph/` 下的 `model_loader.py`、`compat.py`、`patches.py` 逻辑整体平移到 `model/`，
> `graph/` 模块聚焦于图抓取。

#### ConfigParser

从 config.json 中提取结构化的模型参数，供后续模块使用：

```python
@dataclass
class ModelProfile:
    """模型结构参数，从 config.json 解析"""
    model_type: str              # "llama", "qwen2", "deepseek_v3", ...
    num_layers: int
    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int
    vocab_size: int
    max_position_embeddings: int
    # MoE 相关（可选）
    num_experts: int | None = None
    num_shared_experts: int | None = None
    topk: int | None = None
    # MLA 相关（可选）
    kv_lora_rank: int | None = None
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    # MTP 相关（可选）
    num_mtp_heads: int | None = None
```

### 3.2 图抓取模块（Graph Capture）

**职责**：通过 `TorchDispatchMode` 拦截 aten 算子序列，构建 Raw OpGraph。

```
python/zrt/
├── capture/
│   ├── __init__.py
│   ├── tracer.py            # 入口：run_capture(model, inputs, phase) -> OpGraph
│   ├── dispatch.py          # RecordingDispatch（沿用现有，输出改为 IR OpNode）
│   ├── tracker.py           # ModuleTracker（沿用现有）
│   ├── input_builder.py     # 构建 prefill/decode/train 阶段的 FakeTensor 输入
│   └── phase.py             # Phase 枚举 + 多阶段编排 (prefill → decode)
```

#### Input Builder

根据模型配置和推理阶段自动生成输入：

```python
class InputBuilder:
    """根据 ModelProfile + Phase 构建模型输入"""

    def build(self, profile: ModelProfile, phase: Phase,
              batch_size: int, seq_len: int,
              fake_mode: FakeTensorMode) -> dict[str, Any]:
        """
        返回 model.forward(**kwargs) 所需的全部输入。
        对于 decode 阶段，自动注入 past_key_values。
        """
        ...
```

### 3.3 图变换模块（Graph Transform）

**核心思想**：图变换是一系列 Pass 的有序执行，每个 Pass 接收 OpGraph 输出新的 OpGraph。

```
python/zrt/
├── transform/
│   ├── __init__.py
│   ├── pipeline.py          # TransformPipeline: Pass 编排和执行
│   ├── base.py              # GraphPass 抽象基类
│   ├── fusion/
│   │   ├── __init__.py
│   │   ├── engine.py        # FusionEngine（沿用现有三阶段引擎）
│   │   └── rules.py         # FusionRule 基类 + 匹配框架
│   ├── parallel/
│   │   ├── __init__.py
│   │   ├── tensor_parallel.py    # TP 切分 pass
│   │   ├── pipeline_parallel.py  # PP 切分 pass
│   │   ├── expert_parallel.py    # EP 切分 pass
│   │   ├── sequence_parallel.py  # SP pass
│   │   ├── data_parallel.py      # DP/ZeRO pass
│   │   └── comm_inserter.py      # 在切分点插入通信算子 (all_reduce, all_gather, ...)
│   ├── optim/
│   │   ├── __init__.py
│   │   ├── eplb.py               # Expert-Level Load Balancing
│   │   ├── shared_expert.py      # 共享专家外置
│   │   ├── mtp.py                # Multi-Token Prediction head
│   │   ├── quantization.py       # 低精度量化 (W8A8, W4A16, ...)
│   │   ├── kv_cache_quant.py     # KV cache 量化
│   │   └── recompute.py          # 重计算策略（训练场景）
│   └── analysis/
│       ├── __init__.py
│       ├── flops_counter.py      # FLOPs/MACs 估算 pass
│       ├── memory_estimator.py   # 显存占用估算 pass
│       └── roofline.py           # Roofline 分析标注 pass
```

#### Pass 接口

```python
class GraphPass(ABC):
    """图变换 Pass 基类"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def run(self, graph: OpGraph, context: TransformContext) -> OpGraph:
        """输入图，输出变换后的新图（不修改原图）"""
        ...

@dataclass
class TransformContext:
    """Pass 执行上下文，在 Pipeline 各 Pass 之间传递共享信息"""
    hw_spec: HardwareSpec
    platform: PlatformRulePack
    model_profile: ModelProfile
    phase: Phase
    # 并行配置
    parallel_config: ParallelConfig | None = None
    # 累计信息
    inserted_comms: list[OpNode] = field(default_factory=list)
```

#### 并行切分示例（Tensor Parallel）

```python
class TensorParallelPass(GraphPass):
    """
    对 Linear 算子按指定维度切分，插入通信算子。

    规则：
    - QKV projection: 按 head 维度切分 (column parallel)
    - O projection: 按 hidden 维度切分 (row parallel)，后接 all_reduce
    - MLP gate/up: column parallel
    - MLP down: row parallel，后接 all_reduce
    - Embedding: 可选 vocab parallel
    """

    def run(self, graph: OpGraph, ctx: TransformContext) -> OpGraph:
        new_graph = graph.clone()
        tp_degree = ctx.parallel_config.tp_degree
        for node in new_graph.topo_sort():
            split_spec = self._match_split_rule(node, ctx)
            if split_spec:
                self._split_node(new_graph, node, split_spec, tp_degree)
                if split_spec.needs_comm:
                    comm_node = self._make_comm_node(split_spec.comm_type,
                                                     node, tp_degree)
                    new_graph.insert_after(node, comm_node)
        return new_graph
```

#### 图变换 Pipeline 编排

```python
class TransformPipeline:
    """
    按顺序执行一系列 GraphPass。
    支持条件跳过、dry-run、中间结果快照。
    """

    def __init__(self):
        self._passes: list[GraphPass] = []

    def add(self, pass_: GraphPass, condition: Callable | None = None):
        self._passes.append((pass_, condition))

    def run(self, graph: OpGraph, context: TransformContext) -> OpGraph:
        current = graph
        for pass_, condition in self._passes:
            if condition and not condition(context):
                continue
            current = pass_.run(current, context)
        return current

# 典型编排
def build_inference_pipeline(ctx: TransformContext) -> TransformPipeline:
    pipe = TransformPipeline()
    # 1. 平台融合
    pipe.add(FusionPass(ctx.platform.get_fusion_rules()))
    # 2. 量化标注
    pipe.add(QuantizationPass(), condition=lambda c: c.parallel_config.quant is not None)
    # 3. 并行切分（按类型依次）
    pipe.add(TensorParallelPass(), condition=lambda c: c.parallel_config.tp_degree > 1)
    pipe.add(ExpertParallelPass(), condition=lambda c: c.parallel_config.ep_degree > 1)
    pipe.add(SequenceParallelPass(), condition=lambda c: c.parallel_config.sp_enabled)
    pipe.add(PipelineParallelPass(), condition=lambda c: c.parallel_config.pp_degree > 1)
    # 4. 通信算子插入
    pipe.add(CommInserterPass())
    # 5. 分析标注
    pipe.add(FlopsCounterPass())
    pipe.add(MemoryEstimatorPass())
    pipe.add(RooflineAnnotatorPass())
    return pipe
```

### 3.4 算子仿真器 Hub（Op Simulator Hub）

**核心思想**：统一接口，多种后端，支持回退链。

```
python/zrt/
├── simulator/
│   ├── __init__.py
│   ├── hub.py               # SimulatorHub: 统一查询入口 + 后端路由
│   ├── base.py              # OpSimulator 抽象基类
│   ├── result.py            # SimResult 数据类
│   ├── backends/
│   │   ├── __init__.py
│   │   ├── roofline.py      # 理论公式建模 (Roofline Model)
│   │   ├── profile_db.py    # 真机 profiling 数据查表
│   │   ├── tiling_sim.py    # Tiling 级算子仿真器接口
│   │   ├── regression.py    # 回归拟合模型
│   │   └── custom.py        # 用户自定义仿真器接口
│   ├── models/              # 预训练的拟合模型文件
│   │   ├── matmul_910b.pkl
│   │   └── ...
│   └── profile_data/        # 真机 profiling 数据
│       ├── ascend_910b/
│       └── nvidia_h100/
```

#### 仿真器接口

```python
@dataclass
class SimResult:
    """单个算子的仿真结果"""
    op_node_id: str
    latency_us: float           # 算子耗时（微秒）
    compute_us: float           # 其中计算部分
    memory_us: float            # 其中访存部分
    flops: int                  # 浮点运算量
    memory_read_bytes: int      # 读取字节数
    memory_write_bytes: int     # 写入字节数
    arithmetic_intensity: float # 计算强度 (FLOPs/Byte)
    hw_utilization: float       # 硬件利用率 (0~1)
    bottleneck: str             # "compute" | "memory" | "latency"
    backend_used: str           # 实际使用的仿真后端名称
    confidence: float           # 置信度 (0~1)

class OpSimulator(ABC):
    """算子仿真器基类"""

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def priority(self) -> int:
        """优先级，值越大越优先使用（越精确）"""
        return 0

    @abstractmethod
    def can_simulate(self, op_node: OpNode, hw_spec: HardwareSpec) -> bool:
        """判断此后端是否支持该算子"""
        ...

    @abstractmethod
    def simulate(self, op_node: OpNode, hw_spec: HardwareSpec) -> SimResult:
        ...
```

#### SimulatorHub 路由逻辑

```python
class SimulatorHub:
    """
    统一入口，按优先级链式查询多个后端。
    支持：
    - 注册/注销后端
    - 按算子类型路由
    - 自动回退（高精度后端不支持时回退到低精度）
    - 结果缓存
    - 独立对外提供 REST API（可选）
    """

    def __init__(self):
        self._backends: list[OpSimulator] = []  # 按优先级排序
        self._cache: dict[str, SimResult] = {}

    def register(self, backend: OpSimulator):
        self._backends.append(backend)
        self._backends.sort(key=lambda b: b.priority, reverse=True)

    def simulate(self, op_node: OpNode, hw_spec: HardwareSpec) -> SimResult:
        cache_key = self._make_cache_key(op_node, hw_spec)
        if cache_key in self._cache:
            return self._cache[cache_key]

        for backend in self._backends:
            if backend.can_simulate(op_node, hw_spec):
                result = backend.simulate(op_node, hw_spec)
                self._cache[cache_key] = result
                return result

        # 兜底：理论 Roofline
        return self._roofline_fallback(op_node, hw_spec)

    def simulate_graph(self, graph: OpGraph, hw_spec: HardwareSpec) -> list[SimResult]:
        """批量仿真整个图的所有算子"""
        return [self.simulate(node, hw_spec)
                for node in graph.topo_sort()]
```

#### 理论公式后端（Roofline）

```python
class RooflineSimulator(OpSimulator):
    """
    基于 Roofline Model 的理论公式仿真。
    无需任何硬件数据，根据 FLOPs 和访存量 + 硬件峰值规格估算。

    latency = max(compute_time, memory_time)
    compute_time = FLOPs / peak_tflops
    memory_time  = (read_bytes + write_bytes) / bandwidth
    """
    name = "roofline"
    priority = 0  # 最低优先级，兜底

    def can_simulate(self, op_node, hw_spec):
        return True  # 任何算子都能估

    def simulate(self, op_node, hw_spec):
        flops = self._estimate_flops(op_node)
        read_bytes, write_bytes = self._estimate_memory(op_node)
        peak = hw_spec.compute.get_peak_flops(op_node.outputs[0].dtype)
        bw = hw_spec.memory.bandwidth_gbps * 1e9  # bytes/s

        compute_us = flops / peak / 1e6
        memory_us = (read_bytes + write_bytes) / bw * 1e6
        latency_us = max(compute_us, memory_us)
        ...
```

#### 真机数据拟合后端

```python
class RegressionSimulator(OpSimulator):
    """
    基于真机 profiling 数据训练的回归模型。
    特征：op_type + input_shapes + dtype → latency_us
    精度高于 Roofline，但需要预先采集+训练。
    """
    name = "regression"
    priority = 50

    def __init__(self, model_dir: str):
        self._models = self._load_models(model_dir)

    def can_simulate(self, op_node, hw_spec):
        key = (op_node.op_type, hw_spec.name)
        return key in self._models
```

### 3.5 图执行器（Graph Executor）

**核心思想**：基于 DAG 拓扑+资源约束的多流调度仿真，产生每个算子的起止时间。

```
python/zrt/
├── executor/
│   ├── __init__.py
│   ├── scheduler.py         # 多流 DAG 调度器
│   ├── stream.py            # Stream 抽象（计算流、通信流）
│   ├── timeline.py          # Timeline: 存储调度结果
│   ├── overlap.py           # 通算掩盖分析
│   └── memory_plan.py       # 显存 timeline 规划
```

#### 多流调度器

```python
@dataclass
class StreamConfig:
    """流配置"""
    num_compute_streams: int = 1   # 计算流数
    num_comm_streams: int = 1      # 通信流数
    # 通信流与计算流可以并行 → 实现通算掩盖

@dataclass
class ScheduledOp:
    """调度后的算子"""
    op_node: OpNode
    sim_result: SimResult
    stream_id: int
    start_us: float
    end_us: float

class DAGScheduler:
    """
    基于拓扑排序 + 就绪队列 + 多流资源分配的仿真调度。

    调度策略:
    - 计算算子 → 分配到计算流
    - 通信算子 (all_reduce, all_gather, ...) → 分配到通信流
    - 同一流内串行，不同流可并行
    - 依赖满足才能入就绪队列
    """

    def schedule(self, graph: OpGraph, sim_results: dict[str, SimResult],
                 stream_config: StreamConfig) -> Timeline:
        timeline = Timeline()
        ready_queue = PriorityQueue()  # 按最早可开始时间排序
        stream_available = {i: 0.0 for i in range(stream_config.total_streams)}

        for node in self._find_initial_nodes(graph):
            ready_queue.put((0.0, node))

        while not ready_queue.empty():
            earliest_start, node = ready_queue.get()
            stream_id = self._select_stream(node, stream_config, stream_available)
            actual_start = max(earliest_start, stream_available[stream_id])
            duration = sim_results[node.id].latency_us
            end_time = actual_start + duration

            timeline.add(ScheduledOp(
                op_node=node,
                sim_result=sim_results[node.id],
                stream_id=stream_id,
                start_us=actual_start,
                end_us=end_time,
            ))
            stream_available[stream_id] = end_time

            # 释放后继节点
            for succ in graph.successors(node.id):
                if timeline.all_predecessors_done(succ, graph):
                    succ_start = timeline.latest_predecessor_end(succ, graph)
                    ready_queue.put((succ_start, succ))

        return timeline
```

#### 通算掩盖分析

```python
class OverlapAnalyzer:
    """分析通信与计算的重叠效果"""

    def analyze(self, timeline: Timeline) -> OverlapReport:
        compute_intervals = timeline.intervals_by_type("compute")
        comm_intervals = timeline.intervals_by_type("communication")

        total_compute = sum_durations(compute_intervals)
        total_comm = sum_durations(comm_intervals)
        overlap = compute_intersection(compute_intervals, comm_intervals)

        return OverlapReport(
            total_compute_us=total_compute,
            total_comm_us=total_comm,
            overlap_us=overlap,
            exposed_comm_us=total_comm - overlap,  # 未被掩盖的通信
            overlap_ratio=overlap / total_comm if total_comm > 0 else 1.0,
        )
```

### 3.6 报表生成器（Report Generator）

```
python/zrt/
├── report/
│   ├── __init__.py
│   ├── engine.py            # ReportEngine: 编排各 writer
│   ├── excel_writer.py      # 算子级明细 Excel（沿用现有 + 扩展）
│   ├── html_writer.py       # 交互式 HTML 报表（timeline 可视化）
│   ├── json_writer.py       # JSON 结构化数据（供其他工具消费）
│   ├── onnx_writer.py       # ONNX 计算图（沿用现有）
│   └── summary.py           # 端到端性能汇总
```

#### 端到端性能汇总

```python
@dataclass
class E2ESummary:
    """端到端性能汇总"""
    model_name: str
    hardware: str
    phase: str                       # prefill / decode
    parallel_config: str             # "TP8-PP2-EP4"

    # 总体指标
    total_latency_ms: float          # 单次推理延迟
    throughput_tokens_per_sec: float  # 吞吐 (tokens/s)
    time_to_first_token_ms: float    # TTFT (prefill)
    time_per_output_token_ms: float  # TPOT (decode)

    # 分解
    compute_ms: float
    communication_ms: float
    exposed_comm_ms: float           # 未被掩盖的通信
    memory_peak_gb: float

    # 细分
    breakdown_by_component: dict[str, float]   # attention: 40%, mlp: 50%, ...
    breakdown_by_layer: list[float]            # 每层耗时
    bottleneck_ops: list[str]                  # Top-N 瓶颈算子
```

---

## 4. Application Layer — 应用层

### 4.1 Orchestrator（Pipeline 编排器）

将 Core Layer 的各模块串联为完整工作流：

```python
class Orchestrator:
    """
    一键式推理性能预测流程编排。
    """

    def run(self, config: RunConfig) -> RunResult:
        # 1. 加载模型
        model, model_profile, fake_mode = self.model_manager.load(
            config.model_id, config.num_layers)

        # 2. 抓取计算图（prefill + decode）
        graphs = {}
        for phase in config.phases:
            graphs[phase] = self.capture.run(model, model_profile, phase,
                                             config.batch_size, config.seq_len,
                                             fake_mode)
        fake_mode.__exit__(None, None, None)

        # 3. 加载硬件 + 平台规则
        hw_spec = self.hw_registry.get(config.hardware)
        platform = self.platform_registry.get(hw_spec.vendor)

        # 4. 图变换
        transform_ctx = TransformContext(hw_spec, platform, model_profile,
                                         config.parallel_config)
        pipeline = build_inference_pipeline(transform_ctx)
        for phase, graph in graphs.items():
            graphs[phase] = pipeline.run(graph, transform_ctx)

        # 5. 算子仿真
        sim_results = {}
        for phase, graph in graphs.items():
            sim_results[phase] = self.simulator_hub.simulate_graph(graph, hw_spec)

        # 6. 图执行（多流调度）
        timelines = {}
        for phase, graph in graphs.items():
            timelines[phase] = self.scheduler.schedule(
                graph, sim_results[phase], config.stream_config)

        # 7. 生成报表
        self.report_engine.generate(graphs, sim_results, timelines, config)

        return RunResult(graphs, sim_results, timelines)
```

### 4.2 配置寻优（Config Search）

```python
class ConfigSearchEngine:
    """
    在给定模型+硬件下，搜索最优并行策略和优化配置。

    搜索空间:
    - TP degree: [1, 2, 4, 8]
    - PP degree: [1, 2, 4]
    - EP degree: [1, 2, 4, 8, ...num_experts]
    - SP: on/off
    - quantization: [none, w8a8, w4a16, ...]
    - micro_batch_size: [1, 2, 4, 8, ...]
    - recompute_strategy: [none, selective, full]

    搜索策略:
    - grid_search: 全量枚举（小搜索空间）
    - bayesian: 贝叶斯优化（大搜索空间）
    - rule_based: 基于经验规则的启发式剪枝
    """

    def search(self, model_id: str, hardware: str,
               objective: str = "throughput",  # "throughput" | "latency" | "cost"
               constraints: dict | None = None,  # e.g. {"memory_gb": 64}
               strategy: str = "rule_based",
               max_trials: int = 100,
               ) -> list[SearchResult]:
        ...
```

### 4.3 硬件对比（HW Compare）

```python
class HardwareComparator:
    """
    固定模型+配置，对比不同硬件的性能表现。
    输出对比报表（表格 + 雷达图数据）。
    """

    def compare(self, model_id: str,
                hardware_list: list[str],
                parallel_config: ParallelConfig,
                ) -> ComparisonReport:
        results = {}
        for hw_name in hardware_list:
            results[hw_name] = self.orchestrator.run(RunConfig(
                model_id=model_id,
                hardware=hw_name,
                parallel_config=parallel_config,
            ))
        return self._build_comparison(results)
```

### 4.4 瓶颈分析（Bottleneck Analysis）

```python
class BottleneckAnalyzer:
    """
    识别性能瓶颈：
    - 算子级：Top-N 耗时算子，compute-bound vs memory-bound
    - 模块级：attention vs MLP vs communication
    - 系统级：通算比、显存是否成为限制
    """

    def analyze(self, timeline: Timeline, graph: OpGraph) -> BottleneckReport:
        ...
```

---

## 5. 完整目录结构

```
python/zrt/
├── __init__.py
│
├── ir/                          # ① 计算图 IR（基础数据结构）
│   ├── __init__.py
│   ├── types.py                 #    DType, Shape, TensorMeta, DeviceType
│   ├── op_node.py               #    OpNode
│   ├── edge.py                  #    Edge
│   ├── sub_graph.py             #    SubGraph
│   ├── op_graph.py              #    OpGraph + 图操作 API
│   ├── graph_utils.py           #    拓扑排序、依赖分析
│   └── serialization.py         #    JSON 序列化
│
├── model/                       # ② 模型管理器
│   ├── __init__.py
│   ├── loader.py                #    FakeTensor 模型加载
│   ├── compat.py                #    transformers 版本 shim
│   ├── patches.py               #    运行时 monkey-patch
│   ├── registry.py              #    本地模型注册表
│   └── config_parser.py         #    ModelProfile 提取
│
├── capture/                     # ③ 图抓取
│   ├── __init__.py
│   ├── tracer.py                #    run_capture() 入口
│   ├── dispatch.py              #    aten dispatch 拦截
│   ├── tracker.py               #    模块路径追踪
│   ├── input_builder.py         #    阶段输入构建
│   └── phase.py                 #    Phase 枚举 + 多阶段编排
│
├── transform/                   # ④ 图变换
│   ├── __init__.py
│   ├── pipeline.py              #    TransformPipeline
│   ├── base.py                  #    GraphPass ABC
│   ├── fusion/                  #    算子融合
│   │   ├── engine.py
│   │   └── rules.py
│   ├── parallel/                #    并行策略
│   │   ├── tensor_parallel.py
│   │   ├── pipeline_parallel.py
│   │   ├── expert_parallel.py
│   │   ├── sequence_parallel.py
│   │   ├── data_parallel.py
│   │   └── comm_inserter.py
│   ├── optim/                   #    优化 Pass
│   │   ├── eplb.py
│   │   ├── shared_expert.py
│   │   ├── mtp.py
│   │   ├── quantization.py
│   │   ├── kv_cache_quant.py
│   │   └── recompute.py
│   └── analysis/                #    分析 Pass
│       ├── flops_counter.py
│       ├── memory_estimator.py
│       └── roofline.py
│
├── simulator/                   # ⑤ 算子仿真器
│   ├── __init__.py
│   ├── hub.py                   #    SimulatorHub 统一入口
│   ├── base.py                  #    OpSimulator ABC
│   ├── result.py                #    SimResult
│   ├── backends/
│   │   ├── roofline.py          #    理论公式
│   │   ├── profile_db.py        #    真机查表
│   │   ├── regression.py        #    回归拟合
│   │   ├── tiling_sim.py        #    Tiling 级仿真接口
│   │   └── custom.py            #    用户自定义
│   ├── models/                  #    拟合模型文件
│   └── profile_data/            #    真机数据
│
├── executor/                    # ⑥ 图执行器
│   ├── __init__.py
│   ├── scheduler.py             #    DAG 多流调度
│   ├── stream.py                #    Stream 抽象
│   ├── timeline.py              #    Timeline 数据
│   ├── overlap.py               #    通算掩盖分析
│   └── memory_plan.py           #    显存 timeline
│
├── report/                      # ⑦ 报表生成器
│   ├── __init__.py
│   ├── engine.py                #    ReportEngine
│   ├── excel_writer.py          #    算子明细 Excel
│   ├── html_writer.py           #    交互式 HTML
│   ├── json_writer.py           #    结构化 JSON
│   ├── onnx_writer.py           #    ONNX 图
│   └── summary.py               #    E2E 汇总
│
├── hardware/                    # ⑧ 硬件配置
│   ├── __init__.py
│   ├── registry.py
│   ├── spec.py                  #    HardwareSpec
│   └── configs/                 #    内置 YAML
│       ├── nvidia_a100_80g.yaml
│       ├── nvidia_h100_sxm.yaml
│       ├── ascend_910b.yaml
│       └── ...
│
├── platforms/                   # ⑨ 平台规则包
│   ├── __init__.py
│   ├── base.py                  #    PlatformRulePack ABC
│   ├── ascend/
│   ├── nvidia/
│   └── generic/
│
├── app/                         # ⑩ 应用层
│   ├── __init__.py
│   ├── orchestrator.py          #    全流程编排
│   ├── config_search.py         #    配置寻优
│   ├── hw_compare.py            #    硬件对比
│   ├── bottleneck.py            #    瓶颈分析
│   └── cli.py                   #    CLI 入口
│
├── api/                         # ⑪ 对外 API（可选）
│   ├── __init__.py
│   ├── server.py                #    FastAPI 服务
│   ├── routes/
│   │   ├── simulate.py          #    /api/simulate  算子仿真独立接口
│   │   ├── trace.py             #    /api/trace     图抓取接口
│   │   └── predict.py           #    /api/predict   端到端预测
│   └── schemas.py               #    请求/响应模型
│
└── graph/                       # 现有代码（渐进迁移）
    └── ...
```

---

## 6. 关键设计决策

### 6.1 如何做到新模型天然适配

```
┌──────────────────────────────────────────────────────────────┐
│                  新模型接入流程                                │
│                                                              │
│  标准架构 (llama/qwen/mistral)                               │
│  ──────────────────────────────                              │
│  只需提供 HF model_id → 自动完成全流程                        │
│  （transformers 已支持的架构，无需任何额外代码）                 │
│                                                              │
│  自定义架构 (deepseek_v3 等)                                  │
│  ──────────────────────────────                              │
│  1. hf_models/<name>/ 放入模型文件（只读）                     │
│  2. compat.py 添加注册表条目（2 行）                           │
│  3. patches.py 添加运行时兼容 patch（按需）                    │
│  无需改动 capture / transform / simulator / executor 任何代码  │
│                                                              │
│  全新模块类型 (新的 attention 机制等)                          │
│  ──────────────────────────────                              │
│  可能需要:                                                    │
│  - fusion_rules 中添加新的融合规则                             │
│  - parallel/ 中调整切分策略                                   │
│  - classifier 中添加新组件类别                                │
│  但以上都是"添加"而非"修改"                                    │
└──────────────────────────────────────────────────────────────┘
```

### 6.2 多流并行与通算掩盖建模

```
时间轴示例 (TP=2, 1 compute stream + 1 comm stream per rank):

Stream 0 (Compute): ┃ QKV_proj ┃ Attn ┃ O_proj ┃    wait    ┃ Gate ┃ Up ┃ ...
Stream 1 (Comm):    ┃          ┃      ┃        ┃ AllReduce  ┃      ┃    ┃ ...
                    ─────────────────────────────────────────────────────────→ t

Pipeline Parallel 微批次交叠:

Rank 0: ┃ MB0-Stage0 ┃ MB1-Stage0 ┃ MB2-Stage0 ┃    idle    ┃ MB0-grad  ┃
Rank 1: ┃   idle     ┃ MB0-Stage1 ┃ MB1-Stage1 ┃ MB2-Stage1 ┃   idle    ┃
         ─────────────────────────────────────────────────────────────────→ t

MoE Expert Parallel (EP=4, 8 experts):

Stream 0 (Compute): ┃ Gate ┃   ┃ Expert0,1 ┃           ┃ Combine ┃
Stream 1 (Comm):    ┃      ┃ A2A_dispatch ┃ A2A_combine ┃         ┃
                    ───────────────────────────────────────────────→ t
                         通信与部分计算重叠
```

### 6.3 算子仿真器选择策略

```
查询: simulate(matmul, shape=[4096,4096], dtype=bf16, hw=910B)
                    │
                    ▼
        ┌─── Tiling Simulator (priority=100) ───┐
        │   can_simulate? → 检查是否有 910B 的   │
        │   tiling 仿真器实现                    │
        │   ✗ 不支持                             │
        └────────────────────────────────────────┘
                    │
                    ▼
        ┌─── Regression Model (priority=50) ────┐
        │   can_simulate? → 检查是否有训练好的    │
        │   matmul_910b.pkl 模型                 │
        │   ✓ 支持 → 返回 SimResult              │
        └────────────────────────────────────────┘

如果 regression 也不支持:
                    │
                    ▼
        ┌─── Profile DB (priority=30) ──────────┐
        │   can_simulate? → 查表精确匹配         │
        │   shape+dtype 完全命中 → 返回           │
        └────────────────────────────────────────┘
                    │
                    ▼
        ┌─── Roofline (priority=0, 兜底) ───────┐
        │   can_simulate? → True (任何算子)       │
        │   理论公式估算 → 返回                   │
        └────────────────────────────────────────┘
```

### 6.4 可扩展性预留

| 扩展点 | 机制 | 示例 |
|--------|------|------|
| 新硬件 | 添加 YAML + PlatformRulePack | 新增 TPU 支持 |
| 新融合规则 | PlatformRulePack.get_fusion_rules() | FlashMLA |
| 新并行策略 | 新增 GraphPass 子类 | Ring Attention |
| 新仿真后端 | 新增 OpSimulator 子类并 register | NN-based predictor |
| 新优化项 | 新增 optim/ 下的 Pass | Speculative Decoding |
| 新输出格式 | 新增 report/ 下的 Writer | PDF 报告 |
| 新模型架构 | compat 注册 + patches | Jamba, RWKV |
| 训练场景 | Phase 枚举添加 train，recompute pass | 训练 FLOPs 估算 |
| 对外 API | api/ FastAPI 服务 | 算子仿真微服务 |

---

## 7. 配置体系

### 7.1 运行配置

```yaml
# run_config.yaml
model:
  id: "deepseek-ai/DeepSeek-V3-0324"
  num_layers: 61                    # 0 = 全量
  local_path: null                  # 可选，优先于 id

hardware: "ascend_910b"

phases: ["prefill", "decode"]

workload:
  batch_size: 1
  seq_len: 4096
  output_len: 512                   # decode 生成长度

parallel:
  tp_degree: 8
  pp_degree: 1
  ep_degree: 8
  dp_degree: 1
  sp_enabled: true

optimization:
  quantization: "w8a8"
  kv_cache_quant: "int8"
  eplb: true
  shared_expert_external: true
  mtp_heads: 1
  recompute: "none"                 # none | selective | full

executor:
  num_compute_streams: 1
  num_comm_streams: 1

simulator:
  backends: ["regression", "roofline"]  # 按优先级列出
  profile_data_dir: null

output:
  dir: "output/DeepSeek-V3-0324_910B_TP8"
  formats: ["excel", "html", "onnx", "json"]
```

### 7.2 集群配置

```yaml
# cluster_config.yaml
name: "Training Cluster A"
nodes:
  - count: 16
    hardware: "ascend_910b"
    devices_per_node: 8
    intra_node:
      type: "HCCS"
      bandwidth_gbps: 392
    inter_node:
      type: "RoCE"
      bandwidth_gbps: 200
```

---

## 8. CLI 接口设计

```bash
# 单次推理性能预测
zrt predict deepseek-ai/DeepSeek-V3-0324 \
    --hardware ascend_910b \
    --tp 8 --ep 8 \
    --batch-size 1 --seq-len 4096 \
    --output-dir output/dsv3_910b

# 从 YAML 配置运行
zrt predict --config run_config.yaml

# 硬件对比
zrt compare deepseek-ai/DeepSeek-V3-0324 \
    --hardware ascend_910b,nvidia_h100 \
    --tp 8 --ep 8

# 配置寻优
zrt search deepseek-ai/DeepSeek-V3-0324 \
    --hardware ascend_910b \
    --objective throughput \
    --constraint "memory_gb<=64"

# 仅抓取图（兼容现有功能）
zrt capture deepseek-ai/DeepSeek-V3-0324 \
    --layers 4 --phase prefill,decode

# 算子仿真独立查询
zrt simulate --op matmul \
    --shape 4096,4096,4096 --dtype bf16 \
    --hardware ascend_910b

# 启动 API 服务
zrt serve --port 8080
```

---

## 9. 与现有代码的关系

现有 `python/zrt/graph/` 是本架构的原型实现，对应关系：

| 现有模块 | 新架构位置 | 迁移策略 |
|---------|-----------|---------|
| `model_loader.py` | `model/loader.py` | 直接平移 |
| `compat.py` | `model/compat.py` | 直接平移 |
| `patches.py` | `model/patches.py` | 直接平移 |
| `dispatch.py` | `capture/dispatch.py` | 输出类型改为 IR OpNode |
| `tracker.py` | `capture/tracker.py` | 直接平移 |
| `fusion.py` | `transform/fusion/engine.py` | 三阶段引擎平移，规则改为从 PlatformRulePack 加载 |
| `fusion_rules.py` | `platforms/*/fusion_rules.py` | 按平台拆分 |
| `classifier.py` | `transform/analysis/` 或 `ir/` | 组件分类逻辑保留 |
| `excel_writer.py` | `report/excel_writer.py` | 扩展列（增加性能数据列） |
| `graph_builder.py` | `capture/tracer.py` | 合并到图构建流程 |
| `graph_exporter.py` | `report/onnx_writer.py` | 直接平移 |
| `main.py` | `app/orchestrator.py` + `app/cli.py` | 拆分为编排器和 CLI |

**迁移原则**：渐进式，`graph/` 保持可用，新模块逐步替代，最终 `graph/` 成为 `capture/` 的别名。

---

## 10. 开发路线建议

```
Phase 1 — 基础 (当前已有 + IR 改造)
├── ir/ 核心数据结构
├── model/ (从 graph/ 平移)
├── capture/ (从 graph/ 平移，输出 IR)
├── hardware/ (YAML 加载)
└── report/ (从 graph/ 平移)

Phase 2 — 仿真能力
├── simulator/backends/roofline.py (理论公式)
├── executor/scheduler.py (单流调度)
├── transform/fusion/ (从 graph/ 平移)
├── transform/analysis/flops_counter.py
└── report/summary.py (E2E 汇总)

Phase 3 — 并行建模
├── transform/parallel/ (TP/PP/EP/SP)
├── transform/parallel/comm_inserter.py
├── executor/ 多流调度 + 通算掩盖
└── simulator/backends/profile_db.py

Phase 4 — 优化项
├── transform/optim/ (量化、EPLB、MTP、重计算)
├── platforms/ 规则包
└── simulator/backends/regression.py

Phase 5 — 应用层
├── app/orchestrator.py
├── app/config_search.py
├── app/hw_compare.py
├── app/bottleneck.py
└── api/ 服务化
```
