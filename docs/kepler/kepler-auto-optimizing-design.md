# Kepler 并行策略自动寻优 — 总体设计

## 1. 概述

### 1.1 背景

当前 Kepler 的并行策略（TP/DP/EP/world_size 等）完全由用户手动配置。用户需要反复尝试不同组合，每次手动调整参数后运行 `simulate`，对比结果，才能找到满足时延要求的策略。这个过程低效且依赖经验。

### 1.2 目标

用户给定以下输入后，系统自动搜索并输出最优并行策略：

| 输入 | 来源 |
|------|------|
| 模型配置（层数、hidden_dim、注意力头数等） | 模型编辑器 或 预设模型 |
| `input_length` / `output_length` | 推理配置面板 |
| 目标 TPOT（平均 decode 时延上限） | 用户指定 |
| 硬件配置（GPU 型号、显存、带宽等） | 硬件配置面板 |

输出：满足 TPOT 目标的前提下，**使用最少 GPU** 的并行策略（world_size / TP / DP / embed_tp / o_tp / lmhead_tp / EP 等）。

### 1.3 核心思路

利用现有 `POST /api/simulate` 引擎作为评估函数，在合法策略空间内搜索，找到满足约束的最优解。

```
搜索空间 → 候选策略 → simulate() → 过滤(OOM/TPOT超标) → 排序 → 最优策略
```

---

## 2. 搜索算法

### 2.1 策略空间定义

当前约束（来自 `useInferenceStore` 的 useEffect）：

| 约束 | 公式 |
|------|------|
| PP 固定 | `pp_size = 1` |
| CP 固定 | `cp_size = 1` |
| EP 跟随 world_size | `ep_size = world_size` |
| World size 约束 | `world_size = tp_size × dp_size` |

### 2.2 搜索空间

搜索变量共 **6 个维度**：

| 变量 | 候选值 | 说明 |
|------|--------|------|
| `world_size` | {1, 2, 4, 8, 16, 32, 64, 128, 256, 512} | 2 的幂，上限由用户指定 |
| `tp_size` | {d \| d 整除 world_size} | world_size 的所有因子 |
| `dp_size` | world_size / tp_size | 由前两者确定，但作为独立维度参与排序 |
| `embed_tp_size` | {1, tp_size} 或 factors(tp_size) | 嵌入层 TP 度 |
| `o_tp_size` | {1, tp_size} 或 factors(tp_size) | 输出投影 TP 度 |
| `lmhead_tp_size` | {1, tp_size} 或 factors(tp_size) | LM Head TP 度 |

`external_shared_expert_rank_size` 搜索期间保持 0。

**搜索粒度两级**：

- **粗粒度**（默认）：专用 TP（`embed_tp`, `o_tp`, `lmhead_tp`）只取 `{1, tp_size}`，每个 (world_size, tp, dp) 组合产生 2³ = 8 个子候选
- **细粒度**（可选）：专用 TP 取 tp_size 的全部因子，子候选更多但可能找到更优解

示例：world_size=8, tp=4, dp=2, 粗粒度下产生 8 个子候选：

```
(4,2, 1,1,1)  (4,2, 4,1,1)  (4,2, 1,4,1)  (4,2, 1,1,4)
(4,2, 4,4,1)  (4,2, 4,1,4)  (4,2, 1,4,4)  (4,2, 4,4,4)
```

### 2.3 搜索策略

**从 world_size=1 开始递增搜索**：

1. 对每个 `world_size`，枚举所有合法 `(tp, dp)` 组合
2. 对每个 `(tp, dp)`，枚举专用 TP 组合（粗粒度：8 个）
3. 每个候选调用一次 `simulate`
4. 若当前 world_size 下已有策略满足 TPOT 要求，**停止增大 world_size**（目标是最少 GPU）
5. 同一 world_size 内，若某个 tp 值已满足目标且专用 TP=全 tp_size，可跳过其余专用 TP 组合

### 2.4 评估与排序

每个候选策略通过 `simulate` 得到：
- `tpot_ms`：平均 decode 时延
- `max_peak_mem_gb`：单 GPU 峰值显存
- `is_oom`：是否显存溢出

排序规则（依次）：
1. `is_oom == true` → 淘汰
2. `tpot_ms <= target_tpot` → 满足约束的优先
3. `world_size` 升序（最少 GPU 优先）
4. `dp_size` 升序（同 GPU 下优先数据并行，减少通信开销）
5. `tpot_ms` 升序（同配置下选延迟最低）

### 2.5 剪枝优化

- 若 `world_size=N` 下**所有** TP/DP 组合均 OOM 或远超目标，则跳过该 world_size 内剩余未评估的专用 TP 组合
- 若某 `(world_size, tp, dp)` 在专用 TP=全 tp_size 时已满足目标，跳过该组合的其余子候选（降低专用 TP 不会改善 TPOT）
- 若某 tp 配置下 `dp_size` 较小但仍 OOM，更大 tp（即更小 dp）也不会缓解 OOM → 剪枝

### 2.6 搜索规模估算

worst case（max_world_size=512，粗粒度）：

| world_size | 因子数 | × 8 子候选 | 累计 |
|------------|--------|------------|------|
| 1 | 1 | 8 | 8 |
| 2 | 2 | 16 | 24 |
| 4 | 3 | 24 | 48 |
| 8 | 4 | 32 | 80 |
| 16 | 5 | 40 | 120 |
| 32 | 6 | 48 | 168 |
| 64 | 7 | 56 | 224 |
| 128 | 8 | 64 | 288 |
| 256 | 9 | 72 | 360 |
| 512 | 10 | 80 | 440 |

早停后实际评估量远小于此值（通常 world_size ≤ 64 即命中目标）。

---

## 3. 后端 API 设计

### 3.1 文件结构

```
backend/kepler/web/
├── routes/
│   └── optimize.py      ← 新增：路由层，处理 HTTP 请求
├── services/
│   └── optimizer.py     ← 新增：搜索算法，复用 SimulationService
├── schemas.py           ← 修改：新增 Pydantic 模型
└── app.py               ← 修改：注册 optimize 路由
```

### 3.2 schemas.py 新增模型

在现有 `schemas.py` 末尾追加以下 Pydantic 模型：

```python
# ── 自动寻优 ───────────────────────────────────────────

class OptimizeRequest(BaseModel):
    """自动寻优请求"""
    # 模型来源 — 与 SimulateRequest 一致
    model_name: Optional[str] = None
    model_json: Optional[dict] = None
    hf_config_json: Optional[dict] = None

    # 推理参数
    input_length: int = 2048
    output_length: int = 512

    # 优化目标
    target_tpot_ms: float = Field(..., gt=0, description="目标 decode 时延上限 (ms)")
    max_world_size: int = Field(default=512, ge=1, le=1024)

    # 搜索粒度
    fine_grained: bool = False

    # 量化配置 — 搜索期间不变
    quant_global: str = "fp16"
    quant_mlp: str = "fp16"
    quant_shared_expert: str = "fp16"
    quant_routed_expert: str = "fp16"
    quant_kv_cache: str = "fp16"

    # 硬件配置 — 与 SimulateRequest 一致
    hardwares: list[HardwareEntry] = Field(default_factory=list)
    hardware_name: str = ""


class StrategyResult(BaseModel):
    """单个候选策略的评估结果"""
    world_size: int
    tp_size: int
    dp_size: int
    embed_tp_size: int
    o_tp_size: int
    lmhead_tp_size: int
    strategy_label: str                     # "TP8_DP2"
    tpot_ms: float
    max_peak_mem_gb: float
    total_mem_gb: float                     # max_peak_mem_gb × world_size
    is_oom: bool
    meets_target: bool


class SearchSummary(BaseModel):
    """搜索过程摘要"""
    total_candidates: int
    evaluated: int                          # 实际调用 simulate 的次数
    pruned: int
    oom_count: int
    elapsed_ms: float


class OptimizeResponse(BaseModel):
    """自动寻优响应"""
    optimal: Optional[StrategyResult] = None
    candidates: list[StrategyResult] = Field(default_factory=list)
    search_summary: SearchSummary
```

### 3.3 routes/optimize.py — 路由层

遵循现有 `routes/simulate.py` 的极简风格：`APIRouter` → 委托 `Service`。

```python
from __future__ import annotations

from fastapi import APIRouter

from ..schemas import OptimizeRequest, OptimizeResponse
from ..services.optimizer import OptimizerService

router = APIRouter(prefix="/api", tags=["optimize"])


@router.post("/optimize", response_model=OptimizeResponse)
async def optimize(req: OptimizeRequest):
    return OptimizerService().optimize(req)
```

### 3.4 services/optimizer.py — 搜索引擎

核心类 `OptimizerService`，复用 `SimulationService` 作为评估函数。

```python
from __future__ import annotations

import time
import math
from typing import Iterator

from ..schemas import (
    OptimizeRequest, OptimizeResponse, StrategyResult,
    SearchSummary, ParallelConfig, RequestConfig, QuantConfig,
    WorkloadEntry, HardwareEntry, SimulateRequest,
)
from .simulation import SimulationService
from ...utils.log import logger


def _factors(n: int) -> list[int]:
    """返回 n 的所有因子，升序排列。"""
    result = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)
    return sorted(result)


class OptimizerService:
    """并行策略自动寻优引擎。

    搜索空间: world_size(2的幂) × tp(因子) × embed_tp × o_tp × lmhead_tp
    评估函数: SimulationService.simulate()
    目标:     满足 TPOT 约束下使用最少 GPU。
    """

    def __init__(self):
        self._sim = SimulationService()

    # ── 公开入口 ────────────────────────────────────────

    def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        t0 = time.perf_counter()
        candidates: list[StrategyResult] = []

        total = 0
        evaluated = 0

        for world_size in self._iter_world_sizes(req.max_world_size):
            tp_list = _factors(world_size)

            for tp in tp_list:
                dp = world_size // tp

                # 生成专用 TP 子候选
                sub_tps = self._iter_sub_tps(tp, req.fine_grained)

                for (embed_tp, o_tp, lmhead_tp) in sub_tps:
                    total += 1
                    parallel = ParallelConfig(
                        world_size=world_size,
                        tp_size=tp,
                        dp_size=dp,
                        pp_size=1,
                        ep_size=world_size,
                        cp_size=1,
                        embed_tp_size=embed_tp,
                        o_tp_size=o_tp,
                        lmhead_tp_size=lmhead_tp,
                    )

                    result = self._evaluate(req, parallel)
                    evaluated += 1
                    candidates.append(result)

                    logger.info(
                        "candidate %s WS=%d TP=%d DP=%d eTP=%d oTP=%d lmTP=%d "
                        "tpot=%.2f oom=%s meets=%s",
                        result.strategy_label, world_size, tp, dp,
                        embed_tp, o_tp, lmhead_tp,
                        result.tpot_ms, result.is_oom, result.meets_target,
                    )

            # 早停：当前 world_size 下已有策略满足目标，不再尝试更大 world_size
            if any(c.meets_target and c.world_size == world_size for c in candidates):
                logger.info("early stop at world_size=%d — target met", world_size)
                break

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # 排序
        candidates.sort(key=self._rank_key)

        optimal = next((c for c in candidates if c.meets_target), None)

        oom_count = sum(1 for c in candidates if c.is_oom)
        pruned = total - evaluated

        logger.info(
            "optimize done — total=%d evaluated=%d pruned=%d oom=%d "
            "optimal=%s elapsed=%.0fms",
            total, evaluated, pruned, oom_count,
            optimal.strategy_label if optimal else "NONE",
            elapsed_ms,
        )

        return OptimizeResponse(
            optimal=optimal,
            candidates=candidates,
            search_summary=SearchSummary(
                total_candidates=total,
                evaluated=evaluated,
                pruned=pruned,
                oom_count=oom_count,
                elapsed_ms=round(elapsed_ms, 1),
            ),
        )

    # ── 搜索空间迭代器 ──────────────────────────────────

    @staticmethod
    def _iter_world_sizes(max_ws: int) -> Iterator[int]:
        """生成 2 的幂 world_size 序列，从 1 到 max_world_size。"""
        ws = 1
        while ws <= max_ws:
            yield ws
            ws *= 2

    @staticmethod
    def _iter_sub_tps(tp: int, fine: bool) -> list[tuple[int, int, int]]:
        """生成 (embed_tp, o_tp, lmhead_tp) 候选列表。

        粗粒度: {1, tp}^3 → 8 个组合
        细粒度: factors(tp)^3 → 更多组合
        """
        if fine:
            vals = _factors(tp)
        else:
            vals = [1, tp]
        return [(e, o, l) for e in vals for o in vals for l in vals]

    # ── 评估 ────────────────────────────────────────────

    def _evaluate(
        self, req: OptimizeRequest, parallel: ParallelConfig,
    ) -> StrategyResult:
        """用给定并行策略跑一次 simulate，提取关键指标。"""
        # 构建临时的 SimulateRequest
        sim_req = SimulateRequest(
            model_name=req.model_name,
            model_json=req.model_json,
            hf_config_json=req.hf_config_json,
            hardwares=req.hardwares,
            hardware_name=req.hardware_name,
            workloads=[WorkloadEntry(
                request=RequestConfig(
                    phase="decode",
                    batch_size=1,
                    input_length=req.input_length,
                    output_length=req.output_length,
                ),
                parallel=parallel,
                quant=QuantConfig(
                    quant_global=req.quant_global,
                    quant_mlp=req.quant_mlp,
                    quant_shared_expert=req.quant_shared_expert,
                    quant_routed_expert=req.quant_routed_expert,
                    quant_kv_cache=req.quant_kv_cache,
                ),
            )],
        )

        # 调用现有仿真引擎
        sim_result = self._sim.simulate(sim_req)
        single = sim_result.results[0].result if sim_result.results else None

        if single is None:
            return StrategyResult(
                world_size=parallel.world_size,
                tp_size=parallel.tp_size,
                dp_size=parallel.dp_size,
                embed_tp_size=parallel.embed_tp_size,
                o_tp_size=parallel.o_tp_size,
                lmhead_tp_size=parallel.lmhead_tp_size,
                strategy_label=self._make_label(parallel),
                tpot_ms=float("inf"),
                max_peak_mem_gb=0,
                total_mem_gb=0,
                is_oom=True,
                meets_target=False,
            )

        return StrategyResult(
            world_size=parallel.world_size,
            tp_size=parallel.tp_size,
            dp_size=parallel.dp_size,
            embed_tp_size=parallel.embed_tp_size,
            o_tp_size=parallel.o_tp_size,
            lmhead_tp_size=parallel.lmhead_tp_size,
            strategy_label=single.strategy,
            tpot_ms=single.tpot_ms,
            max_peak_mem_gb=single.peak_mem_gb,
            total_mem_gb=single.peak_mem_gb * parallel.world_size,
            is_oom=single.oom,
            meets_target=(not single.oom and single.tpot_ms <= req.target_tpot_ms),
        )

    # ── 排序 ────────────────────────────────────────────

    @staticmethod
    def _rank_key(c: StrategyResult) -> tuple:
        """排序键：OOM淘汰 → 满足目标优先 → GPU最少 → DP升序 → 延迟最低。"""
        return (
            0 if not c.is_oom else 1,          # OOM 排最后
            0 if c.meets_target else 1,         # 满足目标优先
            c.world_size,                        # GPU 少优先
            -c.dp_size,                          # DP 大优先（通信少）
            c.tpot_ms,                           # 延迟低优先
        )

    # ── 工具 ────────────────────────────────────────────

    @staticmethod
    def _make_label(p: ParallelConfig) -> str:
        parts = []
        if p.tp_size > 1: parts.append(f"TP{p.tp_size}")
        if p.dp_size > 1: parts.append(f"DP{p.dp_size}")
        return "_".join(parts) if parts else "single"
```

### 3.5 app.py 注册路由

在现有 `app.py` 中新增两行（遵循现有 import + include 模式）：

```python
# 在现有 import 后追加
from .routes.optimize import router as optimize_router

# 在现有 app.include_router 后追加
app.include_router(optimize_router)
```

完整 diff：
```diff
 from .routes.simulate import router as simulate_router
 from .routes.library import router as library_router
+from .routes.optimize import router as optimize_router

 app.include_router(simulate_router)
 app.include_router(library_router)
+app.include_router(optimize_router)
```

### 3.6 端点规格

```
POST /api/optimize
```

**请求示例**：
```json
{
  "model_name": "kepler-v1",
  "input_length": 2048,
  "output_length": 512,
  "target_tpot_ms": 20.0,
  "max_world_size": 128,
  "quant_global": "fp8",
  "hardwares": [{"name": "H800", "config": "H800"}]
}
```

**响应示例**：
```json
{
  "optimal": {
    "world_size": 16,
    "tp_size": 8,
    "dp_size": 2,
    "embed_tp_size": 8,
    "o_tp_size": 1,
    "lmhead_tp_size": 8,
    "strategy_label": "TP8_DP2",
    "tpot_ms": 12.34,
    "max_peak_mem_gb": 38.5,
    "total_mem_gb": 616.0,
    "is_oom": false,
    "meets_target": true
  },
  "candidates": [...],
  "search_summary": {
    "total_candidates": 80,
    "evaluated": 48,
    "pruned": 32,
    "oom_count": 12,
    "elapsed_ms": 3240.5
  }
}
```

---

## 4. 前端设计

### 4.1 设计理念

**不新增独立面板，而是嵌入现有工作流。** 在步骤 2（负载配置页）顶部提供「手动仿真 / 自动寻优」模式切换。两个模式共享请求配置和量化配置，仅并行策略区域内容不同。

用户无需跳转页面即可在手动和自动之间切换，硬件配置在步骤 3 中可见，寻优在步骤 3 触发。

### 4.2 组件结构

```
App.tsx 步骤 2 → WorkloadConfigPanel
│
├── [模式切换条]                    ← 页面顶端，全宽双按钮
│   ┌──────────┬──────────┐
│   │ 手动仿真  │ 自动寻优  │
│   └──────────┴──────────┘
│
├── 请求配置        (两种模式共用)
│   Phase / Batch Size / Input Length / Output Length
│   MTP Tokens / MTP Ratio / 平均接受tokens数量
│
├── 并行策略        (内容随模式切换)
│   ┌─ 手动模式 ───────────────────┐
│   │ TP × DP × PP = EP = WS badge│
│   │ CP chip                      │
│   │ 专用 TP: Embed O LMHead ExtSE│
│   └──────────────────────────────┘
│   ┌─ 自动模式 ───────────────────┐
│   │ 目标 TPOT: [ 20 ] ms         │
│   │ 最大 GPU 数: [ 128 ]          │
│   │ 提示：硬件配置在步骤 3 中设置  │
│   └──────────────────────────────┘
│
└── 量化配置        (两种模式共用)
    权重: 全局 / MLP / 共享专家 / 路由专家
    KV Cache / 激活值
```

### 4.3 交互流程

```
1. 用户在步骤 2 配置请求参数和量化
2. 选择「手动仿真」：手动填写 TP/DP/EP 等
   选择「自动寻优」：填写目标 TPOT 和最大 GPU 数
3. 进入步骤 3，配置硬件 → 点击「开始仿真」
   - 手动模式：POST /api/simulate（现有流程）
   - 自动模式：POST /api/optimize（新增，内部调用多次 simulate）
4. 步骤 4 展示结果：
   - 手动模式：单个仿真结果（现有流程）
   - 自动模式：最优策略 + 候选对比表 + 搜索摘要
```

### 4.4 状态管理

**不新增独立 store。** 寻优相关状态直接加入现有 `useInferenceStore`（文件：`WorkloadConfigPanel.tsx`）：

```typescript
// InferenceParams 新增字段
optimizeMode: 'manual' | 'auto'  // 默认 'manual'
targetTpotMs: number              // 默认 20
maxWorldSize: number              // 默认 128
```

手动模式的并行策略字段（`tp_size`、`dp_size` 等）与自动模式的寻优参数共享同一 store，模式切换时仅 UI 渲染不同。

### 4.5 App.tsx 步骤 3 改造

步骤 3 的「开始仿真」按钮根据当前模式调用不同 API：

```typescript
async function handleRun() {
  const p = useInferenceStore.getState()
  // ... 准备 model_json, hf_config_json, hardwares ...

  if (p.optimizeMode === 'auto') {
    // 调用寻优 API
    const result = await runOptimize({
      model_json: modelJson,
      hf_config_json: hfConfigJson,
      input_length: p.input_length,
      output_length: p.output_length,
      target_tpot_ms: p.targetTpotMs,
      max_world_size: p.maxWorldSize,
      quant_global: p.quant_global,
      // ... 其他 quant 字段
      hardwares: hwList,
    })
    // 展示寻优结果
  } else {
    // 现有手动仿真流程
    const result = await runSimulate(payload)
  }
}
```

### 4.6 进度反馈

搜索最多 440 次 simulate 调用（worst case，早停后实际 ~50-100 次）。

| 方案 | 描述 |
|------|------|
| **A: 一次请求** | `POST /optimize` 同步返回全部结果，前端按钮显示 "寻优中..." 并禁用 |
| **B: SSE 流式** | 后端每评估一个候选推送结果，前端实时更新进度条 |

**Phase 1 采用方案 A**（搜索预计 10-30s），按钮文案变为 "自动寻优中..." 并显示 spinner。

### 4.7 文件清单

| 文件 | 操作 | 说明 |
|------|------|------|
| `src/components/ConfigForm/WorkloadConfigPanel.tsx` | 修改 | 新增模式切换条 + 自动寻优输入区 + InferenceParams 扩展 |
| `src/api/library.ts` | 修改 | 新增 `runOptimize()` 函数及类型 |
| `src/App.tsx` | 修改 | 步骤 3 handleRun 根据模式分流 API |
| `src/index.css` | 修改 | 新增 opt-mode-bar、opt-mode-btn、opt-auto-section 样式 |
| `backend/kepler/web/schemas.py` | 修改 | 新增 OptimizeRequest/OptimizeResponse/StrategyResult schema |
| `backend/kepler/web/routes/optimize.py` | **新增** | 寻优 API 路由 |
| `backend/kepler/web/services/optimizer.py` | **新增** | 搜索算法 |
| `backend/kepler/web/app.py` | 修改 | 注册 optimize 路由 |

---

## 5. 数据流

```
步骤 2: 负载配置
  │
  ├─ 手动模式: 用户填写 TP/DP/EP/专用 TP
  └─ 自动模式: 用户填写 target_tpot_ms / max_world_size
  │
  ▼
步骤 3: 硬件配置 + 开始仿真
  │
  ├─ 手动模式:
  │   读取 useInferenceStore (手动并行策略)
  │   → POST /api/simulate
  │   → 展示单个仿真结果
  │
  └─ 自动模式:
      读取 useInferenceStore (target_tpot_ms + max_world_size)
      + useModelStore (model_json)
      + getHardwareConfigs()
      → POST /api/optimize
        │
        ├─ OptimizerService 枚举候选策略
        ├─ 每个候选调用 SimulationService.simulate()
        ├─ 早停 + 排序 → 最优策略
        │
        ▼
      返回 OptimizeResponse (optimal + candidates + search_summary)
      → 步骤 4 展示最优策略 + 候选对比表
```

---

## 6. 实现计划

### Phase 1：后端搜索引擎

1. `backend/kepler/web/schemas.py` — 新增 OptimizeRequest/OptimizeResponse/StrategyResult/SearchSummary schema
2. `backend/kepler/web/services/optimizer.py` — OptimizerService：枚举候选 + simulate 评估 + 排序 + 早停
3. `backend/kepler/web/routes/optimize.py` — `POST /api/optimize`
4. `backend/kepler/web/app.py` — 注册路由

### Phase 2：前端改造

5. `src/components/ConfigForm/WorkloadConfigPanel.tsx`
   - `InferenceParams` 新增 `optimizeMode`、`targetTpotMs`、`maxWorldSize`
   - 页面顶端添加模式切换条（手动仿真 / 自动寻优）
   - 并行策略区域按模式条件渲染（手动 → TP/DP 流水线，自动 → 目标输入）
6. `src/api/library.ts` — 新增 `runOptimize()` + 请求/响应类型
7. `src/App.tsx` — 步骤 3 `handleRun` 根据 `optimizeMode` 分流 API
8. `src/index.css` — 新增 opt-mode-bar、opt-mode-btn、opt-auto-section 样式

### Phase 3：结果展示

9. 步骤 4 根据模式展示不同结果：
   - 手动模式：现有 ResultsPanel
   - 自动模式：最优策略卡片 + 候选对比表 + 搜索摘要

---

## 7. 约束与风险

| 风险 | 缓解 |
|------|------|
| 搜索时间过长（细粒度 440+ 候选） | 默认粗粒度；早停策略；world_size 从 1 递增 |
| simulate 非幂等或依赖全局状态 | 每次调用独立构建 context，不共享可变状态 |
| TPOT 目标设置过低导致无解 | 返回空 optimal + 完整候选列表，让用户看到最接近的策略 |
| 专用 TP 组合爆炸 | 粗粒度仅 {1, tp_size}，细粒度可选 |
| 前端长时间无响应 | 搜索在后端异步执行，前端显示 loading 状态 |

---

## 8. 后续扩展

- **吞吐量目标**：除 TPOT 外，支持以吞吐量（tokens/s）为目标
- **多目标帕累托**：展示时延 vs GPU 数的帕累托前沿
- **SSE 流式进度**：实时推送每个候选的评估结果
- **约束编程**：支持固定某个维度（如 DP 必须 ≥ 4）进行条件搜索
- **缓存**：对相同模型+硬件的评估结果做缓存，避免重复 simulate
