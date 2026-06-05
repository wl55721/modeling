# Kepler 成本耗时计算设计

## 概述

Kepler 的成本耗时计算分为四个层级，自底向上依次为：**算子（Operator）→ 层（Layer）→ Rank → 模型（Model）**。每个词级负责不同粒度的耗时聚合与时间轴编排。

---

## 1. 算子级别（Operator）

### 1.1 算子分类

| 类型 | 基类 | static_cost_us | 瓶颈 |
|------|------|---------------|------|
| Cube 算子 | `OpCubeBase` | 5 us | 矩阵计算 FLOPS |
| Vector 算子 | `OpVectorBase` | 2 us | 向量计算 FLOPS |
| 混合算子 | `OpMixBase` | 3 us | Cube + Vector FLOPS |
| 通信算子 | `OpCommBase` | 10 us | 通信带宽 + 显存带宽 |

### 1.2 耗时计算公式

每个算子调用 `calc_cost_us(chip)` 计算耗时，结果存入 `OperatorExecuteResult`。

**第一步：计算计算耗时 `comp_us`（微秒）**

```
chip_flops = spec_cube_fp16 × compute_ratio × 2   // 单位：GFLOPS（用于 Cube/混合算子）
chip_flops = spec_vect_fp16 × compute_ratio × 2   // 单位：GFLOPS（用于 Vector 算子）

// compute_flops_str 表达式求值（如 "B * S * h * 2"）
compute_flops = eval(compute_flops_str, context)

compute_cost_ns = compute_flops / chip_flops       // 结果：纳秒
comp_us = compute_cost_ns / 1000                   // 纳秒 → 微秒
```

**第二步：计算访存耗时 `mem_us`（微秒）**

```
chip_gmem_bw = spec_bw_memory × bw_gmem_ratio       // 单位：GiB/s（已×1024 转换自 TB/s）

bw_bytes = input_bytes + param_bytes + cache_bytes + output_bytes

mem_us = bw_bytes / 1024 / chip_gmem_bw             // KiB / GiB/s = 微秒
```

**第三步：计算通信耗时 `comm_us`（仅 OpCommBase，微秒）**

```
// 根据 rank_size 选择通信层级
comm_bw = bwsio  (rank_size ≤ bwsio_limit)
       = intra  (rank_size ≤ superpod_limit)
       = inter  (rank_size >  superpod_limit)

comm_us = comm_bytes / 1024 / comm_bw               // KiB / GiB/s = 微秒
```

**第四步：汇总总耗时**

```
                 ┌ comp_us    (计算为主)
total_cost_us =  │ mem_us     (访存为主)   + static_cost_us
                 └ max(comm_us, gmem_cost_us)  (通信为主)  // 仅通信算子
                 
bound_type = comp_us > mem_us ? "compute" : "memory"
```

### 1.3 start_time_ns / end_time_ns

拓扑执行完成后填充：

```
start_time_ns = max(finish_time[pred])   // 所有前驱节点中最晚完成时间（纳秒）
end_time_ns   = start_time_ns + total_cost_us × 1000   // us → ns
```

---

## 2. 层级别（Layer）

### 2.1 数据来源

层信息来自 `ModelConfig.layers`，每个 Layer 定义了：
- `layer_idx`：层序号（≥0 为真实层，-2=START层，-1=Prolog层，999999=Epilog层，1000000=END层）
- `repeat`：真实层的重复次数，其他层都为1
- `rank_ops`：`{rank_idx: [op_ids]}` 映射

### 2.2 耗时汇总

在 `_build_rank_response` 中，算子按 `layer_idx` 聚合到 Layer：

```
// 在同一rank同一层内所有算子的时间跨度
layer_start_ns  = min(op.start_time_ns)   // 该层最早算子开始时间
layer_end_ns    = max(op.end_time_ns)     // 该层最晚算子结束时间
layer_cost_ns   = layer_end_ns - layer_start_ns
```

### 2.3 时间轴调整（`_adjust_start_end_time`）

按 `layer_idx` 排序后，按 `repeat` 展开为连续时间轴：

```
cumulative_ns = 0
for layer in sorted_layers:
    // 第1次 repeat 内的层
    layer.start_time_ns = cumulative_ns
    layer.end_time_ns   = cumulative_ns + layer_cost_ns
    
    // 该层内算子的时间偏移到全局时间轴
    for op_id in layer.op_ids:
        op.start_time_ns = cumulative_ns + op.start_time_ns   // FIXME: 当前实现有偏差
        op.end_time_ns   = op.start_time_ns + op.total_cost_us × 1000
    
    cumulative_ns += layer_cost_ns × repeat   // 累加 repeat 次
```

---

## 3. Rank 级别

### 3.1 数据来源

Rank 信息来自 `ModelConfig.ranks`，每个 Rank 定义了：
- `rank_idx`：Rank 编号
- `ops`：该 Rank 拥有的算子 ID 列表

### 3.2 耗时汇总

```
rank_start_ns = min(op.start_time_ns)   // 该 Rank 最早算子开始时间
rank_end_ns   = max(op.end_time_ns)     // 该 Rank 最晚算子结束时间
rank_total_ms = (rank_end_ns - rank_start_ns) / 1_000_000   // ns → ms
```

### 3.3 显存计算

```
param_bytes = Σ(layer.param_bytes × layer.repeat)   // 所有层的参数字节 × repeat
io_bytes    = Σ(layer.io_bytes × layer.repeat)      // 所有层的 IO 字节 × repeat
peak_mem_gb = noise_gb + (param_bytes + io_bytes) / 1024³
oom         = peak_mem_gb > mem_capacity_gb
```

---

## 4. 模型级别（Model）

### 4.1 总体耗时

```
model_start_ns = min(rank.start_time_ns)    // 所有 Rank 的最早开始时间
model_end_ns   = max(rank.end_time_ns)      // 所有 Rank 的最晚结束时间
overall_cost_ms = (model_end_ns - model_start_ns) / 1_000_000   // ns → ms
```

### 4.2 派生指标

```
// avg_accept_tokens: 每次 decode step 平均输出 token 数（含 MTP）
// 对于 prefill 阶段，avg_accept_tokens = 1
// 对于 decode 阶段，avg_accept_tokens = 1 + Σ(p.ratio_mtp_tokens^i) for i=0..num_mtp_tokens

tpot_ms = overall_cost_ms / avg_accept_tokens   // 每个输出 token 的耗时
tps     = batch_size / (tpot_ms / 1000)           // tokens per second
qps     = 1000 / overall_cost_ms                  // queries per second
```

### 4.3 并行策略字符串

```
strategy = "TP{tp}_DP{dp}_EP{ep}"   // 仅包含值 > 1 的维度
```

---

## 5. 单位汇总

| 字段 | 单位 | 取值范围 |
|------|------|---------|
| `compute_flops` | 条浮点操作（次） | 由算子表达式决定 |
| `spec_cube_fp16` | GFLOPS | ~2250 (B300) |
| `spec_vect_fp16` | GFLOPS | ~160 (B300) |
| `spec_bw_memory` | TiB/s（存储时）×1024→GiB/s | 8.0 TB/s (B300) |
| `compute_cost_us` | 微秒 (us) | 由 FLOPS/带宽决定 |
| `total_cost_us` | 微秒 (us) | 单次算子执行 |
| `static_cost_us` | 微秒 (us) | 2/3/5/10（按算子类型） |
| `start_time_ns` | 纳秒 (ns) | 全局时间轴 |
| `end_time_ns` | 纳秒 (ns) | 全局时间轴 |
| `layer_cost_ns` | 纳秒 (ns) | 单次层执行 |
| `total_cost_ms` | 毫秒 (ms) | Rank/模型总耗时 |
| `peak_mem_gb` | GiB | 显存占用 |
| `ttot_ms` | 毫秒 (ms) | 模型总时延 |
| `tpot_ms` | 毫秒 (ms) | 每 token 时延 |
| `tps` | tokens/sec | 吞吐 |
| `qps` | queries/sec | 查询吞吐 |

## 6. 核心公式速查

```
算子单次耗时:  total_cost_us  = max(comp_us, mem_us) + static_cost_us
层跨度:        layer_cost_ns  = max(op.end_time_ns) - min(op.start_time_ns)
层累计:        layer_total    = layer_cost_ns × repeat × num_ranks
Rank 耗时:     rank_total_ms  = (rank_end_ns - rank_start_ns) / 1e6
模型耗时:      ttot_ms        = model_span_ns / 1e6
每 Token 耗时: tpot_ms        = ttot_ms / avg_accept_tokens
吞吐:          tps            = batch_size / (tpot_ms / 1000)
```

## 7. 注意事项

1. **显存带宽单位**：存储时使用 TB/s（如 8.0），加载到 `spec_bw_memory` 时乘以 1024 转换为 GiB/s，使公式 `bytes / 1024 / bw` 正确得到微秒。

2. **1024 vs 1000**：计算耗时中 `compute_cost_ns / 1000` 使用 1000（十进制），因为 GFLOPS = 10^9 FLOPS/s。显存带宽公式中 `/1024` 将 bytes 转为 KiB，配合 GiB/s 得到微秒。

3. **repeat 折叠**：`_adjust_start_end_time` 将 layer 按 `repeat` 次数在时间轴上展开。`layer_cost_ns × repeat` 累加到 `cumulative_ns`。

4. **算子时间偏差**：当前 `r.start_time_ns = cumulative_ns + r.start_time_ns` 将算子原始绝对时间累加到层基址，存在重复计算风险。正确做法应先保存层原始起始时间，计算算子相对偏移后再叠加。

5. **通信层级选择**：
   - `rank_size ≤ bwsio_limit`：片内通信（NVLink/EP总线）
   - `rank_size ≤ superpod_limit`：节点内通信（InfiniBand/RoCE）
   - `rank_size > superpod_limit`：跨节点通信（跨机架）
