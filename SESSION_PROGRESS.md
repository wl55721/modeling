# Session Progress

## 当前文件状态

| 文件 | 状态 |
|------|------|
| `ARCHITECTURE.md` | ✅ V2 完成，含 10 个章节 |
| `python/zrt/graph/*` | ✅ 现有图抓取+融合引擎，可用 |
| `python/zrt/ir/` | ✅ OpGraph IR 完整实现 + NetworkX 适配器 |
| `python/zrt/graph/main.py` | ✅ capture 层已迁移：`run_trace/run_trace_phases` 输出 `OpGraph` |
| `python/zrt/hardware/` | ✅ 完整实现：spec.py + registry.py + 5 个 YAML 配置 |
| `python/zrt/simulator/` | ✅ 完整实现：Phase 1 核心 Roofline 仿真器 |
| `python/zrt/transform/` | ✅ 完整实现：4-stage Transform Pipeline |
| `python/zrt/executor/` | ✅ 完整实现：DAGScheduler + Timeline |

## 已解决的问题

### transform/ 模块实现（本次）

**方案**：方案 B（先抓图再变换），两者都要（拓扑变化 + stream 注解）

**新增文件**：
- `python/zrt/transform/__init__.py`：公开 API
- `python/zrt/transform/base.py`：GraphPass ABC
- `python/zrt/transform/context.py`：TransformContext, ParallelConfig, StreamConfig, QuantConfig
- `python/zrt/transform/pipeline.py`：TransformPipeline + build_default_pipeline()
- `python/zrt/transform/parallel/tensor_parallel.py`：TensorParallelPass（column/row parallel shape修改 + 注解）
- `python/zrt/transform/parallel/expert_parallel.py`：ExpertParallelPass（MoE EP 注解）
- `python/zrt/transform/parallel/comm_inserter.py`：CommInserterPass（插入 comm.all_reduce / comm.all_to_all 节点，边重连）
- `python/zrt/transform/fusion/pass_.py`：FusionPass（现有引擎的 stub 适配器）
- `python/zrt/transform/optim/passes.py`：QuantizationPass / EPLBPass / SharedExpertPass / MTPPass
- `python/zrt/transform/analysis/passes.py`：FlopsPass + RooflinePass + StreamAssignPass

**核心设计**：
- TP column parallel：outputs 最后一维 / tp，同步更新出边 tensor shape
- TP row parallel：inputs[0] 最后一维 / tp，标注 comm_after=all_reduce
- CommInserter：在 row-parallel 节点后插入 comm.all_reduce，EP expert 块前后插入 comm.all_to_all，边重连正确
- StreamAssignPass：compute 节点 → stream 0..num_compute-1，comm 节点 → stream num_compute..total-1，round-robin

**测试**：
- `tests/test_transform.py`：18 个测试全部通过（2.08s）
- TP shape 修改验证
- comm 节点插入和边重连验证
- 多流 stream_id 分配验证（含多 compute/comm 流配置）
- 端到端 pipeline 验证

## 已完成（本次）：Executor / DAGScheduler

**文件**：
- `python/zrt/executor/scheduler.py`：DAGScheduler + Timeline + ScheduledOp
- `python/zrt/executor/__init__.py`：公开 API

**核心设计**：
- 拓扑序 + list scheduling：`start = max(前驱完成时间, 所在 stream 可用时间)`
- `latency_us` 优先从 annotations 读取，fallback 到 Roofline 估算（需传 hw_spec），再 fallback 到 1 µs
- `Timeline.overlap_us = compute_time + comm_time - total_latency`（量化通算掩盖收益）
- `Timeline.ops_on_stream(id)` 返回指定 stream 的按时间排序的 op 列表

**测试**：
- `tests/test_executor.py`：14 个测试全部通过（1.28s）
- 覆盖：单节点、线性链、依赖顺序、同 stream 串行化、不同 stream 并行、overlap 量化、无 overlap 线性链、latency fallback、完整 pipeline 集成

## 下一步待办

1. ~~**FusionPass 真正接入 OpGraph IR**~~ ✅ 已完成

2. ~~**E2ESummary 报表**~~ ✅ 已完成
   - `python/zrt/report/summary.py`：E2ESummary + build_summary()
   - 16 个测试全部通过

3. **capture 层迁移（可选后续）**
   - 将 `export_all` 从 `nx.DiGraph` 改为直接接收 `OpGraph`
