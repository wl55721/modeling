# Runner

Runner is a lightweight graph-based LLM model runner. It takes a `GlobalGraph` as input and traverses every node in topological order, assigning each a simulated `(start, end, duration)` to produce a timeline for the whole multi-rank multi-stream graph.

## Simulation Workflow

Simulation runs as a three-step pipeline:

1. **Per-node duration** — query the `ops/` cost model for every node's ideal duration.
2. **Ideal timeline** — topo-order walk that assigns `(start, end)` under same/cross-stream and cross-rank ordering rules, ignoring dual-stream contention.
3. **Contention correction** — scan the ideal timeline, apply the dual-stream resource-sharing scale on overlapping windows, and propagate stretched durations to a fixed point.

The separation keeps step 2 a pure topo-order walk and confines the nonlinear part of the simulation to step 3.

---

### Step 1 — Per-Node Duration

Duration comes from `ops/` cost models, keyed on `(op_type, inputs, outputs)`:

- **Theoretical** — roofline from shape / dtype / op type.
- **DB** — measured lookup keyed by op signature.

The Runner is agnostic to which model is used; it only consumes `OpResult` fields (`static_cost`, `compute_time`, `memory_time`, …).

### Step 2 — Ideal Timeline

For each node, let `preds` be its predecessors across the entire global graph (both same-rank and cross-rank edges):

```
start    = max(pred.end for pred in preds)  # 0 if no predecessors
end      = start + duration
```

Ordering rules:

- **Same rank, same stream** — predecessors on the same stream enforce strict serialization.
- **Same rank, different streams** — the two streams (compute / communication) can overlap; they synchronize only where an explicit dataflow edge bridges them.
- **Cross-rank** — communication ops anchor both endpoints; downstream nodes on the receiving rank wait for the matching send to complete.

### Step 3 — Contention Correction

When two streams on the same rank run concurrently, they share the device's compute units and HBM bandwidth — they do not each get a full copy. For every overlap window between stream A and stream B, the Runner applies a resource-contention scale:

```
duration_effective = duration_ideal * contention_scale(resource_mix)
```

- **Compute-bound × compute-bound** — both contend for SM / tensor cores; scale closer to 1 / (combined utilization fraction).
- **Memory-bound × memory-bound** — both contend for HBM bandwidth; similar scaling.
- **Compute-bound × memory-bound** — minimal contention; scale close to 1.0 — the desired overlap case (e.g. communication kernels overlapping dense matmuls).

Because stretching one node shifts the start times of its successors, correction iterates until the timeline reaches a fixed point. A node's effective duration may therefore be piecewise across its `[start, end]` range.

---

## Output

Per node:

| Field | Meaning |
|-------|---------|
| `start` | Simulated start time (µs) |
| `end` | Simulated end time (µs) |
| `duration` | `end - start`, after contention correction |

Aggregations (per rank, per stream, critical path, total latency, peak memory) are derived from these per-node timings.
