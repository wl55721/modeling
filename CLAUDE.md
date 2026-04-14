# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A lightweight graph-based execution model for simulating LLM inference performance. The pipeline traces ATen-level ops from HuggingFace models into a `networkx.DiGraph`, rewrites that graph to model parallelism and runtime features, then walks it in topological order against per-op cost models to estimate memory and latency.

## Environment & Commands

- **Python env**: `conda run -n py311` (Python 3.11)
- **Run tests**: `conda run -n py311 python -m pytest tests/ -v`
- **Run one test**: `conda run -n py311 python -m pytest tests/<file>::<Class>::<test> -v -s`
- **Log level**: `LOG_LEVEL` env var (DEBUG/INFO/WARNING/ERROR)

`python/` is not an installed package. Scripts and tests must prepend `sys.path.insert(0, "python")` before importing `zrt`.

## Architecture

Four cooperating components under `python/zrt/`:

### Capturer (`capturer/`)
Traces ATen-level operator sequences from HuggingFace model definitions (loaded on `torch.device("meta")` via `TorchDispatchMode`), records shapes/dtypes/module paths per op, and exports to CSV. The CSV is the boundary between capture and the rest of the system.

### Graph (`graph/`)
Builds a `networkx.DiGraph` from the captured CSV. Each node is an operator (or a fused group of operators). Supports operator fusion to produce higher-level computation graphs.

### Adapter (`adapter/`)
Rewrites the captured graph to model runtime features and parallelism. Target features:
- TP / DP / EP parallel
- MTP (DeepSeek-series)
- Prefix Cache
- Chunked Prefill
- Context Parallel via RingAttention

Adapter output is a **multi-rank multi-stream directed graph**: a global graph whose subgraphs represent individual ranks. Per-rank subgraphs may differ depending on enabled features. Each rank subgraph can carry multiple streams to model CUDA stream overlap (e.g. compute vs. communication).

### Runner (`runner/`)
Lightweight graph executor. Walks the `networkx.DiGraph` in topological order and assigns each node a `(start, end, duration)`. A node's start time is the max end time across its predecessors, accounting for stream and rank boundaries.

### Ops (`ops/`)
Cost models for simulating operator execution (memory + time):
- **Theoretical**: Roofline-based analytical model.
- **DB**: Looks up measured performance by op signature (inputs, dtype, shape).

### Common (`common/`)
Shared primitives — tensor metadata, logging, small utilities — used by every component above.

## Layout

```
docs/                  # design docs
python/zrt/            # core library ("Zhanlu Runtime")
  capturer/            # HF → CSV op trace
  graph/               # node types, graph builder, fusion
  adapter/             # feature/parallelism rewrites → multi-rank multi-stream graph
  runner/              # topo-order simulator
  ops/                 # cost models
  common/              # logger, tensor base, utilities
scripts/               # build / env / setup scripts
tests/                 # pytest suite
deepseek_v3_ops.csv    # reference op trace (DeepSeek-V3)
```

## Repo Status

The repo is currently an early skeleton — most subdirectories under `python/zrt/` are empty. Treat the component descriptions above as the intended design, and verify actual file layout with `ls` before assuming a module exists.
