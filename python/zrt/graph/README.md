# Global Graph

The global graph is a multi_rank multi_stream directed graph.

## Rank

Each rank represents a compuation graph on actual device.

### Keypoints

- Each rank is a subgraph.
- There are at most 2 stream in each rank.
- There are dependencies between ranks, usually around communication operators.

## Node

Each node represents an op, holding an op instance and its inputs. A Node carries the following attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `index` | `int` | Unique id of the node within the global graph. |
| `op_name` | `str` | Name of the op (e.g. `aten.add.Tensor`). |
| `layer` | `Optional[int]` | Layer index, if the op belongs to a transformer block. |
| `module_path` | `str` | Fully-qualified module path where the op was invoked. |
| `component` | `str` | High-level component label (e.g. `attn.q_proj`, `moe.gate`). |
| `stream` | `int` | Stream id (0 or 1) — used to model compute / communication overlap. |
| `inputs` | `List[TensorBase]` | Input tensor metadata (shape + dtype). |
| `outputs` | `List[TensorBase]` | Output tensor metadata (shape + dtype). |

## Builder

`GraphBuilder` turns the fused op sequence produced by Capturer into a `GlobalGraph`. It takes:

- `raw_graph: nx.DiGraph` — the linear fused op sequence from Capturer, with one node per fused op and edges representing dataflow.
- `rt_config: RuntimeConfig` — parallelism, disaggregation, and runtime-feature settings.

Flow:

1. **Seed the GlobalGraph** — create a single `Rank` for the driver and mirror every op from `raw_graph` onto it. At this stage the GlobalGraph is a one-rank, one-stream copy of the fused sequence.
2. **Hand off to the Adapter** — the Adapter consumes this seeded GlobalGraph and rewrites it per `rt_config`: splitting it across ranks for TP/DP/EP, inserting communication ops, adding cross-rank edges, and assigning stream ids so compute and communication can overlap.

The Builder itself stays minimal — it constructs the faithful initial graph; all feature- and parallelism-aware rewrites live in the Adapter.