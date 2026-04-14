from typing import Dict

import pytest

from zrt.common.chip_spec import ChipSpec, Vendor
from zrt.common.tensor_base import DType
from zrt.graph.graph import GlobalGraph
from zrt.graph.node import Node
from zrt.ops.op_base import OpResult
from zrt.runner.runner import Runner, Timing


# ---------------------------------------------------------------- fixtures


@pytest.fixture
def chip() -> ChipSpec:
    return ChipSpec(
        name="test-chip",
        vendor=Vendor.NVIDIA,
        cube_tflops={DType.FLOAT16: 1000.0},
        vector_tflops={DType.FLOAT32: 100.0},
        hbm_bandwidth_gbps=1000.0,
        hbm_capacity_gb=80.0,
    )


def _make_node(index: int, op: str = "aten::add", stream: int = 0) -> Node:
    return Node(
        index=index,
        aten_op=op,
        layer=None,
        module_path="m",
        component="c",
        stream=stream,
    )


def _fixed_cost(
    table: Dict[int, OpResult],
) -> "callable":
    """Cost fn that returns per-node-index pre-built OpResults."""
    def fn(node: Node, chip: ChipSpec) -> OpResult:
        return table[node.index]
    return fn


def _dump(runner: Runner) -> None:
    """Pretty-print per-node OpResult + Timing. Visible with `pytest -s`."""
    print()
    print(f"--- Runner dump (total_latency={runner.total_latency():.3f}us) ---")
    header = f"{'node':<32} {'start':>8} {'end':>8} {'dur':>8} {'mem_B':>10}"
    print(header)
    for node, t in sorted(runner.timings.items(), key=lambda kv: kv[1].start):
        r = runner.op_results[node]
        print(
            f"{repr(node)[:32]:<32} "
            f"{t.start:>8.3f} {t.end:>8.3f} {t.duration:>8.3f} "
            f"{r.peak_memory():>10.0f}"
        )
    for rank, peak in runner.peak_memory.items():
        print(f"  rank {rank} peak_memory = {peak:.0f} B")


def _simple_result(duration_us: float, mem_bytes: float = 0.0) -> OpResult:
    return OpResult(
        static_cost=0.0,
        total_compute_flops=0.0,
        total_compute_time=duration_us,
        compute_formula="",
        total_memory_bytes=mem_bytes,
        total_memory_time=0.0,
        memory_formula="",
    )


# --------------------------------------------------------- ideal timeline


def test_single_chain_topo_order(chip):
    g = GlobalGraph()
    r = g.create_rank(0)
    a, b, c = _make_node(0), _make_node(1), _make_node(2)
    for n in (a, b, c):
        r.add_op_node(n)
    r.add_op_edge(a, b)
    r.add_op_edge(b, c)

    costs = {0: _simple_result(1.0), 1: _simple_result(2.0), 2: _simple_result(3.0)}
    runner = Runner(g, chip_spec=chip, cost_fn=_fixed_cost(costs))
    timings = runner.run()
    _dump(runner)

    assert timings[a] == Timing(start=0.0, end=1.0, duration=1.0)
    assert timings[b] == Timing(start=1.0, end=3.0, duration=2.0)
    assert timings[c] == Timing(start=3.0, end=6.0, duration=3.0)
    assert runner.total_latency() == 6.0


def test_parallel_branches_start_at_max_pred(chip):
    # a -> b, a -> c, (b,c) -> d
    g = GlobalGraph()
    r = g.create_rank(0)
    a, b, c, d = (_make_node(i) for i in range(4))
    for n in (a, b, c, d):
        r.add_op_node(n)
    r.add_op_edge(a, b)
    r.add_op_edge(a, c)
    r.add_op_edge(b, d)
    r.add_op_edge(c, d)

    costs = {
        0: _simple_result(1.0),
        1: _simple_result(2.0),   # b ends at 3
        2: _simple_result(5.0),   # c ends at 6 -> d starts at 6
        3: _simple_result(1.0),
    }
    runner = Runner(g, chip_spec=chip, cost_fn=_fixed_cost(costs))
    timings = runner.run()
    _dump(runner)

    assert timings[d].start == 6.0
    assert timings[d].end == 7.0


# ------------------------------------------------------ contention correction


def test_contention_stretches_overlapping_nodes(chip):
    """Two nodes on two streams of the same rank overlap; contention_fn=2.0
    should double durations until fixed point."""
    g = GlobalGraph()
    r = g.create_rank(0)
    a = _make_node(0, stream=0)
    b = _make_node(1, stream=1)
    r.add_op_node(a)
    r.add_op_node(b)

    costs = {0: _simple_result(10.0), 1: _simple_result(10.0)}
    runner = Runner(
        g,
        chip_spec=chip,
        cost_fn=_fixed_cost(costs),
        contention_fn=lambda x, y: 2.0,
    )
    runner.run()
    _dump(runner)

    # Symmetric: both streams stretched against the other.
    assert runner.durations[a] >= 20.0
    assert runner.durations[b] >= 20.0


def test_no_contention_when_non_overlapping(chip):
    g = GlobalGraph()
    r = g.create_rank(0)
    a = _make_node(0, stream=0)
    b = _make_node(1, stream=1)
    r.add_op_node(a)
    r.add_op_node(b)
    # Force ordering via cross-stream edge so they don't overlap
    r.add_op_edge(a, b)

    costs = {0: _simple_result(5.0), 1: _simple_result(5.0)}
    runner = Runner(
        g,
        chip_spec=chip,
        cost_fn=_fixed_cost(costs),
        contention_fn=lambda x, y: 10.0,
    )
    runner.run()
    _dump(runner)

    assert runner.durations[a] == 5.0
    assert runner.durations[b] == 5.0


# -------------------------------------------------------------- peak memory


def test_peak_memory_per_rank(chip):
    """Rank 0: two overlapping nodes on different streams, 100 + 200 bytes.
    Peak should be 300."""
    g = GlobalGraph()
    r0 = g.create_rank(0)
    r1 = g.create_rank(1)

    a = _make_node(0, stream=0)  # rank 0
    b = _make_node(1, stream=1)  # rank 0
    c = _make_node(2, stream=0)  # rank 1
    r0.add_op_node(a)
    r0.add_op_node(b)
    r1.add_op_node(c)

    costs = {
        0: _simple_result(10.0, mem_bytes=100.0),
        1: _simple_result(10.0, mem_bytes=200.0),
        2: _simple_result(10.0, mem_bytes=50.0),
    }
    runner = Runner(g, chip_spec=chip, cost_fn=_fixed_cost(costs))
    runner.run()
    _dump(runner)

    assert runner.peak_memory[0] == 300.0
    assert runner.peak_memory[1] == 50.0


def test_peak_memory_single_node(chip):
    g = GlobalGraph()
    r = g.create_rank(0)
    a = _make_node(0, stream=0)
    r.add_op_node(a)

    costs = {0: _simple_result(5.0, mem_bytes=200.0)}
    runner = Runner(g, chip_spec=chip, cost_fn=_fixed_cost(costs))
    runner.run()
    _dump(runner)

    assert runner.peak_memory[0] == 200.0


# -------------------------------------------------------- chip_spec plumbing


def test_cost_fn_receives_chip_spec(chip):
    seen = {}

    def cost(node: Node, c: ChipSpec) -> OpResult:
        seen["chip"] = c
        return _simple_result(1.0)

    g = GlobalGraph()
    r = g.create_rank(0)
    r.add_op_node(_make_node(0))
    Runner(g, chip_spec=chip, cost_fn=cost).run()

    assert seen["chip"] is chip
