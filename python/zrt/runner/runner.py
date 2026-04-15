from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List

import networkx as nx

from zrt.common.chip_spec import ChipSpec
from zrt.graph.graph import GlobalGraph
from zrt.graph.node import Node
from zrt.ops.op_base import OpResult


@dataclass
class Timing:
    start: float     # us
    end: float       # us
    duration: float  # us


CostFn = Callable[[Node, ChipSpec], OpResult]
ContentionFn = Callable[[Node, Node], float]


from zrt.ops.op_base import get_class_by_name

def _default_cost(node: Node, chip: ChipSpec) -> OpResult:
    try:
        # 根据 node 的 op_name 获取对应的算子类
        op_class = get_class_by_name(node.op_name)
        # 创建算子实例
        op = op_class(chip)
        # 设置输入和输出张量
        op.inputs = node.inputs
        op.outputs = node.outputs
        # 调用 get_memory_cost 方法获取成本信息
        return op.get_memory_cost()
    except ValueError:
        # 如果找不到对应的算子类，返回默认值
        return OpResult(
            static_cost=0.0,
            total_compute_flops=0.0,
            total_compute_time=1.0,
            compute_formula="",
            total_memory_bytes=0.0,
            total_memory_time=0.0,
            memory_formula="",
        )


def _default_contention(a: Node, b: Node) -> float:
    return 1.0


class Runner:
    """Simulate a `GlobalGraph` and produce per-node `(start, end, duration)`.

    Four-step workflow (see `runner/README.md`):

    1. per-node `OpResult` via `cost_fn`, from which ideal duration and
       memory footprint are derived
    2. ideal timeline via topo-order walk
    3. contention correction via `contention_fn` on same-rank cross-stream
       overlaps, iterated until a fixed point
    4. per-rank peak instantaneous memory from the final timeline
    """

    def __init__(
        self,
        graph: GlobalGraph,
        chip_spec: ChipSpec,
        cost_fn: CostFn = _default_cost,
        contention_fn: ContentionFn = _default_contention,
        max_contention_iters: int = 16,
        convergence_eps: float = 1e-6,
    ):
        self.graph = graph
        self.chip_spec = chip_spec
        self.cost_fn = cost_fn
        self.contention_fn = contention_fn
        self.max_contention_iters = max_contention_iters
        self.convergence_eps = convergence_eps

        self.op_results: Dict[Node, OpResult] = {}
        self.durations: Dict[Node, float] = {}
        self.timings: Dict[Node, Timing] = {}
        self.peak_memory: Dict[int, float] = {}  # rank -> bytes

    # ------------------------------------------------------------------ run

    def run(self) -> Dict[Node, Timing]:
        self._step1_op_results()
        self._step2_ideal_timeline()
        self._step3_contention_correction()
        self._step4_peak_memory()
        return self.timings

    # --------------------------------------------------------------- step 1

    def _step1_op_results(self) -> None:
        self.op_results = {n: self.cost_fn(n, self.chip_spec) for n in self.graph.nodes}
        self.durations = {n: r.duration() for n, r in self.op_results.items()}

    # --------------------------------------------------------------- step 2

    def _step2_ideal_timeline(self) -> None:
        timings: Dict[Node, Timing] = {}
        for node in nx.topological_sort(self.graph):
            preds = list(self.graph.predecessors(node))
            start = max((timings[p].end for p in preds), default=0.0)
            dur = self.durations[node]
            timings[node] = Timing(start=start, end=start + dur, duration=dur)
        self.timings = timings

    # --------------------------------------------------------------- step 3

    def _step3_contention_correction(self) -> None:
        for _ in range(self.max_contention_iters):
            if not self._apply_contention_scales():
                return
            self._step2_ideal_timeline()

    def _apply_contention_scales(self) -> bool:
        """Stretch each node's duration by the worst-case contention scale it
        sees from any concurrent node on the other stream of the same rank.
        Returns True if any duration changed.
        """
        streams_by_rank = self._group_nodes_by_rank_stream()
        changed = False
        new_durations = dict(self.durations)

        for rank_streams in streams_by_rank.values():
            if len(rank_streams) < 2:
                continue
            all_nodes = [n for nodes in rank_streams.values() for n in nodes]
            for a in all_nodes:
                worst = 1.0
                for b in all_nodes:
                    if a is b or a.stream == b.stream:
                        continue
                    if not self._overlaps(a, b):
                        continue
                    scale = self.contention_fn(a, b)
                    if scale > worst:
                        worst = scale
                stretched = self.durations[a] * worst
                if stretched - new_durations[a] > self.convergence_eps:
                    new_durations[a] = stretched
                    changed = True

        if changed:
            self.durations = new_durations
        return changed

    # --------------------------------------------------------------- step 4

    def _step4_peak_memory(self) -> None:
        """Compute per-rank peak instantaneous memory.

        A node is memory-active during its `[start, end]` window and contributes
        `OpResult.total_memory_bytes` bytes. For each rank we sweep time-ordered
        alloc/free events and record the maximum concurrent footprint.
        """
        peak: Dict[int, float] = {}
        for rank, streams in self._group_nodes_by_rank_stream().items():
            events: List[tuple] = []
            for nodes in streams.values():
                for n in nodes:
                    bytes_ = self.op_results[n].peak_memory()
                    if bytes_ <= 0:
                        continue
                    t = self.timings[n]
                    events.append((t.start, 0, bytes_))   # alloc first at same t
                    events.append((t.end, 1, -bytes_))    # free after alloc ties
            events.sort()
            running = 0.0
            rank_peak = 0.0
            for _, _, delta in events:
                running += delta
                if running > rank_peak:
                    rank_peak = running
            peak[rank] = rank_peak
        self.peak_memory = peak

    # -------------------------------------------------------------- helpers

    def _group_nodes_by_rank_stream(self) -> Dict[int, Dict[int, List[Node]]]:
        out: Dict[int, Dict[int, List[Node]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for node in self.graph.nodes:
            rank = self.graph.nodes[node].get("rank")
            if rank is None:
                continue
            out[rank][node.stream].append(node)
        return out

    def _overlaps(self, a: Node, b: Node) -> bool:
        ta = self.timings[a]
        tb = self.timings[b]
        return ta.start < tb.end and tb.start < ta.end

    # ---------------------------------------------------------------- stats

    def total_latency(self) -> float:
        return max((t.end for t in self.timings.values()), default=0.0)
