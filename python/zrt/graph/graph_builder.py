import networkx as nx

from python.zrt.config.runtime_config import RuntimeConfig
from python.zrt.graph.graph import GlobalGraph


DRIVER_RANK_ID = 0


class GraphBuilder:
    """Seed a single-rank `GlobalGraph` from a fused op sequence.

    Parallelism- and feature-aware rewrites (TP/DP/EP, MTP, cross-rank
    communication, stream assignment, etc.) happen downstream in the Adapter.
    """

    def __init__(self, raw_graph: nx.DiGraph, rt_config: RuntimeConfig):
        self.raw_graph = raw_graph
        self.rt_config = rt_config

    def build(self) -> GlobalGraph:
        gg = GlobalGraph()
        rank = gg.create_rank(DRIVER_RANK_ID)
        clones = {node: node.clone() for node in self.raw_graph.nodes()}
        for clone in clones.values():
            rank.add_op_node(clone)
        for src, dst in self.raw_graph.edges():
            rank.add_op_edge(clones[src], clones[dst])
        return gg
