from typing import Dict, Iterator

import networkx as nx

from zrt.graph.node import Node


MAX_STREAMS_PER_RANK = 2


class Rank(nx.DiGraph):
    """A per-device computation subgraph.

    Nodes are `Node` instances; each carries its own `stream` attribute
    (0 or 1) to model CUDA stream overlap. A Rank is bound to a parent
    `GlobalGraph`; adding a node to the Rank also registers it on the global
    graph so both views stay in sync.
    """

    def __init__(self, rank_id: int, parent: "GlobalGraph"):
        super().__init__()
        self.rank_id = rank_id
        self.parent = parent

    def add_op_node(self, node: Node) -> None:
        if node.stream < 0 or node.stream >= MAX_STREAMS_PER_RANK:
            raise ValueError(
                f"node.stream {node.stream} out of range "
                f"[0, {MAX_STREAMS_PER_RANK})"
            )
        self.add_node(node)
        self.parent.add_node(node, rank=self.rank_id)

    def add_op_edge(self, src: Node, dst: Node) -> None:
        if src not in self or dst not in self:
            raise KeyError(
                f"edge endpoints must exist in rank {self.rank_id}: "
                f"{src!r} -> {dst!r}"
            )
        self.add_edge(src, dst)
        self.parent.add_edge(src, dst)

    def op_nodes(self) -> Iterator[Node]:
        return iter(self.nodes)

    def __repr__(self) -> str:
        return (
            f"Rank(id={self.rank_id}, "
            f"nodes={self.number_of_nodes()}, "
            f"edges={self.number_of_edges()})"
        )


class GlobalGraph(nx.DiGraph):
    """Multi-rank multi-stream directed graph.

    A `DiGraph` holding every `Node` across all ranks, plus the cross-rank
    edges (typically around communication operators). Each rank also keeps
    its own `Rank` subgraph view; nodes added to a `Rank` are mirrored here.
    """

    def __init__(self):
        super().__init__()
        self.ranks: Dict[int, Rank] = {}

    def create_rank(self, rank_id: int) -> Rank:
        if rank_id in self.ranks:
            raise ValueError(f"rank {rank_id} already exists")
        rank = Rank(rank_id, self)
        self.ranks[rank_id] = rank
        return rank

    def get_rank(self, rank_id: int) -> Rank:
        return self.ranks[rank_id]

    def add_cross_rank_edge(self, src: Node, dst: Node) -> None:
        """Add a dependency between nodes on different ranks.

        Both nodes must already be registered on the global graph (i.e.
        added via `Rank.add_op_node`).
        """
        if src not in self or dst not in self:
            raise KeyError(f"unknown node in cross-rank edge: {src!r} -> {dst!r}")
        src_rank = self.nodes[src].get("rank")
        dst_rank = self.nodes[dst].get("rank")
        if src_rank == dst_rank:
            raise ValueError(
                "cross-rank edge endpoints must be on different ranks; "
                f"both on rank {src_rank}"
            )
        self.add_edge(src, dst)

    def __repr__(self) -> str:
        return (
            f"GlobalGraph(ranks={len(self.ranks)}, "
            f"nodes={self.number_of_nodes()}, "
            f"edges={self.number_of_edges()})"
        )
