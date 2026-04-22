"""NetworkX-based computation graph for operator fusion."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


class ComputeGraph:
    """Directed acyclic graph of operations, wrapping networkx.DiGraph."""

    def __init__(self, name: str = "graph"):
        self.name = name
        self._graph = nx.DiGraph()
        self._next_id = 0

    def add_node(self, op_name: str, attrs: Optional[Dict[str, Any]] = None,
                 name: Optional[str] = None) -> int:
        node_id = self._next_id
        self._next_id += 1
        self._graph.add_node(
            node_id,
            op_name=op_name,
            name=name or op_name,
            attrs=attrs or {},
        )
        return node_id

    def add_edge(self, src_id: int, dst_id: int,
                 tensor_id: Optional[int] = None):
        self._graph.add_edge(src_id, dst_id, tensor_id=tensor_id)

    def topo_order(self) -> List[int]:
        return list(nx.topological_sort(self._graph))

    def predecessors(self, node_id: int) -> List[int]:
        return list(self._graph.predecessors(node_id))

    def successors(self, node_id: int) -> List[int]:
        return list(self._graph.successors(node_id))

    def node_attrs(self, node_id: int) -> Dict[str, Any]:
        return dict(self._graph.nodes[node_id])

    def edge_data(self, src_id: int, dst_id: int) -> Dict[str, Any]:
        return dict(self._graph.edges[src_id, dst_id])

    def edges(self) -> List[Tuple[int, int, Dict[str, Any]]]:
        return [(u, v, d) for u, v, d in self._graph.edges(data=True)]

    @property
    def nodes(self) -> List[int]:
        return list(self._graph.nodes)

    @property
    def num_nodes(self) -> int:
        return self._graph.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self._graph.number_of_edges()

    @property
    def nx_graph(self) -> nx.DiGraph:
        return self._graph

    def __repr__(self):
        return f"ComputeGraph({self.name!r}, nodes={self.num_nodes}, edges={self.num_edges})"
