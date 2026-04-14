from typing import List, Mapping, Optional

from python.zrt.common.tensor_base import TensorBase


class Node:
    """A single captured op. One CSV row → one Node."""

    _next_n_id: int = 0

    @classmethod
    def _get_next_n_id(cls) -> int:
        n_id = cls._next_n_id
        cls._next_n_id += 1
        return n_id

    def __init__(
        self,
        index: int,
        aten_op: str,
        layer: Optional[int],
        module_path: str,
        component: str,
        stream: int = 0,
        inputs: Optional[List[TensorBase]] = None,
        outputs: Optional[List[TensorBase]] = None,
    ):
        self.n_id = Node._get_next_n_id()

        self.index = index
        self.op_name = aten_op
        self.layer = layer
        self.module_path = module_path
        self.component = component
        self.stream = stream

        self.inputs: List[TensorBase] = inputs if inputs is not None else []
        self.outputs: List[TensorBase] = outputs if outputs is not None else []

    @classmethod
    def from_csv_row(cls, row: Mapping[str, str]) -> "Node":
        """Build a Node from a csv.DictReader row.

        Strips whitespace from every field — the reference CSV is space-padded
        for readability.
        """
        def g(key: str) -> str:
            for k, v in row.items():
                if k.strip() == key:
                    return (v or "").strip()
            raise KeyError(key)

        layer_str = g("Layer")
        layer = int(layer_str) if layer_str else None

        return cls(
            index=int(g("Index")),
            aten_op=g("Aten Op"),
            module_path=g("Module Path"),
            layer=layer,
            component=g("Component"),
            inputs=TensorBase.from_parsed(g("Input Shapes"), g("Input Dtypes")),
            outputs=TensorBase.from_parsed(g("Output Shapes"), g("Output Dtypes")),
        )

    @classmethod
    def from_csv(cls, path: str) -> List["Node"]:
        """Parse a captured-ops CSV into a list of Nodes, preserving row order."""
        import csv

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            return [cls.from_csv_row(row) for row in reader]

    def clone(self) -> "Node":
        """Return a deep copy of this Node.

        Tensor metadata lists are copied so downstream passes can mutate
        inputs/outputs without affecting the source graph.
        """
        return Node(
            index=self.index,
            aten_op=self.op_name,
            layer=self.layer,
            module_path=self.module_path,
            component=self.component,
            stream=self.stream,
            inputs=list(self.inputs),
            outputs=list(self.outputs),
        )

    def __repr__(self) -> str:
        layer = "" if self.layer is None else f" L{self.layer}"
        return (
            f"Node(n_id={self.n_id} #{self.index} {self.op_name}{layer} "
            f"{self.component!r} in={len(self.inputs)} out={len(self.outputs)})"
        )
