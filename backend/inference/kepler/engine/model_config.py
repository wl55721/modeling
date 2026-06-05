"""
模型配置数据类 —— 与 simulate API 中 model_json 的结构一一对应。
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class OperatorConfig:
    op_id: int
    op_name: str
    layer_idx: int = -1
    rank_idx: int = 0
    op_module: str = ""
    inputs: list[dict] = field(default_factory=list)
    params: list[dict] = field(default_factory=list)
    outputs: list[dict] = field(default_factory=list)
    compute_flops: str = "0"

    @classmethod
    def from_dict(cls, d: dict) -> OperatorConfig:
        return cls(
            op_id=d.get("op_id", 0),
            op_name=d.get("op_name", ""),
            layer_idx=d.get("layer_idx", -1),
            rank_idx=d.get("rank_idx", 0),
            op_module=d.get("op_module", ""),
            inputs=d.get("inputs", []),
            params=d.get("params", []),
            outputs=d.get("outputs", []),
            compute_flops=d.get("compute_flops", "0"),
        )


@dataclass
class Layer:
    name: str
    layer_idx: int
    repeat: int = 1
    kind: str = "regular"
    ops: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Layer:
        ops: list[int] = []
        rank_ops_raw = d.get("rank_ops", {})
        if isinstance(rank_ops_raw, dict):
            for k in sorted(rank_ops_raw.keys(), key=int):
                v = rank_ops_raw[k]
                if isinstance(v, list):
                    ops.extend(v)
        return cls(
            name=d.get("name", ""),
            layer_idx=d.get("layer_idx", 0),
            repeat=d.get("repeat", 1),
            kind=d.get("kind", "regular"),
            ops=ops,
        )


@dataclass
class Edge:
    source: int
    target: int

    @classmethod
    def from_dict(cls, d: dict) -> Edge:
        return cls(
            source=d.get("from", d.get("source", 0)),
            target=d.get("to", d.get("target", 0)),
        )


@dataclass
class Rank:
    rank_idx: int
    ops: list[int] = field(default_factory=list)
    layer_ids: list[int] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict) -> Rank:
        layers = [Layer.from_dict(ld) for ld in d.get("layers", [])]
        # backward compat: build layer_ids from layers if not provided
        layer_ids = d.get("layer_ids", [])
        if not layer_ids:
            layer_ids = [l.layer_idx for l in layers]
        return cls(
            rank_idx=d.get("rank_idx", 0),
            ops=d.get("ops", []),
            layer_ids=layer_ids,
            layers=layers,
        )


@dataclass
class ModelConfig:
    """模型配置，与前端 exportModel 输出的 JSON 结构一一对应。"""
    name: str = ""
    operators: list[OperatorConfig] = field(default_factory=list)
    layers: list[Layer] = field(default_factory=list)
    edges: list[Edge] = field(default_factory=list)
    ranks: list[Rank] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict) -> ModelConfig:
        ranks = [Rank.from_dict(r) for r in data.get("ranks", [])]

        # collect layers from ranks (new format); fallback to top-level layers
        layers = []
        if any(r.layers for r in ranks):
            seen = set()
            for r in ranks:
                for layer in r.layers:
                    if layer.layer_idx not in seen:
                        layers.append(layer)
                        seen.add(layer.layer_idx)
        else:
            layers = [Layer.from_dict(l) for l in data.get("layers", [])]
            for rank in ranks:
                rank_ops_set = set(rank.ops)
                for layer in layers:
                    if any(op_id in rank_ops_set for op_id in layer.ops):
                        rank.layers.append(layer)

        return cls(
            name=data.get("name", ""),
            operators=[OperatorConfig.from_dict(o) for o in data.get("operators", [])],
            layers=layers,
            edges=[Edge.from_dict(e) for e in data.get("edges", [])],
            ranks=ranks,
        )

    @property
    def layer_repeat(self) -> dict[int, int]:
        """返回 {layer_idx: repeat} 映射，仅包含 repeat > 0 的层。"""
        repeat_dict = {
            l.layer_idx: l.repeat
            for l in self.layers
            if 0 <= l.layer_idx <= 1000 and l.repeat > 0
        }
        repeat_dict[-2] = 1
        repeat_dict[-1] = 1
        repeat_dict[900] = 1
        repeat_dict[1000] = 1

        return repeat_dict

    @property
    def total_operators(self) -> int:
        """考虑 repeat 后的总算子执行次数。"""
        repeat_map = self.layer_repeat
        total = 0
        for op in self.operators:
            total += repeat_map.get(op.layer_idx, 1)
        return total
