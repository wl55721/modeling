"""Graph-based operator fusion driven by JSON fusion rules.

Reads fusion rules (from JSON or FusionSpec list), matches them against a
ComputeGraph by walking successor chains, and produces a new fused graph
with correctly merged edges.

Usage::

    from screenshot_ops.fusion_pass import FusionRule, FusionPass
    rules = FusionRule.from_specs(specs)
    fused_graph, result = FusionPass(rules).apply(graph)
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from screenshot_ops.compute_graph import ComputeGraph


@dataclass
class FusionRule:
    """A fusion pattern loaded from JSON or derived from FusionSpec."""

    rule_name: str
    module_class: str
    aten_op_sequence: List[str]
    num_sub_ops: int = 0
    fusion_level: str = "leaf"
    occurrences: int = 1
    example_module_path: str = ""
    input_map: List[Dict] = field(default_factory=list)
    output_map: List[Dict] = field(default_factory=list)

    @classmethod
    def from_json(cls, path: str | Path) -> List["FusionRule"]:
        """Load fusion rules from a JSON file."""
        data = json.loads(Path(path).read_text())
        rules = []
        for entry in data:
            rules.append(cls(
                rule_name=entry.get("rule_name", entry.get("module_class", "")),
                module_class=entry["module_class"],
                aten_op_sequence=entry["aten_op_sequence"],
                num_sub_ops=entry.get("num_sub_ops", len(entry["aten_op_sequence"])),
                fusion_level=entry.get("fusion_level", "leaf"),
                occurrences=entry.get("occurrences", 1),
                example_module_path=entry.get("example_module_path", ""),
                input_map=entry.get("input_map", []),
                output_map=entry.get("output_map", []),
            ))
        return rules

    @classmethod
    def from_specs(cls, specs) -> List["FusionRule"]:
        """Create FusionRules from FusionSpec objects (fusion.py)."""
        rules = []
        for s in specs:
            rules.append(cls(
                rule_name=s.module_class,
                module_class=s.module_class,
                aten_op_sequence=s.aten_op_sequence,
                num_sub_ops=s.num_sub_ops,
                fusion_level=s.fusion_level,
                occurrences=s.occurrences,
                example_module_path=s.example_module_path,
                input_map=s.input_map,
                output_map=s.output_map,
            ))
        return rules

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule_name": self.rule_name,
            "module_class": self.module_class,
            "aten_op_sequence": self.aten_op_sequence,
            "num_sub_ops": self.num_sub_ops,
            "fusion_level": self.fusion_level,
            "occurrences": self.occurrences,
            "example_module_path": self.example_module_path,
            "input_map": self.input_map,
            "output_map": self.output_map,
        }


@dataclass
class FusionResult:
    """Result of applying fusion to a graph."""
    original_nodes: int
    fused_nodes: int
    fusions_applied: List[str] = field(default_factory=list)

    @property
    def nodes_eliminated(self) -> int:
        return self.original_nodes - self.fused_nodes

    def summary(self) -> str:
        lines = [
            f"Fusion: {self.original_nodes} ops -> {self.fused_nodes} ops "
            f"({self.nodes_eliminated} eliminated)",
            f"Fusions applied: {len(self.fusions_applied)}",
        ]
        for f in self.fusions_applied[:20]:
            lines.append(f"  - {f}")
        if len(self.fusions_applied) > 20:
            lines.append(f"  ... and {len(self.fusions_applied) - 20} more")
        return "\n".join(lines)


class FusionPass:
    """Apply fusion rules to a ComputeGraph, producing a new fused graph.

    The original graph is not modified.

    Algorithm (two-phase, inspired by xPU-simulator FusionPass):
      1. **Match phase**: Group nodes by ``(module_path, layer)``. Within
         each group, collect consecutive runs sharing the same
         ``module_class``. If a run's module_class matches a rule, the
         entire run is fused (regardless of the exact op sequence, since
         the run already represents the real op order from the trace).
      2. **Rewrite phase**: Build a new ComputeGraph. Create fused
         replacement nodes for matched groups, copy unmatched nodes,
         then rebuild edges — internal edges within a fused group are
         dropped, external edges are redirected through the fused node.
    """

    def __init__(self, rules: List[FusionRule]):
        self.rules = rules
        self._class_to_rule: Dict[str, FusionRule] = {}
        for r in rules:
            if r.module_class:
                self._class_to_rule[r.module_class] = r

    def apply(self, graph: ComputeGraph) -> Tuple[ComputeGraph, FusionResult]:
        original_count = graph.num_nodes
        fusions_applied: List[str] = []

        fused_ids: set = set()
        pending_fusions: List[Tuple[FusionRule, List[int]]] = []

        order = graph.topo_order()

        i = 0
        while i < len(order):
            node_id = order[i]
            if node_id in fused_ids:
                i += 1
                continue

            attrs = graph.node_attrs(node_id).get("attrs", {})
            module_class = attrs.get("module_class", "")
            module_path = attrs.get("module_path", "")
            layer = attrs.get("layer", "")

            rule = self._class_to_rule.get(module_class)
            if rule is None:
                i += 1
                continue

            run = [node_id]
            j = i + 1
            while j < len(order):
                next_id = order[j]
                if next_id in fused_ids:
                    break
                next_attrs = graph.node_attrs(next_id).get("attrs", {})
                if (next_attrs.get("module_class") == module_class
                        and next_attrs.get("module_path") == module_path
                        and next_attrs.get("layer") == layer):
                    run.append(next_id)
                    j += 1
                else:
                    break

            if len(run) >= 2:
                pending_fusions.append((rule, run))
                for n in run:
                    fused_ids.add(n)
                names = [graph.node_attrs(n).get("name", str(n)) for n in run]
                fusions_applied.append(
                    f"{rule.rule_name}({len(run)} ops): {' + '.join(names[:3])}"
                    + (f" + ... ({len(names) - 3} more)" if len(names) > 3 else "")
                )

            i = j if len(run) >= 2 else i + 1

        new_graph = ComputeGraph(graph.name + "_fused")
        node_map: Dict[int, int] = {}
        fused_replacement: Dict[int, int] = {}

        for rule, match_node_ids in pending_fusions:
            first_attrs = graph.node_attrs(match_node_ids[0])
            last_attrs = graph.node_attrs(match_node_ids[-1])

            fused_name = f"fused_{rule.rule_name}"
            fused_node_id = new_graph.add_node(
                op_name=fused_name,
                name=fused_name,
                attrs={
                    "rule_name": rule.rule_name,
                    "module_class": rule.module_class,
                    "aten_op_sequence": rule.aten_op_sequence,
                    "num_sub_ops": len(match_node_ids),
                    "fusion_level": rule.fusion_level,
                    "module_path": first_attrs.get("attrs", {}).get("module_path", ""),
                    "layer": first_attrs.get("attrs", {}).get("layer", ""),
                    "component": first_attrs.get("attrs", {}).get("component", ""),
                    "input_shapes": first_attrs.get("attrs", {}).get("input_shapes", ""),
                    "input_dtypes": first_attrs.get("attrs", {}).get("input_dtypes", ""),
                    "output_shapes": last_attrs.get("attrs", {}).get("output_shapes", ""),
                    "output_dtypes": last_attrs.get("attrs", {}).get("output_dtypes", ""),
                    "input_map": rule.input_map,
                    "output_map": rule.output_map,
                    "_matched_node_ids": match_node_ids,
                },
            )

            for old_id in match_node_ids:
                fused_replacement[old_id] = fused_node_id

        for node_id in graph.topo_order():
            if node_id in fused_ids:
                continue
            old_attrs = graph.node_attrs(node_id)
            new_id = new_graph.add_node(
                op_name=old_attrs["op_name"],
                name=old_attrs["name"],
                attrs=old_attrs.get("attrs", {}),
            )
            node_map[node_id] = new_id

        for old_id, new_id in fused_replacement.items():
            node_map[old_id] = new_id

        seen_edges: set = set()
        for src_id in graph.topo_order():
            src_new = node_map.get(src_id)
            if src_new is None:
                continue
            for dst_id in graph.successors(src_id):
                dst_new = node_map.get(dst_id)
                if dst_new is None:
                    continue
                if src_new == dst_new:
                    continue
                edge_key = (src_new, dst_new)
                if edge_key not in seen_edges:
                    new_graph.add_edge(src_new, dst_new)
                    seen_edges.add(edge_key)

        fusion_result = FusionResult(
            original_nodes=original_count,
            fused_nodes=new_graph.num_nodes,
            fusions_applied=fusions_applied,
        )

        return new_graph, fusion_result


def export_fusion_rules_json(
    rules: List[FusionRule],
    output_path: Path,
) -> Path:
    """Export fusion rules to a JSON file alongside the Excel output."""
    json_path = output_path.with_name(output_path.stem + "_fusion_rules.json")
    json_data = [r.to_dict() for r in rules]
    json_path.write_text(json.dumps(json_data, indent=2))
    return json_path


def load_fusion_rules_json(path: str | Path) -> List[FusionRule]:
    """Load fusion rules from a JSON file."""
    return FusionRule.from_json(path)
