"""Lightweight data-flow graph built from dispatch-recorded ops.

This is the equivalent of xPU-simulator's DispatchExtractor._build_graph,
but without the full networkx DAG.  We only need two things for correct
fusion I/O mapping:

1. ``tensor_producer``: which *recorded* op (by record idx) produced each
   tensor ID.
2. ``passthroughs``: for skip ops (view/reshape/…), a mapping
   output_tensor_id → input_tensor_id so we can follow the chain through
   ops that were not recorded.

The key operation is ``resolve_id(tid)``: follow the passthrough chain until
we reach a tensor that was either (a) produced by a recorded op or (b) came
from outside the trace entirely.  This lets ``_compute_fused_io`` correctly
classify tensors as internal or external even when skipped reshape ops sit
between the producer and consumer.

Example problem this solves
---------------------------
Inside an MLP fused group::

    mul  (inside)  → produces tensor 500
    view (skipped) → produces tensor 501 (passthrough: 501 → 500)
    mm   (inside)  → consumes tensor 501

Without graph:
    produced_ids = {500, …}   (501 is not recorded as produced)
    consumed_ids = {501, …}
    external_input_ids = consumed − produced → 501 looks external  ✗

With graph:
    resolve_id(501) = 500  (follow passthrough)
    500 IS in produced_ids → 501 is internal  ✓
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DataFlowGraph:
    """Minimal data-flow index for a recorded op sequence.

    Attributes
    ----------
    tensor_producer:
        Maps tensor_id → record ``idx`` of the op that produced it.
        Only covers tensors produced by *recorded* (non-skip) ops.
    passthroughs:
        Maps skip-op output_tensor_id → skip-op input_tensor_id.
        Populated by ``TensorTracker`` during dispatch interception.
    """
    tensor_producer: Dict[int, int] = field(default_factory=dict)
    passthroughs: Dict[int, int] = field(default_factory=dict)

    def resolve_id(self, tid: int) -> int:
        """Follow the passthrough chain to the canonical tensor ID.

        The canonical ID is either the first one not in ``passthroughs``
        (i.e., produced by a recorded op or an external input) or, if a
        cycle is detected, the starting ID.
        """
        visited: set = set()
        while tid in self.passthroughs:
            if tid in visited:
                break  # cycle guard
            visited.add(tid)
            tid = self.passthroughs[tid]
        return tid

    def is_produced_by_any(self, tid: int, record_indices: set) -> bool:
        """Return True if *tid* (after passthrough resolution) was produced
        by one of the records in *record_indices*."""
        resolved = self.resolve_id(tid)
        producer = self.tensor_producer.get(resolved)
        return producer is not None and producer in record_indices


def build_graph(
    records: List[Dict[str, Any]],
    passthroughs: Dict[int, int],
) -> DataFlowGraph:
    """Build a ``DataFlowGraph`` from a list of op records and skip-op passthroughs.

    Parameters
    ----------
    records:
        The list of op records produced by ``RecordingDispatch`` (each has
        ``idx`` and ``_output_ids``).
    passthroughs:
        The ``TensorTracker.passthroughs`` dict populated during tracing.
    """
    tensor_producer: Dict[int, int] = {}
    for rec in records:
        for tid in rec.get("_output_ids", []):
            tensor_producer[tid] = rec["idx"]
    return DataFlowGraph(tensor_producer=tensor_producer, passthroughs=passthroughs)
