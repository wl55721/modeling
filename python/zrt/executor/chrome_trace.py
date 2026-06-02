"""Chrome Trace exporter — convert pipeline scheduling results to JSON.

Supports three input types:
  - ``PPStitchedTimeline``   : device x microbatch pipeline grid tasks
  - ``Timeline`` (per-device) : per-op trace from DAGScheduler
  - ``List[Timeline]``       : multi-device per-op trace, one per GPU

Chrome Trace event format (JSON array of dicts):

  {
    "ph":  "X",            // complete event (has duration)
    "name": "op_name",     // display name
    "cat":  "compute",     // category for filtering
    "pid":  device_id,     // process = GPU device
    "tid":  stage_or_stream,  // thread = virtual stage or compute/comm stream
    "ts":   123456.0,      // timestamp in microseconds
    "dur":  789.0,         // duration in microseconds
    "args": {"desc": "..."}
  }

  VPP (Virtual Pipeline Parallelism): devices with multiple virtual stages
  show each stage on a separate thread (tid).  This gives the "mbs blocks"
  visual grouping in Chrome Trace.

Usage
-----
>>> from python.zrt.executor.chrome_trace import ChromeTraceExporter
>>> exporter = ChromeTraceExporter()
>>> exporter.export_stitched(result, "pipeline.json")      # PP grid (device view)
>>> exporter.export_per_stage(timelines, "detail.json")    # per-op detail
>>> exporter.export_combined(result, timelines, "all.json") # combined
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from python.zrt.executor.pp_stitcher import PPStitchedTimeline
    from python.zrt.executor.scheduler import ScheduledOp, Timeline


# ── colour palette ────────────────────────────────────────────────────────────

_FWD_COLOR = "#1a3a6b"
_BWD_COLOR = "#8FBC8F"
_BWD_DX_COLOR = "#6B8E6B"
_BWD_DW_COLOR = "#A8D8A8"

_COLORS: dict[str, str] = {
    "fwd_compute":        "good",
    "fwd_comm":           "olive",
    "fwd_p2p":            "terracotta",
    "bwd_compute":        "blue",
    "bwd_comm":           "purple",
    "bwd_p2p":            "magenta",
    "bwd_dx_compute":     "blue",
    "bwd_dw_compute":     "navy",
    "memory":             "yellow",
    "idle":               "grey",
    "bubble":             "light_grey",
    "warmup":             "orange",
    "cooldown":           "teal",
}

_NAMES: dict[str, str] = {
    "fwd_compute":        "[c]",
    "fwd_comm":           "[n]",
    "fwd_p2p":            "[p2p]",
    "bwd_compute":        "[c]",
    "bwd_comm":           "[n]",
    "bwd_p2p":            "[p2p]",
    "bwd_dx_compute":     "dX [c]",
    "bwd_dw_compute":     "dW [c]",
    "memory":             "MEM",
    "idle":               "idle",
    "bubble":             "bubble",
}


@dataclass
class ChromeTraceEvent:
    """A single Chrome Trace complete event (ph="X")."""

    name: str
    cat: str
    pid: int
    tid: int
    ts: float
    dur: float
    args: dict = field(default_factory=dict)
    color: str | None = None

    def to_dict(self) -> dict:
        d = {
            "ph": "X",
            "name": self.name,
            "cat": self.cat,
            "pid": self.pid,
            "tid": self.tid,
            "ts": self.ts,
            "dur": self.dur,
            "args": self.args,
        }
        if self.color:
            d["color"] = self.color
        return d


class ChromeTraceExporter:
    """Export pipeline scheduling results as Chrome Trace JSON.

    Parameters
    ----------
    time_unit : str
        "us" (default) or "ns".  Chrome Trace ``ts`` / ``dur`` fields
        are always in microseconds; ``ns`` multiplies by 1000.
    """

    _MIN_VISIBLE_US = 1.0

    def __init__(
        self,
        time_unit: str = "us",
        *,
        trace_ep_waves: bool = False,
        ep_wave_k: int = 0,
    ) -> None:
        self._mult = 1000.0 if time_unit == "ns" else 1.0
        self._time_unit = time_unit
        self._trace_ep_waves = trace_ep_waves
        self._ep_wave_k = max(0, int(ep_wave_k))

    # ── metadata helpers ──────────────────────────────────────────────────

    @staticmethod
    def _process_name_meta(pid: int, name: str) -> dict:
        return {"ph": "M", "pid": pid, "ts": 0, "name": "process_name", "args": {"name": name}}

    @staticmethod
    def _thread_name_meta(pid: int, tid: int, name: str) -> dict:
        return {"ph": "M", "pid": pid, "tid": tid, "ts": 0, "name": "thread_name", "args": {"name": name}}

    @staticmethod
    def _sort_index_meta(pid: int, sort_index: int) -> dict:
        return {"ph": "M", "pid": pid, "ts": 0, "name": "sort_index", "args": {"sort_index": sort_index}}

    @staticmethod
    def _thread_sort_index_meta(pid: int, tid: int, sort_index: int) -> dict:
        return {"ph": "M", "pid": pid, "tid": tid, "ts": 0, "name": "thread_sort_index", "args": {"sort_index": sort_index}}

    # ── device layout helpers (VPP support) ─────────────────────────────

    @staticmethod
    def _compute_device_layout(
        result: "PPStitchedTimeline",
    ) -> tuple[int, dict[int, list[int]], dict[tuple[int, int], int]]:
        """Compute VPP layout from stitched timeline.

        Returns
        -------
        vpp : int
            Max number of virtual stages per device.
        device_stages : dict[int, list[int]]
            device_id → sorted list of stage_ids on that device.
        stage_to_tid : dict[(device_id, stage_id), int]
            Maps (device, virtual_stage) → thread index on that device.
        """
        stages_per_dev: dict[int, set[int]] = {}
        for task in result.tasks:
            stages_per_dev.setdefault(task.stream_id, set()).add(task.stage_id)
        device_stages: dict[int, list[int]] = {
            d: sorted(s) for d, s in stages_per_dev.items()
        }
        vpp = max(len(s) for s in device_stages.values()) if device_stages else 1
        stage_to_tid: dict[tuple[int, int], int] = {}
        for d, stages in device_stages.items():
            for idx, s in enumerate(stages):
                stage_to_tid[(d, s)] = idx
        return vpp, device_stages, stage_to_tid

    def _grid_meta_events(
        self, pp: int, *, vpp: int = 1,
        device_stages: dict[int, list[int]] | None = None,
    ) -> list[dict]:
        meta: list[dict] = []
        for d in range(pp):
            meta.append(self._process_name_meta(d, f"GPU {d}"))
            meta.append(self._sort_index_meta(d, d))
            if vpp <= 1:
                meta.append(self._thread_name_meta(d, 0, "Grid Schedule"))
            else:
                stages = (device_stages or {}).get(d, [])
                for idx, s in enumerate(stages):
                    meta.append(self._thread_name_meta(d, idx, f"Stage {s}"))
                    meta.append(self._thread_sort_index_meta(d, idx, idx))
        return meta

    def _per_stage_meta_events(self, pp: int) -> list[dict]:
        meta: list[dict] = []
        for d in range(pp):
            meta.append(self._process_name_meta(d, f"GPU {d}"))
            meta.append(self._sort_index_meta(d, d))
            meta.append(self._thread_name_meta(d, 0, "Compute Ops"))
            meta.append(self._thread_sort_index_meta(d, 0, 0))
            meta.append(self._thread_name_meta(d, 1, "Comm Ops"))
            meta.append(self._thread_sort_index_meta(d, 1, 1))
        return meta

    def _combined_meta_events(
        self, pp: int, *, vpp: int = 1,
        device_stages: dict[int, list[int]] | None = None,
    ) -> list[dict]:
        meta: list[dict] = []
        detail_base = max(vpp, 2)
        for d in range(pp):
            meta.append(self._process_name_meta(d, f"GPU {d}"))
            meta.append(self._sort_index_meta(d, d))
            if vpp <= 1:
                meta.append(self._thread_name_meta(d, 0, "Grid Schedule"))
                meta.append(self._thread_sort_index_meta(d, 0, 0))
            else:
                stages = (device_stages or {}).get(d, [])
                for idx, s in enumerate(stages):
                    meta.append(self._thread_name_meta(d, idx, f"Stage {s}"))
                    meta.append(self._thread_sort_index_meta(d, idx, idx))
            meta.append(self._thread_name_meta(d, detail_base + 0, "Compute Ops"))
            meta.append(self._thread_sort_index_meta(d, detail_base + 0, vpp))
            meta.append(self._thread_name_meta(d, detail_base + 1, "Comm Ops"))
            meta.append(self._thread_sort_index_meta(d, detail_base + 1, vpp + 1))
        return meta

    # ── deduplication ─────────────────────────────────────────────────────

    @staticmethod
    def _deduplicate(events: list[dict]) -> list[dict]:
        """Merge X-events that share (pid, tid, ts, name, cat) into one with count."""
        meta = [e for e in events if e.get("ph") != "X"]
        x_events = [e for e in events if e.get("ph") == "X"]

        groups: dict[tuple, list[dict]] = {}
        for e in x_events:
            key = (e["pid"], e["tid"], e["ts"], e["name"], e.get("cat", ""))
            groups.setdefault(key, []).append(e)

        merged: list[dict] = []
        for key, group in groups.items():
            if len(group) == 1:
                merged.append(group[0])
            else:
                first = dict(group[0])
                first["args"] = dict(first.get("args", {}))
                first["args"]["count"] = len(group)
                merged.append(first)

        return meta + merged

    # ── public API ────────────────────────────────────────────────────────

    def export_stitched(
        self, result: "PPStitchedTimeline", path: str | None = None,
    ) -> str:
        """Export PPStitchedTimeline (device x microbatch grid).

        pid = device (stream_id), tid = stage chunk index within device.

        For VPP, each device gets one thread per virtual stage so the
        grid blocks for each stage appear in separate rows within the
        same GPU process.

        Metadata events name each process "GPU N" and threads
        "Stage N" (VPP) or "Grid Schedule".
        """
        events: list[dict] = []
        n_devices = result.pp
        vpp, device_stages, stage_to_tid = self._compute_device_layout(result)
        events.extend(self._grid_meta_events(n_devices, vpp=vpp, device_stages=device_stages))

        for task in result.tasks:
            cat = self._grid_cat(task)
            name = self._name_for_task(task)
            color_val = self._color_for_task(task)
            pid = task.stream_id
            tid = stage_to_tid.get((pid, task.stage_id), 0)
            events.append(ChromeTraceEvent(
                name=name,
                cat=cat,
                pid=pid,
                tid=tid,
                ts=task.start_us * self._mult,
                dur=max(task.latency_us, self._MIN_VISIBLE_US) * self._mult,
                color=color_val,
                args={
                    "phase": task.phase,
                    "mb": task.mb_id,
                    "stage": task.stage_id,
                    "device": pid,
                    "dep_count": len(task.dependencies),
                },
            ).to_dict())

        if result.warmup_us > 0:
            for d in range(n_devices):
                events.append(self._instant(
                    name=_NAMES.get("warmup", "warmup"),
                    pid=d, tid=0,
                    ts=result.warmup_us * self._mult,
                    args={"phase": "warmup", "dur_us": result.warmup_us},
                ))
        if result.cooldown_us > 0:
            cooldown_ts = (result.step_time_us - result.cooldown_us) * self._mult
            for d in range(n_devices):
                events.append(self._instant(
                    name=_NAMES.get("cooldown", "cooldown"),
                    pid=d, tid=0,
                    ts=cooldown_ts,
                    args={"phase": "cooldown", "dur_us": result.cooldown_us},
                ))

        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_per_stage(
        self,
        timelines: list["Timeline"],
        path: str | None = None,
        *,
        M: int = 1,
        pp_stitched: "PPStitchedTimeline | None" = None,
        replicate: bool = True,
    ) -> str:
        """Export per-device DAGScheduler Timelines (per-op detail).

        ``timelines[d]`` maps to ``pid=d`` (one process per GPU).
        Within each device, ops on different streams are rendered on
        separate ``tid`` values (0 = "Compute Ops", 1 = "Comm Ops").

        When ``replicate=True`` and ``M > 1``, each microbatch's ops are
        expanded as time-offset replicas aligned with the pipeline grid
        schedule.  When ``replicate=False``, only one clean reference copy
        of each per-device timeline is exported.

        Zero-latency ops receive a minimum visible duration so they
        are not invisible in Chrome Trace.

        For VPP, each device's single Timeline already aggregates all
        virtual stages — the per-stage distinction is visible in the
        corresponding grid view (export_stitched / export_stitched_detailed).
        """
        pp = len(timelines)
        events: list[dict] = []
        events.extend(self._per_stage_meta_events(pp))

        grid_slot: dict[tuple[int, int, str], float] = {}
        if pp_stitched is not None and M > 1:
            for task in pp_stitched.tasks:
                if task.phase in ("fwd", "bwd", "bwd_dx", "bwd_dw"):
                    grid_slot[(task.stage_id, task.mb_id, task.phase)] = task.start_us

        for s, tl in enumerate(timelines):
            fwd_lat = tl.phase_latency("fwd")
            bwd_lat = tl.phase_latency("bwd")
            if fwd_lat == 0.0 and bwd_lat == 0.0:
                fwd_lat = tl.total_latency_us
            stage_total = fwd_lat + bwd_lat

            num_replicas = M if replicate else 1

            for m in range(num_replicas):
                fwd_base = grid_slot.get((s, m, "fwd"), m * stage_total)
                bwd_base = grid_slot.get(
                    (s, m, "bwd"),
                    grid_slot.get((s, m, "bwd_dx"), m * stage_total + fwd_lat),
                )

                skip_ids: set[str] = set()
                for idx, op in enumerate(tl.scheduled_ops):
                    if op.node_id in skip_ids:
                        continue
                    cat = "communication" if op.stream_type == "comm" else "compute"
                    if replicate:
                        name = f"m{m}:{op.phase}:{op.op_type}" if op.phase else f"m{m}:{op.op_type}"
                    else:
                        name = f"{op.phase}:{op.op_type}" if op.phase else op.op_type

                    if op.phase == "fwd" or not op.phase:
                        base = fwd_base
                        rel_start = op.start_us
                    else:
                        base = bwd_base
                        rel_start = op.start_us - fwd_lat

                    dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us

                    if self._trace_ep_waves and self._is_ep_dispatch(op):
                        group = self._find_ep_wave_group(tl.scheduled_ops, idx)
                        if group is not None and self._ep_wave_count(*group) > 1:
                            _dispatch, expert_ops, combine, blockers, dispatch_ready = group
                            dispatch_ready_rel = (
                                dispatch_ready
                                if op.phase == "fwd" or not op.phase
                                else dispatch_ready - fwd_lat
                            )
                            combine_region_end = (
                                combine.start_us + combine.latency_us
                                if combine.phase == "fwd" or not combine.phase
                                else combine.start_us - fwd_lat + combine.latency_us
                            )
                            events.extend(self._ep_wave_events(
                                op, expert_ops, combine,
                                pid=s,
                                detail_base=0,
                                base=base,
                                mb=m,
                                phase=op.phase,
                                name_prefix=f"m{m}:" if replicate else "",
                                rel_start=lambda candidate: (
                                    candidate.start_us
                                    if candidate.phase == "fwd" or not candidate.phase
                                    else candidate.start_us - fwd_lat
                                ),
                                blockers=blockers,
                                dispatch_ready_us=dispatch_ready_rel,
                                region_end_us=combine_region_end,
                            ))
                            skip_ids.update(
                                {op.node_id, combine.node_id}
                                | {expert.node_id for expert in expert_ops}
                            )
                            continue

                    events.append(ChromeTraceEvent(
                        name=name,
                        cat=cat,
                        pid=s,
                        tid=op.stream_id,
                        ts=(base + rel_start) * self._mult,
                        dur=dur_us * self._mult,
                        args={
                            "phase": op.phase,
                            "op_type": op.op_type,
                            "stream_type": op.stream_type,
                            "mb": m,
                        },
                    ).to_dict())

        events = self._deduplicate(events)
        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_combined(
        self,
        stitched: "PPStitchedTimeline",
        timelines: list["Timeline"],
        path: str | None = None,
    ) -> str:
        """Combined export: PP grid + per-device op detail shared on same pids.

        pid = device id (stream_id).

        Within each device:
          tid 0..vpp-1  = "Grid Schedule" per virtual stage (VPP) or
                          tid 0 = "Grid Schedule" (non-VPP)
          tid base+0    = "Compute Ops"    (per-op compute detail)
          tid base+1    = "Comm Ops"       (per-op communication detail)
          base = max(vpp, 2)

        Grid task categories use ``mb_{mb}`` so the same
        microbatch index gets the same colour across all stages.
        """
        n_devices = stitched.pp
        vpp, device_stages, stage_to_tid = self._compute_device_layout(stitched)
        detail_base = max(vpp, 2)
        events: list[dict] = []
        events.extend(self._combined_meta_events(
            n_devices, vpp=vpp, device_stages=device_stages,
        ))

        for task in stitched.tasks:
            pid = task.stream_id
            tid = stage_to_tid.get((pid, task.stage_id), 0)
            events.append(ChromeTraceEvent(
                name=self._name_for_task(task),
                cat=self._grid_cat(task),
                pid=pid,
                tid=tid,
                ts=task.start_us * self._mult,
                dur=max(task.latency_us, self._MIN_VISIBLE_US) * self._mult,
                color=self._color_for_task(task),
                args={
                    "phase": task.phase,
                    "mb": task.mb_id,
                    "stage": task.stage_id,
                    "device": pid,
                    "view": "grid",
                    "dep_count": len(task.dependencies),
                },
            ).to_dict())

        for d, tl in enumerate(timelines):
            for op in tl.scheduled_ops:
                cat = "communication" if op.stream_type == "comm" else "compute"
                name = f"{op.phase}:{op.op_type}" if op.phase else op.op_type
                dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us
                events.append(ChromeTraceEvent(
                    name=name,
                    cat=cat,
                    pid=d,
                    tid=detail_base + op.stream_id,
                    ts=op.start_us * self._mult,
                    dur=dur_us * self._mult,
                    args={
                        "phase": op.phase,
                        "op_type": op.op_type,
                        "stream_type": op.stream_type,
                        "view": "detail",
                    },
                ).to_dict())

        events = self._deduplicate(events)
        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    def export_stitched_detailed(
        self,
        stitched: "PPStitchedTimeline",
        timelines: list["Timeline"],
        path: str | None = None,
    ) -> str:
        """Stitched grid with per-device detail on separate tids (same pid).

        pid = device id (stream_id).
        Within each device:
          tid 0..vpp-1       = grid-level fwd/bwd blocks per virtual stage
          tid base + 0..N    = per-op detail from DAGScheduler Timeline
          base = max(vpp, 2)

        All microbatches share the same detail rows — they are distinguished
        by time offsets from the pipeline schedule, naturally serialised on
        their physical stream.
        """
        n_devices = stitched.pp
        vpp, device_stages, stage_to_tid = self._compute_device_layout(stitched)
        detail_base = max(vpp, 2)
        events: list[dict] = []
        events.extend(self._combined_meta_events(
            n_devices, vpp=vpp, device_stages=device_stages,
        ))

        grid_index: dict[tuple[int, int, str], float] = {}
        for task in stitched.tasks:
            key = (task.stage_id, task.mb_id, task.phase)
            grid_index[key] = task.start_us

        for task in stitched.tasks:
            pid = task.stream_id
            tid = stage_to_tid.get((pid, task.stage_id), 0)
            events.append(ChromeTraceEvent(
                name=self._name_for_task(task),
                cat=self._grid_cat(task),
                pid=pid,
                tid=tid,
                ts=task.start_us * self._mult,
                dur=max(task.latency_us, self._MIN_VISIBLE_US) * self._mult,
                color=self._color_for_task(task),
                args={
                    "phase": task.phase, "mb": task.mb_id,
                    "stage": task.stage_id, "device": pid, "view": "grid",
                },
            ).to_dict())

        for d, tl in enumerate(timelines):
            fwd_lat = tl.phase_latency("fwd")
            bwd_lat = tl.phase_latency("bwd")

            # Index compute ops by node_id for CoC start-time shift
            compute_index: dict[str, tuple[float, float]] = {}
            for op in tl.scheduled_ops:
                if op.stream_type != "comm":
                    compute_index[op.node_id] = (op.start_us, op.latency_us)

            for m in range(stitched.M):
                fwd_base = grid_index.get((d, m, "fwd"), 0.0)
                skip_ids: set[str] = set()
                for idx, op in enumerate(tl.scheduled_ops):
                    if op.node_id in skip_ids:
                        continue
                    if op.phase == "fwd":
                        dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us
                        ts = (fwd_base + op.start_us) * self._mult
                        if op.overlap_type not in ("none", "") and op.overlap_target:
                            ts = self._shift_overlap_comm_op(op, compute_index, fwd_base, ts)
                        if self._trace_ep_waves and self._is_ep_dispatch(op):
                            group = self._find_ep_wave_group(tl.scheduled_ops, idx)
                            if group is not None and self._ep_wave_count(*group) > 1:
                                _dispatch, expert_ops, combine, blockers, dispatch_ready = group
                                events.extend(self._ep_wave_events(
                                    op, expert_ops, combine,
                                    pid=d,
                                    detail_base=detail_base,
                                    base=fwd_base,
                                    mb=m,
                                    phase="fwd",
                                    name_prefix=f"m{m}:",
                                    rel_start=lambda candidate: candidate.start_us,
                                    blockers=blockers,
                                    dispatch_ready_us=dispatch_ready,
                                    region_end_us=combine.start_us + combine.latency_us,
                                ))
                                skip_ids.update(
                                    {op.node_id, combine.node_id}
                                    | {expert.node_id for expert in expert_ops}
                                )
                                continue
                        events.append(ChromeTraceEvent(
                            name=f"m{m}:{op.phase}:{op.op_type}" if op.phase else f"m{m}:{op.op_type}",
                            cat="compute" if op.stream_type != "comm" else "communication",
                            pid=d,
                            tid=detail_base + op.stream_id,
                            ts=ts,
                            dur=dur_us * self._mult,
                            args={
                                "phase": "fwd",
                                "mb": m,
                                "op_type": op.op_type,
                                "view": "detail",
                            },
                        ).to_dict())

                bwd_base = grid_index.get((d, m, "bwd"), grid_index.get((d, m, "bwd_dx"), 0.0))
                skip_ids = set()
                for idx, op in enumerate(tl.scheduled_ops):
                    if op.node_id in skip_ids:
                        continue
                    if "bwd" in op.phase:
                        dur_us = max(op.latency_us, self._MIN_VISIBLE_US) if op.stream_type == "comm" else op.latency_us
                        relative_start = op.start_us - fwd_lat if len(op.phase) > 0 and "fwd" not in op.phase else op.start_us
                        ts = (bwd_base + relative_start) * self._mult
                        if op.overlap_type not in ("none", "") and op.overlap_target:
                            ts = self._shift_overlap_comm_op(op, compute_index, bwd_base, ts)
                        if self._trace_ep_waves and self._is_ep_dispatch(op):
                            group = self._find_ep_wave_group(tl.scheduled_ops, idx)
                            if group is not None and self._ep_wave_count(*group) > 1:
                                _dispatch, expert_ops, combine, blockers, dispatch_ready = group
                                dispatch_ready_rel = (
                                    dispatch_ready - fwd_lat
                                    if len(op.phase) > 0 and "fwd" not in op.phase
                                    else dispatch_ready
                                )
                                events.extend(self._ep_wave_events(
                                    op, expert_ops, combine,
                                    pid=d,
                                    detail_base=detail_base,
                                    base=bwd_base,
                                    mb=m,
                                    phase="bwd",
                                    name_prefix=f"m{m}:",
                                    rel_start=lambda candidate: (
                                        candidate.start_us - fwd_lat
                                        if len(candidate.phase) > 0 and "fwd" not in candidate.phase
                                        else candidate.start_us
                                    ),
                                    blockers=blockers,
                                    dispatch_ready_us=dispatch_ready_rel,
                                    region_end_us=(
                                        combine.start_us - fwd_lat + combine.latency_us
                                        if len(combine.phase) > 0 and "fwd" not in combine.phase
                                        else combine.start_us + combine.latency_us
                                    ),
                                ))
                                skip_ids.update(
                                    {op.node_id, combine.node_id}
                                    | {expert.node_id for expert in expert_ops}
                                )
                                continue
                        events.append(ChromeTraceEvent(
                            name=f"m{m}:{op.phase}:{op.op_type}" if op.phase else f"m{m}:{op.op_type}",
                            cat="compute" if op.stream_type != "comm" else "communication",
                            pid=d,
                            tid=detail_base + op.stream_id,
                            ts=ts,
                            dur=dur_us * self._mult,
                            args={
                                "phase": "bwd",
                                "mb": m,
                                "op_type": op.op_type,
                                "view": "detail",
                            },
                        ).to_dict())

        doc = self._build_doc(events)
        if path:
            self._write(path, doc)
        return doc

    # ── internals ─────────────────────────────────────────────────────────

    @staticmethod
    def _is_ep_dispatch(op) -> bool:
        return (
            op.stream_type == "comm"
            and op.parallelism_tag == "ep"
            and op.op_type == "comm.all_to_all"
            and op.comm_role == "dispatch"
        )

    @staticmethod
    def _is_ep_combine(op) -> bool:
        return (
            op.stream_type == "comm"
            and op.parallelism_tag == "ep"
            and op.op_type == "comm.all_to_all"
            and op.comm_role == "combine"
        )

    @staticmethod
    def _is_ep_expert_compute(op) -> bool:
        if op.stream_type == "comm":
            return False
        text = " ".join((
            op.op_type or "",
            op.component or "",
            op.scope or "",
            op.module_class or "",
            op.node_id or "",
        )).lower()
        return (
            "groupedmatmul" in text
            or "grouped_gate_up" in text
            or "routed_expert_ffn" in text
        )

    @staticmethod
    def _is_ep_wave_excluded_compute(op) -> bool:
        text = " ".join((
            op.op_type or "",
            op.module_class or "",
            op.component or "",
            op.node_id or "",
        )).lower()
        return "moe_gate" in text or "gate" == (op.module_class or "").lower()

    @classmethod
    def _is_ep_wave_compute_candidate(cls, op) -> bool:
        text = " ".join((
            op.op_type or "",
            op.module_class or "",
            op.component or "",
            op.scope or "",
            op.node_id or "",
        )).lower()
        if "shared_expert" in text or "shared_experts" in text:
            return False
        return (
            cls._is_ep_expert_compute(op)
            or "grouped_silu" in text
            or ".ffn.moe" in text
        )

    def _find_ep_wave_group(self, ops, dispatch_idx: int):
        dispatch = ops[dispatch_idx]
        expert_candidates = []
        blockers = []
        pre_region_compute = []
        in_expert_region = False
        expert_region_start = 0.0
        for op in ops[dispatch_idx + 1:]:
            if op.phase != dispatch.phase:
                continue
            if not self._same_ep_region_scope(dispatch.scope, op.scope):
                continue
            if self._is_ep_combine(op):
                if not expert_candidates:
                    return None
                expert_candidates.sort(key=lambda candidate: candidate.start_us)
                expert_candidates = self._trim_ep_region_at_down(expert_candidates)
                if not expert_candidates:
                    return None
                blockers.sort(key=lambda candidate: candidate.start_us)
                region_end = expert_candidates[-1].start_us
                blockers = [
                    blocker for blocker in blockers
                    if blocker.start_us <= region_end
                ]
                pre_region_compute = [
                    candidate for candidate in ops
                    if candidate.phase == dispatch.phase
                    and candidate.stream_type != "comm"
                    and self._same_ep_region_scope(dispatch.scope, candidate.scope)
                    and candidate.start_us < expert_region_start
                ]
                dispatch_ready = max(
                    (candidate.end_us for candidate in pre_region_compute),
                    default=dispatch.start_us,
                )
                return dispatch, expert_candidates, op, blockers, dispatch_ready
            if self._is_ep_expert_compute(op):
                in_expert_region = True
                if not expert_candidates:
                    expert_region_start = op.start_us
                expert_candidates.append(op)
                continue
            if not in_expert_region and op.stream_type != "comm":
                pre_region_compute.append(op)
                continue
            if in_expert_region and op.stream_type != "comm" and op.start_us >= expert_region_start:
                if self._is_ep_wave_excluded_compute(op):
                    blockers.append(op)
                elif self._is_ep_wave_compute_candidate(op):
                    expert_candidates.append(op)
        return None

    @staticmethod
    def _trim_ep_region_at_down(expert_ops):
        last_down_idx = None
        for idx, op in enumerate(expert_ops):
            text = " ".join((op.op_type or "", op.node_id or "", op.scope or "")).lower()
            if "grouped_down" in text:
                last_down_idx = idx
        if last_down_idx is None:
            return expert_ops
        return expert_ops[:last_down_idx + 1]

    @staticmethod
    def _same_ep_region_scope(dispatch_scope: str, op_scope: str) -> bool:
        if not dispatch_scope or not op_scope:
            return True
        if dispatch_scope == op_scope:
            return True
        marker = ".ffn."
        if marker not in dispatch_scope or marker not in op_scope:
            return False
        return dispatch_scope.split(marker, 1)[0] == op_scope.split(marker, 1)[0]

    @staticmethod
    def _ep_expert_score(op) -> tuple[int, float]:
        text = " ".join((
            op.op_type or "",
            op.component or "",
            op.scope or "",
            op.module_class or "",
            op.node_id or "",
        )).lower()
        is_grouped = int("groupedmatmul" in text or "grouped_mm" in text)
        return is_grouped, op.latency_us

    def _ep_wave_count(self, *ops) -> int:
        for op in ops:
            if getattr(op, "ep_wave_k", 0) > 1:
                return int(op.ep_wave_k)
        return self._ep_wave_k if self._ep_wave_k > 1 else 1

    def _ep_wave_events(
        self,
        dispatch,
        expert_ops,
        combine,
        *,
        pid: int,
        detail_base: int,
        base: float,
        mb: int,
        phase: str,
        name_prefix: str,
        rel_start,
        blockers=(),
        dispatch_ready_us: float | None = None,
        region_end_us: float | None = None,
    ) -> list[dict]:
        waves = self._ep_wave_count(dispatch, *expert_ops, combine)
        if waves <= 1:
            return []

        dispatch_start = max(
            rel_start(dispatch),
            dispatch_ready_us if dispatch_ready_us is not None else rel_start(dispatch),
        )
        dispatch_lat = dispatch.latency_us / waves
        combine_lat = combine.latency_us / waves
        expert_index = {op.node_id: idx for idx, op in enumerate(expert_ops)}
        expert_lats = {op.node_id: op.latency_us / waves for op in expert_ops}
        expert_starts = {op.node_id: rel_start(op) for op in expert_ops}
        blocker_intervals = [
            (rel_start(op), rel_start(op) + op.latency_us)
            for op in blockers
        ]
        expert_name_counts: dict[str, int] = {}
        for op in expert_ops:
            expert_name_counts[op.op_type] = expert_name_counts.get(op.op_type, 0) + 1

        events: list[dict] = []
        stream_free = {
            "comm": dispatch_start,
            "compute": dispatch_start,
        }
        end_time: dict[tuple[str, int], float] = {}
        unscheduled = {
            (role, wave)
            for wave in range(waves)
            for role in (
                ["dispatch"]
                + [f"expert:{op.node_id}" for op in expert_ops]
                + ["combine"]
            )
        }
        role_priority = {"combine": 0, "expert": 1, "dispatch": 2}

        def deps_done(role: str, wave: int) -> bool:
            if role == "dispatch":
                return wave == 0 or ("dispatch", wave - 1) in end_time
            if role.startswith("expert:"):
                idx = expert_index[role.split(":", 1)[1]]
                if idx == 0:
                    return ("dispatch", wave) in end_time
                prev = expert_ops[idx - 1]
                return (f"expert:{prev.node_id}", wave) in end_time
            last = expert_ops[-1]
            return (f"expert:{last.node_id}", wave) in end_time

        def earliest_start(role: str, wave: int) -> float:
            if role == "dispatch":
                dep_ready = dispatch_start
                stream = "comm"
            elif role.startswith("expert:"):
                op_id = role.split(":", 1)[1]
                idx = expert_index[op_id]
                if idx == 0:
                    dep_ready = max(end_time[("dispatch", wave)], expert_starts[op_id])
                else:
                    prev = expert_ops[idx - 1]
                    dep_ready = end_time[(f"expert:{prev.node_id}", wave)]
                stream = "compute"
            else:
                last = expert_ops[-1]
                dep_ready = end_time[(f"expert:{last.node_id}", wave)]
                stream = "comm"
            return max(dep_ready, stream_free[stream])

        def avoid_compute_blockers(start: float, dur: float) -> float:
            adjusted = start
            changed = True
            while changed:
                changed = False
                for blocker_start, blocker_end in blocker_intervals:
                    if adjusted < blocker_end and adjusted + dur > blocker_start:
                        adjusted = blocker_end
                        changed = True
            return adjusted

        while unscheduled:
            ready = [
                (
                    earliest_start(role, wave),
                    role_priority["expert" if role.startswith("expert:") else role],
                    wave,
                    role,
                )
                for role, wave in unscheduled
                if deps_done(role, wave)
            ]
            if not ready:
                break
            start, _priority, wave, role = min(ready)

            if role == "dispatch":
                op = dispatch
                stream = "comm"
                dur = dispatch_lat
                cat = "communication.ep.dispatch"
                event_role = "dispatch"
                event_name = f"{name_prefix}{phase}:wave{wave}-dispatch"
            elif role.startswith("expert:"):
                op_id = role.split(":", 1)[1]
                op = expert_ops[expert_index[op_id]]
                stream = "compute"
                dur = expert_lats[op_id]
                start = avoid_compute_blockers(start, dur)
                cat = "compute.ep.expert"
                event_role = "expert"
                event_name = self._ep_expert_event_name(
                    name_prefix,
                    phase,
                    wave,
                    op,
                    expert_ops[:expert_index[op_id]],
                    expert_name_counts,
                )
            else:
                op = combine
                stream = "comm"
                dur = combine_lat
                cat = "communication.ep.combine"
                event_role = "combine"
                event_name = f"{name_prefix}{phase}:wave{wave}-combine"

            end = start + dur
            stream_free[stream] = end
            end_time[(role, wave)] = end
            unscheduled.remove((role, wave))
            events.append(self._wave_event(
                event_name,
                cat,
                pid,
                detail_base + op.stream_id,
                base + start,
                dur,
                op,
                mb,
                wave,
                waves,
                event_role,
            ))

        if region_end_us is not None and not blockers:
            self._fit_wave_events_to_region(
                events,
                base + dispatch_start,
                base + region_end_us,
            )

        return events

    def _fit_wave_events_to_region(
        self,
        events: list[dict],
        region_start_us: float,
        region_end_us: float,
    ) -> None:
        if not events or region_end_us <= region_start_us:
            return
        synth_end_us = max(
            (event["ts"] + event["dur"]) / self._mult
            for event in events
            if event.get("ph") == "X"
        )
        if synth_end_us <= region_end_us:
            return
        scale = (region_end_us - region_start_us) / (synth_end_us - region_start_us)
        if scale <= 0:
            return
        for event in events:
            if event.get("ph") != "X":
                continue
            ts_us = event["ts"] / self._mult
            dur_us = event["dur"] / self._mult
            event["ts"] = (region_start_us + (ts_us - region_start_us) * scale) * self._mult
            event["dur"] = dur_us * scale * self._mult
            event.setdefault("args", {})["ep_wave_fit_scale"] = scale

    @staticmethod
    def _ep_expert_event_name(
        name_prefix: str,
        phase: str,
        wave: int,
        op,
        previous_ops,
        name_counts: dict[str, int],
    ) -> str:
        base = f"{name_prefix}{phase}:wave{wave}-expert-{op.op_type}"
        if name_counts.get(op.op_type, 0) <= 1:
            return base
        duplicate_index = sum(1 for prev in previous_ops if prev.op_type == op.op_type)
        if duplicate_index == 0:
            return base
        suffix = (op.node_id or f"{duplicate_index}").split("_")[-1]
        return f"{base}:{suffix}"

    def _wave_event(
        self,
        name: str,
        cat: str,
        pid: int,
        tid: int,
        ts_us: float,
        dur_us: float,
        op,
        mb: int,
        wave: int,
        waves: int,
        role: str,
    ) -> dict:
        return ChromeTraceEvent(
            name=name,
            cat=cat,
            pid=pid,
            tid=tid,
            ts=ts_us * self._mult,
            dur=max(dur_us, self._MIN_VISIBLE_US if op.stream_type == "comm" else 0.0) * self._mult,
            args={
                "phase": op.phase,
                "mb": mb,
                "op_type": op.op_type,
                "view": "detail",
                "parallelism": "ep",
                "role": role,
                "wave": wave,
                "waves": waves,
                "original_node": op.node_id,
                "stream_type": op.stream_type,
            },
        ).to_dict()

    def _shift_overlap_comm_op(
        self,
        op,
        compute_index: dict[str, tuple[float, float]],
        base_us: float,
        original_ts: float,
    ) -> float:
        """Shift overlap comm op start time to visualize overlap in trace.

        Handles three overlap types:
        - CoC: K-wave shift (target_lat * (K-1) / K)
        - P2P/Ring-CP: full overlap (shift by target_lat)
        - Others: no shift

        Args:
            op: ScheduledOp with overlap_type, overlap_target, coc_tile_k
            compute_index: dict mapping node_id -> (start_us, latency_us) - NOT multiplied
            base_us: base timestamp for the current microbatch - NOT multiplied
            original_ts: original timestamp from scheduler - ALREADY multiplied by self._mult

        Returns:
            Shifted timestamp to show overlap in trace visualization.
        """
        overlap_type = op.overlap_type
        if overlap_type in ("none", ""):
            return original_ts

        target_key = op.overlap_target
        if not target_key or ":" not in target_key:
            return original_ts

        target_id = target_key.split(":", 1)[1]
        entry = compute_index.get(target_id)
        if entry is None:
            return original_ts

        target_start_us, target_lat = entry  # NOT multiplied
        if target_lat <= 0:
            return original_ts

        if overlap_type == "coc":
            coc_tile_k = getattr(op, "coc_tile_k", 4)
            if coc_tile_k <= 1:
                return original_ts
            shift_us = target_lat * (coc_tile_k - 1) / coc_tile_k  # NOT multiplied
            return original_ts - shift_us * self._mult

        elif overlap_type in ("p2p_overlap", "ring_cp"):
            # P2P should overlap with target compute
            # Strategy: shift P2P backward by target_latency to show overlap
            
            # Compute desired P2P start time (NOT multiplied)
            # P2P starts at same time as target (parallel execution)
            desired_start_us = base_us + target_start_us
            
            # Shift amount (NOT multiplied)
            # current = base_us + op.start_us
            # desired = base_us + target_start_us
            # shift = desired - current = target_start_us - op.start_us
            
            # But P2P might already start before target (scheduler put it earlier)
            # In that case, we want to shift forward to align
            
            # Alternative: shift backward by target_lat to show full overlap
            # This makes P2P appear during compute's execution window
            
            # Simple approach: shift by target_latency (shows P2P completely hidden)
            shift_us = target_lat
            
            # Apply shift (multiplied)
            shifted_ts = original_ts - shift_us * self._mult
            
# Clamp to >= 0 (can't have negative time in trace)
            return max(0.0, shifted_ts)

        return original_ts

    def _coc_shift_ts(
        self,
        op,
        compute_index: dict[str, tuple[float, float]],
        base_us: float,
        original_ts: float,
    ) -> float:
        """Legacy wrapper for backward compatibility."""
        return self._shift_overlap_comm_op(
            op, compute_index, base_us, original_ts
        )

    @staticmethod
    def _grid_cat(task) -> str:
        if task.phase == "fwd":
            return "fwd"
        return "bwd"

    @staticmethod
    def _name_for_task(task) -> str:
        mapping = {"fwd": "F", "bwd": "B", "bwd_dx": "B_dx", "bwd_dw": "B_dw"}
        prefix = mapping.get(task.phase, task.phase.upper()[:4])
        return f"{prefix} {task.mb_id}"

    @staticmethod
    def _color_for_task(task) -> str:
        if task.phase == "fwd":
            return _FWD_COLOR
        if task.phase == "bwd_dx":
            return _BWD_DX_COLOR
        if task.phase == "bwd_dw":
            return _BWD_DW_COLOR
        return _BWD_COLOR

    @staticmethod
    def _instant(name: str, pid: int, tid: int, ts: float, args: dict) -> dict:
        return {
            "ph": "i",
            "name": name,
            "pid": pid,
            "tid": tid,
            "ts": ts,
            "s": "g",   # scope = global
            "args": args,
        }

    def _build_doc(self, events: list[dict]) -> str:
        return json.dumps(
            {
                "traceEvents": events,
                "displayTimeUnit": "ns" if self._time_unit == "ns" else "ms",
            },
            indent=2,
            ensure_ascii=False,
        )

    @staticmethod
    def _write(path: str, content: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
