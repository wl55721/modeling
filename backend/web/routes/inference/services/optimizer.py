from __future__ import annotations

import time
import math
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Iterator

from ..schemas import (
    OptimizeRequest, OptimizeResponse, StrategyResult,
    SearchSummary, HardwareOptimizeResult,
    ParallelConfig, WorkloadEntry, SimulateRequest,
)
from .simulation import SimulationService
from backend.utils.log import logger


def _factors(n: int) -> list[int]:
    result = []
    for i in range(1, int(math.isqrt(n)) + 1):
        if n % i == 0:
            result.append(i)
            if i != n // i:
                result.append(n // i)
    return sorted(result)


def _to_power_of_two(v: int) -> int:
    p = 1
    while p < v:
        p *= 2
    return p


def _make_label(p: ParallelConfig) -> str:
    parts = []
    if p.tp_size > 1: parts.append(f"TP{p.tp_size}")
    if p.dp_size > 1: parts.append(f"DP{p.dp_size}")
    return "_".join(parts) if parts else "single"


class OptimizerService:

    def __init__(self):
        self._sim = SimulationService()

    # ── public ───────────────────────────────────────────

    def optimize(self, req: OptimizeRequest) -> OptimizeResponse:
        t0 = time.perf_counter()

        # Determine hardware list: use hardwares array or fallback to hardware_name
        hw_names: list[str] = []
        for h in req.hardwares:
            hw_names.append(h.name)
        if not hw_names and req.hardware_name:
            hw_names.append(req.hardware_name)

        if not hw_names:
            # No hardware specified — single pass with empty name
            hw_names = [""]

        all_candidates: list[StrategyResult] = []
        total = 0
        evaluated = 0
        hw_results: list[HardwareOptimizeResult] = []

        for hw_name in hw_names:
            hw_candidates, hw_total, hw_evaluated = self._optimize_for_hardware(req, hw_name)
            total += hw_total
            evaluated += hw_evaluated
            all_candidates.extend(hw_candidates)

            hw_candidates.sort(key=self._rank_key)
            hw_optimal = next((c for c in hw_candidates if c.meets_target), None)
            hw_oom = sum(1 for c in hw_candidates if c.is_oom)

            hw_results.append(HardwareOptimizeResult(
                hardware_name=hw_name,
                optimal=hw_optimal,
                candidates=hw_candidates,
                search_summary=SearchSummary(
                    total_candidates=hw_total,
                    evaluated=hw_evaluated,
                    pruned=0,
                    oom_count=hw_oom,
                    elapsed_ms=0,
                ),
            ))

        elapsed_ms = (time.perf_counter() - t0) * 1000

        all_candidates.sort(key=self._rank_key)
        optimal = next((c for c in all_candidates if c.meets_target), None)
        oom_count = sum(1 for c in all_candidates if c.is_oom)

        logger.info(
            "optimize done — total=%d evaluated=%d oom=%d "
            "optimal=%s bs=%d tps=%.1f elapsed=%.0fms",
            total, evaluated, oom_count,
            optimal.strategy_label if optimal else "NONE",
            optimal.batch_size if optimal else 0,
            optimal.tps if optimal else 0,
            elapsed_ms,
        )

        return OptimizeResponse(
            optimal=optimal,
            candidates=all_candidates,
            search_summary=SearchSummary(
                total_candidates=total,
                evaluated=evaluated,
                pruned=0,
                oom_count=oom_count,
                elapsed_ms=round(elapsed_ms, 1),
            ),
            hardware_results=hw_results,
        )

    def _optimize_for_hardware(
        self, req: OptimizeRequest, hw_name: str,
    ) -> tuple[list[StrategyResult], int, int]:
        """Run optimization for a single hardware."""
        opt = req.workload.optimize
        max_workers = max(1, min(8, (os.cpu_count() or 4)))
        candidates: list[StrategyResult] = []
        total = 0
        evaluated = 0

        for world_size in self._iter_world_sizes(opt.min_world_size, opt.max_world_size):
            tp_list = _factors(world_size)
            sub_tps = self._iter_sub_tps(world_size, opt)

            lo = max(1, opt.batch_size_min)
            hi = opt.batch_size_max
            searched_bs: set[int] = set()
            ws_best_tps = 0.0
            ws_best_bs = 0

            while lo <= hi:
                mid = (lo + hi) // 2
                if mid in searched_bs:
                    break
                searched_bs.add(mid)

                tasks: list[tuple[int, ParallelConfig]] = []
                for tp in tp_list:
                    dp = world_size // tp
                    for (embed_tp, o_tp, lmhead_tp) in sub_tps:
                        total += 1
                        tasks.append((mid, ParallelConfig(
                            world_size=world_size,
                            tp_size=tp,
                            dp_size=dp,
                            pp_size=1,
                            ep_size=world_size,
                            cp_size=1,
                            embed_tp_size=embed_tp,
                            o_tp_size=o_tp,
                            lmhead_tp_size=lmhead_tp,
                        )))

                futures_map: dict = {}
                with ThreadPoolExecutor(max_workers=max_workers) as executor:
                    for bs, p in tasks:
                        f = executor.submit(self._evaluate, req, p, bs, hw_name)
                        futures_map[f] = (p, bs)

                    bs_has_meeting = False
                    for future in as_completed(futures_map):
                        p, bs = futures_map[future]
                        result = future.result()
                        evaluated += 1

                        if result.meets_target:
                            bs_has_meeting = True
                            if result.tps > ws_best_tps:
                                ws_best_tps = result.tps
                                ws_best_bs = bs

                        candidates.append(result)

                if bs_has_meeting:
                    lo = mid + 1
                else:
                    hi = mid - 1

            if ws_best_tps > 0:
                logger.info(
                    "%s WS=%d best: TPS=%.1f BS=%d",
                    hw_name, world_size, ws_best_tps, ws_best_bs,
                )

        return candidates, total, evaluated

    # ── iterators ────────────────────────────────────────

    @staticmethod
    def _iter_world_sizes(min_ws: int, max_ws: int) -> Iterator[int]:
        ws = _to_power_of_two(max(1, min_ws))
        while ws <= max_ws:
            yield ws
            ws *= 2

    @staticmethod
    def _iter_sub_tps(ws: int, opt) -> list[tuple[int, int, int]]:
        def valid_vals(min_v: int, max_v: int) -> list[int]:
            if max_v <= 0:
                max_v = ws
            vals = _factors(ws) if opt.fine_grained else [1, ws]
            return [v for v in vals if min_v <= v <= min(max_v, ws)]

        e_vals = valid_vals(opt.embed_tp_min, opt.embed_tp_max)
        o_vals = valid_vals(opt.o_tp_min, opt.o_tp_max)
        l_vals = valid_vals(opt.lmhead_tp_min, opt.lmhead_tp_max)

        combos = [(e, o, l) for e in e_vals for o in o_vals for l in l_vals]
        full = (ws, ws, ws)
        if full in combos:
            combos.remove(full)
            combos.insert(0, full)
        return combos

    # ── evaluate ─────────────────────────────────────────

    def _evaluate(
        self, req: OptimizeRequest, parallel: ParallelConfig, batch_size: int = 1,
        hardware_name: str = "",
    ) -> StrategyResult:
        wl = req.workload

        req_copy = wl.request.model_copy(update={"batch_size": batch_size})

        # Only pass the specific hardware being evaluated (not all hardwares)
        target_hw = None
        if hardware_name and req.hardwares:
            for h in req.hardwares:
                if h.name == hardware_name:
                    target_hw = h
                    break

        sim_req = SimulateRequest(
            model_name=req.model_name,
            model_json=req.model_json,
            hf_config_json=req.hf_config_json,
            hardwares=[target_hw] if target_hw else req.hardwares,
            hardware_name=hardware_name if not target_hw else "",
            workloads=[WorkloadEntry(
                request=req_copy,
                parallel=parallel,
                quant=wl.quant,
            )],
        )

        try:
            sim_result = self._sim.simulate(sim_req)
            single = sim_result.results[0].result if sim_result.results else None
        except Exception as e:
            logger.warning("evaluate failed for %s: %s", _make_label(parallel), e)
            single = None

        if single is None:
            return StrategyResult(
                world_size=parallel.world_size,
                tp_size=parallel.tp_size,
                dp_size=parallel.dp_size,
                embed_tp_size=parallel.embed_tp_size,
                o_tp_size=parallel.o_tp_size,
                lmhead_tp_size=parallel.lmhead_tp_size,
                batch_size=batch_size,
                strategy_label=_make_label(parallel),
                tpot_ms=float("inf"),
                tps=0,
                max_peak_mem_gb=0,
                total_mem_gb=0,
                is_oom=True,
                meets_target=False,
            )

        return StrategyResult(
            world_size=parallel.world_size,
            tp_size=parallel.tp_size,
            dp_size=parallel.dp_size,
            embed_tp_size=parallel.embed_tp_size,
            o_tp_size=parallel.o_tp_size,
            lmhead_tp_size=parallel.lmhead_tp_size,
            batch_size=batch_size,
            strategy_label=single.strategy,
            tpot_ms=single.tpot_ms,
            tps=single.tps,
            max_peak_mem_gb=single.peak_mem_gb,
            total_mem_gb=single.peak_mem_gb * parallel.world_size,
            is_oom=single.oom,
            meets_target=(not single.oom and single.tpot_ms <= req.workload.optimize.target_tpot_ms),
        )

    # ── ranking ──────────────────────────────────────────

    @staticmethod
    def _rank_key(c: StrategyResult) -> tuple:
        return (
            0 if not c.is_oom else 1,
            0 if c.meets_target else 1,
            c.world_size,
            -c.tps,
        )
