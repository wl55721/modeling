from __future__ import annotations

from dataclasses import replace

from zrt.training.compose.stage import StageTime
from zrt.training.spec.strategy import PPSched, Strategy


def schedule_window_factor(strategy: Strategy) -> float:
    """Conservative compute-window factor for MoE F/B A2A hiding."""
    sched = strategy.pp_schedule
    vpp = max(1, strategy.vpp_chunks)
    if sched == PPSched.DUALPIPE_V:
        return min(1.0, 0.85 + 0.05 * (vpp - 1))
    if sched == PPSched.DUALPIPE:
        return 0.85
    if sched == PPSched.INTERLEAVED:
        return min(0.75, 0.50 + 0.05 * (vpp - 1))
    return 0.50


def steady_fraction(strategy: Strategy) -> float:
    """Fraction of microbatches with cross-microbatch MoE FB windows."""
    m = max(1, strategy.num_microbatches())
    pp = max(1, strategy.pp)
    vpp = max(1, strategy.vpp_chunks)
    sched = strategy.pp_schedule
    if pp <= 1:
        return 0.0
    if sched in (PPSched.DUALPIPE, PPSched.DUALPIPE_V):
        bubble_slots = (pp - 1) / (2.0 * vpp)
        return m / (m + bubble_slots) if m + bubble_slots > 0 else 1.0
    boundary = min(m, 2 * (pp - 1))
    return max(0.0, (m - boundary) / m)


def apply_moe_fb_overlap(stage_times: list[StageTime], strategy: Strategy) -> list[StageTime]:
    """Apply schedule-level MoE F/B EP A2A hiding to per-stage timings."""
    if not strategy.moe_fb_overlap or strategy.ep <= 1:
        return stage_times

    factor = schedule_window_factor(strategy)
    steady = steady_fraction(strategy)
    if factor <= 0.0 or steady <= 0.0:
        return stage_times

    adjusted: list[StageTime] = []
    for st in stage_times:
        ep_total = max(0.0, st.ep_exposed)
        if ep_total <= 0.0:
            adjusted.append(st)
            continue

        fwd_compute = max(0.0, st.fwd - st.comm_fwd)
        bwd_compute = max(0.0, st.bwd - st.comm_bwd)
        hide_window = (fwd_compute + bwd_compute) * factor * steady
        hidden = min(ep_total, hide_window)
        if hidden <= 0.0:
            adjusted.append(replace(
                st,
                ep_fb_total=ep_total,
                ep_fb_exposed=ep_total,
            ))
            continue

        exposed = max(0.0, ep_total - hidden)
        fwd_share = st.comm_fwd / (st.comm_fwd + st.comm_bwd) if (st.comm_fwd + st.comm_bwd) > 0 else 0.5
        hidden_fwd = min(st.comm_fwd, hidden * fwd_share)
        hidden_bwd = min(st.comm_bwd, hidden - hidden_fwd)

        adjusted.append(replace(
            st,
            fwd=max(0.0, st.fwd - hidden_fwd),
            bwd=max(0.0, st.bwd - hidden_bwd),
            bwd_dx=max(0.0, st.bwd_dx - hidden_bwd),
            comm_fwd=max(0.0, st.comm_fwd - hidden_fwd),
            comm_bwd=max(0.0, st.comm_bwd - hidden_bwd),
            ep_exposed=exposed,
            ep_hidden=st.ep_hidden + hidden,
            ep_fb_total=ep_total,
            ep_fb_hidden=hidden,
            ep_fb_exposed=exposed,
            ep_fb_steady_hidden=hidden,
        ))

    return adjusted
