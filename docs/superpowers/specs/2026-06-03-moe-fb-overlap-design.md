# MoE FB Overlap Redesign

Date: 2026-06-03

## Context

The current EP overlap model mixes two different concepts:

- `ep_overlap_waves` is used by the normal EP stage composer as an intra-layer
  K-wave overlap between EP A2A and expert GEMM.
- The same `ep_overlap_waves` value is also the right hardware input for
  MegaMoE, where dispatch/compute/combine wave pipelining is an explicit fused
  operator model.

The normal EP path should no longer model MindSpeed-style MoE overlap through
`ep_overlap_waves`. The MindSpeed feature is a schedule-level MoE forward/backward
communication overlap: MoE A2A communication from one microbatch can be hidden by
Attention/MLP compute windows from other microbatches under 1F1B, VPP,
DualPipe, or DualPipeV schedules.

## Goals

- Make `ep_overlap_waves` a MegaMoE-only hardware parameter.
- Remove the incorrect normal EP K-wave hidden-communication path.
- Add a separate `moe_fb_overlap` capability for MindSpeed-style MoE
  forward/backward A2A overlap.
- Let PP schedule, VPP chunks, and microbatch count provide overlap windows.
- Use the same MoE FB overlap result in numeric reports and trace view.
- Keep the implementation conservative and calibration-friendly.

## Non-Goals

- Do not rewrite the full scheduler into a fully timeline-driven MoE stream
  model in this change.
- Do not remove MegaMoE wave pipelining.
- Do not treat VPP, 1F1B, or DualPipe as overlap features by themselves.
  They only provide different windows consumed by `moe_fb_overlap`.

## New Semantics

### Configuration

`ep_overlap_waves`

- Remains in hardware/system specs.
- Is consumed only by MegaMoE fused-operator modelling.
- Must not reduce normal EP A2A exposed communication.

`ep_overlap`

- Should no longer trigger normal EP K-wave hidden communication.
- Existing config compatibility can be preserved, but it should not imply
  MindSpeed MoE FB overlap.

`moe_fb_overlap`

- New strategy-level boolean.
- Enables MindSpeed-style MoE forward/backward A2A overlap.
- Applies to `1f1b`, `interleaved`/VPP, `dualpipe`, and `dualpipev`.
- The schedule determines the available overlap windows; the flag determines
  whether MoE A2A is allowed to consume those windows.

`dualbatch`

- Remains a pipeline scheduling / dual-batch capability.
- Should not be the sole switch for EP hidden communication.

## MoE FB Model

The model exposes four semantic A2A event classes:

- `fwd_dispatch`
- `fwd_combine`
- `bwd_dispatch`
- `bwd_combine`

For each bottleneck stage, compute:

- total EP A2A time per class
- available hiding window per class
- hidden amount
- exposed amount

The report-level identities must hold:

```text
ep_total_ms = ep_exposed_ms + ep_hidden_ms
ep_hidden_ms = ep_fb_hidden_ms + mega_moe_hidden_ms
```

If MegaMoE is not active, `mega_moe_hidden_ms` is zero.

The initial estimator should be conservative:

- Steady-state microbatches can consume schedule windows.
- Warmup and cooldown have reduced or zero cross-microbatch windows.
- Optional boundary self-overlap can be reported separately only when the
  implementation has enough layer-internal structure to justify it.

## Schedule Windows

The PP schedule provides windows; it does not directly hide EP communication.

`1f1b`

- Provides limited steady-state overlap windows.
- Warmup/cooldown windows are smaller.

`interleaved` / VPP

- Provides more fragmented windows due to virtual stages.
- Reduces bubble and may increase opportunity, but does not enable overlap
  without `moe_fb_overlap`.

`dualpipe`

- Provides stronger forward/backward concurrency windows.
- Usually has the largest MoE FB hiding opportunity among non-VPP schedules.

`dualpipev`

- Combines DualPipe-style concurrency with virtual-stage placement.
- Uses the same `moe_fb_overlap` model with VPP-aware window sizing.

## Implementation Shape

### Normal Configuration Path

Update the per-stage composer so normal EP A2A no longer calls
`_wave_overlap_saved` or consumes `gpu.ep_overlap_waves`.

Normal EP stage time should keep:

- raw EP communication
- exposed EP communication
- zero hidden EP communication unless `moe_fb_overlap` or MegaMoE contributes

MegaMoE keeps its existing wave pipeline model and continues to consume
`ep_overlap_waves`.

### Graph Capture Path

Add MoE FB overlap as a separate post-stage analysis:

1. Extract MoE A2A nodes by role and phase.
2. Aggregate bottleneck-stage A2A totals by the four semantic classes.
3. Read PP schedule metadata, VPP chunks, microbatch count, and
   warmup/steady/cooldown information.
4. Estimate hidden/exposed EP A2A using schedule windows.
5. Store the result in graph metadata.

This should not be mixed into generic `overlap_type` formula handling. Existing
`ring_cp`, `coc`, `mc2`, and P2P overlap logic remains separate.

### Report Fields

Keep existing fields for compatibility:

- `ep_total_ms`
- `ep_exposed_ms`
- `ep_hidden_ms`
- `hidden_comm_ms`

Add or expose detailed MoE fields:

- `ep_fb_total_ms`
- `ep_fb_hidden_ms`
- `ep_fb_exposed_ms`
- `ep_fb_steady_hidden_ms`
- `ep_fb_boundary_hidden_ms`
- `mega_moe_hidden_ms`

Reports should label `ep_hidden_ms` as an aggregate, not as K-wave hidden time.

### Trace View

Deprecate the old normal EP use of `trace_ep_waves`.

Introduce `trace_moe_fb_overlap` or equivalent trace mode:

- Show the four semantic MoE A2A event classes.
- Show schedule windows used for hiding.
- Show hidden and exposed segments separately.
- Link hidden segments to the compute window that covers them when possible.

MegaMoE may keep a separate wave visualization because that is a real
dispatch/compute/combine fused-operator model.

## File-Level Targets

Expected implementation files:

- `python/zrt/training/spec/strategy.py`
- `python/zrt/training/io/config_loader.py`
- `python/zrt/training/search/training_search_util.py`
- `python/zrt/training/compose/stage.py`
- `python/zrt/training/compose/schedules.py`
- `python/zrt/transform/analysis/training.py`
- `python/zrt/transform/analysis/modeller.py`
- `python/zrt/executor/chrome_trace.py`
- `python/zrt/training/spec/report.py`
- report exporters under `python/zrt/training/io/`

MegaMoE files should only be touched to preserve or clarify ownership:

- `python/zrt/training/models/mega_moe.py`
- `python/zrt/transform/analysis/passes.py`

## Tests

Add or update tests for:

- Normal EP no longer hides communication through `ep_overlap_waves`.
- MegaMoE still uses `ep_overlap_waves` and still reports hidden communication.
- `moe_fb_overlap=False` keeps EP FB hidden at zero.
- `moe_fb_overlap=True` produces non-negative hidden/exposed totals and preserves
  `ep_total = ep_exposed + ep_hidden`.
- VPP/1F1B/DualPipe/DualPipeV produce different windows but require the same
  `moe_fb_overlap` flag.
- Trace view renders MoE FB overlap metadata and no longer presents normal EP as
  old K-wave overlap.

## Rollout

1. Introduce config/report fields and compatibility plumbing.
2. Remove normal EP K-wave hidden path while preserving MegaMoE behavior.
3. Add MoE FB overlap estimator and metadata.
4. Update reports.
5. Update trace view.
6. Update tests and calibration snapshots.

## Open Calibration Questions

- Exact boundary self-overlap for warmup/cooldown should be calibrated against
  MindSpeed traces before being reported as hidden time.
- Schedule-window sizing should start conservative, then be tuned with real
  traces for 1F1B, VPP, DualPipe, and DualPipeV.
- If future calibration requires exact stream semantics, this design can evolve
  into a fully timeline-driven model.
