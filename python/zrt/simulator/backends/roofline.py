"""Roofline model simulator — universal fallback backend.

Formulas
--------
matmul (mm / bmm / addmm / linear):
    FLOPs  = 2 * M * N * K
    read   = (M*K + K*N) * itemsize
    write  = M*N * itemsize

flash-attention / SDPA:
    FLOPs  ≈ 4 * B * H * Sq * Sk * D   (QK + AV matmuls, ignoring softmax)
    read   = (Q + K + V) bytes
    write  = output bytes

layer_norm / rms_norm:
    FLOPs  ≈ 5 * numel(input)
    read   = numel(input + weight + bias) * itemsize
    write  = numel(output) * itemsize

softmax:
    FLOPs  ≈ 5 * numel(input)
    read   = numel(input) * itemsize
    write  = numel(output) * itemsize

elementwise (add, mul, silu, gelu, ...):
    FLOPs  = ops_per_elem * numel(output)
    read   = sum(numel(input_i)) * itemsize
    write  = numel(output) * itemsize

embedding:
    FLOPs  = 0
    read   = numel(output) * itemsize   (cache-miss dominated)
    write  = numel(output) * itemsize

default fallback:
    FLOPs  = numel(output)  (1 op / element, conservative)
    read   = total_input_bytes
    write  = total_output_bytes
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from python.zrt.ir.types import DType
from python.zrt.simulator.base import OpSimulator
from python.zrt.simulator.result import SimResult

if TYPE_CHECKING:
    from python.zrt.ir.node import OpNode
    from python.zrt.hardware.spec import HardwareSpec


# ── helpers ───────────────────────────────────────────────────────────────────

def _numel(shape: tuple[int, ...]) -> int:
    if not shape:
        return 1
    n = 1
    for d in shape:
        n *= max(d, 0)
    return n


def _primary_dtype(node: "OpNode") -> DType:
    """Return the dominant dtype for compute-throughput lookup."""
    if node.outputs:
        return node.outputs[0].dtype
    if node.inputs:
        return node.inputs[0].dtype
    return DType.BF16


def _itemsize(node: "OpNode") -> float:
    return _primary_dtype(node).itemsize


# ── per-op formula functions ──────────────────────────────────────────────────
# Each returns (flops: float, read_bytes: float, write_bytes: float)

FMR = tuple[float, float, float]   # (flops, read_bytes, write_bytes)


def _mm(node: "OpNode") -> FMR:
    """aten.mm.default: A=(M,K) @ B=(K,N) → (M,N)"""
    if len(node.inputs) < 2:
        return _default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) < 2 or len(b.shape) < 2:
        return _default(node)
    M, K = a.shape[-2], a.shape[-1]
    N = b.shape[-1]
    it = a.dtype.itemsize
    flops = 2.0 * M * N * K
    read  = (M * K + K * N) * it
    write = M * N * it
    return flops, read, write


def _addmm(node: "OpNode") -> FMR:
    """aten.addmm.default: bias + mat1 @ mat2"""
    if len(node.inputs) < 3:
        return _default(node)
    # inputs: [bias, mat1, mat2]
    mat1, mat2 = node.inputs[1], node.inputs[2]
    bias = node.inputs[0]
    if len(mat1.shape) < 2 or len(mat2.shape) < 2:
        return _default(node)
    M, K = mat1.shape[0], mat1.shape[1]
    N = mat2.shape[1]
    it = mat1.dtype.itemsize
    flops = 2.0 * M * N * K + M * N   # mm + bias add
    read  = (M * K + K * N + _numel(bias.shape)) * it
    write = M * N * it
    return flops, read, write


def _bmm(node: "OpNode") -> FMR:
    """aten.bmm.default: (B,M,K) @ (B,K,N) → (B,M,N)"""
    if len(node.inputs) < 2:
        return _default(node)
    a, b = node.inputs[0], node.inputs[1]
    if len(a.shape) < 3 or len(b.shape) < 3:
        return _mm(node)   # fallback to 2-D mm
    B, M, K = a.shape[0], a.shape[1], a.shape[2]
    N = b.shape[2]
    it = a.dtype.itemsize
    flops = 2.0 * B * M * N * K
    read  = (B * M * K + B * K * N) * it
    write = B * M * N * it
    return flops, read, write


def _linear(node: "OpNode") -> FMR:
    """aten.linear.default: input=(*,I), weight=(O,I), optional bias=(O,)"""
    if len(node.inputs) < 2:
        return _default(node)
    inp, weight = node.inputs[0], node.inputs[1]
    if len(inp.shape) < 1 or len(weight.shape) < 2:
        return _default(node)
    I = inp.shape[-1]
    O = weight.shape[0]
    batch = _numel(inp.shape[:-1])
    it = inp.dtype.itemsize
    flops = 2.0 * batch * O * I
    read  = (batch * I + O * I) * it
    write = batch * O * it
    if len(node.inputs) >= 3:   # bias
        bias = node.inputs[2]
        flops += batch * O
        read  += _numel(bias.shape) * it
    return flops, read, write


def _scaled_dot_product_attention(node: "OpNode") -> FMR:
    """aten._scaled_dot_product_flash_attention / scaled_dot_product_attention.

    Query shape: (N, H, Sq, D)  or  (N, Sq, H, D)
    Key   shape: (N, H, Sk, D)  or  (N, Sk, H, D)
    Value shape: (N, H, Sk, Dv) or  (N, Sk, H, Dv)
    """
    if len(node.inputs) < 3:
        return _default(node)
    q, k, v = node.inputs[0], node.inputs[1], node.inputs[2]
    if len(q.shape) < 4 or len(k.shape) < 4:
        return _default(node)
    # Assume (N, H, Sq, D) layout
    N, H, Sq, D = q.shape[0], q.shape[1], q.shape[2], q.shape[3]
    Sk = k.shape[2]
    it = q.dtype.itemsize
    # QK matmul: 2*N*H*Sq*Sk*D,  AV matmul: 2*N*H*Sq*Sk*D
    flops = 4.0 * N * H * Sq * Sk * D
    # Softmax ops ~ 5*N*H*Sq*Sk (sub-dominant, included for completeness)
    flops += 5.0 * N * H * Sq * Sk
    read  = (N*H*Sq*D + N*H*Sk*D + N*H*Sk*D) * it   # Q + K + V
    write = (N*H*Sq*D) * it                           # output
    return flops, read, write


def _layer_norm(node: "OpNode") -> FMR:
    """aten.layer_norm.default / aten.native_layer_norm.default"""
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # mean(N) + var(2N) + norm(N) + scale(N) + shift(N) ≈ 5N flops
    flops = 5.0 * n
    # read: input + weight + bias (last dim)
    weight_size = inp.shape[-1] if inp.shape else 1
    read  = (n + 2 * weight_size) * it
    write = n * it
    return flops, read, write


def _rms_norm(node: "OpNode") -> FMR:
    """Fused rms_norm: fewer ops than layer_norm (no mean subtraction)."""
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # pow(N) + mean(N) + rsqrt(1) + mul(N) + scale(N) ≈ 4N flops
    flops = 4.0 * n
    weight_size = inp.shape[-1] if inp.shape else 1
    read  = (n + weight_size) * it
    write = n * it
    return flops, read, write


def _softmax(node: "OpNode") -> FMR:
    """aten._softmax.default / aten.softmax.int"""
    if not node.inputs:
        return _default(node)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    # max(N) + sub(N) + exp(N) + sum(N) + div(N) ≈ 5N
    flops = 5.0 * n
    read  = n * it
    write = n * it
    return flops, read, write


def _elementwise(node: "OpNode", ops_per_elem: float = 1.0) -> FMR:
    """Generic elementwise op: one (or a few) ops per output element."""
    if not node.outputs:
        return _default(node)
    out = node.outputs[0]
    n_out = _numel(out.shape)
    it = out.dtype.itemsize
    flops = ops_per_elem * n_out
    read  = float(node.total_input_bytes())
    write = float(node.total_output_bytes())
    return flops, read, write


def _embedding(node: "OpNode") -> FMR:
    """aten.embedding.default: random HBM reads, negligible FLOPs."""
    if not node.outputs:
        return _default(node)
    out = node.outputs[0]
    n = _numel(out.shape)
    it = out.dtype.itemsize
    flops = 0.0
    read  = n * it
    write = n * it
    return flops, read, write


def _default(node: "OpNode") -> FMR:
    """Conservative fallback: 1 flop / output element."""
    n_out = sum(_numel(o.shape) for o in node.outputs) if node.outputs else 1
    it = _itemsize(node)
    flops = float(n_out)
    read  = float(node.total_input_bytes())
    write = float(node.total_output_bytes())
    return flops, read, write


# ── fused op formulas ─────────────────────────────────────────────────────────

def _fused_attention(node: "OpNode") -> FMR:
    """flash_attn / sdpa / npu_fusion_attention / attn / mla_attn.

    Assumes external inputs are ordered: [Q, K, V, ...].
    Falls back to default if shapes are insufficient.
    """
    if len(node.inputs) >= 3:
        return _scaled_dot_product_attention(node)
    # single-tensor attention (e.g., compact fused node)
    return _default(node)


def _fused_norm(node: "OpNode") -> FMR:
    """rms_norm / layer_norm / add_rms_norm / add_layer_norm."""
    if not node.inputs:
        return _default(node)
    # For add_norm variants: FLOPs = 4-5 * N + N (for the add)
    inp = node.inputs[0]
    n = _numel(inp.shape)
    it = inp.dtype.itemsize
    flops = 6.0 * n     # 5 for norm + 1 for residual add
    weight_size = inp.shape[-1] if inp.shape else 1
    read  = (n * 2 + weight_size) * it   # input + residual + weight
    write = n * it
    return flops, read, write


def _fused_mlp(node: "OpNode") -> FMR:
    """gated_mlp / mlp: contains 2-3 matmuls + activation.

    Inputs are typically: [hidden=(B,S,H), gate_w=(I,H), up_w=(I,H), down_w=(H,I)]
    Where I = intermediate_size, H = hidden_size.
    We estimate from available weight shapes.
    """
    if len(node.inputs) < 2:
        return _default(node)

    hidden = node.inputs[0]
    it = hidden.dtype.itemsize

    if len(hidden.shape) < 2:
        return _default(node)

    batch = _numel(hidden.shape[:-1])   # B*S
    H = hidden.shape[-1]

    # Collect intermediate sizes from weight tensors
    mm_flops = 0.0
    mm_read  = batch * H * it   # hidden state read once
    mm_write = 0.0

    # Each weight matrix contributes one matmul
    for w in node.inputs[1:]:
        if len(w.shape) < 2:
            continue
        # weight shape: (out_features, in_features) or (in, out)
        s0, s1 = w.shape[0], w.shape[1]
        # Infer which dim matches H
        if s1 == H:
            O = s0
        elif s0 == H:
            O = s1
        else:
            O = max(s0, s1)
        mm_flops += 2.0 * batch * H * O
        mm_read  += s0 * s1 * it        # read weight
        mm_write += batch * O * it

    # Elementwise (activation + mul for gated MLP)
    n_out = _numel(node.outputs[0].shape) if node.outputs else batch * H
    elem_flops = 4.0 * (n_out // 2 if n_out > batch * H else n_out)

    flops = mm_flops + elem_flops
    read  = mm_read
    write = (node.outputs[0].mem_bytes if node.outputs
             else batch * H * it)
    return flops, read, write


def _fused_moe_gate(node: "OpNode", with_topk: bool = False) -> FMR:
    """moe_gate / moe_gate_topk: small matmul + softmax (+ topk)."""
    # Dominant cost: one matmul to compute gate scores
    if len(node.inputs) >= 2:
        flops, read, write = _linear(node) if len(node.inputs[1].shape) >= 2 else _default(node)
        # Add softmax cost
        n_out = sum(_numel(o.shape) for o in node.outputs) if node.outputs else 1
        flops += 5.0 * n_out
        if with_topk:
            flops += n_out   # topk comparison ops
        return flops, read, write
    return _default(node)


# ── op → formula dispatch table ──────────────────────────────────────────────

# Maps op_type (exact match) → formula function.
# Check is "op_type starts with <key>" for prefix entries.

_EXACT_FORMULAS: dict[str, "callable"] = {
    # matmul
    "aten.mm.default":                  _mm,
    "aten.mm":                          _mm,
    "aten.addmm.default":               _addmm,
    "aten.addmm":                       _addmm,
    "aten.bmm.default":                 _bmm,
    "aten.bmm":                         _bmm,
    "aten.linear.default":              _linear,
    "aten.linear":                      _linear,
    # attention
    "aten._scaled_dot_product_flash_attention.default": _scaled_dot_product_attention,
    "aten.scaled_dot_product_attention.default":        _scaled_dot_product_attention,
    "aten._scaled_dot_product_efficient_attention.default": _scaled_dot_product_attention,
    # norm
    "aten.layer_norm.default":          _layer_norm,
    "aten.layer_norm":                  _layer_norm,
    "aten.native_layer_norm.default":   _layer_norm,
    # softmax
    "aten._softmax.default":            _softmax,
    "aten.softmax.int":                 _softmax,
    "aten.special_softmax.int":         _softmax,
    # elementwise — 1 op/elem
    "aten.add.Tensor":                  _elementwise,
    "aten.add_.Tensor":                 _elementwise,
    "aten.sub.Tensor":                  _elementwise,
    "aten.mul.Tensor":                  _elementwise,
    "aten.mul.Scalar":                  _elementwise,
    "aten.div.Tensor":                  _elementwise,
    "aten.div.Scalar":                  _elementwise,
    "aten.neg.default":                 _elementwise,
    "aten.abs.default":                 _elementwise,
    "aten.relu.default":                _elementwise,
    "aten.relu_.default":               _elementwise,
    "aten.tanh.default":                _elementwise,
    "aten.exp.default":                 _elementwise,
    "aten.log.default":                 _elementwise,
    "aten.sqrt.default":                _elementwise,
    "aten.rsqrt.default":               _elementwise,
    "aten.pow.Tensor_Scalar":           _elementwise,
    "aten.pow.Tensor_Tensor":           _elementwise,
    # activation — ~4 ops/elem
    "aten.silu.default":                lambda n: _elementwise(n, 4.0),
    "aten.silu_.default":               lambda n: _elementwise(n, 4.0),
    "aten.gelu.default":                lambda n: _elementwise(n, 4.0),
    "aten.sigmoid.default":             lambda n: _elementwise(n, 4.0),
    # embedding
    "aten.embedding.default":           _embedding,
    # reduction
    "aten.mean.dim":                    lambda n: _elementwise(n, 1.0),
    "aten.sum.dim_IntList":             lambda n: _elementwise(n, 1.0),
    "aten.var.correction":              lambda n: _elementwise(n, 3.0),
    # memory / shape — trivial compute
    "aten.copy_.default":               lambda n: (0.0, float(n.total_input_bytes()), float(n.total_output_bytes())),
    # fused semantic labels (from fusion engine)
    "rms_norm":                         _rms_norm,
    "layer_norm":                       _layer_norm,
    "add_rms_norm":                     _fused_norm,
    "add_layer_norm":                   _fused_norm,
    "flash_attn":                       _fused_attention,
    "sdpa":                             _fused_attention,
    "npu_fusion_attention":             _fused_attention,
    "attn":                             _fused_attention,
    "mla_attn":                         _fused_attention,
    "gated_mlp":                        _fused_mlp,
    "mlp":                              _fused_mlp,
    "moe_gate":                         lambda n: _fused_moe_gate(n, with_topk=False),
    "moe_gate_topk":                    lambda n: _fused_moe_gate(n, with_topk=True),
    "npu_moe_gate":                     lambda n: _fused_moe_gate(n, with_topk=False),
    "npu_moe_gate_topk":                lambda n: _fused_moe_gate(n, with_topk=True),
    "moe_block":                        _fused_mlp,
    "moe_expert":                       _fused_mlp,
    "rope":                             lambda n: _elementwise(n, 2.0),
}


def _shape_ops_fmr(node: "OpNode") -> FMR:
    """Shape/view/permute ops: near-zero compute, read ≈ write."""
    it = _itemsize(node)
    n = _numel(node.outputs[0].shape) if node.outputs else 1
    return 0.0, n * it, n * it


_SHAPE_OP_PREFIXES: tuple[str, ...] = (
    "aten.view", "aten._unsafe_view", "aten.reshape",
    "aten.expand", "aten.squeeze", "aten.unsqueeze",
    "aten.permute", "aten.transpose", "aten.contiguous",
    "aten.flatten", "aten.as_strided", "aten.select",
    "aten.slice", "aten.clone", "aten.t.", "aten.chunk",
    "aten.split", "aten.unbind", "aten.detach", "aten.alias",
    "aten.cat", "aten.stack",
)


# ── RooflineSimulator ─────────────────────────────────────────────────────────

class RooflineSimulator(OpSimulator):
    """Theoretical Roofline model — universal fallback backend.

    Uses pre-registered analytic formulas keyed by op_type.
    Any op without a formula uses the default fallback (1 flop / output elem).

    This backend always returns True from ``can_simulate()``, making it the
    guaranteed last resort in ``SimulatorHub``.
    """

    name = "roofline"
    priority = 0

    def can_simulate(self, node: "OpNode", hw: "HardwareSpec") -> bool:
        return True

    def simulate(self, node: "OpNode", hw: "HardwareSpec") -> SimResult:
        flops, read_bytes, write_bytes = self._fmr(node)
        total_bytes = read_bytes + write_bytes

        dtype = _primary_dtype(node)
        peak  = hw.peak_flops(dtype)        # ops/s
        bw    = hw.hbm_bandwidth()          # bytes/s

        compute_us = (flops / peak * 1e6)   if peak > 0 else 0.0
        memory_us  = (total_bytes / bw * 1e6) if bw > 0  else 0.0

        # Latency bound: kernel launch overhead (minimum ~1 µs for GPUs/NPUs)
        latency_us = max(compute_us, memory_us, 1e-3)

        ai = flops / total_bytes if total_bytes > 0 else math.inf

        if compute_us > 0 or memory_us > 0:
            bound = "compute" if compute_us >= memory_us else "memory"
        else:
            bound = "latency"

        hw_util = 0.0
        if peak > 0 and latency_us > 0:
            actual_rate = flops / (latency_us * 1e-6)
            hw_util = min(1.0, actual_rate / peak)

        return SimResult(
            op_node_id        = node.id,
            latency_us        = latency_us,
            compute_us        = compute_us,
            memory_us         = memory_us,
            flops             = int(flops),
            read_bytes        = int(read_bytes),
            write_bytes       = int(write_bytes),
            arithmetic_intensity = ai,
            bound             = bound,
            hw_utilization    = hw_util,
            backend           = self.name,
            confidence        = 0.3,
        )

    # ── FLOPs / Memory formula dispatch ──────────────────────────────────────

    def _fmr(self, node: "OpNode") -> FMR:
        op = node.op_type

        # 1. Exact match
        fn = _EXACT_FORMULAS.get(op)
        if fn is not None:
            return fn(node)

        # 2. Shape / transparent ops
        for prefix in _SHAPE_OP_PREFIXES:
            if op.startswith(prefix):
                return _shape_ops_fmr(node)

        # 3. Fused node: sum sub-op estimates if fused_from is available
        if node.is_fused and node.fused_from:
            return self._fused_decompose(node)

        # 4. Fallback
        return _default(node)

    def _fused_decompose(self, node: "OpNode") -> FMR:
        """Sum up FLOPs/memory for all sub-ops listed in fused_from.

        Since we don't have intermediate tensor shapes, we use the node's
        external inputs and outputs to estimate the dominant matmul costs.
        For everything else, we use the output-element heuristic.
        """
        total_flops = 0.0
        total_read  = float(node.total_input_bytes())
        total_write = float(node.total_output_bytes())

        for sub_op in node.fused_from:
            fn = _EXACT_FORMULAS.get(sub_op)
            if fn is not None:
                # Reuse the node's shapes as a proxy
                f, r, w = fn(node)
                total_flops += f
            else:
                # Unknown sub-op: 1 flop / output elem
                total_flops += sum(_numel(o.shape) for o in node.outputs)

        # Clamp read/write to at least actual tensor bytes
        total_read  = max(total_read,  float(node.total_input_bytes()))
        total_write = max(total_write, float(node.total_output_bytes()))

        return total_flops, total_read, total_write
