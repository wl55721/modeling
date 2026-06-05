from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import TYPE_CHECKING, Literal

from ..common.tensor_base import TensorBase

if TYPE_CHECKING:
    from ..chips.config import AIChipConfig

BoundType = Literal["compute", "memory", "communication", "none"]


@dataclass
class TensorInfo:
    name: str = ""
    shape: list[int] = field(default_factory=list)
    dtype: str = ""


@dataclass
class OperatorExecuteResult:
    op_id: int = 0
    op_name: str = ""
    layer_idx: int = 0
    rank_idx: int = 0
    compute_flops: int = 0
    input_bytes: int = 0
    param_bytes: int = 0
    cache_bytes: int = 0
    output_bytes: int = 0
    bw_bytes: int = 0
    comm_bytes: int = 0
    compute_cost_us: float = 0.0
    bw_cost_us: float = 0.0
    comm_cost_us: float = 0.0
    total_cost_us: float = 0.0
    static_cost_us: float = 0.0
    bound_type: BoundType = "none"
    start_time_ns: int = 0
    end_time_ns: int = 0
    inputs_info: list[TensorInfo] = field(default_factory=list)
    params_info: list[TensorInfo] = field(default_factory=list)
    outputs_info: list[TensorInfo] = field(default_factory=list)


class OperatorBase:

    def __init__(self, weights: list[TensorBase] | None = None, is_vector: bool = False, **attrs):
        self.inputs: list[TensorBase] = []
        self.params: list[TensorBase] = list(weights) if weights else []
        self.caches: list[TensorBase] = []
        self.outputs: list[TensorBase] = []
        self.cfg: dict = attrs
        self.op_id: int = 0
        self.op_name: str = ""
        self.layer_idx: int = 0
        self.rank_idx: int = 0
        self.op_module: str = ""
        self.compute_flops_str: str = "0"
        self.compute_flops: int = 0
        self.sfu_compute_flops: int = 0
        self.input_bytes: int = 0
        self.param_bytes: int = 0
        self.cache_bytes: int = 0
        self.output_bytes: int = 0
        self.bw_bytes: int = 0
        self.comm_bytes: int = 0
        self.static_cost_us = 0
        self.execute_result = OperatorExecuteResult()
        self.is_fusable = True # 是否可融合优化
        self.is_vector = is_vector

    def __call__(self, input_tensors: list[TensorBase], out_tensors: list[TensorBase]) -> list[TensorBase]:
        self.inputs = input_tensors or []
        self.outputs = out_tensors or []

        if not self.inputs:
            return self.outputs
        
        self.dynamic_update_b_s()

        self.calc_compute_flops()
        self.calc_bw_bytes()
        self.calc_comm_bytes()
        return self.outputs
    
    def dynamic_update_b_s(self):
        if not self.inputs:
            return self.outputs
        
        q_len, kv_len = self.parse_q_and_kv_length()
        is_attn_mlp_lmhead = (
            "attn" in self.op_module
            or "mlp" in self.op_module
            or "lm_head" in self.op_module
            or "mhc" in self.op_module
        )

        is_moe_gate_topk = (
            "moe" in self.op_module 
            and ("gate" in self.op_module or "topk" in self.op_module)
        )

        batch_size = self.cfg.get('batch_size', 1)
        dp_size = self.cfg.get('dp_size', 1)
        attn_bs = math.ceil(batch_size / dp_size)
        if is_attn_mlp_lmhead or is_moe_gate_topk:
            self.update_bsz_qlen_kvlen(attn_bs, q_len, kv_len)
            return
        
        if "routed_expert" in self.op_module:
            top_k = self.cfg.get('top_k', 6)
            ep_size = self.cfg.get('ep_size', 16)
            bsz = math.ceil(math.ceil(batch_size / ep_size) * top_k)
            self.update_bsz_qlen_kvlen(bsz, q_len, kv_len)
            return
        
        if "shared_expert" in self.op_module:
            external_shared_expert_rank_size = self.cfg.get('external_shared_expert_rank_size', 0)
            if external_shared_expert_rank_size > 0:
                n_shared_experts = self.cfg.get('n_shared_experts', 1)
                bsz = math.ceil(batch_size * n_shared_experts / external_shared_expert_rank_size)
            else:
                bsz = attn_bs
            self.update_bsz_qlen_kvlen(bsz, q_len, kv_len)
            return
        
        if "moe" in self.op_module and ("MoEDispatch" in self.op_name or "MoECombine" in self.op_name):
            # TODO: 负载不均衡
            bsz = attn_bs
            self.update_bsz_qlen_kvlen(bsz, q_len, kv_len)
            return
        
        if "moe" in self.op_module:
            bsz = attn_bs
            self.update_bsz_qlen_kvlen(bsz, q_len, kv_len)
            return
        
        if "mtp" in self.op_module:
            self.update_bsz_qlen_kvlen(attn_bs, q_len, kv_len)
            return

    def parse_q_and_kv_length(self):
        in_len = self.cfg.get('input_length', 1)
        out_len = self.cfg.get('output_length', 1)
        is_prefill = self.cfg.get('phase', "decode") == "prefill"

        if "mtp" in self.op_module:
            return self.parse_mtp_q_and_kv_length()

        num_mtp_tokens = self.cfg.get('num_mtp_tokens', 0)
        prefix_hit_ratio = self.cfg.get('prefix_hit_ratio', 0)

        qkv_list = ["input_norm", "attn"]
        if is_prefill and prefix_hit_ratio > 0 and self._contains_any(qkv_list, self.op_module):
            kv_len = math.ceil(in_len * prefix_hit_ratio) // 128 * 128
            q_len = in_len - kv_len
        elif is_prefill:
            q_len = in_len
            kv_len = 0
        else: # decode
            q_len = 1 if num_mtp_tokens == 0 else (1 + num_mtp_tokens)
            kv_len = in_len + out_len // 2
        
        return q_len, kv_len
    
    def parse_mtp_q_and_kv_length(self):
        in_len = self.cfg.get('input_length', 1)
        out_len = self.cfg.get('output_length', 1)
        is_prefill = self.cfg.get('phase', "decode") == "prefill"
        if is_prefill:
            if self.layer_idx == 980: # the first mtp layer_idx is 980
                q_len = in_len + 1
                kv_len = 0
            else:
                q_len = 1
                kv_len = in_len + (self.layer_idx - 980)
        else: # decode
            avg_accept_tokens = self.cfg.get('avg_accept_tokens', 1)
            if self.layer_idx == 980:
                q_len = round(avg_accept_tokens)
                kv_len = in_len + out_len // 2
            else:
                q_len = 1
                kv_len = in_len + round(avg_accept_tokens) + (self.layer_idx - 980 - 1) + out_len // 2
        return q_len, kv_len
    
    def update_bsz_qlen_kvlen(self, bsz, qlen, kvlen):
        x_shape = self.inputs[0].shape
        if len(x_shape) >= 3:
            x_shape[0] = bsz
            x_shape[1] = qlen

        o_shape = self.outputs[0].shape
        if len(o_shape) >= 3:
            o_shape[0] = bsz
            o_shape[1] = qlen
    
        self.cfg["B"] = bsz
        self.cfg["S"] = qlen

    def _contains_any(self, lst, text):
        if "o_proj" in text:
            return False
        return any(item in text for item in lst)

    # ── 自动成本计算 ──────────────────────────────────

    def calc_compute_flops(self):
        self.compute_flops = eval(self.compute_flops_str, {"__builtins__": {}}, self.cfg)

    def calc_bw_bytes(self):
        self.input_bytes = sum(t.nbytes for t in self.inputs)
        self.param_bytes = sum(t.nbytes for t in self.params)
        self.output_bytes = sum(t.nbytes for t in self.outputs)
        self.bw_bytes = self.input_bytes + self.param_bytes + self.cache_bytes + self.output_bytes

    def calc_comm_bytes(self):
        self.comm_bytes = 0

    # ── 耗时计算 ──────────────────────────────────────

    def fused_optim(self, chip: AIChipConfig):
        """融合优化：根据芯片规格调整输入输出大小，模拟 L2 Cache 效果。"""
        l2_cache_bytes = chip.l2_cache_size * 1024 * 1024
        if self.input_bytes < l2_cache_bytes:
            self.input_bytes = 0
        if self.output_bytes < l2_cache_bytes:
            self.output_bytes = 0

    def calc_bw_gmem_cost_us(self, chip: AIChipConfig) -> float:
        """计算显存带宽各部分耗时（微秒），返回明细 dict。"""
        if self.is_fusable and chip.fused_optim:
            self.fused_optim(chip)

        chip_gmem_bw = chip.spec_bw_memory * chip.bw_gmem_ratio
        if self.is_vector and chip.vendor == "HUAWEI":
            chip_gmem_bw *= 1.01
        
        self.bw_bytes = self.input_bytes + self.param_bytes + self.cache_bytes + self.output_bytes
        # KB/GB = us
        bw_gmem_cost_us = self.bw_bytes / 1024.0 / chip_gmem_bw if chip_gmem_bw > 0 else 0

        return bw_gmem_cost_us

    def calc_compute_cost_us(self, chip: AIChipConfig) -> float:
        if self.is_vector:
            chip_flops = chip.spec_vect_fp16 * chip.compute_ratio * 2
        else:
            chip_flops = chip.spec_cube_fp16 * chip.compute_ratio * 2

        chip_flops_sfu = chip.spec_sfu_fp16 * chip.compute_ratio * 2

        if chip_flops_sfu == 0:
            self.compute_flops += self.sfu_compute_flops
            compute_cost_ns = self.compute_flops / chip_flops if chip_flops > 0 else 0
        else:
            compute_cost_ns = (self.compute_flops / chip_flops if chip_flops > 0 else 0) \
                + (self.sfu_compute_flops / chip_flops_sfu if chip_flops_sfu > 0 else 0)
            
        return compute_cost_ns / 1000.0

    def calc_cost_us(self, chip: AIChipConfig) -> float:
        """根据芯片对象计算耗时并填充 execute_result。"""

        comp_us = self.calc_compute_cost_us(chip)
        mem_us = self.calc_bw_gmem_cost_us(chip)

        self.execute_result = OperatorExecuteResult(
            op_id=self.op_id,
            op_name=self.op_name,
            layer_idx=self.layer_idx,
            rank_idx=self.rank_idx,
            compute_flops=self.compute_flops,
            input_bytes=self.input_bytes,
            param_bytes=self.param_bytes,
            cache_bytes=self.cache_bytes,
            output_bytes=self.output_bytes,
            bw_bytes=self.bw_bytes,
            comm_bytes=self.comm_bytes,
            compute_cost_us=comp_us,
            bw_cost_us=mem_us,
            comm_cost_us=0,
            total_cost_us=max(comp_us, mem_us) + self.static_cost_us,
            static_cost_us=self.static_cost_us,
            bound_type="compute" if comp_us > mem_us else "memory",
            inputs_info=[TensorInfo(t.name, t.shape, t.dtype.name.lower()) for t in self.inputs],
            params_info=[TensorInfo(t.name, t.shape, t.dtype.name.lower()) for t in self.params],
            outputs_info=[TensorInfo(t.name, t.shape, t.dtype.name.lower()) for t in self.outputs],
        )
        return self.execute_result.total_cost_us


class OpCubeBase(OperatorBase):
    """Cube 单元算子基类：矩阵乘法等密集计算，瓶颈在 Cube FLOPS。"""

    compute_unit = "cube"

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        super().__init__(weights, is_vector=False, **attrs)
        self.static_cost_us = 5


class OpVectorBase(OperatorBase):
    """Vector 单元算子基类：逐元素操作，瓶颈在 Vector FLOPS。"""

    compute_unit = "vector"

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        super().__init__(weights, is_vector=True, **attrs)
        self.static_cost_us = 2


class OpMixBase(OperatorBase):
    """混合单元算子基类：同时使用 Cube 和 Vector，瓶颈在总 FLOPS。"""

    compute_unit = "mix"

    def __init__(self, weights: list[TensorBase] | None = None, **attrs):
        super().__init__(weights, is_vector=False, **attrs)
        self.static_cost_us = 3


class OpCommBase(OperatorBase):
    """通信算子基类：无计算、无权重，带宽消耗来自输入/输出张量传输。"""

    def __init__(self, weights: list[TensorBase] | None = None, comm_tensor: TensorBase | None = None, **attrs):
        super().__init__(weights, is_vector=False, **attrs)
        self.comm_ratio = 1.0
        self.static_cost_us = 10
        self.comm_tensor: TensorBase | None = comm_tensor
        self.rank_size: int = attrs.get("rank_size", 1)

    def calc_comm_bytes(self):
        self.comm_bytes = self.comm_tensor.nbytes if self.comm_tensor else 0

    def calc_cost_us(self, chip: AIChipConfig) -> float:
        chip_gmem_bw = chip.spec_bw_memory * chip.bw_gmem_ratio
        gmem_cost_us = self.bw_bytes / 1024.0 / chip_gmem_bw if chip_gmem_bw > 0 else 0

        if self.rank_size <= chip.bwsio_limit:
            comm_bw = chip.spec_comm_bwsio * chip.comm_bwsio_ratio * self.comm_ratio
        elif self.rank_size <= chip.superpod_limit:
            comm_bw = chip.spec_comm_intra * chip.comm_intra_ratio * self.comm_ratio
        else:
            comm_bw = chip.spec_comm_inter * chip.comm_inter_ratio * self.comm_ratio
        comm_cost_us = self.comm_bytes / 1024.0 / comm_bw if comm_bw > 0 else 0
        total_cost_us = max(comm_cost_us, gmem_cost_us) + self.static_cost_us
        self.execute_result = OperatorExecuteResult(
            op_id=self.op_id,
            op_name=self.op_name,
            layer_idx=self.layer_idx,
            rank_idx=self.rank_idx,
            compute_flops=0,
            input_bytes=0,
            param_bytes=0,
            cache_bytes=0,
            output_bytes=0,
            bw_bytes=self.comm_bytes,
            comm_bytes=self.comm_bytes,
            compute_cost_us=0.0,
            bw_cost_us=gmem_cost_us,
            comm_cost_us=comm_cost_us,
            total_cost_us=total_cost_us,
            static_cost_us=self.static_cost_us,
            bound_type="communication",
            inputs_info=[TensorInfo(t.name, t.shape, t.dtype.name.lower()) for t in self.inputs],
            params_info=[TensorInfo(t.name, t.shape, t.dtype.name.lower()) for t in self.params],
            outputs_info=[TensorInfo(t.name, t.shape, t.dtype.name.lower()) for t in self.outputs],
        )
        return self.execute_result.total_cost_us
