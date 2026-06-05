from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


# ── 请求 ───────────────────────────────────────────────

class RequestConfig(BaseModel):
    phase: str = "decode"
    batch_size: int = 1
    input_length: int = 2048
    output_length: int = 512
    num_mtp_tokens: int = 2
    ratio_mtp_tokens: float = 0.75
    avg_accept_tokens: float = 2.53
    prefix_hit_ratio: float = 0


class ParallelConfig(BaseModel):
    world_size: int = 8
    tp_size: int = 1
    dp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    cp_size: int = 1
    embed_tp_size: int = 1
    o_tp_size: int = 1
    lmhead_tp_size: int = 1
    external_shared_expert_rank_size: int = 0


class QuantConfig(BaseModel):
    quant_global: str = "fp16"
    quant_mlp: str = "fp16"
    quant_shared_expert: str = "fp16"
    quant_routed_expert: str = "fp16"
    quant_kv_cache: str = "fp16"
    quant_activation: str = "fp16"


class HardwareEntry(BaseModel):
    name: str = ""
    config: str | dict = Field(default_factory=dict)


class WorkloadEntry(BaseModel):
    request: RequestConfig = Field(default_factory=RequestConfig)
    parallel: ParallelConfig = Field(default_factory=ParallelConfig)
    quant: QuantConfig = Field(default_factory=QuantConfig)


class SimulateRequest(BaseModel):
    # 负载配置列表
    workloads: list[WorkloadEntry] = Field(default_factory=list)
    # 硬件配置列表
    hardwares: list[HardwareEntry] = Field(default_factory=list)
    hardware_name: str = ""  # 兼容旧字段
    # 模型来源：内置模型名称 或 自定义模型 JSON
    model_name: Optional[str] = None
    model_json: Optional[dict] = None
    # HuggingFace 模型配置 JSON（可替代 model_json）
    hf_config_json: Optional[dict] = None


# ── 响应 ───────────────────────────────────────────────

class PerGPUResult(BaseModel):
    gpu_index: int
    memory_noise_gb: float
    peak_memory_gb: float
    memory_capacity_gb: float
    oom: bool
    weight_memory_gb: float
    kv_cache_memory_gb: float
    activation_memory_gb: float


class TensorInfo(BaseModel):
    name: str = ""
    shape: list[int] = []
    dtype: str = ""


class OperatorResult(BaseModel):
    op_id: int
    op_name: str
    layer_idx: int
    rank_idx: int = 0
    op_module: str
    compute_cost_us: float
    mem_cost_us: float
    comm_cost_us: float
    bound_type: str
    total_cost_us: float
    noise_us: float
    start_time_ns: int = 0
    end_time_ns: int = 0
    inputs_info: list[TensorInfo] = []
    params_info: list[TensorInfo] = []
    outputs_info: list[TensorInfo] = []


class OperatorStatistics(BaseModel):
    op_name: str
    total_cost_us: float = 0.0
    num_calls: int = 0
    avg_cost_us: float = 0.0


class LayerResultPerRank(BaseModel):
    layer_idx: int
    rank_idx: int = 0
    layer_cost_ns: int = 0
    param_bytes: int = 0
    io_bytes: int = 0
    op_ids: list[int] = []
    repeat: int = 1
    start_time_ns: int = 0
    end_time_ns: int = 0


class RankResult(BaseModel):
    rank_idx: int
    repeat: int = 1
    total_cost_ms: float = 0.0
    param_bytes: int = 0
    io_bytes: int = 0
    noise_gb: float = 0.0
    mem_capacity_gb: float = 0.0
    peak_mem_gb: float = 0.0
    oom: bool = False
    num_ops: int = 0
    ops: list[int] = []
    # 层前和层后算子，分别构建特殊的层，放到layers中，统一处理
    layers: list[LayerResultPerRank] = Field(default_factory=list)
    start_time_ns: int = 0
    end_time_ns: int = 0


class SimulateResponse(BaseModel):
    hardware_name: str = ""
    tpot_ms: float
    tps: float
    qps: float
    prefill_latency_ms: float
    decode_latency_per_token_ms: float
    peak_mem_gb: float
    oom: bool
    operators: list[OperatorResult]
    op_statistics: list[OperatorStatistics]
    ranks: list[RankResult] = Field(default_factory=list)
    strategy: str
    timestamp: str
    start_time_ns: int = 0
    end_time_ns: int = 0


class SimulateSingleResult(BaseModel):
    hardware_name: str
    result: SimulateResponse


class SimulateMultiResponse(BaseModel):
    results: list[SimulateSingleResult]


class SaveHardwareRequest(BaseModel):
    hw_env: str = ""
    chip_name: str = ""
    vendor: str = ""
    spec_memory_size: int = 0
    spec_cube_fp16: float = 0
    spec_vect_fp16: float = 0
    spec_sfu_fp16: float = 0
    spec_bw_memory: float = 0
    spec_comm_intra: float = 0
    spec_comm_inter: float = 0
    spec_comm_bwsio: float = 0
    spec_l2cache_size: float = 0
    memory_noise: float = 0
    compute_ratio: float = 1.0
    bw_gmem_ratio: float = 1.0
    comm_intra_ratio: float = 1.0
    comm_inter_ratio: float = 1.0
    comm_bwsio_ratio: float = 1.0
    l2_cache_ratio: float = 1.0
    superpod_limit: int = 1
    bwsio_limit: int = 1
    cube_core_cnt: int = 0
    vector_core_cnt: int = 0
    sfu_core_cnt: int = 0
    cube_freq: float = 0.0


# ── 自动寻优 ───────────────────────────────────────────

class OptimizeSpec(BaseModel):
    """寻优目标参数"""
    target_tpot_ms: float = Field(..., gt=0, description="目标 decode 时延上限 (ms)")
    min_world_size: int = Field(default=8, ge=1, le=1024)
    max_world_size: int = Field(default=512, ge=1, le=1024)
    fine_grained: bool = True # 搜索粒度
    embed_tp_min: int = Field(default=2, ge=1)
    embed_tp_max: int = Field(default=16, ge=1)
    o_tp_min: int = Field(default=2, ge=1)
    o_tp_max: int = Field(default=16, ge=1)
    lmhead_tp_min: int = Field(default=2, ge=1)
    lmhead_tp_max: int = Field(default=16, ge=1)
    batch_size_min: int = Field(default=1, ge=1)
    batch_size_max: int = Field(default=65536, ge=1)


class OptimizeWorkload(BaseModel):
    """自动寻优的工作负载描述"""
    request: RequestConfig = Field(default_factory=RequestConfig)
    optimize: OptimizeSpec
    quant: QuantConfig = Field(default_factory=QuantConfig)


class OptimizeRequest(BaseModel):
    """自动寻优请求"""
    model_name: Optional[str] = None
    model_json: Optional[dict] = None
    hf_config_json: Optional[dict] = None

    workload: OptimizeWorkload

    hardwares: list[HardwareEntry] = Field(default_factory=list)
    hardware_name: str = ""


class StrategyResult(BaseModel):
    """单个候选策略的评估结果"""
    world_size: int
    tp_size: int
    dp_size: int
    embed_tp_size: int
    o_tp_size: int
    lmhead_tp_size: int
    batch_size: int = 1
    strategy_label: str
    tpot_ms: float
    tps: float
    max_peak_mem_gb: float
    total_mem_gb: float
    is_oom: bool
    meets_target: bool


class SearchSummary(BaseModel):
    """搜索过程摘要"""
    total_candidates: int
    evaluated: int
    pruned: int
    oom_count: int
    elapsed_ms: float


class HardwareOptimizeResult(BaseModel):
    """单个硬件的寻优结果"""
    hardware_name: str
    optimal: Optional[StrategyResult] = None
    candidates: list[StrategyResult] = Field(default_factory=list)
    search_summary: SearchSummary


class OptimizeResponse(BaseModel):
    """自动寻优响应"""
    optimal: Optional[StrategyResult] = None  # 全局最优
    candidates: list[StrategyResult] = Field(default_factory=list)  # 全部候选
    search_summary: SearchSummary
    hardware_results: list[HardwareOptimizeResult] = Field(default_factory=list)
