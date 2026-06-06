export interface PerGPUResult {
  gpu_index: number
  memory_noise_gb: number
  peak_memory_gb: number
  memory_capacity_gb: number
  oom: boolean
  weight_memory_gb: number
  kv_cache_memory_gb: number
  activation_memory_gb: number
}

export interface TensorInfo {
  name: string
  shape: number[]
  dtype: string
}

export interface OperatorResult {
  op_id: number
  op_name: string
  layer_idx: number
  rank_idx: number
  op_module: string
  compute_cost_us: number
  mem_cost_us: number
  comm_cost_us: number
  bound_type: string
  total_cost_us: number
  noise_us: number
  start_time_ns: number
  end_time_ns: number
  inputs_info: TensorInfo[]
  params_info: TensorInfo[]
  outputs_info: TensorInfo[]
}

export interface OperatorStatistics {
  op_name: string
  total_cost_us: number
  num_calls: number
  avg_cost_us: number
}

export interface LayerResultPerRank {
  layer_idx: number
  rank_idx: number
  layer_cost_ns: number
  param_bytes: number
  io_bytes: number
  op_ids: number[]
  repeat: number
  start_time_ns: number
  end_time_ns: number
}

export interface RankResult {
  rank_idx: number
  repeat: number
  total_cost_ms: number
  peak_mem_gb: number
  noise_gb: number
  mem_capacity_gb: number
  oom: boolean
  num_ops: number
  ops: number[]
  param_bytes: number
  io_bytes: number
  layers: LayerResultPerRank[]
  start_time_ns: number
  end_time_ns: number
}

export interface EstimationResult {
  hardware_name: string
  tpot_ms: number
  tps: number
  qps: number
  prefill_latency_ms: number
  decode_latency_per_token_ms: number
  peak_mem_gb: number
  oom: boolean
  operators: OperatorResult[]
  op_statistics: OperatorStatistics[]
  ranks: RankResult[]
  strategy: string
  timestamp: string
  start_time_ns: number
  end_time_ns: number
}

export interface SimulateSingleResult {
  hardware_name: string
  result: EstimationResult
}
