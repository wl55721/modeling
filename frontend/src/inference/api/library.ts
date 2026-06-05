import axios from 'axios'
import type { OperatorDef, ModelStructure, HardwareSpec } from '../types/model'
import type { SimulateSingleResult } from '../types/results'

const api = axios.create({ baseURL: '/api' })

export async function fetchOperatorList(): Promise<string[]> {
  const { data } = await api.get('/library/operators')
  return data
}

export async function fetchOperator(name: string): Promise<OperatorDef> {
  const { data } = await api.get(`/library/operators/${name}`)
  return data
}

export async function fetchModelList(): Promise<string[]> {
  const { data } = await api.get('/library/models')
  return data
}

export async function fetchModel(name: string): Promise<ModelStructure> {
  const { data } = await api.get(`/library/models/${name}`)
  return data
}

export async function saveModel(model: ModelStructure): Promise<void> {
  await api.post('/library/models', model)
}

export async function deleteModel(name: string): Promise<void> {
  await api.delete(`/library/models/${name}`)
}

export interface HardwareListItem {
  name: string
  chip: string
  vendor: string
}

export async function fetchHardwareList(): Promise<HardwareListItem[]> {
  const { data } = await api.get('/library/hardwares')
  return data
}

export async function fetchHardware(name: string): Promise<HardwareSpec> {
  const { data } = await api.get(`/library/hardwares/${name}`)
  return data
}

export async function saveHardware(hw: HardwareSpec): Promise<void> {
  await api.post('/library/hardwares', hw)
}

export async function deleteHardware(name: string): Promise<void> {
  await api.delete(`/library/hardwares/${name}`)
}

export async function fetchModuleList(): Promise<string[]> {
  const { data } = await api.get('/library/modules')
  return data
}

export async function fetchModule(name: string): Promise<OperatorDef[]> {
  const { data } = await api.get(`/library/modules/${name}`)
  return data
}

export async function fetchHfConfigList(): Promise<string[]> {
  const { data } = await api.get('/library/hf_configs')
  return data
}

export async function fetchHfConfig(name: string): Promise<object> {
  const { data } = await api.get(`/library/hf_configs/${name}`)
  return data
}

export interface WorkloadRequest {
  phase: string
  batch_size: number
  input_length: number
  output_length: number
  num_mtp_tokens: number
  ratio_mtp_tokens: number
  avg_accept_tokens: number
  prefix_hit_ratio: number
}

export interface WorkloadParallel {
  world_size: number
  tp_size: number
  dp_size: number
  pp_size: number
  ep_size: number
  cp_size: number
  embed_tp_size: number
  o_tp_size: number
  lmhead_tp_size: number
  external_shared_expert_rank_size: number
}

export interface WorkloadQuant {
  quant_global: string
  quant_mlp: string
  quant_shared_expert: string
  quant_routed_expert: string
  quant_kv_cache: string
  quant_activation: string
}

export interface WorkloadEntry {
  request: WorkloadRequest
  parallel: WorkloadParallel
  quant: WorkloadQuant
}

export interface HardwareEntry {
  name: string
  config: Record<string, unknown>
}

export interface SimulateParams {
  workloads: WorkloadEntry[]
  hardwares: HardwareEntry[]
  model_name?: string
  model_json?: object
  hf_config_json?: object
}

export interface SimulateMultiResponse {
  results: SimulateSingleResult[]
}

export async function runSimulate(params: SimulateParams): Promise<SimulateMultiResponse> {
  const { data } = await api.post('/simulate', params)
  return data
}

// ── Auto-optimize ──

export interface OptimizeParams {
  model_name?: string
  model_json?: object
  hf_config_json?: object
  workload: {
    request: WorkloadRequest
    optimize: { target_tpot_ms: number; min_world_size?: number; max_world_size?: number; fine_grained?: boolean; embed_tp_min?: number; embed_tp_max?: number; o_tp_min?: number; o_tp_max?: number; lmhead_tp_min?: number; lmhead_tp_max?: number; batch_size_min?: number; batch_size_max?: number }
    quant: WorkloadQuant
  }
  hardwares: HardwareEntry[]
}

export interface StrategyResult {
  world_size: number
  tp_size: number
  dp_size: number
  embed_tp_size: number
  o_tp_size: number
  lmhead_tp_size: number
  batch_size: number
  strategy_label: string
  tpot_ms: number
  tps: number
  max_peak_mem_gb: number
  total_mem_gb: number
  is_oom: boolean
  meets_target: boolean
}

export interface HardwareOptimizeResult {
  hardware_name: string
  optimal: StrategyResult | null
  candidates: StrategyResult[]
  search_summary: {
    total_candidates: number
    evaluated: number
    pruned: number
    oom_count: number
    elapsed_ms: number
  }
}

export interface OptimizeResult {
  optimal: StrategyResult | null
  candidates: StrategyResult[]
  search_summary: {
    total_candidates: number
    evaluated: number
    pruned: number
    oom_count: number
    elapsed_ms: number
  }
  hardware_results: HardwareOptimizeResult[]
}

export async function runOptimize(params: OptimizeParams): Promise<OptimizeResult> {
  const { data } = await api.post('/optimize', params)
  return data
}
