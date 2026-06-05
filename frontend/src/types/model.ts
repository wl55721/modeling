export interface TensorMeta {
  name: string
  shape: string
  dtype: string
  factor?: number
}

export interface OperatorDef {
  name: string
  module: string
  description: string
  category?: string
  inputs: TensorMeta[]
  params: TensorMeta[]
  outputs: TensorMeta[]
  compute_flops: string
  rank_size?: string
}

export interface ModuleDef {
  name: string
  label: string
  operatorNames: string[]
  isBuiltin: boolean
}

export interface OpNodeData {
  index: number
  operator: OperatorDef
  label: string
  rank: number
  position?: { x: number; y: number }
}

export interface LayerConfig {
  id: string
  name: string
  repeat: number
  layerIdx: number
  kind: 'regular' | 'mtp'
  rankOps: Record<number, number[]>
}

export interface ModelStructure {
  name: string
  architecture: string
  num_layers: number
  hidden_dim: number
  vocab_size: number
  layers: OpNodeData[]
}

export interface HardwareSpec {
  hw_env: string
  chip_name: string
  vendor: string
  spec_memory_size: number
  spec_cube_fp16: number
  spec_vect_fp16: number
  spec_sfu_fp16: number
  spec_bw_memory: number
  spec_comm_intra: number
  spec_comm_inter: number
  spec_comm_bwsio: number
  spec_l2cache_size: number
  memory_noise: number
  compute_ratio: number
  bw_gmem_ratio: number
  comm_intra_ratio: number
  comm_inter_ratio: number
  comm_bwsio_ratio: number
  l2_cache_ratio: number
  superpod_limit: number
  bwsio_limit: number
  cube_core_cnt: number
  vector_core_cnt: number
  sfu_core_cnt: number
  cube_freq: number
}
