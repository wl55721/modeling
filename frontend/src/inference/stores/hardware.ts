import { create } from 'zustand'
import * as api from '../api/library'

export interface HwFields {
  spec_memory_size: number
  spec_bw_memory: number
  spec_comm_intra: number
  spec_comm_inter: number
  spec_comm_bwsio: number
  compute_ratio: number
  bw_gmem_ratio: number
  comm_intra_ratio: number
  comm_inter_ratio: number
  comm_bwsio_ratio: number
  l2_cache_ratio: number
  memory_noise: number
  cube_core_cnt: number
  vector_core_cnt: number
  sfu_core_cnt: number
  cube_freq: number
  spec_cube_fp16: number
  spec_vect_fp16: number
  spec_sfu_fp16: number
  spec_l2cache_size: number
  superpod_limit: number
  bwsio_limit: number
  chip_name: string
}

export interface HwConfig {
  id: string
  name: string
  builtinName: string
  enabled: boolean
  fields: HwFields
}

const DEFAULT_FIELDS: HwFields = {
  spec_memory_size: 80,
  spec_bw_memory: 2.0,
  spec_comm_intra: 600,
  spec_comm_inter: 100,
  spec_comm_bwsio: 0,
  compute_ratio: 0.7,
  bw_gmem_ratio: 0.6,
  comm_intra_ratio: 0.5,
  comm_inter_ratio: 0.5,
  comm_bwsio_ratio: 0,
  l2_cache_ratio: 0,
  memory_noise: 5,
  cube_core_cnt: 0,
  vector_core_cnt: 0,
  sfu_core_cnt: 0,
  cube_freq: 0,
  spec_cube_fp16: 0,
  spec_vect_fp16: 0,
  spec_sfu_fp16: 0,
  spec_l2cache_size: 0,
  superpod_limit: 0,
  bwsio_limit: 0,
  chip_name: '',
}

let nextId = 1

interface MultiHwState {
  configs: HwConfig[]
  activeConfigId: string | null
  setActiveConfig: (id: string) => void
  addConfig: () => void
  removeConfig: (id: string) => void
  duplicateConfig: (id: string) => void
  updateConfigName: (id: string, name: string) => void
  setConfigBuiltin: (id: string, builtinName: string) => void
  toggleEnabled: (id: string) => void
  updateFields: (id: string, patch: Partial<HwFields>) => void
  loadBuiltinFields: (id: string, builtinName: string) => Promise<void>
}

export const useHardwareStore = create<MultiHwState>((set, get) => ({
  configs: [{
    id: String(nextId++),
    name: 'A5_POD',
    builtinName: 'A5_POD',
    enabled: true,
    fields: { ...DEFAULT_FIELDS },
  }],
  activeConfigId: '1',

  setActiveConfig: (id) => set({ activeConfigId: id }),

  addConfig: () => {
    const id = String(nextId++)
    set((s) => ({
      configs: [...s.configs, {
        id,
        name: `配置 ${s.configs.length + 1}`,
        builtinName: '',
        enabled: true,
        fields: { ...DEFAULT_FIELDS, chip_name: '' },
      }],
      activeConfigId: id,
    }))
  },

  removeConfig: (id) => {
    const { configs, activeConfigId } = get()
    if (configs.length <= 1) return
    const remaining = configs.filter((c) => c.id !== id)
    set({
      configs: remaining,
      activeConfigId: activeConfigId === id ? remaining[0].id : activeConfigId,
    })
  },

  duplicateConfig: (id) => {
    const cfg = get().configs.find((c) => c.id === id)
    if (!cfg) return
    const newId = String(nextId++)
    set((s) => ({
      configs: [...s.configs, {
        id: newId,
        name: `${cfg.name} (副本)`,
        builtinName: cfg.builtinName,
        enabled: true,
        fields: JSON.parse(JSON.stringify(cfg.fields)),
      }],
      activeConfigId: newId,
    }))
  },

  updateConfigName: (id, name) => set((s) => ({
    configs: s.configs.map((c) => c.id === id ? { ...c, name } : c),
  })),

  setConfigBuiltin: (id, builtinName) => set((s) => ({
    configs: s.configs.map((c) => c.id === id ? { ...c, builtinName } : c),
  })),

  toggleEnabled: (id) => set((s) => ({
    configs: s.configs.map((c) => c.id === id ? { ...c, enabled: !c.enabled } : c),
  })),

  updateFields: (id, patch) => set((s) => ({
    configs: s.configs.map((c) => c.id === id ? { ...c, fields: { ...c.fields, ...patch } } : c),
  })),

  loadBuiltinFields: async (id, builtinName) => {
    try {
      const hw = await api.fetchHardware(builtinName)
      const fields: HwFields = {
        chip_name: hw.chip_name || builtinName,
        spec_memory_size: hw.spec_memory_size ?? 80,
        spec_bw_memory: hw.spec_bw_memory ?? 2.0,
        spec_comm_intra: hw.spec_comm_intra ?? 600,
        spec_comm_inter: hw.spec_comm_inter ?? 100,
        spec_comm_bwsio: hw.spec_comm_bwsio ?? 0,
        compute_ratio: hw.compute_ratio ?? 0.7,
        bw_gmem_ratio: hw.bw_gmem_ratio ?? 0.6,
        comm_intra_ratio: hw.comm_intra_ratio ?? 0.5,
        comm_inter_ratio: hw.comm_inter_ratio ?? 0.5,
        comm_bwsio_ratio: hw.comm_bwsio_ratio ?? 0,
        l2_cache_ratio: hw.l2_cache_ratio ?? 0,
        memory_noise: hw.memory_noise ?? 5,
        cube_core_cnt: hw.cube_core_cnt ?? 0,
        vector_core_cnt: hw.vector_core_cnt ?? 0,
        sfu_core_cnt: hw.sfu_core_cnt ?? 0,
        cube_freq: hw.cube_freq ?? 0,
        spec_cube_fp16: hw.spec_cube_fp16 ?? 0,
        spec_vect_fp16: hw.spec_vect_fp16 ?? 0,
        spec_sfu_fp16: hw.spec_sfu_fp16 ?? 0,
        spec_l2cache_size: hw.spec_l2cache_size ?? 0,
        superpod_limit: hw.superpod_limit ?? 0,
        bwsio_limit: hw.bwsio_limit ?? 0,
      }
      set((s) => ({
        configs: s.configs.map((c) => c.id === id ? { ...c, fields, builtinName, name: builtinName } : c),
      }))
    } catch { /* ignore */ }
  },
}))

export function getHardwareConfigs(): { name: string; config: Record<string, unknown> }[] {
  return useHardwareStore.getState().configs
    .filter((c) => c.enabled)
    .map((c) => {
      const f = c.fields
      const isAscend = f.cube_core_cnt > 0
      const hasBwsio = f.spec_comm_bwsio > 0
      const hasSfuFlops = f.spec_sfu_fp16 > 0
      const out: Record<string, unknown> = {
        name: c.builtinName || f.chip_name || 'Custom',
        chip: f.chip_name || 'Custom',
        spec_memory_size: f.spec_memory_size,
        spec_bw_memory: f.spec_bw_memory,
        spec_comm_intra: f.spec_comm_intra,
        spec_comm_inter: f.spec_comm_inter,
        compute_ratio: f.compute_ratio,
        bw_gmem_ratio: f.bw_gmem_ratio,
        comm_intra_ratio: f.comm_intra_ratio,
        comm_inter_ratio: f.comm_inter_ratio,
        memory_noise: f.memory_noise,
        spec_cube_fp16: f.spec_cube_fp16,
        spec_vect_fp16: f.spec_vect_fp16,
        spec_l2cache_size: f.spec_l2cache_size,
        superpod_limit: f.superpod_limit,
      }
      // Ascend-specific: compute via core_cnt * freq
      if (isAscend) {
        out.cube_core_cnt = f.cube_core_cnt
        out.vector_core_cnt = f.vector_core_cnt
        out.sfu_core_cnt = f.sfu_core_cnt
        out.cube_freq = f.cube_freq
      }
      // Optional fields
      if (hasBwsio) {
        out.spec_comm_bwsio = f.spec_comm_bwsio
        out.comm_bwsio_ratio = f.comm_bwsio_ratio
      }
      if (hasSfuFlops) out.spec_sfu_fp16 = f.spec_sfu_fp16
      if (f.l2_cache_ratio > 0) out.l2_cache_ratio = f.l2_cache_ratio
      if (f.bwsio_limit > 0) out.bwsio_limit = f.bwsio_limit
      return { name: c.name, config: out }
    })
}

export type RatioKey = 'compute_ratio' | 'bw_gmem_ratio' | 'comm_intra_ratio' | 'comm_inter_ratio' | 'comm_bwsio_ratio' | 'l2_cache_ratio'
export const BASE_RATIO_FIELDS: { key: RatioKey; label: string }[] = [
  { key: 'compute_ratio', label: '算力效率' },
  { key: 'bw_gmem_ratio', label: '带宽效率' },
  { key: 'comm_intra_ratio', label: '机内通信效率' },
  { key: 'comm_inter_ratio', label: '机间通信效率' },
]
