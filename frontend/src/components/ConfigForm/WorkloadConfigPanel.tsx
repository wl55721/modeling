import { useEffect } from 'react'
import { cn } from '../../utils/classnames'
import { create } from 'zustand'
import { useModelStore } from '../../stores/model'

function toPowerOfTwo(v: number, dir: 'up' | 'down'): number {
  if (v < 1) return 1
  let p = 1
  while (p < v) p *= 2
  if (dir === 'down' && p > v && p > 1) p /= 2
  return Math.max(1, p)
}

export interface InferenceParams {
  phase: string
  batch_size: number
  input_length: number
  output_length: number
  num_mtp_tokens: number
  ratio_mtp_tokens: number
  prefix_hit_ratio: number
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
  quant_global: string
  quant_mlp: string
  quant_shared_expert: string
  quant_routed_expert: string
  quant_kv_cache: string
  quant_activation: string
  optimizeMode: 'manual' | 'auto'
  targetTpotMs: number
  minWorldSize: number
  maxWorldSize: number
  embedTpMin: number
  embedTpMax: number
  oTpMin: number
  oTpMax: number
  lmheadTpMin: number
  lmheadTpMax: number
  batchSizeMin: number
  batchSizeMax: number
}

export const useInferenceStore = create<InferenceParams>(() => ({
  phase: 'decode',
  batch_size: 1,
  input_length: 8192,
  output_length: 1024,
  num_mtp_tokens: 2,
  ratio_mtp_tokens: 0.75,
  prefix_hit_ratio: 0.0,
  world_size: 8,
  tp_size: 8,
  dp_size: 4,
  pp_size: 1,
  ep_size: 8,
  cp_size: 1,
  embed_tp_size: 8,
  o_tp_size: 8,
  lmhead_tp_size: 8,
  external_shared_expert_rank_size: 0,
  quant_global: 'bf16',
  quant_mlp: 'bf16',
  quant_shared_expert: 'bf16',
  quant_routed_expert: 'bf16',
  quant_kv_cache: 'bf16',
  quant_activation: 'bf16',
  optimizeMode: 'manual',
  targetTpotMs: 20,
  minWorldSize: 8,
  maxWorldSize: 64,
  embedTpMin: 2,
  embedTpMax: 16,
  oTpMin: 2,
  oTpMax: 16,
  lmheadTpMin: 2,
  lmheadTpMax: 16,
  batchSizeMin: 1,
  batchSizeMax: 8192,
}))

const QUANT_GROUPS = [
  {
    label: '权重',
    fields: [
      { key: 'quant_global' as const, label: '全局', options: ['bf16', 'int8', 'fp8', 'fp4', 'int4'] },
      { key: 'quant_mlp' as const, label: 'MLP', options: ['bf16', 'int8', 'fp8', 'fp4', 'int4'] },
      { key: 'quant_shared_expert' as const, label: '共享专家', options: ['bf16', 'int8', 'fp8', 'fp4', 'int4'] },
      { key: 'quant_routed_expert' as const, label: '路由专家', options: ['bf16', 'int8', 'fp8', 'fp4', 'int4'] },
    ],
  },
  {
    label: 'KV Cache / 激活值',
    oneRow: true,
    fields: [
      { key: 'quant_kv_cache' as const, label: 'KV Cache', options: ['bf16', 'int8', 'fp8'] },
      { key: 'quant_activation' as const, label: '激活值', options: ['bf16', 'fp8', 'int8'] },
    ],
  },
]

export default function WorkloadConfig() {
  const p = useInferenceStore()
  const set = useInferenceStore.setState
  const mtpCount = useModelStore((s) => s.layers.filter((l) => l.kind === 'mtp').length)

  // num_mtp_tokens: auto=0 when no MTP layers, user-editable otherwise
  const hasMtp = mtpCount > 0
  useEffect(() => {
    if (!hasMtp && p.num_mtp_tokens !== 0) {
      set({ num_mtp_tokens: 0 })
    }
  }, [hasMtp, p.num_mtp_tokens, set])

  // PP is locked at 1
  useEffect(() => {
    if (p.pp_size !== 1) {
      set({ pp_size: 1 })
    }
  }, [p.pp_size, set])

  // CP is locked at 1
  useEffect(() => {
    if (p.cp_size !== 1) {
      set({ cp_size: 1 })
    }
  }, [p.cp_size, set])

  // World Size = TP × DP × PP — auto-sync when TP or DP changes
  useEffect(() => {
    const ws = p.tp_size * p.dp_size * p.pp_size
    if (p.world_size !== ws) {
      set({ world_size: ws })
    }
  }, [p.tp_size, p.dp_size, p.pp_size, p.world_size, set])

  // EP always tracks World Size
  useEffect(() => {
    if (p.ep_size !== p.world_size) {
      set({ ep_size: p.world_size })
    }
  }, [p.world_size, p.ep_size, set])

  return (
    <div className="inference-config">
      {/* ── Mode toggle ── */}
      <div className="opt-mode-bar">
        <button className={cn('opt-mode-btn', p.optimizeMode === 'manual' && 'active')}
          onClick={() => set({ optimizeMode: 'manual' } as any)}>手动仿真</button>
        <button className={cn('opt-mode-btn', p.optimizeMode === 'auto' && 'active')}
          onClick={() => set({ optimizeMode: 'auto' } as any)}>自动寻优</button>
      </div>

      {/* ── Request Config ── */}
      <div className="section">
        <div className="section-title">请求配置</div>
        <div className={cn('req-row', p.optimizeMode === 'auto' && 'req-row-auto')}>
          <div className={cn('req-card', p.optimizeMode === 'auto' && 'req-card-phase')}>
            <span className="req-label">Phase</span>
            <select className="req-select" value={p.phase}
              onChange={(e) => set({ phase: e.target.value })}>
              <option value="prefill">Prefill</option>
              <option value="decode">Decode</option>
            </select>
          </div>
          <div className={cn('req-card', p.optimizeMode === 'auto' && 'req-card-bs')}>
            <span className="req-label">Batch Size</span>
            {p.optimizeMode === 'auto' ? (
              <div className="hw-spec-input-wrap">
                <input type="number" className="req-input req-input-bs-min" value={p.batchSizeMin || ''} min={1}
                  onChange={(e) => set({ batchSizeMin: +e.target.value } as any)} />
                <span className="req-sep">—</span>
                <input type="number" className="req-input req-input-bs-max" value={p.batchSizeMax || ''} min={1}
                  onChange={(e) => set({ batchSizeMax: +e.target.value } as any)} />
              </div>
            ) : (
              <input type="number" className="req-input" value={p.batch_size || ''} min={1}
                onChange={(e) => set({ batch_size: +e.target.value })} />
            )}
          </div>
          <div className={cn('req-card', p.optimizeMode === 'auto' && 'req-card-in')}>
            <span className="req-label">Input</span>
            <input type="number" className="req-input" value={p.input_length || ''} min={1}
              onChange={(e) => set({ input_length: +e.target.value })} />
          </div>
          <div className={cn('req-card', p.optimizeMode === 'auto' && 'req-card-sm')}>
            <span className="req-label">Output</span>
            <input type="number" className="req-input" value={p.output_length || ''} min={1}
              onChange={(e) => set({ output_length: +e.target.value })} />
          </div>
          {p.optimizeMode === 'manual' && (
          <div className="req-card">
            <span className="req-label">Prefix Hit Ratio</span>
            <input type="number" className="req-input" value={p.prefix_hit_ratio ?? ''} min={0} max={1} step={0.05}
              onChange={(e) => set({ prefix_hit_ratio: +e.target.value })} />
          </div>
          )}
        </div>
        <div className="req-row req-mtp-row">
          <div className={cn('req-card', !hasMtp && 'req-card-readonly')}>
            <span className="req-label">MTP Tokens</span>
            {hasMtp ? (
              <input type="number" className="req-input" value={p.num_mtp_tokens || ''} min={1}
                onChange={(e) => set({ num_mtp_tokens: +e.target.value })} />
            ) : (
              <>
                <span className="req-value">0</span>
                <span className="req-hint">在模型配置中添加 MTP 层后可修改该值</span>
              </>
            )}
          </div>
          <div className="req-card">
            <span className="req-label">MTP Ratio</span>
            <input type="number" className="req-input" value={p.ratio_mtp_tokens || ''} min={0} max={1} step={0.05}
              onChange={(e) => set({ ratio_mtp_tokens: +e.target.value })} />
          </div>
          <div className="req-card req-card-computed">
            <span className="req-label">平均接受 tokens</span>
            <span className="req-result">{p.phase === 'prefill'
  ? '1'
  : [...Array(p.num_mtp_tokens + 1)].reduce((sum, _, i) => sum + Math.pow(p.ratio_mtp_tokens, i), 0).toFixed(2)}</span>
          </div>
        </div>
      </div>

      {/* ── Parallel Strategy ── */}
      <div className="section">
        <div className="section-title">
          <span>并行策略</span>
          <span className="section-badge">World Size: {p.world_size}</span>
        </div>

        {p.optimizeMode === 'manual' ? (<>
          <div className="parallel-pipeline">
          <div className="parallel-chip">
            <span className="parallel-abbr">TP</span>
            <input type="number" className="parallel-input" value={p.tp_size || ''} min={1}
              onChange={(e) => set({ tp_size: +e.target.value })} />
            <span className="parallel-name">张量并行</span>
          </div>

          <span className="parallel-connector multiply">×</span>

          <div className="parallel-chip">
            <span className="parallel-abbr">DP</span>
            <input type="number" className="parallel-input" value={p.dp_size || ''} min={1}
              onChange={(e) => set({ dp_size: +e.target.value })} />
            <span className="parallel-name">数据并行</span>
          </div>

          <span className="parallel-connector multiply">×</span>

          <div className="parallel-chip locked">
            <span className="parallel-abbr">PP</span>
            <input type="number" className="parallel-input" value={1} disabled />
            <span className="parallel-lock">内置</span>
          </div>

          <span className="parallel-connector equals">=</span>

          <div className="parallel-chip auto-derived">
            <span className="parallel-abbr">EP</span>
            <input type="number" className="parallel-input" value={p.world_size} disabled />
            <span className="parallel-lock">= World Size</span>
          </div>

          <span className="parallel-connector equals">=</span>

          <span className="parallel-result match">{p.world_size}</span>

          <div className="parallel-chip cp-chip locked">
            <span className="parallel-abbr">CP</span>
            <input type="number" className="parallel-input" value={1} disabled />
            <span className="parallel-lock">内置</span>
          </div>
        </div>

        <div className="parallel-tp-extra">
          <div className="parallel-chip">
            <span className="parallel-abbr">Embed</span>
            <input type="number" className="parallel-input" value={p.embed_tp_size || ''} min={1}
              onChange={(e) => set({ embed_tp_size: +e.target.value })} />
            <span className="parallel-name">嵌入层</span>
          </div>
          <div className="parallel-chip">
            <span className="parallel-abbr">O</span>
            <input type="number" className="parallel-input" value={p.o_tp_size || ''} min={1}
              onChange={(e) => set({ o_tp_size: +e.target.value })} />
            <span className="parallel-name">输出投影</span>
          </div>
          <div className="parallel-chip">
            <span className="parallel-abbr">LMHead</span>
            <input type="number" className="parallel-input" value={p.lmhead_tp_size || ''} min={1}
              onChange={(e) => set({ lmhead_tp_size: +e.target.value })} />
            <span className="parallel-name">LM Head</span>
          </div>
          <div className="parallel-chip chip-extse">
            <span className="parallel-abbr">ExtSE</span>
            <input type="number" className="parallel-input" value={p.external_shared_expert_rank_size || ''} min={0}
              onChange={(e) => set({ external_shared_expert_rank_size: +e.target.value })} />
            <span className="parallel-name">外置共享专家</span>
          </div>
        </div>
        </>) : (
          <div className="opt-auto-section">
            <div className="opt-auto-row">
              <div className="req-card">
                <span className="req-label">目标 TPOT</span>
                <div className="hw-spec-input-wrap">
                  <input type="number" className="req-input" value={p.targetTpotMs || ''} min={1}
                    onChange={(e) => set({ targetTpotMs: +e.target.value } as any)} />
                  <span className="req-unit">ms</span>
                </div>
              </div>
              <div className="req-card">
                <span className="req-label">最小卡数</span>
                <input type="number" className="req-input" value={p.minWorldSize || ''} min={1}
                  onChange={(e) => set({ minWorldSize: +e.target.value } as any)}
                  onBlur={(e) => set({ minWorldSize: toPowerOfTwo(+e.target.value, 'up') } as any)} />
              </div>
              <div className="req-card">
                <span className="req-label">最大卡数</span>
                <input type="number" className="req-input" value={p.maxWorldSize || ''} min={1}
                  onChange={(e) => set({ maxWorldSize: +e.target.value } as any)}
                  onBlur={(e) => set({ maxWorldSize: toPowerOfTwo(+e.target.value, 'up') } as any)} />
              </div>
            </div>
            <div className="opt-auto-row">
              <div className="req-card">
                <span className="req-label">Embed TP 范围</span>
                <div className="hw-spec-input-wrap">
                  <input type="number" className="req-input" value={p.embedTpMin || ''} min={1}
                    onChange={(e) => set({ embedTpMin: +e.target.value } as any)} />
                  <span className="req-sep">—</span>
                  <input type="number" className="req-input" value={p.embedTpMax || ''} min={0}
                    onChange={(e) => set({ embedTpMax: +e.target.value } as any)} />
                </div>
              </div>
              <div className="req-card">
                <span className="req-label">O TP 范围</span>
                <div className="hw-spec-input-wrap">
                  <input type="number" className="req-input" value={p.oTpMin || ''} min={1}
                    onChange={(e) => set({ oTpMin: +e.target.value } as any)} />
                  <span className="req-sep">—</span>
                  <input type="number" className="req-input" value={p.oTpMax || ''} min={0}
                    onChange={(e) => set({ oTpMax: +e.target.value } as any)} />
                </div>
              </div>
              <div className="req-card">
                <span className="req-label">LMHead TP 范围</span>
                <div className="hw-spec-input-wrap">
                  <input type="number" className="req-input" value={p.lmheadTpMin || ''} min={1}
                    onChange={(e) => set({ lmheadTpMin: +e.target.value } as any)} />
                  <span className="req-sep">—</span>
                  <input type="number" className="req-input" value={p.lmheadTpMax || ''} min={0}
                    onChange={(e) => set({ lmheadTpMax: +e.target.value } as any)} />
                </div>
              </div>
            </div>
            <div className="opt-auto-hint">
              自动搜索满足时延目标的最优并行策略（最少 GPU），硬件配置在步骤 3 中设置<br/>
              <span className="opt-auto-hint-sub">TP 范围：min 起始值，max 上限（0 = 自动跟随 World Size）</span>
            </div>
          </div>
        )}
      </div>

      {/* ── Quantization ── */}
      <div className="section">
        <div className="section-title">量化配置</div>
        <div className="quant-groups">
          {QUANT_GROUPS.map((g) => (
            <div key={g.label} className="quant-group">
              <div className="quant-group-label">{g.label}</div>
              <div className={`quant-grid${(g as any).oneRow ? ' quant-grid-row' : ''}`}>
                {g.fields.map((f) => (
                  <div key={f.key} className="quant-item">
                    <label className="quant-label">{f.label}</label>
                    <select className={`quant-select${f.key === 'quant_global' ? ' quant-global' : ''}`}
                      value={p[f.key]}
                      onChange={(e) => {
                        const v = e.target.value
                        if (f.key === 'quant_global') {
                          set({ quant_global: v, quant_mlp: v, quant_shared_expert: v, quant_routed_expert: v } as any)
                        } else {
                          set({ [f.key]: v } as any)
                        }
                      }}>
                      {(f as any).options.map((o: string) => (<option key={o} value={o}>{o.toUpperCase()}</option>))}
                    </select>
                  </div>
                ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
