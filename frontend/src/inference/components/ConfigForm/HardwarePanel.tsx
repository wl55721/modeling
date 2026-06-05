import { useEffect, useState } from 'react'
import * as api from '../../api/library'
import { useHardwareStore, BASE_RATIO_FIELDS, type RatioKey, type HwFields } from '../../stores/hardware'
import { cn } from '../../utils/classnames'

export default function HardwarePanel() {
  const store = useHardwareStore()
  const [list, setList] = useState<api.HardwareListItem[]>([])
  const [vendor, setVendor] = useState('HUAWEI')

  useEffect(() => { api.fetchHardwareList().then(setList) }, [])

  useEffect(() => {
    const cfg = store.configs[0]
    if (cfg && cfg.builtinName && cfg.fields.chip_name === '') {
      store.loadBuiltinFields(cfg.id, cfg.builtinName)
    }
  }, [])

  const active = store.configs.find((c) => c.id === store.activeConfigId)
  const fields = active?.fields

  const vendors = [...new Set(list.map((h) => h.vendor).filter(Boolean))].sort()
  const filtered = vendor ? list.filter((h) => h.vendor === vendor) : list

  // Auto-sync vendor and builtin fields when switching active config
  const activeBuiltinName = active?.builtinName
  useEffect(() => {
    if (!activeBuiltinName || list.length === 0) return
    const hw = list.find((h) => h.name === activeBuiltinName)
    if (hw && hw.vendor) {
      setVendor(hw.vendor)
    }
    // Load builtin fields if this config hasn't loaded them yet
    if (active && active.builtinName && active.fields.chip_name === '') {
      store.loadBuiltinFields(active.id, active.builtinName)
    }
  }, [active?.id, activeBuiltinName, list])
  const hasEnabled = store.configs.some((c) => c.enabled)

  const num = (v: string, k: keyof HwFields) => {
    if (!active) return
    store.updateFields(active.id, { [k]: parseFloat(v) || 0 })
  }

  const isAscendChip = !!(active?.builtinName?.match(/^A\d/) || fields?.cube_core_cnt)
  const showAscendCompute = isAscendChip
  const showNvidiaCompute = !isAscendChip
  const hasBwsio = !!(active?.builtinName?.match(/^A\d/) || fields?.spec_comm_bwsio)

  const ratioFields = [
    ...BASE_RATIO_FIELDS,
    ...(hasBwsio ? [{ key: 'comm_bwsio_ratio' as RatioKey, label: 'BWSIO通信效率' }] : []),
    { key: 'l2_cache_ratio' as RatioKey, label: 'L2缓存效率' },
  ]

  function handleBuiltinSelect(name: string) {
    if (!active || !name) return
    store.setConfigBuiltin(active.id, name)
    store.updateConfigName(active.id, name)
    store.loadBuiltinFields(active.id, name)
  }

  return (
    <div className="section">
      <div className="section-title">硬件规格配置</div>

      {/* ═══ Area A: Config list ═══ */}
      {!hasEnabled && (
        <div className="hw-warning-banner">未选择任何硬件环境进行仿真</div>
      )}
      <div className="hw-config-list">
        {store.configs.map((cfg) => (
          <div
            key={cfg.id}
            className={cn('hw-config-card', cfg.id === store.activeConfigId && 'active', !cfg.enabled && 'disabled')}
          >
            <input
              type="checkbox"
              className="hw-config-checkbox"
              checked={cfg.enabled}
              onChange={() => store.toggleEnabled(cfg.id)}
            />
            <div className="hw-config-info" onClick={() => store.setActiveConfig(cfg.id)}>
              <span className="hw-config-name">{cfg.name}</span>
              <span className="hw-config-summary">
                {cfg.builtinName && <span className="hw-config-chip">{cfg.builtinName}</span>}
                {cfg.fields.chip_name && cfg.fields.chip_name !== cfg.builtinName && (
                  <span className="hw-config-chip">{cfg.fields.chip_name}</span>
                )}
                <span className="hw-config-spec">{cfg.fields.spec_memory_size}GB</span>
                <span className="hw-config-spec">{cfg.fields.spec_bw_memory}TB/s</span>
              </span>
            </div>
            <div className="hw-config-actions">
              <button className="hw-config-btn" title="编辑" onClick={() => store.setActiveConfig(cfg.id)}>✎</button>
              <button className="hw-config-btn" title="复制" onClick={() => store.duplicateConfig(cfg.id)}>⧉</button>
              <button className="hw-config-btn" title="移除" disabled={store.configs.length <= 1}
                onClick={() => store.removeConfig(cfg.id)}>×</button>
            </div>
          </div>
        ))}
      </div>
      <button className="hw-add-btn" onClick={() => store.addConfig()}>+ 添加硬件环境</button>

      {/* ═══ Area B: Active config editor ═══ */}
      {active && fields && (
        <div className="hw-editor-section">
          <div className="hw-editor-title">正在编辑：{active.name}</div>

          {/* chip selector + name — single row */}
          <div className="hw-chip-select">
            <div className="hw-chip-select-col">
              <label className="hw-chip-label">配置名称</label>
              <input value={active.name}
                onChange={(e) => store.updateConfigName(active.id, e.target.value)}
                placeholder="输入配置名称" />
            </div>
            <div className="hw-chip-select-col">
              <label className="hw-chip-label">硬件厂商</label>
              <select value={vendor} onChange={(e) => setVendor(e.target.value)}>
                <option value="">全部厂商</option>
                {vendors.map((v) => (<option key={v} value={v}>{v}</option>))}
              </select>
            </div>
            <div className="hw-chip-select-col">
              <label className="hw-chip-label">选择内置硬件配置</label>
              <select value={active.builtinName} onChange={(e) => handleBuiltinSelect(e.target.value)}>
                <option value="">-- 不选择 --</option>
                {filtered.map((h) => (<option key={h.name} value={h.name}>{h.name}</option>))}
              </select>
            </div>
            <div className="hw-chip-select-col">
              <label className="hw-chip-label">芯片名称</label>
              <input value={fields.chip_name}
                onChange={(e) => store.updateFields(active.id, { chip_name: e.target.value })}
                placeholder="输入芯片名称" />
            </div>
          </div>

          {/* spec fields */}
          <div className="hw-specs">

            {/* ── Basic specs ── */}
            <div className="hw-spec-group">
              <div className="hw-spec-group-title">基本规格</div>
              <div className="hw-spec-row three-col">
                <div className="hw-spec-card">
                  <span className="hw-spec-label">显存</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.spec_memory_size || ''}
                      onChange={(e) => num(e.target.value, 'spec_memory_size')} />
                    <span className="hw-spec-unit">GB</span>
                  </div>
                </div>
                <div className="hw-spec-card">
                  <span className="hw-spec-label">显存带宽</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.spec_bw_memory || ''}
                      onChange={(e) => num(e.target.value, 'spec_bw_memory')} step="0.1" />
                    <span className="hw-spec-unit">TB/s</span>
                  </div>
                </div>
                <div className="hw-spec-card">
                  <span className="hw-spec-label">L2缓存</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.spec_l2cache_size || ''}
                      onChange={(e) => num(e.target.value, 'spec_l2cache_size')} />
                    <span className="hw-spec-unit">MB</span>
                  </div>
                </div>
              </div>

              {showAscendCompute && (
                <>
                  <div className="hw-spec-row three-col">
                    <div className="hw-spec-card">
                      <span className="hw-spec-label">Cube核心数</span>
                      <div className="hw-spec-input-wrap">
                        <input type="number" value={fields.cube_core_cnt || ''}
                          onChange={(e) => num(e.target.value, 'cube_core_cnt')} />
                      </div>
                    </div>
                    <div className="hw-spec-card">
                      <span className="hw-spec-label">Vector核心数</span>
                      <div className="hw-spec-input-wrap">
                        <input type="number" value={fields.vector_core_cnt || ''}
                          onChange={(e) => num(e.target.value, 'vector_core_cnt')} />
                      </div>
                    </div>
                    <div className="hw-spec-card">
                      <span className="hw-spec-label">SFU核心数</span>
                      <div className="hw-spec-input-wrap">
                        <input type="number" value={fields.sfu_core_cnt || ''}
                          onChange={(e) => num(e.target.value, 'sfu_core_cnt')} />
                      </div>
                    </div>
                  </div>
                  <div className="hw-spec-row">
                    <div className="hw-spec-card">
                      <span className="hw-spec-label">Cube频率</span>
                      <div className="hw-spec-input-wrap">
                        <input type="number" value={fields.cube_freq || ''}
                          onChange={(e) => num(e.target.value, 'cube_freq')} step="0.1" />
                        <span className="hw-spec-unit">GHz</span>
                      </div>
                    </div>
                  </div>
                </>
              )}

              {showNvidiaCompute && (
                <div className="hw-spec-row three-col">
                  <div className="hw-spec-card">
                    <span className="hw-spec-label">FP16算力 (Cube)</span>
                    <div className="hw-spec-input-wrap">
                      <input type="number" value={fields.spec_cube_fp16 || ''}
                        onChange={(e) => num(e.target.value, 'spec_cube_fp16')} />
                      <span className="hw-spec-unit">TFLOPS</span>
                    </div>
                  </div>
                  <div className="hw-spec-card">
                    <span className="hw-spec-label">FP16算力 (Vector)</span>
                    <div className="hw-spec-input-wrap">
                      <input type="number" value={fields.spec_vect_fp16 || ''}
                        onChange={(e) => num(e.target.value, 'spec_vect_fp16')} />
                      <span className="hw-spec-unit">TFLOPS</span>
                    </div>
                  </div>
                  <div className="hw-spec-card">
                    <span className="hw-spec-label">SFU算力</span>
                    <div className="hw-spec-input-wrap">
                      <input type="number" value={fields.spec_sfu_fp16 || ''}
                        onChange={(e) => num(e.target.value, 'spec_sfu_fp16')} />
                      <span className="hw-spec-unit">TFLOPS</span>
                    </div>
                  </div>
                </div>
              )}
              <div className="hw-spec-row three-col">
                <div className="hw-spec-card">
                  <span className="hw-spec-label">机内通信（单向）</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.spec_comm_intra || ''}
                      onChange={(e) => num(e.target.value, 'spec_comm_intra')} />
                    <span className="hw-spec-unit">GB/s</span>
                  </div>
                </div>
                <div className="hw-spec-card">
                  <span className="hw-spec-label">机间通信（单向）</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.spec_comm_inter || ''}
                      onChange={(e) => num(e.target.value, 'spec_comm_inter')} />
                    <span className="hw-spec-unit">GB/s</span>
                  </div>
                </div>
                {hasBwsio && (
                  <div className="hw-spec-card">
                    <span className="hw-spec-label">BWSIO通信</span>
                    <div className="hw-spec-input-wrap">
                      <input type="number" value={fields.spec_comm_bwsio || ''}
                        onChange={(e) => num(e.target.value, 'spec_comm_bwsio')} />
                      <span className="hw-spec-unit">GB/s</span>
                    </div>
                  </div>
                )}
              </div>

              {/* memory noise — inside基本规格 */}
              <div className="hw-spec-row">
                <div className="hw-spec-card">
                  <span className="hw-spec-label">常驻噪声显存</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.memory_noise || ''}
                      onChange={(e) => num(e.target.value, 'memory_noise')} step="0.5" />
                    <span className="hw-spec-unit">GB</span>
                  </div>
                </div>
              </div>
            </div>

            {/* ── Efficiency ratios ── */}
            <div className="hw-spec-group">
              <div className="hw-spec-group-title">效率系数</div>
              <div className="hw-ratio-grid">
                {ratioFields.map(({ key, label }) => (
                  <div key={key} className="hw-ratio-item">
                    <span className="hw-ratio-label">{label}</span>
                    <span className="hw-ratio-pct">{Math.round(((fields[key] as number) || 0) * 100)}%</span>
                    <input type="number" className="hw-ratio-input"
                      value={(fields[key] as number) || ''}
                      onChange={(e) => num(e.target.value, key as keyof HwFields)}
                      step="0.05" min="0" max="1" />
                  </div>
                ))}
              </div>
            </div>

            {/* ── Limits ── */}
            <div className="hw-spec-group">
              <div className="hw-spec-group-title">集群限制</div>
              <div className="hw-spec-row">
                <div className="hw-spec-card">
                  <span className="hw-spec-label">单POD或Server内的卡数</span>
                  <div className="hw-spec-input-wrap">
                    <input type="number" value={fields.superpod_limit || ''}
                      onChange={(e) => num(e.target.value, 'superpod_limit')} />
                  </div>
                </div>
                {hasBwsio && (
                  <div className="hw-spec-card">
                    <span className="hw-spec-label">BWSIO上限（单卡内DIE数量）</span>
                    <div className="hw-spec-input-wrap">
                      <input type="number" value={fields.bwsio_limit || ''}
                        onChange={(e) => num(e.target.value, 'bwsio_limit')} />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
