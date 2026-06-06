import { useState, useCallback, useRef } from 'react'
import html2canvas from 'html2canvas'
import jsPDF from 'jspdf'
import { useInferenceStore } from '../ConfigForm/WorkloadConfigPanel'
import { useModelStore } from '../../stores/model'
import { useHardwareStore } from '../../stores/hardware'
import type { SimulateSingleResult } from '../../types/results'

const PDF_STYLE = `
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; color: #1a1a1a; line-height: 1.5; background: #fff; }
  h2 { font-size: 15px; font-weight: 700; color: #1a1a1a; margin-bottom: 10px; padding-bottom: 4px; border-bottom: 2px solid #3b6fb6; }
  h3 { font-size: 12px; font-weight: 600; color: #333; margin: 10px 0 4px; padding-bottom: 3px; border-bottom: 1px solid #ddd; }
  h4 { font-size: 10px; font-weight: 600; color: #555; margin: 6px 0 3px; }
  table { width: 100%; border-collapse: collapse; font-size: 9px; margin-bottom: 6px; }
  th, td { padding: 2px 5px; border: 1px solid #ddd; text-align: left; }
  th { background: #f0f0f0; font-weight: 600; font-size: 8px; }
  .kv { font-weight: 500; color: #555; background: #fafafa; }
  .sec-label { font-weight: 600; color: #333; background: #f5f5f5; font-size: 8px; text-transform: uppercase; }
  .results-table td { text-align: center; }
  .results-table td:first-child { text-align: left; font-weight: 500; }
  .pdf-section { margin-bottom: 14px; }
  img.canvas-img { width: 100%; border: 1px solid #eee; border-radius: 4px; }
`

function fm(gb: number) { return (gb ?? 0).toFixed(2) + ' GB' }
function fl(ms: number) { return (ms ?? 0) < 1000 ? (ms ?? 0).toFixed(2) + ' ms' : ((ms ?? 0) / 1000).toFixed(2) + ' s' }
function fc(us: number) { return (us ?? 0) < 1000 ? (us ?? 0).toFixed(1) + ' us' : ((us ?? 0) / 1000).toFixed(2) + ' ms' }
function fb(bytes: number) { const b = bytes ?? 0; return b >= 1024 * 1024.0 * 1024.0 ? (b / 1024.0 / 1024.0 / 1024.0).toFixed(2) + ' GB' : (b / 1024.0 / 1024.0).toFixed(1) + ' MB' }

const DONUT_COLORS = ['#4f5eb1', '#3b6fb6', '#4d8c57', '#c4504a', '#e67e22', '#8e6bb8', '#2e86c1', '#b866cc', '#9e9e9e', '#7a9fcf', '#d4842a', '#5b6abf']

function DonutSvg({ stats, totalUs }: { stats: { op_name: string; total_cost_us: number }[]; totalUs: number }) {
  const size = 180; const cx = size / 2; const cy = size / 2
  const radius = 62; const stroke = 18
  const circumference = 2 * Math.PI * radius
  const gapAngle = 0.02

  let accumulated = 0
  const slices = stats.map((s, i) => {
    const pct = s.total_cost_us / totalUs
    const gapLen = gapAngle * radius
    const arcLen = pct * circumference - gapLen
    const dashOffset = -(accumulated / totalUs * circumference + gapLen / 2)
    accumulated += s.total_cost_us
    return { name: s.op_name, pct, dashOffset, arcLen: Math.max(arcLen, 0.5), color: DONUT_COLORS[i % DONUT_COLORS.length] }
  })

  return (
    <div style={{ textAlign: 'center', marginBottom: 6 }}>
      <svg viewBox={`0 0 ${size} ${size}`} width={size} height={size}>
        <circle cx={cx} cy={cy} r={radius} fill="none" stroke="#e8e8e8" strokeWidth={stroke} />
        {slices.map((s) => (
          <circle key={s.name} cx={cx} cy={cy} r={radius} fill="none"
            stroke={s.color} strokeWidth={stroke}
            strokeDasharray={`${s.arcLen} ${circumference - s.arcLen}`}
            strokeDashoffset={s.dashOffset}
            strokeLinecap="round"
            transform={`rotate(-90 ${cx} ${cy})`}
          >
            <title>{s.name}: {(s.pct * 100).toFixed(1)}%</title>
          </circle>
        ))}
        <text x={cx} y={cy - 6} textAnchor="middle" fontSize={11} fill="#888">总计</text>
        <text x={cx} y={cy + 10} textAnchor="middle" fontSize={12} fontWeight={600} fill="#333">{fc(totalUs)}</text>
      </svg>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '2px 10px', justifyContent: 'center', marginTop: 6, fontSize: 9 }}>
        {slices.map(s => (
          <span key={s.name} style={{ display: 'inline-flex', alignItems: 'center', gap: 3 }}>
            <span style={{ width: 8, height: 8, borderRadius: 2, background: s.color, display: 'inline-block' }} />
            {s.name} {(s.pct * 100).toFixed(1)}%
          </span>
        ))}
      </div>
    </div>
  )
}

export default function PdfExport({ results }: { results: SimulateSingleResult[] | null }) {
  const [busy, setBusy] = useState(false)
  const exportRef = useRef<HTMLDivElement>(null)
  const canvasImgRef = useRef<HTMLImageElement>(null)

  async function addImageToPdf(pdf: jsPDF, canvas: HTMLCanvasElement, y: number, contentW: number, pageH: number, margin: number): Promise<number> {
    const imgH = (canvas.height / canvas.width) * contentW
    const pageContentH = pageH - margin * 2

    // if section fits on current page, add and return new y
    if (y + imgH <= pageH - margin) {
      pdf.addImage(canvas.toDataURL('image/png'), 'PNG', margin, y, contentW, imgH)
      return y + imgH + 4
    }

    // section doesn't fit — start new page
    pdf.addPage()
    let newY = margin

    // if still too tall for a single page, split
    if (imgH > pageContentH) {
      const ratio = canvas.width / contentW
      let srcY = 0
      while (srcY < canvas.height) {
        const sliceH = Math.min(canvas.height - srcY, pageContentH * ratio)
        const pc = document.createElement('canvas')
        pc.width = canvas.width; pc.height = sliceH
        pc.getContext('2d')!.drawImage(canvas, 0, srcY, canvas.width, sliceH, 0, 0, canvas.width, sliceH)
        pdf.addImage(pc.toDataURL('image/png'), 'PNG', margin, newY, contentW, (sliceH / canvas.width) * contentW)
        srcY += sliceH
        if (srcY < canvas.height) { pdf.addPage(); newY = margin }
        else newY = margin + (sliceH / canvas.width) * contentW
      }
      return newY
    }

    pdf.addImage(canvas.toDataURL('image/png'), 'PNG', margin, margin, contentW, imgH)
    return margin + imgH + 4
  }

  const handleExport = useCallback(async () => {
    if (busy || !results) return
    setBusy(true)
    try {
      const pdf = new jsPDF('p', 'mm', 'a4')
      const pageW = pdf.internal.pageSize.getWidth()
      const pageH = pdf.internal.pageSize.getHeight()
      const margin = 8
      const contentW = pageW - margin * 2
      let y = margin

      // capture model canvas — use html2canvas with onclone to fix CSS vars
      const rfViewport = document.querySelector('.react-flow__viewport') as HTMLElement | null
      if (rfViewport && canvasImgRef.current) {
        try {
          const rfContainer = rfViewport.closest('.react-flow') as HTMLElement
          if (rfContainer) {
            const c = await html2canvas(rfContainer, {
              scale: 2, useCORS: true, backgroundColor: '#ffffff', logging: false,
              onclone: (clonedDoc) => {
                // Replace CSS custom properties on all elements in the clone
                clonedDoc.querySelectorAll('*').forEach((el: Element) => {
                  const htmlEl = el as HTMLElement
                  const st = htmlEl.style
                  if (!st) return
                  // fix common CSS variable patterns
                  for (let i = st.length - 1; i >= 0; i--) {
                    const prop = st[i]
                    const val = st.getPropertyValue(prop)
                    if (val && val.includes('var(')) {
                      // resolve var(--xy-edge-stroke) etc with defaults
                      st.setProperty(prop, val.replace(/var\([^)]+\)/g, '#cccccc'), 'important')
                    }
                  }
                })
              },
            })
            canvasImgRef.current.src = c.toDataURL('image/png')
          }
        } catch { /* skip if capture fails */ }
      }
      await new Promise(r => setTimeout(r, 300))

      // capture each section individually to avoid splitting headers
      if (exportRef.current) {
        const sections = exportRef.current.querySelectorAll('.pdf-section')
        for (const sec of sections) {
          const c = await html2canvas(sec as HTMLElement, {
            scale: 2, useCORS: true, backgroundColor: '#ffffff', logging: false,
          })
          y = await addImageToPdf(pdf, c, y, contentW, pageH, margin)
        }
      }

      pdf.save(`kepler-simulation-${new Date().toISOString().slice(0, 10)}.pdf`)
    } catch (err) {
      console.error('PDF export failed:', err)
    } finally {
      setBusy(false)
    }
  }, [busy, results])

  if (!results) return null

  const p = useInferenceStore.getState()
  const m = useModelStore.getState()
  const hwConfigs = useHardwareStore.getState().configs.filter(c => c.enabled)
  const names = results.map(r => r.hardware_name)

  return (
    <>
      {/* hidden container with all export content */}
      <div className="pdf-export-hidden" ref={exportRef}>
        <div style={{ background: '#fff', padding: 20, width: 740 }}>
          <style>{PDF_STYLE}</style>

          {/* 1. Model canvas */}
          <div className="pdf-section">
            <h2>模型结构图</h2>
            <img ref={canvasImgRef} className="canvas-img" alt="model canvas" />
          </div>

          {/* 2. Workload config */}
          <div className="pdf-section">
            <h2>负载配置</h2>
            <table>
              <tbody>
                <tr><td className="kv">模型</td><td>{m.modelName || '—'}</td></tr>
                <tr><td className="kv">Phase</td><td>{p.phase}</td></tr>
                <tr><td className="kv">Batch Size</td><td>{p.batch_size}</td></tr>
                <tr><td className="kv">Input Length</td><td>{p.input_length}</td></tr>
                <tr><td className="kv">Output Length</td><td>{p.output_length}</td></tr>
                <tr><td className="kv">Prefix Hit Ratio</td><td>{p.prefix_hit_ratio}</td></tr>
                <tr><td className="kv">MTP Tokens</td><td>{p.num_mtp_tokens}</td></tr>
                <tr><td className="kv">MTP Ratio</td><td>{p.ratio_mtp_tokens}</td></tr>
                <tr><td className="kv">World Size</td><td>{p.world_size}</td></tr>
                <tr><td className="kv">TP / DP / PP</td><td>{p.tp_size} / {p.dp_size} / {p.pp_size}</td></tr>
                <tr><td className="kv">EP / CP</td><td>{p.ep_size} / {p.cp_size}</td></tr>
                <tr><td className="kv">Embed TP</td><td>{p.embed_tp_size}</td></tr>
                <tr><td className="kv">O TP</td><td>{p.o_tp_size}</td></tr>
                <tr><td className="kv">LMHead TP</td><td>{p.lmhead_tp_size}</td></tr>
                <tr><td className="kv">ExtSE Rank</td><td>{p.external_shared_expert_rank_size}</td></tr>
              </tbody>
            </table>
            <h3>量化配置</h3>
            <table>
              <tbody>
                <tr><td className="kv">全局</td><td>{p.quant_global?.toUpperCase()}</td></tr>
                <tr><td className="kv">MLP</td><td>{p.quant_mlp?.toUpperCase()}</td></tr>
                <tr><td className="kv">共享专家</td><td>{p.quant_shared_expert?.toUpperCase()}</td></tr>
                <tr><td className="kv">路由专家</td><td>{p.quant_routed_expert?.toUpperCase()}</td></tr>
                <tr><td className="kv">KV Cache</td><td>{p.quant_kv_cache?.toUpperCase()}</td></tr>
                <tr><td className="kv">激活值</td><td>{p.quant_activation?.toUpperCase()}</td></tr>
              </tbody>
            </table>
          </div>

          {/* 3. Hardware config */}
          {hwConfigs.length > 0 && (
            <div className="pdf-section">
              <h2>硬件配置</h2>
              {hwConfigs.map(cfg => (
                <div key={cfg.id} style={{ marginBottom: 6 }}>
                  <h4>{cfg.name}</h4>
                  <table>
                    <tbody>
                      <tr><td className="kv">芯片</td><td>{cfg.fields.chip_name || cfg.builtinName || '—'}</td></tr>
                      <tr><td className="kv">显存</td><td>{cfg.fields.spec_memory_size} GB</td></tr>
                      <tr><td className="kv">显存带宽</td><td>{cfg.fields.spec_bw_memory} TB/s</td></tr>
                      <tr><td className="kv">L2 缓存</td><td>{cfg.fields.spec_l2cache_size} MB</td></tr>
                      <tr><td className="kv">机内/机间通信</td><td>{cfg.fields.spec_comm_intra} / {cfg.fields.spec_comm_inter} GB/s</td></tr>
                      <tr><td className="kv">计算/显存/通信效率</td><td>{Math.round((cfg.fields.compute_ratio || 0) * 100)}% / {Math.round((cfg.fields.bw_gmem_ratio || 0) * 100)}% / {Math.round((cfg.fields.comm_intra_ratio || 0) * 100)}% / {Math.round((cfg.fields.comm_inter_ratio || 0) * 100)}%</td></tr>
                      <tr><td className="kv">常驻噪声</td><td>{cfg.fields.memory_noise} GB</td></tr>
                    </tbody>
                  </table>
                </div>
              ))}
            </div>
          )}

          {/* 4. Simulation results — model comparison */}
          <div className="pdf-section">
            <h2>仿真结果 · 模型统计</h2>
            <table className="results-table">
              <thead>
                <tr>
                  <th>指标</th>
                  {names.map((n, i) => <th key={i}>{n}</th>)}
                </tr>
              </thead>
              <tbody>
                <tr><td className="sec-label" colSpan={names.length + 1}>时延指标</td></tr>
                <tr><td>TPOT</td>{results.map((r, i) => <td key={i}>{fl(r.result.tpot_ms)}</td>)}</tr>
                <tr><td>Prefill</td>{results.map((r, i) => <td key={i}>{fl(r.result.prefill_latency_ms)}</td>)}</tr>
                <tr><td>Decode/Token</td>{results.map((r, i) => <td key={i}>{fl(r.result.decode_latency_per_token_ms)}</td>)}</tr>
                <tr><td className="sec-label" colSpan={names.length + 1}>吞吐指标</td></tr>
                <tr><td>TPS</td>{results.map((r, i) => <td key={i}>{r.result.tps.toFixed(2)}</td>)}</tr>
                <tr><td>QPS</td>{results.map((r, i) => <td key={i}>{r.result.qps.toFixed(2)}</td>)}</tr>
                <tr><td className="sec-label" colSpan={names.length + 1}>显存总览</td></tr>
                <tr><td>总显存占用</td>{results.map((r, i) => <td key={i}>{fm(r.result.peak_mem_gb)}</td>)}</tr>
                <tr><td>OOM</td>{results.map((r, i) => <td key={i}>{r.result.oom ? 'OOM!' : 'OK'}</td>)}</tr>
                <tr><td>策略</td>{results.map((r, i) => <td key={i} style={{ fontSize: 8 }}>{r.result.strategy}</td>)}</tr>
              </tbody>
            </table>
          </div>

          {/* 5. Per-hardware Rank statistics */}
          {results.map((r) => {
            const ranks = r.result.ranks
            if (!ranks || ranks.length === 0) return null
            const maxCost = Math.max(...ranks.map(rs => rs.total_cost_ms))
            const maxMem = Math.max(...ranks.map(rs => rs.peak_mem_gb))
            const oomCount = ranks.filter(rs => rs.oom).length
            return (
              <div className="pdf-section" key={r.hardware_name}>
                <h2>Rank 统计 · {r.hardware_name}</h2>
                <table>
                  <tbody>
                    <tr><td className="kv">总 Ranks</td><td>{ranks.length}</td>
                    <td className="kv">最慢耗时</td><td>{fl(maxCost)}</td>
                    <td className="kv">最大显存</td><td>{fm(maxMem)}</td>
                    <td className="kv">OOM</td><td>{oomCount > 0 ? `${oomCount}/${ranks.length}` : '全部 OK'}</td></tr>
                  </tbody>
                </table>
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th><th>耗时</th><th>峰值显存</th>
                      <th>权重</th><th>激活</th><th>噪声</th><th>容量</th>
                      <th>算子数</th><th>状态</th>
                    </tr>
                  </thead>
                  <tbody>
                    {ranks.map(rs => (
                      <tr key={rs.rank_idx} style={rs.oom ? { background: '#fff0f0' } : undefined}>
                        <td>Rank {rs.rank_idx}</td>
                        <td>{fl(rs.total_cost_ms)}</td>
                        <td>{fm(rs.peak_mem_gb)}</td>
                        <td>{fb(rs.param_bytes)}</td>
                        <td>{fb(rs.io_bytes)}</td>
                        <td>{fm(rs.noise_gb)}</td>
                        <td>{fm(rs.mem_capacity_gb)}</td>
                        <td>{rs.num_ops}</td>
                        <td>{rs.oom ? 'OOM!' : 'OK'}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )
          })}

          {/* 6. Per-hardware Operator statistics */}
          {results.map((r) => {
            const stats = [...r.result.op_statistics]
              .filter(s => s.op_name !== 'start' && s.op_name !== 'end')
              .sort((a, b) => b.total_cost_us - a.total_cost_us)
            const totalUs = stats.reduce((sum, s) => sum + s.total_cost_us, 0)
            if (stats.length === 0) return null
            return (
              <div className="pdf-section" key={`op-${r.hardware_name}`}>
                <h2>算子统计 · {r.hardware_name}</h2>
                <DonutSvg stats={stats} totalUs={totalUs} />
                <table>
                  <thead>
                    <tr>
                      <th>#</th><th>算子</th><th>调用次数</th>
                      <th>总耗时</th><th>平均耗时</th><th>占比</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.map((s, i) => (
                      <tr key={s.op_name}>
                        <td>{i + 1}</td>
                        <td style={{ fontWeight: 500 }}>{s.op_name}</td>
                        <td>{s.num_calls}</td>
                        <td>{fc(s.total_cost_us)}</td>
                        <td>{fc(s.avg_cost_us)}</td>
                        <td>{(s.total_cost_us / totalUs * 100).toFixed(1)}%</td>
                      </tr>
                    ))}
                    <tr style={{ fontWeight: 600, background: '#f5f5f5' }}>
                      <td colSpan={3}>总计</td>
                      <td>{fc(totalUs)}</td>
                      <td colSpan={2}>{stats.length} 个算子</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )
          })}
        </div>
      </div>

      <button className="btn primary pdf-export-btn" onClick={handleExport} disabled={busy}>
        {busy ? '导出中...' : '导出 PDF'}
      </button>
    </>
  )
}
