import { useState, useRef } from 'react'
import { useInferenceStore } from '../ConfigForm/WorkloadConfigPanel'
import type { OptimizeResult, StrategyResult, HardwareOptimizeResult } from '../../api/library'

// ── helpers ──────────────────────────────────────────────

function fm(gb: number) { return gb.toFixed(1) + ' GB' }

function toPowerOfTwo(v: number): number {
  if (v < 1) return 1
  let p = 1
  while (p < v) p *= 2
  if (p > v && p > 1) {
    const lower = p / 2
    return (v - lower) < (p - v) ? lower : p
  }
  return Math.max(1, p)
}

// ── TPS vs world_size bar chart ─────────────────────────────

const COLORS = ['#4f5eb1', '#c4504a', '#4d8c57', '#e67e22', '#8e6bb8']
const COLOR_GRADS = [
  ['#5b6fd4', '#3d4d9e'],
  ['#e0605a', '#a8403a'],
  ['#5da86a', '#3d7048'],
  ['#f08c3e', '#c06820'],
  ['#a484cc', '#7254a0'],
]

function WsTpsChart({
  hwResults, hwIdx, setHwIdx,
}: {
  hwResults: HardwareOptimizeResult[]
  hwIdx: number
  setHwIdx: (i: number) => void
}) {
  const [tooltip, setTooltip] = useState<{ x: number; y: number; hw: string; c: StrategyResult; color: string } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  if (hwResults.length === 0) return null

  // gather per-hw per-ws best TPS
  const allData: { hw: string; ws: number; c: StrategyResult }[] = []
  for (const hw of hwResults) {
    const wsBest = new Map<number, StrategyResult>()
    for (const c of hw.candidates.filter(c => c.meets_target)) {
      const cur = wsBest.get(c.world_size)
      if (!cur || c.tps > cur.tps) wsBest.set(c.world_size, c)
    }
    for (const [ws, c] of wsBest) {
      allData.push({ hw: hw.hardware_name, ws, c })
    }
  }

  const isSingle = hwResults.length === 1
  const hwNames = hwResults.map(h => h.hardware_name)

  const allWs = [...new Set(allData.map(d => d.ws))].sort((a, b) => a - b)
  if (allWs.length < 1) return null
  const wsIndex = new Map(allWs.map((ws, i) => [ws, i]))

  const w = 520; const h = 220
  const pad = { top: 16, right: 16, bottom: 32, left: 48 }
  const iw = w - pad.left - pad.right; const ih = h - pad.top - pad.bottom

  const maxTps = Math.max(...allData.map(d => d.c.tps)) * 1.12 || 1

  // categorical X: equal spacing regardless of WS value
  const numWs = allWs.length
  const sidePad = iw * 0.10
  function xPos(idx: number) {
    if (numWs === 1) return pad.left + iw / 2
    return pad.left + sidePad + idx / (numWs - 1) * (iw - sidePad * 2)
  }

  // bar geometry
  const hwCount = hwNames.length
  const groupWidth = numWs > 1 ? iw / numWs : 72
  const totalBarWidth = Math.min(groupWidth * 0.32, numWs > 1 ? groupWidth * 0.32 : 18)
  const barGap = 3
  const barWidth = Math.max(8, (totalBarWidth - barGap * (hwCount - 1)) / hwCount)
  const barRadius = Math.min(3, barWidth / 3)

  // per-hw WS→data lookup
  const hwDataMap = new Map<string, Map<number, typeof allData[0]>>()
  for (const hwName of hwNames) {
    const m = new Map<number, typeof allData[0]>()
    for (const d of allData) {
      if (d.hw === hwName) m.set(d.ws, d)
    }
    hwDataMap.set(hwName, m)
  }

  function handleBarEnter(e: React.MouseEvent, d: typeof allData[0], color: string) {
    const rect = (e.target as Element).closest('svg')?.getBoundingClientRect()
    if (!rect) return
    const svgW = rect.width; const svgH = rect.height
    const px = rect.left + (xPos(wsIndex.get(d.ws)!) / w) * svgW
    const barH = d.c.tps / maxTps * ih
    const py = rect.top + (pad.top + ih - barH) / h * svgH
    setTooltip({ x: px, y: py, hw: d.hw, c: d.c, color })
  }

  // rounded-top bar path
  function barPath(x: number, y: number, bw: number, bh: number, r: number) {
    if (bh < r * 2) {
      // bar too short, use plain rect
      return `M${x},${y + bh} L${x},${y} L${x + bw},${y} L${x + bw},${y + bh} Z`
    }
    return `M${x},${y + bh} L${x},${y + r} Q${x},${y} ${x + r},${y} L${x + bw - r},${y} Q${x + bw},${y} ${x + bw},${y + r} L${x + bw},${y + bh} Z`
  }

  return (
    <div className="ws-tps-chart-wrap" ref={containerRef}>
      {!isSingle && (
        <div className="ws-chart-tabs">
          {hwNames.map((n, i) => (
            <button key={n} className={`ws-chart-tab${i === hwIdx ? ' active' : ''}`}
              style={{ '--tab-color': COLORS[i % COLORS.length] } as React.CSSProperties}
              onClick={() => setHwIdx(i)}>
              <span className="ws-chart-dot" style={{ background: COLORS[i % COLORS.length] }} />
              {n}
            </button>
          ))}
        </div>
      )}

      <svg viewBox={`0 0 ${w} ${h}`} className="ws-tps-chart" onMouseLeave={() => setTooltip(null)}>
        <defs>
          {COLOR_GRADS.map(([top, bot], i) => (
            <linearGradient key={i} id={`bar-grad-${i}`} x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor={top} />
              <stop offset="100%" stopColor={bot} />
            </linearGradient>
          ))}
        </defs>

        {/* Y-axis grid + baseline */}
        <line x1={pad.left} x2={w - pad.right} y1={pad.top + ih} y2={pad.top + ih}
          stroke="var(--border)" strokeWidth="1" />
        {[0.25, 0.5, 0.75, 1].map(frac => {
          const y = pad.top + ih * (1 - frac)
          return (
            <g key={frac}>
              <line x1={pad.left} x2={w - pad.right} y1={y} y2={y}
                stroke="var(--border)" strokeWidth="0.5" strokeDasharray="4,4" opacity="0.6" />
              <text x={pad.left - 8} y={y + 4} textAnchor="end" fontSize="10" fill="var(--text-muted)">
                {(maxTps * frac).toFixed(0)}
              </text>
            </g>
          )
        })}

        {/* X-axis labels with tick marks */}
        {allWs.map((ws, i) => {
          const cx = xPos(i)
          return (
            <g key={ws}>
              <line x1={cx} x2={cx} y1={pad.top + ih} y2={pad.top + ih + 4}
                stroke="var(--border)" strokeWidth="1" />
              <text x={cx} y={h - 10} textAnchor="middle" fontSize="10" fill="var(--text-muted)" fontWeight={500}>{ws}</text>
            </g>
          )
        })}
        <text x={w / 2} y={h - 2} textAnchor="middle" fontSize="8" fill="var(--text-muted)" opacity="0.6">World Size</text>

        {/* bars */}
        {hwNames.map((hwName, hi) => {
          const map = hwDataMap.get(hwName)
          if (!map) return null
          const active = isSingle || hwIdx === hi
          const opacity = isSingle ? 1 : active ? 1 : 0.3
          const color = COLORS[hi % COLORS.length]
          return allWs.map((ws, i) => {
            const d = map.get(ws)
            if (!d) return null
            const cx = xPos(i)
            const barX = cx - totalBarWidth / 2 + hi * (barWidth + barGap)
            const barH = Math.max(d.c.tps / maxTps * ih, 2)
            const barY = pad.top + ih - barH
            const fill = active ? `url(#bar-grad-${hi % COLOR_GRADS.length})` : color
            const labelY = barY - 4
            return (
              <g key={`${hwName}-${ws}`} className="ws-bar-group">
                <path d={barPath(barX, barY, barWidth, barH, barRadius)}
                  fill={fill} opacity={opacity}
                  style={{ cursor: 'pointer', transition: 'opacity 0.2s' }}
                  onMouseEnter={e => handleBarEnter(e, d, color)}
                />
                {active && barH > 12 && (
                  <text x={barX + barWidth / 2} y={labelY} textAnchor="middle"
                    fontSize="9" fontWeight={500} fill="var(--text-primary)" opacity={0.85}>
                    {d.c.tps.toFixed(0)}
                  </text>
                )}
              </g>
            )
          })
        })}
      </svg>

      <div className="ws-chart-label">TPS ↑ &nbsp;—&nbsp; World Size →</div>

      {/* tooltip */}
      {tooltip && (
        <div className="ws-chart-tooltip" style={{ left: tooltip.x + 12, top: tooltip.y - 10, borderColor: tooltip.color }}>
          <div className="ws-chart-tt-hw">{tooltip.hw}</div>
          <div className="ws-chart-tt-strategy">
            <span className="ws-chart-tt-dot" style={{ background: tooltip.color }} />
            {tooltip.c.strategy_label}
          </div>
          <div className="ws-chart-tt-grid">
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">WS</span><span className="ws-chart-tt-val">{tooltip.c.world_size}</span></div>
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">TP</span><span className="ws-chart-tt-val">{tooltip.c.tp_size}</span></div>
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">DP</span><span className="ws-chart-tt-val">{tooltip.c.dp_size}</span></div>
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">BS</span><span className="ws-chart-tt-val">{tooltip.c.batch_size}</span></div>
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">Emb</span><span className="ws-chart-tt-val">{tooltip.c.embed_tp_size}</span></div>
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">O</span><span className="ws-chart-tt-val">{tooltip.c.o_tp_size}</span></div>
            <div className="ws-chart-tt-cell"><span className="ws-chart-tt-label">LM</span><span className="ws-chart-tt-val">{tooltip.c.lmhead_tp_size}</span></div>
            <div className="ws-chart-tt-cell" />
          </div>
          <div className="ws-chart-tt-sep" />
          <div className="ws-chart-tt-row">
            <span className="ws-chart-tt-kv">TPOT <strong>{tooltip.c.tpot_ms.toFixed(1)}ms</strong></span>
            <span className="ws-chart-tt-kv">TPS <strong>{tooltip.c.tps.toFixed(1)}</strong></span>
            <span className="ws-chart-tt-kv">显存 <strong>{fm(tooltip.c.max_peak_mem_gb)}</strong></span>
          </div>
        </div>
      )}
    </div>
  )
}

// ── optimal config statistics ─────────────────────────────

function OptimalStatsTable({ hwResults, onApply, optimal }: { hwResults: HardwareOptimizeResult[]; onApply: (c: StrategyResult) => void; optimal: StrategyResult | null }) {
  const [applied, setApplied] = useState(false)
  const [selIdx, setSelIdx] = useState(-1)
  if (hwResults.length === 0) return null

  const rows: { hw: string; ws: number; c: StrategyResult; isOptimal: boolean }[] = []
  for (const hw of hwResults) {
    const wsBest = new Map<number, StrategyResult>()
    for (const c of hw.candidates.filter(c => c.meets_target)) {
      const cur = wsBest.get(c.world_size)
      if (!cur || c.tps > cur.tps) wsBest.set(c.world_size, c)
    }
    for (const [ws, c] of [...wsBest.entries()].sort((a, b) => a[0] - b[0])) {
      const isOpt = optimal != null &&
        c.world_size === optimal.world_size &&
        c.tp_size === optimal.tp_size &&
        c.dp_size === optimal.dp_size &&
        c.batch_size === optimal.batch_size &&
        c.strategy_label === optimal.strategy_label
      rows.push({ hw: hw.hardware_name, ws, c, isOptimal: isOpt })
    }
  }

  if (rows.length === 0) return null

  function handleApply() {
    if (selIdx < 0) return
    const r = rows[selIdx]
    if (r) {
      onApply(r.c)
      setApplied(true)
    }
  }

  return (
    <div className="opt-candidates-wrap">
      <div className="opt-candidates-header">
        <span className="opt-candidates-title">最优配置统计</span>
        <button className="btn primary opt-apply-btn"
          onClick={handleApply} disabled={selIdx < 0 || applied}>
          {applied ? '✓ 已应用' : '应用策略'}
        </button>
      </div>
      <div className="op-stats-scroll" style={{ maxHeight: 240 }}>
        <table className="opt-candidates-table">
          <thead>
            <tr>
              {hwResults.length > 1 && <th>硬件</th>}
              <th>WS</th><th>策略</th><th>TP</th><th>DP</th>
              <th>Emb</th><th>O</th><th>LM</th><th>BS</th>
              <th>TPOT</th><th>TPS</th><th>显存</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={i} className={i === selIdx ? 'opt-row-best' : r.isOptimal ? 'opt-row-ok' : ''}
                onClick={() => { setSelIdx(i); setApplied(false) }}
                style={{ cursor: 'pointer' }}>
                {hwResults.length > 1 && <td>{r.hw}</td>}
                <td>{r.ws}</td>
                <td className="opt-td-strategy">{r.c.strategy_label}{r.isOptimal ? ' ★' : ''}</td>
                <td>{r.c.tp_size}</td><td>{r.c.dp_size}</td>
                <td>{r.c.embed_tp_size}</td><td>{r.c.o_tp_size}</td><td>{r.c.lmhead_tp_size}</td>
                <td>{r.c.batch_size}</td>
                <td>{r.c.tpot_ms.toFixed(1)}ms</td>
                <td>{r.c.tps.toFixed(1)}</td>
                <td>{fm(r.c.max_peak_mem_gb)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── main ──────────────────────────────────────────────────

export default function OptimizeResults({ result }: { result: OptimizeResult }) {
  const hwResults = result.hardware_results || []
  const [hwIdx, setHwIdx] = useState(0)
  const [filterStatus, setFilterStatus] = useState('')
  const [filterWs, setFilterWs] = useState('')
  const set = useInferenceStore.setState
  const s = result.search_summary

  const currentHw = hwResults.length > 0 ? hwResults[hwIdx] : null
  const displayOpt = currentHw?.optimal ?? result.optimal

  function applyStrategy(c: StrategyResult) {
    set({
      tp_size: c.tp_size,
      dp_size: c.dp_size,
      world_size: c.world_size,
      ep_size: c.world_size,
      embed_tp_size: c.embed_tp_size,
      o_tp_size: c.o_tp_size,
      lmhead_tp_size: c.lmhead_tp_size,
      batch_size: c.batch_size,
    } as any)
  }

  const totalEval = s.evaluated
  const meetTotal = result.candidates.filter(c => c.meets_target).length
  const oomTotal = s.oom_count
  const notMeet = totalEval - meetTotal - oomTotal

  return (
    <div className="opt-results">
      {/* ── 统计概览 ── */}
      <div className="opt-summary-bar">
        <span>共寻优 <strong>{totalEval}</strong> 组</span>
        <span>耗时 <strong>{(s.elapsed_ms / 1000).toFixed(1)}s</strong></span>
        <span className="opt-stat-ok">满足 TPOT <strong>{meetTotal}</strong></span>
        <span className="opt-stat-over">未满足 <strong>{notMeet}</strong></span>
        <span className="opt-stat-oom">OOM <strong>{oomTotal}</strong></span>
      </div>

      {!displayOpt && (
        <div className="opt-no-result">
          <span className="opt-no-icon">!</span>
          <div>
            <p>未找到满足目标 TPOT 的策略</p>
            <p className="opt-no-hint">请尝试：放宽目标 TPOT、增加最大卡数</p>
          </div>
        </div>
      )}

      {/* ═══ Section 1: TPS 变化趋势 ═══ */}
      <WsTpsChart hwResults={hwResults} hwIdx={hwIdx} setHwIdx={setHwIdx} />

      {/* ═══ Section 2: 最优配置统计 ═══ */}
      <OptimalStatsTable hwResults={hwResults} onApply={applyStrategy} optimal={currentHw?.optimal ?? result.optimal} />

      {/* ═══ Section 3: 硬件 Tab + 候选策略 ═══ */}
      {hwResults.length > 0 && (
        <div className="opt-candidates-wrap">
          {/* hardware tabs */}
          <div className="opt-candidates-header">
            <span className="opt-candidates-title">候选策略列表</span>
            <span className="opt-candidates-meta">
              耗时 {(s.elapsed_ms / 1000).toFixed(1)}s
            </span>
          </div>
          {hwResults.length > 1 && (
            <div className="hw-selector">
              {hwResults.map((hw, i) => (
                <button key={hw.hardware_name}
                  className={`hw-selector-btn${i === hwIdx ? ' active' : ''}`}
                  onClick={() => setHwIdx(i)}>
                  {hw.hardware_name}
                </button>
              ))}
            </div>
          )}
          {/* candidate list for selected hardware */}
          {(() => {
            const hw = hwResults[hwIdx]
            const allCandidates = hw.candidates
            const hwOpt = hw.optimal
            // apply filters
            const filterWsNum = filterWs ? parseInt(filterWs) : NaN
            const candidates = allCandidates.filter(c => {
              if (filterStatus === 'meets' && !c.meets_target) return false
              if (filterStatus === 'not_meet' && (c.meets_target || c.is_oom)) return false
              if (filterStatus === 'oom' && !c.is_oom) return false
              if (!isNaN(filterWsNum) && c.world_size !== filterWsNum) return false
              return true
            })
            const meetCount = allCandidates.filter(c => c.meets_target).length
            const oomCount = hw.search_summary.oom_count
            return (
              <>
                <div className="opt-hw-stats">
                  达标 <strong>{meetCount}</strong> · OOM <strong>{oomCount}</strong> · 总计 <strong>{allCandidates.length}</strong>
                  {candidates.length !== allCandidates.length && <> · 筛选 <strong>{candidates.length}</strong></>}
                </div>
                <div className="opt-filter-bar">
                  <select className="opt-filter-select" value={filterStatus}
                    onChange={(e) => setFilterStatus(e.target.value)}>
                    <option value="">全部状态</option>
                    <option value="meets">达标</option>
                    <option value="not_meet">未满足</option>
                    <option value="oom">OOM</option>
                  </select>
                  <input type="number" className="opt-filter-ws" value={filterWs}
                    onChange={(e) => setFilterWs(e.target.value)}
                    onBlur={(e) => { const v = parseInt(e.target.value); if (v > 0) setFilterWs(String(toPowerOfTwo(v))) }}
                    placeholder="WS 过滤" />
                </div>
                <div className="op-stats-scroll" style={{ maxHeight: 320 }}>
                  <table className="opt-candidates-table">
                    <thead>
                      <tr>
                        <th>#</th><th>策略</th><th>WS</th><th>TP</th><th>DP</th>
                        <th>Emb</th><th>O</th><th>LM</th><th>BS</th>
                        <th>TPOT</th><th>TPS</th><th>显存</th><th>达标</th>
                      </tr>
                    </thead>
                    <tbody>
                      {candidates.map((c, i) => (
                        <tr key={i} className={
                          c === hwOpt ? 'opt-row-best'
                          : c.meets_target ? 'opt-row-ok'
                          : c.is_oom ? 'opt-row-oom'
                          : 'opt-row-over'
                        }>
                          <td className="os-num os-muted">{i + 1}</td>
                          <td className="opt-td-strategy">{c.strategy_label}{c === hwOpt ? ' ★' : ''}</td>
                          <td>{c.world_size}</td><td>{c.tp_size}</td><td>{c.dp_size}</td>
                          <td>{c.embed_tp_size}</td><td>{c.o_tp_size}</td><td>{c.lmhead_tp_size}</td>
                          <td>{c.batch_size}</td>
                          <td>{c.is_oom ? '—' : c.tpot_ms.toFixed(1) + 'ms'}</td>
                          <td>{c.is_oom ? '—' : c.tps.toFixed(1)}</td>
                          <td>{c.is_oom ? '—' : fm(c.max_peak_mem_gb)}</td>
                          <td>{c.is_oom ? 'OOM' : c.meets_target ? '✓' : '✗'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            )
          })()}
        </div>
      )}
    </div>
  )
}