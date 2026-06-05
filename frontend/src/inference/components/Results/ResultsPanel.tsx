import { Fragment, useEffect, useRef, useState } from 'react'
import { cn } from '../../utils/classnames'
import type { SimulateSingleResult, OperatorStatistics, OperatorResult } from '../../types/results'

function fm(gb: number) { return (gb ?? 0).toFixed(2) + ' GB' }
function fl(ms: number) { return (ms ?? 0) < 1000 ? (ms ?? 0).toFixed(2) + ' ms' : ((ms ?? 0) / 1000).toFixed(2) + ' s' }
function fc(us: number) { return (us ?? 0) < 1000 ? (us ?? 0).toFixed(1) + ' us' : ((us ?? 0) / 1000).toFixed(2) + ' ms' }
function fb(bytes: number) { const b = bytes ?? 0; return b >= 1024 * 1024.0 * 1024.0 ? (b / 1024.0 / 1024.0 / 1024.0).toFixed(2) + ' GB' : (b / 1024.0 / 1024.0).toFixed(1) + ' MB' }
function oc(oom: boolean) { return oom ? 'oom-yes' : 'oom-no' }

const PIE_COLORS = [
  '#4f5eb1', '#3b6fb6', '#4d8c57', '#c4504a', '#e67e22',
  '#8e6bb8', '#2e86c1', '#b866cc', '#9e9e9e', '#7a9fcf',
  '#d4842a', '#5b6abf', '#6b8e23', '#cd853f', '#4682b4',
]

type TabKey = 'model' | 'opstats' | 'rank'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'model', label: '模型统计' },
  { key: 'rank', label: 'Rank 统计' },
  { key: 'opstats', label: '算子统计' },
]

export default function ResultsPanel({ results }: { results: SimulateSingleResult[] | null }) {
  const [tab, setTab] = useState<TabKey>('model')
  const [hwIdx, setHwIdx] = useState(0)

  if (!results || results.length === 0) {
    return (
      <div className="results-panel">
        <div className="empty-hint"><p>点击"运行估算"按钮</p><p>查看推理建模结果</p></div>
      </div>
    )
  }

  const names = results.map((r) => r.hardware_name)
  const r0 = results[hwIdx].result

  // Best-value computation
  const bestTPOT = Math.min(...results.map((r) => r.result.tpot_ms))
  const bestPrefill = Math.min(...results.map((r) => r.result.prefill_latency_ms))
  const bestDecode = Math.min(...results.map((r) => r.result.decode_latency_per_token_ms))
  const bestTPS = Math.max(...results.map((r) => r.result.tps))
  const bestQPS = Math.max(...results.map((r) => r.result.qps))
  const bestMemory = Math.min(...results.map((r) => r.result.peak_mem_gb))
  function best(v: number, bestV: number) { return v === bestV ? ' metric-best' : '' }

  // Pre-compute operator stats
  const sortedOpStats = [...r0.op_statistics]
    .filter((s) => s.op_name !== 'start' && s.op_name !== 'end')
    .sort((a, b) => b.total_cost_us - a.total_cost_us)
  const totalOpUs = sortedOpStats.reduce((sum, s) => sum + s.total_cost_us, 0)

  // Pre-compute operator details
  const allDetailOps = r0.operators.filter(op => op.op_name !== 'start' && op.op_name !== 'end')
  const availableRanks = [...new Set(allDetailOps.map(op => op.rank_idx))].filter(r => r != null).sort((a, b) => a - b)
  const availableLayers = [...new Set(allDetailOps.map(op => op.layer_idx))].sort((a, b) => a - b)

  // Pre-compute rank stats
  const rankStats = r0.ranks
  const maxCost = Math.max(...rankStats.map(r => r.total_cost_ms))
  const maxMem = Math.max(...rankStats.map(r => r.peak_mem_gb))
  const oomCount = rankStats.filter(r => r.oom).length

  return (
    <div className="results-panel">
      <div className="results-tabs">
        {TABS.map(t => {
          const count = t.key === 'model' ? names.length
            : t.key === 'rank' ? rankStats.length
            : sortedOpStats.length
          return (
            <button key={t.key} className={`results-tab${tab === t.key ? ' active' : ''}`} onClick={() => setTab(t.key)}>
              {t.label}<span className="tab-count">{count}</span>
            </button>
          )
        })}
      </div>

      <div className="results-body">
        {tab === 'model' && (
          <div className="section">
            <div className="results-compare-wrap">
              <table className="results-compare-table">
                <thead>
                  <tr>
                    <th>指标</th>
                    {names.map((n, i) => <th key={i}>{n}</th>)}
                  </tr>
                </thead>
                <tbody>
                  <tr><td className="rc-label" colSpan={names.length + 1}>时延指标</td></tr>
                  <tr>
                    <td>TPOT</td>
                    {results.map((r, i) => <td key={i} className={best(r.result.tpot_ms, bestTPOT)}>{fl(r.result.tpot_ms)}</td>)}
                  </tr>
                  <tr>
                    <td>Prefill Latency</td>
                    {results.map((r, i) => <td key={i} className={best(r.result.prefill_latency_ms, bestPrefill)}>{fl(r.result.prefill_latency_ms)}</td>)}
                  </tr>
                  <tr>
                    <td>Decode / Token</td>
                    {results.map((r, i) => <td key={i} className={best(r.result.decode_latency_per_token_ms, bestDecode)}>{fl(r.result.decode_latency_per_token_ms)}</td>)}
                  </tr>
                  <tr><td className="rc-label" colSpan={names.length + 1}>吞吐指标</td></tr>
                  <tr>
                    <td>TPS</td>
                    {results.map((r, i) => <td key={i} className={best(r.result.tps, bestTPS)}>{r.result.tps.toFixed(2)}</td>)}
                  </tr>
                  <tr>
                    <td>QPS</td>
                    {results.map((r, i) => <td key={i} className={best(r.result.qps, bestQPS)}>{r.result.qps.toFixed(2)}</td>)}
                  </tr>
                  <tr><td className="rc-label" colSpan={names.length + 1}>显存总览</td></tr>
                  <tr>
                    <td>总显存占用</td>
                    {results.map((r, i) => <td key={i} className={best(r.result.peak_mem_gb, bestMemory)}>{fm(r.result.peak_mem_gb)}</td>)}
                  </tr>
                  <tr>
                    <td>OOM</td>
                    {results.map((r, i) => <td key={i}><span className={`oom-badge ${oc(r.result.oom)}`}>{r.result.oom ? 'OOM!' : 'OK'}</span></td>)}
                  </tr>
                  <tr>
                    <td>并行策略</td>
                    {results.map((r, i) => <td key={i} className="rc-strategy">{r.result.strategy}</td>)}
                  </tr>
                </tbody>
              </table>
            </div>
          </div>
        )}

        {tab === 'opstats' && (
          <div className="section">
            <HardwareSelector names={names} selected={hwIdx} onChange={setHwIdx} />
            <div className="section-title">算子统计</div>
            <DonutChart stats={sortedOpStats} totalUs={totalOpUs} />
            <OpStatsTable stats={sortedOpStats} totalUs={totalOpUs} />
            <div className="section-title">算子详情</div>
            <OpTensorTable ops={allDetailOps} ranks={availableRanks} layers={availableLayers} />
          </div>
        )}

        {tab === 'rank' && (
          <div className="section">
            <HardwareSelector names={names} selected={hwIdx} onChange={setHwIdx} />
            <div className="rank-summary">
              <div className="rank-summary-item">
                <span className="rank-summary-label">总 Ranks</span>
                <span className="rank-summary-value">{rankStats.length}</span>
              </div>
              <div className="rank-summary-item">
                <span className="rank-summary-label">最慢耗时</span>
                <span className="rank-summary-value">{fl(maxCost)}</span>
              </div>
              <div className="rank-summary-item">
                <span className="rank-summary-label">最大峰值显存</span>
                <span className="rank-summary-value">{fm(maxMem)}</span>
              </div>
              <div className="rank-summary-item">
                <span className="rank-summary-label">OOM</span>
                <span className={`rank-summary-value ${oomCount > 0 ? 'oom-warn' : ''}`}>
                  {oomCount > 0 ? `${oomCount}/${rankStats.length}` : '全部 OK'}
                </span>
              </div>
            </div>
            <div className="rank-grid">
              {rankStats.map((rs) => (
                <div key={rs.rank_idx}
                  className={`rank-cell ${rs.oom ? 'rank-cell-oom' : 'rank-cell-ok'}`}
                  title={`Rank ${rs.rank_idx}: ${fl(rs.total_cost_ms)} | ${fm(rs.peak_mem_gb)} | 权重 ${fb(rs.param_bytes)} | 激活 ${fb(rs.io_bytes)} | ${rs.num_ops} ops`}>
                  <span className="rank-cell-idx">{rs.rank_idx}</span>
                </div>
              ))}
            </div>
            <RankDetailTable rankStats={rankStats} />
            <RankLayersTable rankStats={rankStats} />
          </div>
        )}
      </div>
    </div>
  )
}

function HardwareSelector({ names, selected, onChange }: { names: string[]; selected: number; onChange: (i: number) => void }) {
  if (names.length <= 1) return null
  return (
    <div className="hw-selector">
      {names.map((n, i) => (
        <button key={n} className={`hw-selector-btn${i === selected ? ' active' : ''}`} onClick={() => onChange(i)}>
          {n}
        </button>
      ))}
    </div>
  )
}

function DonutChart({ stats, totalUs }: { stats: OperatorStatistics[]; totalUs: number }) {
  const radius = 72
  const stroke = 20
  const size = 200
  const cx = size / 2; const cy = size / 2
  const circumference = 2 * Math.PI * radius
  const gapAngle = 0.02
  const [showAllLegend, setShowAllLegend] = useState(false)

  let accumulated = 0
  const slices = stats.map((s) => {
    const pct = s.total_cost_us / totalUs
    const gapLen = gapAngle * radius
    const arcLen = pct * circumference - gapLen
    const dashOffset = -(accumulated / totalUs * circumference + gapLen / 2)
    accumulated += s.total_cost_us
    return { ...s, pct, dashOffset, arcLen: Math.max(arcLen, 0.5) }
  })

  const mainOp = slices.reduce((a, b) => a.pct > b.pct ? a : b, slices[0])

  function sliceColor(i: number, name: string) {
    return name === mainOp.op_name ? 'var(--signal-red)' : PIE_COLORS[i % PIE_COLORS.length]
  }

  return (
    <div className="donut-chart-wrap">
      <div className="donut-chart-box">
        <svg viewBox={`0 0 ${size} ${size}`} className="donut-chart">
          <circle cx={cx} cy={cy} r={radius} fill="none"
            stroke="var(--bg-panel)" strokeWidth={stroke} />
          {slices.map((s, i) => (
            <circle key={s.op_name} cx={cx} cy={cy} r={radius} fill="none"
              stroke={sliceColor(i, s.op_name)}
              strokeWidth={stroke}
              strokeDasharray={`${s.arcLen} ${circumference - s.arcLen}`}
              strokeDashoffset={s.dashOffset}
              strokeLinecap="round"
              transform={`rotate(-90 ${cx} ${cy})`}
              className={s.op_name === mainOp.op_name ? 'donut-slice donut-slice-top' : 'donut-slice'}
            >
              <title>{s.op_name}: {(s.pct * 100).toFixed(1)}% — {fc(s.total_cost_us)}</title>
            </circle>
          ))}
        </svg>
        <div className="donut-center">
          <span className="donut-center-label">总计</span>
          <span className="donut-center-value">{fc(totalUs)}</span>
        </div>
      </div>
      <div className={`donut-legend${slices.length > 10 ? ' donut-legend-cols2' : ''}`}>
        {(showAllLegend ? slices : slices.slice(0, 10)).map((s, i) => {
          const isTop = s.op_name === mainOp.op_name
          const color = isTop ? 'var(--signal-red)' : PIE_COLORS[i % PIE_COLORS.length]
          return (
          <div key={s.op_name} className={`donut-legend-item${isTop ? ' donut-legend-top' : ''}`}>
            <div className="donut-legend-head">
              <span className="donut-dot" style={{ background: color }} />
              <span className="donut-legend-name" title={s.op_name}>{s.op_name}</span>
              <span className="donut-legend-pct">{(s.pct * 100).toFixed(1)}%</span>
            </div>
            <div className="donut-legend-bar">
              <span className="donut-legend-bar-fill"
                style={{ width: (s.pct / mainOp.pct * 100).toFixed(0) + '%', background: color }} />
            </div>
          </div>
        )})}
        {slices.length > 10 && (
          <button className="rank-show-all" onClick={() => setShowAllLegend(!showAllLegend)}>
            {showAllLegend ? '收起' : `显示全部 ${slices.length} 个算子`}
          </button>
        )}
      </div>
    </div>
  )
}

function OpStatsTable({ stats, totalUs }: { stats: OperatorStatistics[]; totalUs: number }) {
  type SortKey = 'total_cost_us' | 'avg_cost_us'
  const [sortKey, setSortKey] = useState<SortKey>('total_cost_us')
  const [sortAsc, setSortAsc] = useState(false)

  const sorted = [...stats].sort((a, b) => {
    const dir = sortAsc ? 1 : -1
    return (a[sortKey] - b[sortKey]) * dir
  })
  const display = sorted

  function handleSort(key: SortKey) {
    if (sortKey === key) setSortAsc(!sortAsc)
    else { setSortKey(key); setSortAsc(false) }
  }

  function SortArrow({ k }: { k: SortKey }) {
    if (sortKey !== k) return <span className="os-sort-icon os-sort-off">↕</span>
    return <span className="os-sort-icon">{sortAsc ? '↑' : '↓'}</span>
  }

  return (
    <div className="op-stats-wrap">
      <div className="op-stats-scroll">
      <table className="op-stats-table">
        <thead>
          <tr>
            <th className="os-right">#</th>
            <th>算子</th>
            <th className="os-right">调用次数</th>
            <th className="os-right os-sort-th" onClick={() => handleSort('total_cost_us')}>
              总耗时 (us) <SortArrow k="total_cost_us" />
            </th>
            <th className="os-right os-sort-th" onClick={() => handleSort('avg_cost_us')}>
              平均耗时 (us) <SortArrow k="avg_cost_us" />
            </th>
            <th className="os-right">占比</th>
          </tr>
        </thead>
        <tbody>
          {display.map((s, i) => (
            <tr key={s.op_name}>
              <td className="os-right os-num os-muted">{i + 1}</td>
              <td className="os-name">{s.op_name}</td>
              <td className="os-right os-num">{s.num_calls}</td>
              <td className="os-right os-num os-total">{fc(s.total_cost_us)}</td>
              <td className="os-right os-num">{fc(s.avg_cost_us)}</td>
              <td className="os-right os-num">
                <span className="os-pct-bar" style={{ width: (s.total_cost_us / totalUs * 100).toFixed(0) + '%' }} />
                {(s.total_cost_us / totalUs * 100).toFixed(1)}%
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  )
}

function nsToMs(ns: number) { return (ns / 1_000_000).toFixed(3) }

function RankLayersTable({ rankStats }: { rankStats: { rank_idx: number; layers: { layer_idx: number; layer_cost_ns: number; repeat: number; param_bytes: number; io_bytes: number; start_time_ns: number; end_time_ns: number }[] }[] }) {
  const [expanded, setExpanded] = useState<Set<number>>(new Set())
  const toggle = (r: number) => setExpanded(prev => { const next = new Set(prev); if (next.has(r)) next.delete(r); else next.add(r); return next })

  return (
    <div className="rank-layers-wrap">
      <div className="rank-layers-title">层详情</div>
      {rankStats.map((rs) => {
        const layers = rs.layers || []
        if (layers.length === 0) return null
        const isOpen = expanded.has(rs.rank_idx)
        const layerCostNs = layers.reduce((s, l) => s + l.layer_cost_ns * l.repeat, 0)
        const layerIoBytes = layers.reduce((s, l) => s + l.io_bytes * l.repeat, 0)
        const layerParamBytes = layers.reduce((s, l) => s + l.param_bytes * l.repeat, 0)
        return (
          <div key={rs.rank_idx} className="rank-layer-group">
            <div className="rank-layer-group-header" onClick={() => toggle(rs.rank_idx)}>
              <span className={cn('op-cat-arrow', isOpen && 'open')}>&#9654;</span>
              <span className="rank-layer-rank">Rank {rs.rank_idx}</span>
              <span className="rank-layer-summary">
                {layers.length} 层 · 总耗时 {nsToMs(layerCostNs)}ms · 权重 {fb(layerParamBytes)} · 激活 {fb(layerIoBytes)}
              </span>
            </div>
            {isOpen && (
              <table className="rank-layer-table">
                <thead>
                  <tr>
                    <th>layer_idx</th>
                    <th className="os-right">repeat</th>
                    <th className="os-right">权重</th>
                    <th className="os-right">激活</th>
                    <th className="os-right">开始</th>
                    <th className="os-right">结束</th>
                    <th className="os-right">耗时</th>
                  </tr>
                </thead>
                <tbody>
                  {layers.map((l) => (
                    <tr key={l.layer_idx}>
                      <td className="os-name">{l.layer_idx < 0 ? (l.layer_idx === -1 ? '层前' : 'start') : l.layer_idx >= 900 ? (l.layer_idx >= 980 ? `MTP ${l.layer_idx}` : '层后') : `Layer ${l.layer_idx}`}</td>
                      <td className="os-right os-num">×{l.repeat}</td>
                      <td className="os-right os-num">{fb(l.param_bytes)}</td>
                      <td className="os-right os-num">{fb(l.io_bytes)}</td>
                      <td className="os-right os-num">{nsToMs(l.start_time_ns)}ms</td>
                      <td className="os-right os-num">{nsToMs(l.end_time_ns)}ms</td>
                      <td className="os-right os-num">{nsToMs(l.layer_cost_ns)}ms</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        )
      })}
    </div>
  )
}

function RankDetailTable({ rankStats }: { rankStats: { rank_idx: number; total_cost_ms: number; peak_mem_gb: number; noise_gb: number; mem_capacity_gb: number; oom: boolean; num_ops: number; param_bytes: number; io_bytes: number }[] }) {
  const [showAll, setShowAll] = useState(false)
  const display = showAll ? rankStats : rankStats.slice(0, 16)
  return (
    <>
      <table className="rank-detail-table">
        <thead>
          <tr>
            <th>Rank</th>
            <th className="os-right">耗时</th>
            <th className="os-right">峰值显存</th>
            <th className="os-right">权重</th>
            <th className="os-right">激活</th>
            <th className="os-right">噪声</th>
            <th className="os-right">容量</th>
            <th className="os-right">算子数</th>
            <th className="os-right">状态</th>
          </tr>
        </thead>
        <tbody>
          {display.map((rs) => (
            <tr key={rs.rank_idx} className={rs.oom ? 'rank-row-oom' : ''}>
              <td className="os-name">Rank {rs.rank_idx}</td>
              <td className="os-right os-num">{fl(rs.total_cost_ms)}</td>
              <td className="os-right os-num">{fm(rs.peak_mem_gb)}</td>
              <td className="os-right os-num">{fb(rs.param_bytes)}</td>
              <td className="os-right os-num">{fb(rs.io_bytes)}</td>
              <td className="os-right os-num">{fm(rs.noise_gb)}</td>
              <td className="os-right os-num">{fm(rs.mem_capacity_gb)}</td>
              <td className="os-right os-num">{rs.num_ops}</td>
              <td className="os-right"><span className={`oom-badge ${oc(rs.oom)}`}>{rs.oom ? 'OOM!' : 'OK'}</span></td>
            </tr>
          ))}
        </tbody>
      </table>
      {rankStats.length > 16 && (
        <button className="rank-show-all" onClick={() => setShowAll(!showAll)}>
          {showAll ? '收起' : `显示全部 ${rankStats.length} ranks`}
        </button>
      )}
    </>
  )
}

function tensorShapeStr(shape: number[]) {
  if (!shape || shape.length === 0) return '—'
  return '[' + shape.join(', ') + ']'
}

const OP_COLS = [
  { key: 'op_name', label: '算子', always: true },
  { key: 'layer_idx', label: '层' },
  { key: 'rank_idx', label: 'Rank' },
  { key: 'op_module', label: 'Module' },
  { key: 'compute_cost_us', label: '计算耗时 (us)' },
  { key: 'mem_cost_us', label: '访存耗时 (us)' },
  { key: 'comm_cost_us', label: '通信耗时 (us)' },
  { key: 'noise_us', label: '噪声 (us)' },
  { key: 'total_cost_us', label: '总耗时 (us)' },
  { key: 'bound_type', label: 'Bound' },
  { key: 'inputs_info', label: 'Inputs' },
  { key: 'params_info', label: 'Params' },
  { key: 'outputs_info', label: 'Outputs' },
] as const

function layerLabel(idx: number) {
  if (idx === -1) return '层前'
  if (idx === -2) return 'start'
  if (idx >= 980) return `MTP ${idx}`
  if (idx >= 900) return '层后'
  if (idx === 1000) return 'end'
  return `Layer ${idx}`
}

function OpTensorTable({ ops, ranks, layers }: { ops: OperatorResult[]; ranks: number[]; layers: number[] }) {
  const [selectedRank, setSelectedRank] = useState<number>(-1)
  const [selectedLayer, setSelectedLayer] = useState<number | null>(null)
  const ALL_LAYERS = null
  const [search, setSearch] = useState('')
  const HIDDEN_BY_DEFAULT = new Set(['rank_idx', 'op_module', 'bound_type', 'compute_cost_us', 'mem_cost_us', 'comm_cost_us', 'noise_us'])
  const [visibleCols, setVisibleCols] = useState<Set<string>>(
    new Set(OP_COLS.map(c => c.key).filter(k => !HIDDEN_BY_DEFAULT.has(k)))
  )
  const [showColPicker, setShowColPicker] = useState(false)
  const colPickerRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    function handleClick(e: MouseEvent) {
      if (colPickerRef.current && !colPickerRef.current.contains(e.target as Node)) {
        setShowColPicker(false)
      }
    }
    if (showColPicker) document.addEventListener('mousedown', handleClick)
    return () => document.removeEventListener('mousedown', handleClick)
  }, [showColPicker])

  const q = search.toLowerCase().trim()
  let filtered = selectedRank === -1 ? ops : ops.filter(op => op.rank_idx === selectedRank)
  if (selectedLayer !== null) filtered = filtered.filter(op => op.layer_idx === selectedLayer)
  if (q) filtered = filtered.filter(op => op.op_name.toLowerCase().includes(q))
  const display = filtered

  function toggleCol(key: string) {
    setVisibleCols(prev => {
      const next = new Set(prev)
      if (next.has(key)) next.delete(key)
      else next.add(key)
      return next
    })
  }

  function cellClass(key: string, ...extra: string[]) {
    return `col-${key} ${extra.join(' ')}`.trim()
  }

  function renderCell(op: OperatorResult, key: string) {
    switch (key) {
      case 'op_name': return <td className={cellClass(key, 'os-name')}>{op.op_name}</td>
      case 'bound_type': return <td className={cellClass(key, 'os-num')}>{op.bound_type || '—'}</td>
      case 'layer_idx': return <td className={cellClass(key, 'os-num')}>{op.layer_idx}</td>
      case 'rank_idx': return <td className={cellClass(key, 'os-num')}>{op.rank_idx}</td>
      case 'op_module': return <td className={cellClass(key, 'os-num')}>{op.op_module || '—'}</td>
      case 'compute_cost_us': return <td className={cellClass(key, 'os-right', 'os-num')}>{(op.compute_cost_us ?? 0).toFixed(2)}</td>
      case 'mem_cost_us': return <td className={cellClass(key, 'os-right', 'os-num')}>{(op.mem_cost_us ?? 0).toFixed(2)}</td>
      case 'comm_cost_us': return <td className={cellClass(key, 'os-right', 'os-num')}>{(op.comm_cost_us ?? 0).toFixed(2)}</td>
      case 'noise_us': return <td className={cellClass(key, 'os-right', 'os-num')}>{(op.noise_us ?? 0).toFixed(2)}</td>
      case 'total_cost_us': return <td className={cellClass(key, 'os-right', 'os-num', 'os-total')}>{(op.total_cost_us ?? 0).toFixed(2)}</td>
      case 'inputs_info': return <td className={cellClass(key, 'os-tensors')}>{renderTensors(op.inputs_info)}</td>
      case 'params_info': return <td className={cellClass(key, 'os-tensors')}>{renderTensors(op.params_info)}</td>
      case 'outputs_info': return <td className={cellClass(key, 'os-tensors')}>{renderTensors(op.outputs_info)}</td>
      default: return <td className={cellClass(key)} />;
    }
  }

  return (
    <div className="op-stats-wrap">
      <div className="op-detail-toolbar">
        <select className="op-rank-select" value={selectedRank} onChange={e => setSelectedRank(Number(e.target.value))}>
          <option value={-1}>全部 Rank ({ops.length})</option>
          {ranks.map(r => <option key={r} value={r}>Rank {r} ({ops.filter(op => op.rank_idx === r).length})</option>)}
        </select>
        <select className="op-rank-select" value={selectedLayer === null ? 'all' : selectedLayer} onChange={e => { const v = e.target.value; setSelectedLayer(v === 'all' ? null : Number(v)) }}>
          <option value="all">全部层 ({ops.length})</option>
          {layers.map(l => <option key={l} value={l}>{layerLabel(l)} ({ops.filter(op => op.layer_idx === l).length})</option>)}
        </select>
        <input className="op-search-input" type="text" placeholder="搜索算子名称..." value={search} onChange={e => setSearch(e.target.value)} />
        <span className="op-detail-count">{filtered.length} 个算子</span>
        <div className="op-col-picker-wrap" ref={colPickerRef}>
          <button className="op-col-picker-btn" onClick={() => setShowColPicker(!showColPicker)}>
            列选择 ▾
          </button>
          {showColPicker && (
            <div className="op-col-picker-drop">
              {OP_COLS.filter(c => !('always' in c)).map(c => (
                <label key={c.key} className="op-col-picker-item">
                  <input type="checkbox" checked={visibleCols.has(c.key)} onChange={() => toggleCol(c.key)} />
                  {c.label}
                </label>
              ))}
            </div>
          )}
        </div>
      </div>
      <div className="op-stats-scroll">
      <table className="op-stats-table op-detail-table">
        <thead>
          <tr>
            {OP_COLS.map(c => (('always' in c) || visibleCols.has(c.key)) && (
              <th key={c.key} className={'col-' + c.key + (c.key.endsWith('_us') ? ' os-right' : '')}>
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {display.map((op) => (
            <tr key={op.op_id}>
              {OP_COLS.map(c => (('always' in c) || visibleCols.has(c.key)) && (
                <Fragment key={c.key}>{renderCell(op, c.key)}</Fragment>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      </div>
    </div>
  )
}

function renderTensors(tensors: { name: string; shape: number[]; dtype: string }[] | undefined) {
  if (!tensors || tensors.length === 0) return <span className="os-muted">—</span>
  return tensors.map((t, i) => (
    <span key={i} className="tensor-chip" title={t.name || `t${i}`}>
      <span className="tensor-shape">{tensorShapeStr(t.shape)}</span><span className="tensor-dtype">{t.dtype}</span>
    </span>
  ))
}

