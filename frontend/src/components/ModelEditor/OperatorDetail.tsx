import { useState } from 'react'
import { useModelStore, toList } from '../../stores/model'
import { cn } from '../../utils/classnames'
import { DTYPES } from '../../constants/operators'
import ModuleSelect from './ModuleSelect'
import type { OperatorDef } from '../../types/model'

function TensorTable({ items, emptyHint }: { items: { name: string; shape: string; dtype: string }[]; emptyHint: string }) {
  if (items.length === 0) return <div className="op-ro-empty">{emptyHint}</div>
  return (
    <table className="op-ro-table">
      <thead>
        <tr><th>Name</th><th>Shape</th><th>Dtype</th></tr>
      </thead>
      <tbody>
        {items.map((t, i) => (
          <tr key={i}>
            <td className="op-ro-name">{t.name || '-'}</td>
            <td className="op-ro-shape"><code>{t.shape || '-'}</code></td>
            <td className="op-ro-dtype">{t.dtype || '-'}</td>
          </tr>
        ))}
      </tbody>
    </table>
  )
}

function ReadOnlyOpDetail({ operator }: { operator: OperatorDef }) {
  const inputs = toList(operator.inputs)
  const params = toList(operator.params)
  const outputs = toList(operator.outputs)

  return (
    <div className="mc-detail">
      <div className="op-ro-header">
        <h4>{operator.name}</h4>
        {operator.category && <span className="op-ro-category">{operator.category}</span>}
      </div>
      {operator.description && <p className="op-ro-desc">{operator.description}</p>}

      <div className="op-ro-field">
        <label>Module</label>
        <span className="op-ro-value">{operator.module || operator.name}</span>
      </div>

      <h4>Inputs</h4>
      <TensorTable items={inputs} emptyHint="无输入" />

      <h4>Params</h4>
      <TensorTable items={params} emptyHint="无参数" />

      <h4>Outputs</h4>
      <TensorTable items={outputs} emptyHint="无输出" />

      <h4>Compute FLOPs</h4>
      <pre className="op-ro-flops">{operator.compute_flops || '0'}</pre>
    </div>
  )
}

export function SectionHeader({ title, count, collapsed, onToggle, onAdd }: {
  title: string; count: number; collapsed: boolean; onToggle: () => void; onAdd: () => void
}) {
  return (
    <div className="tr-header">
      <button className={cn('mc-collapse-btn', collapsed && 'collapsed')} onClick={onToggle}>&#9654;</button>
      <h4>{title}</h4>
      {collapsed && <span className="tr-count">{count}</span>}
      <span className="mc-spacer" />
      <button className="tr-add" onClick={onAdd}>+</button>
    </div>
  )
}

function OpDetail({ operator, onChange }: { operator: OperatorDef; onChange: (section: string, data: any) => void }) {
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set())

  const toggle = (section: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(section)) next.delete(section)
      else next.add(section)
      return next
    })
  }

  const addItem = (section: string, arr: any[]) => {
    onChange(section, [...arr, { name: '', shape: '', dtype: '' }])
    setCollapsed((prev) => {
      const next = new Set(prev)
      next.delete(section)
      return next
    })
  }

  const itemRows = (section: string, arr: any[]) => {
    if (collapsed.has(section)) return null
    return arr.map((t, i) => (
      <div key={i} className="mc-op-row">
        <input className="tr-name" value={t.name} placeholder="name"
          onChange={(e) => { const next = [...arr]; next[i] = { ...next[i], name: e.target.value }; onChange(section, next) }} />
        <input className="tr-shape" value={t.shape} placeholder="shape"
          onChange={(e) => { const next = [...arr]; next[i] = { ...next[i], shape: e.target.value }; onChange(section, next) }} />
        <select className="tr-dtype" value={t.dtype}
          onChange={(e) => { const next = [...arr]; next[i] = { ...next[i], dtype: e.target.value }; onChange(section, next) }}>
          {DTYPES.map((dt) => <option key={dt} value={dt}>{dt}</option>)}
        </select>
        <button className="tr-remove" onClick={() => onChange(section, arr.filter((_, j) => j !== i))}>×</button>
      </div>
    ))
  }

  const inputs = toList(operator.inputs)
  const params = toList(operator.params)
  const outputs = toList(operator.outputs)

  return (
    <div className="mc-detail">
      <h4>Module</h4>
      <ModuleSelect value={operator.module || ''}
        onChange={(v) => onChange('module', v)} />

      <SectionHeader title="Inputs" count={inputs.length} collapsed={collapsed.has('inputs')}
        onToggle={() => toggle('inputs')} onAdd={() => addItem('inputs', inputs)} />
      {itemRows('inputs', inputs)}

      <SectionHeader title="Params" count={params.length} collapsed={collapsed.has('params')}
        onToggle={() => toggle('params')} onAdd={() => addItem('params', params)} />
      {itemRows('params', params)}

      <SectionHeader title="Outputs" count={outputs.length} collapsed={collapsed.has('outputs')}
        onToggle={() => toggle('outputs')} onAdd={() => addItem('outputs', outputs)} />
      {itemRows('outputs', outputs)}

      <h4>Compute FLOPs</h4>
      <textarea className="mc-flops" value={operator.compute_flops || '0'} rows={5}
        onChange={(e) => onChange('compute_flops', e.target.value)} />
    </div>
  )
}

export default function OperatorDetail() {
  const nodes = useModelStore((s) => s.nodes)
  const layers = useModelStore((s) => s.layers)
  const topGlobalIndices = useModelStore((s) => s.topGlobalIndices)
  const bottomGlobalIndices = useModelStore((s) => s.bottomGlobalIndices)
  const selectedNodeId = useModelStore((s) => s.selectedNodeId)
  const selectedOperator = useModelStore((s) => s.selectedOperator)
  const updateSection = useModelStore((s) => s.updateNodeSection)

  if (selectedNodeId) {
    const selNode = nodes.find((n) => String(n.index) === selectedNodeId)
    if (!selNode) return <div className="config-panel"><div className="empty-hint">未找到算子</div></div>

    const isBuiltin = selNode.operator.name === 'start' || selNode.operator.name === 'end'
    let layerIdx: number
    if (isBuiltin) layerIdx = selNode.operator.name === 'start' ? -2 : 1000000
    else if (topGlobalIndices.includes(selNode.index)) layerIdx = -1
    else if (bottomGlobalIndices.includes(selNode.index)) layerIdx = 999999
    else {
      const l = layers.find((l) => Object.values(l.rankOps).flat().includes(selNode.index))
      layerIdx = l ? l.layerIdx : -1
    }
    const rankIdx = isBuiltin ? -1 : selNode.rank

    return (
      <div className="config-panel">
        <div className="layer-info">
          <strong>{selNode.operator.name}</strong>
          <div className="layer-info-meta">layer_idx: {layerIdx} &nbsp; rank_idx: {rankIdx}</div>
        </div>
        <div className="config-body">
          <OpDetail operator={selNode.operator}
            onChange={(section, data) => updateSection(selNode.index, section, data)} />
        </div>
      </div>
    )
  }

  if (selectedOperator) {
    return (
      <div className="config-panel">
        <div className="layer-info">
          <strong>算子详情</strong>
        </div>
        <div className="config-body">
          <ReadOnlyOpDetail operator={selectedOperator} />
        </div>
      </div>
    )
  }

  return (
    <div className="config-panel">
      <div className="empty-hint">点击画布中的算子或算子库中的算子</div>
    </div>
  )
}
