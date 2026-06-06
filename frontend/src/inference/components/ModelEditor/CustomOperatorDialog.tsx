import { useState, useCallback } from 'react'
import { DTYPES } from '../../constants/operators'
import { SectionHeader } from './OperatorDetail'
import ModuleSelect from './ModuleSelect'
import type { OperatorDef, TensorMeta } from '../../types/model'

const COMPUTE_UNITS = [
  { value: 'cube', label: 'Cube（矩阵乘/密集计算）' },
  { value: 'vector', label: 'Vector（逐元素操作）' },
  { value: 'mix', label: 'Mix（混合计算）' },
  { value: 'sfu', label: 'SFU（特殊函数：exp/sin/sigmoid 等）' },
  { value: 'communication', label: 'Communication（通信）' },
]

function emptyOp(): OperatorDef {
  return { name: '', module: '', description: '', inputs: [], params: [], outputs: [], compute_flops: '0', compute_unit: 'cube' }
}

function TensorRows({ sectionId, arr, collapsed, onChange }: {
  sectionId: string; arr: TensorMeta[]; collapsed: Set<string>; onChange: (arr: TensorMeta[]) => void
}) {
  if (collapsed.has(sectionId)) return null
  return arr.map((t, i) => (
    <div key={i} className="mc-op-row">
      <input className="tr-name" value={t.name} placeholder="name"
        onChange={(e) => { const next = [...arr]; next[i] = { ...next[i], name: e.target.value }; onChange(next) }} />
      <input className="tr-shape" value={t.shape} placeholder="shape"
        onChange={(e) => { const next = [...arr]; next[i] = { ...next[i], shape: e.target.value }; onChange(next) }} />
      <select className="tr-dtype" value={t.dtype}
        onChange={(e) => { const next = [...arr]; next[i] = { ...next[i], dtype: e.target.value }; onChange(next) }}>
        {DTYPES.map((dt) => <option key={dt} value={dt}>{dt}</option>)}
      </select>
      <button className="tr-remove" onClick={() => onChange(arr.filter((_, j) => j !== i))}>×</button>
    </div>
  ))
}

export default function CustomOperatorDialog({
  initial,
  onSave,
  onCancel,
}: {
  initial?: OperatorDef
  onSave: (op: OperatorDef) => void
  onCancel: () => void
}) {
  const [op, setOp] = useState<OperatorDef>(initial ?? emptyOp())
  const [collapsed, setCollapsed] = useState<Set<string>>(new Set(['inputs', 'params', 'outputs']))
  const [error, setError] = useState('')

  const isEdit = !!initial

  const toggleCollapse = useCallback((section: string) => {
    setCollapsed((prev) => {
      const next = new Set(prev)
      if (next.has(section)) next.delete(section); else next.add(section)
      return next
    })
  }, [])

  const addToSection = useCallback((section: keyof OperatorDef) => {
    setOp((prev) => {
      const arr = prev[section] as TensorMeta[]
      return { ...prev, [section]: [...arr, { name: '', shape: '', dtype: '' }] }
    })
    setCollapsed((prev) => {
      const next = new Set(prev)
      next.delete(section as string)
      return next
    })
  }, [])

  function handleSave() {
    if (!op.name.trim()) { setError('算子名称不能为空'); return }
    if (!op.module.trim()) { setError('请选择 Module'); return }
    const existingName = op.name.trim()
    onSave({ ...op, name: existingName, module: op.module.trim(), description: op.description.trim() })
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{isEdit ? '编辑自定义算子' : '新建自定义算子'}</h3>
          <button className="modal-close" onClick={onCancel}>×</button>
        </div>
        <div className="modal-body">
          {error && <div className="modal-error">{error}</div>}

          <label className="mod-field">
            <span>算子名称</span>
            <input value={op.name} placeholder="输入算子名称"
              onChange={(e) => { setError(''); setOp({ ...op, name: e.target.value }) }} />
          </label>

          <label className="mod-field">
            <span>Module</span>
            <ModuleSelect value={op.module}
              onChange={(v) => setOp({ ...op, module: v })} />
          </label>

          <label className="mod-field">
            <span>描述</span>
            <input value={op.description} placeholder="简短描述"
              onChange={(e) => setOp({ ...op, description: e.target.value })} />
          </label>

          <label className="mod-field">
            <span>计算单元</span>
            <select value={op.compute_unit || 'cube'}
              onChange={(e) => setOp({ ...op, compute_unit: e.target.value })}>
              {COMPUTE_UNITS.map((u) => (
                <option key={u.value} value={u.value}>{u.label}</option>
              ))}
            </select>
          </label>

          <SectionHeader title="Inputs" count={op.inputs.length} collapsed={collapsed.has('inputs')}
            onToggle={() => toggleCollapse('inputs')} onAdd={() => addToSection('inputs')} />
          <TensorRows sectionId="inputs" arr={op.inputs} collapsed={collapsed}
            onChange={(arr) => setOp({ ...op, inputs: arr })} />

          <SectionHeader title="Params" count={op.params.length} collapsed={collapsed.has('params')}
            onToggle={() => toggleCollapse('params')} onAdd={() => addToSection('params')} />
          <TensorRows sectionId="params" arr={op.params} collapsed={collapsed}
            onChange={(arr) => setOp({ ...op, params: arr })} />

          <SectionHeader title="Outputs" count={op.outputs.length} collapsed={collapsed.has('outputs')}
            onToggle={() => toggleCollapse('outputs')} onAdd={() => addToSection('outputs')} />
          <TensorRows sectionId="outputs" arr={op.outputs} collapsed={collapsed}
            onChange={(arr) => setOp({ ...op, outputs: arr })} />

          <label className="mod-field">
            <span>Compute FLOPs</span>
            <textarea className="mc-flops" value={op.compute_flops} rows={4}
              onChange={(e) => setOp({ ...op, compute_flops: e.target.value })} />
          </label>
        </div>
        <div className="modal-footer">
          <button className="btn" onClick={onCancel}>取消</button>
          <button className="btn primary" onClick={handleSave}>保存</button>
        </div>
      </div>
    </div>
  )
}
