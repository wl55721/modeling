import { useState, useMemo, useCallback } from 'react'
import { useModelStore } from '../../stores/model'
import type { ModuleDef } from '../../types/model'

export default function CustomModuleDialog({
  initial,
  onSave,
  onCancel,
}: {
  initial?: ModuleDef
  onSave: (mod: ModuleDef) => void
  onCancel: () => void
}) {
  const operators = useModelStore((s) => s.operators)
  const customOperators = useModelStore((s) => s.customOperators)
  const allOps = useMemo(() => {
    const existingNames = new Set(operators.map((o) => o.name))
    const uniqueCustoms = customOperators.filter((o) => !existingNames.has(o.name))
    return [...operators, ...uniqueCustoms]
  }, [operators, customOperators])

  const [name, setName] = useState(initial?.label ?? '')
  const [selected, setSelected] = useState<Set<string>>(
    () => new Set(initial?.operatorNames ?? [])
  )
  const [error, setError] = useState('')

  const isEdit = !!initial

  const toggle = useCallback((opName: string) => {
    setSelected((prev) => {
      const next = new Set(prev)
      if (next.has(opName)) next.delete(opName)
      else next.add(opName)
      return next
    })
    setError('')
  }, [])

  function handleSave() {
    const trimmed = name.trim()
    if (!trimmed) { setError('module名称不能为空'); return }
    if (selected.size === 0) { setError('请至少选择一个算子'); return }
    onSave({ name: trimmed, label: trimmed, operatorNames: [...selected], isBuiltin: false })
  }

  return (
    <div className="modal-overlay" onClick={onCancel}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>{isEdit ? '编辑自定义MODULE' : '新建自定义MODULE'}</h3>
          <button className="modal-close" onClick={onCancel}>×</button>
        </div>
        <div className="modal-body">
          {error && <div className="modal-error">{error}</div>}

          <label className="mod-field">
            <span>module名称</span>
            <input value={name} placeholder="输入模块名称"
              onChange={(e) => { setError(''); setName(e.target.value) }} />
          </label>

          <div className="mod-field">
            <span>选择算子 ({selected.size})</span>
            <div className="mod-check-list">
              {allOps.map((op) => (
                <label key={op.name} className="mod-check-item">
                  <input type="checkbox"
                    checked={selected.has(op.name)}
                    onChange={() => toggle(op.name)} />
                  <span className="mod-check-name">{op.name}</span>
                  <span className="mod-check-desc">{op.description}</span>
                </label>
              ))}
            </div>
          </div>
        </div>
        <div className="modal-footer">
          <button className="btn" onClick={onCancel}>取消</button>
          <button className="btn primary" onClick={handleSave}>保存</button>
        </div>
      </div>
    </div>
  )
}
