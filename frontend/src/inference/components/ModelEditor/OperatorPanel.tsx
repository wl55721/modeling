import { useState, useCallback, useMemo } from 'react'
import { useModelStore } from '../../stores/model'
import { cn } from '../../utils/classnames'
import { openFileDialog } from '../../utils/file'
import type { OperatorDef } from '../../types/model'
import CustomOperatorDialog from './CustomOperatorDialog'

const CATEGORY_COLORS: Record<string, string> = {
  Flow: '#9e9e9e',
  Embedding: '#3b6fb6',
  Normalization: '#8e6bb8',
  Attention: '#c4504a',
  Position: '#d4842a',
  Activation: '#e67e22',
  Linear: '#4d8c57',
  MoE: '#2e86c1',
  Communication: '#b866cc',
  Compressor: '#6b8e23',
  MTP: '#cd853f',
  Torch: '#7a9fcf',
}

const CATEGORY_ORDER = ['Flow', 'Embedding', 'Normalization', 'Attention', 'Position', 'Activation', 'Linear', 'MoE', 'Communication', 'Compressor', 'MTP', 'Torch']

function CategoryGroup({ name, color, ops, forceOpen, isCustom, onAddCustom, onEditCustom, onDeleteCustom, onDuplicateCustom, onImport, onExport }: {
  name: string; color: string; ops: OperatorDef[]; forceOpen: boolean
  isCustom?: boolean
  onAddCustom?: () => void
  onEditCustom?: (op: OperatorDef) => void
  onDeleteCustom?: (name: string) => void
  onDuplicateCustom?: (op: OperatorDef) => void
  onImport?: () => void
  onExport?: () => void
}) {
  const [open, setOpen] = useState(false)
  const selectOperator = useModelStore((s) => s.selectOperator)
  const selectedOperator = useModelStore((s) => s.selectedOperator)

  const isOpen = forceOpen || open

  const onDragStart = useCallback((event: React.DragEvent, op: OperatorDef) => {
    event.dataTransfer.setData('application/kepler-operator', op.name)
    event.dataTransfer.effectAllowed = 'move'
  }, [])

  if (ops.length === 0 && !isCustom) return null
  return (
    <div className="op-category">
      <div className="op-cat-header" onClick={() => setOpen(!open)}>
        <span className={cn('op-cat-arrow', isOpen && 'open')}>&#9654;</span>
        <span className="op-cat-name">{name}</span>
        <span className="op-cat-count">{ops.length}</span>
        {isCustom && (
          <>
            <button className="op-cat-import" onClick={(e) => { e.stopPropagation(); onImport?.() }}
              title="导入">导入</button>
            <button className="op-cat-export" onClick={(e) => { e.stopPropagation(); onExport?.() }}
              title="导出">导出</button>
            <button className="op-cat-add" onClick={(e) => { e.stopPropagation(); onAddCustom?.() }}
              title="新建自定义算子">+</button>
          </>
        )}
      </div>
      {isOpen && (
        <div className="op-cat-items">
          {ops.length === 0 && isCustom && (
            <div className="op-empty-hint">点击 + 新建自定义算子</div>
          )}
          {ops.map((op) => (
            <div key={op.name}
              className={cn('operator-item', selectedOperator?.name === op.name && 'op-selected', isCustom && 'op-custom')}
              style={{ ['--cat-color' as string]: color }}
              draggable
              onDragStart={(e) => onDragStart(e, op)}
              onClick={() => selectOperator(op)}
              onDoubleClick={() => selectOperator(null)}>
              <span className="op-info">
                <span className="op-name">{op.name}</span>
                <span className="op-desc">{op.description}</span>
              </span>
              {isCustom && (
                <span className="op-custom-actions">
                  <button className="op-dup-btn" onClick={(e) => { e.stopPropagation(); onDuplicateCustom?.(op) }}
                    title="复制">⧉</button>
                  <button className="op-edit-btn" onClick={(e) => { e.stopPropagation(); onEditCustom?.(op) }}
                    title="编辑">&#9998;</button>
                  <button className="op-del-btn" onClick={(e) => { e.stopPropagation(); onDeleteCustom?.(op.name) }}
                    title="删除">×</button>
                </span>
              )}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

export default function OperatorPanel() {
  const operators = useModelStore((s) => s.operators)
  const customOperators = useModelStore((s) => s.customOperators)
  const addCustomOperator = useModelStore((s) => s.addCustomOperator)
  const removeCustomOperator = useModelStore((s) => s.removeCustomOperator)
  const updateCustomOperator = useModelStore((s) => s.updateCustomOperator)
  const importCustomOperators = useModelStore((s) => s.importCustomOperators)

  const [search, setSearch] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editingOp, setEditingOp] = useState<OperatorDef | undefined>(undefined)

  const isSearching = search.trim().length > 0
  const q = search.toLowerCase().trim()

  const customNames = useMemo(() => new Set(customOperators.map((o) => o.name)), [customOperators])
  const mergedOps = useMemo(() => {
    const existingNames = new Set(operators.map((o) => o.name))
    const uniqueCustoms = customOperators.filter((o) => !existingNames.has(o.name))
    return [...operators, ...uniqueCustoms]
  }, [operators, customOperators])

  const filtered = useMemo(() => {
    const grouped = new Map<string, OperatorDef[]>()
    for (const op of mergedOps) {
      if (customNames.has(op.name)) continue
      const cat = op.category || 'Other'
      if (!grouped.has(cat)) grouped.set(cat, [])
      grouped.get(cat)!.push(op)
    }
    const cats = [...grouped.keys()].sort((a, b) => {
      const ai = CATEGORY_ORDER.indexOf(a)
      const bi = CATEGORY_ORDER.indexOf(b)
      if (ai >= 0 && bi >= 0) return ai - bi
      if (ai >= 0) return -1
      if (bi >= 0) return 1
      return a.localeCompare(b)
    })
    return cats.map((cat) => ({
      name: cat,
      color: CATEGORY_COLORS[cat] || '#9e9e9e',
      ops: grouped.get(cat)!.filter((op) => {
        if (!isSearching) return true
        return op.name.toLowerCase().includes(q) || op.description.toLowerCase().includes(q)
      }),
    })).filter((c) => c.ops.length > 0)
  }, [mergedOps, isSearching, q])

  const customFiltered = useMemo(() => {
    return mergedOps.filter((op) => customNames.has(op.name) && (!isSearching ||
      op.name.toLowerCase().includes(q) || op.description.toLowerCase().includes(q)))
  }, [mergedOps, customNames, isSearching, q])

  async function handleImport() {
    const file = await openFileDialog('.json')
    if (!file) return
    try {
      const text = await file.text()
      const data = JSON.parse(text)
      const ops = Array.isArray(data) ? data : (data.operators || [])
      const count = importCustomOperators(ops)
      if (count === 0) alert('没有新算子可导入（名称重复）')
    } catch { alert('导入失败：JSON 格式无效') }
  }

  function handleExport() {
    if (customOperators.length === 0) { alert('没有自定义算子可导出'); return }
    const json = JSON.stringify(customOperators, null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = 'kepler-custom-operators.json'
    a.click()
  }

  function handleSave(op: OperatorDef) {
    if (editingOp) {
      updateCustomOperator(editingOp.name, op)
    } else {
      const ok = addCustomOperator(op)
      if (!ok) { alert('算子名称已存在'); return }
    }
    setDialogOpen(false)
    setEditingOp(undefined)
  }

  function handleEdit(op: OperatorDef) {
    setEditingOp(op)
    setDialogOpen(true)
  }

  function handleAdd() {
    setEditingOp(undefined)
    setDialogOpen(true)
  }

  function handleDuplicate(op: OperatorDef) {
    const allNames = new Set(mergedOps.map((o) => o.name))
    let base = op.name + '_copy'
    let name = base
    for (let i = 1; allNames.has(name); i++) name = base + i
    addCustomOperator({ ...op, name, inputs: [...op.inputs], params: [...op.params], outputs: [...op.outputs] })
  }

  return (
    <div className="operator-panel">
      <h3>算子库</h3>
      <div className="operator-search-wrap">
        <input
          className="operator-search"
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索算子..."
        />
        {isSearching && (
          <button className="operator-search-clear" onClick={() => setSearch('')}>×</button>
        )}
      </div>
      <div className="operator-list">
        {filtered.map((cat) => (
          <CategoryGroup key={cat.name} name={cat.name} color={cat.color} ops={cat.ops} forceOpen={isSearching} />
        ))}
        <CategoryGroup
          name="自定义"
          color="#d4842a"
          ops={customFiltered}
          forceOpen={isSearching}
          isCustom
          onAddCustom={handleAdd}
          onEditCustom={handleEdit}
          onDeleteCustom={removeCustomOperator}
          onDuplicateCustom={handleDuplicate}
          onImport={handleImport}
          onExport={handleExport}
        />
        {isSearching && filtered.every((c) => c.ops.length === 0) && customFiltered.length === 0 && (
          <div className="op-empty-search">无匹配算子</div>
        )}
      </div>
      {dialogOpen && (
        <CustomOperatorDialog
          initial={editingOp}
          onSave={handleSave}
          onCancel={() => { setDialogOpen(false); setEditingOp(undefined) }}
        />
      )}
    </div>
  )
}
