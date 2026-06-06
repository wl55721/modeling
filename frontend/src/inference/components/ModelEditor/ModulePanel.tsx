import { useState, useCallback, useMemo } from 'react'
import { useModelStore } from '../../stores/model'
import { cn } from '../../utils/classnames'
import type { ModuleDef, OperatorDef } from '../../types/model'
import CustomModuleDialog from './CustomModuleDialog'

const MODULE_LABELS: Record<string, string> = {
  embedding: 'Embedding',
  lmhead: 'LM Head',
  mla_indexer_compressor: 'MLA Indexer & Compressor',
  mla_kv_compressor: 'MLA KV Compressor',
  mla_prolog_epilog_v4: 'MLA Prolog & Epilog V4',
  mlp: 'MLP',
  moe_dispatch_combine: 'MoE Dispatch & Combine',
  moe_routed_expert: 'MoE Routed Expert',
  moe_shared_expert: 'MoE Shared Expert',
  mtp: 'MTP',
}

function ModuleItem({ mod, isCustom, selected, onEdit, onDelete, getOps, onSelectOp, onSelect, onDoubleClick }: {
  mod: ModuleDef
  isCustom?: boolean
  selected?: boolean
  onEdit?: (mod: ModuleDef) => void
  onDelete?: (name: string) => void
  getOps?: () => OperatorDef[]
  onSelectOp: (op: OperatorDef) => void
  onSelect: (mod: ModuleDef) => void
  onDoubleClick: () => void
}) {
  const [open, setOpen] = useState(false)
  const ops = getOps?.() ?? []

  const onDragStart = useCallback((event: React.DragEvent) => {
    event.dataTransfer.setData('application/kepler-module', mod.name)
    event.dataTransfer.effectAllowed = 'move'
  }, [mod.name])

  return (
    <div className="mod-item-wrap">
      <div className={cn('operator-item module-item', selected && 'op-selected')}
        draggable
        onDragStart={onDragStart}
        onClick={() => { onSelect(mod); setOpen(!open) }}
        onDoubleClick={(e) => { e.stopPropagation(); onDoubleClick() }}>
        <span className={cn('op-cat-arrow', open && 'open')}>&#9654;</span>
        <span className="op-info">
          <span className="op-name">{mod.label}</span>
        </span>
        {isCustom && (
          <span className="op-custom-actions">
            <button className="op-edit-btn" onClick={(e) => { e.stopPropagation(); onEdit?.(mod) }}
              title="编辑">&#9998;</button>
            <button className="op-del-btn" onClick={(e) => { e.stopPropagation(); onDelete?.(mod.name) }}
              title="删除">×</button>
          </span>
        )}
      </div>
      {open && (
        <div className="mod-op-list">
          {ops.map((op, i) => (
            <div key={`${op.name}-${i}`} className="mod-op-item"
              onClick={(e) => { e.stopPropagation(); onSelectOp(op) }}>
              {op.name}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ModuleGroup({ modules, isCustom, onAdd, onEdit, onDelete, moduleDefs, onSelectOp, selectedModule, onSelect, onDoubleClick }: {
  modules: ModuleDef[]
  isCustom?: boolean
  onAdd?: () => void
  onEdit?: (mod: ModuleDef) => void
  onDelete?: (name: string) => void
  moduleDefs?: Record<string, OperatorDef[]>
  onSelectOp: (op: OperatorDef) => void
  selectedModule: ModuleDef | null
  onSelect: (mod: ModuleDef) => void
  onDoubleClick: () => void
}) {
  const [open, setOpen] = useState(!isCustom)

  if (modules.length === 0 && !isCustom) return null
  return (
    <div className="op-category">
      <div className="op-cat-header" onClick={() => setOpen(!open)}>
        <span className={cn('op-cat-arrow', open && 'open')}>&#9654;</span>
        <span className="op-cat-name">{isCustom ? '自定义MODULE' : '内置module'}</span>
        <span className="op-cat-count">{modules.length}</span>
        {isCustom && (
          <button className="op-cat-add" onClick={(e) => { e.stopPropagation(); onAdd?.() }}
            title="新建自定义MODULE">+</button>
        )}
      </div>
      {open && (
        <div className="op-cat-items">
          {modules.length === 0 && isCustom && (
            <div className="op-empty-hint">点击 + 新建自定义MODULE</div>
          )}
          {modules.map((mod) => (
            <ModuleItem key={mod.name} mod={mod} isCustom={isCustom}
              selected={selectedModule?.name === mod.name}
              onEdit={onEdit} onDelete={onDelete} onSelectOp={onSelectOp}
              onSelect={onSelect} onDoubleClick={onDoubleClick}
              getOps={moduleDefs && mod.isBuiltin
                ? () => moduleDefs[mod.name] || []
                : undefined} />
          ))}
        </div>
      )}
    </div>
  )
}

export default function ModulePanel() {
  const moduleList = useModelStore((s) => s.moduleList)
  const moduleDefs = useModelStore((s) => s.moduleDefs)
  const customModules = useModelStore((s) => s.customModules)
  const addCustomModule = useModelStore((s) => s.addCustomModule)
  const removeCustomModule = useModelStore((s) => s.removeCustomModule)
  const updateCustomModule = useModelStore((s) => s.updateCustomModule)
  const selectOperator = useModelStore((s) => s.selectOperator)
  const selectedModule = useModelStore((s) => s.selectedModule)
  const selectModule = useModelStore((s) => s.selectModule)

  const [search, setSearch] = useState('')
  const [dialogOpen, setDialogOpen] = useState(false)
  const [editingMod, setEditingMod] = useState<ModuleDef | undefined>(undefined)

  const isSearching = search.trim().length > 0
  const q = search.toLowerCase().trim()

  const builtinModules = useMemo(() => {
    return moduleList
      .map((name): ModuleDef => ({
        name,
        label: MODULE_LABELS[name] || name,
        operatorNames: (moduleDefs[name] || []).map((op: OperatorDef) => op.name),
        isBuiltin: true,
      }))
      .filter((m) => m.operatorNames.length > 0)
  }, [moduleList, moduleDefs])

  const filteredBuiltin = useMemo(() => {
    if (!isSearching) return builtinModules
    return builtinModules.filter((m) => m.label.toLowerCase().includes(q))
  }, [builtinModules, isSearching, q])

  const filteredCustom = useMemo(() => {
    if (!isSearching) return customModules
    return customModules.filter((m) => m.label.toLowerCase().includes(q))
  }, [customModules, isSearching, q])

  function handleSave(mod: ModuleDef) {
    if (editingMod) {
      updateCustomModule(editingMod.name, mod)
    } else {
      const ok = addCustomModule(mod)
      if (!ok) { alert('module名称已存在'); return }
    }
    setDialogOpen(false)
    setEditingMod(undefined)
  }

  function handleEdit(mod: ModuleDef) {
    setEditingMod(mod)
    setDialogOpen(true)
  }

  function handleAdd() {
    setEditingMod(undefined)
    setDialogOpen(true)
  }

  return (
    <div className="module-panel">
      <h3>Module库（算子集）</h3>
      <div className="operator-search-wrap">
        <input
          className="operator-search"
          type="text"
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索模块..."
        />
        {isSearching && (
          <button className="operator-search-clear" onClick={() => setSearch('')}>×</button>
        )}
      </div>
      <div className="operator-list">
        <ModuleGroup modules={filteredBuiltin} moduleDefs={moduleDefs}
          onSelectOp={selectOperator}
          selectedModule={selectedModule}
          onSelect={selectModule}
          onDoubleClick={() => selectModule(null)} />
        <ModuleGroup
          modules={filteredCustom}
          isCustom
          onAdd={handleAdd}
          onEdit={handleEdit}
          onDelete={removeCustomModule}
          onSelectOp={selectOperator}
          selectedModule={selectedModule}
          onSelect={selectModule}
          onDoubleClick={() => selectModule(null)}
        />
        {isSearching && filteredBuiltin.length === 0 && filteredCustom.length === 0 && (
          <div className="op-empty-search">无匹配模块</div>
        )}
      </div>
      {dialogOpen && (
        <CustomModuleDialog
          initial={editingMod}
          onSave={handleSave}
          onCancel={() => { setDialogOpen(false); setEditingMod(undefined) }}
        />
      )}
    </div>
  )
}
