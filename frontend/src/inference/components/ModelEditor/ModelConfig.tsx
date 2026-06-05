import { useMemo, useState, useCallback, useEffect } from 'react'
import { useModelStore } from '../../stores/model'
import { cn } from '../../utils/classnames'
import { openFileDialog } from '../../utils/file'
import { fetchHfConfigList, fetchHfConfig, fetchModelList, fetchModel } from '../../api/library'
import type { LayerConfig, OperatorDef, OpNodeData, ModuleDef } from '../../types/model'

function RankLabel({ rank, updateRankIndex }: { rank: number; activeRank: number; updateRankIndex: (oldIdx: number, newIdx: number) => boolean }) {
  const [editing, setEditing] = useState(false)
  const [value, setValue] = useState(String(rank))

  const startEdit = (e: React.MouseEvent) => {
    e.stopPropagation()
    setValue(String(rank))
    setEditing(true)
  }

  const commit = () => {
    setEditing(false)
    const n = parseInt(value)
    if (!isNaN(n) && n >= 0 && n !== rank) {
      updateRankIndex(rank, n)
    }
  }

  if (editing) {
    return (
      <input className="rank-edit-input" value={value}
        onChange={(e) => setValue(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => { if (e.key === 'Enter') commit() }}
        onClick={(e) => e.stopPropagation()}
        autoFocus
      />
    )
  }

  return <span className="rank-label" onDoubleClick={startEdit} title="双击编辑编号">Rank {rank}</span>
}

function GlobalRow({ op, arrPos, isSelected, position, rank, itemIdx, selectNode, onRemove, onReorder, isFirst, isLast }: {
  op: { index: number; name: string }; arrPos: number; isSelected: boolean; position: 'top' | 'bottom'
  rank: number; itemIdx: number
  selectNode: (id: string | null) => void; onRemove: () => void
  onReorder: (rank: number, from: number, to: number) => void
  isFirst: boolean; isLast: boolean
}) {
  const sel = String(op.index)
  const badgeClass = position === 'top' ? 'mc-global-badge-top' : 'mc-global-badge-bottom'
  const actionable = itemIdx >= 0

  return (
    <div className={cn('mc-op-row', 'mc-global-row', isSelected && 'mc-op-sel')}
      onClick={() => selectNode(sel)}
      onDoubleClick={() => selectNode(null)}>
      {actionable && (
        <span className="mc-reorder">
          <button className="mc-arrow-btn" disabled={isFirst}
            onClick={(e) => { e.stopPropagation(); onReorder(rank, itemIdx, itemIdx - 1) }}>▲</button>
          <button className="mc-arrow-btn" disabled={isLast}
            onClick={(e) => { e.stopPropagation(); onReorder(rank, itemIdx, itemIdx + 1) }}>▼</button>
        </span>
      )}
      <span className="mc-op-idx">#{arrPos}</span>
      <span className="mc-op-name">{op.name}</span>
      <span className={cn('mc-global-badge', badgeClass)}>{position === 'top' ? '层前' : '层后'}</span>
      <button className="mc-remove-btn" onClick={(e) => { e.stopPropagation(); onRemove() }}>×</button>
    </div>
  )
}

function LayerBlock({ layer, layerIdx, nodes, selOp, selModule, onAddModule, isFirst, isLast, indexMap, selectedNodeId, selectNode, activeRank }: {
  layer: LayerConfig; layerIdx: number; nodes: OpNodeData[]; selOp: OperatorDef | null
  selModule: ModuleDef | null; onAddModule: (layerId: string) => void
  isFirst: boolean; isLast: boolean; indexMap: Map<number, number>
  selectedNodeId: string | null; selectNode: (id: string | null) => void; activeRank: number
}) {
  const removeLayer = useModelStore((s) => s.removeLayer)
  const duplicateLayer = useModelStore((s) => s.duplicateLayer)
  const setRepeat = useModelStore((s) => s.setLayerRepeat)
  const moveUp = useModelStore((s) => s.moveLayerUp)
  const moveDown = useModelStore((s) => s.moveLayerDown)
  const setLayerIndex = useModelStore((s) => s.setLayerIndex)
  const removeOp = useModelStore((s) => s.removeOpFromLayer)
  const addNodeToLayer = useModelStore((s) => s.addNodeToLayer)
  const reorderLayerOps = useModelStore((s) => s.reorderLayerOps)

  const rankOpIndices = layer.rankOps[activeRank] || []
  const layerOps = nodes.filter((n) => rankOpIndices.includes(n.index))
  const [collapsed, setCollapsed] = useState(false)
  const [editingIdx, setEditingIdx] = useState(false)
  const [idxValue, setIdxValue] = useState(String(layer.layerIdx))
  const hasOps = layerOps.length > 0

  const isMtp = layer.kind === 'mtp'

  const commitIdx = () => {
    setEditingIdx(false)
    const n = parseInt(idxValue)
    if (!isNaN(n) && n >= 0) setLayerIndex(layer.id, n)
    else setIdxValue(String(layer.layerIdx))
  }

  return (
    <div className="mc-layer">
      <div className={cn('mc-layer-header', layer.kind === 'mtp' && 'mc-mtp-header')}>
        <span className="mc-reorder">
          <button className="mc-arrow-btn" disabled={isFirst} onClick={() => moveUp(layer.id)}>▲</button>
          <button className="mc-arrow-btn" disabled={isLast} onClick={() => moveDown(layer.id)}>▼</button>
        </span>
        {isMtp ? (
          <span className="mc-layer-idx mc-layer-idx-fixed">{layer.layerIdx}</span>
        ) : editingIdx ? (
          <input className="mc-layer-idx-input" value={idxValue}
            onChange={(e) => setIdxValue(e.target.value)}
            onBlur={commitIdx}
            onKeyDown={(e) => { if (e.key === 'Enter') commitIdx() }}
            autoFocus />
        ) : (
          <span className="mc-layer-idx" onDoubleClick={() => { setIdxValue(String(layerIdx)); setEditingIdx(true) }}
            title="双击编辑序号">{layerIdx}</span>
        )}
        <span className="mc-layer-name-wrap">
          <span className="mc-layer-name mc-layer-name-fixed">{isMtp ? `mtp_${layer.layerIdx}` : `Layer_${layer.layerIdx}`}</span>
          <span className="mc-repeat">
            ×<input type="number" value={layer.repeat} min={1}
              disabled={isMtp}
              onChange={(e) => setRepeat(layer.id, parseInt(e.target.value) || 1)} />
          </span>
        </span>
        {layer.kind === 'mtp' && <span className="mc-kind-badge">MTP</span>}
        {collapsed && <span className="mc-collapsed-count">{layerOps.length} ops</span>}
        <span className="mc-spacer" />
        {selModule && (
          <button className="mc-add-op-btn" onClick={() => onAddModule(layer.id)}>
            + {selModule.label}
          </button>
        )}
        {!selModule && selOp && (
          <button className="mc-add-op-btn" onClick={() => addNodeToLayer(layer.id, selOp)}>
            + {selOp.name}
          </button>
        )}
        <button className="mc-dup-btn" onClick={() => duplicateLayer(layer.id)} title="复制层">⧉</button>
        {hasOps && (
          <button className={cn('mc-collapse-btn', collapsed && 'collapsed')}
            onClick={() => setCollapsed(!collapsed)} title={collapsed ? '展开' : '折叠'}>
            &#9654;
          </button>
        )}
        <button className="mc-remove-btn" onClick={() => removeLayer(layer.id)}>×</button>
      </div>
      {!collapsed && (
      <div className="mc-layer-ops">
        {layerOps.map((op, itemIdx, arr) => {
          const arrPos = indexMap.get(op.index) ?? op.index
          const selId = String(op.index)
          const isSel = selectedNodeId === selId
          const desc = op.operator?.description || op.operator?.module || ''
          const isFirst = itemIdx === 0
          const isLast = itemIdx === arr.length - 1
          return (
            <div key={op.index}
              className={cn('mc-op-row', isSel && 'mc-op-sel')}
              onClick={() => selectNode(selId)}
              onDoubleClick={() => selectNode(null)}>
              <span className="mc-reorder">
                <button className="mc-arrow-btn" disabled={isFirst}
                  onClick={(e) => { e.stopPropagation(); reorderLayerOps(layer.id, activeRank, itemIdx, itemIdx - 1) }}>▲</button>
                <button className="mc-arrow-btn" disabled={isLast}
                  onClick={(e) => { e.stopPropagation(); reorderLayerOps(layer.id, activeRank, itemIdx, itemIdx + 1) }}>▼</button>
              </span>
              <span className="mc-op-idx">#{arrPos}</span>
              <div className="mc-op-info">
                <span className="mc-op-name">{op.operator?.name || ''}</span>
                {desc && <span className="mc-op-desc">{desc}</span>}
              </div>
              <button className="mc-remove-btn" onClick={(e) => { e.stopPropagation(); removeOp(layer.id, op.index) }}>×</button>
            </div>
          )
        })}
        {layerOps.length === 0 && <div className="mc-empty">选中一个算子或 module 后，点击 + 按钮</div>}
      </div>
      )}
    </div>
  )
}

export default function ModelConfig({ activeTab, onSwitchTab }: { activeTab: 'editor' | 'json'; onSwitchTab: (tab: 'editor' | 'json') => void }) {
  const nodes = useModelStore((s) => s.nodes)
  const layers = useModelStore((s) => s.layers)
  const topGlobalIndices = useModelStore((s) => s.topGlobalIndices)
  const bottomGlobalIndices = useModelStore((s) => s.bottomGlobalIndices)
  const addLayer = useModelStore((s) => s.addLayer)
  const addMtpLayer = useModelStore((s) => s.addMtpLayer)
  const addTopGlobal = useModelStore((s) => s.addTopGlobal)
  const addBottomGlobal = useModelStore((s) => s.addBottomGlobal)
  const removeTopGlobal = useModelStore((s) => s.removeTopGlobal)
  const removeBottomGlobal = useModelStore((s) => s.removeBottomGlobal)
  const reorderTopGlobals = useModelStore((s) => s.reorderTopGlobals)
  const reorderBottomGlobals = useModelStore((s) => s.reorderBottomGlobals)
  const addOpToLayer = useModelStore((s) => s.addOpToLayer)
  const addNodeToLayer = useModelStore((s) => s.addNodeToLayer)
  const selectedNodeId = useModelStore((s) => s.selectedNodeId)
  const selectNode = useModelStore((s) => s.selectNode)
  const selectedOperator = useModelStore((s) => s.selectedOperator)
  const selectedModule = useModelStore((s) => s.selectedModule)
  const moduleDefs = useModelStore((s) => s.moduleDefs)
  const operators = useModelStore((s) => s.operators)
  const customOperators = useModelStore((s) => s.customOperators)
  const modelName = useModelStore((s) => s.modelName)
  const setModelName = useModelStore((s) => s.setModelName)
  const exportModel = useModelStore((s) => s.exportModel)
  const clearNodes = useModelStore((s) => s.clearNodes)
  const importFromJSON = useModelStore((s) => s.importFromJSON)
  const ranks = useModelStore((s) => s.ranks)
  const activeRank = useModelStore((s) => s.activeRank)
  const addRank = useModelStore((s) => s.addRank)
  const duplicateRank = useModelStore((s) => s.duplicateRank)
  const removeRank = useModelStore((s) => s.removeRank)
  const setActiveRank = useModelStore((s) => s.setActiveRank)
  const updateRankIndex = useModelStore((s) => s.updateRankIndex)
  const hfConfigText = useModelStore((s) => s.hfConfigText)
  const setHfConfigText = useModelStore((s) => s.setHfConfigText)
  const [jsonText, setJsonText] = useState(hfConfigText)

  // sync JSON tab content when hfConfigText changes from editor tab (e.g. built-in model load)
  useEffect(() => { setJsonText(hfConfigText) }, [hfConfigText])

  const [builtinHfConfigs, setBuiltinHfConfigs] = useState<string[]>([])
  const [selectedBuiltin, setSelectedBuiltin] = useState('')
  const [builtinModels, setBuiltinModels] = useState<string[]>([])
  const [selectedBuiltinModel, setSelectedBuiltinModel] = useState('')

  useEffect(() => {
    fetchHfConfigList().then(setBuiltinHfConfigs).catch(() => {})
  }, [])

  useEffect(() => {
    fetchModelList().then(setBuiltinModels).catch(() => {})
  }, [])

  async function handleSelectBuiltin(name: string) {
    setSelectedBuiltin(name)
    if (!name) return
    try {
      const data = await fetchHfConfig(name)
      const text = JSON.stringify(data, null, 2)
      setJsonText(text); setHfConfigText(text)
    } catch { alert('加载模型失败') }
  }

  async function handleLoadBuiltinModel() {
    if (!selectedBuiltinModel) return
    try {
      const data = await fetchModel(selectedBuiltinModel)
      importFromJSON(data)
      if (data.name) setModelName(data.name)
      try {
        const hfConfig = await fetchHfConfig(selectedBuiltinModel)
        const text = JSON.stringify(hfConfig, null, 2)
        setHfConfigText(text)
        setSelectedBuiltin(selectedBuiltinModel)
      } catch { /* no matching HF config */ }
      onSwitchTab('editor')
    } catch { alert('加载内置模型失败') }
  }

  async function handleImportLocalModel() {
    const file = await openFileDialog('.json')
    if (!file) return
    try {
      const data = JSON.parse(await file.text())
      importFromJSON(data)
      if (data.name) setModelName(data.name)
      onSwitchTab('editor')
    } catch { alert('JSON 格式无效') }
  }

  function handleExportModel() {
    const name = modelName || 'custom-model'
    const json = JSON.stringify(exportModel(name), null, 2)
    const blob = new Blob([json], { type: 'application/json' })
    const a = document.createElement('a')
    a.href = URL.createObjectURL(blob)
    a.download = `${name}.json`
    a.click()
  }

  async function handleImportJSON() {
    const file = await openFileDialog('.json')
    if (!file) return
    try {
      const text = await file.text()
      JSON.parse(text) // validate
      setJsonText(text); setHfConfigText(text)
    } catch { alert('JSON 格式无效') }
  }

  const { nodeList, topGlobals, bottomGlobals } = useMemo(() => {
    const list = nodes.map((n) => ({ index: n.index, name: n.operator.name, rank: n.rank }))
    const layerSet = new Set(layers.flatMap((l) => Object.values(l.rankOps).flat()))
    const isBuiltin = (n: typeof list[number]) => n.name === 'start' || n.name === 'end'
    const topRaw = list.filter((n) => topGlobalIndices.includes(n.index) && !layerSet.has(n.index) && !isBuiltin(n))
    const bottomRaw = list.filter((n) => bottomGlobalIndices.includes(n.index) && !layerSet.has(n.index) && !isBuiltin(n))
    const regLayers = layers.filter((l) => l.kind !== 'mtp')
    const mtpOnly = layers.filter((l) => l.kind === 'mtp')
    // build logically ordered list: by rank (top → regular layers → bottom → MTP)
    const middle: { index: number; name: string; rank: number }[] = []
    for (const rank of ranks) {
      for (const n of topRaw) { if (n.rank === rank) middle.push(n) }
      for (const l of regLayers) {
        for (const oi of (l.rankOps[rank] || [])) {
          const n = list.find((x) => x.index === oi)
          if (n) middle.push(n)
        }
      }
      for (const n of bottomRaw) { if (n.rank === rank) middle.push(n) }
      for (const l of mtpOnly) {
        for (const oi of (l.rankOps[rank] || [])) {
          const n = list.find((x) => x.index === oi)
          if (n) middle.push(n)
        }
      }
    }
    const startNode = list.find((n) => n.name === 'start')
    const endNode = list.find((n) => n.name === 'end')
    const ordered: { index: number; name: string; rank: number }[] = []
    if (startNode) ordered.push(startNode)
    ordered.push(...middle)
    if (endNode) ordered.push(endNode)
    // include builtins in display lists so they render as global rows
    const topDisplay = startNode ? [startNode, ...topRaw] : topRaw
    const bottomDisplay = endNode ? [...bottomRaw, endNode] : bottomRaw
    return {
      nodeList: ordered,
      topGlobals: topDisplay,
      bottomGlobals: bottomDisplay,
    }
  }, [nodes, layers, topGlobalIndices, bottomGlobalIndices, ranks])

  // resolve operators from selected module (built-in or custom)
  const moduleOps = useMemo((): OperatorDef[] => {
    if (!selectedModule) return []
    if (selectedModule.isBuiltin) return moduleDefs[selectedModule.name] || []
    const allOps = [...operators, ...customOperators]
    return selectedModule.operatorNames
      .map((n) => allOps.find((o) => o.name === n))
      .filter(Boolean) as OperatorDef[]
  }, [selectedModule, moduleDefs, operators, customOperators])

  const topLabel = selectedModule ? `+ 层前 ${selectedModule.label}` : selectedOperator ? `+ 层前 ${selectedOperator.name}` : '+ 层前'
  const bottomLabel = selectedModule ? `+ 层后 ${selectedModule.label}` : selectedOperator ? `+ 层后 ${selectedOperator.name}` : '+ 层后'

  const handleAddTop = useCallback(() => {
    if (selectedModule) {
      moduleOps.forEach((op) => ranks.forEach((r) => addTopGlobal(op, r)))
    } else if (selectedOperator) {
      ranks.forEach((r) => addTopGlobal(selectedOperator, r))
    }
  }, [selectedModule, selectedOperator, addTopGlobal, ranks, moduleOps])
  const handleAddBottom = useCallback(() => {
    if (selectedModule) {
      moduleOps.forEach((op) => ranks.forEach((r) => addBottomGlobal(op, r)))
    } else if (selectedOperator) {
      ranks.forEach((r) => addBottomGlobal(selectedOperator, r))
    }
  }, [selectedModule, selectedOperator, addBottomGlobal, ranks, moduleOps])
  const handleAddLayer = useCallback(() => addLayer(), [addLayer])
  const handleAddMtpLayer = useCallback(() => addMtpLayer(), [addMtpLayer])

  const addModuleToLayer = useCallback((layerId: string) => {
    if (selectedModule) moduleOps.forEach((op) => addNodeToLayer(layerId, op))
  }, [selectedModule, moduleOps, addNodeToLayer])

  const regularLayers = layers.filter((l) => l.kind !== 'mtp')
  const mtpLayers = layers.filter((l) => l.kind === 'mtp')

  const indexMap = useMemo(() => {
    const map = new Map<number, number>()
    for (let i = 0; i < nodeList.length; i++) map.set(nodeList[i].index, i)
    return map
  }, [nodeList])

  const assignTargetNode = useMemo(() => {
    if (!selectedNodeId || selectedOperator || layers.length === 0) return null
    return nodes.find((n) => String(n.index) === selectedNodeId) || null
  }, [selectedNodeId, selectedOperator, layers, nodes])

  return (
    <div className="config-panel">
      {activeTab === 'editor' && (
      <>
      <div className="mc-import-export-bar">
        <select className="mc-builtin-select" value={selectedBuiltinModel}
          onChange={(e) => setSelectedBuiltinModel(e.target.value)}>
          <option value="">内置模型</option>
          {builtinModels.map((m) => (
            <option key={m} value={m}>{m}</option>
          ))}
        </select>
        <button className="btn-sm" onClick={handleLoadBuiltinModel}
          disabled={!selectedBuiltinModel}>加载</button>
        <button className="btn-sm" onClick={handleImportLocalModel}>导入本地模型</button>
        <button className="btn-sm" onClick={handleExportModel}>导出模型</button>
      </div>
      <label className="mc-model-label">
        <span className="mc-model-label-text">模型名称</span>
        <input className="mc-model-name" value={modelName}
          onChange={(e) => setModelName(e.target.value)}
          placeholder="输入模型名称" />
      </label>
      <div className="mc-three-bar">
        <button className="mc-bar-btn" disabled={!selectedOperator && !selectedModule}
          onClick={handleAddTop}>{topLabel}</button>
        <button className="mc-bar-btn primary" onClick={handleAddLayer}>+ 新建层</button>
        <button className="mc-bar-btn" disabled={!selectedOperator && !selectedModule}
          onClick={handleAddBottom}>{bottomLabel}</button>
        <button className="mc-bar-btn mc-mtp-btn" onClick={handleAddMtpLayer}
          disabled={mtpLayers.length >= 1}>+ MTP层</button>
        <button className="mc-bar-btn mc-clear-btn" onClick={clearNodes}>清空</button>
      </div>

      <div className="rank-tab-bar">
        {ranks.map((r) => (
          <button key={r} className={cn('rank-tab', activeRank === r && 'active')}
            onClick={() => setActiveRank(r)}>
            <RankLabel rank={r} activeRank={activeRank} updateRankIndex={updateRankIndex} />
            <span className="rank-tab-dup"
              onClick={(e) => { e.stopPropagation(); duplicateRank(r) }} title="复制 Rank">⧉</span>
            {r > 0 && <span className="rank-tab-remove"
              onClick={(e) => { e.stopPropagation(); removeRank(r) }}>×</span>}
          </button>
        ))}
        <button className="rank-tab-add" onClick={addRank}>+ Rank</button>
      </div>

      <div className="config-body">
        {layers.length === 0 && topGlobals.length === 0 && bottomGlobals.length === 0 ? (
          <div className="mc-empty-state">
            <p>点击「+ 新建层」添加层级结构</p>
            <p>或在算子库中选择算子后，点击「+ 层前/层后」添加全局算子</p>
          </div>
        ) : (
          <>
            {(() => {
              const filtered = topGlobals.filter(op => op.rank === activeRank)
              const nonBuiltin = filtered.filter(op => op.name !== 'start' && op.name !== 'end')
              let idx = 0
              return filtered.map((op) => {
                const pos = indexMap.get(op.index) ?? op.index
                const isBuiltin = op.name === 'start' || op.name === 'end'
                const itemIdx = isBuiltin ? -1 : idx++
                const isFirst = itemIdx === 0
                const isLast = itemIdx === nonBuiltin.length - 1
                return <GlobalRow key={`t${op.index}`} op={op} arrPos={pos} isSelected={selectedNodeId === String(op.index)} position="top" rank={activeRank} itemIdx={itemIdx} selectNode={selectNode} onRemove={() => removeTopGlobal(op.index)} onReorder={reorderTopGlobals} isFirst={isFirst} isLast={isLast} />
              })
            })()}

            {topGlobals.some(op => op.rank === activeRank) && layers.length > 0 && <div className="mc-section-divider" />}

            {regularLayers.map((layer, i, arr) => (
              <div key={layer.id}>
                <LayerBlock layer={layer} layerIdx={layer.layerIdx} nodes={nodes} selOp={selectedOperator} selModule={selectedModule} onAddModule={addModuleToLayer} isFirst={i === 0} isLast={i === arr.length - 1} indexMap={indexMap} selectedNodeId={selectedNodeId} selectNode={selectNode} activeRank={activeRank} />
              </div>
            ))}

            {bottomGlobals.some(op => op.rank === activeRank && op.name !== 'end') && layers.length > 0 && <div className="mc-section-divider" />}

            {(() => {
              const filtered = bottomGlobals.filter(op => op.rank === activeRank && op.name !== 'end')
              let idx = 0
              return filtered.map((op, i, arr) => {
                const pos = indexMap.get(op.index) ?? op.index
                const itemIdx = idx++
                const isFirst = i === 0
                const isLast = i === arr.length - 1
                return <GlobalRow key={`b${op.index}`} op={op} arrPos={pos} isSelected={selectedNodeId === String(op.index)} position="bottom" rank={activeRank} itemIdx={itemIdx} selectNode={selectNode} onRemove={() => removeBottomGlobal(op.index)} onReorder={reorderBottomGlobals} isFirst={isFirst} isLast={isLast} />
              })
            })()}

            {mtpLayers.length > 0 && <div className="mc-section-divider mc-mtp-divider" />}

            {mtpLayers.map((layer, i, arr) => (
              <div key={layer.id}>
                <LayerBlock layer={layer} layerIdx={layer.layerIdx} nodes={nodes} selOp={selectedOperator} selModule={selectedModule} onAddModule={addModuleToLayer} isFirst={i === 0} isLast={i === arr.length - 1} indexMap={indexMap} selectedNodeId={selectedNodeId} selectNode={selectNode} activeRank={activeRank} />
              </div>
            ))}

            {(() => {
              const endNode = bottomGlobals.find(op => op.rank === activeRank && op.name === 'end')
              if (!endNode) return null
              const pos = indexMap.get(endNode.index) ?? endNode.index
              return (
                <>
                  {(regularLayers.length > 0 || mtpLayers.length > 0) && <div className="mc-section-divider" />}
                  <GlobalRow key={`b${endNode.index}`} op={endNode} arrPos={pos} isSelected={selectedNodeId === String(endNode.index)} position="bottom" rank={activeRank} itemIdx={-1} selectNode={selectNode} onRemove={() => removeBottomGlobal(endNode.index)} onReorder={reorderBottomGlobals} isFirst={true} isLast={true} />
                </>
              )
            })()}

            {assignTargetNode && (
            <div className="mc-assign">
              <span className="section-title">分配至</span>
              {layers.map((l) => (
                <button key={l.id} className="mc-assign-btn"
                  onClick={() => addOpToLayer(l.id, assignTargetNode.index)}>
                  {l.name}
                </button>
              ))}
            </div>
          )}
          </>
        )}
      </div>
      </>
      )}

      {activeTab === 'json' && (
        <>
          <div className="mc-json-actions">
            <select className="mc-builtin-select" value={selectedBuiltin}
              onChange={(e) => handleSelectBuiltin(e.target.value)}>
              <option value="">内置 HF config</option>
              {builtinHfConfigs.map((m) => (
                <option key={m} value={m}>{m}</option>
              ))}
            </select>
            <button className="btn-sm" onClick={handleImportJSON}>导入 config.json</button>
          </div>
          <div className="config-body">
            <textarea className="mc-json-editor" value={jsonText}
              onChange={(e) => { setJsonText(e.target.value); setHfConfigText(e.target.value) }} spellCheck={false} />
          </div>
        </>
      )}
    </div>
  )
}
