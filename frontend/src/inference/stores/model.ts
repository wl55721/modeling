import { create } from 'zustand'
import type { OperatorDef, OpNodeData, LayerConfig, TensorMeta, ModuleDef } from '../types/model'
import * as api from '../api/library'

let _nextId = 1

const START_OPERATOR: OperatorDef = {
  name: 'start', module: '__builtin__', description: '模型入口',
  inputs: [], params: [], outputs: [{ name: 'output', shape: '-', dtype: '-' }], compute_flops: '0',
}
const END_OPERATOR: OperatorDef = {
  name: 'end', module: '__builtin__', description: '模型出口',
  inputs: [{ name: 'input', shape: '-', dtype: '-' }], params: [], outputs: [], compute_flops: '0',
}
const START_INDEX = _nextId++
const END_INDEX = _nextId++
export const BUILTIN_INDICES = new Set([START_INDEX, END_INDEX])

function _makeBuiltinNodes(): OpNodeData[] {
  return [
    { index: START_INDEX, operator: { ...START_OPERATOR }, label: 'start', rank: 0, position: { x: 0, y: 0 } },
    { index: END_INDEX, operator: { ...END_OPERATOR }, label: 'end', rank: 0, position: { x: 0, y: 300 } },
  ]
}

export interface EdgeData { id: string; source: string; target: string }

interface ModelState {
  operators: OperatorDef[]
  nodes: OpNodeData[]
  layers: LayerConfig[]
  edges: EdgeData[]
  disconnected: Set<string>
  selectedNodeId: string | null
  selectedOperator: OperatorDef | null
  selectedModule: ModuleDef | null
  modelName: string
  setModelName: (name: string) => void
  hfConfigText: string
  setHfConfigText: (text: string) => void
  topGlobalIndices: number[]
  bottomGlobalIndices: number[]
  operatorList: string[]
  customOperators: OperatorDef[]
  customModules: ModuleDef[]
  moduleList: string[]
  moduleDefs: Record<string, OperatorDef[]>

  addCustomOperator: (op: OperatorDef) => boolean
  removeCustomOperator: (name: string) => void
  updateCustomOperator: (oldName: string, op: OperatorDef) => void
  importCustomOperators: (ops: OperatorDef[]) => number

  addCustomModule: (mod: ModuleDef) => boolean
  removeCustomModule: (name: string) => void
  updateCustomModule: (oldName: string, mod: ModuleDef) => void

  loadOperatorList: () => Promise<void>
  loadModuleList: () => Promise<void>
  addNode: (operator: OperatorDef, pos?: { x: number; y: number }, rank?: number) => void
  removeNode: (index: number) => void
  selectNode: (id: string | null) => void
  updateNodeSection: (index: number, section: string, data: any) => void
  moveNode: (index: number, pos: { x: number; y: number }) => void
  addLayer: (kind?: 'regular' | 'mtp') => void
  addMtpLayer: () => void
  duplicateLayer: (id: string) => void
  removeLayer: (id: string) => void
  setLayerRepeat: (id: string, repeat: number) => void
  addOpToLayer: (layerId: string, opIndex: number) => void
  addNodeToLayer: (layerId: string, op: OperatorDef) => void
  removeOpFromLayer: (layerId: string, opIndex: number) => void
  renameLayer: (id: string, name: string) => void
  moveLayerUp: (id: string) => void
  moveLayerDown: (id: string) => void
  setLayerIndex: (id: string, newIdx: number) => void
  selectOperator: (op: OperatorDef | null) => void
  selectModule: (mod: ModuleDef | null) => void
  addTopGlobal: (op: OperatorDef, rank?: number) => void
  addBottomGlobal: (op: OperatorDef, rank?: number) => void
  removeTopGlobal: (index: number) => void
  removeBottomGlobal: (index: number) => void
  reorderTopGlobals: (rank: number, fromIdx: number, toIdx: number) => void
  reorderBottomGlobals: (rank: number, fromIdx: number, toIdx: number) => void
  reorderLayerOps: (layerId: string, rank: number, fromIdx: number, toIdx: number) => void
  addEdge: (source: string, target: string) => boolean
  removeEdge: (edgeId: string) => void
  wouldCreateCycle: (source: string, target: string) => boolean
  exportModel: (name: string) => object
  importFromJSON: (json: object) => void
  reorderCount: number
  clearNodes: () => void
  ranks: number[]
  activeRank: number
  addRank: () => void
  duplicateRank: (sourceRank: number) => void
  removeRank: (rank: number) => void
  setActiveRank: (rank: number) => void
  updateRankIndex: (oldIdx: number, newIdx: number) => boolean
  reorderNodes: () => void
}

function _renumberMtp(layers: LayerConfig[]): LayerConfig[] {
  let idx = 980
  return layers.map((l) => {
    if (l.kind !== 'mtp') return l
    const n = idx++
    return { ...l, layerIdx: n, name: `mtp_${n}` }
  })
}

function _reorderNodes(s: ModelState, newLayers: LayerConfig[]) {
  const topByRank = new Map<number, OpNodeData[]>()
  const bottomByRank = new Map<number, OpNodeData[]>()
  for (const idx of s.topGlobalIndices) {
    const n = s.nodes.find((n) => n.index === idx)
    if (n) { const r = n.rank; if (!topByRank.has(r)) topByRank.set(r, []); topByRank.get(r)!.push(n) }
  }
  for (const idx of s.bottomGlobalIndices) {
    const n = s.nodes.find((n) => n.index === idx)
    if (n) { const r = n.rank; if (!bottomByRank.has(r)) bottomByRank.set(r, []); bottomByRank.get(r)!.push(n) }
  }
  const rankSet = new Set<number>()
  for (const r of topByRank.keys()) rankSet.add(r)
  for (const r of bottomByRank.keys()) rankSet.add(r)
  for (const l of newLayers) for (const r of Object.keys(l.rankOps)) rankSet.add(Number(r))
  const allRanks = [...rankSet].sort((a, b) => a - b)
  const regularLayers = newLayers.filter((l) => l.kind !== 'mtp')
  const mtpLayers = newLayers.filter((l) => l.kind === 'mtp')
  const middle: OpNodeData[] = []
  for (const rank of allRanks) {
    for (const n of topByRank.get(rank) || []) if (!BUILTIN_INDICES.has(n.index)) middle.push(n)
    for (const l of regularLayers) {
      for (const i of (l.rankOps[rank] || [])) {
        const n = s.nodes.find((n) => n.index === i)
        if (n) middle.push(n)
      }
    }
    for (const n of bottomByRank.get(rank) || []) if (!BUILTIN_INDICES.has(n.index)) middle.push(n)
    for (const l of mtpLayers) {
      for (const i of (l.rankOps[rank] || [])) {
        const n = s.nodes.find((n) => n.index === i)
        if (n) middle.push(n)
      }
    }
  }
  // collect any unassigned nodes (not in top/bottom and not in any layer)
  const assigned = new Set(middle.map((n) => n.index))
  for (const n of s.nodes) {
    if (!assigned.has(n.index) && !BUILTIN_INDICES.has(n.index)) middle.push(n)
  }
  // ensure start is first, end is last
  const startNode = s.nodes.find((n) => n.index === START_INDEX)
  const endNode = s.nodes.find((n) => n.index === END_INDEX)
  const result: OpNodeData[] = []
  if (startNode) result.push(startNode)
  result.push(...middle)
  if (endNode) result.push(endNode)
  return result
}

function _setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false
  for (const x of a) if (!b.has(x)) return false
  return true
}

export function computeAutoEdgePairs(
  nodes: OpNodeData[],
  manualEdges: EdgeData[],
  disconnected: Set<string>,
): [number, number][] {
  const manualSet = new Set(manualEdges.map((e) => `${e.source}-${e.target}`))

  // Step 1: build graph from explicit edges + cross-rank auto edges for predecessor computation
  const adj = new Map<number, number[]>()
  const indeg = new Map<number, number>()
  for (const n of nodes) { adj.set(n.index, []); indeg.set(n.index, 0) }

  for (const e of manualEdges) {
    const s = parseInt(e.source), t = parseInt(e.target)
    if (isNaN(s) || isNaN(t)) continue
    adj.get(s)?.push(t)
    indeg.set(t, (indeg.get(t) || 0) + 1)
  }

  const rankFirst = new Map<number, number>()
  const rankLast = new Map<number, number>()
  for (const n of nodes) {
    if (BUILTIN_INDICES.has(n.index)) continue
    if (!rankFirst.has(n.rank)) rankFirst.set(n.rank, n.index)
    rankLast.set(n.rank, n.index)
  }
  // Cross-rank edges (always needed for correct predecessor propagation)
  for (const firstIdx of rankFirst.values()) {
    if (!manualSet.has(`${START_INDEX}-${firstIdx}`) && !disconnected.has(`${START_INDEX}-${firstIdx}`)) {
      adj.get(START_INDEX)?.push(firstIdx)
      indeg.set(firstIdx, (indeg.get(firstIdx) || 0) + 1)
    }
  }

  // Compute direct predecessors (reverse adjacency)
  const directPreds = new Map<number, Set<number>>()
  for (const n of nodes) directPreds.set(n.index, new Set())
  for (const [from, toList] of adj) {
    for (const to of toList) {
      directPreds.get(to)?.add(from)
    }
  }

  // Step 2: build output pairs
  const pairs: [number, number][] = []
  const add = (srcIdx: number, tgtIdx: number) => {
    const key = `${srcIdx}-${tgtIdx}`
    const revKey = `${tgtIdx}-${srcIdx}`
    if (!manualSet.has(key) && !disconnected.has(key) && !disconnected.has(revKey)) {
      pairs.push([srcIdx, tgtIdx])
    }
  }

  // Same-rank auto edges with concurrency detection
  for (let i = 1; i < nodes.length; i++) {
    if (nodes[i - 1].rank !== nodes[i].rank) continue
    if (BUILTIN_INDICES.has(nodes[i - 1].index) || BUILTIN_INDICES.has(nodes[i].index)) continue
    const prevIdx = nodes[i - 1].index
    const currIdx = nodes[i].index
    const prevP = directPreds.get(prevIdx)
    const currP = directPreds.get(currIdx)

    // Rule 1: sparse graph (no pred info) → default to serial
    if (!prevP || !currP || prevP.size === 0 || currP.size === 0) {
      add(prevIdx, currIdx)
      continue
    }
    // Rule 2: same predecessors → concurrent, skip
    if (_setsEqual(prevP, currP)) continue
    // Rule 3: prev must be in curr's predecessor set for a real dependency
    if (!currP.has(prevIdx)) continue
    add(prevIdx, currIdx)
  }

  for (const firstIdx of rankFirst.values()) add(START_INDEX, firstIdx)
  for (const lastIdx of rankLast.values()) add(lastIdx, END_INDEX)
  return pairs
}

function _buildAdjacency(nodes: OpNodeData[], edges: EdgeData[], disconnected: Set<string>): Map<string, Set<string>> {
  const adj = new Map<string, Set<string>>()
  for (const n of nodes) adj.set(String(n.index), new Set())
  for (const e of edges) {
    if (!adj.has(e.source)) adj.set(e.source, new Set())
    adj.get(e.source)?.add(e.target)
  }
  for (const [src, tgt] of computeAutoEdgePairs(nodes, edges, disconnected)) {
    if (!adj.has(String(src))) adj.set(String(src), new Set())
    adj.get(String(src))?.add(String(tgt))
  }
  return adj
}

export function toList(v: unknown): TensorMeta[] {
  if (Array.isArray(v)) return v.map((x: Record<string, unknown>) => ({ name: String(x.name || ''), shape: String(x.shape || ''), dtype: String(x.dtype || '') }))
  if (v && typeof v === 'object') return Object.entries(v as Record<string, unknown>).map(([k, val]) => ({ name: k, shape: String(val), dtype: '' }))
  return []
}

export const useModelStore = create<ModelState>((set, get) => ({
  operators: [],
  nodes: _makeBuiltinNodes(),
  layers: [],
  edges: [],
  disconnected: new Set<string>(),
  selectedNodeId: null,
  selectedOperator: null,
  selectedModule: null,
  modelName: 'custom-model',
  setModelName: (name) => set({ modelName: name }),
  hfConfigText: '',
  setHfConfigText: (text) => set({ hfConfigText: text }),
  topGlobalIndices: [START_INDEX],
  bottomGlobalIndices: [END_INDEX],
  reorderCount: 0,
  ranks: [0],
  activeRank: 0,
  operatorList: [],
  moduleList: [],
  moduleDefs: {},
  customOperators: (() => {
    try {
      const raw = localStorage.getItem('kepler-custom-operators')
      return raw ? JSON.parse(raw) : []
    } catch { return [] }
  })(),
  customModules: (() => {
    try {
      const raw = localStorage.getItem('kepler-custom-modules')
      return raw ? JSON.parse(raw) : []
    } catch { return [] }
  })(),

  addCustomOperator(op: OperatorDef) {
    const { customOperators, operators } = get()
    if (customOperators.some((o) => o.name === op.name)) return false
    if (operators.some((o) => o.name === op.name)) return false
    const next = [...customOperators, op]
    localStorage.setItem('kepler-custom-operators', JSON.stringify(next))
    set({ customOperators: next })
    return true
  },
  removeCustomOperator(name: string) {
    const next = get().customOperators.filter((o) => o.name !== name)
    localStorage.setItem('kepler-custom-operators', JSON.stringify(next))
    set({ customOperators: next })
  },
  updateCustomOperator(oldName: string, op: OperatorDef) {
    const next = get().customOperators.map((o) => o.name === oldName ? op : o)
    localStorage.setItem('kepler-custom-operators', JSON.stringify(next))
    set({ customOperators: next })
  },
  importCustomOperators(ops: OperatorDef[]) {
    const { customOperators, operators } = get()
    const existingNames = new Set([
      ...customOperators.map((o) => o.name),
      ...operators.map((o) => o.name),
    ])
    const newOps = ops.filter((op) => !existingNames.has(op.name))
    if (newOps.length === 0) return 0
    const next = [...customOperators, ...newOps]
    localStorage.setItem('kepler-custom-operators', JSON.stringify(next))
    set({ customOperators: next })
    return newOps.length
  },

  addCustomModule(mod: ModuleDef) {
    const { customModules } = get()
    if (customModules.some((m) => m.name === mod.name)) return false
    const next = [...customModules, { ...mod, isBuiltin: false }]
    localStorage.setItem('kepler-custom-modules', JSON.stringify(next))
    set({ customModules: next })
    return true
  },
  removeCustomModule(name: string) {
    const next = get().customModules.filter((m) => m.name !== name)
    localStorage.setItem('kepler-custom-modules', JSON.stringify(next))
    set({ customModules: next })
  },
  updateCustomModule(oldName: string, mod: ModuleDef) {
    const next = get().customModules.map((m) =>
      m.name === oldName ? { ...mod, isBuiltin: false } : m
    )
    localStorage.setItem('kepler-custom-modules', JSON.stringify(next))
    set({ customModules: next })
  },

  async loadOperatorList() {
    const { operators } = get()
    if (operators.length > 0) return
    const names = await api.fetchOperatorList()
    const ops = (await Promise.all(names.map((n) => api.fetchOperator(n))))
      .map((op) => ({ ...op, module: op.module || op.name }))
    set({ operatorList: names, operators: ops })
  },

  async loadModuleList() {
    const { moduleList } = get()
    if (moduleList.length > 0) return
    const names = await api.fetchModuleList()
    const results = await Promise.all(names.map((n) => api.fetchModule(n)))
    const defs: Record<string, OperatorDef[]> = {}
    for (let i = 0; i < names.length; i++) {
      defs[names[i]] = results[i].map((op) => ({ ...op, module: op.module || op.name }))
    }
    set({ moduleList: names, moduleDefs: defs })
  },

  addNode(operator: OperatorDef, pos?: { x: number; y: number }, rank?: number) {
    set((s) => {
      const idx = _nextId++
      const newNode = { index: idx, operator, label: `${operator.name}_0`, rank: rank ?? s.activeRank, position: pos }
      const newNodes = _reorderNodes({ ...s, nodes: [...s.nodes, newNode] }, s.layers)
      return { nodes: newNodes }
    })
  },

  removeNode(index: number) {
    if (BUILTIN_INDICES.has(index)) return
    set((s) => {
      const filtered = s.nodes.filter((l) => l.index !== index)
      const selectedId = s.selectedNodeId === String(index) ? null : s.selectedNodeId
      const nid = String(index)
      return { nodes: filtered, edges: s.edges.filter((e) => e.source !== nid && e.target !== nid), selectedNodeId: selectedId }
    })
  },

  selectNode(id: string | null) {
    set({ selectedNodeId: id })
  },

  updateNodeSection(index: number, section: string, data: any) {
    set((s) => ({
      nodes: s.nodes.map((l) =>
        l.index === index ? {
          ...l,
          operator: { ...l.operator, [section]: data },
        } : l
      ),
    }))
  },

  moveNode(index: number, pos: { x: number; y: number }) {
    set((s) => ({
      nodes: s.nodes.map((l) => (l.index === index ? { ...l, position: pos } : l)),
    }))
  },

  addLayer(kind: 'regular' | 'mtp' = 'regular') {
    if (kind === 'mtp') { get().addMtpLayer(); return }
    const id = (_nextId++).toString(36)
    set((s) => ({
      layers: [...s.layers, { id, name: `Layer_${s.layers.length}`, repeat: 1, layerIdx: s.layers.length, kind: 'regular', rankOps: {} }],
    }))
  },

  addMtpLayer() {
    const id = (_nextId++).toString(36)
    set((s) => {
      if (s.layers.filter((l) => l.kind === 'mtp').length >= 1) return s
      const next = [...s.layers, { id, name: 'mtp_980', repeat: 1, layerIdx: 980, kind: 'mtp' as const, rankOps: {} as Record<number, number[]> }]
      return { layers: _renumberMtp(next) }
    })
  },

  duplicateLayer(id: string) {
    set((s) => {
      const srcIdx = s.layers.findIndex((l) => l.id === id)
      if (srcIdx < 0) return s
      const src = s.layers[srcIdx]
      if (src.kind === 'mtp' && s.layers.filter((l) => l.kind === 'mtp').length >= 1) return s
      const oldToNew = new Map<number, number>()
      const newNodes: OpNodeData[] = []
      const newRankOps: Record<number, number[]> = {}
      for (const rank of Object.keys(src.rankOps)) {
        const r = Number(rank)
        newRankOps[r] = []
        for (const oi of (src.rankOps[r] || [])) {
          const orig = s.nodes.find((n) => n.index === oi)
          if (!orig) continue
          const newIdx = _nextId++
          oldToNew.set(oi, newIdx)
          newRankOps[r].push(newIdx)
          newNodes.push({
            index: newIdx,
            operator: structuredClone(orig.operator),
            label: orig.label,
            rank: r,
          })
        }
      }
      const newId = (_nextId++).toString(36)
      const nextRegIdx = src.kind === 'mtp' ? 0
        : Math.max(...s.layers.filter(l => l.kind !== 'mtp').map(l => l.layerIdx), 0) + 1
      const newLayer: LayerConfig = { id: newId, name: src.kind === 'mtp' ? `mtp_${src.layerIdx}` : `Layer_${nextRegIdx}`, repeat: src.repeat, layerIdx: nextRegIdx, kind: src.kind || 'regular', rankOps: newRankOps }
      const newLayers = [...s.layers]
      newLayers.splice(srcIdx + 1, 0, newLayer)
      // insert new nodes after the last source-layer node in the array
      const allSrcOpIndices = new Set(Object.values(src.rankOps).flat())
      let insertPos = 0
      for (let i = 0; i < s.nodes.length; i++) {
        if (allSrcOpIndices.has(s.nodes[i].index)) insertPos = i + 1
      }
      const allNodes = [...s.nodes]
      allNodes.splice(insertPos, 0, ...newNodes)
      return { layers: _renumberMtp(newLayers), nodes: allNodes }
    })
  },

  removeLayer(id: string) {
    set((s) => {
      const layer = s.layers.find((l) => l.id === id)
      if (!layer) return s
      const allOpIndices = new Set(Object.values(layer.rankOps).flat())
      const newNodes = s.nodes.filter((n) => !allOpIndices.has(n.index))
      const newEdges = s.edges.filter((e) => !allOpIndices.has(parseInt(e.source)) && !allOpIndices.has(parseInt(e.target)))
      return { layers: _renumberMtp(s.layers.filter((l) => l.id !== id)), nodes: newNodes, edges: newEdges }
    })
  },

  setLayerRepeat(id: string, repeat: number) {
    set((s) => ({
      layers: s.layers.map((l) => l.id === id ? { ...l, repeat: Math.max(1, repeat | 0) } : l),
    }))
  },

  addOpToLayer(layerId: string, opIndex: number) {
    set((s) => {
      const node = s.nodes.find((n) => n.index === opIndex)
      const r = node?.rank ?? s.activeRank
      const newLayers = s.layers.map((l) => {
        if (l.id !== layerId) return l
        const ops = l.rankOps[r] || []
        if (ops.includes(opIndex)) return l
        return { ...l, rankOps: { ...l.rankOps, [r]: [...ops, opIndex] } }
      })
      return {
        layers: newLayers,
        nodes: _reorderNodes(s, newLayers),
        reorderCount: s.reorderCount + 1,
      }
    })
  },

  addNodeToLayer(layerId: string, op: OperatorDef) {
    set((s) => {
      const idx = _nextId++
      const rank = s.activeRank
      const node = { index: idx, operator: op, label: `${op.name}_0`, rank }
      // insert after: all nodes of lower ranks + this rank's top globals + preceding layers' rank ops + current layer's rank ops
      let insertPos = 0
      const targetLayerIdx = s.layers.findIndex((l) => l.id === layerId)
      for (const n of s.nodes) {
        if (n.rank < rank) { insertPos++; continue }
        if (n.rank > rank) continue
        const inTop = s.topGlobalIndices.includes(n.index)
        if (inTop) { insertPos++; continue }
        const ownerLayerIdx = s.layers.findIndex((l) => (l.rankOps[rank] || []).includes(n.index))
        if (ownerLayerIdx < 0) { insertPos++; continue }
        if (ownerLayerIdx < targetLayerIdx) { insertPos++; continue }
        if (ownerLayerIdx === targetLayerIdx) { insertPos++; continue }
        break
      }
      const newNodes = [...s.nodes]
      newNodes.splice(insertPos, 0, node)
      const targetLayer = s.layers[targetLayerIdx]
      const ops = targetLayer.rankOps[rank] || []
      return {
        nodes: newNodes,
        layers: s.layers.map((l) =>
          l.id === layerId ? { ...l, rankOps: { ...l.rankOps, [rank]: [...ops, idx] } } : l
        ),
      }
    })
  },

  removeOpFromLayer(layerId: string, opIndex: number) {
    set((s) => ({
      layers: s.layers.map((l) => {
        if (l.id !== layerId) return l
        const newRankOps = { ...l.rankOps }
        for (const r of Object.keys(newRankOps)) {
          newRankOps[Number(r)] = (newRankOps[Number(r)] || []).filter((i) => i !== opIndex)
        }
        return { ...l, rankOps: newRankOps }
      }),
      nodes: s.nodes.filter((n) => n.index !== opIndex),
      edges: s.edges.filter((e) => e.source !== String(opIndex) && e.target !== String(opIndex)),
    }))
  },

  renameLayer(id: string, name: string) {
    set((s) => ({
      layers: s.layers.map((l) => l.id === id ? { ...l, name } : l),
    }))
  },

  moveLayerUp(id: string) {
    set((s) => {
      const idx = s.layers.findIndex((l) => l.id === id)
      if (idx <= 0) return s
      const arr = [...s.layers]
      ;[arr[idx - 1], arr[idx]] = [arr[idx], arr[idx - 1]]
      return { layers: arr, nodes: _reorderNodes(s, arr), reorderCount: s.reorderCount + 1 }
    })
  },

  moveLayerDown(id: string) {
    set((s) => {
      const idx = s.layers.findIndex((l) => l.id === id)
      if (idx < 0 || idx >= s.layers.length - 1) return s
      const arr = [...s.layers]
      ;[arr[idx], arr[idx + 1]] = [arr[idx + 1], arr[idx]]
      return { layers: arr, nodes: _reorderNodes(s, arr), reorderCount: s.reorderCount + 1 }
    })
  },

  setLayerIndex(id: string, newIdx: number) {
    if (newIdx < 0) return
    set((s) => {
      const target = s.layers.find((l) => l.id === id)
      if (!target || target.layerIdx === newIdx) return s
      // swap with existing layer that has the same target idx
      const other = s.layers.find((l) => l.id !== id && l.layerIdx === newIdx)
      const newLayers = s.layers.map((l) => {
        if (l.id === id) return { ...l, layerIdx: newIdx }
        if (other && l.id === other.id) return { ...l, layerIdx: target.layerIdx }
        return l
      })
      return { layers: newLayers, nodes: _reorderNodes(s, newLayers), reorderCount: s.reorderCount + 1 }
    })
  },

  reorderNodes() {
    set((s) => ({
      nodes: _reorderNodes(s, s.layers),
      reorderCount: s.reorderCount + 1,
    }))
  },

  selectOperator(op: OperatorDef | null) {
    set({ selectedOperator: op })
  },

  selectModule(mod: ModuleDef | null) {
    set({ selectedModule: mod })
  },

  addTopGlobal(op: OperatorDef, rank?: number) {
    const r = rank ?? 0
    set((s) => {
      const idx = _nextId++
      const node = { index: idx, operator: op, label: `${op.name}_0`, rank: r }
      // insert after: nodes of lower ranks + same-rank top globals
      let pos = 0
      for (const n of s.nodes) {
        if (n.rank < r) { pos++; continue }
        if (n.rank > r) break
        if (s.topGlobalIndices.includes(n.index)) { pos++; continue }
        break
      }
      const newNodes = [...s.nodes]
      newNodes.splice(pos, 0, node)
      return { nodes: newNodes, topGlobalIndices: [...s.topGlobalIndices, idx] }
    })
  },

  addBottomGlobal(op: OperatorDef, rank?: number) {
    const r = rank ?? 0
    set((s) => {
      const idx = _nextId++
      const node = { index: idx, operator: op, label: `${op.name}_0`, rank: r }
      // insert after: all nodes of ranks <= r, before next rank
      let pos = 0
      for (const n of s.nodes) {
        if (n.rank < r) { pos++; continue }
        if (n.rank > r) break
        if (s.bottomGlobalIndices.includes(n.index)) { pos++; continue }
        pos++
      }
      const newNodes = [...s.nodes]
      newNodes.splice(pos, 0, node)
      return { nodes: newNodes, bottomGlobalIndices: [...s.bottomGlobalIndices, idx] }
    })
  },

  removeTopGlobal(index: number) {
    if (BUILTIN_INDICES.has(index)) return
    set((s) => ({
      topGlobalIndices: s.topGlobalIndices.filter((i) => i !== index),
      nodes: s.nodes.filter((n) => n.index !== index),
      edges: s.edges.filter((e) => e.source !== String(index) && e.target !== String(index)),
    }))
  },

  removeBottomGlobal(index: number) {
    if (BUILTIN_INDICES.has(index)) return
    set((s) => ({
      bottomGlobalIndices: s.bottomGlobalIndices.filter((i) => i !== index),
      nodes: s.nodes.filter((n) => n.index !== index),
      edges: s.edges.filter((e) => e.source !== String(index) && e.target !== String(index)),
    }))
  },

  reorderTopGlobals(rank: number, fromIdx: number, toIdx: number) {
    set((s) => {
      const rankIndices = s.topGlobalIndices.filter((i) => {
        const n = s.nodes.find((x) => x.index === i)
        return n && n.rank === rank && !BUILTIN_INDICES.has(i)
      })
      if (fromIdx < 0 || fromIdx >= rankIndices.length || toIdx < 0 || toIdx >= rankIndices.length) return s
      const [moved] = rankIndices.splice(fromIdx, 1)
      rankIndices.splice(toIdx, 0, moved)
      const others = s.topGlobalIndices.filter((i) => {
        const n = s.nodes.find((x) => x.index === i)
        return !n || n.rank !== rank || BUILTIN_INDICES.has(i)
      })
      const nextState = { ...s, topGlobalIndices: [...others, ...rankIndices] }
      return { topGlobalIndices: nextState.topGlobalIndices, nodes: _reorderNodes(nextState, s.layers), reorderCount: s.reorderCount + 1 }
    })
  },

  reorderBottomGlobals(rank: number, fromIdx: number, toIdx: number) {
    set((s) => {
      const rankIndices = s.bottomGlobalIndices.filter((i) => {
        const n = s.nodes.find((x) => x.index === i)
        return n && n.rank === rank && !BUILTIN_INDICES.has(i)
      })
      if (fromIdx < 0 || fromIdx >= rankIndices.length || toIdx < 0 || toIdx >= rankIndices.length) return s
      const [moved] = rankIndices.splice(fromIdx, 1)
      rankIndices.splice(toIdx, 0, moved)
      const others = s.bottomGlobalIndices.filter((i) => {
        const n = s.nodes.find((x) => x.index === i)
        return !n || n.rank !== rank || BUILTIN_INDICES.has(i)
      })
      const nextState = { ...s, bottomGlobalIndices: [...others, ...rankIndices] }
      return { bottomGlobalIndices: nextState.bottomGlobalIndices, nodes: _reorderNodes(nextState, s.layers), reorderCount: s.reorderCount + 1 }
    })
  },

  reorderLayerOps(layerId: string, rank: number, fromIdx: number, toIdx: number) {
    set((s) => {
      const newLayers = s.layers.map((l) => {
        if (l.id !== layerId) return l
        const ops = [...(l.rankOps[rank] || [])]
        if (fromIdx < 0 || fromIdx >= ops.length || toIdx < 0 || toIdx >= ops.length) return l
        const [moved] = ops.splice(fromIdx, 1)
        ops.splice(toIdx, 0, moved)
        return { ...l, rankOps: { ...l.rankOps, [rank]: ops } }
      })
      return { layers: newLayers, nodes: _reorderNodes(s, newLayers), reorderCount: s.reorderCount + 1 }
    })
  },

  addEdge(source: string, target: string) {
    if (source === target) return false
    const state = get()
    const orderedNodes = _reorderNodes(state, state.layers)
    const adj = _buildAdjacency(orderedNodes, state.edges, state.disconnected)
    const visited = new Set<string>()
    const stack = [target]
    while (stack.length > 0) {
      const node = stack.pop()!
      if (node === source) return false // would create cycle
      if (visited.has(node)) continue
      visited.add(node)
      for (const next of adj.get(node) || []) {
        if (!visited.has(next)) stack.push(next)
      }
    }
    set((s) => {
      // clear any prior disconnection so manual connect overrides auto-edge removal
      const disconnected = new Set(s.disconnected)
      disconnected.delete(`${source}-${target}`)
      return {
        edges: [...s.edges, { id: `e-${source}-${target}-${_nextId++}`, source, target }],
        disconnected,
      }
    })
    return true
  },

  wouldCreateCycle(source: string, target: string) {
    if (source === target) return true
    const state = get()
    const orderedNodes = _reorderNodes(state, state.layers)
    const adj = _buildAdjacency(orderedNodes, state.edges, state.disconnected)
    const visited = new Set<string>()
    const stack = [target]
    while (stack.length > 0) {
      const node = stack.pop()!
      if (node === source) return true
      if (visited.has(node)) continue
      visited.add(node)
      for (const next of adj.get(node) || []) {
        if (!visited.has(next)) stack.push(next)
      }
    }
    return false
  },

  removeEdge(edgeId: string) {
    set((s) => {
      const disconnected = new Set(s.disconnected)
      const autoMatch = edgeId.match(/^auto-(\d+)-(\d+)$/)
      if (autoMatch) {
        const key = `${autoMatch[1]}-${autoMatch[2]}`
        // don't re-add to disconnected if a manual edge already covers this pair
        // (prevents stale re-addition from ReactFlow's onEdgesChange re-entrancy)
        const hasManual = s.edges.some(e => e.source === autoMatch[1] && e.target === autoMatch[2])
        if (!hasManual) {
          disconnected.add(key)
        }
      } else {
        // also track explicit edge removals so auto-edge system won't re-add them
        const edge = s.edges.find(e => e.id === edgeId)
        if (edge) disconnected.add(`${edge.source}-${edge.target}`)
      }
      return {
        edges: s.edges.filter((e) => e.id !== edgeId),
        disconnected,
      }
    })
  },

  exportModel(name: string) {
    const { nodes, layers, edges, disconnected, topGlobalIndices, bottomGlobalIndices, ranks } = get()
    const topSet = new Set(topGlobalIndices)
    const bottomSet = new Set(bottomGlobalIndices)
    const isBuiltin = (n: OpNodeData) => BUILTIN_INDICES.has(n.index)

    // build canonical order matching canvas: start → middle → end
    const ordered: OpNodeData[] = []
    const startNode = nodes.find((n) => n.operator.name === 'start')
    const endNode = nodes.find((n) => n.operator.name === 'end')
    if (startNode) ordered.push(startNode)
    const regularLayers = layers.filter((l) => l.kind !== 'mtp')
    const mtpLayers = layers.filter((l) => l.kind === 'mtp')
    for (const rank of ranks) {
      for (const n of nodes) if (topSet.has(n.index) && n.rank === rank && !isBuiltin(n)) ordered.push(n)
      for (const l of regularLayers) {
        for (const oi of (l.rankOps[rank] || [])) {
          const n = nodes.find((x) => x.index === oi)
          if (n) ordered.push(n)
        }
      }
      for (const n of nodes) if (bottomSet.has(n.index) && n.rank === rank && !isBuiltin(n)) ordered.push(n)
      for (const l of mtpLayers) {
        for (const oi of (l.rankOps[rank] || [])) {
          const n = nodes.find((x) => x.index === oi)
          if (n) ordered.push(n)
        }
      }
    }
    if (endNode) ordered.push(endNode)
    const posMap = new Map<number, number>() // node.index → ordered position
    ordered.forEach((n, i) => posMap.set(n.index, i))

    // operators in canonical order (including start/end)
    const opLayerMap = new Map<number, number>()
    for (let li = 0; li < layers.length; li++) {
      for (const ops of Object.values(layers[li].rankOps)) {
        for (const oi of (ops as number[])) opLayerMap.set(oi, layers[li].kind === 'mtp' ? 980 : layers[li].layerIdx)
      }
    }
    const operators = ordered.map((n) => {
      const op = n.operator
      let layerIdx: number
      if (isBuiltin(n)) layerIdx = n.operator.name === 'start' ? -2 : 1000
      else if (opLayerMap.has(n.index)) layerIdx = opLayerMap.get(n.index)!
      else if (topSet.has(n.index)) layerIdx = -1
      else if (bottomSet.has(n.index)) layerIdx = 900
      else layerIdx = -1
      return {
        op_id: posMap.get(n.index)!,
        op_name: op.name,
        layer_idx: layerIdx,
        rank_idx: isBuiltin(n) ? -1 : (n.rank ?? 0),
        op_module: op.module,
        inputs: toList(op.inputs),
        params: toList(op.params),
        outputs: toList(op.outputs),
        compute_flops: op.compute_flops || '0',
        compute_unit: op.compute_unit || 'cube',
      }
    })

    // collect all edges matching canvas: manual + same-rank auto + cross-rank auto
    const allEdges = [...edges]
    for (const [src, tgt] of computeAutoEdgePairs(ordered, edges, disconnected)) {
      allEdges.push({ id: `auto-${src}-${tgt}`, source: String(src), target: String(tgt) })
    }

    const edgeDefs = allEdges
      .filter((e) => {
        const s = parseInt(e.source), t = parseInt(e.target)
        return posMap.has(s) && posMap.has(t)
      })
      .map((e) => ({
        from: posMap.get(parseInt(e.source))!,
        to: posMap.get(parseInt(e.target))!,
      }))

    // rank summaries: ordered op_ids per rank (excluding builtins)
    const rankDefs = ranks.map((r) => {
      const opIds: number[] = []
      for (const n of ordered) {
        if (n.rank === r && !isBuiltin(n)) opIds.push(posMap.get(n.index)!)
      }
      const rankLayers: { layer_idx: number; repeat: number; kind: string; ops: number[] }[] = []
      for (const l of layers) {
        const lopIds = (l.rankOps[r] || [])
          .map((oi: number) => posMap.get(oi))
          .filter((i): i is number => i !== undefined)
        if (lopIds.length > 0) {
          rankLayers.push({ layer_idx: l.layerIdx, repeat: l.repeat, kind: l.kind || 'regular', ops: lopIds })
        }
      }
      return { rank_idx: r, ops: opIds, layers: rankLayers }
    })

    return {
      name,
      num_ops: operators.length,
      num_layers: layers.length,
      num_edges: edgeDefs.length,
      operators,
      edges: edgeDefs,
      ranks: rankDefs,
    }
  },

  importFromJSON(json: any) {
    const { operators: opList, edges: edgeList } = json
    // support both new format (layers in ranks) and old format (top-level layers)
    const layerList: any[] = Array.isArray(json.layers) ? json.layers
      : (Array.isArray(json.ranks) ? json.ranks.flatMap((r: any) => (r.layers || []).map((l: any) => ({ ...l, rank_idx: r.rank_idx, rank_ops: { [r.rank_idx]: l.ops } }))) : [])
    if (!Array.isArray(opList)) return

    const existingOps = get().operators
    const opDefMap = new Map(existingOps.map((o) => [o.name, o]))
    const idMap = new Map<number, number>() // old op_id → new node index

    // build operator defs and nodes, preserving builtins
    const nodes: OpNodeData[] = _makeBuiltinNodes()
    const topGlobals: number[] = [START_INDEX]
    const bottomGlobals: number[] = [END_INDEX]

    for (const op of opList) {
      if (op.op_name === 'start') { idMap.set(op.op_id, START_INDEX); continue }
      if (op.op_name === 'end') { idMap.set(op.op_id, END_INDEX); continue }
      const baseDef = opDefMap.get(op.op_name)
      const operator: OperatorDef = {
        name: op.op_name,
        module: op.op_module || op.op_name,
        description: baseDef?.description || '',
        inputs: Array.isArray(op.inputs) ? op.inputs : toList(op.inputs),
        params: Array.isArray(op.params) ? op.params : toList(op.params),
        outputs: Array.isArray(op.outputs) ? op.outputs : toList(op.outputs),
        compute_flops: op.compute_flops || '0',
        compute_unit: op.compute_unit || 'cube',
      }
      const idx = _nextId++
      nodes.push({ index: idx, operator, label: `${op.op_name}_${op.op_id}`, rank: op.rank_idx ?? 0 })
      idMap.set(op.op_id, idx)

      if (op.layer_idx === -1) topGlobals.push(idx)
      else if (op.layer_idx === 900 || op.layer_idx === 999999) bottomGlobals.push(idx)
    }

    // build layers
    const layers: LayerConfig[] = []
    if (Array.isArray(layerList)) {
      for (const l of layerList) {
        let rankOps: Record<number, number[]>
        if (l.rank_ops) {
          // new format: per-rank operators
          rankOps = {}
          for (const [r, ops] of Object.entries(l.rank_ops as Record<string, number[]>)) {
            rankOps[Number(r)] = ops.map((oid: number) => idMap.get(oid)).filter((i) => i !== undefined) as number[]
          }
        } else {
          // old format: single op_idx + rank_idx
          const ops = (Array.isArray(l.op_idx) ? l.op_idx : [])
            .map((oid: number) => idMap.get(oid))
            .filter((i: number | undefined) => i !== undefined) as number[]
          const r = l.rank_idx ?? 0
          rankOps = ops.length > 0 ? { [r]: ops } : {}
        }
        layers.push({
          id: (_nextId++).toString(36),
          name: l.kind === 'mtp' ? `mtp_${l.layer_idx ?? 0}` : `Layer_${l.layer_idx ?? 0}`,
          repeat: l.repeat || 1,
          layerIdx: l.layer_idx ?? 0,
          kind: l.kind || 'regular',
          rankOps,
        })
      }
    }

    // reconstruct ranks from imported data (prefer explicit ranks field)
    let ranks: number[] = []
    if (Array.isArray(json.ranks)) {
      ranks = json.ranks.map((r: any) => r.rank_idx ?? r).filter((r: number) => r >= 0)
    }
    if (ranks.length === 0) {
      const rankSet = new Set<number>()
      for (const n of nodes) rankSet.add(n.rank)
      for (const l of layers) for (const r of Object.keys(l.rankOps)) rankSet.add(Number(r))
      if (!rankSet.has(0)) rankSet.add(0)
      ranks = [...rankSet].sort((a, b) => a - b)
    }

    // build node order: by rank (top globals → regular layers → bottom globals → MTP layers)
    const orderedMiddle: OpNodeData[] = []
    const topSet = new Set(topGlobals)
    const bottomSet = new Set(bottomGlobals)
    const regLayers = layers.filter((l) => l.kind !== 'mtp')
    const mtpOnlyLayers = layers.filter((l) => l.kind === 'mtp')
    for (const rank of ranks) {
      for (const n of nodes) {
        if (topSet.has(n.index) && n.rank === rank && !BUILTIN_INDICES.has(n.index)) orderedMiddle.push(n)
      }
      for (const l of regLayers) {
        for (const idx of (l.rankOps[rank] || [])) {
          const n = nodes.find((x) => x.index === idx)
          if (n) orderedMiddle.push(n)
        }
      }
      for (const n of nodes) {
        if (bottomSet.has(n.index) && n.rank === rank && !BUILTIN_INDICES.has(n.index)) orderedMiddle.push(n)
      }
      for (const l of mtpOnlyLayers) {
        for (const idx of (l.rankOps[rank] || [])) {
          const n = nodes.find((x) => x.index === idx)
          if (n) orderedMiddle.push(n)
        }
      }
    }
    for (const n of nodes) {
      if (BUILTIN_INDICES.has(n.index)) continue
      if (topSet.has(n.index)) continue
      if (bottomSet.has(n.index)) continue
      if (!layers.some((l) => Object.values(l.rankOps).flat().includes(n.index))) {
        orderedMiddle.push(n)
      }
    }
    const orderedNodes: OpNodeData[] = []
    const importStart = nodes.find((n) => n.index === START_INDEX)
    const importEnd = nodes.find((n) => n.index === END_INDEX)
    if (importStart) orderedNodes.push(importStart)
    orderedNodes.push(...orderedMiddle)
    if (importEnd) orderedNodes.push(importEnd)

    const edges: EdgeData[] = []
    if (Array.isArray(edgeList)) {
      for (const e of edgeList) {
        const srcIdx = idMap.get(e.from)
        const tgtIdx = idMap.get(e.to)
        if (srcIdx !== undefined && tgtIdx !== undefined) {
          edges.push({ id: `e-${srcIdx}-${tgtIdx}-${_nextId++}`, source: String(srcIdx), target: String(tgtIdx) })
        }
      }
    }

    // filter out edges that would be auto-generated (consecutive same-rank pairs),
    // so they adapt to reordering instead of persisting as stale manual edges
    const autoPairs = computeAutoEdgePairs(orderedNodes, [], new Set())
    const autoKeySet = new Set(autoPairs.map(([s, t]) => `${s}-${t}`))
    const manualEdges = edges.filter((e) => !autoKeySet.has(`${e.source}-${e.target}`))

    set({
      nodes: orderedNodes,
      layers: _renumberMtp(layers),
      topGlobalIndices: topGlobals,
      bottomGlobalIndices: bottomGlobals,
      edges: manualEdges,
      disconnected: new Set(),
      selectedNodeId: null,
      reorderCount: get().reorderCount + 1,
      ranks,
    })
  },


  addRank() {
    set((s) => {
      let r = s.ranks.length
      while (s.ranks.includes(r)) r++
      return { ranks: [...s.ranks, r].sort((a, b) => a - b) }
    })
  },

  duplicateRank(sourceRank: number) {
    set((s) => {
      let newRank = s.ranks.length
      while (s.ranks.includes(newRank)) newRank++
      const newNodes: OpNodeData[] = []
      const newTopGlobals: number[] = []
      const newBottomGlobals: number[] = []
      const indexMap = new Map<number, number>() // old index → new index

      // copy top globals for source rank (skip builtins)
      for (const idx of s.topGlobalIndices) {
        if (BUILTIN_INDICES.has(idx)) continue
        const n = s.nodes.find((n) => n.index === idx)
        if (n && n.rank === sourceRank) {
          const newIdx = _nextId++
          indexMap.set(n.index, newIdx)
          newTopGlobals.push(newIdx)
          newNodes.push({ index: newIdx, operator: structuredClone(n.operator), label: n.label, rank: newRank })
        }
      }
      // copy layer ops for source rank
      const newLayerRankOps: { layerId: string; ops: number[] }[] = []
      for (const l of s.layers) {
        const srcOps = l.rankOps[sourceRank] || []
        if (srcOps.length === 0) continue
        const copied: number[] = []
        for (const oi of srcOps) {
          if (indexMap.has(oi)) { copied.push(indexMap.get(oi)!); continue }
          const n = s.nodes.find((n) => n.index === oi)
          if (!n) continue
          const newIdx = _nextId++
          indexMap.set(n.index, newIdx)
          newNodes.push({ index: newIdx, operator: structuredClone(n.operator), label: n.label, rank: newRank })
          copied.push(newIdx)
        }
        newLayerRankOps.push({ layerId: l.id, ops: copied })
      }
      // copy bottom globals for source rank (skip builtins)
      for (const idx of s.bottomGlobalIndices) {
        if (BUILTIN_INDICES.has(idx)) continue
        const n = s.nodes.find((n) => n.index === idx)
        if (n && n.rank === sourceRank) {
          if (indexMap.has(n.index)) { newBottomGlobals.push(indexMap.get(n.index)!); continue }
          const newIdx = _nextId++
          indexMap.set(n.index, newIdx)
          newBottomGlobals.push(newIdx)
          newNodes.push({ index: newIdx, operator: structuredClone(n.operator), label: n.label, rank: newRank })
        }
      }

      // insert new nodes after the last source-rank node
      let insertPos = 0
      for (let i = 0; i < s.nodes.length; i++) {
        if (s.nodes[i].rank === sourceRank) insertPos = i + 1
      }
      const allNodes = [...s.nodes]
      allNodes.splice(insertPos, 0, ...newNodes)

      // update layers with new rankOps
      const newLayers = s.layers.map((l) => {
        const entry = newLayerRankOps.find((e) => e.layerId === l.id)
        if (!entry) return l
        return { ...l, rankOps: { ...l.rankOps, [newRank]: entry.ops } }
      })

      return {
        nodes: allNodes,
        layers: newLayers,
        topGlobalIndices: [...s.topGlobalIndices, ...newTopGlobals],
        bottomGlobalIndices: [...s.bottomGlobalIndices, ...newBottomGlobals],
        ranks: [...s.ranks, newRank].sort((a, b) => a - b),
        activeRank: newRank,
      }
    })
  },

  updateRankIndex(oldIdx: number, newIdx: number) {
    if (oldIdx === newIdx) return true
    const s = get()
    if (s.ranks.includes(newIdx)) return false
    set({
      nodes: s.nodes.map((n) => n.rank === oldIdx ? { ...n, rank: newIdx } : n),
      layers: s.layers.map((l) => {
        const ops = l.rankOps[oldIdx] || []
        if (ops.length === 0) return l
        const newRankOps = { ...l.rankOps }
        delete newRankOps[oldIdx]
        newRankOps[newIdx] = [...(newRankOps[newIdx] || []), ...ops]
        return { ...l, rankOps: newRankOps }
      }),
      ranks: s.ranks.map((r) => r === oldIdx ? newIdx : r).sort((a, b) => a - b),
      activeRank: s.activeRank === oldIdx ? newIdx : s.activeRank,
    })
    return true
  },

  removeRank(rank: number) {
    if (rank === 0) return
    set((s) => {
      const removedIndices = new Set<number>()
      for (const n of s.nodes) {
        if (n.rank === rank) removedIndices.add(n.index)
      }
      return {
        nodes: s.nodes.filter((n) => n.rank !== rank),
        topGlobalIndices: s.topGlobalIndices.filter((i) => !removedIndices.has(i)),
        bottomGlobalIndices: s.bottomGlobalIndices.filter((i) => !removedIndices.has(i)),
        layers: s.layers.map((l) => {
          if (!l.rankOps[rank]) return l
          const newRankOps = { ...l.rankOps }
          delete newRankOps[rank]
          return { ...l, rankOps: newRankOps }
        }),
        edges: s.edges.filter((e) => {
          const sIdx = parseInt(e.source), tIdx = parseInt(e.target)
          return !removedIndices.has(sIdx) && !removedIndices.has(tIdx)
        }),
        ranks: s.ranks.filter((r) => r !== rank),
        activeRank: s.activeRank === rank ? 0 : s.activeRank,
      }
    })
  },

  setActiveRank(rank: number) {
    set({ activeRank: rank })
  },

  clearNodes() {
    set({ nodes: _makeBuiltinNodes(), layers: [], topGlobalIndices: [START_INDEX], bottomGlobalIndices: [END_INDEX], edges: [], disconnected: new Set(), selectedNodeId: null, ranks: [0], activeRank: 0 })
  },
}))
