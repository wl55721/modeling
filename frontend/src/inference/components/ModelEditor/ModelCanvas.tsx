import { useEffect, useMemo, useCallback, useRef, useState } from 'react'
import {
  ReactFlow, Background, Controls,
  useReactFlow, type Node, type Edge, type NodeTypes, type Connection, type EdgeChange,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'

import { useModelStore, computeAutoEdgePairs, BUILTIN_INDICES } from '../../stores/model'
import type { OpNodeData, OperatorDef } from '../../types/model'
import LayerNode from './LayerNode'

const nodeTypes: NodeTypes = { layer: LayerNode }
const defaultEdgeOptions = {
  animated: true,
  style: { stroke: '#aaa', strokeWidth: 2 },
  markerEnd: { type: 'arrowclosed' as const, width: 10, height: 10, color: '#aaa' },
} as const

function avgPredTrack(arrIdx: string, directPreds: Map<string, Set<string>>, nodeTrack: Map<string, number>): number {
  const preds = directPreds.get(arrIdx) || new Set()
  let sum = 0, count = 0
  for (const pred of preds) {
    const t = nodeTrack.get(pred)
    if (t !== undefined) { sum += t; count++ }
  }
  return count > 0 ? sum / count : Infinity
}

function setsEqual<T>(a: Set<T>, b: Set<T>): boolean {
  if (a.size !== b.size) return false
  for (const x of a) if (!b.has(x)) return false
  return true
}

function computeLayoutLevels(
  ol: OpNodeData[],
  edges: { source: string; target: string }[],
  rawToOrder: Map<number, number>,
  disconnected: Set<string>,
): { levels: string[][]; directPreds: Map<string, Set<string>> } {
  const indeg = new Map<string, number>()
  const adj = new Map<string, string[]>()
  for (let i = 0; i < ol.length; i++) {
    indeg.set(String(i), 0); adj.set(String(i), [])
  }

  const addEdge = (from: string, to: string) => {
    adj.get(from)?.push(to)
    indeg.set(to, (indeg.get(to) || 0) + 1)
  }

  // Pass 1: explicit edges
  for (const e of edges) {
    const sIdx = parseInt(e.source), tIdx = parseInt(e.target)
    if (isNaN(sIdx) || isNaN(tIdx)) continue
    const so = rawToOrder.get(sIdx)
    const to = rawToOrder.get(tIdx)
    if (so != null && to != null) {
      addEdge(String(so), String(to))
    }
  }

  const isBuiltin = (n: OpNodeData) => n.operator.name === 'start' || n.operator.name === 'end'

  // Cross-rank auto edges (always needed for predecessor propagation)
  const rankFirst = new Map<number, number>()
  const rankLast = new Map<number, number>()
  ol.forEach((n, i) => {
    if (n.operator.module === '__builtin__') return
    if (!rankFirst.has(n.rank)) rankFirst.set(n.rank, i)
    rankLast.set(n.rank, i)
  })
  const startOrd = ol.findIndex((n) => n.operator.name === 'start')
  const endOrd = ol.findIndex((n) => n.operator.name === 'end')
  const addCrossAuto = (srcOrd: number, tgtOrd: number, srcIdx: number, tgtIdx: number) => {
    const hasManual = edges.some((e) => parseInt(e.source) === srcIdx && parseInt(e.target) === tgtIdx)
    if (!hasManual && !disconnected.has(`${srcIdx}-${tgtIdx}`)) {
      addEdge(String(srcOrd), String(tgtOrd))
    }
  }
  if (startOrd >= 0 && endOrd >= 0) {
    if (rankFirst.size === 0) {
      // No non-builtin nodes — wire start→end directly so they don't overlap
      addEdge(String(startOrd), String(endOrd))
    } else {
      for (const firstOrd of rankFirst.values()) {
        addCrossAuto(startOrd, firstOrd, ol[startOrd].index, ol[firstOrd].index)
      }
      for (const lastOrd of rankLast.values()) {
        addCrossAuto(lastOrd, endOrd, ol[lastOrd].index, ol[endOrd].index)
      }
    }
  }

  // Compute direct predecessors (reverse adjacency)
  const directPreds = new Map<string, Set<string>>()
  for (let i = 0; i < ol.length; i++) directPreds.set(String(i), new Set())
  for (const [from, toList] of adj) {
    for (const to of toList) {
      directPreds.get(to)?.add(from)
    }
  }

  // Pass 2: same-rank auto edges with concurrency detection
  for (let i = 1; i < ol.length; i++) {
    if (ol[i - 1].rank !== ol[i].rank) continue
    if (isBuiltin(ol[i - 1]) || isBuiltin(ol[i])) continue
    const hasManual = edges.some((e) => parseInt(e.source) === ol[i - 1].index && parseInt(e.target) === ol[i].index)
    if (hasManual) continue
    if (disconnected.has(`${ol[i - 1].index}-${ol[i].index}`)) continue
    if (disconnected.has(`${ol[i].index}-${ol[i - 1].index}`)) continue

    const prevP = directPreds.get(String(i - 1))
    const currP = directPreds.get(String(i))

    // Rule 1: sparse graph (no pred info) → default to serial
    if (!prevP || !currP || prevP.size === 0 || currP.size === 0) {
      addEdge(String(i - 1), String(i))
      continue
    }
    // Rule 2: same predecessors → concurrent, skip
    if (setsEqual(prevP, currP)) continue
    // Rule 3: prev must be in curr's predecessor set for a real dependency
    if (!currP.has(String(i - 1))) continue
    addEdge(String(i - 1), String(i))
  }

  // Compute depths (longest-path) + topological order
  const finalIndeg = new Map(indeg)
  const depth = new Map<string, number>()
  const queue = [...finalIndeg.entries()].filter(([, d]) => d === 0).map(([n]) => { depth.set(n, 0); return n })
  const order: string[] = []
  while (queue.length > 0) {
    const n = queue.shift()!
    order.push(n)
    const curDepth = depth.get(n) || 0
    for (const m of adj.get(n) || []) {
      depth.set(m, Math.max(depth.get(m) || 0, curDepth + 1))
      const d = (finalIndeg.get(m) || 1) - 1
      finalIndeg.set(m, d)
      if (d === 0) queue.push(m)
    }
  }

  // Group by depth
  const maxDepth = Math.max(...depth.values(), 0)
  const levels: string[][] = Array.from({ length: maxDepth + 1 }, () => [])
  for (const n of order) {
    const d = depth.get(n) ?? 0
    levels[d].push(n)
  }

  // Recompute direct preds from the final graph (Pass 1 + Pass 2 edges)
  const finalDirectPreds = new Map<string, Set<string>>()
  for (let i = 0; i < ol.length; i++) finalDirectPreds.set(String(i), new Set())
  for (const [from, toList] of adj) {
    for (const to of toList) {
      finalDirectPreds.get(to)?.add(from)
    }
  }

  return { levels: levels.filter(l => l.length > 0), directPreds: finalDirectPreds }
}

function ArrowPad({ getViewport, setViewport, rfFitView }: { getViewport: () => { x: number; y: number; zoom: number }; setViewport: (vp: { x: number; y: number; zoom: number }) => void; rfFitView: (opts?: any) => void }) {
  const timerRef = useRef<ReturnType<typeof setInterval> | undefined>(undefined)
  const padRef = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null)
  const dragRef = useRef<{ sx: number; sy: number; px: number; py: number } | null>(null)
  const step = 50

  const startPan = useCallback((dx: number, dy: number) => {
    const pan = () => {
      const vp = getViewport()
      setViewport({ x: vp.x + dx, y: vp.y + dy, zoom: vp.zoom })
    }
    pan()
    timerRef.current = setInterval(pan, 60)
  }, [getViewport, setViewport])

  const stopPan = useCallback(() => {
    clearInterval(timerRef.current)
  }, [])

  const onDragStart = useCallback((e: React.MouseEvent) => {
    e.preventDefault()
    const cur = pos || { x: 0, y: 0 }
    dragRef.current = { sx: e.clientX, sy: e.clientY, px: cur.x, py: cur.y }
    const onMove = (ev: MouseEvent) => {
      if (!dragRef.current) return
      setPos({ x: dragRef.current.px + ev.clientX - dragRef.current.sx, y: dragRef.current.py + ev.clientY - dragRef.current.sy })
    }
    const onUp = () => { dragRef.current = null; document.removeEventListener('mousemove', onMove); document.removeEventListener('mouseup', onUp) }
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)
  }, [pos])

  useEffect(() => () => clearInterval(timerRef.current), [])

  return (
    <div className="canvas-arrows" ref={padRef}
      style={pos ? { transform: `translate(${pos.x}px, ${pos.y}px)` } : undefined}>
      <button className="canvas-arrow canvas-arrow-up"
        onMouseDown={() => startPan(0, step)} onMouseUp={stopPan} onMouseLeave={stopPan} title="上移">▲</button>
      <div className="canvas-arrow-mid">
        <button className="canvas-arrow canvas-arrow-left"
          onMouseDown={() => startPan(step, 0)} onMouseUp={stopPan} onMouseLeave={stopPan} title="左移">◀</button>
        <button className="canvas-arrow canvas-arrow-reset"
          onMouseDown={onDragStart} title="拖拽移动 / 双击居中" onDoubleClick={() => rfFitView({ padding: 0.2, duration: 300 })}>⊙</button>
        <button className="canvas-arrow canvas-arrow-right"
          onMouseDown={() => startPan(-step, 0)} onMouseUp={stopPan} onMouseLeave={stopPan} title="右移">▶</button>
      </div>
      <button className="canvas-arrow canvas-arrow-down"
        onMouseDown={() => startPan(0, -step)} onMouseUp={stopPan} onMouseLeave={stopPan} title="下移">▼</button>
    </div>
  )
}

export default function ModelCanvas() {
  const storeNodes = useModelStore((s) => s.nodes)
  const edges = useModelStore((s) => s.edges)
  const layers = useModelStore((s) => s.layers)
  const topGlobalIndices = useModelStore((s) => s.topGlobalIndices)
  const bottomGlobalIndices = useModelStore((s) => s.bottomGlobalIndices)
  const disconnected = useModelStore((s) => s.disconnected)
  const ranks = useModelStore((s) => s.ranks)
  const reorderCount = useModelStore((s) => s.reorderCount)
  const addNode = useModelStore((s) => s.addNode)
  const addEdge = useModelStore((s) => s.addEdge)
  const wouldCreateCycle = useModelStore((s) => s.wouldCreateCycle)
  const removeEdge = useModelStore((s) => s.removeEdge)
  const moveNode = useModelStore((s) => s.moveNode)
  const { screenToFlowPosition, fitView: rfFitView, getViewport, setViewport } = useReactFlow()

  // logically ordered node list; rawToOrder maps node.index → ordered position (for layout)
  const { orderedList, rawToOrder } = useMemo(() => {
    const nodeByIdx = new Map(storeNodes.map((n) => [n.index, n]))
    const isBuiltin = (n: OpNodeData) => n.operator.name === 'start' || n.operator.name === 'end'
    const middle: OpNodeData[] = []

    for (const rank of ranks) {
      // tops: use topGlobalIndices order (authoritative)
      for (const idx of topGlobalIndices) {
        if (BUILTIN_INDICES.has(idx)) continue
        const n = nodeByIdx.get(idx)
        if (n && n.rank === rank) middle.push(n)
      }
      // regular layers
      for (const l of layers) {
        if (l.kind === 'mtp') continue
        for (const oi of (l.rankOps[rank] || [])) {
          const n = nodeByIdx.get(oi)
          if (n) middle.push(n)
        }
      }
      // bottoms: use bottomGlobalIndices order (authoritative)
      for (const idx of bottomGlobalIndices) {
        if (BUILTIN_INDICES.has(idx)) continue
        const n = nodeByIdx.get(idx)
        if (n && n.rank === rank) middle.push(n)
      }
      // MTP layers
      for (const l of layers) {
        if (l.kind !== 'mtp') continue
        for (const oi of (l.rankOps[rank] || [])) {
          const n = nodeByIdx.get(oi)
          if (n) middle.push(n)
        }
      }
    }
    const used = new Set(middle.map((n) => n.index))
    for (const n of storeNodes) {
      if (!used.has(n.index) && !isBuiltin(n)) middle.push(n)
    }
    const startNode = storeNodes.find((n) => n.operator.name === 'start')
    const endNode = storeNodes.find((n) => n.operator.name === 'end')
    const order: OpNodeData[] = []
    if (startNode) order.push(startNode)
    order.push(...middle)
    if (endNode) order.push(endNode)
    const r2o = new Map<number, number>()  // node.index → ordered position
    order.forEach((n, i) => { r2o.set(n.index, i) })
    return { orderedList: order, rawToOrder: r2o }
  }, [storeNodes, layers, topGlobalIndices, bottomGlobalIndices, ranks])

  const [nodeLayerMap, nodeColorMap] = useMemo(() => {
    const layerMap = new Map<number, number>()
    const colorMap = new Map<number, number>()
    const topSet = new Set(topGlobalIndices)
    const bottomSet = new Set(bottomGlobalIndices)
    for (const n of storeNodes) {
      if (n.operator.name === 'start') { layerMap.set(n.index, -2); colorMap.set(n.index, 0); continue }
      if (n.operator.name === 'end') { layerMap.set(n.index, 1000000); colorMap.set(n.index, 1); continue }
      if (topSet.has(n.index)) { layerMap.set(n.index, -1); colorMap.set(n.index, 0); continue }
      if (bottomSet.has(n.index)) { layerMap.set(n.index, 999999); colorMap.set(n.index, 1); continue }
      let found = false
      for (let li = 0; li < layers.length; li++) {
        const allOps = Object.values(layers[li].rankOps).flat()
        if (allOps.includes(n.index)) {
          layerMap.set(n.index, li); colorMap.set(n.index, li + 2); found = true; break
        }
      }
      if (!found) { layerMap.set(n.index, -1); colorMap.set(n.index, 0) }
    }
    return [layerMap, colorMap] as const
  }, [storeNodes, layers, topGlobalIndices, bottomGlobalIndices])

  const rfNodes: Node[] = useMemo(() => {
    let y = 50
    return orderedList.map((opNode, i) => {
      const pos = opNode.position || { x: 250, y }
      if (!opNode.position) y += 120
      return {
        id: String(opNode.index), type: 'layer', position: pos,
        data: { index: opNode.index, operator: opNode.operator, label: opNode.label, colorIdx: nodeColorMap.get(opNode.index) ?? 0, layerIdx: nodeLayerMap.get(opNode.index) ?? -1, rank: opNode.rank, orderIdx: i },
      }
    })
  }, [orderedList, nodeColorMap, nodeLayerMap])

  const allEdges: Edge[] = useMemo(() => {
    const autoEs: Edge[] = []
    for (const [srcRaw, tgtRaw] of computeAutoEdgePairs(orderedList, edges, disconnected)) {
      autoEs.push({ id: `auto-${srcRaw}-${tgtRaw}`, source: String(srcRaw), target: String(tgtRaw), animated: true, selectable: true })
    }
    const manualEs: Edge[] = edges.map((e) => ({
      ...e, animated: true, selectable: true
    }))
    return [...autoEs, ...manualEs]
  }, [orderedList, edges, disconnected])

  const onConnect = useCallback((conn: Connection) => {
    if (conn.source && conn.target) {
      if (wouldCreateCycle(conn.source, conn.target)) {
        console.warn('Cannot add edge: would create a cycle')
        return
      }
      addEdge(conn.source, conn.target)
    }
  }, [addEdge, wouldCreateCycle])

  const onNodeDragStop = useCallback((_event: React.MouseEvent, node: Node) => {
    const d = node.data as any
    moveNode(d.index, node.position)
  }, [moveNode])

  const removeCanvasEdge = useCallback((edgeId: string) => {
    const match = edgeId.match(/^auto-(\d+)-(\d+)$/)
    if (match) {
      removeEdge(`auto-${match[1]}-${match[2]}`)
    } else {
      removeEdge(edgeId)
    }
  }, [removeEdge])

  const onEdgesChange = useCallback((changes: EdgeChange[]) => {
    for (const c of changes) {
      if (c.type === 'remove') removeCanvasEdge(c.id)
    }
  }, [removeCanvasEdge])

  const onEdgeDoubleClick = useCallback((_event: React.MouseEvent, edge: Edge) => {
    removeCanvasEdge(edge.id)
  }, [removeCanvasEdge])

  const moveNodeRef = useRef(moveNode)
  moveNodeRef.current = moveNode
  const rfFitViewRef = useRef(rfFitView)
  rfFitViewRef.current = rfFitView

  const autoLayout = useCallback(() => {
    const moveNode = moveNodeRef.current
    const rfFitView = rfFitViewRef.current
    const ol = orderedList
    if (ol.length === 0) return

    const st = useModelStore.getState()
    const { levels, directPreds } = computeLayoutLevels(ol, st.edges, rawToOrder, st.disconnected)

    // Keep only non-empty levels (start, middle, end all go through the same pipeline)
    const fullLevels = levels.filter(l => l.length > 0)

    if (ranks.length <= 1) {
      const rowSpacing = 56
      const trackSpacing = 200

      // ================================================================
      // Step 1: Top-down — compute depth (y) and per-level track layout
      // ================================================================
      const nodeTrack = new Map<string, number>()
      let nextTrack = 0

      // Level 0 (typically start): initial track assignment left-to-right by ascending index
      if (fullLevels.length > 0) {
        const sorted = [...fullLevels[0]].sort((a, b) => {
          const na = ol[parseInt(a)]; const nb = ol[parseInt(b)]
          return (na?.index ?? 0) - (nb?.index ?? 0)
        })
        for (const arrIdx of sorted) nodeTrack.set(arrIdx, nextTrack++)
      }

      // Subsequent levels: inherit track from leftmost free predecessor, iterate until stable
      for (let iter = 0; iter < 10; iter++) {
        let changed = false
        for (let li = 1; li < fullLevels.length; li++) {
          const usedTracks = new Set<number>()
          const sorted = [...fullLevels[li]].sort((a, b) => {
            const d = avgPredTrack(a, directPreds, nodeTrack) - avgPredTrack(b, directPreds, nodeTrack)
            if (d !== 0) return d
            const na = ol[parseInt(a)]; const nb = ol[parseInt(b)]
            return (na?.index ?? 0) - (nb?.index ?? 0)
          })
          for (const arrIdx of sorted) {
            const preds = directPreds.get(arrIdx) || new Set()
            const predArr = [...preds].sort((a, b) =>
              (nodeTrack.get(a) ?? Infinity) - (nodeTrack.get(b) ?? Infinity))
            let track: number | undefined
            for (const pred of predArr) {
              const pt = nodeTrack.get(pred)
              if (pt !== undefined && !usedTracks.has(pt)) { track = pt; break }
            }
            if (track === undefined || usedTracks.has(track)) track = nextTrack++
            if (nodeTrack.get(arrIdx) !== track) { nodeTrack.set(arrIdx, track); changed = true }
            usedTracks.add(track)
          }
        }
        if (!changed) break
      }

      // ================================================================
      // Step 2: Global track compaction — same track → same x across all levels
      // ================================================================
      const allTracks = [...new Set(nodeTrack.values())].sort((a, b) => a - b)
      const trackToX = new Map<number, number>()
      const totalW = (allTracks.length - 1) * trackSpacing
      allTracks.forEach((t, i) => trackToX.set(t, i * trackSpacing - totalW / 2))

      // ================================================================
      // Step 3: Position nodes — unified x per track, y per depth
      // ================================================================
      const totalH = (fullLevels.length - 1) * rowSpacing

      for (let li = 0; li < fullLevels.length; li++) {
        const y = li * rowSpacing - totalH / 2

        for (const arrIdx of fullLevels[li]) {
          const node = ol[parseInt(arrIdx)]
          if (!node) continue
          const isBuiltin = node.operator.name === 'start' || node.operator.name === 'end'
          const x = isBuiltin ? 0 : (trackToX.get(nodeTrack.get(arrIdx) ?? 0) ?? 0)
          moveNode(node.index, { x, y })
        }
      }
    } else {
      const rowSpacing = 56
      const trackSpacing = 200
      const rankGap = trackSpacing

      // ================================================================
      // Step 1: Per-rank — independent 3-step pipeline for each rank
      // ================================================================
      type RankResult = {
        rank: number
        nodeTrack: Map<string, number>
        trackToX: Map<number, number>
        maxWidth: number
        layouts: { levelIdx: number; arrIndices: string[] }[]
      }
      const rankResults: RankResult[] = []

      for (const rank of ranks) {
        // Extract this rank's sub-levels (only levels where this rank has nodes)
        const rankLevels: string[][] = []
        const levelIdxMap: number[] = [] // rankLevels[i] → fullLevels index
        for (let li = 0; li < fullLevels.length; li++) {
          const rankNodes = fullLevels[li].filter(arrIdx => {
            const n = ol[parseInt(arrIdx)]
            return n && n.rank === rank
              && n.operator.name !== 'start' && n.operator.name !== 'end'
          })
          if (rankNodes.length > 0) {
            rankLevels.push(rankNodes)
            levelIdxMap.push(li)
          }
        }
        if (rankLevels.length === 0) continue

        // --- Step 1a: track assignment (same logic as single rank) ---
        const nodeTrack = new Map<string, number>()
        let nextTrack = 0

        if (rankLevels.length > 0) {
          const sorted = [...rankLevels[0]].sort((a, b) => {
            const na = ol[parseInt(a)]; const nb = ol[parseInt(b)]
            return (na?.index ?? 0) - (nb?.index ?? 0)
          })
          for (const arrIdx of sorted) nodeTrack.set(arrIdx, nextTrack++)
        }

        for (let iter = 0; iter < 10; iter++) {
          let changed = false
          for (let li = 1; li < rankLevels.length; li++) {
            const usedTracks = new Set<number>()
            const sorted = [...rankLevels[li]].sort((a, b) => {
              const d = avgPredTrack(a, directPreds, nodeTrack) - avgPredTrack(b, directPreds, nodeTrack)
              if (d !== 0) return d
              const na = ol[parseInt(a)]; const nb = ol[parseInt(b)]
              return (na?.index ?? 0) - (nb?.index ?? 0)
            })
            for (const arrIdx of sorted) {
              const preds = directPreds.get(arrIdx) || new Set()
              const predArr = [...preds].sort((a, b) =>
                (nodeTrack.get(a) ?? Infinity) - (nodeTrack.get(b) ?? Infinity))
              let track: number | undefined
              for (const pred of predArr) {
                const pt = nodeTrack.get(pred)
                if (pt !== undefined && !usedTracks.has(pt)) { track = pt; break }
              }
              if (track === undefined || usedTracks.has(track)) track = nextTrack++
              if (nodeTrack.get(arrIdx) !== track) { nodeTrack.set(arrIdx, track); changed = true }
              usedTracks.add(track)
            }
          }
          if (!changed) break
        }

        // --- Step 1b: global compaction within this rank ---
        const allTracks = [...new Set(nodeTrack.values())].sort((a, b) => a - b)
        const trackToX = new Map<number, number>()
        const maxWidth = (allTracks.length - 1) * trackSpacing
        allTracks.forEach((t, i) => trackToX.set(t, i * trackSpacing - maxWidth / 2))

        const layouts: { levelIdx: number; arrIndices: string[] }[] = []
        for (let rli = 0; rli < rankLevels.length; rli++) {
          layouts.push({ levelIdx: levelIdxMap[rli], arrIndices: rankLevels[rli] })
        }

        rankResults.push({ rank, nodeTrack, trackToX, maxWidth, layouts })
      }

      // ================================================================
      // Step 2: Position rank columns side by side
      // ================================================================
      rankResults.sort((a, b) => a.rank - b.rank)
      const rankOffset = new Map<number, number>()
      let cumX = 0
      for (const rr of rankResults) {
        rankOffset.set(rr.rank, cumX)
        cumX += rr.maxWidth + rankGap
      }
      const totalRankWidth = cumX - rankGap

      // ================================================================
      // Step 3: Apply positions — unified x per track within each rank
      // ================================================================
      const totalH = (fullLevels.length - 1) * rowSpacing

      for (const rr of rankResults) {
        const rankCenterX = rankOffset.get(rr.rank)! + rr.maxWidth / 2 - totalRankWidth / 2

        for (const { levelIdx, arrIndices } of rr.layouts) {
          const y = levelIdx * rowSpacing - totalH / 2

          for (const arrIdx of arrIndices) {
            const node = ol[parseInt(arrIdx)]
            if (!node) continue
            const x = rankCenterX + (rr.trackToX.get(rr.nodeTrack.get(arrIdx) ?? 0) ?? 0)
            moveNode(node.index, { x, y })
          }
        }
      }

      // Builtins (start/end): center globally across all rank columns
      for (let li = 0; li < fullLevels.length; li++) {
        for (const arrIdx of fullLevels[li]) {
          const node = ol[parseInt(arrIdx)]
          if (!node) continue
          const isBuiltin = node.operator.name === 'start' || node.operator.name === 'end'
          if (!isBuiltin) continue
          const y = li * rowSpacing - totalH / 2
          moveNode(node.index, { x: 0, y })
        }
      }
    }
    setTimeout(() => rfFitView({ padding: 0.2, duration: 200 }), 50)
  }, [orderedList, rawToOrder, ranks])

  // auto-layout on node count change or layer reorder (debounced)
  const prevLen = useRef(storeNodes.length)
  const prevReorder = useRef(reorderCount)
  const layoutTimer = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  useEffect(() => {
    const countChanged = storeNodes.length !== prevLen.current
    const reordered = reorderCount !== prevReorder.current
    prevLen.current = storeNodes.length
    prevReorder.current = reorderCount
    if ((!countChanged && !reordered) || storeNodes.length === 0) return
    clearTimeout(layoutTimer.current)
    layoutTimer.current = setTimeout(() => autoLayout(), 120)
    return () => clearTimeout(layoutTimer.current)
  }, [storeNodes.length, reorderCount, autoLayout])

  const onDragOver = useCallback((event: React.DragEvent) => {
    event.preventDefault()
    event.dataTransfer.dropEffect = 'move'
  }, [])

  const onDrop = useCallback((event: React.DragEvent) => {
    event.preventDefault()

    const opName = event.dataTransfer.getData('application/kepler-operator')
    if (opName) {
      const store = useModelStore.getState()
      const op = store.operators.find((o) => o.name === opName)
        || store.customOperators.find((o) => o.name === opName)
      if (!op) return
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY })
      addNode(op, { x: flowPos.x - 85, y: flowPos.y - 30 })
      return
    }

    const moduleName = event.dataTransfer.getData('application/kepler-module')
    if (moduleName) {
      const store = useModelStore.getState()
      const flowPos = screenToFlowPosition({ x: event.clientX, y: event.clientY })

      // Check built-in modules first (from store.moduleDefs), then custom
      let opsToAdd: OperatorDef[] | null = null
      if (store.moduleDefs[moduleName]) {
        opsToAdd = store.moduleDefs[moduleName]
      } else {
        const customMod = store.customModules.find((m) => m.name === moduleName)
        if (customMod) {
          const allOps = [...store.operators, ...store.customOperators]
          opsToAdd = customMod.operatorNames
            .map((name) => allOps.find((o) => o.name === name))
            .filter(Boolean) as OperatorDef[]
        }
      }
      if (!opsToAdd || opsToAdd.length === 0) return

      const n = opsToAdd.length
      opsToAdd.forEach((op, i) => {
        addNode(op, {
          x: flowPos.x + (i - (n - 1) / 2) * 30 - 85,
          y: flowPos.y + i * 30 - 30,
        })
      })
    }
  }, [addNode, screenToFlowPosition])

  // keyboard panning
  useEffect(() => {
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement || e.target instanceof HTMLSelectElement) return
      const step = 50
      const vp = getViewport()
      switch (e.key) {
        case 'ArrowLeft': setViewport({ x: vp.x + step, y: vp.y, zoom: vp.zoom }); break
        case 'ArrowRight': setViewport({ x: vp.x - step, y: vp.y, zoom: vp.zoom }); break
        case 'ArrowUp': setViewport({ x: vp.x, y: vp.y + step, zoom: vp.zoom }); break
        case 'ArrowDown': setViewport({ x: vp.x, y: vp.y - step, zoom: vp.zoom }); break
      }
    }
    window.addEventListener('keydown', onKeyDown)
    return () => window.removeEventListener('keydown', onKeyDown)
  }, [getViewport, setViewport])

  return (
    <div className="canvas-wrap">
      <ReactFlow nodes={rfNodes} edges={allEdges} nodeTypes={nodeTypes}
        onConnect={onConnect} onNodeDragStop={onNodeDragStop}
        onEdgesChange={onEdgesChange}
        onEdgeDoubleClick={onEdgeDoubleClick}
        onDragOver={onDragOver}
        onDrop={onDrop}
        defaultEdgeOptions={defaultEdgeOptions}
        edgesFocusable
        fitView>
        <Background gap={16} /><Controls />
        <div className="auto-layout-btn" onClick={autoLayout} title="自动布局">自动布局</div>
        <ArrowPad getViewport={getViewport} setViewport={setViewport} rfFitView={rfFitView} />
      </ReactFlow>
    </div>
  )
}
