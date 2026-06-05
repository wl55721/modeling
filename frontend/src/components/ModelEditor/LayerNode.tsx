import { memo, useRef, useLayoutEffect, useState } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useModelStore } from '../../stores/model'
import { cn } from '../../utils/classnames'
import type { OperatorDef } from '../../types/model'

type LayerData = { index: number; operator: OperatorDef; label: string; colorIdx?: number; layerIdx?: number; rank?: number; orderIdx?: number }

const LAYER_COLORS = ['#3b6fb6', '#8e6bb8', '#4d8c57', '#c4504a', '#e67e22', '#2e86c1', '#8e44ad', '#16a085']

function LayerNode({ data }: NodeProps) {
  const d = data as unknown as LayerData
  const selectedNodeId = useModelStore((s) => s.selectedNodeId)
  const selectNode = useModelStore((s) => s.selectNode)
  const removeNode = useModelStore((s) => s.removeNode)

  const displayIdx = d.orderIdx ?? 0
  const borderColor = LAYER_COLORS[(d.colorIdx ?? 0) % LAYER_COLORS.length]

  const op = d.operator
  const isStart = op.name === 'start'
  const isEnd = op.name === 'end'
  const isBuiltin = isStart || isEnd
  const layerLabel = (d.layerIdx === -1 || d.layerIdx === -2) ? '层前' : (d.layerIdx === 999999 || d.layerIdx === 1000000) ? '层后' : `Layer ${d.layerIdx}`

  const nameRef = useRef<HTMLSpanElement>(null)
  const [nameSize, setNameSize] = useState(11)
  useLayoutEffect(() => {
    const el = nameRef.current
    if (el && el.scrollWidth > el.clientWidth && nameSize === 11) {
      setNameSize(10)
    }
  }, [d.operator.name, nameSize])

  const nodeStyle: React.CSSProperties = isBuiltin
    ? {
        width: 85,
        backgroundColor: isStart ? '#fff' : '#1a1a1a',
        borderColor: isStart ? '#ccc' : '#333',
        color: isStart ? '#1a1a1a' : '#fff',
      }
    : { backgroundColor: borderColor + '66' }

  return (
    <div
      className={cn('layer-node', selectedNodeId === String(d.index) && 'selected')}
      style={nodeStyle}
      onClick={() => selectNode(String(d.index))}
    >
      <div className="node-tooltip">
        <div>{layerLabel}{d.rank != null && d.rank > 0 ? ` [Rank ${d.rank}]` : ''}</div>
        <div>{op.module}</div>
      </div>
      <Handle type="target" position={Position.Top} />
      <div className="node-body">
        <div className="node-title">
          <span className="node-index" style={isBuiltin ? { color: isStart ? '#999' : '#888' } : undefined}>#{displayIdx}</span>
          <span ref={nameRef} className="node-optype" style={{ fontSize: nameSize, color: isBuiltin ? nodeStyle.color : undefined }}>{d.operator.name}</span>
          {!isBuiltin && <button className="node-remove" onClick={(e) => { e.stopPropagation(); removeNode(d.index) }}>×</button>}
        </div>
      </div>
      <Handle type="source" position={Position.Bottom} />
    </div>
  )
}

export default memo(LayerNode)
