import { useRef, useEffect } from 'react'

interface Props {
  side: 'left' | 'right'
  defaultWidth: number
  minWidth?: number
  maxWidth?: number
  children: React.ReactNode
}

export default function ResizablePanel({ side, defaultWidth, minWidth = 280, maxWidth = 750, children }: Props) {
  const panelRef = useRef<HTMLDivElement>(null)
  const handleRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handle = handleRef.current
    const panel = panelRef.current
    if (!handle || !panel) return

    let dragging = false
    let startX = 0
    let startW = 0

    const onDown = (e: MouseEvent) => {
      e.preventDefault()
      dragging = true
      startX = e.clientX
      startW = panel.offsetWidth
      document.body.style.cursor = 'col-resize'
      document.body.style.userSelect = 'none'
    }

    const onMove = (e: MouseEvent) => {
      if (!dragging) return
      const delta = side === 'left' ? e.clientX - startX : startX - e.clientX
      const newW = Math.min(maxWidth, Math.max(minWidth, startW + delta))
      panel.style.width = newW + 'px'
    }

    const onUp = () => {
      dragging = false
      document.body.style.cursor = ''
      document.body.style.userSelect = ''
    }

    handle.addEventListener('mousedown', onDown)
    document.addEventListener('mousemove', onMove)
    document.addEventListener('mouseup', onUp)

    // init
    panel.style.width = defaultWidth + 'px'

    return () => {
      handle.removeEventListener('mousedown', onDown)
      document.removeEventListener('mousemove', onMove)
      document.removeEventListener('mouseup', onUp)
    }
  }, [side, defaultWidth, minWidth, maxWidth])

  return (
    <>
      {side === 'left' && <div ref={panelRef} className="resize-panel">{children}</div>}
      {side === 'left' && <div ref={handleRef} className="resize-handle" />}
      {side === 'right' && <div ref={handleRef} className="resize-handle" />}
      {side === 'right' && <div ref={panelRef} className="resize-panel">{children}</div>}
    </>
  )
}
