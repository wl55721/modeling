import { useState, useRef, useEffect, useCallback } from 'react'
import { MODULE_GROUPS } from '../../constants/operators'
import { cn } from '../../utils/classnames'

export default function ModuleSelect({ value, onChange }: {
  value: string
  onChange: (v: string) => void
}) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    if (!open) return
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', onDown)
    return () => document.removeEventListener('mousedown', onDown)
  }, [open])

  const handleSelect = useCallback((m: string) => {
    onChange(m)
    setOpen(false)
  }, [onChange])

  return (
    <div className="mod-select" ref={ref}>
      <button className={cn('mod-select-btn', open && 'open')}
        onClick={() => setOpen(!open)}>
        <span className={cn('mod-select-val', !value && 'placeholder')}>
          {value || '选择 Module...'}
        </span>
        <span className="mod-select-arrow">&#9662;</span>
      </button>
      {open && (
        <div className="mod-select-drop">
          {MODULE_GROUPS.map((g) => (
            <div key={g.label} className="mod-select-group">
              <div className="mod-select-group-label">{g.label}</div>
              <div className="mod-select-grid">
                {g.modules.map((m) => (
                  <button key={m}
                    className={cn('mod-select-opt', m === value && 'active')}
                    onClick={() => handleSelect(m)}>
                    {m}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
