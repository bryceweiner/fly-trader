// Global search (fly_trader/ui/api/search.py): a token by symbol or address, a Kalshi market by ticker or title, a
// transaction by signature or hash. ↑↓ to move, Enter to open (the inspector, or the page it lives on), Esc to close;
// "/" anywhere focuses it.
import { useEffect, useRef, useState } from 'react'
import { useNavigate } from 'react-router'
import { useApp, type InspectTarget } from '../context'
import { readTopic } from '../lib/api'
import { Icon } from './ui'

type Target = InspectTarget | { kind: 'route'; to: string } | null
type Result = { kind: 'token' | 'market' | 'tx' | 'note'; label: string; sub: string; target: Target }
type Answer = { q: string; results: Result[]; note: string | null }

const KIND_ICON: Record<string, string> = { token: 'bx-coin', market: 'bx-target-lock', tx: 'bx-receipt', note: 'bx-info-circle' }
const KIND_LOOK: Record<string, [string, string]> = {
  token: ['rgb(255 47 185 / 24%)', '#ff8fd8'], market: ['rgb(13 202 240 / 18%)', '#7ad8f5'], tx: ['rgb(139 92 246 / 26%)', '#c4b1ff'], note: ['rgb(255 255 255 / 12%)', '#fff'],
}
const DEBOUNCE_MS = 300

export function SearchBox() {
  const { openInspector } = useApp()
  const navigate = useNavigate()
  const [text, setText] = useState('')
  const [open, setOpen] = useState(false)
  const [ans, setAns] = useState<Answer | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [active, setActive] = useState(0)
  const input = useRef<HTMLInputElement>(null)
  const box = useRef<HTMLDivElement>(null)
  const seq = useRef(0)
  const q = text.trim()

  useEffect(() => {
    if (q.length < 2) { setAns(null); setBusy(false); setError(null); return }
    const my = ++seq.current
    setBusy(true)
    const h = setTimeout(() => {
      readTopic<Answer>('search.query', { q })
        .then((a) => { if (my === seq.current) { setAns(a); setError(null); setActive(0) } })
        .catch((e: Error) => { if (my === seq.current) setError(e.message) })
        .finally(() => { if (my === seq.current) setBusy(false) })
    }, DEBOUNCE_MS)
    return () => clearTimeout(h)
  }, [q])

  useEffect(() => {
    const key = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement | null
      if (e.key === '/' && !(el && (el.tagName === 'INPUT' || el.tagName === 'TEXTAREA' || el.tagName === 'SELECT' || el.isContentEditable))) {
        e.preventDefault(); input.current?.focus()
      }
    }
    const click = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false) }
    window.addEventListener('keydown', key)
    document.addEventListener('mousedown', click)
    return () => { window.removeEventListener('keydown', key); document.removeEventListener('mousedown', click) }
  }, [])

  const results = (ans && ans.q === q ? ans.results : []).filter((r) => r.target)
  const notes = ans && ans.q === q ? ans.results.filter((r) => !r.target) : []

  const choose = (r: Result | undefined) => {
    if (!r || !r.target) return
    if (r.target.kind === 'route') navigate(r.target.to)
    else openInspector(r.target)
    setOpen(false); setText(''); input.current?.blur()
  }

  const onKey = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'ArrowDown') { e.preventDefault(); setOpen(true); setActive((a) => Math.min(results.length - 1, a + 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setActive((a) => Math.max(0, a - 1)) }
    else if (e.key === 'Enter') { e.preventDefault(); choose(results[active] ?? results[0]) }
    else if (e.key === 'Escape') { setOpen(false); input.current?.blur() }
  }

  const status = q.length === 0 ? null : q.length < 2 ? 'Type at least 2 characters.' : error ? `Search failed: ${error}`
    : busy && !results.length ? 'Searching…' : ans && ans.q === q && !results.length ? ans.note || 'Nothing matches.' : null

  const kindName = (k: string) => (k === 'token' ? 'Token' : k === 'market' ? 'Kalshi market' : k === 'tx' ? 'Transaction' : 'Note')
  return (
    <div className="search-bar-box dropdown-large" ref={box} style={{ maxWidth: 600 }}>
      <i className="bx bx-search" />
      <input ref={input} className="form-control search-control" placeholder="Search a token, mint, ticker or signature…  ( / )" value={text}
        role="combobox" aria-expanded={open && q.length > 0} aria-controls="search-results" aria-autocomplete="list"
        onChange={(e) => { setText(e.target.value); setOpen(true) }} onFocus={() => setOpen(true)} onKeyDown={onKey} />
      {open && q.length > 0 && (
        <div className="dropdown-menu show" id="search-results" role="listbox"
          style={{ position: 'absolute', left: 0, top: 'calc(100% + 12px)', width: 'min(600px, 94vw)', background: '#24152f', border: '1px solid rgb(255 255 255 / 14%)', padding: 0 }}>
          <div className="msg-header" style={{ background: '#2b1938' }}>
            <p className="msg-header-title">Search</p>
            {results.length > 0 && <p className="msg-header-badge">{results.length} result{results.length === 1 ? '' : 's'}</p>}
            {busy && <span className="spinner-border spinner-border-sm" style={{ color: 'rgb(255 255 255 / 60%)' }} />}
          </div>
          <div style={{ maxHeight: 420, overflowY: 'auto' }}>
            {status && <div className="dropdown-item" style={{ color: 'var(--fly-muted)', whiteSpace: 'normal' }}>{status}</div>}
            {results.map((r, i) => {
              const [bg, color] = KIND_LOOK[r.kind] ?? KIND_LOOK.note
              return (
                <div key={`${r.kind}:${r.label}:${i}`} role="option" aria-selected={i === active} className="dropdown-item"
                  style={{ cursor: 'pointer', whiteSpace: 'normal', background: i === active ? 'rgb(255 255 255 / 12%)' : undefined }}
                  onMouseEnter={() => setActive(i)} onMouseDown={(e) => { e.preventDefault(); choose(r) }}>
                  <div className="d-flex align-items-center">
                    <div className="notify" style={{ background: bg, color, flexShrink: 0 }}><Icon name={KIND_ICON[r.kind] ?? 'bx-circle'} /></div>
                    <div className="flex-grow-1" style={{ minWidth: 0 }}>
                      <h6 className={`msg-name ${r.kind === 'market' || r.kind === 'tx' ? 'mono' : ''}`}>{r.label}
                        <span className="msg-time float-end">{kindName(r.kind)} · {r.target?.kind === 'route' ? `→ ${r.target.to}` : 'inspect'}</span></h6>
                      <p className="msg-info" style={{ whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', color: 'rgb(255 255 255 / 65%)' }}>{r.sub}</p>
                    </div>
                  </div>
                </div>
              )
            })}
            {notes.map((r, i) => <div key={`n${i}`} className="dropdown-item" style={{ whiteSpace: 'normal', color: 'var(--fly-muted)' }}><Icon name="info" /> {r.label}: {r.sub}</div>)}
          </div>
          <div className="msg-footer" style={{ fontSize: 12.5, fontWeight: 400, color: 'rgb(255 255 255 / 60%)' }}>↑ ↓ to move · Enter to open · Esc to close</div>
        </div>
      )}
    </div>
  )
}
