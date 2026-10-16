// The notifications bell in Dashtreme's header dropdown (fly_trader/ui/api/notifications.py): warnings and errors from
// the event log, the rails' circuit events and every live-money fill, newest first. Unread = newer than what this browser
// last marked read.
import { useEffect, useRef, useState } from 'react'
import { NavLink } from 'react-router'
import { useApp } from '../context'
import { useNow, useStored, useTopic } from '../lib/hooks'
import { ago, dt } from '../lib/format'
import { Icon, Segmented } from './ui'

type Note = { id: string; ts: string; level: 'info' | 'warning' | 'error' | 'live'; source: string; text: string; chain?: string; mint?: string; ticker?: string }

// level → (icon, Dashtreme's light background, icon colour)
const LEVEL_LOOK: Record<string, [string, string, string]> = {
  error: ['bx-error-circle', 'rgb(244 17 39 / 22%)', '#ff7d90'], warning: ['bx-error', 'rgb(255 193 7 / 20%)', '#ffd65c'],
  live: ['bx-money', 'rgb(255 47 185 / 26%)', '#ff8fd8'], info: ['bx-info-circle', 'rgb(13 202 240 / 18%)', '#7ad8f5'],
}
const LEVEL_TITLE: Record<string, string> = { error: 'Error', warning: 'Warning', live: 'Live money', info: 'Rails' }
const FILTERS = ['all', 'error', 'warning', 'live', 'info'] as const
type Filter = (typeof FILTERS)[number]

export function Notifications() {
  const { data } = useTopic<{ items: Note[] }>('notifications.recent')
  const { openInspector } = useApp()
  const [readTs, setReadTs] = useStored<number>('notif_read_ms', 0)
  const [open, setOpen] = useState(false)
  const [filter, setFilter] = useStored<Filter>('notif_filter', 'all')
  const box = useRef<HTMLDivElement>(null)
  const now = useNow(10000)
  const items = data?.items ?? []
  const unread = items.filter((x) => Date.parse(x.ts) > readTs).length
  const shown = filter === 'all' ? items : items.filter((x) => x.level === filter)
  useEffect(() => {
    if (!open) return
    const click = (e: MouseEvent) => { if (box.current && !box.current.contains(e.target as Node)) setOpen(false) }
    const key = (e: KeyboardEvent) => e.key === 'Escape' && setOpen(false)
    document.addEventListener('mousedown', click)
    window.addEventListener('keydown', key)
    return () => { document.removeEventListener('mousedown', click); window.removeEventListener('keydown', key) }
  }, [open])
  const markRead = () => setReadTs(Math.max(readTs, ...items.map((x) => Date.parse(x.ts))))
  const go = (x: Note) => {
    if (x.ticker) openInspector({ kind: 'market', ticker: x.ticker })
    else if (x.mint && x.chain) openInspector({ kind: 'token', chain: x.chain, mint: x.mint })
    else return
    setOpen(false)
  }
  return (
    <div ref={box} className="dropdown-large">
      <a className="nav-link position-relative" role="button" tabIndex={0} aria-label={`Notifications${unread ? `: ${unread} unread` : ''}`} aria-expanded={open}
        onClick={() => setOpen(!open)} onKeyDown={(e) => (e.key === 'Enter' || e.key === ' ') && setOpen(!open)}>
        {unread > 0 && <span className="alert-count">{unread > 99 ? '99+' : unread}</span>}
        <i className="bx bx-bell" />
      </a>
      {open && (
        <div className="dropdown-menu dropdown-menu-end show" role="dialog" aria-label="Notifications"
          style={{ position: 'absolute', right: 0, top: 'calc(100% + 14px)', width: 'min(400px, 94vw)', background: '#24152f', border: '1px solid rgb(255 255 255 / 14%)' }}>
          <div className="msg-header" style={{ background: '#2b1938' }}>
            <p className="msg-header-title">Notifications</p>
            <div className="d-flex align-items-center gap-2">
              {unread > 0 && <p className="msg-header-badge">{unread} New</p>}
              <button type="button" className="link-btn msg-header-clear" style={{ fontSize: 12.5 }} onClick={markRead} disabled={!unread}>Mark all read</button>
            </div>
          </div>
          <div className="px-3 py-2" style={{ borderBottom: '1px solid rgb(255 255 255 / 12%)' }}>
            <Segmented options={FILTERS} value={filter} onChange={setFilter} labels={{ all: 'All', error: 'Errors', warning: 'Warnings', live: 'Live', info: 'Info' }} />
          </div>
          <div className="header-notifications-list" style={{ height: 'auto', maxHeight: 380, overflowY: 'auto' }}>
            {shown.length === 0 && <div className="dropdown-item" style={{ color: 'var(--fly-muted)' }}>{data ? 'Nothing to report.' : 'Loading…'}</div>}
            {shown.map((x) => {
              const [icon, bg, color] = LEVEL_LOOK[x.level] ?? LEVEL_LOOK.info
              const clickable = !!(x.ticker || (x.mint && x.chain))
              const isNew = Date.parse(x.ts) > readTs
              return (
                <div key={x.id} className="dropdown-item" role={clickable ? 'button' : undefined} onClick={clickable ? () => go(x) : undefined}
                  style={{ cursor: clickable ? 'pointer' : 'default', whiteSpace: 'normal', background: isNew ? 'rgb(255 255 255 / 6%)' : undefined,
                    borderLeft: isNew ? '3px solid var(--fly-magenta)' : '3px solid transparent' }}>
                  <div className="d-flex align-items-center">
                    <div className="notify" style={{ background: bg, color, flexShrink: 0 }}><Icon name={icon} /></div>
                    <div className="flex-grow-1" style={{ minWidth: 0 }}>
                      <h6 className="msg-name">{LEVEL_TITLE[x.level] ?? x.level} · {x.source}
                        <span className="msg-time float-end" title={`${dt(x.ts, true)} UTC`}>{ago(x.ts, now)}</span></h6>
                      <p className="msg-info" style={{ whiteSpace: 'normal', wordBreak: 'break-word', color: 'rgb(255 255 255 / 72%)' }}>{x.text}</p>
                    </div>
                  </div>
                </div>
              )
            })}
          </div>
          <div className="msg-footer d-flex justify-content-center gap-4">
            <NavLink to="/logs" onClick={() => setOpen(false)} style={{ color: '#fff' }}><Icon name="logs" /> Logs</NavLink>
            <NavLink to="/processes" onClick={() => setOpen(false)} style={{ color: '#fff' }}><Icon name="settings" /> Processes &amp; event log</NavLink>
          </div>
        </div>
      )}
    </div>
  )
}
