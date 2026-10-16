// The console's building blocks in Dashtreme's markup: cards with a title row and rule, data/static widgets (label,
// value, icon, delta, sparkline, progress), colour static widgets, border alerts, breadcrumb page heads, progress lists.
import { useId, useLayoutEffect, useRef, useState, type ReactNode } from 'react'
import { createPortal } from 'react-dom'

// Semantic icon names (the server sends these with status widgets) → Boxicons.
const ICONS: Record<string, string> = {
  payments: 'bx-money', receipt: 'bx-receipt', candlestick: 'bx-line-chart', psychology: 'bx-chip', block: 'bx-block', upgrade: 'bx-up-arrow-circle',
  neurology: 'bx-brain', swap: 'bx-transfer-alt', model_training: 'bx-analyse', sensors: 'bx-broadcast', history: 'bx-history', visibility: 'bx-show',
  query_stats: 'bx-search-alt', savings: 'bx-coin-stack', dashboard: 'bx-grid-alt', trades: 'bx-transfer', database: 'bx-data', link: 'bx-link',
  settings: 'bx-cog', shield: 'bx-shield-quarter', logs: 'bx-terminal', info: 'bx-info-circle', warning: 'bx-error', error: 'bx-error-circle',
  success: 'bx-check-circle', play: 'bx-play', stop: 'bx-stop', pause: 'bx-pause', reset: 'bx-reset', undo: 'bx-undo', refresh: 'bx-refresh',
  score: 'bx-target-lock', wallet: 'bx-wallet', bell: 'bx-bell', search: 'bx-search', download: 'bx-download', external: 'bx-link-external',
  copy: 'bx-copy', folder: 'bx-folder-open', terminal: 'bx-terminal', autorenew: 'bx-sync', race: 'bx-flag', category: 'bx-category',
  pulse: 'bx-pulse', rocket: 'bx-rocket', lock: 'bx-lock', key: 'bx-key', trash: 'bx-trash', time: 'bx-time', fly: 'bx-bug',
  wealth: 'bx-wallet-alt', pnl: 'bx-dollar-circle', count: 'bx-list-check', open: 'bx-briefcase-alt-2', drawdown: 'bx-trending-down',
  up: 'bx-trending-up', target: 'bx-bullseye', calendar: 'bx-calendar', layers: 'bx-layer', chart: 'bx-bar-chart-alt-2', percent: 'bx-pie-chart-alt-2',
}

export function Icon({ name, className = '' }: { name?: string | null; className?: string }) {
  if (!name) return null
  const cls = name.startsWith('bx') ? name : ICONS[name] || 'bx-circle'
  return <i className={`bx ${cls} ${className}`} aria-hidden="true" />
}

/** A hover/focus bubble with a help text, positioned against the viewport (never clipped by cards or tables). */
export function Tip({ text, children, wide }: { text?: ReactNode; children: ReactNode; wide?: boolean }) {
  const ref = useRef<HTMLSpanElement>(null)
  const bub = useRef<HTMLDivElement>(null)
  const [open, setOpen] = useState(false)
  const [pos, setPos] = useState<{ left: number; top: number } | null>(null)
  useLayoutEffect(() => {
    if (!open || !ref.current || !bub.current) return
    const r = ref.current.getBoundingClientRect(), b = bub.current.getBoundingClientRect()
    let left = r.left + r.width / 2 - b.width / 2
    left = Math.max(8, Math.min(left, window.innerWidth - b.width - 8))
    let top = r.bottom + 8
    if (top + b.height > window.innerHeight - 8) top = r.top - b.height - 8
    setPos({ left, top })
  }, [open])
  if (!text) return <>{children}</>
  return (
    <span className="tip-wrap" ref={ref} onMouseEnter={() => setOpen(true)} onMouseLeave={() => { setOpen(false); setPos(null) }}
      onFocus={() => setOpen(true)} onBlur={() => { setOpen(false); setPos(null) }}>
      {children}
      {open && createPortal(
        <div ref={bub} className="tip-bubble" style={{ left: pos?.left ?? -9999, top: pos?.top ?? -9999, maxWidth: wide ? 520 : undefined }}>{text}</div>,
        document.body)}
    </span>
  )
}

export function Help({ text }: { text?: ReactNode }) {
  if (!text) return null
  return <Tip text={text}><i className="bx bx-help-circle help-dot" tabIndex={0} /></Tip>
}

export type BadgeData = { text: string; color: string; icon?: string | null; help?: string | null }

/** A small pill (tables, worker cards); page-level state uses colour widgets instead. */
export function Badge({ text, color, icon, help }: BadgeData) {
  const b = <span className={`fbadge ${color}`} tabIndex={help ? 0 : undefined}><Icon name={icon} />{text}</span>
  return help ? <Tip text={help}>{b}</Tip> : b
}

/** Dashtreme's card: a title row (icon, title, actions on the right), a rule, the body. */
export function Card({ title, icon, sub, actions, children, className = '', bodyClass = '' }: {
  title?: ReactNode; icon?: string; sub?: ReactNode; actions?: ReactNode; children?: ReactNode; className?: string; bodyClass?: string
}) {
  return (
    <div className={`card ${className}`}>
      {(title || actions) && (
        <>
          <div className="card-head">
            {title && <h6 className="title"><Icon name={icon} />{title}</h6>}
            {actions && <div className="actions">{actions}</div>}
            {sub && <span className="sub">{sub}</span>}
          </div>
          <hr />
        </>
      )}
      <div className={`card-body ${bodyClass}`}>{children}</div>
    </div>
  )
}

/** Dashtreme's breadcrumb row: group | home › page, tools on the right. */
export function Breadcrumb({ group, page, icon, children }: { group: string; page: string; icon?: string; children?: ReactNode }) {
  return (
    <div className="page-breadcrumb">
      <div className="breadcrumb-title pe-3">{group}</div>
      <nav aria-label="breadcrumb" className="ps-1">
        <ol className="breadcrumb mb-0 p-0 align-items-center">
          <li className="breadcrumb-item"><Icon name={icon || 'bx-home-alt'} /></li>
          <li className="breadcrumb-item active" aria-current="page">{page}</li>
        </ol>
      </nav>
      {children && <div className="tools">{children}</div>}
    </div>
  )
}

/** Dashtreme's section heading: an uppercase title over a rule. */
export function SectionTitle({ children }: { children: ReactNode }) {
  return <><h6 className="section-title">{children}</h6><hr /></>
}

export type WidgetColor = 'success' | 'danger' | 'warning' | 'info' | 'violet' | 'magenta' | 'teal' | 'dark'
/** The server's semantic colours (status) → colour-widget colours. */
export const STATE_COLOR: Record<string, WidgetColor> = { green: 'success', red: 'danger', orange: 'warning', blue: 'info', violet: 'violet', magenta: 'magenta', gray: 'dark', teal: 'teal' }

function deltaClass(delta: ReactNode, tone: 'normal' | 'off' | 'inverse'): '' | 'up' | 'down' {
  if (delta === null || delta === undefined || delta === '' || tone === 'off') return ''
  const s = String(delta).trim()
  const neg = s.startsWith('-') || s.startsWith('−')
  return (tone === 'inverse' ? !neg : neg) ? 'down' : 'up'
}

type WidgetProps = {
  label: ReactNode; value: ReactNode; delta?: ReactNode | null; tone?: 'normal' | 'off' | 'inverse'; help?: ReactNode
  icon?: string | null; spark?: number[] | null; sparkType?: 'area' | 'bar'; progress?: number | null; color?: WidgetColor | null; compact?: boolean
}

function WidgetInner({ label, value, delta, tone = 'normal', help, icon, spark, sparkType = 'area', progress }: WidgetProps) {
  const dc = deltaClass(delta, tone)
  return (
    <>
      <div className="w-body">
        <div className="w-text">
          <p className="w-label">{label}<Help text={help} /></p>
          <h4 className="w-value">{value}</h4>
          {delta !== null && delta !== undefined && delta !== '' && (
            <p className={`w-delta ${dc}`}>{dc === 'up' && <i className="bx bxs-up-arrow align-middle" style={{ fontSize: 10 }} />}
              {dc === 'down' && <i className="bx bxs-down-arrow align-middle" style={{ fontSize: 10 }} />}{delta}</p>
          )}
        </div>
        {icon && <div className="w-icon"><Icon name={icon} /></div>}
      </div>
      {spark && spark.length > 1 && <Sparkline data={spark} type={sparkType} />}
      {progress !== null && progress !== undefined && (
        <div className="w-progress"><div className="progress"><div className="progress-bar" style={{ width: `${Math.max(0, Math.min(1, progress)) * 100}%` }} /></div></div>
      )}
    </>
  )
}

/**
 * Streamlit's st.metric as a Dashtreme widget tile (inside a card's `Metrics` grid): label, value, an optional delta
 * (tone "normal" colours it by sign, "off" keeps it neutral), icon, sparkline, progress bar or colour.
 */
export function Metric(p: WidgetProps & { plain?: boolean }) {
  const cls = `widget ${p.plain ? 'plain' : ''} ${p.compact ? 'compact' : ''} ${p.color ? `cw cw-${p.color}` : ''}`
  return <div className={cls}><WidgetInner {...p} /></div>
}

/** A standalone Dashtreme widget card (page-level KPI rows: put several in a `WidgetGrid`). */
export function Widget(p: WidgetProps) {
  return <div className={`card radius-10 widget ${p.compact ? 'compact' : ''} ${p.color ? `cw cw-${p.color}` : ''}`}><WidgetInner {...p} /></div>
}

export function WidgetGrid({ children, min = 210 }: { children: ReactNode; min?: number }) {
  return <div className="widget-grid" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))` }}>{children}</div>
}

export function Metrics({ children, min = 160 }: { children: ReactNode; min?: number }) {
  return <div className="metrics" style={{ gridTemplateColumns: `repeat(auto-fit, minmax(${min}px, 1fr))` }}>{children}</div>
}

/** A smooth white sparkline (Dashtreme's data widgets): an area with a fading fill, or bars. */
export function Sparkline({ data, height = 52, type = 'area', color = '#fff' }: { data: number[]; height?: number; type?: 'area' | 'bar'; color?: string }) {
  const id = useId().replace(/:/g, '')
  const w = 300, pad = 4, vals = data.filter((v) => Number.isFinite(v))
  if (vals.length < 2) return null
  const lo = Math.min(...vals), hi = Math.max(...vals), span = hi - lo || 1
  const y = (v: number) => height - pad - ((v - lo) / span) * (height - 2 * pad)
  if (type === 'bar') {
    const bw = w / data.length
    return (
      <svg className="w-spark" viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" width="100%" height={height}>
        {data.map((v, i) => <rect key={i} x={i * bw + bw * 0.22} width={bw * 0.56} y={y(v)} height={height - y(v)} rx={Math.min(3, bw * 0.2)} fill={color} opacity={0.9} />)}
      </svg>
    )
  }
  const pts = data.map((v, i) => [(i / (data.length - 1)) * w, y(v)] as const)
  let d = `M${pts[0][0]},${pts[0][1]}`
  for (let i = 0; i < pts.length - 1; i++) {        // Catmull-Rom → Bézier: the template's smooth curves
    const p0 = pts[Math.max(0, i - 1)], p1 = pts[i], p2 = pts[i + 1], p3 = pts[Math.min(pts.length - 1, i + 2)]
    d += ` C${p1[0] + (p2[0] - p0[0]) / 6},${p1[1] + (p2[1] - p0[1]) / 6} ${p2[0] - (p3[0] - p1[0]) / 6},${p2[1] - (p3[1] - p1[1]) / 6} ${p2[0]},${p2[1]}`
  }
  return (
    <svg className="w-spark" viewBox={`0 0 ${w} ${height}`} preserveAspectRatio="none" width="100%" height={height}>
      <defs><linearGradient id={`g${id}`} x1="0" y1="0" x2="0" y2="1"><stop offset="0" stopColor={color} stopOpacity="0.45" /><stop offset="1" stopColor={color} stopOpacity="0" /></linearGradient></defs>
      <path d={`${d} L${w},${height} L0,${height} Z`} fill={`url(#g${id})`} />
      <path d={d} fill="none" stroke={color} strokeWidth="2.2" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

const CALLOUT_ICON = { info: 'bx-info-square', success: 'bxs-check-circle', warning: 'bx-error', error: 'bxs-message-square-x' } as const

/** Dashtreme's border alert (st.info / success / warning / error). */
export function Callout({ kind = 'info', icon, children }: { kind?: 'info' | 'success' | 'warning' | 'error'; icon?: string; children: ReactNode }) {
  return <div className={`callout ${kind}`} role={kind === 'error' ? 'alert' : undefined}><Icon name={icon || CALLOUT_ICON[kind]} /><div>{children}</div></div>
}

export function Caption({ children }: { children: ReactNode }) {
  return children ? <div className="caption">{children}</div> : null
}

export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>
}

/** The little markdown the server's sentences use: **bold**, `code`, paragraphs (blank line) and line breaks ("  \n"). */
export function Md({ text, className = '' }: { text?: string | null; className?: string }) {
  if (!text) return null
  const inline = (s: string, k: number) => {
    const out: ReactNode[] = []
    const re = /(\*\*[^*]+\*\*|`[^`]+`)/g
    let last = 0, m: RegExpExecArray | null, i = 0
    while ((m = re.exec(s))) {
      if (m.index > last) out.push(s.slice(last, m.index))
      const t = m[0]
      out.push(t.startsWith('**') ? <strong key={`${k}-${i++}`}>{t.slice(2, -2)}</strong> : <code key={`${k}-${i++}`}>{t.slice(1, -1)}</code>)
      last = m.index + t.length
    }
    if (last < s.length) out.push(s.slice(last))
    return out
  }
  return (
    <div className={`narrative ${className}`}>
      {text.split(/\n\s*\n/).map((para, pi) => (
        <p key={pi}>{para.split(/ {2}\n|\n/).map((line, li, arr) => <span key={li}>{inline(line, pi * 100 + li)}{li < arr.length - 1 && <br />}</span>)}</p>
      ))}
    </div>
  )
}

export function Segmented<T extends string>({ options, value, onChange, labels }: {
  options: readonly T[]; value: T; onChange: (v: T) => void; labels?: Partial<Record<T, ReactNode>>
}) {
  return (
    <div className="seg" role="tablist">
      {options.map((o) => <button key={o} type="button" role="tab" aria-selected={o === value} className={o === value ? 'on' : ''} onClick={() => onChange(o)}>{labels?.[o] ?? o}</button>)}
    </div>
  )
}

export function Pills<T extends string>({ options, value, onChange }: { options: readonly T[]; value: T[]; onChange: (v: T[]) => void }) {
  return (
    <div className="pills">
      {options.map((o) => {
        const on = value.includes(o)
        return <button key={o} type="button" aria-pressed={on} className={on ? 'on' : ''} onClick={() => onChange(on ? value.filter((x) => x !== o) : [...value, o])}>{o}</button>
      })}
    </div>
  )
}

export function Toggle({ label, checked, onChange, help, disabled }: { label: ReactNode; checked: boolean; onChange: (v: boolean) => void; help?: ReactNode; disabled?: boolean }) {
  return (
    <label className="form-check form-switch d-inline-flex align-items-center gap-2 mb-0" style={{ cursor: disabled ? 'default' : 'pointer' }}>
      <input className="form-check-input" type="checkbox" role="switch" checked={checked} disabled={disabled} onChange={(e) => onChange(e.target.checked)} />
      <span className="form-check-label" style={{ fontSize: 14, color: '#fff' }}>{label}</span>
      <Help text={help} />
    </label>
  )
}

export function Progress({ value, text }: { value: number; text?: ReactNode }) {
  const v = Math.max(0, Math.min(1, value || 0))
  return (
    <div className="my-2">
      {text && <div className="caption mb-1">{text}</div>}
      <div className="progress"><div className="progress-bar" style={{ width: `${v * 100}%` }} /></div>
    </div>
  )
}

/** Dashtreme's "impressions by country" list: label, value on the right, a thin white bar under each. */
export function ProgressList({ items, max }: { items: { label: ReactNode; value: ReactNode; frac: number; onClick?: () => void }[]; max?: number }) {
  const m = max ?? Math.max(1e-9, ...items.map((i) => i.frac))
  return (
    <ul className="plist">
      {items.map((it, i) => (
        <li key={i} onClick={it.onClick} style={it.onClick ? { cursor: 'pointer' } : undefined}>
          <div className="row1"><span>{it.label}</span><span className="v">{it.value}</span></div>
          <div className="progress"><div className="progress-bar" style={{ width: `${Math.max(0, Math.min(1, it.frac / m)) * 100}%` }} /></div>
        </li>
      ))}
    </ul>
  )
}

/** A key/value list (Dashtreme's legend tables under donuts). */
export function KVList({ items }: { items: { label: ReactNode; value: ReactNode; dot?: string }[] }) {
  return (
    <ul className="kv">
      {items.map((it, i) => <li key={i}>{it.dot && <span style={{ width: 9, height: 9, borderRadius: '50%', background: it.dot, display: 'inline-block' }} />}<span>{it.label}</span><span className="v">{it.value}</span></li>)}
    </ul>
  )
}

/** The row of centred figures under a chart (Dashtreme's "45.87M Overall Visitor | 15:48 Visitor Duration"). */
export function RowGroup({ items }: { items: { label: ReactNode; value: ReactNode }[] }) {
  return <div className="row-group-cells">{items.map((it, i) => <div key={i}><div className="v">{it.value}</div><div className="l">{it.label}</div></div>)}</div>
}

export function Expander({ title, icon, children, open }: { title: ReactNode; icon?: string; children: ReactNode; open?: boolean }) {
  return (
    <details className="expander" open={open}>
      <summary><Icon name={icon} />{title}</summary>
      <div className="body">{children}</div>
    </details>
  )
}

/** Loading skeleton, error callout, or the content — what every topic-driven panel renders. */
export function Loaded<T>({ state, children, height = 90 }: { state: { data: T | undefined; error?: string; loading: boolean }; children: (d: T) => ReactNode; height?: number }) {
  if (state.data === undefined) {
    if (state.error) return <Callout kind="error">{state.error}</Callout>
    return <div className="skeleton" style={{ height }} />
  }
  return (
    <>
      {state.error && <Callout kind="error">{state.error} (showing the last good data)</Callout>}
      {children(state.data)}
    </>
  )
}

export function Address({ value }: { value: string }) {
  return (
    <span className="d-flex gap-2 align-items-start">
      <code className="addr flex-grow-1">{value}</code>
      <button type="button" className="btn btn-sm btn-light" title="Copy" onClick={() => navigator.clipboard?.writeText(value)}><Icon name="copy" /></button>
    </span>
  )
}
