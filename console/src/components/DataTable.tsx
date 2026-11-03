// Dashtreme's DataTables, in React: "Show N entries", search, Copy / CSV / Print, sortable striped rows, "Showing x to y
// of z entries" and pagination. Columns render cells and give raw values for sorting, searching and export.
import { useEffect, useMemo, useState, type ReactNode } from 'react'
import { download, toCsv } from '../lib/csv'
import { useStored } from '../lib/hooks'
import { useFeedback } from './feedback'
import { Help } from './ui'

export type Col<R> = {
  key: string
  label: string
  /** Render the cell (default: the value as text). */
  render?: (row: R) => ReactNode
  /** The raw value for sorting, filtering and export (default: row[key]). */
  value?: (row: R) => unknown
  num?: boolean
  wrap?: boolean
  help?: string
  width?: number
}

const SIZES = [10, 25, 50, 100, 0] as const        // 0 = all

function pages(cur: number, total: number): (number | '…')[] {
  if (total <= 7) return Array.from({ length: total }, (_, i) => i)
  const out: (number | '…')[] = [0]
  const lo = Math.max(1, cur - 1), hi = Math.min(total - 2, cur + 1)
  if (lo > 1) out.push('…')
  for (let i = lo; i <= hi; i++) out.push(i)
  if (hi < total - 2) out.push('…')
  out.push(total - 1)
  return out
}

/** ``compact`` fits a narrow card: header labels wrap onto two lines and the cells pad less. */
export function DataTable<R extends Record<string, unknown>>({ rows, cols, height = 560, filename = 'table', onRowClick, empty = 'No rows.', search = true, initialSort, pageSize, compact = false }: {
  rows: R[]; cols: Col<R>[]; height?: number; filename?: string; onRowClick?: (r: R) => void; empty?: ReactNode; search?: boolean
  initialSort?: { key: string; desc?: boolean }; pageSize?: number; compact?: boolean
}) {
  const fb = useFeedback()
  const [sort, setSort] = useState<{ key: string; desc: boolean } | null>(initialSort ? { key: initialSort.key, desc: !!initialSort.desc } : null)
  const [filter, setFilter] = useState('')
  const [size, setSize] = useStored<number>(`dt:${filename}:size`, pageSize ?? 10)
  const [page, setPage] = useState(0)
  const val = (c: Col<R>, r: R) => (c.value ? c.value(r) : r[c.key])
  const shown = useMemo(() => {
    let out = rows
    const f = filter.trim().toLowerCase()
    if (f) out = out.filter((r) => cols.some((c) => { const v = val(c, r); return v !== null && v !== undefined && String(v).toLowerCase().includes(f) }))
    if (sort) {
      const c = cols.find((x) => x.key === sort.key)
      if (c) {
        out = [...out].sort((a, b) => {
          const x = val(c, a), y = val(c, b)
          if (x === y) return 0
          if (x === null || x === undefined) return 1
          if (y === null || y === undefined) return -1
          const r = typeof x === 'number' && typeof y === 'number' ? x - y : String(x).localeCompare(String(y), undefined, { numeric: true })
          return sort.desc ? -r : r
        })
      }
    }
    return out
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rows, cols, sort, filter])
  const per = size > 0 ? size : Math.max(1, shown.length)
  const npages = Math.max(1, Math.ceil(shown.length / per))
  useEffect(() => { if (page > npages - 1) setPage(npages - 1) }, [npages, page])
  if (!rows.length) return <div className="empty">{empty}</div>
  const start = Math.min(page, npages - 1) * per
  const view = shown.slice(start, start + per)
  const click = (k: string) => setSort((s) => (!s || s.key !== k ? { key: k, desc: false } : !s.desc ? { key: k, desc: true } : null))
  const exportRows = () => shown.map((r) => cols.map((c) => val(c, r)))
  const copy = async () => {
    const text = [cols.map((c) => c.label).join('\t'), ...exportRows().map((r) => r.map((v) => (v === null || v === undefined ? '' : typeof v === 'object' ? JSON.stringify(v) : String(v))).join('\t'))].join('\n')
    try { await navigator.clipboard.writeText(text); fb.toast(`Copied ${shown.length} rows`, 'success', 2500) } catch { fb.toast('The browser refused the clipboard', 'error') }
  }
  const print = () => {
    const w = window.open('', '_blank'); if (!w) return
    const esc = (v: unknown) => String(v ?? '').replace(/[&<>]/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;' })[m]!)
    w.document.write(`<title>${esc(filename)}</title><style>body{font:12px system-ui;margin:16px}table{border-collapse:collapse;width:100%}th,td{border:1px solid #ccc;padding:4px 6px;text-align:left}th{background:#eee}</style>`
      + `<h3>${esc(filename)}</h3><table><thead><tr>${cols.map((c) => `<th>${esc(c.label)}</th>`).join('')}</tr></thead><tbody>`
      + exportRows().map((r) => `<tr>${r.map((v) => `<td>${esc(typeof v === 'object' && v !== null ? JSON.stringify(v) : v)}</td>`).join('')}</tr>`).join('') + '</tbody></table>')
    w.document.close(); w.focus(); w.print()
  }
  return (
    <div className="dt">
      <div className="dt-top">
        <label className="dt-len">Show
          <select className="form-select form-select-sm" value={size} onChange={(e) => { setSize(Number(e.target.value)); setPage(0) }}>
            {SIZES.map((s) => <option key={s} value={s}>{s || 'All'}</option>)}
          </select>entries</label>
        <div className="btn-group dt-buttons" role="group">
          <button type="button" className="btn btn-sm btn-light" onClick={copy}>Copy</button>
          <button type="button" className="btn btn-sm btn-light" onClick={() => download(`${filename}.csv`, toCsv(cols.map((c) => c.label), exportRows()))}>CSV</button>
          <button type="button" className="btn btn-sm btn-light" onClick={print}>Print</button>
        </div>
        {search && <label className="dt-search">Search:<input className="form-control form-control-sm" value={filter} onChange={(e) => { setFilter(e.target.value); setPage(0) }} /></label>}
      </div>
      <div className="ftable-wrap" style={{ maxHeight: height }}>
        <table className={`ftable ${compact ? 'compact' : ''}`}>
          <thead>
            <tr>
              {cols.map((c) => (
                <th key={c.key} className={`${c.num ? 'num' : ''} ${sort?.key === c.key ? 'sorted' : ''}`} style={c.width ? { width: c.width } : undefined} onClick={() => click(c.key)}>
                  {c.label}<Help text={c.help} />
                  <i className={`bx sort ${sort?.key === c.key ? (sort.desc ? 'bx-sort-down' : 'bx-sort-up') : 'bx-sort-alt-2'}`} />
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {view.map((r, i) => (
              <tr key={start + i} className={onRowClick ? 'clickable' : ''} onClick={onRowClick ? () => onRowClick(r) : undefined}>
                {cols.map((c) => {
                  const v = c.render ? c.render(r) : (val(c, r) as ReactNode)
                  return <td key={c.key} className={`${c.num ? 'num' : ''} ${c.wrap ? 'wrap' : ''}`}>{v === null || v === undefined || v === '' ? '—' : v}</td>
                })}
              </tr>
            ))}
            {!view.length && <tr><td colSpan={cols.length} className="text-center text-muted2">No matching rows</td></tr>}
          </tbody>
        </table>
      </div>
      <div className="dt-bottom">
        <span>Showing {shown.length ? start + 1 : 0} to {start + view.length} of {shown.length} entries{shown.length !== rows.length ? ` (filtered from ${rows.length})` : ''}</span>
        {npages > 1 && (
          <ul className="pagination pagination-sm">
            <li className={`page-item ${page === 0 ? 'disabled' : ''}`}><a className="page-link" onClick={() => setPage(Math.max(0, page - 1))}>Prev</a></li>
            {pages(page, npages).map((p, i) => p === '…'
              ? <li key={`e${i}`} className="page-item disabled"><span className="page-link">…</span></li>
              : <li key={p} className={`page-item ${p === page ? 'active' : ''}`}><a className="page-link" onClick={() => setPage(p)}>{p + 1}</a></li>)}
            <li className={`page-item ${page >= npages - 1 ? 'disabled' : ''}`}><a className="page-link" onClick={() => setPage(Math.min(npages - 1, page + 1))}>Next</a></li>
          </ul>
        )}
      </div>
    </div>
  )
}

/** A 0..1 score as a bar with its number (Streamlit's ProgressColumn). */
export function BarCell({ value, max = 1, digits = 3 }: { value: number | null | undefined; max?: number; digits?: number }) {
  if (value === null || value === undefined) return <>—</>
  return (
    <span className="bar-cell">
      <span className="track"><span className="fill" style={{ width: `${Math.max(0, Math.min(1, value / max)) * 100}%` }} /></span>
      <span className="num">{value.toFixed(digits)}</span>
    </span>
  )
}

/** Dashtreme's status pill in a table cell. */
export function Pill({ children, color = 'gray' }: { children: ReactNode; color?: string }) {
  return <span className={`fbadge ${color}`} style={{ minWidth: 86, justifyContent: 'center' }}>{children}</span>
}
