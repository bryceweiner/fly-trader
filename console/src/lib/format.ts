// The console's number and time formats (the same as the Python side's, fly_trader/ui/state.py).

type Num = number | string | null | undefined

export function n(x: Num): number | null {
  if (x === null || x === undefined || x === '') return null
  const v = typeof x === 'number' ? x : Number(x)
  return Number.isFinite(v) ? v : null
}

function fixed(v: number, nd: number, signed: boolean, group = true): string {
  const s = Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: nd, maximumFractionDigits: nd, useGrouping: group })
  return (v < 0 ? '-' : signed ? '+' : '') + s
}

/** "1,234.5678 SOL" — a book amount in its market's unit. */
export function amount(x: Num, unit = 'SOL', nd = 4, signed = false): string {
  const v = n(x)
  return v === null ? '—' : `${fixed(v, nd, signed)} ${unit}`
}

/** A fraction as a percent: 0.0123 → "+1.23%". */
export function pct(x: Num, nd = 2, signed = true): string {
  const v = n(x)
  return v === null ? '—' : `${fixed(v * 100, nd, signed, false)}%`
}

/** A value that is already in percent: 1.23 → "+1.23%". */
export function pctPts(x: Num, nd = 2, signed = true): string {
  const v = n(x)
  return v === null ? '—' : `${fixed(v, nd, signed, false)}%`
}

export function usd(x: Num, nd = 2, signed = false): string {
  const v = n(x)
  if (v === null) return '—'
  return `${signed && v >= 0 ? '+' : ''}${v < 0 ? '-' : ''}$${Math.abs(v).toLocaleString('en-US', { minimumFractionDigits: nd, maximumFractionDigits: nd })}`
}

export function num(x: Num, nd = 2, signed = false): string {
  const v = n(x)
  return v === null ? '—' : fixed(v, nd, signed)
}

/** Significant digits like Python's %.4g. */
export function sig(x: Num, digits = 4, signed = false): string {
  const v = n(x)
  if (v === null) return '—'
  const s = Math.abs(v) >= 1e-4 && Math.abs(v) < 1e15 ? String(Number(v.toPrecision(digits))) : v.toExponential(digits - 1)
  return (signed && v >= 0 ? '+' : '') + s
}

/** Significant digits as a plain decimal, never scientific notation: 3.079e-7 → "0.0000003079". */
export function plain(x: Num, digits = 4): string {
  const v = n(x)
  return v === null ? '—' : v.toLocaleString('en-US', { maximumSignificantDigits: digits })
}

export function sci(x: Num, digits = 3): string {
  const v = n(x)
  return v === null ? '—' : v.toExponential(digits)
}

export function int(x: Num): string {
  const v = n(x)
  return v === null ? '—' : Math.round(v).toLocaleString('en-US')
}

function toDate(ts: string | number | Date | null | undefined): Date | null {
  if (ts === null || ts === undefined || ts === '') return null
  const d = ts instanceof Date ? ts : new Date(typeof ts === 'number' ? ts * 1000 : ts)
  return Number.isNaN(d.getTime()) ? null : d
}

/** "12 s ago", "3 min ago", "1.5 h ago", "4 d ago". */
export function ago(ts: string | number | Date | null | undefined, now = Date.now()): string {
  const d = toDate(ts)
  if (!d) return '—'
  const s = Math.max(0, (now - d.getTime()) / 1000)
  return s < 120 ? `${s.toFixed(0)} s ago` : s < 7200 ? `${(s / 60).toFixed(0)} min ago` : s < 172800 ? `${(s / 3600).toFixed(1)} h ago` : `${(s / 86400).toFixed(0)} d ago`
}

const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
const pad = (v: number) => String(v).padStart(2, '0')

/** "Oct 16 14:05" in UTC (the console's times are all UTC). */
export function dt(ts: string | number | Date | null | undefined, seconds = false): string {
  const d = toDate(ts)
  if (!d) return '—'
  return `${MONTHS[d.getUTCMonth()]} ${d.getUTCDate()} ${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}${seconds ? ':' + pad(d.getUTCSeconds()) : ''}`
}

/** "14:05" UTC. */
export function hm(ts: string | number | Date | null | undefined): string {
  const d = toDate(ts)
  return d ? `${pad(d.getUTCHours())}:${pad(d.getUTCMinutes())}` : '—'
}

export function shortAddr(a: string | null | undefined, head = 6, tail = 4): string {
  if (!a) return '—'
  return a.length <= head + tail + 1 ? a : `${a.slice(0, head)}…${a.slice(-tail)}`
}

export function tone(x: Num): 'pos' | 'neg' | '' {
  const v = n(x)
  return v === null || v === 0 ? '' : v > 0 ? 'pos' : 'neg'
}
