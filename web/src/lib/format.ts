/** Number, time and address formatting. Pure functions, unit-tested in format.test.ts. */
import { formatUnits, parseUnits } from 'viem'

export const LAMPORTS_PER_SOL = 1_000_000_000
const MINUS = '−'

function sign(n: number): string {
  return n < 0 ? MINUS : ''
}

function group(n: number, min: number, max: number): string {
  return new Intl.NumberFormat('en-US', { minimumFractionDigits: min, maximumFractionDigits: max }).format(Math.abs(n))
}

/** 1_234_500_000 lamports -> "1.2345" (SOL, up to `digits` decimals, at least 2). */
export function sol(lamports: number | null | undefined, digits = 4): string {
  if (lamports == null || !Number.isFinite(lamports)) return '—'
  const v = lamports / LAMPORTS_PER_SOL
  return sign(v) + group(v, Math.min(2, digits), digits)
}

/** Signed SOL amount: "+0.0420" / "−0.0100". */
export function solDelta(lamports: number | null | undefined, digits = 4): string {
  if (lamports == null || !Number.isFinite(lamports)) return '—'
  return (lamports > 0 ? '+' : '') + sol(lamports, digits)
}

/** Token amount from base units (wei string or bigint) with thousands separators. */
export function token(amount: bigint | string | null | undefined, decimals = 18, maxFrac = 2): string {
  if (amount == null || amount === '') return '—'
  let v: bigint
  try {
    v = typeof amount === 'bigint' ? amount : BigInt(amount)
  } catch {
    return '—'
  }
  const neg = v < 0n
  const s = formatUnits(neg ? -v : v, decimals)
  const [int, frac = ''] = s.split('.')
  const f = frac.slice(0, maxFrac).replace(/0+$/, '')
  const intGrouped = int.replace(/\B(?=(\d{3})+(?!\d))/g, ',')
  return (neg ? MINUS : '') + intGrouped + (f ? '.' + f : '')
}

/** Token base units -> float, for USD maths only (never for amounts sent on-chain). */
export function tokenFloat(amount: bigint | string, decimals = 18): number {
  return Number(formatUnits(typeof amount === 'bigint' ? amount : BigInt(amount), decimals))
}

/** "$1,234.56"; small prices keep 4 significant digits: "$0.00004125". */
export function usd(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return '—'
  const a = Math.abs(x)
  if (a !== 0 && a < 1) {
    return sign(x) + '$' + new Intl.NumberFormat('en-US', { maximumSignificantDigits: 4 }).format(a)
  }
  return sign(x) + '$' + group(a, 2, 2)
}

/** "$41.2K", "$1.35M" */
export function usdCompact(x: number | null | undefined): string {
  if (x == null || !Number.isFinite(x)) return '—'
  return (
    sign(x) +
    '$' +
    new Intl.NumberFormat('en-US', { notation: 'compact', maximumFractionDigits: 2 }).format(Math.abs(x))
  )
}

/** Price with 4 significant digits and no currency sign, for chart axes. */
export function price(x: number): string {
  if (!Number.isFinite(x)) return ''
  const a = Math.abs(x)
  if (a >= 1) return sign(x) + group(a, 2, 4)
  return sign(x) + new Intl.NumberFormat('en-US', { maximumSignificantDigits: 4 }).format(a)
}

/** 27.897 -> "+27.90 %"; ratio=true treats input as a fraction (0.279 -> "+27.90 %"). */
export function pct(x: number | null | undefined, ratio = false, digits = 2): string {
  if (x == null || !Number.isFinite(x)) return '—'
  const v = ratio ? x * 100 : x
  return (v > 0 ? '+' : sign(v)) + Math.abs(v).toFixed(digits) + ' %'
}

/** Seconds -> "6d 23h 59m" / "2h 05m 10s" / "45s". */
export function duration(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—'
  const s = Math.max(0, Math.floor(seconds))
  const d = Math.floor(s / 86400)
  const h = Math.floor((s % 86400) / 3600)
  const m = Math.floor((s % 3600) / 60)
  const r = s % 60
  const p2 = (n: number) => String(n).padStart(2, '0')
  if (d > 0) return `${d}d ${p2(h)}h ${p2(m)}m`
  if (h > 0) return `${h}h ${p2(m)}m ${p2(r)}s`
  if (m > 0) return `${m}m ${p2(r)}s`
  return `${r}s`
}

/** Seconds since -> "12 s ago" / "5 min ago" / "3 h ago" / "2 d ago". */
export function ago(seconds: number): string {
  if (!Number.isFinite(seconds)) return '—'
  const s = Math.max(0, Math.round(seconds))
  if (s < 60) return `${s} s ago`
  if (s < 3600) return `${Math.floor(s / 60)} min ago`
  if (s < 86400) return `${Math.floor(s / 3600)} h ago`
  return `${Math.floor(s / 86400)} d ago`
}

/** Unix seconds -> "2026-09-28 00:00 UTC". */
export function utc(ts: number | null | undefined, withTime = true): string {
  if (ts == null || !Number.isFinite(ts) || ts <= 0) return '—'
  const iso = new Date(ts * 1000).toISOString()
  return withTime ? `${iso.slice(0, 10)} ${iso.slice(11, 16)} UTC` : iso.slice(0, 10)
}

/** "0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3" -> "0x2fC7…b3D3" */
export function short(a: string | null | undefined, head = 4, tail = 4): string {
  if (!a) return '—'
  const h = a.startsWith('0x') ? head + 2 : head
  return a.length <= h + tail + 1 ? a : `${a.slice(0, h)}…${a.slice(-tail)}`
}

/** Next Monday 00:00 UTC strictly after `nowS` (unix seconds). */
export function nextMondayUtc(nowS: number): number {
  const d = new Date(nowS * 1000)
  const day = d.getUTCDay() // 0 = Sunday
  const midnight = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000
  const daysAhead = (8 - day) % 7 || 7
  return midnight + daysAhead * 86400
}

/**
 * Parses what a person typed into a token amount in base units.
 * Returns null for anything that is not a plain positive decimal with at most `decimals` fraction digits.
 */
export function parseAmount(input: string, decimals: number): bigint | null {
  const s = input.trim().replace(/,/g, '')
  if (!/^\d*\.?\d*$/.test(s) || s === '' || s === '.') return null
  const frac = s.split('.')[1] ?? ''
  if (frac.length > decimals) return null
  try {
    const v = parseUnits(s, decimals)
    return v > 0n ? v : null
  } catch {
    return null
  }
}
