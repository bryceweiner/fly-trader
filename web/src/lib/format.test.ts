import { describe, expect, it } from 'vitest'
import { ago, duration, nextMondayUtc, parseAmount, pct, short, sol, solDelta, token, usd, usdCompact, utc } from './format'

describe('format', () => {
  it('sol', () => {
    expect(sol(1_234_500_000)).toBe('1.2345')
    expect(sol(0)).toBe('0.00')
    expect(sol(-2_000_000)).toBe('−0.002')
    expect(sol(12_345_678_900_000, 2)).toBe('12,345.68')
    expect(sol(null)).toBe('—')
    expect(solDelta(42_000_000)).toBe('+0.042')
  })
  it('token (wei strings above 2^53)', () => {
    expect(token('412750000000000000123456789')).toBe('412,750,000')
    expect(token(1_500_000_000_000_000_000n)).toBe('1.5')
    expect(token('1234567', 6, 2)).toBe('1.23')
    expect(token('not a number')).toBe('—')
  })
  it('usd', () => {
    expect(usd(1234.5)).toBe('$1,234.50')
    expect(usd(0.0000412497)).toBe('$0.00004125')
    expect(usd(-3)).toBe('−$3.00')
    expect(usdCompact(41249.79)).toBe('$41.25K')
  })
  it('pct', () => {
    expect(pct(27.897)).toBe('+27.90 %')
    expect(pct(-0.0344, true)).toBe('−3.44 %')
    expect(pct(null)).toBe('—')
  })
  it('durations and ages', () => {
    expect(duration(6 * 86400 + 23 * 3600 + 59 * 60 + 10)).toBe('6d 23h 59m')
    expect(duration(2 * 3600 + 5 * 60 + 10)).toBe('2h 05m 10s')
    expect(duration(45)).toBe('45s')
    expect(ago(12)).toBe('12 s ago')
    expect(ago(301)).toBe('5 min ago')
  })
  it('utc and short', () => {
    expect(utc(1790553600)).toBe('2026-09-28 00:00 UTC')
    expect(short('0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3')).toBe('0x2fC7…b3D3')
    expect(short('HdkTVFM1vaZYZeFc9YsDzPT8fmFgq1CL7z8g2m1ujHDo')).toBe('HdkT…jHDo')
  })
  it('nextMondayUtc', () => {
    const wed = Date.UTC(2026, 8, 23, 12) / 1000
    const monday = Date.UTC(2026, 8, 28) / 1000
    expect(nextMondayUtc(wed)).toBe(monday)
    expect(nextMondayUtc(monday)).toBe(monday + 7 * 86400) // strictly after
    expect(nextMondayUtc(monday - 1)).toBe(monday)
    expect(new Date(nextMondayUtc(wed) * 1000).getUTCDay()).toBe(1)
  })
  it('parseAmount', () => {
    expect(parseAmount('1.5', 18)).toBe(1_500_000_000_000_000_000n)
    expect(parseAmount('1,000', 6)).toBe(1_000_000_000n)
    expect(parseAmount('.5', 6)).toBe(500_000n)
    expect(parseAmount('0', 18)).toBeNull()
    expect(parseAmount('-1', 18)).toBeNull()
    expect(parseAmount('1e5', 18)).toBeNull()
    expect(parseAmount('0.1234567', 6)).toBeNull()
    expect(parseAmount('', 18)).toBeNull()
  })
})
