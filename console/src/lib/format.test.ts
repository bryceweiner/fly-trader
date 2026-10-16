import { describe, expect, it } from 'vitest'
import { toCsv } from './csv'
import { ago, amount, dt, hm, pct, pctPts, shortAddr, sig, usd } from './format'
import { topicKey } from './live'

describe('formats match the Python side (fly_trader/ui/state.py)', () => {
  it('amounts', () => {
    expect(amount(1234.56789, 'SOL')).toBe('1,234.5679 SOL')
    expect(amount(0.5, 'ETH', 4, true)).toBe('+0.5000 ETH')
    expect(amount(-0.5, 'ETH', 4, true)).toBe('-0.5000 ETH')
    expect(amount(null)).toBe('—')
  })
  it('percents', () => {
    expect(pct(0.0123)).toBe('+1.23%')
    expect(pct(-0.0123, 1)).toBe('-1.2%')
    expect(pct(0.5, 0, false)).toBe('50%')
    expect(pctPts(12.345)).toBe('+12.35%')
    expect(pct(undefined)).toBe('—')
  })
  it('dollars', () => {
    expect(usd(1300)).toBe('$1,300.00')
    expect(usd(-2.5, 2, true)).toBe('-$2.50')
    expect(usd(2.5, 2, true)).toBe('+$2.50')
  })
  it('significant digits like %g', () => {
    expect(sig(0.123456)).toBe('0.1235')
    expect(sig(5)).toBe('5')
  })
  it('times in UTC', () => {
    const now = Date.parse('2026-10-16T12:00:00Z')
    expect(ago('2026-10-16T11:59:30Z', now)).toBe('30 s ago')
    expect(ago('2026-10-16T11:00:00Z', now)).toBe('60 min ago')
    expect(ago('2026-10-15T12:00:00Z', now)).toBe('24.0 h ago')
    expect(ago('2026-10-10T12:00:00Z', now)).toBe('6 d ago')
    expect(dt('2026-10-16T09:05:07Z')).toBe('Oct 16 09:05')
    expect(dt('2026-10-16T09:05:07Z', true)).toBe('Oct 16 09:05:07')
    expect(hm('2026-10-16T09:05:07Z')).toBe('09:05')
  })
  it('addresses', () => {
    expect(shortAddr('E8APqcyZpz21zDQU6TfAzquNoJ3YgTjTxxRP18roUch3')).toBe('E8APqc…Uch3')
  })
})

describe('helpers', () => {
  it('topic keys ignore param order', () => {
    expect(topicKey('t', { b: 1, a: 'x' })).toBe(topicKey('t', { a: 'x', b: 1 }))
    expect(topicKey('t', { a: 1 })).not.toBe(topicKey('t', { a: 2 }))
  })
  it('csv quotes what needs quoting', () => {
    expect(toCsv(['a', 'b'], [[1, 'x,y'], [null, 'say "hi"']])).toBe('a,b\r\n1,"x,y"\r\n,"say ""hi"""')
  })
})
