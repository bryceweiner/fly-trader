import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  buildRoute,
  clampSlippage,
  DEFAULT_SLIPPAGE_BPS,
  formatSlippage,
  getRoute,
  isAllowedRouter,
  isSuspiciousQuote,
  KyberError,
  MAX_SLIPPAGE_BPS,
  MIN_SLIPPAGE_BPS,
  minReceived,
  NATIVE,
  parseSlippagePercent,
  priceImpact,
  SLIPPAGE_LADDER_BPS,
  type Route,
} from './kyber'

const ROUTER = '0x6131B5fae19EA4f9D964eAc0408E4408b66337b5'
const FLY = '0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3'
const EVIL = '0x000000000000000000000000000000000000dEaD'

function mockFetch(data: unknown) {
  const fn = vi.fn(async () => new Response(JSON.stringify({ code: 0, message: 'successfully', data }), { status: 200 }))
  vi.stubGlobal('fetch', fn)
  return fn
}
afterEach(() => vi.unstubAllGlobals())

describe('router allowlist', () => {
  it('accepts the allowlisted router in any case', () => {
    expect(isAllowedRouter(ROUTER)).toBe(true)
    expect(isAllowedRouter(ROUTER.toLowerCase())).toBe(true)
  })
  it('rejects anything else', () => {
    expect(isAllowedRouter(EVIL)).toBe(false)
    expect(isAllowedRouter('')).toBe(false)
    expect(isAllowedRouter(undefined)).toBe(false)
    expect(isAllowedRouter('0x1234')).toBe(false)
  })
  it('getRoute refuses a route for another router', async () => {
    mockFetch({ routeSummary: { amountOut: '1' }, routerAddress: EVIL })
    await expect(getRoute(NATIVE, FLY, 1n)).rejects.toBeInstanceOf(KyberError)
  })
  it('buildRoute refuses calldata for another router', async () => {
    const route = { routeSummary: { amountOut: '1' }, routerAddress: ROUTER } as unknown as Route
    mockFetch({ data: '0x', routerAddress: EVIL, transactionValue: '0', amountIn: '1', amountOut: '1' })
    await expect(buildRoute(route, EVIL, 100)).rejects.toThrow(/allowlisted/)
  })
  it('getRoute sends the client id and accepts the allowlisted router', async () => {
    const fn = mockFetch({ routeSummary: { amountOut: '5' }, routerAddress: ROUTER.toLowerCase() })
    const r = await getRoute(NATIVE, FLY, 10n)
    expect(r.routerAddress).toBe(ROUTER)
    const [url, init] = fn.mock.calls[0] as unknown as [string, RequestInit]
    expect(url).toContain('/robinhood/api/v1/routes?')
    expect((init.headers as Record<string, string>)['x-client-id']).toBe('fly-trader')
  })
  it('buildRoute sends slippage in bps and a 20-minute deadline', async () => {
    const fn = mockFetch({ data: '0xabc', routerAddress: ROUTER, transactionValue: '10', amountIn: '10', amountOut: '5' })
    await buildRoute({ routeSummary: {} as never, routerAddress: ROUTER }, EVIL, 150, 1_000)
    const body = JSON.parse((fn.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)
    expect(body).toMatchObject({ slippageTolerance: 150, deadline: 2_200, sender: EVIL, recipient: EVIL })
  })
})

describe('quote maths', () => {
  it('minReceived applies slippage in bps', () => {
    expect(minReceived(10_000n, 100)).toBe(9_900n)
    expect(minReceived(1_000_000n, 50)).toBe(995_000n)
  })
  it('slippage starts at 0.2 %, is clamped to 0.2–10 %, and steps by 0.5 % up to 10 %', () => {
    expect(DEFAULT_SLIPPAGE_BPS).toBe(20)
    expect(MIN_SLIPPAGE_BPS).toBe(20)
    expect(MAX_SLIPPAGE_BPS).toBe(1000)
    expect(clampSlippage(0)).toBe(20)
    expect(clampSlippage(19)).toBe(20)
    expect(clampSlippage(99_999)).toBe(1000)
    expect(clampSlippage(NaN)).toBe(20)
    expect(clampSlippage(75)).toBe(75)
    expect(SLIPPAGE_LADDER_BPS).toEqual([20, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000])
  })
  it('custom entry accepts any value in range and refuses the rest', () => {
    expect(parseSlippagePercent('0.2')).toBe(20)
    expect(parseSlippagePercent('0,75')).toBe(75)
    expect(parseSlippagePercent(' 10 ')).toBe(1000)
    expect(parseSlippagePercent('0.19')).toBeNull()
    expect(parseSlippagePercent('10.01')).toBeNull()
    expect(parseSlippagePercent('-1')).toBeNull()
    expect(parseSlippagePercent('abc')).toBeNull()
    expect(parseSlippagePercent('')).toBeNull()
    expect(parseSlippagePercent('1e1')).toBeNull()
  })
  it('formatSlippage', () => {
    expect(formatSlippage(20)).toBe('0.2 %')
    expect(formatSlippage(75)).toBe('0.75 %')
    expect(formatSlippage(150)).toBe('1.5 %')
    expect(formatSlippage(1000)).toBe('10 %')
  })
  it('priceImpact', () => {
    expect(priceImpact(100, 98)).toBeCloseTo(0.02)
    expect(priceImpact(2.66, 0)).toBeNull()
    expect(priceImpact(0.0405, 0.0516)).toBeCloseTo(-0.274, 2)
  })
  it('a quote that claims a gain is suspicious; small rounding is not', () => {
    expect(isSuspiciousQuote(priceImpact(0.0405, 0.0516))).toBe(true)
    expect(isSuspiciousQuote(priceImpact(100, 101))).toBe(false)
    expect(isSuspiciousQuote(priceImpact(100, 98))).toBe(false)
    expect(isSuspiciousQuote(null)).toBe(false)
  })
})
