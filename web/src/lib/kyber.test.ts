import { afterEach, describe, expect, it, vi } from 'vitest'
import { buildRoute, clampSlippage, getRoute, isAllowedRouter, KyberError, minReceived, NATIVE, priceImpact, type Route } from './kyber'

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
  it('clampSlippage keeps it within 0.01 %–50 %', () => {
    expect(clampSlippage(0)).toBe(1)
    expect(clampSlippage(99_999)).toBe(5000)
    expect(clampSlippage(NaN)).toBe(100)
  })
  it('priceImpact', () => {
    expect(priceImpact(100, 98)).toBeCloseTo(0.02)
    expect(priceImpact(2.66, 0)).toBeNull()
  })
})
