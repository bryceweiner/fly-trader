/**
 * Audit of the KyberSwap client (2026-09-25). Tests marked "AUDIT FINDING" assert the behaviour the page should
 * have and fail against the current code; the report names the fix.
 */
import { encodeFunctionData, parseAbi, type Address, type Hex } from 'viem'
import { afterEach, describe, expect, it, vi } from 'vitest'
import {
  buildRoute,
  clampSlippage,
  DEFAULT_SLIPPAGE_BPS,
  getRoute,
  isSuspiciousQuote,
  KyberError,
  MAX_SLIPPAGE_BPS,
  MIN_SLIPPAGE_BPS,
  minReceived,
  NATIVE,
  parseSlippagePercent,
  priceImpact,
  SLIPPAGE_LADDER_BPS,
  SLIPPAGE_STEP_BPS,
  type Route,
} from './kyber'

const ROUTER = '0x6131B5fae19EA4f9D964eAc0408E4408b66337b5' as Address
const FLY = '0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3' as Address
const USER = '0x1111111111111111111111111111111111111111' as Address
const EVIL = '0x000000000000000000000000000000000000dEaD' as Address

afterEach(() => vi.unstubAllGlobals())

function respond(body: unknown, status = 200) {
  const fn = vi.fn(async () => new Response(typeof body === 'string' ? body : JSON.stringify(body), { status }))
  vi.stubGlobal('fetch', fn)
  return fn
}
const ok = (data: unknown) => respond({ code: 0, message: 'successfully', data })

/** MetaAggregationRouterV2.swap (selector 0xe21fd0e9), the calldata /route/build returns. */
const routerAbi = parseAbi([
  'struct SwapDescriptionV2 { address srcToken; address dstToken; address[] srcReceivers; uint256[] srcAmounts; address[] feeReceivers; uint256[] feeAmounts; address dstReceiver; uint256 amount; uint256 minReturnAmount; uint256 flags; bytes permit; }',
  'struct SwapExecutionParams { address callTarget; address approveTarget; bytes targetData; SwapDescriptionV2 desc; bytes clientData; }',
  'function swap(SwapExecutionParams execution) payable returns (uint256 returnAmount, uint256 gasUsed)',
])
function swapCalldata(o: { src?: Address; dst?: Address; to?: Address; amount?: bigint; minOut?: bigint; fee?: [Address, bigint] }): Hex {
  return encodeFunctionData({
    abi: routerAbi,
    functionName: 'swap',
    args: [
      {
        callTarget: EVIL,
        approveTarget: EVIL,
        targetData: '0x',
        desc: {
          srcToken: o.src ?? NATIVE,
          dstToken: o.dst ?? FLY,
          srcReceivers: [],
          srcAmounts: [],
          feeReceivers: o.fee ? [o.fee[0]] : [],
          feeAmounts: o.fee ? [o.fee[1]] : [],
          dstReceiver: o.to ?? USER,
          amount: o.amount ?? 1000n,
          minReturnAmount: o.minOut ?? 998_000n,
          flags: 0n,
          permit: '0x',
        },
        clientData: '0x',
      },
    ],
  })
}
const summary = (o: Partial<Record<'tokenIn' | 'tokenOut' | 'amountIn' | 'amountOut', string>> = {}) => ({
  tokenIn: NATIVE,
  tokenOut: FLY,
  amountIn: '1000',
  amountOut: '1000000',
  amountInUsd: '1',
  amountOutUsd: '0.99',
  gas: '1',
  gasUsd: '0.01',
  ...o,
})
const route = (): Route => ({ routeSummary: summary(), routerAddress: ROUTER })

describe('kyber transport errors all surface as KyberError', () => {
  it('network failure', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => { throw new TypeError('offline') }))
    await expect(getRoute(NATIVE, FLY, 1n)).rejects.toThrow(/could not be reached/)
  })
  it('HTTP 500 with a non-JSON body', async () => {
    respond('<html>bad gateway</html>', 502)
    await expect(getRoute(NATIVE, FLY, 1n)).rejects.toThrow(/KyberSwap error 502/)
  })
  it('code != 0 carries the API message as text', async () => {
    respond({ code: 4008, message: 'route not found' })
    await expect(getRoute(NATIVE, FLY, 1n)).rejects.toThrow('KyberSwap: route not found')
  })
  it('code 0 without data is an error, not an undefined route', async () => {
    respond({ code: 0, message: 'ok' })
    await expect(getRoute(NATIVE, FLY, 1n)).rejects.toBeInstanceOf(KyberError)
  })
  it('a missing routerAddress is refused', async () => {
    ok({ routeSummary: summary() })
    await expect(getRoute(NATIVE, FLY, 1n)).rejects.toThrow(/\(none\)/)
  })
  it('getRoute sends the exact wei amount as a decimal string (no Number precision loss)', async () => {
    const big = 123_456_789_012_345_678_901_234_567n
    const fn = ok({ routeSummary: summary({ tokenIn: FLY, tokenOut: NATIVE, amountIn: big.toString() }), routerAddress: ROUTER })
    await getRoute(FLY, NATIVE, big)
    const url = new URL((fn.mock.calls[0] as unknown as [string])[0])
    expect(url.searchParams.get('amountIn')).toBe(big.toString())
    expect(url.searchParams.get('tokenIn')).toBe(FLY)
    expect(url.searchParams.get('tokenOut')).toBe(NATIVE)
  })
})

describe('buildRoute never sends slippage outside 0.2–10 %', () => {
  for (const [given, sent] of [[5000, MAX_SLIPPAGE_BPS], [-5, MIN_SLIPPAGE_BPS], [NaN, DEFAULT_SLIPPAGE_BPS], [Infinity, DEFAULT_SLIPPAGE_BPS], [74.6, 75]] as const) {
    it(`${given} bps -> ${sent} bps`, async () => {
      const fn = ok({ data: '0x', routerAddress: ROUTER, transactionValue: '0', amountIn: '1000', amountOut: '1' })
      await buildRoute(route(), USER, given).catch(() => undefined)
      const body = JSON.parse((fn.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)
      expect(body.slippageTolerance).toBe(sent)
    })
  }
  it('sender and recipient are both the connected account; deadline is now + 20 min', async () => {
    const fn = ok({ data: '0x', routerAddress: ROUTER, transactionValue: '0', amountIn: '1000', amountOut: '1' })
    await buildRoute(route(), USER, 20, 1_700_000_000).catch(() => undefined)
    const body = JSON.parse((fn.mock.calls[0] as unknown as [string, RequestInit])[1].body as string)
    expect(body.sender).toBe(USER)
    expect(body.recipient).toBe(USER)
    expect(body.deadline).toBe(1_700_000_000 + 1200)
    expect(body.routeSummary).toEqual(route().routeSummary)
  })
})

describe('slippage input and ladder', () => {
  it('ladder is the single source: sorted, unique, in range, 0.5 % steps after the 0.2 % floor', () => {
    expect(SLIPPAGE_LADDER_BPS[0]).toBe(MIN_SLIPPAGE_BPS)
    expect(SLIPPAGE_LADDER_BPS.at(-1)).toBe(MAX_SLIPPAGE_BPS)
    expect(SLIPPAGE_LADDER_BPS).toContain(DEFAULT_SLIPPAGE_BPS)
    expect(new Set(SLIPPAGE_LADDER_BPS).size).toBe(SLIPPAGE_LADDER_BPS.length)
    for (let i = 1; i < SLIPPAGE_LADDER_BPS.length; i++) {
      expect(SLIPPAGE_LADDER_BPS[i]).toBeGreaterThan(SLIPPAGE_LADDER_BPS[i - 1])
      expect(SLIPPAGE_LADDER_BPS[i] % SLIPPAGE_STEP_BPS).toBe(0)
    }
    for (const b of SLIPPAGE_LADDER_BPS) expect(clampSlippage(b)).toBe(b)
  })
  it.each([
    ['1e1', null], ['0x10', null], ['  5 ', 500], ['-1', null], ['-0.5', null], ['NaN', null], ['Infinity', null],
    ['1,5', 150], ['1.2.3', null], ['.5', 50], ['5.', null], ['10.004', 1000], ['10.005', null], ['0.195', 20],
    ['0.194', null], ['+1', null], ['1 000', null], ['１', null], ['0', null], ['00010', 1000], ['1,5,', null],
  ] as const)('parseSlippagePercent(%j) = %s', (s, want) => {
    expect(parseSlippagePercent(s)).toBe(want)
  })
  it('clampSlippage handles infinities and fractions', () => {
    expect(clampSlippage(-Infinity)).toBe(DEFAULT_SLIPPAGE_BPS)
    expect(clampSlippage(20.4)).toBe(20)
    expect(clampSlippage(999.6)).toBe(1000)
  })
  it('minReceived floors and is always <= amountOut, never negative', () => {
    expect(minReceived(1n, 20)).toBe(0n)
    expect(minReceived(10_001n, 20)).toBe(9_980n)
    const big = 2n ** 200n
    expect(minReceived(big, 1000)).toBe((big * 9000n) / 10000n)
    expect(minReceived(0n, 1000)).toBe(0n)
    expect(minReceived(1000n, 99_999)).toBe(900n) // clamped to 10 %
  })
  it('suspicious-quote guard: exactly 2 % gain is tolerated, anything more is flagged', () => {
    expect(isSuspiciousQuote(-0.02)).toBe(false)
    expect(isSuspiciousQuote(-0.0201)).toBe(true)
    expect(priceImpact(NaN, 1)).toBeNull()
    expect(priceImpact(1, Infinity)).toBe(-Infinity)
    expect(isSuspiciousQuote(priceImpact(1, Infinity))).toBe(true)
  })
})

/*
 * AUDIT FINDING (high): the page trusts KyberSwap's route and calldata beyond the router address and ETH value.
 * Nothing checks that the quote is for the tokens and amount asked for, nor that the calldata pays the user
 * (dstReceiver) at least minReceived (minReturnAmount). A compromised or buggy API could return calldata to the
 * allowlisted router that sends the output to another address or with minReturnAmount = 0, and the page would
 * prompt it (the eth_call pre-flight succeeds for such calldata).
 */
describe('AUDIT FINDING: route and calldata are verified against what the user asked for', () => {
  it('getRoute refuses a routeSummary for another output token', async () => {
    ok({ routeSummary: summary({ tokenOut: EVIL }), routerAddress: ROUTER })
    await expect(getRoute(NATIVE, FLY, 1000n)).rejects.toBeInstanceOf(KyberError)
  })
  it('getRoute refuses a routeSummary for another input token', async () => {
    ok({ routeSummary: summary({ tokenIn: EVIL }), routerAddress: ROUTER })
    await expect(getRoute(NATIVE, FLY, 1000n)).rejects.toBeInstanceOf(KyberError)
  })
  it('getRoute refuses a routeSummary for another amountIn', async () => {
    ok({ routeSummary: summary({ amountIn: '999999' }), routerAddress: ROUTER })
    await expect(getRoute(NATIVE, FLY, 1000n)).rejects.toBeInstanceOf(KyberError)
  })
  it('buildRoute refuses a build whose amountIn differs from the route', async () => {
    ok({ data: swapCalldata({ amount: 5000n }), routerAddress: ROUTER, transactionValue: '5000', amountIn: '5000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).rejects.toBeInstanceOf(KyberError)
  })
  it('buildRoute refuses calldata that pays someone else (dstReceiver)', async () => {
    ok({ data: swapCalldata({ to: EVIL }), routerAddress: ROUTER, transactionValue: '1000', amountIn: '1000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).rejects.toBeInstanceOf(KyberError)
  })
  it('buildRoute refuses calldata whose minReturnAmount is below quote x (1 - slippage)', async () => {
    ok({ data: swapCalldata({ minOut: 0n }), routerAddress: ROUTER, transactionValue: '1000', amountIn: '1000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).rejects.toBeInstanceOf(KyberError)
  })
  it('buildRoute refuses calldata for other tokens than the route', async () => {
    ok({ data: swapCalldata({ dst: EVIL }), routerAddress: ROUTER, transactionValue: '1000', amountIn: '1000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).rejects.toBeInstanceOf(KyberError)
  })
  it('buildRoute refuses calldata that lists a fee, even one inside the slippage floor', async () => {
    const floor = minReceived(1_000_000n, 20)
    ok({ data: swapCalldata({ minOut: floor, fee: [EVIL, 1n] }), routerAddress: ROUTER, transactionValue: '1000', amountIn: '1000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).rejects.toThrow(/pays a fee/)
    ok({ data: swapCalldata({ minOut: floor, fee: [EVIL, 0n] }), routerAddress: ROUTER, transactionValue: '1000', amountIn: '1000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).rejects.toThrow(/pays a fee/)
  })
  it('buildRoute accepts well-formed calldata for the user at the right floor', async () => {
    ok({ data: swapCalldata({ minOut: minReceived(1_000_000n, 20) }), routerAddress: ROUTER, transactionValue: '1000', amountIn: '1000', amountOut: '1000000' })
    await expect(buildRoute(route(), USER, 20)).resolves.toMatchObject({ routerAddress: ROUTER })
  })
})
