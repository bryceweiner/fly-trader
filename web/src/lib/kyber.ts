/**
 * KyberSwap aggregator on Robinhood Chain: quote (routes) and calldata (route/build).
 * The only contract we ever send a swap to, or approve, is the allowlisted router.
 */
import { getAddress, isAddress, isAddressEqual, type Address, type Hex } from 'viem'
import { config } from '../config'

export const NATIVE = '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE' as Address
/** Slippage in basis points: the page starts at 0.2 %, offers 0.5 % steps up to 10 %, and accepts any custom value
 *  in that range. Nothing outside it is ever sent to KyberSwap. */
export const MIN_SLIPPAGE_BPS = 20
export const MAX_SLIPPAGE_BPS = 1000
export const SLIPPAGE_STEP_BPS = 50
export const DEFAULT_SLIPPAGE_BPS = MIN_SLIPPAGE_BPS
export const SLIPPAGE_LADDER_BPS: readonly number[] = [
  MIN_SLIPPAGE_BPS,
  ...Array.from({ length: MAX_SLIPPAGE_BPS / SLIPPAGE_STEP_BPS }, (_, i) => SLIPPAGE_STEP_BPS * (i + 1)),
]
export const DEADLINE_S = 20 * 60

export interface RouteSummary {
  tokenIn: string
  amountIn: string
  amountInUsd: string
  tokenOut: string
  amountOut: string
  amountOutUsd: string
  gas: string
  gasUsd: string
  [k: string]: unknown
}

export interface Route {
  routeSummary: RouteSummary
  routerAddress: Address
}

export interface BuiltRoute {
  amountIn: string
  amountOut: string
  amountInUsd?: string
  amountOutUsd?: string
  data: Hex
  routerAddress: Address
  transactionValue: string
}

export class KyberError extends Error {}

export function isAllowedRouter(addr: string | null | undefined, allowed: string = config.kyber.router): boolean {
  return !!addr && isAddress(addr, { strict: false }) && isAddressEqual(addr as Address, allowed as Address)
}

function assertRouter(addr: string | undefined): Address {
  if (!isAllowedRouter(addr)) {
    throw new KyberError(`Refusing the swap: KyberSwap returned router ${addr ?? '(none)'}, not the allowlisted ${config.kyber.router}.`)
  }
  return getAddress(addr!)
}

async function kyber<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(`${config.kyber.api}${path}`, {
      ...init,
      headers: { 'x-client-id': config.kyber.clientId, ...(init?.body ? { 'content-type': 'application/json' } : {}) },
    })
  } catch {
    throw new KyberError('KyberSwap could not be reached.')
  }
  const body = (await res.json().catch(() => null)) as { code?: number; message?: string; data?: T } | null
  if (!res.ok || !body || body.code !== 0 || !body.data) {
    throw new KyberError(body?.message ? `KyberSwap: ${body.message}` : `KyberSwap error ${res.status}`)
  }
  return body.data
}

export async function getRoute(tokenIn: Address, tokenOut: Address, amountIn: bigint, signal?: AbortSignal): Promise<Route> {
  const qs = new URLSearchParams({ tokenIn, tokenOut, amountIn: amountIn.toString(), gasInclude: 'true' })
  const data = await kyber<{ routeSummary: RouteSummary; routerAddress: string }>(`/routes?${qs}`, { signal })
  return { routeSummary: data.routeSummary, routerAddress: assertRouter(data.routerAddress) }
}

export async function buildRoute(
  route: Route,
  account: Address,
  slippageBps: number,
  nowS = Math.floor(Date.now() / 1000),
): Promise<BuiltRoute> {
  const data = await kyber<BuiltRoute>('/route/build', {
    method: 'POST',
    body: JSON.stringify({
      routeSummary: route.routeSummary,
      sender: account,
      recipient: account,
      slippageTolerance: clampSlippage(slippageBps),
      deadline: nowS + DEADLINE_S,
      source: config.kyber.clientId,
    }),
  })
  return { ...data, routerAddress: assertRouter(data.routerAddress) }
}

export function clampSlippage(bps: number): number {
  if (!Number.isFinite(bps)) return DEFAULT_SLIPPAGE_BPS
  return Math.min(MAX_SLIPPAGE_BPS, Math.max(MIN_SLIPPAGE_BPS, Math.round(bps)))
}

/** A typed percentage ("0.75") as basis points, or null when it is not a number in 0.2–10 %. */
export function parseSlippagePercent(input: string): number | null {
  const s = input.trim().replace(',', '.')
  if (!/^\d*\.?\d+$/.test(s)) return null
  const bps = Math.round(parseFloat(s) * 100)
  return bps >= MIN_SLIPPAGE_BPS && bps <= MAX_SLIPPAGE_BPS ? bps : null
}

export const formatSlippage = (bps: number) => `${(bps / 100).toFixed(bps % 100 === 0 ? 0 : bps % 10 === 0 ? 1 : 2)} %`

/** The least the router will deliver: amountOut × (1 − slippage). */
export function minReceived(amountOut: bigint, slippageBps: number): bigint {
  return (amountOut * BigInt(10_000 - clampSlippage(slippageBps))) / 10_000n
}

/** Fractional loss between USD in and USD out (0.012 = 1.2 %; negative when the quote claims a gain). Null when
 *  either side is unpriced. */
export function priceImpact(amountInUsd: number, amountOutUsd: number): number | null {
  if (!(amountInUsd > 0) || !(amountOutUsd > 0)) return null
  return 1 - amountOutUsd / amountInUsd
}

/** A swap cannot create value: a quote whose USD out exceeds USD in by more than this is mispriced by the quoter
 *  (seen on 2026-09-24 for $FLY sells through the Uniswap v4 pool: +27 % "gain", then the router refused at every
 *  slippage up to 10 %). The page shows such a quote as suspicious instead of as 0 % impact. */
export const SUSPICIOUS_GAIN = 0.02
export const isSuspiciousQuote = (impact: number | null) => impact != null && impact < -SUSPICIOUS_GAIN
