/**
 * KyberSwap aggregator on Robinhood Chain: quote (routes) and calldata (route/build).
 * The only contract we ever send a swap to, or approve, is the allowlisted router.
 */
import { getAddress, isAddress, isAddressEqual, type Address, type Hex } from 'viem'
import { config } from '../config'

export const NATIVE = '0xEeeeeEeeeEeEeeEeEeEeeEEEeeeeEeeeeeeeEEeE' as Address
export const DEFAULT_SLIPPAGE_BPS = 100
export const MAX_SLIPPAGE_BPS = 5000
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
  return Math.min(MAX_SLIPPAGE_BPS, Math.max(1, Math.round(bps)))
}

/** The least the router will deliver: amountOut × (1 − slippage). */
export function minReceived(amountOut: bigint, slippageBps: number): bigint {
  return (amountOut * BigInt(10_000 - clampSlippage(slippageBps))) / 10_000n
}

/** Fractional loss between USD in and USD out (0.012 = 1.2 %). Null when either side is unpriced. */
export function priceImpact(amountInUsd: number, amountOutUsd: number): number | null {
  if (!(amountInUsd > 0) || !(amountOutUsd > 0)) return null
  return 1 - amountOutUsd / amountInUsd
}
