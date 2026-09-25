/**
 * KyberSwap aggregator on Robinhood Chain: quote (routes) and calldata (route/build).
 * The only contract we ever send a swap to, or approve, is the allowlisted router.
 */
import { decodeFunctionData, getAddress, isAddress, isAddressEqual, parseAbi, type Address, type Hex } from 'viem'
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

/** The two MetaAggregationRouterV2 entry points /route/build returns (selectors 0xe21fd0e9 and 0x8af033fb). */
const ROUTER_ABI = parseAbi([
  'struct SwapDescriptionV2 { address srcToken; address dstToken; address[] srcReceivers; uint256[] srcAmounts; address[] feeReceivers; uint256[] feeAmounts; address dstReceiver; uint256 amount; uint256 minReturnAmount; uint256 flags; bytes permit; }',
  'struct SwapExecutionParams { address callTarget; address approveTarget; bytes targetData; SwapDescriptionV2 desc; bytes clientData; }',
  'function swap(SwapExecutionParams execution) payable returns (uint256 returnAmount, uint256 gasUsed)',
  'function swapSimpleMode(address caller, SwapDescriptionV2 desc, bytes executorData, bytes clientData) returns (uint256 returnAmount, uint256 gasUsed)',
])

/** How far the build's amountOut may fall below the quote the page showed (KyberSwap re-prices at build time). */
export const BUILD_DRIFT_BPS = 50

const sameToken = (a: string | undefined, b: string) => !!a && isAddress(a, { strict: false }) && isAddressEqual(a as Address, b as Address)

function big(v: unknown): bigint | null {
  try {
    return typeof v === 'string' && /^\d+$/.test(v) ? BigInt(v) : null
  } catch {
    return null
  }
}

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
  const routerAddress = assertRouter(data.routerAddress)
  const rs = data.routeSummary
  if (!rs || !sameToken(rs.tokenIn, tokenIn) || !sameToken(rs.tokenOut, tokenOut) || big(rs.amountIn) !== amountIn || big(rs.amountOut) == null) {
    throw new KyberError('Refusing the quote: KyberSwap answered for a different swap than the one asked for.')
  }
  return { routeSummary: rs, routerAddress }
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
  const built = { ...data, routerAddress: assertRouter(data.routerAddress) }
  checkBuild(built, route, account, slippageBps)
  return built
}

/** Decodes the router calldata and refuses it unless it swaps exactly the quoted tokens and amount, pays `account`,
 *  and enforces at least the slippage floor the page showed. Guards against a compromised or buggy API: the router
 *  allowlist alone would still let calldata send the output elsewhere or with no floor. */
export function checkBuild(built: BuiltRoute, route: Route, account: Address, slippageBps: number): void {
  const rs = route.routeSummary
  const refuse = (why: string) => {
    throw new KyberError(`Refusing the swap: KyberSwap built ${why}.`)
  }
  const amountIn = big(rs.amountIn)
  const quotedOut = big(rs.amountOut)
  const builtOut = big(built.amountOut)
  if (amountIn == null || quotedOut == null || builtOut == null) return refuse('a transaction without readable amounts')
  if (big(built.amountIn) !== amountIn) return refuse('a transaction for a different amount')
  if (builtOut * 10_000n < quotedOut * BigInt(10_000 - BUILD_DRIFT_BPS)) return refuse('a transaction that delivers much less than quoted')
  let desc
  try {
    const call = decodeFunctionData({ abi: ROUTER_ABI, data: built.data })
    desc = call.functionName === 'swap' ? call.args[0].desc : call.args[1]
  } catch {
    return refuse('calldata the page cannot verify')
  }
  if (!isAddressEqual(desc.dstReceiver, account)) return refuse('a transaction that pays another address')
  if (!sameToken(desc.srcToken, rs.tokenIn) || !sameToken(desc.dstToken, rs.tokenOut)) return refuse('a transaction for other tokens')
  if (desc.amount !== amountIn) return refuse('a transaction for a different amount')
  // The router takes any listed fee out of the swapped amount (src) or the output (dst). minReturnAmount still bounds
  // the loss, but a fee would silently eat the slippage budget, and the page promises no fee: builds without our
  // fee parameters carry empty lists (checked on mainnet 2026-09-25).
  if (desc.feeReceivers.length || desc.feeAmounts.some((x) => x !== 0n)) return refuse('a transaction that pays a fee')
  if (desc.minReturnAmount < minReceived(builtOut, slippageBps)) return refuse('a transaction without the slippage limit you chose')
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
