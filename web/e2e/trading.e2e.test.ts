/**
 * The Trading page's buy and sell paths, end to end: real KyberSwap quotes and calldata (src/lib/kyber.ts), the same
 * checks trading.ts makes before sending, the calldata simulated on Robinhood Chain mainnet itself (eth_call with
 * state overrides, so no real funds move), and the wallet plumbing (exact approval, sending, error wording) on an
 * anvil fork of mainnet.
 *
 *   anvil --fork-url https://rpc.mainnet.chain.robinhood.com --port 8546 && web/e2e/run.sh trading
 *
 * The slippage ladder is measured by eth_call on mainnet (the source of truth, no funds move). The swap is then also
 * sent for real on the fork at the measured boundary. A fork more than a few minutes old drifts from the chain and
 * KyberSwap's executor then reverts early (custom error 0x8727a7f9); that is reported, not counted as a page failure.
 */
import { sendTransaction } from '@wagmi/core'
import { createPublicClient, decodeAbiParameters, encodeAbiParameters, erc20Abi, http, isAddressEqual, keccak256, parseEther, type Address, type Hex } from 'viem'
import { beforeAll, describe, expect, it } from 'vitest'
import { config } from '../src/config'
import { robinhood } from '../src/lib/chains'
import { approveExact, ensureChain, humanError } from '../src/lib/evm'
import { buildRoute, DEFAULT_SLIPPAGE_BPS, getRoute, isSuspiciousQuote, minReceived, NATIVE, priceImpact, SLIPPAGE_LADDER_BPS, type BuiltRoute, type Route } from '../src/lib/kyber'
import { pub, rpc, walletFor } from './harness'

const FLY = config.fly.address as Address
const USDG = config.usdg!.address
const ROUTER = config.kyber.router
const SLIPPAGE = DEFAULT_SLIPPAGE_BPS // 0.2 %, the page default
const MAINNET_RPC = 'https://rpc.mainnet.chain.robinhood.com'

/** The real chain, read-only: eth_call only. Its public RPC drops connections now and then, hence the retries. */
const mainnet = createPublicClient({ chain: robinhood, transport: http(MAINNET_RPC, { timeout: 60_000, retryCount: 6, retryDelay: 2_000 }) })

/** KyberSwap's public API throttles bursts ("service temporarily overloaded"); a person clicking never hits that,
 *  a test suite does. Same call, spaced out. */
async function patient<T>(fn: () => Promise<T>, tries = 12): Promise<T> {
  for (let i = 0; ; i++) {
    try {
      return await fn()
    } catch (e) {
      if (i + 1 >= tries || !/overloaded|rate|429|reached/i.test(String((e as Error).message))) throw e
      await new Promise((r) => setTimeout(r, 10_000 * (i + 1)))
    }
  }
}
const quote = (a: Address, b: Address, amt: bigint) => patient(() => getRoute(a, b, amt))
const build = (r: Route, who: Address, bps: number) => patient(() => buildRoute(r, who, bps))
const LONG = 20 * 60_000 // Kyber's throttle can last minutes

/** What trading.ts does between the quote and the wallet prompt. */
function pageChecks(built: BuiltRoute, tokenIn: Address, amount: bigint) {
  const value = BigInt(built.transactionValue || '0')
  const expected = isAddressEqual(tokenIn, NATIVE) ? amount : 0n
  if (value !== expected) throw new Error('Refusing the swap: KyberSwap built a transaction with an unexpected ETH value.')
  expect(isAddressEqual(built.routerAddress, ROUTER)).toBe(true)
  expect(built.data).toMatch(/^0x[0-9a-f]{8,}$/)
  return value
}

/** MetaAggregationRouterV2.swap returns (uint256 returnAmount, uint256 gasUsed). */
function swapReturn(data: Hex): bigint {
  return decodeAbiParameters([{ type: 'uint256' }, { type: 'uint256' }], data)[0]
}

/** $FLY is a plain OpenZeppelin ERC-20: balances live in slot 0, allowances in slot 1 (checked on 2026-09-24). */
const balanceSlot = (owner: Address) => keccak256(encodeAbiParameters([{ type: 'address' }, { type: 'uint256' }], [owner, 0n]))
const allowanceSlot = (owner: Address, spender: Address) =>
  keccak256(encodeAbiParameters([{ type: 'address' }, { type: 'bytes32' }], [spender, keccak256(encodeAbiParameters([{ type: 'address' }, { type: 'uint256' }], [owner, 1n]))]))
const word = (n: bigint) => ('0x' + n.toString(16).padStart(64, '0')) as Hex
/** Kyber's build rounds its amountOut independently of the route summary: within 0.1 % is the same quote. */
const near = (a: bigint, b: bigint) => (a > b ? a - b : b - a) <= b / 1000n

/** Sends the built swap on the fork from the page's wallet path. Returns the receipt status, or 'drift' when the
 *  executor refused because the fork no longer matches the chain (see the header). */
async function sendOnFork(cfg: Awaited<ReturnType<typeof walletFor>>['cfg'], from: Address, built: BuiltRoute): Promise<'success' | 'reverted' | 'drift'> {
  const value = BigInt(built.transactionValue || '0')
  try {
    await pub.call({ account: from, to: built.routerAddress, data: built.data, value })
  } catch (e) {
    const m = String((e as Error).message)
    if (/0x8727a7f9|Call failed|historical state/.test(m)) {
      console.warn(`fork drift: the executor refused the swap on anvil (${m.split('\n')[0].slice(0, 120)}); mainnet simulation above is authoritative`)
      return 'drift'
    }
    throw e
  }
  const hash = await sendTransaction(cfg, { account: from, to: built.routerAddress, data: built.data, value, chainId: config.evm.chainId })
  const receipt = await pub.waitForTransactionReceipt({ hash })
  return receipt.status
}

describe('trading: buy and sell $FLY through KyberSwap', () => {
  let cfg: Awaited<ReturnType<typeof walletFor>>['cfg']
  let user: Address
  const spend = parseEther('0.01')
  let buyRoute: Route
  let buyBuilt: BuiltRoute // reused by the later tests: every Kyber call counts against its throttle
  let bought: bigint

  beforeAll(async () => {
    expect(config.kyber.enabled).toBe(true)
    expect(await pub.getChainId()).toBe(4663)
    ;({ cfg, address: user } = await walletFor(5))
    await rpc('anvil_setBalance', [user, '0x' + parseEther('10').toString(16)])
    await ensureChain(cfg)
  })

  it('the storage layout the sell simulation relies on holds on mainnet', async () => {
    const poolManager = '0x8366a39CC670B4001A1121B8F6A443A643e40951' as Address
    const bal = await mainnet.readContract({ address: FLY, abi: erc20Abi, functionName: 'balanceOf', args: [poolManager] })
    const raw = await mainnet.getStorageAt({ address: FLY, slot: balanceSlot(poolManager) })
    expect(BigInt(raw!)).toBe(bal)
    expect(bal).toBeGreaterThan(0n)
  })

  it('quote: ETH -> $FLY route comes from the allowlisted router with a priced output', async () => {
    const r = await quote(NATIVE, FLY, spend)
    buyRoute = r
    expect(isAddressEqual(r.routerAddress, ROUTER)).toBe(true)
    expect(BigInt(r.routeSummary.amountOut)).toBeGreaterThan(0n)
    expect(BigInt(r.routeSummary.amountIn)).toBe(spend)
    const impact = priceImpact(parseFloat(r.routeSummary.amountInUsd), parseFloat(r.routeSummary.amountOutUsd))
    expect(impact == null || impact < 0.5).toBe(true)
    expect(minReceived(BigInt(r.routeSummary.amountOut), SLIPPAGE)).toBe((BigInt(r.routeSummary.amountOut) * BigInt(10_000 - SLIPPAGE)) / 10_000n)
  }, LONG)

  it('buy: build calldata, check the ETH value, simulate on mainnet across the slippage ladder', async () => {
    const quoted = BigInt(buyRoute.routeSummary.amountOut)
    const builds = new Map<number, BuiltRoute>()
    const simulate = async (bps: number): Promise<bigint | null> => {
      const built = await build(buyRoute, user, bps)
      builds.set(bps, built)
      buyBuilt ??= built
      expect(pageChecks(built, NATIVE, spend)).toBe(spend)
      expect(near(BigInt(built.amountOut), quoted)).toBe(true)
      try {
        const { data } = await mainnet.call({ account: user, to: built.routerAddress, data: built.data, value: spend, stateOverride: [{ address: user, balance: parseEther('1') }] })
        return swapReturn(data!)
      } catch (e) {
        if (/Return amount is not enough/.test(String((e as Error).message))) return null
        throw e
      }
    }
    // Same method as the sell below: one run at the top of the ladder measures what the pools deliver, then the
    // boundary step and the one under it are confirmed on chain.
    const ladder = SLIPPAGE_LADDER_BPS
    const top = ladder[ladder.length - 1]
    const delivered = await simulate(top)
    expect(delivered, `the buy must succeed within ${top / 100} % slippage`).not.toBeNull()
    bought = delivered!
    // predicted from the measurement, then confirmed on chain; the pools move between blocks, so a predicted
    // step that is within a hair of the floor can slip one step up on confirmation
    const predicted = ladder.find((bps) => minReceived(quoted, bps) <= bought)!
    let boundary = top
    for (const bps of ladder.slice(ladder.indexOf(predicted))) {
      if (bps === top || (await simulate(bps)) != null) {
        boundary = bps
        break
      }
    }
    console.log(`buy ${spend} wei ETH: quoted ${quoted}, delivered ${bought} (${(Number(bought) / Number(quoted)).toFixed(4)} of quote); predicted ${predicted / 100} %, confirmed at ${boundary / 100} %`)
    if (boundary > SLIPPAGE) console.warn(`FINDING: buying $FLY needs ${boundary / 100} % slippage today; the page defaults to ${SLIPPAGE / 100} %`)
    expect(bought).toBeLessThanOrEqual((quoted * 11n) / 10n) // sane: not wildly above the quote
    // and for real on the fork, through the wallet path, at the boundary
    const before = await pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'balanceOf', args: [user] })
    const status = await sendOnFork(cfg, user, builds.get(boundary)!)
    if (status !== 'drift') {
      expect(status).toBe('success')
      const got = (await pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'balanceOf', args: [user] })) - before
      expect(got).toBeGreaterThanOrEqual(minReceived(quoted, boundary))
      console.log(`buy executed on the fork: received ${got} wei FLY`)
    }
  }, LONG)

  it('sell: exact approval to the router only (on the fork), zero ETH value, simulated on mainnet across the slippage ladder', async () => {
    expect(bought).toBeGreaterThan(0n)
    // the approval leg for real on the fork: the page approves exactly the amount, never unlimited
    await rpc('anvil_setStorageAt', [FLY, balanceSlot(user), word(bought)])
    expect(await pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'balanceOf', args: [user] })).toBe(bought)
    const hash0 = await approveExact(cfg, FLY, user, ROUTER, bought)
    expect(hash0).toMatch(/^0x/)
    expect(await pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'allowance', args: [user, ROUTER] })).toBe(bought)
    expect(await approveExact(cfg, FLY, user, ROUTER, bought)).toBeNull() // enough now: no second prompt

    const route = await quote(FLY, NATIVE, bought)
    const quoted = BigInt(route.routeSummary.amountOut)
    const overrides = [
      { address: user, balance: parseEther('1') },
      { address: FLY, stateDiff: [{ slot: balanceSlot(user), value: word(bought) }, { slot: allowanceSlot(user, ROUTER), value: word(bought) }] },
    ]
    const builds = new Map<number, BuiltRoute>()
    const simulate = async (bps: number): Promise<bigint | null> => {
      const built = await build(route, user, bps)
      builds.set(bps, built)
      expect(pageChecks(built, FLY, bought)).toBe(0n)
      expect(near(BigInt(built.amountOut), quoted)).toBe(true)
      try {
        const { data } = await mainnet.call({ account: user, to: built.routerAddress, data: built.data, value: 0n, stateOverride: overrides })
        return swapReturn(data!)
      } catch (e) {
        if (/Return amount is not enough/.test(String((e as Error).message))) return null
        throw e
      }
    }
    // The ladder the page offers: 0.2 %, then 0.5 % steps to 10 %. The router embeds quote × (1 − slippage) as its
    // floor, so one run at the top tells how much the pools really deliver; the boundary step is then confirmed
    // on chain, plus the step below it (which must be refused). This keeps the KyberSwap calls to three or four.
    const ladder = SLIPPAGE_LADDER_BPS
    const top = ladder[ladder.length - 1]
    const delivered = await simulate(top)
    const usdIn = parseFloat(route.routeSummary.amountInUsd)
    const usdOut = parseFloat(route.routeSummary.amountOutUsd)
    const hops = (route.routeSummary.route as { exchange: string }[][] | undefined)?.flat().map((p) => p.exchange).join(' > ') ?? '?'
    console.log(`sell quote: ${usdIn.toFixed(4)} USD in, ${usdOut.toFixed(4)} USD out; pools ${hops}`)
    expect(
      delivered,
      `the sell must succeed within ${top / 100} % slippage. It did not: KyberSwap quoted ${usdOut.toFixed(4)} USD out for ${usdIn.toFixed(4)} USD in` +
        (isSuspiciousQuote(priceImpact(usdIn, usdOut)) ? ' (a quote claiming a gain: the quoter mispriced a hop, and the page now marks it suspicious)' : ''),
    ).not.toBeNull()
    const ratio = Number(delivered!) / Number(quoted)
    const predicted = ladder.find((bps) => minReceived(quoted, bps) <= delivered!)!
    let boundary = top
    for (const bps of ladder.slice(ladder.indexOf(predicted))) {
      if (bps === top || (await simulate(bps)) != null) {
        boundary = bps
        break
      }
    }
    console.log(`sell ${bought} wei FLY: quoted ${quoted}, delivered ${delivered} (${ratio.toFixed(4)} of quote); predicted ${predicted / 100} %, confirmed at ${boundary / 100} %`)
    if (boundary > SLIPPAGE) console.warn(`FINDING: selling $FLY needs ${boundary / 100} % slippage today; the page defaults to ${SLIPPAGE / 100} % and its pre-flight check explains why`)
    expect(delivered!).toBeLessThan(spend) // a round trip through two pools cannot come back with more ETH than went in
    // and for real on the fork: the balance was set above, the exact approval is in place
    const ethBefore = await pub.getBalance({ address: user })
    const status = await sendOnFork(cfg, user, builds.get(boundary)!)
    if (status !== 'drift') {
      expect(status).toBe('success')
      expect(await pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'balanceOf', args: [user] })).toBe(0n)
      expect(await pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'allowance', args: [user, ROUTER] })).toBe(0n)
      console.log(`sell executed on the fork: ETH delta ${(await pub.getBalance({ address: user })) - ethBefore} wei (gas included)`)
    }
  }, LONG)

  it('quote: USDG -> $FLY and $FLY -> USDG both route', async () => {
    const buy = await quote(USDG, FLY, 5_000_000n) // 5 USDG
    expect(BigInt(buy.routeSummary.amountOut)).toBeGreaterThan(0n)
    const sell = await quote(FLY, USDG, parseEther('10000'))
    expect(BigInt(sell.routeSummary.amountOut)).toBeGreaterThan(0n)
  }, LONG)

  it('the page refuses calldata whose ETH value does not match what the user typed', () => {
    const built = buyBuilt
    expect(() => pageChecks({ ...built, transactionValue: (spend + 1n).toString() }, NATIVE, spend)).toThrow(/unexpected ETH value/)
    expect(() => pageChecks({ ...built, transactionValue: '1' }, FLY, spend)).toThrow(/unexpected ETH value/)
  })

  it('a wallet without ETH gets the gas message', async () => {
    const poor = await walletFor(6)
    await rpc('anvil_setBalance', [poor.address, '0x0'])
    const built = buyBuilt // the node refuses for lack of funds before the router ever runs
    let msg = ''
    try {
      await sendTransaction(poor.cfg, { account: poor.address, to: built.routerAddress, data: built.data, value: spend, chainId: config.evm.chainId })
    } catch (e) {
      msg = humanError(e)
    }
    expect(msg).toMatch(/Not enough ETH|insufficient|funds/i)
  })

  it('a swap that reverts shows a readable error, never a raw object', async () => {
    const built = buyBuilt // an ETH-in route sent without the ETH: the router must refuse it
    let msg = ''
    try {
      await pub.call({ account: user, to: built.routerAddress, data: built.data, value: 0n })
    } catch (e) {
      msg = humanError(e)
    }
    expect(msg.length).toBeGreaterThan(0)
    expect(msg).not.toMatch(/^\[object/)
    expect(msg).not.toMatch(/undefined/)
  })
})
