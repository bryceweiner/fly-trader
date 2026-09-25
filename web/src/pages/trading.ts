/** Trading page: market strip, candles, KyberSwap buy/sell, balances. */
import { sendTransaction, type Config } from '@wagmi/core'
import { erc20Abi, formatUnits, getAddress, isAddressEqual, type Address } from 'viem'
import { config, explorer } from '../config'
import type { Wallets, WalletState } from '../lib/appkit'
import { poll } from '../lib/api'
import { publicClient } from '../lib/chains'
import { CandleChart, fetchCandles, fetchMarket, type Interval, type Market } from '../lib/chart'
import { approveExact, ensureChain, humanError, waitOk } from '../lib/evm'
import { parseAmount, pct, short, token, usd, usdCompact } from '../lib/format'
import {
  buildRoute,
  clampSlippage,
  DEFAULT_SLIPPAGE_BPS,
  formatSlippage,
  getRoute,
  isSuspiciousQuote,
  minReceived,
  NATIVE,
  parseSlippagePercent,
  priceImpact,
  SLIPPAGE_LADDER_BPS,
  type Route,
} from '../lib/kyber'
import { $, busy, mountWalletBar, setText, statusLine, tabs, txLink } from '../lib/ui'
import { vaultAbi } from '../lib/vault'

interface Asset {
  symbol: string
  address: Address
  decimals: number
}

const ETH: Asset = { symbol: 'ETH', address: NATIVE, decimals: 18 }
const FLY: Asset | null = config.fly.address ? { symbol: '$FLY', address: config.fly.address, decimals: 18 } : null
const USDG: Asset | null = config.usdg ? { symbol: 'USDG', address: config.usdg.address, decimals: config.usdg.decimals } : null
const ETH_GAS_BUFFER = 500_000_000_000_000n // 0.0005 ETH kept back by "Max"

const state = {
  side: 'buy' as 'buy' | 'sell',
  asset: 'ETH' as 'ETH' | 'USDG',
  slippageBps: DEFAULT_SLIPPAGE_BPS,
  interval: '1h' as Interval,
  market: null as Market | null,
  account: null as Address | null,
  cfg: null as Config | null,
  balances: {} as Record<string, bigint>,
  route: null as Route | null,
  quoteSeq: 0,
}

const status = statusLine('swap-status')
const amountEl = $<HTMLInputElement>('swap-amount')
const swapBtn = $<HTMLButtonElement>('swap-btn')

function sides(): { tokenIn: Asset; tokenOut: Asset } | null {
  const other = state.asset === 'ETH' ? ETH : USDG
  if (!FLY || !other) return null
  return state.side === 'buy' ? { tokenIn: other, tokenOut: FLY } : { tokenIn: FLY, tokenOut: other }
}

/* ---------- static bits ---------- */

function setLink(id: string, href: string | null, text: string) {
  const a = $<HTMLAnchorElement>(id)
  if (href) a.href = href
  a.textContent = text
}
setLink('c-fly', FLY ? explorer.evmAddress(FLY.address) : null, FLY ? getAddress(FLY.address) : 'not configured')
setLink('c-usdg', USDG ? explorer.evmAddress(USDG.address) : null, USDG ? USDG.address : 'mainnet only')
setLink('c-router', explorer.evmAddress(config.kyber.router), config.kyber.router)
setLink('router-link', explorer.evmAddress(config.kyber.router), short(config.kyber.router))
$<HTMLAnchorElement>('c-pool').href = `https://www.geckoterminal.com/${config.market.geckoNetwork}/pools/${config.market.pool}`

if (!config.kyber.enabled) {
  const note = $('network-note')
  note.hidden = false
  note.className = 'notice mb-24'
  note.innerHTML = '<span class="micro">Testnet build</span><p>Swaps are disabled: KyberSwap only runs on Robinhood Chain mainnet. The chart shows fixture data.</p>'
  for (const el of $('swap-form').querySelectorAll<HTMLInputElement | HTMLButtonElement | HTMLSelectElement>('input, button, select')) el.disabled = true
  swapBtn.textContent = 'Swaps are mainnet only'
}
if (!config.market.live) setText('m-source', 'Market data: rehearsal fixtures (testnet build)')

/* ---------- market strip + chart ---------- */

const chart = new CandleChart($('price-chart'))
let fitNext = true

async function refreshMarket() {
  try {
    const m = await fetchMarket()
    state.market = m
    setText('m-price', usd(m.priceUsd))
    setText('m-change', pct(m.change24h), `value ${m.change24h == null ? '' : m.change24h >= 0 ? 'lime' : 'red'}`)
    setText('m-fdv', usdCompact(m.fdvUsd))
    setText('m-liq', usdCompact(m.liquidityUsd))
    setText('m-vol', usdCompact(m.volume24hUsd))
  } catch {
    setText('m-source', 'Market data unavailable right now (GeckoTerminal). Retrying every minute.')
  }
}

async function refreshCandles() {
  const msg = $('chart-msg')
  try {
    const candles = await fetchCandles(state.interval)
    chart.set(candles, fitNext)
    fitNext = false
    msg.hidden = candles.length > 0
    msg.textContent = 'No trades in this range yet.'
  } catch {
    msg.hidden = false
    msg.textContent = 'Chart unavailable right now. Retrying every minute.'
  }
}

tabs($('interval-tabs'), 'interval', (v) => {
  state.interval = v as Interval
  fitNext = true
  void refreshCandles()
})

poll(() => Promise.all([refreshMarket(), refreshCandles()]), 60_000)

/* ---------- balances ---------- */

async function refreshBalances() {
  const user = state.account
  const ids = ['b-eth', 'b-usdg', 'b-fly', 'b-locked']
  if (!user) {
    ids.forEach((id) => setText(id, '—'))
    updateBalanceHint()
    return
  }
  try {
    const client = publicClient()
    const [eth, usdg, fly, locked] = await Promise.all([
      client.getBalance({ address: user }),
      USDG ? client.readContract({ address: USDG.address, abi: erc20Abi, functionName: 'balanceOf', args: [user] }) : null,
      FLY ? client.readContract({ address: FLY.address, abi: erc20Abi, functionName: 'balanceOf', args: [user] }) : null,
      config.vault ? client.readContract({ address: config.vault, abi: vaultAbi, functionName: 'locked', args: [user] }) : null,
    ])
    state.balances = { ETH: eth, ...(usdg != null ? { USDG: usdg } : {}), ...(fly != null ? { $FLY: fly } : {}) }
    setText('b-eth', token(eth, 18, 6))
    setText('b-usdg', usdg == null ? 'n/a' : token(usdg, USDG!.decimals, 2))
    setText('b-fly', fly == null ? 'n/a' : token(fly, 18, 2))
    setText('b-locked', locked == null ? 'vault not deployed' : token(locked, 18, 2))
  } catch {
    ids.forEach((id) => setText(id, 'unavailable'))
  }
  updateBalanceHint()
}

function updateBalanceHint() {
  const s = sides()
  const bal = s ? state.balances[s.tokenIn.symbol] : undefined
  setText('swap-balance', s && bal != null ? `Balance: ${token(bal, s.tokenIn.decimals, 6)} ${s.tokenIn.symbol}` : 'Balance: —')
}

/* ---------- quote ---------- */

function resetQuote(label: string) {
  state.route = null
  for (const id of ['q-out', 'q-min', 'q-impact', 'q-gas']) setText(id, '—')
  swapBtn.textContent = label
  swapBtn.disabled = true
}

function labels() {
  const s = sides()
  setText('swap-asset-label', state.side === 'buy' ? 'Pay with' : 'Receive')
  setText('swap-amount-label', s ? `You ${state.side === 'buy' ? 'pay' : 'sell'} (${s.tokenIn.symbol})` : 'Amount')
  updateBalanceHint()
}

let quoteTimer: number | undefined
let swapping = false
function scheduleQuote() {
  clearTimeout(quoteTimer)
  quoteTimer = window.setTimeout(() => void quote(), 450)
}

async function quote() {
  if (!config.kyber.enabled) return
  const s = sides()
  if (!s) return resetQuote('$FLY not configured')
  const amount = parseAmount(amountEl.value, s.tokenIn.decimals)
  if (!amountEl.value.trim()) return resetQuote('Enter an amount')
  if (amount == null) return resetQuote('Invalid amount')
  const seq = ++state.quoteSeq
  swapBtn.disabled = true
  swapBtn.textContent = 'Finding a route…'
  try {
    const route = await getRoute(s.tokenIn.address, s.tokenOut.address, amount)
    if (seq !== state.quoteSeq) return
    state.route = route
    showQuote(route, s)
  } catch (e) {
    if (seq !== state.quoteSeq) return
    resetQuote('No route')
    status(humanError(e), 'error')
  }
}

function showQuote(route: Route, s: { tokenIn: Asset; tokenOut: Asset }) {
  const r = route.routeSummary
  const out = BigInt(r.amountOut)
  setText('q-out', `${token(out, s.tokenOut.decimals, 6)} ${s.tokenOut.symbol}`)
  setText('q-min', `${token(minReceived(out, state.slippageBps), s.tokenOut.decimals, 6)} ${s.tokenOut.symbol}`)
  let inUsd = parseFloat(r.amountInUsd)
  let outUsd = parseFloat(r.amountOutUsd)
  // Kyber often leaves $FLY unpriced; fall back to the GeckoTerminal pool price for that side.
  const flyUsd = state.market?.priceUsd ?? 0
  if (!(outUsd > 0) && state.side === 'buy' && flyUsd > 0) outUsd = Number(formatUnits(out, 18)) * flyUsd
  if (!(inUsd > 0) && state.side === 'sell' && flyUsd > 0) inUsd = Number(formatUnits(BigInt(r.amountIn), 18)) * flyUsd
  const impact = priceImpact(inUsd, outUsd)
  const impactEl = $('q-impact')
  const suspicious = isSuspiciousQuote(impact)
  impactEl.textContent = impact == null ? 'unknown' : suspicious ? `quote claims a ${pct(-impact, true)} gain: the pools may not honour it` : pct(Math.max(0, impact), true)
  impactEl.className = suspicious || (impact != null && impact > 0.05) ? 'red' : impact != null && impact > 0.02 ? 'amber' : ''
  setText('q-gas', usd(parseFloat(r.gasUsd)))
  const bal = state.balances[s.tokenIn.symbol]
  if (!state.account) {
    swapBtn.textContent = 'Connect a wallet to swap'
    swapBtn.disabled = true
  } else if (bal != null && BigInt(r.amountIn) > bal) {
    swapBtn.textContent = `Not enough ${s.tokenIn.symbol}`
    swapBtn.disabled = true
  } else {
    swapBtn.textContent = state.side === 'buy' ? `Buy $FLY with ${s.tokenIn.symbol}` : `Sell $FLY for ${s.tokenOut.symbol}`
    swapBtn.disabled = swapping // a re-quote while a swap waits in the wallet must not offer a second swap
  }
}

/* ---------- swap ---------- */

async function swap() {
  const s = sides()
  const cfg = state.cfg
  const account = state.account
  if (!s || !cfg || !account || swapping) return
  const amount = parseAmount(amountEl.value, s.tokenIn.decimals)
  if (amount == null) return
  swapping = true
  try {
    status('Checking network…', 'busy')
    await ensureChain(cfg)
    if (!isAddressEqual(s.tokenIn.address, NATIVE)) {
      status(`Approve exactly ${token(amount, s.tokenIn.decimals, 6)} ${s.tokenIn.symbol} for the KyberSwap router in your wallet…`, 'busy')
      await approveExact(cfg, s.tokenIn.address, account, config.kyber.router, amount)
    }
    status('Getting a fresh route…', 'busy')
    const route = await getRoute(s.tokenIn.address, s.tokenOut.address, amount)
    showQuote(route, s)
    const built = await buildRoute(route, account, state.slippageBps)
    const value = BigInt(built.transactionValue || '0')
    const expected = isAddressEqual(s.tokenIn.address, NATIVE) ? amount : 0n
    if (value !== expected) throw new Error('Refusing the swap: KyberSwap built a transaction with an unexpected ETH value.')
    // Dry run on the node first: a route the pools cannot honour at this slippage fails here, with a plain
    // explanation, instead of costing a wallet prompt and a reverted transaction.
    status('Checking the swap against the chain…', 'busy')
    try {
      await publicClient().call({ account, to: built.routerAddress, data: built.data, value })
    } catch (e) {
      // The router measures what arrives at the wallet. A smart account (EIP-7702 delegation or contract) that
      // forwards incoming ETH receives nothing, so every slippage fails; saying "raise the slippage" would mislead.
      if (isAddressEqual(s.tokenOut.address, NATIVE) && /Return amount is not enough/i.test(String((e as Error)?.message)) && (await publicClient().getCode({ address: account }).catch(() => undefined))) {
        throw new Error('The router refused because your wallet did not keep the ETH it was sent: it is a smart account (EIP-7702 or contract) that forwards incoming ETH. Sell for USDG instead, or use another wallet.')
      }
      throw e
    }
    status('Confirm the swap in your wallet…', 'busy')
    const hash = await sendTransaction(cfg, { account, to: built.routerAddress, data: built.data, value, chainId: config.evm.chainId })
    status('Waiting for confirmation…', 'busy')
    await waitOk(cfg, hash)
    status(txLink(hash), 'ok')
    amountEl.value = ''
    resetQuote('Enter an amount')
    await refreshBalances()
  } catch (e) {
    status(humanError(e), 'error')
    swapBtn.disabled = false
  } finally {
    swapping = false
  }
}

/* ---------- inputs ---------- */

tabs($('side-tabs'), 'side', (v) => {
  state.side = v === 'sell' ? 'sell' : 'buy'
  labels()
  scheduleQuote()
})

const assetSel = $<HTMLSelectElement>('swap-asset')
if (!USDG) assetSel.querySelector('option[value="USDG"]')?.remove()
assetSel.addEventListener('change', () => {
  state.asset = assetSel.value === 'USDG' ? 'USDG' : 'ETH'
  labels()
  scheduleQuote()
})

amountEl.addEventListener('input', scheduleQuote)

$('swap-max').addEventListener('click', () => {
  const s = sides()
  const bal = s ? state.balances[s.tokenIn.symbol] : undefined
  if (!s || bal == null) return
  const max = s.tokenIn.symbol === 'ETH' ? (bal > ETH_GAS_BUFFER ? bal - ETH_GAS_BUFFER : 0n) : bal
  amountEl.value = formatUnits(max, s.tokenIn.decimals)
  scheduleQuote()
})

const slipSelect = $<HTMLSelectElement>('slip-select')
const slipInput = $<HTMLInputElement>('slip-input')
slipSelect.replaceChildren(
  ...SLIPPAGE_LADDER_BPS.map((bps) => Object.assign(document.createElement('option'), { value: String(bps), textContent: formatSlippage(bps) })),
  Object.assign(document.createElement('option'), { value: '', textContent: 'Custom' }),
)

function setSlippage(bps: number) {
  state.slippageBps = clampSlippage(bps)
  setText('slip-view', formatSlippage(state.slippageBps))
  slipInput.value = String(state.slippageBps / 100)
  slipSelect.value = SLIPPAGE_LADDER_BPS.includes(state.slippageBps) ? String(state.slippageBps) : ''
  try {
    localStorage.setItem('fly.slippageBps', String(state.slippageBps))
  } catch {
    /* storage blocked: the setting just isn't remembered */
  }
  if (state.route) scheduleQuote()
}
slipSelect.addEventListener('change', () => {
  if (slipSelect.value) setSlippage(Number(slipSelect.value))
  else slipInput.focus()
})
slipInput.addEventListener('change', () => {
  const bps = parseSlippagePercent(slipInput.value)
  if (bps == null) {
    status('Slippage must be a number between 0.2 % and 10 %.', 'error')
    slipInput.value = String(state.slippageBps / 100)
    return
  }
  setSlippage(bps)
})
try {
  const saved = Number(localStorage.getItem('fly.slippageBps'))
  setSlippage(saved > 0 ? saved : DEFAULT_SLIPPAGE_BPS)
} catch {
  setSlippage(DEFAULT_SLIPPAGE_BPS)
}

$('swap-form').addEventListener('submit', (e) => {
  e.preventDefault()
  void busy(swapBtn, swap)
})

labels()

/* ---------- wallets (AppKit loads after the page is readable) ---------- */

function onWallet(s: WalletState, w: Wallets) {
  state.cfg = w.wagmiConfig
  const changed = state.account !== s.evm
  state.account = s.evm
  if (changed) {
    state.balances = {}
    void refreshBalances().then(() => (state.route ? showQuote(state.route, sides()!) : undefined))
  }
}

setTimeout(() => {
  void mountWalletBar({ solana: false, onChange: onWallet })
}, 0)
poll(() => (state.account ? refreshBalances() : undefined), 60_000)
