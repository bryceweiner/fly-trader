/** Vault page: public stats and tables, NAV chart, your position (lock/request/cancel/withdraw) and SOL claims. */
import type { Config } from '@wagmi/core'
import { formatUnits, getAddress, type Address } from 'viem'
import { config, explorer } from '../config'
import type { Wallets, WalletState } from '../lib/appkit'
import { api, ApiError, poll, type Account, type ClaimStatus, type Cursor, type Flow, type HistoryItems, type NavItem, type Settlement, type Stats, type Trade } from '../lib/api'
import { LineChart } from '../lib/chart'
import { ClaimError, runClaim, solSignatureBytes, type ClaimStep } from '../lib/claim'
import { humanError } from '../lib/evm'
import { duration, parseAmount, pct, short, sol, solDelta, token, tokenFloat, usd, utc } from '../lib/format'
import { $, busy, h, link, mountWalletBar, setText, statusLine, tabs, txLink } from '../lib/ui'
import * as vault from '../lib/vault'

const now = () => Math.floor(Date.now() / 1000)
const SOL = (l: number | null | undefined, d = 4) => (l == null ? '—' : `${sol(l, d)} SOL`)

const state = {
  stats: null as Stats | null,
  statsError: null as string | null,
  totals: null as vault.VaultTotals | null,
  evm: null as Address | null,
  sol: null as string | null,
  cfg: null as Config | null,
  wallets: null as Wallets | null,
  position: null as vault.Position | null,
  account: null as Account | null,
  minLamports: config.claimMinLamports as number,
  claimRunning: false,
}

/* ---------- contracts block ---------- */

function addrCell(id: string, href: string | null, text: string) {
  $(id).replaceChildren(href ? link(href, text) : text)
}
addrCell('c-vault', config.vault ? explorer.evmAddress(config.vault) : null, config.vault ? getAddress(config.vault) : 'not deployed yet')
addrCell('c-timelock', config.timelock ? explorer.evmAddress(config.timelock) : null, config.timelock ? getAddress(config.timelock) : 'not deployed yet')
addrCell('c-fly', config.fly.address ? explorer.evmAddress(config.fly.address) : null, config.fly.address ? getAddress(config.fly.address) : 'not configured')

/* ---------- banners ---------- */

function banner(kind: 'info' | 'warn' | 'danger', title: string, body: (Node | string)[]): HTMLElement {
  return h('div', { class: `notice ${kind === 'danger' ? 'danger' : kind === 'info' ? 'info' : ''}` }, h('span', { class: 'micro' }, title), h('p', {}, ...body))
}

function renderBanners() {
  const s = state.stats
  const out: HTMLElement[] = []
  if (!config.vault) {
    out.push(banner('info', 'Vault not deployed yet', ['The FlyVault contract is not live yet, so locking is not open. The fly\'s numbers below are already public.']))
  }
  if (s?.book && s.book !== 'live') {
    out.push(banner('info', 'Paper book', [
      `These are the running fly's real decisions on its paper book (${s.book}): real market, real model, real trades and NAV, but no real SOL moves. `,
      'Deposits show the paper starting bankroll; the wallet address pays test claims only.',
    ]))
  }
  if (state.statsError) {
    out.push(banner('warn', 'Stats unavailable', [state.statsError, ' On-chain data and your position still work.']))
  }
  const paused = state.totals?.paused ?? s?.vault.paused
  if (paused) {
    out.push(banner('danger', 'Vault paused', ['New locks and cancellations are paused. Withdrawal requests and withdrawals still work, and earnings keep accruing.']))
  }
  const up = s?.vault.upgrade_scheduled
  if (up) {
    const delay = state.totals?.withdrawDelay ?? config.withdrawDelayS
    const exitFirst = now() + delay < up.eta
    out.push(
      banner('warn', 'Contract upgrade scheduled', [
        `An upgrade of the vault contract can execute after ${utc(up.eta)} (in ${duration(up.eta - now())}). Operation `,
        h('code', {}, short(up.id, 6, 6)),
        config.timelock ? ' on the ' : '',
        config.timelock ? link(explorer.evmAddress(config.timelock), 'timelock') : '',
        '. ',
        exitFirst
          ? 'A withdrawal requested now is ready before it can execute.'
          : 'A withdrawal requested now would be ready after it can execute.',
      ]),
    )
  }
  if (s?.fly.kill_switch) out.push(banner('danger', 'Kill switch', ['The fly hit its drawdown limit and stopped trading. Claims of what is already owed still work.']))
  else if (s?.fly.state === 'halted') out.push(banner('warn', 'Fly halted', ['The fly is not trading right now. Claims still work.']))
  else if (s?.fly.entries_paused) out.push(banner('info', 'Entries paused', ['The fly is not opening new positions right now; it still manages open ones.']))
  $('banners').replaceChildren(...out)
}

/* ---------- tiles + wallet ---------- */

function renderStats() {
  const s = state.stats
  const flyUsd = s?.prices.fly_usd ?? 0
  const lockedWei = state.totals?.totalLocked ?? (s ? BigInt(s.vault.total_locked || '0') : null)
  const pendingWei = state.totals?.totalPending ?? (s ? BigInt(s.vault.total_pending || '0') : null)
  setText('t-locked', lockedWei == null ? '—' : `${token(lockedWei, 18, 0)} $FLY`)
  setText('t-locked-usd', lockedWei != null && flyUsd > 0 ? usd(tokenFloat(lockedWei) * flyUsd) : '—')
  setText('t-pending', pendingWei == null ? '—' : `${token(pendingWei, 18, 0)} $FLY`)
  setText('t-earners', s ? String(s.vault.earners) : '—')

  if (s) {
    const navSol = s.wallet.nav / 1e9
    setText('t-nav', SOL(s.wallet.nav, 3))
    setText('t-nav-usd', s.prices.sol_usd > 0 ? usd(navSol * s.prices.sol_usd) : '—')
    setText('t-nav-fly', s.prices.sol_usd > 0 && flyUsd > 0 ? token(BigInt(Math.floor((navSol * s.prices.sol_usd) / flyUsd)) * 10n ** 18n, 18, 0) : '—')
    $('w-address').replaceChildren(link(explorer.solAccount(s.fly.wallet), s.fly.wallet))
    if (s.book && s.book !== 'live') $('w-address').append(` (claims only; the trading is paper book ${s.book})`)
    $('c-wallet').replaceChildren(link(explorer.solAccount(s.fly.wallet), short(s.fly.wallet, 6, 6)))
    const L = s.ledger
    setText('w-native', SOL(s.wallet.native))
    setText('w-deposits', SOL(L.deposits))
    setText('w-withdrawals', SOL(L.withdrawals))
    setText('w-owed', SOL(L.reserved))
    setText('w-claimed', SOL(L.claims_paid))
    setText('w-realized', `${solDelta(L.realized)} SOL`, `v ${L.realized >= 0 ? 'lime-text' : 'red'}`)
    setText('w-booked', `${solDelta(L.booked_realized)} SOL (Δ ${solDelta(L.realized - L.booked_realized)})`)
    setText('w-allocated', SOL(L.allocated))
    setText('w-pot', SOL(L.pot))
    const last = s.settlement.last
    setText('w-last', last ? `${utc(last.period_end, false)} · ${SOL(last.allocated)}` : 'none yet')
  }
  renderBanners()
}

function tick() {
  for (const el of document.querySelectorAll<HTMLElement>('[data-ready-at]')) {
    const left = Number(el.dataset.readyAt) - now()
    el.textContent = left > 0 ? `ready in ${duration(left)}` : 'ready now'
  }
}
setInterval(tick, 1000)

async function refreshStats() {
  try {
    state.stats = await api.stats()
    state.statsError = null
  } catch (e) {
    state.statsError =
      e instanceof ApiError && e.status === 503
        ? 'The fly has not published stats yet.'
        : 'The stats relay is not reachable right now.'
  }
  renderStats()
  tick()
}

async function refreshTotals() {
  if (!config.vault) return
  try {
    state.totals = await vault.readTotals()
    renderStats()
  } catch {
    /* RPC trouble: tiles fall back to the fly's indexed numbers */
  }
}

/* ---------- tables ---------- */

function td(text: string | Node, cls = ''): HTMLTableCellElement {
  return h('td', { class: cls }, text)
}
function signed(l: number): HTMLTableCellElement {
  return td(solDelta(l), `num ${l > 0 ? 'lime-text' : l < 0 ? 'red' : ''}`)
}
function empty(tbody: HTMLElement, cols: number, text: string) {
  tbody.replaceChildren(h('tr', {}, h('td', { colspan: cols, class: 'muted' }, text)))
}

const ROWS: { [K in 'settlements' | 'trades' | 'flows']: (x: HistoryItems[K]) => HTMLTableRowElement } = {
  settlements: (x: Settlement) =>
    h(
      'tr',
      {},
      td(utc(x.period_end, false)),
      signed(x.realized),
      td(sol(x.pot), 'num'),
      td(sol(x.allocated), 'num'),
      td(sol(x.carried), 'num'),
      td(String(x.earners), 'num'),
      td(x.status),
    ),
  trades: (x: Trade) =>
    h(
      'tr',
      {},
      td(utc(x.closed_at)),
      td(link(explorer.solToken(x.mint), x.symbol || short(x.mint))),
      td(duration(x.closed_at - x.opened_at), 'num'),
      td(sol(x.cost), 'num'),
      td(sol(x.proceeds), 'num'),
      signed(x.realized),
      td(x.exit_kind || '—'),
    ),
  flows: (x: Flow) =>
    h(
      'tr',
      {},
      td(utc(x.ts)),
      td(h('span', { class: `tag ${x.kind}` }, x.kind)),
      td(`${x.direction === 'out' ? '−' : '+'}${sol(x.lamports)}`, `num ${x.direction === 'in' ? 'lime-text' : ''}`),
      td(x.counterparty ? link(explorer.solAccount(x.counterparty), short(x.counterparty)) : '—'),
      td(x.signature ? link(explorer.solTx(x.signature), short(x.signature, 6, 6)) : '—'),
    ),
}

class Table<K extends 'settlements' | 'trades' | 'flows'> {
  private items = new Map<string, HistoryItems[K]>()
  private next: Cursor | null = null
  private tbody: HTMLElement
  private more: HTMLButtonElement

  constructor(
    private kind: K,
    private sortKey: (x: HistoryItems[K]) => number,
  ) {
    this.tbody = $(`tbl-${kind}`).querySelector('tbody')!
    this.more = document.querySelector<HTMLButtonElement>(`[data-more="${kind}"]`)!
    this.more.addEventListener('click', () => void busy(this.more, () => this.load(this.next)))
    empty(this.tbody, 7, 'Loading…')
  }

  /** before = null loads (or refreshes) the newest page and keeps anything older already loaded. */
  async load(before: Cursor | null = null) {
    try {
      const page = await api.history(this.kind, before, 25)
      for (const it of page.items) this.items.set(String(it.id), it)
      if (before != null || this.items.size === page.items.length) this.next = page.next
      this.render()
    } catch (e) {
      if (!this.items.size) empty(this.tbody, 7, e instanceof ApiError && e.status === 503 ? 'Not published yet.' : 'Unavailable right now.')
    }
  }

  private render() {
    const rows = [...this.items.values()].sort((a, b) => this.sortKey(b) - this.sortKey(a))
    if (!rows.length) empty(this.tbody, 7, 'Nothing yet.')
    else this.tbody.replaceChildren(...rows.map((r) => ROWS[this.kind](r as never)))
    this.more.hidden = this.next == null
  }
}

const tables = [
  new Table('settlements', (x) => x.period_end),
  new Table('trades', (x) => x.closed_at),
  new Table('flows', (x) => x.ts),
]

/* ---------- NAV chart ---------- */

const nav = {
  unit: 'sol' as 'sol' | 'usd',
  points: new Map<number, NavItem>(),
  next: null as Cursor | null,
  chart: null as LineChart | null,
  chartUnit: '',
}

function navChart(): LineChart {
  if (nav.chart && nav.chartUnit === nav.unit) return nav.chart
  const fmt =
    nav.unit === 'usd' ? (v: number) => usd(v) : (v: number) => v.toFixed(3)
  nav.chart?.remove()
  nav.chart = new LineChart($('nav-chart'), fmt)
  nav.chartUnit = nav.unit
  return nav.chart
}

function renderNav(fit: boolean) {
  const pts = [...nav.points.values()].map((p) => ({
    time: p.ts,
    value: nav.unit === 'sol' ? p.nav / 1e9 : (p.nav / 1e9) * p.sol_usd,
  }))
  const msg = $('nav-msg')
  msg.hidden = pts.length > 0
  msg.textContent = 'No NAV history published yet.'
  navChart().set(pts, fit)
  $('nav-more').hidden = nav.next == null
}

async function loadNav(before: Cursor | null, pages = 1) {
  let cursor = before
  try {
    for (let i = 0; i < pages; i++) {
      const page = await api.history('nav', cursor, 500)
      for (const p of page.items) nav.points.set(p.ts, p)
      if (before != null || i > 0 || nav.next == null) nav.next = page.next
      cursor = page.next
      if (cursor == null) break
    }
    renderNav(before == null)
  } catch {
    if (!nav.points.size) {
      $('nav-msg').hidden = false
      $('nav-msg').textContent = 'NAV history is unavailable right now.'
    }
  }
}

tabs($('nav-tabs'), 'unit', (v) => {
  nav.unit = v as typeof nav.unit
  renderNav(true)
})
$<HTMLButtonElement>('nav-more').addEventListener('click', (e) =>
  void busy(e.currentTarget as HTMLButtonElement, () => loadNav(nav.next, 1)),
)

/* ---------- your position ---------- */

const posStatus = () => statusLine('pos-status')

function amountForm(id: string, label: string, max: bigint, cta: string, onSubmit: (amount: bigint) => Promise<void>, disabledNote?: string) {
  const input = h('input', { id: `${id}-amount`, type: 'text', inputmode: 'decimal', autocomplete: 'off', placeholder: '0.0', spellcheck: 'false' })
  const btn = h('button', { type: 'submit', class: 'btn primary' }, cta) as HTMLButtonElement
  if (disabledNote) btn.disabled = true
  const form = h(
    'form',
    { class: 'mini-form', novalidate: true },
    h('label', { for: `${id}-amount` }, label),
    h(
      'div',
      { class: 'input-row' },
      input,
      h('button', { type: 'button', class: 'btn small', onclick: () => (input.value = formatUnits(max, 18)) }, 'Max'),
      btn,
    ),
    disabledNote ? h('p', { class: 'hint' }, disabledNote) : null,
  )
  form.addEventListener('submit', (e) => {
    e.preventDefault()
    const amount = parseAmount(input.value, 18)
    if (amount == null) return posStatus()('Enter a valid amount.', 'error')
    if (amount > max) return posStatus()(`That is more than the ${token(max)} $FLY available.`, 'error')
    void busy(btn, () => onSubmit(amount))
  })
  return form
}

async function tx(run: (step: (m: string) => void) => Promise<`0x${string}`>) {
  const status = posStatus()
  try {
    const hash = await run((m) => status(m, 'busy'))
    status(txLink(hash), 'ok')
  } catch (e) {
    status(humanError(e), 'error')
  }
  await refreshPosition()
  void refreshTotals()
}

function renderPosition() {
  const body = $('position-body')
  const { evm, cfg, position: p, account: a } = state
  if (!evm) {
    body.replaceChildren(h('p', { class: 'muted' }, 'Connect an EVM wallet to see your $FLY, lock, and withdraw.'))
    return
  }
  const paused = state.totals?.paused ?? state.stats?.vault.paused ?? false
  const kv = h(
    'div',
    { class: 'kv compact' },
    h('div', {}, h('span', { class: 'micro' }, '$FLY in wallet'), h('div', { class: 'v' }, p ? token(p.flyBalance) : '—')),
    h('div', {}, h('span', { class: 'micro' }, 'Locked (earning)'), h('div', { class: 'v accent' }, p ? token(p.locked) : '—')),
    h('div', {}, h('span', { class: 'micro' }, 'Pending withdrawal'), h('div', { class: 'v' }, p ? token(p.pending) : '—')),
    h('div', {}, h('span', { class: 'micro' }, 'Share of the vault'), h('div', { class: 'v' }, sharePct(p))),
    h('div', {}, h('span', { class: 'micro' }, 'Earned, all time'), h('div', { class: 'v' }, a ? SOL(a.allocated) : '—')),
    h('div', {}, h('span', { class: 'micro' }, 'Owed to you now'), h('div', { class: 'v lime-text' }, a ? SOL(a.owed) : '—')),
    h('div', {}, h('span', { class: 'micro' }, 'Claimed'), h('div', { class: 'v' }, a ? SOL(a.claimed) : '—')),
    h('div', {}, h('span', { class: 'micro' }, 'Claim in flight'), h('div', { class: 'v' }, a ? SOL(a.in_flight) : '—')),
  )
  const parts: (Node | null)[] = [kv]
  if (!config.vault) {
    parts.push(h('p', { class: 'hint mt-16' }, 'Locking opens once the vault contract is deployed.'))
  } else if (p && cfg) {
    parts.push(
      amountForm('lock', 'Lock $FLY', p.flyBalance, 'Lock', (amt) => tx((step) => vault.lock(cfg, evm, amt, step)), paused ? 'The vault is paused: locking is blocked for now.' : undefined),
      amountForm('req', 'Request a withdrawal (stops earning now; withdrawable after 7 days)', p.locked, 'Request', (amt) =>
        tx((step) => vault.requestWithdrawal(cfg, evm, amt, step)),
      ),
    )
    const pending = p.requests.filter((r) => r.state === 'pending')
    if (pending.length) {
      parts.push(
        h('h4', { class: 'mt-16' }, 'Pending withdrawals'),
        h(
          'ul',
          { class: 'req-rows' },
          ...pending.map((r) => {
            const ready = now() >= r.readyAt
            return h(
              'li',
              {},
              h('span', { class: 'mono' }, `#${r.id} · ${token(r.amount)} $FLY`),
              h('span', { class: 'micro', 'data-ready-at': r.readyAt }, ready ? 'ready now' : `ready in ${duration(r.readyAt - now())}`),
              h('span', { class: 'req-actions' },
                h('button', { type: 'button', class: 'btn small', disabled: paused, title: paused ? 'Paused: cancelling is blocked' : 'Lock this amount again', onclick: (e: Event) => void busy(e.currentTarget as HTMLButtonElement, () => tx((step) => vault.cancelRequest(cfg, evm, r.id, step))) }, 'Cancel'),
                h('button', { type: 'button', class: 'btn small primary', disabled: !ready, onclick: (e: Event) => void busy(e.currentTarget as HTMLButtonElement, () => tx((step) => vault.withdraw(cfg, evm, r.id, step))) }, 'Withdraw'),
              ),
            )
          }),
        ),
      )
    }
    const done = p.requests.length - pending.length
    if (done) parts.push(h('p', { class: 'hint' }, `${done} earlier request${done > 1 ? 's' : ''} cancelled or withdrawn.`))
  } else {
    parts.push(h('p', { class: 'hint mt-16' }, 'Reading your position…'))
  }
  if (a?.allocations.length) {
    parts.push(
      h('details', { class: 'mt-16' },
        h('summary', {}, `Your weekly allocations (${a.allocations.length})`),
        h('div', { class: 'table-wrap' }, h('table', { class: 'data' },
          h('thead', {}, h('tr', {}, h('th', {}, 'Week ending'), h('th', { class: 'num' }, 'SOL'), h('th', { class: 'num' }, 'Share'))),
          h('tbody', {}, ...a.allocations.slice().sort((x, y) => y.period_end - x.period_end).map((x) =>
            h('tr', {}, td(utc(x.period_end, false)), td(sol(x.lamports), 'num'), td(pct(x.share, true), 'num')))),
        )),
      ),
    )
  }
  parts.push(h('p', { class: 'status-line', id: 'pos-status', role: 'status', 'aria-live': 'polite', hidden: true }))
  body.replaceChildren(...parts.filter((x): x is Node => x != null))
}

function sharePct(p: vault.Position | null): string {
  const total = state.totals?.totalLocked ?? (state.stats ? BigInt(state.stats.vault.total_locked || '0') : 0n)
  if (!p || total === 0n) return '—'
  return pct(Number((p.locked * 1_000_000n) / total) / 1_000_000, true)
}

async function refreshPosition() {
  const evm = state.evm
  if (!evm) return renderPosition()
  const [pos, acct] = await Promise.all([
    config.vault ? vault.readPosition(evm).catch(() => state.position) : Promise.resolve(null),
    api.account(evm).catch(() => state.account),
  ])
  if (evm !== state.evm) return
  state.position = pos
  state.account = acct
  // Re-rendering would wipe what someone is typing; only do it when no form input has focus.
  if (!(document.activeElement instanceof HTMLInputElement && $('position-body').contains(document.activeElement))) renderPosition()
  renderClaim()
}

/* ---------- claim ---------- */

const STATUS_TEXT: Record<string, string> = {
  received: 'Received by the relay. Waiting for the fly to pick it up…',
  verified: 'Signatures verified. Preparing the payment…',
  waiting_liquidity: 'Verified. The fly is freeing up SOL to pay you (for example by closing a position)…',
  sending: 'Sending SOL…',
  paid: 'Paid.',
  rejected: 'Rejected.',
  failed: 'Failed.',
}

let claimLog: HTMLElement | null = null
let claimTexts: HTMLElement | null = null

function renderClaim() {
  if (state.claimRunning) return
  const body = $('claim-body')
  const { evm, sol: solAddr, account: a } = state
  if (!state.wallets) {
    body.replaceChildren(h('p', { class: 'muted' }, config.reownProjectId ? 'Loading wallet support…' : 'Claiming needs wallet connection, which is not configured on this build.'))
    return
  }
  const need: string[] = []
  if (!evm) need.push('the EVM wallet that locked $FLY')
  if (!solAddr) need.push('the Solana wallet that should receive the SOL')
  const parts: Node[] = [
    h('div', { class: 'kv compact' },
      h('div', {}, h('span', { class: 'micro' }, 'Owed to you'), h('div', { class: 'v lime-text' }, a ? SOL(a.owed) : '—')),
      h('div', {}, h('span', { class: 'micro' }, 'Minimum claim'), h('div', { class: 'v' }, SOL(state.minLamports, 3))),
    ),
  ]
  if (need.length) {
    parts.push(h('p', { class: 'hint mt-16' }, `Connect ${need.join(' and ')} in the wallet panel.`))
  } else if (a) {
    const inFlight = a.in_flight > 0 || a.claims.some((c) => ['received', 'verified', 'waiting_liquidity', 'sending'].includes(c.status))
    const tooSmall = a.owed < state.minLamports
    const btn = h('button', { type: 'button', class: 'btn primary block mt-16', disabled: inFlight || tooSmall }, inFlight ? 'A claim is in progress' : tooSmall ? 'Nothing to claim yet' : `Claim ${SOL(a.owed)} to ${short(solAddr)}`) as HTMLButtonElement
    btn.addEventListener('click', () => void busy(btn, claim))
    parts.push(
      btn,
      h('p', { class: 'hint' }, 'You will sign two messages, one in each wallet. Signing costs nothing and sends no transaction. The fly then pays everything owed to your EVM address.'),
    )
  }
  claimLog = h('ol', { class: 'claim-log', 'aria-live': 'polite' })
  claimTexts = h('div')
  parts.push(claimTexts, claimLog)
  if (a?.claims.length) {
    parts.push(
      h('details', { class: 'mt-16' }, h('summary', {}, `Your claims (${a.claims.length})`),
        h('ul', { class: 'req-rows' }, ...a.claims.slice().sort((x, y) => y.created_at - x.created_at).map((c) =>
          h('li', {}, h('span', { class: 'mono' }, `#${c.id} · ${SOL(c.lamports)}`), h('span', { class: `tag ${c.status}` }, c.status), c.tx ? link(explorer.solTx(c.tx), short(c.tx, 6, 6)) : h('span', {}, utc(c.created_at)))))),
    )
  }
  body.replaceChildren(...parts)
}

function logStep(text: string | Node, kind: 'busy' | 'ok' | 'error' | 'info' = 'info') {
  claimLog?.append(h('li', { class: kind }, text))
}

async function claim() {
  const w = state.wallets
  const { evm, sol: solAddr } = state
  if (!w || !evm || !solAddr) return
  state.claimRunning = true
  claimLog?.replaceChildren()
  const steps: Record<ClaimStep['step'], string> = {
    challenge: 'Getting a claim challenge from the relay…',
    'sign-evm': 'Sign the first message in your EVM wallet.',
    'sign-sol': 'Sign the second message in your Solana wallet.',
    submit: 'Checking both signatures and sending the claim…',
    status: '',
  }
  let lastStatus = ''
  try {
    const final = await runClaim(
      {
        evm,
        sol: solAddr,
        signEvm: (text) => w.signEvm(text),
        signSol: async (msg) => solSignatureBytes(await w.signSol(msg)),
      },
      (s) => {
        if (s.step === 'sign-evm' && claimTexts) {
          claimTexts.replaceChildren(
            h('details', { class: 'mt-16' }, h('summary', {}, 'What you are signing'),
              h('span', { class: 'micro' }, 'EVM wallet'), h('pre', {}, s.evmText),
              h('span', { class: 'micro' }, 'Solana wallet'), h('pre', {}, s.solText)),
          )
        }
        if (s.step === 'status') {
          if (s.claim.status !== lastStatus) {
            lastStatus = s.claim.status
            logClaimStatus(s.claim)
          }
        } else logStep(steps[s.step], 'busy')
      },
      { minLamports: (n) => (state.minLamports = n || state.minLamports) },
    )
    if (final.status !== 'paid') logStep('Nothing was paid. You can try again.', 'error')
  } catch (e) {
    logStep(e instanceof ClaimError ? e.message : humanError(e), 'error')
  } finally {
    state.claimRunning = false
    void refreshPosition()
  }
}

function logClaimStatus(c: ClaimStatus) {
  const kind = c.status === 'paid' ? 'ok' : c.status === 'rejected' || c.status === 'failed' ? 'error' : 'busy'
  const parts: (Node | string)[] = [`Claim #${c.id}: ${STATUS_TEXT[c.status] ?? c.status}`]
  if (c.reason) parts.push(` ${c.reason}`)
  if (c.lamports) parts.push(` ${SOL(c.lamports)}.`)
  if (c.tx) parts.push(' ', link(explorer.solTx(c.tx), 'View on Solscan'))
  logStep(h('span', {}, ...parts), kind)
}

/* ---------- wallets ---------- */

function onWallet(s: WalletState, w: Wallets) {
  state.wallets = w
  state.cfg = w.wagmiConfig
  const evmChanged = s.evm !== state.evm
  state.evm = s.evm
  state.sol = s.sol
  if (evmChanged) {
    state.position = null
    state.account = null
    renderPosition()
    void refreshPosition()
  }
  renderClaim()
}

/* ---------- start ---------- */

tick()
poll(refreshStats, 60_000)
poll(refreshTotals, 60_000)
poll(() => Promise.all(tables.map((t) => t.load(null))), 300_000)
void loadNav(null, 2)
poll(() => (nav.points.size ? loadNav(null, 1) : undefined), 120_000)
poll(() => (state.evm && !state.claimRunning ? refreshPosition() : undefined), 45_000)
renderPosition()
setTimeout(() => {
  void mountWalletBar({ solana: true, onChange: onWallet }).then((w) => {
    if (!w) renderClaim()
  })
}, 0)
