// Writes dev/fixtures/*.json: realistic relay payloads (docs/vault/SPEC.md §4) for `npm run dev`.
// Deterministic (seeded), so re-running produces the same files. Timestamps are relative to REF;
// dev/mock-api.ts shifts them to "now" when serving.
//   node dev/make-fixtures.mjs
import { mkdirSync, writeFileSync } from 'node:fs'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'
import bs58 from 'bs58'

const OUT = join(dirname(fileURLToPath(import.meta.url)), 'fixtures')
const REF = Date.UTC(2026, 8, 23, 12, 0, 0) / 1000 // Wed 2026-09-23 12:00 UTC
const WEEK = 604800
const L = 1e9

let seed = 0x5eed1234
const rnd = () => ((seed = (seed * 1664525 + 1013904223) >>> 0) / 2 ** 32)
const bytes = (n) => Uint8Array.from({ length: n }, () => Math.floor(rnd() * 256))
const addr = () => bs58.encode(bytes(32))
const sig = () => bs58.encode(bytes(64))
const pick = (a) => a[Math.floor(rnd() * a.length)]
const int = (x) => Math.round(x)

const mondayFloor = (t) => {
  const d = new Date(t * 1000)
  const mid = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000
  return mid - ((d.getUTCDay() + 6) % 7) * 86400
}

const FLY_WALLET = addr()
const FUNDING = addr()
const START = mondayFloor(REF) - 3 * WEEK // the fly started four Mondays ago
const SYMBOLS = ['MOTH', 'BZZT', 'WING', 'LARVA', 'PUPA', 'NECTAR', 'SWARM', 'HIVE', 'ANTNA', 'DROSO', 'FRUIT', 'MAGGOT', 'COMPOUND', 'PROBOS', 'HALTERE']
const MINTS = Object.fromEntries(SYMBOLS.map((s) => [s, addr() .slice(0, 40) + 'pump']))
const EXITS = ['trail', 'target', 'stop', 'bitter', 'satiety', 'turned', 'max_hold', 'dead_bag']

// ---- trades: ~6 per day, slightly positive edge ----
const trades = []
let t = START + 3600
let id = 1
while (t < REF - 1800) {
  const sym = pick(SYMBOLS)
  const cost = int((0.15 + rnd() * 0.3) * L)
  const ret = (rnd() - 0.44) * 0.22 - 0.004
  const hold = int(600 + rnd() * 5400)
  const proceeds = int(cost * (1 + ret))
  trades.push({ id: id++, mint: MINTS[sym], symbol: sym, opened_at: t, closed_at: t + hold, cost, proceeds, realized: proceeds - cost, exit_kind: pick(EXITS) })
  t += int(1800 + rnd() * 12000)
}
const realizedBefore = (ts) => trades.filter((x) => x.closed_at < ts).reduce((a, x) => a + x.realized, 0)

// ---- settlements for every finished week ----
const settlements = []
let allocated = 0
for (let w = 0; mondayFloor(REF) > START + (w + 1) * WEEK - 1; w++) {
  const ps = START + w * WEEK
  const pe = ps + WEEK
  const R = realizedBefore(pe) - int(0.0004 * L * (w + 1)) // fees the trade records miss
  const pot = Math.max(0, R - allocated)
  const alloc = int(pot * 0.999)
  settlements.push({ id: w + 1, period_start: ps, period_end: pe, realized: R, pot, allocated: alloc, carried: pot - alloc, earners: 14 + w * 9, total_weight: String(BigInt(int(2.1e8 + w * 6e7)) * 10n ** 18n * 86400n), status: 'allocated' })
  allocated += alloc
}

// ---- flows ----
const flows = [
  { id: 1, ts: START - 7200, signature: sig(), direction: 'in', kind: 'deposit', counterparty: FUNDING, lamports: 3 * L },
  { id: 2, ts: START + 5 * 86400, signature: sig(), direction: 'in', kind: 'deposit', counterparty: FUNDING, lamports: 2 * L },
  { id: 3, ts: START + 9 * 86400, signature: sig(), direction: 'in', kind: 'profit', counterparty: addr(), lamports: int(0.05 * L) },
]
let fid = 4
let claimsPaid = 0
for (const s of settlements) {
  for (let k = 0; k < 3; k++) {
    const lam = int(s.allocated * (0.05 + rnd() * 0.2))
    claimsPaid += lam
    flows.push({ id: fid++, ts: s.period_end + int(3600 + rnd() * 400000), signature: sig(), direction: 'out', kind: 'claim', counterparty: addr(), lamports: lam })
  }
}
flows.sort((a, b) => b.ts - a.ts)

// ---- NAV history, every 15 min ----
const deposits = 5 * L
const nav = []
for (let ts = START; ts <= REF; ts += 900) {
  const dep = flows.filter((f) => f.kind === 'deposit' && f.ts <= ts).reduce((a, f) => a + f.lamports, 0)
  const paid = flows.filter((f) => f.kind === 'claim' && f.ts <= ts).reduce((a, f) => a + f.lamports, 0)
  const v = dep + realizedBefore(ts) - paid + int((rnd() - 0.5) * 0.04 * L)
  nav.push({ ts, nav: v, index: +(1 + realizedBefore(ts) / deposits).toFixed(5), sol_usd: +(214 + Math.sin(ts / 90000) * 9 + rnd() * 2).toFixed(2) })
}
nav.reverse()

// ---- stats ----
const R = realizedBefore(REF) + 0.05 * L - int(0.0016 * L)
const booked = realizedBefore(REF)
const positions = [0, 1, 2].map((i) => {
  const sym = SYMBOLS[(i * 5) % SYMBOLS.length]
  const cost = int((0.18 + i * 0.07) * L)
  const move = [0.083, -0.021, 0.004][i]
  const entry = [0.00000412, 0.0000187, 0.000093][i]
  return { mint: MINTS[sym], symbol: sym, opened_at: REF - [2400, 900, 5400][i], cost, value: int(cost * (1 + move)), entry_price: entry, mark_price: +(entry * (1 + move)).toPrecision(6), hold_min: [40, 15, 90][i] }
})
const openCost = positions.reduce((a, p) => a + p.cost, 0)
const posValue = positions.reduce((a, p) => a + p.value, 0)
const tokenAccounts = int(0.0061 * L)
const native = deposits + R - claimsPaid - openCost - tokenAccounts
const exitCost = int(posValue * 0.012)
const stats = {
  v: 1,
  ts: REF - 20,
  cluster: 'mainnet',
  fly: { state: 'live', wallet: FLY_WALLET, handover: true, kill_switch: false, entries_paused: false, model: { fly: 62, selector: 59, release: 3 } },
  wallet: { native, token_accounts: tokenAccounts, open_cost: openCost, positions_value: posValue, exit_cost: exitCost, nav: native + tokenAccounts + posValue - exitCost },
  ledger: { deposits, withdrawals: 0, claims_paid: claimsPaid, realized: R, booked_realized: booked, allocated, reserved: allocated - claimsPaid, pot: Math.max(0, R - allocated) },
  prices: { sol_usd: 218.37, fly_usd: 0.0000412, ts: REF - 45 },
  vault: {
    address: '0x0000000000000000000000000000000000000000',
    chain_id: 4663,
    total_locked: (412_750_000n * 10n ** 18n + 123456789n).toString(),
    total_pending: (18_000_000n * 10n ** 18n).toString(),
    earners: 41,
    finalized_block: 70639000,
    paused: false,
    impl: '0x0000000000000000000000000000000000000000',
    upgrade_scheduled: null,
  },
  settlement: { next_at: mondayFloor(REF) + WEEK, last: settlements.at(-1) ?? null },
  positions,
  index: { value: nav[0].index, peak: Math.max(...nav.map((n) => n.index)), drawdown: 0.012 },
}

// ---- account (served for whatever address the page asks about) ----
const allocations = settlements.map((s, i) => ({ period_end: s.period_end, lamports: int(s.allocated * (0.031 + i * 0.004)), weight: String(BigInt(int(6.5e6 + i * 1e6)) * 10n ** 18n * 86400n), share: +(0.031 + i * 0.004).toFixed(4) }))
const accAllocated = allocations.reduce((a, x) => a + x.lamports, 0)
const accClaimed = allocations[0] ? allocations[0].lamports : 0
const account = {
  evm: '0x0000000000000000000000000000000000000000',
  allocated: accAllocated,
  claimed: accClaimed,
  in_flight: 0,
  owed: accAllocated - accClaimed,
  allocations,
  claims: accClaimed ? [{ id: 7, status: 'paid', lamports: accClaimed, sol: addr(), tx: sig(), created_at: settlements[0].period_end + 90000, updated_at: settlements[0].period_end + 90041 }] : [],
}

const challenge = { nonce: '', issued_at: '', expires_at: '', domain: '', uri: '', chain_id: 4663, sol_chain: 'mainnet', request_id: 'fly-vault-claim-v1', min_lamports: 2000000 }
const claimStatus = { id: 0, status: 'paid', reason: null, lamports: 0, tx: sig(), created_at: 0, updated_at: 0 }

// ---- GeckoTerminal stand-ins for testnet builds (Market shape and raw ohlcv rows, newest first) ----
const geckoPool = { priceUsd: 0.0000412, change24h: 27.9, fdvUsd: 41249.79, liquidityUsd: 20325.84, volume24hUsd: 14752.36 }
const ohlcv = []
let p = 0.000021
for (let i = 0; i < 300; i++) {
  const ts = Math.floor(REF / 3600) * 3600 - (299 - i) * 3600
  const o = p
  const c = Math.max(1e-6, o * (1 + (rnd() - 0.48) * 0.08))
  ohlcv.push([ts, o, Math.max(o, c) * (1 + rnd() * 0.03), Math.min(o, c) * (1 - rnd() * 0.03), c, +(rnd() * 3000).toFixed(2)])
  p = c
}
ohlcv.reverse()

mkdirSync(OUT, { recursive: true })
const files = {
  'stats.json': { ref: REF, stats },
  'history_nav.json': nav,
  'history_trades.json': trades.slice().reverse(),
  'history_flows.json': flows,
  'history_settlements.json': settlements.slice().reverse(),
  'account.json': account,
  'challenge.json': challenge,
  'claim_status.json': claimStatus,
  'gecko_pool.json': geckoPool,
  'gecko_ohlcv.json': ohlcv,
}
for (const [name, body] of Object.entries(files)) writeFileSync(join(OUT, name), JSON.stringify(body, null, 1) + '\n')
console.log(`wrote ${Object.keys(files).length} fixtures to ${OUT}: ${trades.length} trades, ${flows.length} flows, ${settlements.length} settlements, ${nav.length} nav points`)
