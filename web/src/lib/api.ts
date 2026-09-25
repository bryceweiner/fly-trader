/** Relay client (docs/vault/SPEC.md §3, payloads §4). Lamports are numbers, $FLY amounts are wei strings. */
import { config } from '../config'

export interface Position {
  mint: string
  symbol: string
  opened_at: number
  cost: number
  value: number
  entry_price: number
  mark_price: number
  hold_min: number
}

export interface Settlement {
  id: number | string
  period_start: number
  period_end: number
  realized: number
  pot: number
  allocated: number
  carried: number
  earners: number
  total_weight: string
  status: string
}

export interface Stats {
  v: number
  ts: number
  /** the book the vault shares: 'live', or a paper book on a dry run */
  book?: string
  cluster: string
  fly: {
    state: 'starting' | 'paper' | 'live' | 'halted' | string
    handover: boolean
    kill_switch: boolean
    entries_paused: boolean
    model: Record<string, number>
  }
  /** as of an hour ago (the fly publishes nothing that would let anyone trade ahead of it) */
  wallet: { native: number; nav: number }
  ledger: {
    deposits: number
    withdrawals: number
    claims_paid: number
    realized: number
    booked_realized: number
    allocated: number
    reserved: number
    pot: number
  }
  prices: { sol_usd: number; fly_usd: number; ts: number }
  vault: {
    address: string
    chain_id: number
    total_locked: string
    total_pending: string
    earners: number
    finalized_block: number
    paused: boolean
    impl: string
    upgrade_scheduled: { eta: number; id: string } | null
  }
  settlement: { last: Settlement | null }
}

export interface NavItem {
  ts: number
  nav: number
  sol_usd: number
}

export interface Trade {
  id: number | string
  mint: string
  symbol: string
  opened_at: number
  closed_at: number
  cost: number
  proceeds: number
  realized: number
  exit_kind: string
}

export interface Flow {
  id: number | string
  ts: number
  direction: 'in' | 'out'
  kind: 'deposit' | 'profit' | 'withdrawal' | 'claim' | string
  lamports: number
}

export interface HistoryItems {
  nav: NavItem
  trades: Trade
  flows: Flow
  settlements: Settlement
}
export type HistoryKind = keyof HistoryItems
export type Cursor = string | number

export interface HistoryPage<K extends HistoryKind> {
  kind: K
  items: HistoryItems[K][]
  next: Cursor | null
}

export interface AccountClaim {
  id: number | string
  status: string
  lamports: number
  sol: string
  tx: string
  created_at: number
  updated_at: number
}

export interface Account {
  evm: string
  allocated: number
  claimed: number
  in_flight: number
  owed: number
  allocations: { period_end: number; lamports: number; weight: string; share: number }[]
  claims: AccountClaim[]
}

export interface Challenge {
  nonce: string
  issued_at: string
  expires_at: string
  domain: string
  uri: string
  chain_id: number
  sol_chain: string
  request_id: string
  min_lamports: number
}

export type ClaimState = 'received' | 'verified' | 'waiting_liquidity' | 'sending' | 'paid' | 'rejected' | 'failed'

export interface ClaimStatus {
  id: number | string
  status: ClaimState
  reason: string | null
  lamports: number | null
  tx: string | null
  created_at: number
  updated_at: number
}

export class ApiError extends Error {
  constructor(
    public status: number,
    message: string,
  ) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response
  try {
    res = await fetch(config.relayBase + path, {
      cache: 'no-store',
      ...init,
      headers: { accept: 'application/json', ...(init?.body ? { 'content-type': 'application/json' } : {}) },
    })
  } catch {
    throw new ApiError(0, 'The relay could not be reached.')
  }
  let body: unknown = null
  try {
    body = await res.json()
  } catch {
    /* non-JSON error page */
  }
  if (!res.ok) {
    const msg = (body as { error?: string } | null)?.error || `Relay error ${res.status}`
    throw new ApiError(res.status, msg)
  }
  return body as T
}

const q = (o: Record<string, string | number | undefined>) =>
  new URLSearchParams(Object.entries(o).filter(([, v]) => v !== undefined && v !== '') as [string, string][]).toString()

export const api = {
  stats: () => request<Stats>('/stats'),
  history: <K extends HistoryKind>(kind: K, before?: Cursor | null, limit = 100) =>
    request<HistoryPage<K>>(`/history?${q({ kind, before: before == null ? undefined : String(before), limit })}`),
  account: (evm: string) => request<Account>(`/account?${q({ evm })}`),
  challenge: (evm: string, sol: string) => request<Challenge>(`/claim/challenge?${q({ evm, sol })}`),
  submitClaim: (body: { nonce: string; evm: string; sol: string; evm_sig: string; sol_sig: string }) =>
    request<{ id: number | string; status: ClaimState }>('/claim', { method: 'POST', body: JSON.stringify(body) }),
  claim: (id: number | string) => request<ClaimStatus>(`/claim/${encodeURIComponent(String(id))}`),
}

/**
 * Calls `fn` now and every `everyMs` while the tab is visible (hidden tabs skip ticks and refresh on return),
 * which keeps a page well inside the relay's 600 reads/hour and GeckoTerminal's 30/min.
 */
export function poll(fn: () => unknown, everyMs: number): () => void {
  let last = 0
  const tick = () => {
    if (document.visibilityState === 'hidden') return
    last = Date.now()
    void fn()
  }
  const timer = setInterval(tick, everyMs)
  const onVis = () => {
    if (document.visibilityState === 'visible' && Date.now() - last > everyMs) tick()
  }
  document.addEventListener('visibilitychange', onVis)
  tick()
  return () => {
    clearInterval(timer)
    document.removeEventListener('visibilitychange', onVis)
  }
}
