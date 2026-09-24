/**
 * Dev-only /api middleware serving dev/fixtures/*.json in the relay's shapes (docs/vault/SPEC.md §3/§4), so
 * `npm run dev` shows full Trading and Vault pages without a backend. Set VITE_RELAY_PROXY to use a real relay.
 * Fixture timestamps are shifted to the present; settlement weeks stay on Monday 00:00 UTC boundaries.
 */
import { randomBytes } from 'node:crypto'
import { readFileSync } from 'node:fs'
import type { IncomingMessage, ServerResponse } from 'node:http'
import { resolve } from 'node:path'
import type { Plugin } from 'vite'

const DIR = resolve(import.meta.dirname, 'fixtures')
const WEEK = 604800
const TIME_KEYS = new Set(['ts', 'opened_at', 'closed_at', 'created_at', 'updated_at', 'next_at'])
const WEEK_KEYS = new Set(['period_start', 'period_end'])

const load = (name: string) => JSON.parse(readFileSync(resolve(DIR, name), 'utf8'))

function mondayFloor(t: number): number {
  const d = new Date(t * 1000)
  const mid = Date.UTC(d.getUTCFullYear(), d.getUTCMonth(), d.getUTCDate()) / 1000
  return mid - ((d.getUTCDay() + 6) % 7) * 86400
}

/** Moves every known timestamp field forward by `delta`; settlement boundaries by whole weeks only. */
function shift<T>(v: T, delta: number): T {
  if (Array.isArray(v)) return v.map((x) => shift(x, delta)) as T
  if (v && typeof v === 'object') {
    const out: Record<string, unknown> = {}
    for (const [k, x] of Object.entries(v)) {
      if (typeof x === 'number' && x > 1e9 && TIME_KEYS.has(k)) out[k] = x + delta
      else if (typeof x === 'number' && x > 1e9 && WEEK_KEYS.has(k)) out[k] = x + Math.floor(delta / WEEK) * WEEK
      else out[k] = shift(x, delta)
    }
    return out as T
  }
  return v
}

function send(res: ServerResponse, status: number, body: unknown) {
  res.statusCode = status
  res.setHeader('Content-Type', 'application/json')
  res.setHeader('Cache-Control', 'no-store')
  res.end(JSON.stringify(body))
}

function readBody(req: IncomingMessage): Promise<string> {
  return new Promise((ok) => {
    let s = ''
    req.on('data', (c) => (s += c))
    req.on('end', () => ok(s))
  })
}

const rfc3339 = (t: number) => new Date(t * 1000).toISOString().replace(/\.\d{3}Z$/, 'Z')

export function devApi(network: 'mainnet' | 'testnet'): Plugin {
  const claims = new Map<number, { created: number; lamports: number; sol: string }>()
  let nextClaim = 100
  return {
    name: 'fly-dev-api',
    apply: 'serve',
    configureServer(server) {
      server.middlewares.use('/api', async (req, res) => {
        const url = new URL(req.url || '/', 'http://dev')
        const now = Math.floor(Date.now() / 1000)
        const { ref, stats } = load('stats.json') as { ref: number; stats: Record<string, unknown> }
        const delta = now - ref
        const path = url.pathname.replace(/\/+$/, '')

        if (req.method === 'GET' && path === '/stats') {
          const s = shift(stats, delta) as Record<string, any>
          s.settlement.next_at = mondayFloor(now) + WEEK
          if (network === 'testnet') {
            s.cluster = 'devnet'
            s.vault.chain_id = 46630
          }
          return send(res, 200, s)
        }

        if (req.method === 'GET' && path === '/history') {
          const kind = url.searchParams.get('kind') || ''
          if (!['nav', 'trades', 'flows', 'settlements'].includes(kind)) return send(res, 400, { error: 'bad kind' })
          const items = shift(load(`history_${kind}.json`) as unknown[], delta)
          const limit = Math.min(500, Math.max(1, Number(url.searchParams.get('limit') || 100)))
          const start = Number(url.searchParams.get('before') || 0)
          const page = items.slice(start, start + limit)
          return send(res, 200, { kind, items: page, next: start + limit < items.length ? String(start + limit) : null })
        }

        if (req.method === 'GET' && path === '/account') {
          const evm = url.searchParams.get('evm') || ''
          if (!/^0x[0-9a-fA-F]{40}$/.test(evm)) return send(res, 400, { error: 'bad evm address' })
          const acct = shift(load('account.json'), delta) as Record<string, any>
          acct.evm = evm
          for (const [id, c] of claims) {
            const st = claimState(now - c.created)
            acct.claims.push({ id, status: st, lamports: c.lamports, sol: c.sol, tx: st === 'paid' ? load('claim_status.json').tx : '', created_at: c.created, updated_at: now })
            if (st === 'paid') {
              acct.claimed += c.lamports
              acct.owed = 0
            } else {
              acct.in_flight = c.lamports
              acct.owed = 0
            }
          }
          return send(res, 200, acct)
        }

        if (req.method === 'GET' && path === '/claim/challenge') {
          const evm = url.searchParams.get('evm') || ''
          const sol = url.searchParams.get('sol') || ''
          if (!/^0x[0-9a-fA-F]{40}$/.test(evm) || !/^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(sol)) return send(res, 400, { error: 'bad address' })
          const host = req.headers.host || 'localhost:5173'
          return send(res, 200, {
            ...load('challenge.json'),
            nonce: randomBytes(16).toString('hex'),
            issued_at: rfc3339(now),
            expires_at: rfc3339(now + 900),
            domain: host,
            uri: `http://${host}/vault.html`,
            chain_id: network === 'testnet' ? 46630 : 4663,
            sol_chain: network === 'testnet' ? 'devnet' : 'mainnet',
          })
        }

        if (req.method === 'POST' && path === '/claim') {
          const body = JSON.parse((await readBody(req)) || '{}')
          if (!body.nonce || !body.evm_sig || !body.sol_sig) return send(res, 400, { error: 'missing fields' })
          const acct = load('account.json')
          const id = nextClaim++
          claims.set(id, { created: now, lamports: acct.owed, sol: body.sol })
          return send(res, 202, { id, status: 'received' })
        }

        const m = path.match(/^\/claim\/(\d+)$/)
        if (req.method === 'GET' && m) {
          const c = claims.get(Number(m[1]))
          if (!c) return send(res, 404, { error: 'unknown claim' })
          const status = claimState(now - c.created)
          const tpl = load('claim_status.json')
          return send(res, 200, {
            id: Number(m[1]),
            status,
            reason: null,
            lamports: c.lamports,
            tx: status === 'paid' ? tpl.tx : null,
            created_at: c.created,
            updated_at: now,
          })
        }

        return send(res, 404, { error: 'not found' })
      })
    },
  }
}

/** Walks a fake claim through the relay's statuses over ~12 s. */
function claimState(age: number): string {
  if (age < 3) return 'received'
  if (age < 6) return 'verified'
  if (age < 10) return 'sending'
  return 'paid'
}
