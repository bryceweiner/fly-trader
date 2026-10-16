// App-wide state: what this install has turned on (the `meta` topic), the memecoin chain the pages show (shared, like
// Streamlit's chain picker), and the fly inspector drawer (opened from any token, market or search result).
import { createContext, useCallback, useContext, useMemo, type ReactNode } from 'react'
import { useSearchParams } from 'react-router'
import { useStored, useTopic } from './lib/hooks'

export type Market = { chain: 'sol' | 'rh'; name: string; unit: string; selector_book: string; fly_book: string; live_book: string; books: string[]; capital: number }
export type Meta = {
  pid: number
  hosts_workers: boolean
  kalshi: boolean
  rh: boolean
  vault: boolean
  markets: Market[]
  kalshi_books: Record<string, string>
  version: string
}

/** What the inspector opens: a memecoin token on a chain, or a Kalshi market ticker. */
export type InspectTarget = { kind: 'token'; chain: string; mint: string } | { kind: 'market'; ticker: string }

type App = {
  meta: Meta | undefined
  chain: Market['chain']
  setChain: (c: Market['chain']) => void
  market: Market | undefined
  inspect: InspectTarget | null
  openInspector: (t: InspectTarget) => void
  closeInspector: () => void
}

const Ctx = createContext<App | null>(null)

export function useApp(): App {
  const v = useContext(Ctx)
  if (!v) throw new Error('useApp outside AppProvider')
  return v
}

export function encodeTarget(t: InspectTarget): string {
  return t.kind === 'token' ? `${t.chain}:${t.mint}` : `kalshi:${t.ticker}`
}

export function decodeTarget(s: string | null): InspectTarget | null {
  if (!s) return null
  const i = s.indexOf(':')
  if (i < 0) return null
  const head = s.slice(0, i), rest = s.slice(i + 1)
  if (!rest) return null
  return head === 'kalshi' ? { kind: 'market', ticker: rest } : { kind: 'token', chain: head, mint: rest }
}

export function AppProvider({ children }: { children: ReactNode }) {
  const { data: meta } = useTopic<Meta>('meta')
  const [stored, setStored] = useStored<Market['chain']>('chain', 'sol')
  const [params, setParams] = useSearchParams()
  const chain: Market['chain'] = meta?.markets.some((m) => m.chain === stored) ? stored : 'sol'
  const market = meta?.markets.find((m) => m.chain === chain) ?? meta?.markets[0]
  const inspectRaw = params.get('inspect')
  const inspect = useMemo(() => decodeTarget(inspectRaw), [inspectRaw])
  const openInspector = useCallback((t: InspectTarget) => {
    setParams((p) => { const n = new URLSearchParams(p); n.set('inspect', encodeTarget(t)); return n })
  }, [setParams])
  const closeInspector = useCallback(() => {
    setParams((p) => { const n = new URLSearchParams(p); n.delete('inspect'); return n })
  }, [setParams])
  const value = useMemo(() => ({ meta, chain, setChain: setStored, market, inspect, openInspector, closeInspector }),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [meta, chain, market, inspect, openInspector, closeInspector])
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>
}
