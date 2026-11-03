// The live fly ticker (fly_trader/ui/api/ticker.py): the newest buys, sells and picks not taken on the fly books, the
// bot wallets' live trades and fills, and the visual fly's Kalshi moves. TickerCard lists them (Overview); LiveToasts,
// mounted on every page, toasts each new live-money one.
import { useEffect, useRef, useState } from 'react'
import { useApp, type InspectTarget } from '../context'
import { Card, Empty, Icon, Loaded, RowGroup, Segmented, Tip } from '../components/ui'
import { useFeedback } from '../components/feedback'
import { useNow, useStored, useTopic } from '../lib/hooks'
import { ago, hm } from '../lib/format'

export type TickerItem = {
  id: string; ts: string; chain: 'sol' | 'rh' | 'kalshi'; book: string | null; kind: string; action: string; side: string | null
  mint: string | null; ticker: string | null; token: string; size: number | null; unit: string; price: number | null; live: boolean
  strategy: string | null; detail: string; text: string; signature?: string | null
}
type TickerData = { items: TickerItem[]; kalshi: boolean; chains: string[] }

const CHAIN_NAME: Record<string, string> = { sol: 'Solana', rh: 'RH', kalshi: 'Kalshi' }
const MAX_TOASTS = 4

export function targetOf(it: { chain: string; mint?: string | null; ticker?: string | null }): InspectTarget | null {
  if (it.ticker) return { kind: 'market', ticker: it.ticker }
  if (it.mint) return { kind: 'token', chain: it.chain, mint: it.mint }
  return null
}

const KIND_LOOK: Record<string, { icon: string; bg: string }> = {
  bought: { icon: 'bx-cart', bg: 'linear-gradient(135deg, #1fa34a, #127a35)' }, sold: { icon: 'bx-dollar-circle', bg: 'linear-gradient(135deg, #ff49c4, #d4168f)' },
  'not taken': { icon: 'bx-block', bg: 'rgb(255 255 255 / 14%)' }, fill: { icon: 'bx-receipt', bg: 'linear-gradient(135deg, #9a6bff, #6d3be0)' },
  posted: { icon: 'bx-time-five', bg: 'linear-gradient(135deg, #2f8fff, #1559d6)' }, filled: { icon: 'bx-check-double', bg: 'linear-gradient(135deg, #1fa34a, #127a35)' },
  settled: { icon: 'bx-flag', bg: 'linear-gradient(135deg, #9a6bff, #6d3be0)' },
}

function sizeText(it: TickerItem): string {
  if (it.size === null || it.size === undefined) return ''
  return it.unit === 'USD' ? `$${it.size.toFixed(2)}` : `${it.size.toFixed(it.unit === 'SOL' ? 4 : 5)} ${it.unit}`
}

function Row({ it, now }: { it: TickerItem; now: number }) {
  const { openInspector } = useApp()
  const t = targetOf(it)
  const look = KIND_LOOK[it.action] ?? KIND_LOOK['not taken']
  const verb = it.action === 'fill' ? `${it.side ?? ''} fill` : it.action
  return (
    <div className="d-flex align-items-center gap-3 p-2" style={{
      borderRadius: 10, border: '1px solid rgb(255 255 255 / 12%)', borderLeft: it.live ? '4px solid var(--fly-magenta)' : '1px solid rgb(255 255 255 / 12%)',
      background: it.live ? 'linear-gradient(90deg, rgb(255 47 185 / 24%), rgb(255 47 185 / 5%))' : 'rgb(255 255 255 / 4%)' }}>
      <div title={verb} style={{ width: 46, height: 46, borderRadius: 10, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 22, color: '#fff', background: look.bg }}>
        <Icon name={look.icon} />
      </div>
      <div style={{ minWidth: 0, flex: 1 }}>
        <h6 className="mb-0 d-flex align-items-center gap-2" style={{ fontSize: 14.5 }}>
          {t ? <button type="button" className="link-btn" style={{ fontWeight: 600, textDecoration: 'none' }} onClick={() => openInspector(t)} title="Open the fly decision inspector">{it.token}</button>
            : <strong>{it.token}</strong>}
          {it.live && <span className="fbadge magenta" style={{ padding: '0.2rem 0.5rem', fontSize: 11 }}><Icon name="payments" />LIVE</span>}
        </h6>
        <p className="mb-0" style={{ fontSize: 12.5, color: 'rgb(255 255 255 / 62%)', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }} title={it.text}>
          <span style={{ color: '#fff' }}>{CHAIN_NAME[it.chain] ?? it.chain} · {verb}</span>{it.strategy ? ` · ${it.strategy}` : ''} — {it.detail}
        </p>
      </div>
      <div className="text-end" style={{ flexShrink: 0 }}>
        <h6 className="mb-0 mono" style={{ fontSize: 13.5 }}>{sizeText(it) || '—'}</h6>
        <Tip text={`${new Date(it.ts).toUTCString()} · ${it.book ?? ''}`}><p className="mb-0" style={{ fontSize: 12, color: 'rgb(255 255 255 / 62%)' }}>{hm(it.ts)} · {ago(it.ts, now)}</p></Tip>
      </div>
    </div>
  )
}

/** The ticker card (Overview): newest first, live-money rows in magenta, filter by chain, pause to read. */
export function TickerCard() {
  const s = useTopic<TickerData>('ticker.recent')
  const [chain, setChain] = useStored<string>('ticker_chain', 'all')
  const [paused, setPaused] = useState<TickerItem[] | null>(null)
  const now = useNow(5000)
  const live = s.data?.items ?? []
  const shown = paused ?? live
  const fresh = paused ? live.filter((x) => !paused.some((p) => p.id === x.id)).length : 0
  const chains = s.data ? ['all', ...s.data.chains] : ['all']
  const pick = chains.includes(chain) ? chain : 'all'
  const rows = pick === 'all' ? shown : shown.filter((x) => x.chain === pick)
  const count = (f: (x: TickerItem) => boolean) => rows.filter(f).length
  return (
    <Card title="Live fly ticker" icon="pulse" sub="buys, sells and picks not taken as they happen; live-money rows in magenta"
      actions={<>
        {chains.length > 2 && <Segmented options={chains} value={pick} onChange={setChain} labels={{ all: 'All', ...CHAIN_NAME }} />}
        <button type="button" className={`btn btn-sm ${paused ? 'btn-primary' : 'btn-light'}`} onClick={() => setPaused(paused ? null : live)}
          title={paused ? 'Resume: show new events as they arrive' : 'Pause: hold the list still to read it'}>
          <Icon name={paused ? 'play' : 'pause'} />{paused ? `Resume${fresh ? ` (${fresh} new)` : ''}` : 'Pause'}
        </button>
      </>}>
      <Loaded state={s} height={260}>
        {() => rows.length ? (
          <>
            <div className="d-flex flex-column gap-2" style={{ maxHeight: 470, overflow: 'auto', paddingRight: 4 }}>
              {rows.map((it) => <Row key={it.id} it={it} now={now} />)}
            </div>
            <div className="mt-3" style={{ margin: '0 -1.1rem -1rem' }}>
              <RowGroup items={[
                { value: count((x) => x.action === 'bought' || x.action === 'filled'), label: 'bought' },
                { value: count((x) => x.action === 'sold' || x.action === 'settled'), label: 'sold / settled' },
                { value: count((x) => x.action === 'not taken'), label: 'not taken' },
                { value: count((x) => x.live), label: 'live money' },
              ]} />
            </div>
          </>
        ) : <Empty>{pick === 'all' ? 'The flies have not traded or passed on anything yet.' : `Nothing on ${CHAIN_NAME[pick] ?? pick} yet.`}</Empty>}
      </Loaded>
    </Card>
  )
}

/** Mounted once by the layout: toasts every new live-money item (a live buy, sell or fill). The first answer only marks
 *  what already happened as seen, so opening the console never replays old trades. */
export function LiveToasts() {
  const { data } = useTopic<TickerData>('ticker.recent')
  const { toast } = useFeedback()
  const { openInspector } = useApp()
  const seen = useRef<Set<string> | null>(null)
  useEffect(() => {
    if (!data) return
    if (seen.current === null) {
      seen.current = new Set(data.items.map((x) => x.id))
      return
    }
    const fresh = data.items.filter((x) => !seen.current!.has(x.id))
    for (const x of fresh) seen.current.add(x.id)
    if (seen.current.size > 5000) seen.current = new Set(data.items.map((x) => x.id))    // only ids still listed can come back
    const liveNew = fresh.filter((x) => x.live).reverse()           // oldest first, so the newest toast lands last
    for (const x of liveNew.slice(-MAX_TOASTS)) {
      const t = targetOf(x)
      toast(
        <span>
          <strong>Live {x.action === 'fill' ? `${x.side ?? ''} fill` : x.action}</strong>{' · '}
          {t ? <button type="button" className="link-btn" onClick={() => openInspector(t)}>{x.token}</button> : x.token}
          <br /><span style={{ color: 'var(--fly-muted)', fontSize: 12.5 }}>{CHAIN_NAME[x.chain] ?? x.chain} · {x.detail}</span>
        </span>, 'live', 15000)
    }
    if (liveNew.length > MAX_TOASTS) toast(`…and ${liveNew.length - MAX_TOASTS} more live trades: see the ticker on the Overview page.`, 'live', 15000)
  }, [data, toast, openInspector])
  return null
}
