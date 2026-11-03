// Robinhood Chain: the Pons indexer, the quote assets' marks, the per-chain verdict and seat, and the ETH wallet's
// intents, transactions, legs and base lots (fly_trader/ui/api/rh.py). Rails (circuit 3) and limits live on the Safety page.
import type { ReactNode } from 'react'
import { PageHead } from '../components/ChainPicker'
import { BarChart } from '../components/Charts'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { Address, Callout, Caption, Card, Icon, Loaded, Metric, Metrics, RowGroup, Segmented, Widget, WidgetGrid, type WidgetColor } from '../components/ui'
import { useApp } from '../context'
import { ago, amount, dt, hm, int, num, pct, shortAddr, sig } from '../lib/format'
import { useNow, useStored, useTopic, type TopicState } from '../lib/hooks'

const OFF = 'Robinhood Chain is off (RH_ENABLED=0).'
type Off = { enabled: false; message: string }
const isOff = (d: { enabled: boolean }): d is Off => d.enabled === false
const off = (d: Off) => <Callout kind="info">{d.message || OFF}</Callout>

type Status = { mode: string; behind_blocks: number; head: number | null; index_through: number | null; minutes_through: string | null; error: string | null }
type IndexerData = Off | { enabled: true; updated_at?: string | null; status: Status | null }
type Hour = { hour: string; swaps: number; minutes: number; launches: number }
type ActivityIdx = Off | { enabled: true; counts: Record<string, number>; hours: Hour[] }
type AssetRow = { asset: string; address: string; class: string; eth_per_unit: number | null; native: boolean; marked_at: string | null; entries: string }
type AssetsData = Off | { enabled: true; rows: AssetRow[] }
type ChainVerdict = { chain: string; name: string; passed: boolean; evaluated: boolean; n: number; mean: number | null; reason: string | null }
type VerdictData = Off | { enabled: true; chains: ChainVerdict[]; replay_at: string | null; handover: { at: string | number | null } | null; live_money: boolean }
type LotRow = { asset: string; symbol: string | null; qty_raw: string; basis_eth: number; source: string; position_id: number | null; opened_at: string }
type WalletData = Off | {
  enabled: true; book: string; address: string | null; address_url: string | null; reconciled: boolean; marked_at: string | null
  native_eth: number | null; wealth: number | null; n_open: number; consistent: boolean; lots: LotRow[]; spark: number[]
}
const VIEWS = ['Intents', 'Transactions', 'Legs'] as const
type View = (typeof VIEWS)[number]
type ActivityData = Off | { enabled: true; view: View; rows: Record<string, unknown>[]; tx_url: string }

const STATE_PILL: Record<string, string> = {
  done: 'green', mined_ok: 'green', failed: 'red', reverted: 'red', aborted: 'gray', dropped: 'gray', replaced: 'orange',
  new: 'blue', leg1: 'blue', leg2: 'blue', signed: 'blue', sent: 'blue',
}

/** An address cut short, the whole of it on hover (and in the CSV). */
const Addr = ({ v }: { v: unknown }) => (v ? <span className="mono" title={String(v)}>{shortAddr(String(v))}</span> : null)
const pill = (v: unknown) => (v ? <Pill color={STATE_PILL[String(v)] ?? 'gray'}>{String(v)}</Pill> : null)

/** A widget slot fed by a topic that may still be loading or be off. */
function Slot<T extends { enabled: boolean }>({ s, label, icon, children }: { s: TopicState<T>; label: string; icon: string; children: (d: Exclude<T, Off>) => ReactNode }) {
  if (!s.data || isOff(s.data)) return <Widget label={label} value={s.error ? 'unavailable' : '…'} icon={icon} help={s.error} />
  return <>{children(s.data as Exclude<T, Off>)}</>
}

function Kpis({ idx, wallet }: { idx: TopicState<IndexerData>; wallet: TopicState<WalletData> }) {
  const now = useNow(5000)
  return (
    <WidgetGrid>
      <Slot s={idx} label="Indexer" icon="sensors">
        {(d) => {
          if (!d.status) return <Widget label="Indexer" value="not run" color="dark" icon="sensors" delta="start the Robinhood Chain feed on Processes" tone="off" />
          const age = d.updated_at ? (now - new Date(d.updated_at).getTime()) / 1000 : null
          const c: WidgetColor = d.status.error ? 'danger' : age !== null && age > 180 ? 'warning' : d.status.mode === 'live' ? 'success' : 'info'
          return <Widget label="Indexer mode" value={d.status.mode} color={c} icon="sensors" delta={`updated ${ago(d.updated_at, now)}`} tone="off" />
        }}
      </Slot>
      <Slot s={idx} label="Blocks behind" icon="layers">
        {(d) => <Widget label="Blocks behind" value={d.status ? int(d.status.behind_blocks) : '—'} icon="layers"
          help={d.status ? `head ${d.status.head ?? '—'}, indexed through ${d.status.index_through ?? '—'}` : null}
          delta={d.status?.index_through ? `through block ${int(d.status.index_through)}` : null} tone="off" />}
      </Slot>
      <Slot s={idx} label="Minutes through" icon="time">
        {(d) => <Widget label="Minutes through" value={d.status?.minutes_through ?? '—'} icon="time" help="The engine trades only minutes the indexer has written." />}
      </Slot>
      <Slot s={wallet} label="Books match chain" icon="shield">
        {(d) => <Widget label="Books match chain" value={d.consistent ? 'yes' : 'NO'} icon="shield" color={!d.reconciled ? 'dark' : d.consistent ? 'success' : 'danger'}
          delta={d.reconciled ? `checked ${ago(d.marked_at, now)}` : 'not reconciled yet'} tone="off"
          help="Token balances vs open positions and lots; native ETH vs every booked flow since the last check." />}
      </Slot>
    </WidgetGrid>
  )
}

function Indexer({ idx }: { idx: TopicState<IndexerData> }) {
  const a = useTopic<ActivityIdx>('rh.index_activity')
  return (
    <Card title="Indexer" sub="Pons launches, graduations and Uniswap v4 swaps from the chain into 1-minute rows in ETH" icon="sensors">
      {idx.data && !isOff(idx.data) && !idx.data.status && <Caption>The indexer has not run yet: start the Robinhood Chain feed on the Processes page.</Caption>}
      {idx.data && !isOff(idx.data) && idx.data.status?.error && <Callout kind="warning">Last error: {idx.data.status.error}</Callout>}
      <Loaded state={a} height={300}>
        {(d) => isOff(d) ? off(d) : (
          <>
            <BarChart height={270} categories={d.hours.map((h) => hm(h.hour))}
              series={[{ name: 'Swaps', data: d.hours.map((h) => h.swaps) }, { name: 'Minutes written', data: d.hours.map((h) => h.minutes) }]} />
            <Caption>Per hour, the last 24 hours (UTC); {int(d.hours.reduce((x, h) => x + h.launches, 0))} launches in that time.</Caption>
            <RowGroup items={Object.entries(d.counts).map(([k, v]) => ({ label: k, value: int(v) }))} />
          </>
        )}
      </Loaded>
    </Card>
  )
}

function Verdict() {
  const s = useTopic<VerdictData>('rh.verdict')
  const now = useNow(5000)
  return (
    <Card title="Verdict and seat" sub="one combined replay judges each chain on its own trades" icon="bx-badge-check">
      <Loaded state={s}>
        {(d) => isOff(d) ? off(d) : (
          <>
            {!d.chains.length ? (
              <Caption>No combined replay verdict yet: the RH books wait for one (the Robinhood corpus must be built and the models trained on both chains).</Caption>
            ) : (
              <>
                <Metrics min={150}>
                  {d.chains.map((c) => (
                    <Metric key={c.chain} label={c.name} value={c.passed ? 'passed' : 'failed'} tone="off" help={c.reason} color={c.passed ? 'success' : 'danger'}
                      icon={c.passed ? 'success' : 'error'} delta={c.evaluated ? `${c.n} trades · mean ${pct(c.mean)}` : null} />
                  ))}
                </Metrics>
                <Caption>Replay {ago(d.replay_at, now)}.</Caption>
              </>
            )}
            {d.handover ? (
              <Callout kind="success" icon="bx-transfer-alt">
                The RH fly holds the seat since {ago(d.handover.at, now)}.
                {d.live_money ? ' It trades the ETH wallet.' : ' Live trading waits for RH_LIVE_ENABLED=1, the key, the pinned address and a funded wallet.'}
              </Callout>
            ) : (
              <Callout kind="info" icon="bx-chair">The RH fly takes the seat when its realized P&amp;L over 14 days is at least the RH selector's, with at least 30 trades.</Callout>
            )}
          </>
        )}
      </Loaded>
    </Card>
  )
}

function Assets() {
  const s = useTopic<AssetsData>('rh.assets')
  const now = useNow(5000)
  const cols: Col<AssetRow>[] = [
    { key: 'asset', label: 'asset', render: (r) => <span title={r.address}><strong>{r.asset}</strong></span> },
    { key: 'class', label: 'class' },
    { key: 'eth_per_unit', label: 'ETH per unit', num: true, render: (r) => sig(r.eth_per_unit, 6) },
    { key: 'marked', label: 'marked', value: (r) => r.marked_at, render: (r) => (r.native ? 'native' : r.marked_at ? ago(r.marked_at, now) : 'never') },
    { key: 'entries', label: 'entries', render: (r) => <Pill color={r.entries === 'open' ? 'green' : 'orange'}>{r.entries}</Pill> },
  ]
  return (
    <Card title="Quote assets" sub="a coin quoted in one of these is bought ETH → base → coin in one transaction and sold back to ETH at once" icon="bx-transfer-alt">
      <Loaded state={s}>
        {(d) => {
          if (isOff(d)) return off(d)
          if (!d.rows.length) return <Caption>No approved quote assets indexed yet.</Caption>
          const open = d.rows.filter((r) => r.entries === 'open').length
          const classes = Object.entries(d.rows.reduce<Record<string, number>>((m, r) => ({ ...m, [r.class]: (m[r.class] ?? 0) + 1 }), {}))
          return (
            <>
              <RowGroup items={[{ label: 'entries open', value: int(open) }, { label: 'blocked: mark stale', value: int(d.rows.length - open) },
                ...classes.map(([k, v]) => ({ label: k, value: int(v) }))]} />
              <div className="mt-3"><DataTable rows={d.rows} cols={cols} filename="rh-quote-assets" /></div>
            </>
          )
        }}
      </Loaded>
    </Card>
  )
}

const LOT_COLS: Col<LotRow>[] = [
  { key: 'asset', label: 'asset', render: (r) => (r.symbol ? <span title={r.asset}>{r.symbol}</span> : <Addr v={r.asset} />) },
  { key: 'qty_raw', label: 'qty_raw', num: true },
  { key: 'basis_eth', label: 'basis_eth', num: true, render: (r) => num(r.basis_eth, 6) },
  { key: 'source', label: 'source', render: (r) => <Pill color="violet">{r.source}</Pill> },
  { key: 'position_id', label: 'position_id', num: true },
  { key: 'opened_at', label: 'opened_at', render: (r) => dt(r.opened_at, true) },
]

function Wallet({ s }: { s: TopicState<WalletData> }) {
  const now = useNow(5000)
  return (
    <Card title="ETH wallet" sub={s.data && !isOff(s.data) ? `book ${s.data.book}` : 'book live_rh'} icon="wallet">
      <Loaded state={s}>
        {(d) => isOff(d) ? off(d) : (
          <>
            {d.address ? <Address value={d.address} /> : <code className="addr">RH_BOT_ADDRESS is not set</code>}
            {d.address_url && <div className="caption"><a href={d.address_url} target="_blank" rel="noreferrer">On the explorer <Icon name="external" /></a></div>}
            <div className="mt-3">
              <Metrics min={150}>
                <Metric compact label="Native ETH" value={amount(d.native_eth, 'ETH', 6)} icon="wallet"
                  help={d.reconciled ? `at the last reconciliation, ${ago(d.marked_at, now)}` : 'not reconciled yet'} />
                <Metric compact label="Live wealth" value={amount(d.wealth, 'ETH', 6)} icon="wealth" spark={d.spark.length > 1 ? d.spark : null} />
                <Metric compact label="Open positions" value={int(d.n_open)} icon="open" />
              </Metrics>
            </div>
            <div className="mt-3 mb-2" style={{ color: '#fff' }}>Base lots (two-leg routes that stopped half way; sold back to ETH hourly)</div>
            <DataTable rows={d.lots} cols={LOT_COLS} filename="rh-base-lots" search={false} empty="No open base lots." />
          </>
        )}
      </Loaded>
    </Card>
  )
}

function Activity() {
  const { openInspector } = useApp()
  const [stored, setView] = useStored<View>('rh_activity', 'Intents')
  const view: View = VIEWS.includes(stored) ? stored : 'Intents'
  const s = useTopic<ActivityData>('rh.activity', { view })
  const txLink = (base: string, h: unknown): ReactNode =>
    h ? <a href={`${base}${String(h)}`} target="_blank" rel="noreferrer" className="mono" title={String(h)}>{shortAddr(String(h), 10, 6)}</a> : null
  const cols = (d: Exclude<ActivityData, Off>): Col<Record<string, unknown>>[] => {
    if (d.view === 'Transactions') return [
      { key: 'created_at', label: 'created_at', render: (r) => dt(r.created_at as string, true) },
      { key: 'kind', label: 'kind' },
      { key: 'nonce', label: 'nonce', num: true },
      { key: 'status', label: 'status', render: (r) => pill(r.status) },
      { key: 'gas_used', label: 'gas_used', num: true, render: (r) => int(r.gas_used as number | null) },
      { key: 'fee_eth', label: 'fee_eth', num: true, render: (r) => sig(r.fee_eth as number | null, 6) },
      { key: 'router', label: 'router', render: (r) => <Addr v={r.router} /> },
      { key: 'hash', label: 'hash', render: (r) => txLink(d.tx_url, r.hash) },
    ]
    if (d.view === 'Legs') return [
      { key: 'ts', label: 'ts', render: (r) => dt(r.ts as string, true) },
      { key: 'leg_no', label: 'leg_no', num: true },
      { key: 'asset_in', label: 'asset_in', render: (r) => <Addr v={r.asset_in} /> },
      { key: 'asset_out', label: 'asset_out', render: (r) => <Addr v={r.asset_out} /> },
      { key: 'amount_in', label: 'amount_in', num: true },
      { key: 'amount_out', label: 'amount_out', num: true },
      { key: 'eth_in', label: 'eth_in', num: true, render: (r) => num(r.eth_in as number | null, 6) },
      { key: 'eth_out', label: 'eth_out', num: true, render: (r) => num(r.eth_out as number | null, 6) },
      { key: 'verified_by', label: 'verified_by', render: (r) => <Pill color={r.verified_by === 'model' ? 'orange' : 'green'}>{String(r.verified_by)}</Pill> },
    ]
    return [
      { key: 'created_at', label: 'created_at', render: (r) => dt(r.created_at as string, true) },
      { key: 'kind', label: 'kind' },
      { key: 'token', label: 'token', render: (r) => r.mint
        ? <button type="button" className="link-btn" title={String(r.mint)} onClick={() => openInspector({ kind: 'token', chain: 'rh', mint: String(r.mint) })}>{String(r.token)}</button>
        : null },
      { key: 'route', label: 'route' },
      { key: 'state', label: 'state', render: (r) => pill(r.state) },
      { key: 'attempts', label: 'attempts', num: true },
      { key: 'error', label: 'error', wrap: true },
    ]
  }
  return (
    <Card title="Live activity" icon="receipt"
      actions={<><span className="caption m-0">Show</span><Segmented options={VIEWS} value={view} onChange={setView} /></>}>
      <Loaded state={s}>
        {(d) => isOff(d) ? off(d) : !d.rows.length ? <Caption>Nothing yet: the live RH book has not traded.</Caption>
          : <DataTable rows={d.rows} cols={cols(d)} filename={`rh-${d.view.toLowerCase()}`} />}
      </Loaded>
    </Card>
  )
}

function Page() {
  const idx = useTopic<IndexerData>('rh.indexer')
  const wallet = useTopic<WalletData>('rh.wallet')
  return (
    <>
      <Kpis idx={idx} wallet={wallet} />
      <div className="row g-4">
        <div className="col-12 col-xl-8"><Indexer idx={idx} /></div>
        <div className="col-12 col-xl-4"><Verdict /></div>
      </div>
      <div className="row g-4">
        <div className="col-12 col-xl-7"><Assets /></div>
        <div className="col-12 col-xl-5"><Wallet s={wallet} /></div>
      </div>
      <Activity />
    </>
  )
}

export default function RH() {
  const { meta } = useApp()
  return (
    <>
      <PageHead title="Robinhood Chain" icon="bx-link" chain={false} />
      {!meta ? <div className="skeleton" style={{ height: 160 }} /> : !meta.rh ? <Callout kind="info">{OFF}</Callout> : <Page />}
    </>
  )
}
