// Kalshi trades: the book's settled results as KPI widgets, its cumulative realized P&L beside the split by strategy, then
// settled positions, resting and recent orders, exchange fills and the decision log as DataTables
// (fly_trader/ui/api/kalshi_trades.py).
import type { ReactNode } from 'react'
import { PageHead } from '../components/ChainPicker'
import { DonutChart, LineChart } from '../components/Charts'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { Caption, Card, Help, Icon, KVList, Loaded, Pills, Segmented, Widget, WidgetGrid } from '../components/ui'
import { useApp } from '../context'
import { dt, int, pct, tone, usd } from '../lib/format'
import { useStored, useTopic } from '../lib/hooks'
import { KalshiDormant } from './KalshiOverview'

type DormantData = { dormant: true }
const isDormant = (d: unknown): d is DormantData => !!d && (d as DormantData).dormant === true

const VIEWS = ['Settled positions', 'Orders', 'Fills', 'Decision log'] as const
type View = (typeof VIEWS)[number]
const VIEW_ICON: Record<View, string> = { 'Settled positions': 'success', Orders: 'bx-list-ul', Fills: 'bx-check-double', 'Decision log': 'bx-list-check' }
const KINDS = ['Taker bought', 'Maker posted', 'Maker filled', 'Maker cancelled', 'Settled', 'Exits', 'Not taken'] as const
type Kind = (typeof KINDS)[number]
const DEFAULT_KINDS: Kind[] = ['Taker bought', 'Maker posted', 'Maker filled', 'Settled']
const KIND_COLOR: Record<string, string> = { 'Taker bought': 'green', 'Maker posted': 'blue', 'Maker filled': 'magenta', 'Maker cancelled': 'gray', Settled: 'violet',
  Exits: 'orange', 'Not taken': 'red' }
const ORDER_COLOR: Record<string, string> = { resting: 'blue', executed: 'green', filled: 'green', canceled: 'gray', cancelled: 'gray', expired: 'gray', error: 'red', rejected: 'red' }
const BOOK_HELP = "Paper books simulate at the quoted book with real fees; live books are the subaccount's own orders and fills."

type SettledRow = {
  ticker: string; market: string; side: string; strategy: string; contracts: number; paid: number; fee: number; result: string; payout: number; pnl: number
  ret: number | null; opened: string; settled: string | null
}
type ByStrategy = { strategy: string; n: number; pnl: number; wins: number }
type SettledData = {
  n: number; pnl: number; avg_ret: number | null; wins: number; fees: number; rows: SettledRow[]; cum: [string, number][]; by_strategy: ByStrategy[]
}
type OrderRow = {
  time: string; status: string; ticker: string; market: string; side: string; price: number; count: number; filled: number; type: string; expires: string | null
  strategy: string; order_id: string; error: string
}
type OrdersData = { resting: number; rows: OrderRow[] }
type FillRow = { time: string; ticker: string; side: string; action: string | null; price: number; count: number; fee: number; taker: boolean; order: string | null; trade: string }
type FillsData = { rows: FillRow[] }
type DecisionRow = { time: string; action: string; market: string | null; side: string | null; edge: number | null; dollars: number; why: string }
type DecisionsData = { rows: DecisionRow[] }

const plain = (v: number | null | undefined) => (v === null || v === undefined ? null : Number(v.toFixed(4)).toLocaleString('en-US', { maximumFractionDigits: 4 }))
const fixed = (v: number | null | undefined, nd: number, signed = false) => (v === null || v === undefined ? null : `${signed && v >= 0 ? '+' : ''}${v.toFixed(nd)}`)
const signedCell = (v: number | null | undefined, nd: number, suffix = '') => {
  const s = fixed(v, nd, true)
  return s === null ? null : <span className={`text-${tone(v) || 'muted2'}`}>{s}{suffix}</span>
}
const SidePill = ({ side }: { side: string | null }) => (side ? <Pill color={side === 'yes' ? 'green' : side === 'no' ? 'red' : 'gray'}>{side}</Pill> : null)

export default function KalshiTrades() {
  const { meta } = useApp()
  const [storedView, setView] = useStored<View>('ktrades_view', 'Settled positions')
  const [storedBook, setBook] = useStored<string>('ktrades_book', 'paper_kalshi_taker')
  const books = meta?.kalshi_books ?? {}
  const view: View = VIEWS.includes(storedView) ? storedView : 'Settled positions'
  const book = storedBook in books || !meta ? storedBook : Object.keys(books)[0] ?? 'paper_kalshi_taker'
  const off = !!meta && !meta.kalshi
  return (
    <>
      <PageHead title="Kalshi trades" icon="bx-receipt" chain={false} status={!off}>
        {!off && (
          <span className="d-inline-flex align-items-center gap-2">
            <label className="form-label mb-0" htmlFor="ktrades-book">Book<Help text={BOOK_HELP} /></label>
            <select id="ktrades-book" className="form-select form-select-sm" style={{ width: 200 }} value={book} onChange={(e) => setBook(e.target.value)}>
              {Object.entries(books).map(([k, label]) => <option key={k} value={k}>{label}</option>)}
            </select>
          </span>
        )}
      </PageHead>
      {off ? <KalshiDormant /> : (
        <>
          <Results book={book} label={books[book]} />
          <Card title={view} icon={VIEW_ICON[view]} sub={books[book]} actions={<Segmented options={VIEWS} value={view} onChange={setView} />}>
            {view === 'Settled positions' ? <Settled book={book} />
              : view === 'Orders' ? <Orders book={book} />
                : view === 'Fills' ? <Fills book={book} />
                  : <Decisions book={book} />}
          </Card>
        </>
      )}
    </>
  )
}

function MarketLink({ ticker, text }: { ticker: string | null; text?: ReactNode }) {
  const { openInspector } = useApp()
  if (!ticker) return null
  return <button type="button" className="link-btn" title={ticker} onClick={() => openInspector({ kind: 'market', ticker })}>{text ?? ticker}</button>
}

/** The book's settled results: KPI widgets, the cumulative realized P&L and the split by strategy. */
function Results({ book, label }: { book: string; label?: string }) {
  const s = useTopic<SettledData | DormantData>('kalshi_trades.settled', { book })
  return (
    <Loaded state={s} height={130}>
      {(d) => isDormant(d) ? <KalshiDormant /> : (
        <>
          <WidgetGrid min={190}>
            <Widget label="Positions" icon="count" value={int(d.n)} delta={label ? `settled in ${label}` : null} tone="off" />
            <Widget label="Realized P&L" icon="pnl" value={usd(d.pnl, 2, true)} spark={d.cum.length > 1 ? d.cum.map(([, v]) => v) : null} />
            <Widget label="Average return" icon="percent" value={pct(d.avg_ret)} help="Per position, on the dollars put down (cost plus fee)." />
            <Widget label="Winners" icon="target" value={d.n ? `${((d.wins / d.n) * 100).toFixed(0)}%` : '—'} delta={d.n ? `${int(d.wins)} of ${int(d.n)}` : null} tone="off"
              progress={d.n ? d.wins / d.n : null} />
            <Widget label="Fees paid" icon="bx-receipt" value={usd(d.fees)} />
          </WidgetGrid>
          <div className="row g-4">
            <div className="col-12 col-xl-8">
              <Card title="Realized P&L" icon="chart" sub="Cumulative, over every settled position in this book (USD)">
                {d.cum.length > 1 ? <LineChart series={[{ name: 'realized P&L', data: d.cum }]} height={260} yLabel="USD" yFormat={(v) => usd(v)} />
                  : <Caption>No settled positions in this book yet.</Caption>}
              </Card>
            </div>
            <div className="col-12 col-xl-4">
              <Card title="By strategy" icon="percent" sub="Settled positions per strategy, with their P&L and winners">
                {!d.by_strategy.length ? <Caption>No settled positions in this book yet.</Caption> : (
                  <>
                    <DonutChart labels={d.by_strategy.map((x) => x.strategy)} values={d.by_strategy.map((x) => x.n)} center={int(d.n)} centerLabel="settled" height={220} />
                    <KVList items={d.by_strategy.map((x, i) => ({
                      label: x.strategy, dot: ['#ffffff', 'rgba(255,255,255,0.6)', 'rgba(255,255,255,0.35)', 'rgba(255,255,255,0.18)', '#ff2fb9', '#8b5cf6', '#b6ff3b', '#3d9bff'][i % 8],
                      value: <>{int(x.n)} · <span className={`text-${tone(x.pnl) || 'muted2'}`}>{usd(x.pnl, 2, true)}</span> · {x.n ? `${((x.wins / x.n) * 100).toFixed(0)}% won` : '—'}</>,
                    }))} />
                  </>
                )}
              </Card>
            </div>
          </div>
        </>
      )}
    </Loaded>
  )
}

function Settled({ book }: { book: string }) {
  const s = useTopic<SettledData | DormantData>('kalshi_trades.settled', { book })
  const cols: Col<SettledRow>[] = [
    { key: 'market', label: 'market', render: (r) => <MarketLink ticker={r.ticker} text={r.market} /> },
    { key: 'side', label: 'side', render: (r) => <SidePill side={r.side} /> },
    { key: 'strategy', label: 'strategy' },
    { key: 'contracts', label: 'contracts', num: true, render: (r) => plain(r.contracts) },
    { key: 'paid', label: 'paid (c)', num: true, render: (r) => fixed(r.paid, 1) },
    { key: 'fee', label: 'fee ($)', num: true, render: (r) => fixed(r.fee, 2) },
    { key: 'result', label: 'result', render: (r) => <Pill color={r.pnl > 0 ? 'green' : r.pnl < 0 ? 'red' : 'gray'}>{r.result}</Pill> },
    { key: 'payout', label: 'payout ($)', num: true, render: (r) => fixed(r.payout, 2) },
    { key: 'pnl', label: 'P&L ($)', num: true, render: (r) => signedCell(r.pnl, 2) },
    { key: 'ret', label: 'return', num: true, render: (r) => signedCell(r.ret, 1, '%') },
    { key: 'opened', label: 'opened', render: (r) => dt(r.opened) },
    { key: 'settled', label: 'settled', render: (r) => dt(r.settled) },
  ]
  return (
    <Loaded state={s}>
      {(d) => isDormant(d) ? <KalshiDormant /> : d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`kalshi-settled-${book}`} />
        : <Caption>No settled positions in this book yet.</Caption>}
    </Loaded>
  )
}

function Orders({ book }: { book: string }) {
  const s = useTopic<OrdersData | DormantData>('kalshi_trades.orders', { book })
  const cols: Col<OrderRow>[] = [
    { key: 'time', label: 'time', render: (r) => dt(r.time, true) },
    { key: 'status', label: 'status', render: (r) => <Pill color={ORDER_COLOR[r.status] ?? 'gray'}>{r.status}</Pill> },
    { key: 'market', label: 'market', render: (r) => <MarketLink ticker={r.ticker} text={r.market} /> },
    { key: 'side', label: 'side', render: (r) => <SidePill side={r.side} /> },
    { key: 'price', label: 'price (c)', num: true },
    { key: 'count', label: 'count', num: true, render: (r) => plain(r.count) },
    { key: 'filled', label: 'filled', num: true, render: (r) => plain(r.filled) },
    { key: 'type', label: 'type' },
    { key: 'expires', label: 'expires', render: (r) => dt(r.expires) },
    { key: 'strategy', label: 'strategy' },
    { key: 'order_id', label: 'exchange id' },
    { key: 'error', label: 'error', wrap: true, render: (r) => (r.error ? <span className="text-neg">{r.error}</span> : null) },
  ]
  return (
    <Loaded state={s}>
      {(d) => isDormant(d) ? <KalshiDormant /> : (
        <>
          <Caption>
            <Pill color="blue">{d.resting} resting</Pill>{' '}the maker arm rests a post-only bid one tick inside the ask and cancels or replaces it when the ask moves; a taker
            order is an IOC at the walk's worst level.
          </Caption>
          {d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`kalshi-orders-${book}`} /> : <Caption>No orders in this book yet.</Caption>}
        </>
      )}
    </Loaded>
  )
}

function Fills({ book }: { book: string }) {
  const s = useTopic<FillsData | DormantData>('kalshi_trades.fills', { book })
  const cols: Col<FillRow>[] = [
    { key: 'time', label: 'time', render: (r) => dt(r.time, true) },
    { key: 'ticker', label: 'market', render: (r) => <MarketLink ticker={r.ticker} /> },
    { key: 'side', label: 'side', render: (r) => <SidePill side={r.side} /> },
    { key: 'action', label: 'action' },
    { key: 'price', label: 'price (c)', num: true, render: (r) => plain(r.price) },
    { key: 'count', label: 'count', num: true, render: (r) => plain(r.count) },
    { key: 'fee', label: 'fee ($)', num: true, render: (r) => plain(r.fee) },
    { key: 'taker', label: 'taker', render: (r) => <Pill color={r.taker ? 'orange' : 'violet'}>{r.taker ? 'taker' : 'maker'}</Pill> },
    { key: 'order', label: 'order' },
    { key: 'trade', label: 'trade' },
  ]
  return (
    <Loaded state={s}>
      {(d) => isDormant(d) ? <KalshiDormant /> : d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`kalshi-fills-${book}`} />
        : <Caption>No exchange fills for this book (paper books have none: their fills are the positions themselves).</Caption>}
    </Loaded>
  )
}

function Decisions({ book }: { book: string }) {
  const [stored, setKinds] = useStored<Kind[]>('kdec_kinds', DEFAULT_KINDS)
  const kinds = (Array.isArray(stored) ? stored : DEFAULT_KINDS).filter((k) => KINDS.includes(k))
  const s = useTopic<DecisionsData | DormantData>('kalshi_trades.decisions', { book, kinds })
  const cols: Col<DecisionRow>[] = [
    { key: 'time', label: 'time', render: (r) => dt(r.time, true) },
    { key: 'action', label: 'action', render: (r) => <Pill color={KIND_COLOR[r.action] ?? 'gray'}>{r.action}</Pill> },
    { key: 'market', label: 'market', render: (r) => <MarketLink ticker={r.market} /> },
    { key: 'side', label: 'side', render: (r) => <SidePill side={r.side} /> },
    { key: 'edge', label: 'edge', num: true, render: (r) => fixed(r.edge, 3) },
    { key: 'dollars', label: 'dollars', num: true, render: (r) => fixed(r.dollars, 2) },
    { key: 'why', label: 'why', wrap: true },
  ]
  return (
    <>
      <div className="d-flex align-items-center gap-2 flex-wrap mb-3">
        <span className="form-label mb-0"><Icon name="bx-filter-alt" className="me-1" />Show</span>
        <Pills options={KINDS} value={kinds} onChange={setKinds} />
      </div>
      <Loaded state={s}>
        {(d) => isDormant(d) ? <KalshiDormant /> : d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`kalshi-decisions-${book}`} />
          : <Caption>No decisions recorded for this book yet.</Caption>}
      </Loaded>
    </>
  )
}
