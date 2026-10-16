// Trades: a book's results as a Dashtreme dashboard — KPI widgets, realized P&L cumulated and per day, trades by strategy
// and by exit — then the trade log (closed trades, the decision log of every buy, sell and pick not taken, fills) for
// either memecoin chain (topics in fly_trader/ui/api/trades.py).
import type { ApexOptions } from 'apexcharts'
import { useApp, type InspectTarget } from '../context'
import { useStored, useTopic } from '../lib/hooks'
import { amount, dt, int, num, pct, pctPts, sci, shortAddr } from '../lib/format'
import { Caption, Card, Empty, Help, KVList, Loaded, Pills, ProgressList, RowGroup, Segmented, Widget, WidgetGrid } from '../components/ui'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { ApexChart, DonutChart, LineChart, fmtAxis } from '../components/Charts'
import { PageHead } from '../components/ChainPicker'

const VIEWS = ['Closed trades', 'Decision log', 'Fills'] as const
type View = (typeof VIEWS)[number]
const SHOW = ['Bought', 'Sold', 'Not taken'] as const
type Show = (typeof SHOW)[number]
const ACTION_COLOR: Record<string, string> = { Bought: 'green', Sold: 'violet', 'Not taken': 'orange' }
const DONUT = ['#ffffff', 'rgba(255,255,255,0.6)', 'rgba(255,255,255,0.35)', 'rgba(255,255,255,0.18)', '#ff2fb9', '#8b5cf6', '#b6ff3b', '#3d9bff']

type Books = { chain: string; unit: string; books: string[]; default: string; help: string }
type Summary = {
  chain: string; book: string; unit: string
  metrics: { n: number; pnl: number | null; avg_ret: number | null; wins: number; win_rate: number | null; open: number; open_cost: number }
  cum: [string, number][]
  daily: { day: string; n: number; pnl: number; wins: number }[]
  exits: { kind: string; n: number; pnl: number }[]
  strategies: { strategy: string | null; n: number; pnl: number; wins: number }[]
}
type ClosedRow = {
  mint: string; token: string; opened: string; closed: string | null; held_min: number | null; cost: number; pnl: number; ret: number | null; exit: string
  strategy: string | null
}
type ClosedData = { chain: string; book: string; unit: string; rows: ClosedRow[] }
type DecisionRow = { ts: string; action: string; kind: string; mint: string | null; token: string; score: number | null; size: number; why: string }
type DecisionData = { chain: string; book: string; unit: string; show: Show[]; rows: DecisionRow[] }
type FillRow = {
  ts: string; side: string | null; mint: string | null; token: string; amount: number; price: number | null; fee: number
  source: string | null; signature: string | null; url: string | null
}
type FillData = { chain: string; book: string; unit: string; rows: FillRow[] }

/** The Streamlit page's number formats: ETH positions are ~20x smaller in number than SOL ones, so they get two more decimals. */
function formats(unit: string) {
  const eth = unit !== 'SOL'
  return { f3: (v: number | null) => num(v, eth ? 5 : 3), f4: (v: number | null) => num(v, eth ? 6 : 4, true) }
}

const signed = (v: number | null | undefined, text: string) => <span className={v == null || v === 0 ? '' : v > 0 ? 'text-pos' : 'text-neg'}>{text}</span>

/** A token cell that opens the fly inspector on that token. */
function tokenCol<R extends { mint: string | null; token: string }>(chain: string, open: (t: InspectTarget) => void): Col<R> {
  return {
    key: 'token', label: 'token',
    render: (r) => r.mint ? <button type="button" className="link-btn" onClick={() => open({ kind: 'token', chain, mint: r.mint as string })}>{r.token}</button> : r.token,
  }
}

export default function Trades() {
  const { chain, market } = useApp()
  const [rawView, setView] = useStored<View>('trades_view', 'Closed trades')
  const view: View = VIEWS.includes(rawView) ? rawView : 'Closed trades'
  const [stored, setBook] = useStored<string>('trades_book', '')
  const b = useTopic<Books>('trades.books', { chain })
  const books = b.data?.books ?? market?.books ?? []
  const fallback = b.data?.default ?? market?.selector_book ?? 'paper_selector'
  const book = books.includes(stored) ? stored : fallback       // the chain's selector book when the remembered one is another chain's
  const help = b.data?.help ?? (market ? `${market.selector_book} is the selector's paper book, ${market.fly_book} the fly's, ${market.live_book} the bot wallet; amounts in ${market.unit}.` : undefined)
  const s = useTopic<Summary>('trades.summary', { chain, book })
  return (
    <>
      <PageHead title="Trades" icon="bx-transfer">
        <label className="d-inline-flex align-items-center gap-2 m-0" htmlFor="trades-book">
          <span className="form-label m-0">Book</span><Help text={help} />
          <select id="trades-book" className="form-select form-select-sm" style={{ width: 200 }} value={book} onChange={(e) => setBook(e.target.value)}>
            {books.map((x) => <option key={x} value={x}>{x}</option>)}
          </select>
        </label>
      </PageHead>
      <Loaded state={s} height={130}>{(d) => <Dashboard d={d} />}</Loaded>
      <Card title="Trade log" icon="trades" sub={`${book} · the newest 300 of each`}
        actions={<Segmented options={VIEWS} value={view} onChange={setView} />}>
        {view === 'Closed trades' ? <ClosedView chain={chain} book={book} /> : view === 'Decision log' ? <DecisionView chain={chain} book={book} /> : <FillsView chain={chain} book={book} />}
      </Card>
    </>
  )
}

// ---------------------------------------------------------------- the book at a glance
function Dashboard({ d }: { d: Summary }) {
  const u = d.unit, m = d.metrics
  const losses = m.n - m.wins
  const best = d.daily.length ? d.daily.reduce((a, x) => (x.pnl > a.pnl ? x : a)) : null
  const worst = d.daily.length ? d.daily.reduce((a, x) => (x.pnl < a.pnl ? x : a)) : null
  const strat = d.strategies.filter((x) => x.n > 0)
  return (
    <>
      <WidgetGrid>
        <Widget label="Trades" value={int(m.n)} icon="count" spark={d.daily.map((x) => x.n)} sparkType="bar" delta={d.daily.length ? `${d.daily.length} trading days` : null} tone="off"
          help="Closed positions in this book; the bars are trades closed per day (last 90 days)." />
        <Widget label="Realized P&L" value={amount(m.pnl, u, 4, true)} icon="pnl" spark={d.cum.map((x) => x[1])}
          color={m.pnl == null || m.pnl === 0 ? null : m.pnl > 0 ? 'success' : 'danger'} help="Realized P&L of every closed trade; the line is its running total." />
        <Widget label="Average return" value={pct(m.avg_ret)} icon="up" help="Per trade, after fees and price impact."
          color={m.avg_ret == null || m.avg_ret === 0 ? null : m.avg_ret > 0 ? 'teal' : 'danger'} />
        <Widget label="Winners" value={m.win_rate === null ? '—' : pct(m.win_rate, 0, false)} icon="target" progress={m.win_rate}
          delta={m.n ? `${int(m.wins)} won · ${int(losses)} lost` : null} tone="off" help="Share of closed trades with a positive realized P&L." />
        <Widget label="Open positions" value={int(m.open)} icon="open" delta={m.open ? `${amount(m.open_cost, u, 4)} at cost` : 'none open'} tone="off"
          color={m.open ? 'info' : 'dark'} help="Positions this book holds right now (not in the trades above until they close)." />
      </WidgetGrid>
      <div className="row g-4">
        <div className="col-12 col-xl-8">
          <Card title="Cumulative realized P&L" icon="chart" sub={`${d.book} · ${u}`}>
            {d.cum.length > 1
              ? <LineChart series={[{ name: `P&L (${u})`, data: d.cum }]} height={280} yFormat={(v) => fmtAxis(v)} />
              : <Empty>No closed trades in this book yet.</Empty>}
            <RowGroup items={[
              { label: 'realized P&L', value: signed(m.pnl, amount(m.pnl, u, 4, true)) },
              { label: 'average per trade', value: signed(m.avg_ret, pct(m.avg_ret)) },
              { label: 'best day', value: best ? signed(best.pnl, `${amount(best.pnl, u, 4, true)} · ${best.day.slice(5)}`) : '—' },
              { label: 'worst day', value: worst ? signed(worst.pnl, `${amount(worst.pnl, u, 4, true)} · ${worst.day.slice(5)}`) : '—' },
            ]} />
          </Card>
        </div>
        <div className="col-12 col-xl-4">
          <Card title="By strategy" icon="percent" sub="closed trades per strategy">
            {strat.length ? (
              <>
                <DonutChart labels={strat.map((x) => x.strategy ?? 'unlabelled')} values={strat.map((x) => x.n)} center={int(m.n)} centerLabel="trades" height={240} colors={DONUT} />
                <KVList items={strat.map((x, i) => ({
                  dot: DONUT[i % DONUT.length], label: x.strategy ?? 'unlabelled',
                  value: <>{int(x.n)} · {pct(x.n ? x.wins / x.n : null, 0, false)} won · {signed(x.pnl, amount(x.pnl, u, 4, true))}</>,
                }))} />
              </>
            ) : <Empty>No closed trades in this book yet.</Empty>}
          </Card>
        </div>
      </div>
      <div className="row g-4">
        <div className="col-12 col-xl-8">
          <Card title="Realized P&L per day" icon="calendar" sub={`last ${d.daily.length || 0} trading days · ${u}`}>
            {d.daily.length ? <DailyBars daily={d.daily} unit={u} /> : <Empty>No closed trades in this book yet.</Empty>}
          </Card>
        </div>
        <div className="col-12 col-xl-4">
          <Card title="Exits" icon="bx-log-out-circle" sub="how the closed trades ended (time = held to the end of their hold)">
            {d.exits.length ? (
              <ProgressList items={d.exits.map((e) => ({ label: e.kind, value: <>{int(e.n)} · {signed(e.pnl, amount(e.pnl, u, 4, true))}</>, frac: e.n }))} />
            ) : <Empty>No closed trades in this book yet.</Empty>}
            {m.n > 0 && (
              <div className="mt-3">
                <ProgressList max={m.n} items={[
                  { label: 'winners', value: int(m.wins), frac: m.wins },
                  { label: 'losers or flat', value: int(losses), frac: losses },
                ]} />
              </div>
            )}
          </Card>
        </div>
      </div>
    </>
  )
}

function DailyBars({ daily, unit }: { daily: Summary['daily']; unit: string }) {
  const options: ApexOptions = {
    chart: { type: 'bar', height: 260, background: 'transparent', foreColor: 'rgba(255,255,255,0.75)', toolbar: { show: false }, zoom: { enabled: false }, animations: { enabled: false },
      parentHeightOffset: 0, fontFamily: 'Inter Variable, Inter, Roboto, system-ui, sans-serif' },
    theme: { mode: 'dark' },
    plotOptions: { bar: { columnWidth: '55%', borderRadius: 3, colors: { ranges: [{ from: -1e12, to: -1e-12, color: '#ff4d6d' }, { from: 0, to: 1e12, color: '#ffffff' }] } } },
    dataLabels: { enabled: false },
    grid: { borderColor: 'rgba(221,221,221,0.08)', padding: { left: 8, right: 12, top: 0 } },
    xaxis: { categories: daily.map((x) => x.day.slice(5)), labels: { rotate: -45, hideOverlappingLabels: true, style: { fontSize: '10.5px' } }, axisBorder: { show: false }, axisTicks: { show: false } },
    yaxis: { labels: { formatter: (v: number) => fmtAxis(v) } },
    tooltip: { theme: 'dark', y: { formatter: (v: number, o?: { dataPointIndex: number }) => {
      const x = o ? daily[o.dataPointIndex] : undefined
      return `${amount(v, unit, 4, true)}${x ? ` · ${x.n} trades, ${x.wins} won` : ''}`
    } } },
    legend: { show: false },
  }
  return <ApexChart options={options} series={[{ name: `P&L (${unit})`, data: daily.map((x) => x.pnl) }]} height={260} />
}

// ---------------------------------------------------------------- the trade log
function ClosedView({ chain, book }: { chain: string; book: string }) {
  const { openInspector } = useApp()
  const s = useTopic<ClosedData>('trades.closed', { chain, book })
  return (
    <Loaded state={s}>
      {(d) => {
        const u = d.unit, f = formats(u)
        const cols: Col<ClosedRow>[] = [
          tokenCol<ClosedRow>(d.chain, openInspector),
          { key: 'opened', label: 'opened', render: (r) => dt(r.opened) },
          { key: 'closed', label: 'closed', render: (r) => dt(r.closed) },
          { key: 'held_min', label: 'held (min)', num: true, render: (r) => num(r.held_min, 0) },
          { key: 'strategy', label: 'strategy' },
          { key: 'cost', label: `cost (${u})`, num: true, render: (r) => f.f3(r.cost) },
          { key: 'pnl', label: `P&L (${u})`, num: true, render: (r) => signed(r.pnl, f.f4(r.pnl)) },
          { key: 'ret', label: 'return', num: true, render: (r) => signed(r.ret, pctPts(r.ret, 2)) },
          { key: 'exit', label: 'exit', render: (r) => <Pill color={r.exit === 'time' ? 'gray' : 'orange'}>{r.exit}</Pill> },
        ]
        return d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`closed-trades-${d.book}`} /> : <Caption>No closed trades in this book yet.</Caption>
      }}
    </Loaded>
  )
}

function DecisionView({ chain, book }: { chain: string; book: string }) {
  const { openInspector } = useApp()
  const [rawShow, setShow] = useStored<Show[]>('dec_kinds', [...SHOW])
  const show = SHOW.filter((x) => Array.isArray(rawShow) && rawShow.includes(x))      // a stable order, so one selection is one subscription
  const s = useTopic<DecisionData>('trades.decisions', { chain, book, show })
  return (
    <>
      <div className="d-flex align-items-center flex-wrap gap-2 mb-3">
        <span className="form-label m-0">Show</span>
        <Pills options={SHOW} value={show} onChange={setShow} />
        {s.data && s.data.rows.length > 0 && (
          <span className="ms-auto d-inline-flex gap-2 flex-wrap">
            {SHOW.filter((k) => show.includes(k)).map((k) => <Pill key={k} color={ACTION_COLOR[k]}>{k} · {s.data!.rows.filter((r) => r.action === k).length}</Pill>)}
          </span>
        )}
      </div>
      <Loaded state={s}>
        {(d) => {
          const u = d.unit, f = formats(u)
          const cols: Col<DecisionRow>[] = [
            { key: 'ts', label: 'time', render: (r) => dt(r.ts, true) },
            { key: 'action', label: 'action', render: (r) => <Pill color={ACTION_COLOR[r.action] ?? 'gray'}>{r.action}</Pill> },
            tokenCol<DecisionRow>(d.chain, openInspector),
            { key: 'score', label: 'score', num: true, render: (r) => num(r.score, 3) },
            { key: 'size', label: `size (${u})`, num: true, render: (r) => f.f3(r.size) },
            { key: 'why', label: 'why', wrap: true },
          ]
          return d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`decisions-${d.book}`} /> : <Caption>No decisions recorded yet.</Caption>
        }}
      </Loaded>
    </>
  )
}

function FillsView({ chain, book }: { chain: string; book: string }) {
  const { openInspector } = useApp()
  const s = useTopic<FillData>('trades.fills', { chain, book })
  return (
    <Loaded state={s}>
      {(d) => {
        const u = d.unit, f = formats(u)
        const cols: Col<FillRow>[] = [
          { key: 'ts', label: 'time', render: (r) => dt(r.ts, true) },
          { key: 'side', label: 'side', render: (r) => r.side ? <Pill color={r.side === 'buy' ? 'green' : r.side === 'sell' ? 'magenta' : 'gray'}>{r.side}</Pill> : null },
          tokenCol<FillRow>(d.chain, openInspector),
          { key: 'amount', label: u, num: true, render: (r) => signed(r.amount, f.f4(r.amount)) },
          { key: 'price', label: `price (${u})`, num: true, render: (r) => sci(r.price, 3) },
          { key: 'fee', label: `fee (${u})`, num: true, render: (r) => num(r.fee, 5) },
          { key: 'source', label: 'source', render: (r) => r.source ? <Pill color={r.source === 'simulated' ? 'gray' : 'blue'}>{r.source}</Pill> : null },
          {
            key: 'signature', label: 'signature',
            render: (r) => r.signature && r.url
              ? <a href={r.url} target="_blank" rel="noreferrer" title={r.signature} className="mono">{shortAddr(r.signature, 10, 6)}</a>
              : r.signature,
          },
        ]
        return d.rows.length ? <DataTable rows={d.rows} cols={cols} filename={`fills-${d.book}`} /> : <Caption>No fills in this book yet.</Caption>
      }}
    </Loaded>
  )
}
