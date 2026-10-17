// Overview, laid out like Dashtreme's index: the selector book's KPI widgets, what the system is doing in plain words
// beside the win rate, the fly's brain, this minute's scoring with the top scores, the race between the selector and the
// plastic fly, the live ticker, and the paper book with its open positions (ui/api/overview.py; the brain ui/api/brain.py).
import { BrainPanel } from '../brain'
import { PageHead } from '../components/ChainPicker'
import { LineChart, RadialGauge, type Series } from '../components/Charts'
import { DataTable, type Col } from '../components/DataTable'
import { HourWheel, hourColor, type HourBucket } from '../components/HourWheel'
import { Callout, Caption, Card, Loaded, Md, Metric, Metrics, ProgressList, RowGroup, Segmented, Widget, WidgetGrid } from '../components/ui'
import { useApp, type Market } from '../context'
import { TickerCard } from '../fly/Ticker'
import { ago, amount, int, num, pct, pctPts, tone } from '../lib/format'
import { useNow, useStored, useTopic } from '../lib/hooks'

type Summary = {
  market: Market
  narrative: string
  marked: boolean
  wealth: number | null
  exposure: number | null
  n_open: number
  drawdown: number | null
  capital: number
  closed: number
  pnl: number | null
  wins: number
  spark: number[]
  pnl_spark: number[]
  dd_spark: number[]
  wealth_help: string
  live: boolean
  label: string
}
type Top = { mint: string; token: string; score: number }
type Minute = {
  unit: string
  runner: boolean
  trading: string
  trading_why: string
  scored_at: string | null
  minute: string | null
  minute_hm: string | null
  mints_traded: number
  eligible: number
  score_max: number
  threshold: number
  picks: number
  entered: number
  exited: number
  latency_s: number | null
  top: Top[]
  eligible_help: string
}
type RaceBook = { book: string; title: string; pnl: number | null; n: number; daily: number[] }
type Race = { hidden: boolean; unit: string; handover?: boolean; success?: string | null; caption?: string | null; metrics?: RaceBook[]; series?: Series[]; days?: string[] }
type Position = { mint: string; token: string; strategy: string; held_min: number; exits_in_min: number; cost: number; ret: number | null }
type Book = { unit: string; history: [string, number][]; positions: Position[]; live?: boolean; book?: string; title?: string; sub?: string }

/** Python's %g: significant digits, trailing zeros dropped, an exponent outside 1e-4..1e±digits. */
function g(x: number | null | undefined, digits = 6): string {
  if (x === null || x === undefined || !Number.isFinite(x)) return '—'
  if (x === 0) return '0'
  const [m, e] = x.toExponential(digits - 1).split('e'), exp = Number(e)
  if (exp < -4 || exp >= digits) return `${m.includes('.') ? m.replace(/\.?0+$/, '') : m}e${exp < 0 ? '-' : '+'}${String(Math.abs(exp)).padStart(2, '0')}`
  return String(Number(x.toPrecision(digits)))
}

const signed = (x: number, nd: number) => `${x >= 0 ? '+' : ''}${x.toFixed(nd)}`
const BOOK_ICON: Record<string, string> = { selector: 'psychology', fly: 'neurology', live: 'payments' }
const bookKind = (book: string) => (book.startsWith('live') ? 'live' : book.endsWith('fly') ? 'fly' : 'selector')

function TokenButton({ chain, mint, token }: { chain: string; mint: string; token: string }) {
  const { openInspector } = useApp()
  return <button type="button" className="link-btn" title={mint} onClick={() => openInspector({ kind: 'token', chain, mint })}>{token}</button>
}

/** The selector book's KPI row, then the narrative beside the win rate. */
function SummaryPanel() {
  const { chain } = useApp()
  const s = useTopic<Summary>('overview.summary', { chain })
  return (
    <Loaded state={s} height={150}>
      {(d) => {
        const u = d.market.unit, winRate = d.closed ? d.wins / d.closed : null
        return (
          <>
            <WidgetGrid>
              <Widget label={d.label} icon="wealth" value={amount(d.wealth, u)} color={d.live ? 'magenta' : null} help={d.wealth_help} spark={d.spark.length > 1 ? d.spark : null}
                delta={d.wealth !== null ? `${signed(d.wealth - d.capital, 4)} ${u} since start` : null} />
              <Widget label="Realized P&L" icon="pnl" value={amount(d.pnl, u, 4, true)} help="Closed trades, after fees and price impact."
                spark={d.pnl_spark.length > 1 ? d.pnl_spark : null} />
              <Widget label="Closed trades" icon="count" value={int(d.closed)} delta={winRate !== null ? `${Math.round(winRate * 100)}% winners` : null} tone="off"
                progress={winRate} />
              <Widget label="Open positions" icon="open" value={int(d.n_open)} delta={d.marked ? `${g(d.exposure ?? 0, 3)} ${u} at cost` : null} tone="off"
                progress={d.marked && d.wealth ? Math.min(1, (d.exposure ?? 0) / d.wealth) : null} />
              <Widget label="Drawdown" icon="drawdown" value={pct(d.drawdown, 2, false)} help="Below the book's peak wealth."
                spark={d.dd_spark.length > 1 ? d.dd_spark.map((x) => -x) : null} />
            </WidgetGrid>
            <div className="row gx-4">
              <div className="col-12 col-xl-8 d-flex">
                <Card title="In plain words" icon="bx-message-square-detail" className="flex-fill"><Md text={d.narrative} /></Card>
              </div>
              <div className="col-12 col-xl-4 d-flex">
                <Card title="Win rate" icon="target" sub={d.live ? 'closed live trades that made money' : 'closed trades of the selector book that made money'} className="flex-fill" bodyClass="pb-0">
                  <RadialGauge value={winRate} label="winners" display={winRate === null ? '—' : `${Math.round(winRate * 100)}%`} height={210} />
                  <RowGroup items={[{ label: 'closed', value: int(d.closed) }, { label: 'winners', value: int(d.wins) }, { label: `realized ${u}`, value: num(d.pnl ?? 0, 3, true) }]} />
                </Card>
              </div>
            </div>
          </>
        )
      }}
    </Loaded>
  )
}

/** This minute's scoring as compact KPI tiles, with the top scores as a progress list beside it. */
function ThisMinute() {
  const { chain } = useApp()
  const m = useTopic<Minute>('overview.minute', { chain })
  const now = useNow(1000)
  return (
    <Loaded state={m} height={160}>
      {(d) => {
        const caption = d.minute ? `Scored ${ago(d.scored_at, now)}` + (d.latency_s !== null && d.latency_s !== undefined ? ` · ${Number(d.latency_s).toFixed(0)} s after the minute closed` : '')
          + (d.trading !== 'trading' ? ` · ${d.trading_why}` : '') : null
        return (
          <div className="row gx-4">
            <div className="col-12 col-xl-8 d-flex">
              <Card title="This minute" icon="time" sub={caption} className="flex-fill">
                {!d.minute ? <Caption>{d.runner ? 'No minute scored yet.' : 'The trading engine is stopped.'}</Caption> : (
                  <Metrics min={250}>
                    <Metric compact icon="time" label="Minute (UTC)" value={d.minute_hm} />
                    <Metric compact icon="layers" label="Tokens traded" value={int(d.mints_traded)} />
                    <Metric compact icon="bx-filter-alt" label="Eligible" value={int(d.eligible)} help={d.eligible_help}
                      progress={d.mints_traded ? d.eligible / d.mints_traded : null} />
                    <Metric compact icon="score" label="Best score" value={d.score_max.toFixed(3)} delta={`needs ${d.threshold.toFixed(3)}`} tone="off"
                      progress={d.threshold > 0 ? Math.max(0, d.score_max) / d.threshold : null} color={d.threshold > 0 && d.score_max >= d.threshold ? 'success' : null} />
                    <Metric compact icon="target" label="Picks" value={int(d.picks)} color={d.picks ? 'violet' : null} />
                    <Metric compact icon="swap" label="Bought / sold" value={`${d.entered} / ${d.exited}`} />
                  </Metrics>
                )}
              </Card>
            </div>
            <div className="col-12 col-xl-4 d-flex">
              <Card title="Top scores" icon="chart" sub="the minute's best-scored tokens (score 0..1)" className="flex-fill">
                {d.top.length ? (
                  <ProgressList max={1} items={d.top.map((t) => ({ label: <TokenButton chain={chain} mint={t.mint} token={t.token} />, value: t.score.toFixed(3), frac: Math.max(0, t.score) }))} />
                ) : <Caption>{d.minute ? 'No token scored this minute.' : 'Nothing scored yet.'}</Caption>}
              </Card>
            </div>
          </div>
        )
      }}
    </Loaded>
  )
}

/** The race: wealth by book as a big area chart, the three 14-day P&L widgets beside it. */
function RacePanel() {
  const { chain } = useApp()
  const r = useTopic<Race>('overview.race', { chain })
  if (!r.data) return r.error ? <Callout kind="error">{r.error}</Callout> : null
  const d = r.data
  if (d.hidden) return null
  const u = d.unit
  return (
    <div className="row gx-4">
      <div className="col-12 col-xl-8 d-flex">
        <Card title="The race: selector vs plastic fly" icon="race" sub={d.success ? null : d.caption} className="flex-fill">
          {r.error && <Callout kind="error">{r.error} (showing the last good data)</Callout>}
          {d.success && <Callout kind="success" icon="swap">{d.success}</Callout>}
          {(d.series ?? []).length > 0 ? <LineChart series={d.series!} height={300} yFormat={(v) => `${v.toFixed(2)} ${u}`} /> : <Caption>No marks in the last 30 days yet.</Caption>}
        </Card>
      </div>
      <div className="col-12 col-xl-4">
        <div className="d-flex flex-column gap-4 mb-4">
          {(d.metrics ?? []).map((m) => (
            <Widget key={m.book} label={m.title} icon={BOOK_ICON[bookKind(m.book)]} value={amount(m.pnl, u, 4, true)} delta={`${m.n} trades`} tone="off"
              help="Realized P&L of closed trades (bars: each day's)." spark={m.daily.some((x) => x !== 0) ? m.daily : null} sparkType="bar"
              color={bookKind(m.book) === 'live' ? 'danger' : null} />
          ))}
        </div>
      </div>
    </div>
  )
}

/** The selector book: its whole wealth history, and the open positions. */
function BookPanel() {
  const { chain } = useApp()
  const b = useTopic<Book>('overview.book', { chain })
  const cols = (u: string): Col<Position>[] => [
    { key: 'token', label: 'token', render: (r) => <TokenButton chain={chain} mint={r.mint} token={r.token} /> },
    { key: 'strategy', label: 'strategy' },
    { key: 'held_min', label: 'held (min)', num: true, render: (r) => num(r.held_min, 0) },
    { key: 'exits_in_min', label: 'exits in (min)', num: true, render: (r) => num(r.exits_in_min, 0) },
    { key: 'cost', label: `cost (${u})`, num: true, render: (r) => g(r.cost, 4) },
    { key: 'ret', label: 'return (gross)', num: true, render: (r) => (r.ret === null ? null : <span className={`text-${tone(r.ret)}`}>{pctPts(r.ret, 2, true)}</span>) },
  ]
  return (
    <div className="row gx-4">
      <div className="col-12 col-xl-7 d-flex">
        <Card title={b.data?.title ?? 'Paper wealth'} icon="wealth" sub={b.data?.sub ?? 'the selector book since it started'} className="flex-fill">
          <Loaded state={b} height={260}>
            {(d) => d.history.length ? <LineChart series={[{ name: 'wealth', data: d.history }]} height={280} yFormat={(v) => `${v.toFixed(2)} ${d.unit}`} /> : <Caption>No marks yet.</Caption>}
          </Loaded>
        </Card>
      </div>
      <div className="col-12 col-xl-5 d-flex">
        <Card title="Open positions" icon="open" className="flex-fill">
          <Loaded state={b}>
            {(d) => d.positions.length ? <DataTable rows={d.positions} cols={cols(d.unit)} pageSize={10} filename="open-positions" /> : <Caption>None open.</Caption>}
          </Loaded>
        </Card>
      </div>
    </div>
  )
}

type HourChain = { chain: string; name: string; unit: string; all: HourBucket[]; live: HourBucket[] }
type Hours = { tz: string; archives: number; chains: HourChain[] }
const LOCAL_SHIFT = Math.round(-new Date().getTimezoneOffset() / 60)
const pad2 = (h: number) => String(h).padStart(2, '0')

/** Every trade ever entered, by hour of day: one wheel per chain, Solana on the left, Robinhood Chain on the right. */
function HoursPanel() {
  const h = useTopic<Hours>('overview.hours')
  const [scope, setScope] = useStored<'all' | 'live'>('hours_scope', 'all')
  const [tz, setTz] = useStored<'UTC' | 'local'>('hours_tz', 'UTC')
  const shift = tz === 'local' ? LOCAL_SHIFT : 0
  const localLabel = `Local (UTC${LOCAL_SHIFT >= 0 ? '+' : ''}${LOCAL_SHIFT})`
  return (
    <Card title="Trades by hour of day" icon="bx-time-five"
      sub={`every trade ever entered, by the hour it opened (${tz === 'UTC' ? 'UTC' : localLabel.toLowerCase()}) · the larger the slice, the busier the hour`}
      actions={<>
        <Segmented options={['all', 'live'] as const} value={scope} onChange={setScope} labels={{ all: 'All books', live: 'Live' }} />
        <Segmented options={['UTC', 'local'] as const} value={tz} onChange={setTz} labels={{ UTC: 'UTC', local: localLabel }} />
      </>}>
      <Loaded state={h} height={420}>
        {(d) => (
          <div className="row g-4">
            {d.chains.map((c) => {
              const b = c[scope]
              const total = b.reduce((s, x) => s + x.trades, 0)
              const shown = Array.from({ length: 24 }, (_, i) => ({ ...(b[(((i - shift) % 24) + 24) % 24] ?? { trades: 0, closed: 0, wins: 0, pnl: 0 }), hour: i }))   // the hour shown
              const ranked = [...shown].sort((x, y) => y.trades - x.trades)
              const quiet = [...shown].sort((x, y) => x.trades - y.trades)[0]
              return (
                <div key={c.chain} className={`col-12 ${d.chains.length > 1 ? 'col-xl-6' : ''}`}>
                  <div className="d-flex align-items-baseline gap-2 mb-1">
                    <h6 className="m-0" style={{ color: '#fff' }}>{c.name}</h6>
                    <span className="caption m-0">{int(total)} trades{scope === 'live' ? ' with real money' : ''}</span>
                  </div>
                  {total ? (
                    <>
                      <HourWheel buckets={b} unit={c.unit} shift={shift} />
                      <div className="d-flex flex-wrap gap-2 justify-content-center mt-1">
                        {ranked.slice(0, 3).map((x, i) => (
                          <span key={x.hour} className="fbadge" style={{ background: hourColor(x.hour), color: '#120a18' }}>
                            #{i + 1} {pad2(x.hour)}:00 · {int(x.trades)} ({((x.trades / total) * 100).toFixed(1)}%)
                          </span>
                        ))}
                        <span className="fbadge gray">quietest {pad2(quiet.hour)}:00 · {int(quiet.trades)}</span>
                      </div>
                    </>
                  ) : <Caption>{scope === 'live' ? 'No live trades on this chain yet.' : 'No trades on this chain yet.'}</Caption>}
                </div>
              )
            })}
          </div>
        )}
      </Loaded>
      <Caption>Counts every memecoin position ever opened, paper and live: the positions table plus every reset's archive (data/pg_archive). Hover a slice for its trades, win rate and realized P&L.</Caption>
    </Card>
  )
}

export default function Overview() {
  return (
    <>
      <PageHead title="Overview" icon="bx-home-alt" />
      <SummaryPanel />
      <BrainPanel variant="memecoin" />
      <ThisMinute />
      <RacePanel />
      <HoursPanel />
      <TickerCard />
      <BookPanel />
    </>
  )
}
