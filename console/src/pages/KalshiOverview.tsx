// Kalshi overview: both arms' books as KPI widgets, wealth by book beside the settled win rates, what the visual fly is
// doing and this minute's scoring, the brain with both flies lit, and the open positions by category
// (fly_trader/ui/api/kalshi_overview.py).
import { BrainPanel } from '../brain'
import { PageHead } from '../components/ChainPicker'
import { DonutChart, LineChart, RadialGauge, type Series } from '../components/Charts'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { Callout, Caption, Card, Icon, KVList, Loaded, Md, Metric, Metrics, ProgressList, SectionTitle, Widget, WidgetGrid } from '../components/ui'
import { useApp } from '../context'
import { ago, int, pct, usd } from '../lib/format'
import { useNow, useTopic } from '../lib/hooks'

type DormantData = { dormant: true }
const isDormant = (d: unknown): d is DormantData => !!d && (d as DormantData).dormant === true

type Arm = {
  arm: string; book: string; label: string; marked: boolean; wealth: number | null; exposure: number; n_open: number; drawdown: number | null
  pnl: number; settled: number; wins: number; spark: number[]
}
type Gap = { live_mean: number | null; mirror_mean: number | null }
type Live = {
  stage: string | null; wealth: number | null; cash: number | null; wealth_taker: number | null; wealth_maker: number | null; error: string | null
  gap: { taker: Gap; maker: Gap }
}
type Minute = {
  minute: string | null; markets_active: number; eligible: number; picks: number; taker_entered: number; maker_posted: number; maker_resting: number
  settled: number; stage: string | null; at: string | null
}
type Summary = {
  narrative: string; capital: number; capital_g: string; arms: Arm[]; live: Live | null; minute: Minute; runner: boolean; trading: string; trading_why: string
}
type OpenRow = {
  book: string; ticker: string; market: string; side: string; strategy: string; contracts: number; paid: number; now: number | null
  closes_in_h: number | null; category: string
}
type Positions = { series: Series[]; rows: OpenRow[]; categories: [string, number][] }

export function KalshiDormant() {
  return (
    <Callout kind="info">
      Kalshi is not set up on this install: set <code>KALSHI_API_KEY_ID</code> (and <code>KALSHI_PRIVATE_KEY_PATH</code>) in <code>.env</code> and restart
      the console. The prediction-market pages join the menu once it is set.
    </Callout>
  )
}

/** Python's f"{x:+.2f}" (no thousands separator). */
const signed2 = (v: number) => `${v >= 0 ? '+' : ''}${v.toFixed(2)}`
/** A float as Streamlit's table shows it: at most four decimals, trailing zeros dropped. */
const plain = (v: number | null | undefined) => (v === null || v === undefined ? null : Number(v.toFixed(4)).toLocaleString('en-US', { maximumFractionDigits: 4 }))
const fixed = (v: number | null | undefined, nd: number) => (v === null || v === undefined ? null : v.toFixed(nd))
const winRate = (a: Arm) => (a.settled ? a.wins / a.settled : null)
const ARM_ICON: Record<string, string> = { taker: 'bx-log-in-circle', maker: 'bx-time-five' }

export default function KalshiOverview() {
  const { meta } = useApp()
  const off = !!meta && !meta.kalshi
  return (
    <>
      <PageHead title="Kalshi overview" icon="bx-show" chain={false} status={!off} />
      {off ? <KalshiDormant /> : <Body />}
    </>
  )
}

function Body() {
  const s = useTopic<Summary | DormantData>('kalshi_overview.summary')
  const p = useTopic<Positions | DormantData>('kalshi_overview.positions')
  if (isDormant(s.data) || isDormant(p.data)) return <KalshiDormant />
  return (
    <>
      <Loaded state={s} height={260}>{(d) => isDormant(d) ? null : <ArmColumns d={d} />}</Loaded>
      <div className="row g-4">
        <div className="col-12 col-xl-8">
          <Card title="Wealth by book" icon="chart" sub="Every Kalshi book's wealth over the last 30 days, in USD">
            <Loaded state={p} height={280}>
              {(d) => isDormant(d) ? null : d.series.length ? <LineChart series={d.series} height={290} yLabel="USD" yFormat={(v) => usd(v)} /> : <Caption>No marks yet.</Caption>}
            </Loaded>
          </Card>
        </div>
        <div className="col-12 col-xl-4">
          <Card title="Settled winners" icon="target" sub="Share of settled and closed positions that made money, per paper arm">
            <Loaded state={s} height={280}>{(d) => isDormant(d) ? null : <WinGauges arms={d.arms} />}</Loaded>
          </Card>
        </div>
      </div>
      <Loaded state={s} height={200}>{(d) => isDormant(d) ? null : <NarrativeRow d={d} />}</Loaded>
      <BrainPanel variant="flies" />
      <Loaded state={p} height={240}>{(d) => isDormant(d) ? null : <OpenPositions d={d} />}</Loaded>
    </>
  )
}

function ArmColumns({ d }: { d: Summary }) {
  return (
    <div className="row g-4">
      {d.arms.map((a) => (
        <div className="col-12 col-xl-6" key={a.book}>
          <SectionTitle><Icon name={ARM_ICON[a.arm]} className="me-2" />{a.label}</SectionTitle>
          <Widget label="Wealth" icon="wealth" value={usd(a.wealth)} delta={a.wealth !== null ? `${signed2(a.wealth - d.capital)} $ since start` : null}
            spark={a.spark.length ? a.spark : null} help={`Started at $${d.capital_g}; open positions at the bid net of the exit fee, resting bids at their collateral.`} />
          <WidgetGrid min={150}>
            <Widget compact label="Settled P&L" icon="pnl" value={usd(a.pnl, 2, true)} help="Settled and closed positions, after fees." />
            <Widget compact label="Settled" icon="count" value={int(a.settled)} delta={a.settled ? `${((a.wins / a.settled) * 100).toFixed(0)}% winners` : null} tone="off"
              progress={winRate(a)} />
            <Widget compact label="Open" icon="open" value={int(a.n_open)} delta={a.marked ? `$${a.exposure.toFixed(2)} at cost` : null} tone="off" />
            <Widget compact label="Drawdown" icon="drawdown" value={pct(a.drawdown, 2, false)} progress={a.drawdown} />
          </WidgetGrid>
        </div>
      ))}
    </div>
  )
}

function WinGauges({ arms }: { arms: Arm[] }) {
  return (
    <>
      <div className="row g-0">
        {arms.map((a) => (
          <div className="col-6" key={a.book}>
            <RadialGauge value={winRate(a)} label={a.arm} display={a.settled ? `${((a.wins / a.settled) * 100).toFixed(0)}%` : '—'} height={210} />
          </div>
        ))}
      </div>
      <KVList items={arms.map((a, i) => ({ label: a.label, value: a.settled ? `${int(a.wins)} of ${int(a.settled)} settled` : 'none settled yet',
        dot: i === 0 ? '#ffffff' : 'rgba(255,255,255,0.5)' }))} />
    </>
  )
}

function NarrativeRow({ d }: { d: Summary }) {
  const now = useNow(5000)
  const m = d.minute
  return (
    <>
      <div className="row g-4">
        <div className="col-12 col-xl-7">
          <Card title="What the visual fly is doing" icon="visibility"><Md text={d.narrative} /></Card>
        </div>
        <div className="col-12 col-xl-5">
          <Card title="This minute" icon="time"
            sub={m.minute ? <>Minute {String(m.minute).slice(11, 16)} UTC · scored {ago(m.at, now)} · {m.stage}{d.trading !== 'trading' ? ` · ${d.trading_why}` : ''}</> : undefined}>
            {!m.minute ? <Caption>{d.runner ? 'No minute scored yet.' : 'The Kalshi engine is stopped.'}</Caption> : (
              <Metrics min={140}>
                <Metric compact label="Markets quoted" icon="bx-store-alt" value={int(m.markets_active)} help="Markets with a message this minute inside the entry window." />
                <Metric compact label="Eligible rows" icon="bx-filter-alt" value={int(m.eligible)} help="Both sides of each market passing the spread, depth, volume and time gates." />
                <Metric compact label="Picks" icon="target" value={int(m.picks)} />
                <Metric compact label="Taker bought" icon={ARM_ICON.taker} value={int(m.taker_entered)} />
                <Metric compact label="Maker posted / resting" icon={ARM_ICON.maker} value={`${m.maker_posted} / ${m.maker_resting}`} />
                <Metric compact label="Settled" icon="success" value={int(m.settled)} />
              </Metrics>
            )}
          </Card>
        </div>
      </div>
      {d.live && <LiveMirror live={d.live} />}
    </>
  )
}

function LiveMirror({ live }: { live: Live }) {
  return (
    <Card title="Live mirror" icon="payments" sub="Both arms mirrored onto the Kalshi subaccount with real dollars">
      <Metrics min={150}>
        <Metric label="Live mirror" icon="payments" value={live.stage || 'not running'} color={live.error ? 'danger' : live.stage ? 'info' : 'dark'} />
        {live.wealth !== null && (
          <>
            <Metric label="Live wealth" icon="wealth" value={usd(live.wealth)} />
            <Metric label="Cash" icon="bx-money" value={usd(live.cash)} />
            <Metric label="Taker" icon={ARM_ICON.taker} value={usd(live.wealth_taker)} />
            <Metric label="Maker" icon={ARM_ICON.maker} value={usd(live.wealth_maker)} />
            <Metric label="Gap · taker" icon="bx-git-compare" value={pct(live.gap.taker.live_mean)} delta={`mirror ${pct(live.gap.taker.mirror_mean)}`} tone="off" />
            <Metric label="Gap · maker" icon="bx-git-compare" value={pct(live.gap.maker.live_mean)} delta={`mirror ${pct(live.gap.maker.mirror_mean)}`} tone="off" />
          </>
        )}
      </Metrics>
      {live.error && <Callout kind="error">{live.error}</Callout>}
    </Card>
  )
}

function OpenPositions({ d }: { d: Positions }) {
  const { openInspector } = useApp()
  const cols: Col<OpenRow>[] = [
    { key: 'book', label: 'book', render: (r) => <Pill color={r.book.startsWith('Taker') ? 'blue' : 'violet'}>{r.book}</Pill> },
    { key: 'market', label: 'market', render: (r) => <button type="button" className="link-btn" title={r.ticker} onClick={() => openInspector({ kind: 'market', ticker: r.ticker })}>{r.market}</button> },
    { key: 'side', label: 'side', render: (r) => <Pill color={r.side === 'yes' ? 'green' : 'red'}>{r.side}</Pill> },
    { key: 'strategy', label: 'strategy' },
    { key: 'contracts', label: 'contracts', num: true, render: (r) => plain(r.contracts) },
    { key: 'paid', label: 'paid (c)', num: true, render: (r) => fixed(r.paid, 1) },
    { key: 'now', label: 'now (c)', num: true, render: (r) => fixed(r.now, 0) },
    { key: 'closes_in_h', label: 'closes in (h)', num: true, render: (r) => fixed(r.closes_in_h, 1) },
    { key: 'category', label: 'category' },
  ]
  const total = d.categories.reduce((a, [, n]) => a + n, 0)
  return (
    <div className="row g-4">
      <div className="col-12 col-xl-8">
        <Card title="Open positions" icon="open" sub="Every open Kalshi position: click a market for its inspector">
          {!d.rows.length ? <Caption>None open.</Caption> : <DataTable rows={d.rows} cols={cols} filename="kalshi-open-positions" />}
        </Card>
      </div>
      <div className="col-12 col-xl-4">
        <Card title="Open by category" icon="percent" sub="Open positions per market category">
          {!d.categories.length ? <Caption>None open.</Caption> : (
            <>
              <DonutChart labels={d.categories.map(([c]) => c)} values={d.categories.map(([, n]) => n)} center={int(total)} centerLabel="open" height={250} />
              <ProgressList items={d.categories.map(([c, n]) => ({ label: c, value: int(n), frac: n / total }))} max={1} />
            </>
          )}
        </Card>
      </div>
    </div>
  )
}
