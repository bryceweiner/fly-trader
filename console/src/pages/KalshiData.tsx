// Kalshi data: the live feed, the history corpus (dataset seed, candles, feature builds) and the catalogue's coverage as
// state widgets, the feed's pulse, the corpus gauge and split, feature builds per day and coverage by category
// (fly_trader/ui/api/kalshi_data.py).
import { PageHead } from '../components/ChainPicker'
import { DonutChart, LineChart, RadialGauge } from '../components/Charts'
import { Caption, Card, Help, KVList, Loaded, ProgressList, RowGroup, STATE_COLOR, Widget, WidgetGrid } from '../components/ui'
import { useApp } from '../context'
import { ago, int } from '../lib/format'
import { useNow, useTopic } from '../lib/hooks'
import { KalshiDormant } from './KalshiOverview'

type DormantData = { dormant: true }
const isDormant = (d: unknown): d is DormantData => !!d && (d as DormantData).dormant === true

type StateBadge = { text: string; color: string }
type Pipelines = {
  feed: {
    badge: StateBadge; events_per_s: number; age: number | null; tickers_quoted: number; settled: number; reconnects: number; private: boolean
    dropped_late: number; updated_at: string | null; per_minute: [string, number][]
  }
  history: {
    badge: StateBadge; settled_markets: number; filled: number; total: number; built: number; errors: number; progress: number | null; start: string
    stage: string | null; eta_h: number | null; seed_done: boolean; updated_at: string | null
  }
  corpus: { status: string; n: number }[]
  features: { n: number; days: number; first: string | null; last: string | null; current: number; per_day: { day: string; markets: number; rows: number }[] }
  categories: { category: string; markets: number }[]
}

const cap = (s: string) => s.charAt(0).toUpperCase() + s.slice(1)
const DONUT = ['#ffffff', 'rgba(255,255,255,0.6)', 'rgba(255,255,255,0.35)', 'rgba(255,255,255,0.18)', '#ff2fb9', '#8b5cf6', '#b6ff3b', '#3d9bff']

export default function KalshiData() {
  const { meta } = useApp()
  const off = !!meta && !meta.kalshi
  return (
    <>
      <PageHead title="Kalshi data" icon="bx-data" chain={false} status={!off} />
      {off ? <KalshiDormant /> : <Panels />}
    </>
  )
}

function Panels() {
  const s = useTopic<Pipelines | DormantData>('kalshi_data.pipelines')
  const now = useNow(5000)
  return (
    <Loaded state={s} height={320}>
      {(d) => isDormant(d) ? <KalshiDormant /> : (
        <>
          <WidgetGrid min={200}>
            <Widget label="Kalshi feed (live)" icon="sensors" color={STATE_COLOR[d.feed.badge.color] ?? 'dark'} value={cap(d.feed.badge.text)}
              delta={`${d.feed.events_per_s.toFixed(0)} messages / s · updated ${ago(d.feed.updated_at, now)}`} tone="off" />
            <Widget label="Kalshi history (training data)" icon="history" color={STATE_COLOR[d.history.badge.color] ?? 'dark'} value={cap(d.history.badge.text)}
              delta={`stage ${d.history.stage || '—'}`} tone="off" />
            <Widget label="Candles filled" icon="bx-bar-chart-square" value={d.history.progress !== null ? `${(d.history.progress * 100).toFixed(1)}%` : '—'}
              delta={`${int(d.history.filled)} of ${int(d.history.total)}`} tone="off" progress={d.history.progress} />
            <Widget label="Features current" icon="layers" color={d.features.n === 0 ? 'dark' : d.features.current === d.features.n ? 'success' : 'warning'}
              value={`${int(d.features.current)}/${int(d.features.n)}`} delta={`${int(d.features.days)} days built`} tone="off"
              help="Parts built with the feature engine's current definitions; training refuses a mix." />
            <Widget label="Dataset seed" icon="bx-cloud-download" color={d.history.seed_done ? 'success' : 'dark'} value={d.history.seed_done ? 'Done' : 'Not yet'}
              delta={`settled markets and trades since ${d.history.start}`} tone="off" />
          </WidgetGrid>

          <div className="row g-4">
            <div className="col-12 col-xl-7">
              <Card title="Kalshi feed" icon="sensors" sub="(live) ticker, trade and lifecycle channels → kalshi_minutes / kalshi_quotes">
                <h6 className="text-white mb-1">Markets with a minute row, per minute (last 3 h)</h6>
                {d.feed.per_minute.length > 1 ? <LineChart series={[{ name: 'markets', data: d.feed.per_minute }]} height={230} />
                  : <Caption>No minute rows in the last three hours.</Caption>}
                <RowGroup items={[
                  { label: 'Messages / s', value: d.feed.events_per_s.toFixed(0) },
                  { label: 'Newest minute', value: d.feed.age !== null ? `${d.feed.age.toFixed(0)} s old` : '—' },
                  { label: <>Markets quoted<Help text="Distinct markets with a quote since the feed started (combos excluded)." /></>, value: int(d.feed.tickers_quoted) },
                  { label: 'Settlements seen', value: int(d.feed.settled) },
                  { label: 'Reconnects', value: int(d.feed.reconnects) },
                ]} />
                <Caption>
                  {d.feed.private ? 'private channels on' : 'public channels only'} · late {int(d.feed.dropped_late)} · updated {ago(d.feed.updated_at, now)}
                </Caption>
              </Card>
            </div>
            <div className="col-12 col-xl-5">
              <Card title="Kalshi history" icon="history" sub="(training data) settled markets seeded from the open dataset, 1-minute candles filled from the API">
                <RadialGauge value={d.history.progress ?? 0} label="candles filled" display={d.history.progress !== null ? `${(d.history.progress * 100).toFixed(0)}%` : '—'} height={220} />
                <RowGroup items={[
                  { label: 'Settled markets', value: int(d.history.settled_markets) },
                  { label: `Candles filled of ${int(d.history.total)}`, value: int(d.history.filled) },
                  { label: 'Built', value: int(d.history.built) },
                  { label: 'Errors', value: <span className={d.history.errors ? 'text-neg' : ''}>{int(d.history.errors)}</span> },
                ]} />
                <Caption>
                  Since {d.history.start} · stage {d.history.stage || '—'}{d.history.eta_h ? ` · about ${d.history.eta_h.toFixed(1)} h left` : ''} · dataset seed{' '}
                  {d.history.seed_done ? 'done' : 'not yet'} · updated {ago(d.history.updated_at, now)}
                </Caption>
              </Card>
            </div>
          </div>

          <div className="row g-4">
            <div className="col-12 col-xl-7">
              <Card title="Kalshi features" icon="bx-table" sub="One row per (market, minute, side) every 5 minutes inside the entry window, through the same feature engine the live engine runs.">
                {d.features.per_day.length > 1 ? (
                  <>
                    <h6 className="text-white mb-1">Feature rows and markets per built day (the last {d.features.per_day.length} days built)</h6>
                    <LineChart height={230} yFormat={(v) => int(v)} series={[
                      { name: 'rows', data: d.features.per_day.map((x) => [String(x.day), x.rows] as [string, number]) },
                      { name: 'markets', data: d.features.per_day.map((x) => [String(x.day), x.markets] as [string, number]) },
                    ]} />
                  </>
                ) : <Caption>No feature days built yet.</Caption>}
                <RowGroup items={[
                  { label: 'Days built', value: int(d.features.days) },
                  { label: 'Range', value: `${d.features.first || '—'} → ${d.features.last || '—'}` },
                  { label: <>Current version<Help text="Parts built with the feature engine's current definitions; training refuses a mix." /></>, value: `${d.features.current}/${d.features.n}` },
                ]} />
              </Card>
            </div>
            <div className="col-12 col-xl-5">
              <Card title="Corpus by status" icon="percent" sub="Every market in the Kalshi corpus registry, by its candle status">
                {!d.corpus.length ? <Caption>No Kalshi corpus yet.</Caption> : (
                  <>
                    <DonutChart labels={d.corpus.map((x) => x.status)} values={d.corpus.map((x) => x.n)} center={int(d.history.total)} centerLabel="markets" height={230} />
                    <KVList items={d.corpus.map((x, i) => ({ label: x.status, value: `${int(x.n)} · ${((x.n / d.history.total) * 100).toFixed(1)}%`, dot: DONUT[i % DONUT.length] }))} />
                  </>
                )}
              </Card>
            </div>
          </div>

          <Card title="Coverage by category" icon="category" sub="Settled markets in the catalogue per category (top 15)">
            {d.categories.length ? (
              <div className="row g-4">
                {[d.categories.slice(0, Math.ceil(d.categories.length / 2)), d.categories.slice(Math.ceil(d.categories.length / 2))].map((part, i) => (
                  <div className="col-12 col-lg-6" key={i}>
                    <ProgressList max={d.categories[0].markets} items={part.map((c) => ({ label: c.category, value: int(c.markets), frac: c.markets }))} />
                  </div>
                ))}
              </div>
            ) : <Caption>No settled markets in the catalogue yet.</Caption>}
          </Card>
        </>
      )}
    </Loaded>
  )
}
