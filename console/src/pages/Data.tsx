// Data pipelines: is each source current, and how far along are the historical builds (fly_trader/ui/api/data.py).
// Solana only (the Robinhood Chain indexer has its own page), so no chain picker.
import { PageHead } from '../components/ChainPicker'
import { LineChart, RadialGauge } from '../components/Charts'
import { Caption, Card, Help, KVList, Loaded, ProgressList, RowGroup, STATE_COLOR, Widget, WidgetGrid, type BadgeData, type WidgetColor } from '../components/ui'
import { ago, hm, int, pct } from '../lib/format'
import { useNow, useTopic } from '../lib/hooks'

type Pipelines = {
  feed: {
    badge: BadgeData; events_per_s: number; eps_spark: number[]; age: number | null; migrates: number; reconnects: number; dropped_band: number; dropped_late: number
    updated_at: string | null; per_minute: [string, number, number][]
  }
  archive: {
    badge: BadgeData; hours_done: number; hours_total: number; progress: number | null; days: number; newest_day: string | null; errors: number
    replay_start: string; last_hour: string; eta_h: number | null; updated_at: string | null; per_day: [string, number, number][]
  }
  features: { n: number; first: string | null; last: string | null; current: number }
  meta: { graduations: number; creators: number; launches: number }
  stats: { badge: BadgeData }
}
type Coverage = { universe: number; covered: number; first: string | null }

const STALE_S = 180          // fly_trader/ui/state.py STALE_FEED_S: no entries or exits past this
const color = (b: BadgeData): WidgetColor => STATE_COLOR[b.color] ?? 'dark'

function TokenStats() {
  const s = useTopic<Coverage>('data.token_stats')
  return (
    <Loaded state={s} height={260}>
      {(c) => (
        <>
          <RadialGauge value={c.universe ? c.covered / c.universe : 0} label="covered" display={c.universe ? pct(c.covered / c.universe, 0, false) : '—'} height={230} />
          <KVList items={[
            { label: <>Tradable tokens with stats<Help text="Tokens that traded in the last hour with a pool above the selector's gate, and how many have Jupiter stats from the last hour." /></>,
              value: `${int(c.covered)}/${int(c.universe)}` },
            { label: 'History since', value: c.first ? c.first.slice(0, 10) : '—' },
          ]} />
        </>
      )}
    </Loaded>
  )
}

export default function Data() {
  const s = useTopic<Pipelines>('data.pipelines')
  const now = useNow(5000)
  return (
    <>
      <PageHead title="Data pipelines" icon="bx-data" chain={false} />
      <Loaded state={s} height={480}>
        {(d) => {
          const f = d.feed, a = d.archive, fp = d.features, m = d.meta
          const featState = fp.n === 0 ? 'none built' : fp.current === fp.n ? 'current' : 'mixed versions'
          const pm = f.per_minute, last = pm[pm.length - 1] ?? ['', 0, 0]
          const avg = pm.length ? pm.reduce((x, r) => x + r[1], 0) / pm.length : 0, peak = pm.reduce((x, r) => Math.max(x, r[1]), 0)
          return (
            <>
              <WidgetGrid>
                <Widget label="Market feed (live)" value={f.badge.text} color={color(f.badge)} icon="sensors" tone="off"
                  delta={f.age !== null ? `newest minute ${f.age.toFixed(0)} s old` : 'no minute written yet'} />
                <Widget label="History archive (training data)" value={a.badge.text} color={color(a.badge)} icon="history" tone="off" delta={`updated ${ago(a.updated_at, now)}`} />
                <Widget label="Training features" value={featState} color={fp.n === 0 ? 'dark' : fp.current === fp.n ? 'success' : 'warning'} icon="bx-table" tone="off"
                  delta={`${fp.current}/${fp.n} days on the current version`} help="Days built with the live feature engine's current definitions; training refuses a mix." />
                <Widget label="Token stats (Jupiter)" value={d.stats.badge.text} color={color(d.stats.badge)} icon="query_stats" tone="off" delta="every 10 minutes, optional" />
              </WidgetGrid>
              <WidgetGrid>
                <Widget label="Events / s" value={f.events_per_s.toFixed(0)} icon="pulse" spark={f.eps_spark.length >= 6 ? f.eps_spark : null}
                  delta={f.eps_spark.length >= 6 ? `last ${f.eps_spark.length} status updates` : null} tone="off" />
                <Widget label="Newest minute" value={f.age !== null ? `${f.age.toFixed(0)} s old` : '—'} icon="time" help="Seconds since the newest complete minute closed."
                  progress={f.age !== null ? Math.min(1, f.age / STALE_S) : null} delta={`stale after ${STALE_S} s`} tone="off" />
                <Widget label="Graduations seen" value={int(f.migrates)} icon="rocket" />
                <Widget label="Reconnects" value={int(f.reconnects)} icon="refresh" />
              </WidgetGrid>
              <div className="row g-4">
                <div className="col-12 col-xl-8">
                  <Card title="Market feed" sub="(live) · PumpSwap trades per minute, last 3 hours" icon="sensors">
                    {f.per_minute.length > 1 ? (
                      <>
                        <LineChart height={260} yLabel="trades per minute" series={[{ name: 'Trades', data: f.per_minute.map(([t, n]) => [t, n]) }]} />
                        <RowGroup items={[
                          { label: `trades in the minute ${hm(last[0])}`, value: int(last[1]) },
                          { label: 'tokens traded that minute', value: int(last[2]) },
                          { label: 'trades per minute, 3 h average', value: int(avg) },
                          { label: 'busiest minute', value: int(peak) },
                        ]} />
                      </>
                    ) : <div className="empty">No minutes written in the last 3 hours.</div>}
                    <Caption>
                      PumpSwap trades → 1-minute candles (pump_minutes) · dropped as outliers {int(f.dropped_band)} · late {int(f.dropped_late)} · updated {ago(f.updated_at, now)}
                    </Caption>
                  </Card>
                </div>
                <div className="col-12 col-xl-4">
                  <Card title="History archive" sub="(training data)" icon="history">
                    <RadialGauge value={a.progress ?? 0} label={a.hours_total && a.hours_done >= a.hours_total ? 'hours · caught up' : 'hours ingested'}
                      display={a.hours_total && a.hours_done < a.hours_total ? `${int(a.hours_done)}/${int(a.hours_total)}` : int(a.hours_done)} height={230} />
                    <RowGroup items={[
                      { label: 'days assembled', value: int(a.days) },
                      { label: 'newest day', value: a.newest_day ? a.newest_day.slice(5) : '—' },
                      { label: 'errors', value: int(a.errors) },
                    ]} />
                    <Caption>
                      pumpapi.io hourly archive since {a.replay_start}, newest first · last hour {a.last_hour}
                      {a.eta_h !== null && a.eta_h > 0 ? ` · about ${a.eta_h.toFixed(1)} h left` : a.hours_total && a.hours_done >= a.hours_total ? ' · caught up' : ''} · updated {ago(a.updated_at, now)}
                    </Caption>
                  </Card>
                </div>
              </div>
              <div className="row g-4">
                <div className="col-12 col-xl-8">
                  <Card title="Days assembled" sub={`graduations per assembled day, last ${a.per_day.length || 0} days`} icon="calendar">
                    {a.per_day.length > 1 ? (
                      <LineChart height={300} yLabel="graduations" curve="straight" series={[{ name: 'Graduations', data: a.per_day.map(([day, g]) => [day, g]) }]} />
                    ) : <div className="empty">No day assembled yet.</div>}
                    {a.per_day.length > 0 && (
                      <RowGroup items={[
                        { label: 'graduations, newest day', value: int(a.per_day[a.per_day.length - 1][1]) },
                        { label: 'tokens written, newest day', value: int(a.per_day[a.per_day.length - 1][2]) },
                        { label: `average over ${a.per_day.length} days`, value: int(a.per_day.reduce((x, r) => x + r[1], 0) / a.per_day.length) },
                      ]} />
                    )}
                  </Card>
                </div>
                <div className="col-12 col-xl-4">
                  <Card title="Training features" icon="bx-table">
                    <RadialGauge value={fp.n ? fp.current / fp.n : 0} label="current version" display={`${fp.current}/${fp.n}`} height={210} />
                    <KVList items={[
                      { label: 'Days built', value: int(fp.n) },
                      { label: 'Range', value: `${fp.first ?? '—'} → ${fp.last ?? '—'}` },
                      { label: <>Current version<Help text="Days built with the live feature engine's current definitions; training refuses a mix." /></>, value: `${fp.current}/${fp.n}` },
                    ]} />
                    <Caption>One row per token-minute of every PumpSwap token, fed through the live feature engine. Built by the history archive worker.</Caption>
                  </Card>
                </div>
              </div>
              <div className="row g-4">
                <div className="col-12 col-xl-6">
                  <Card title="Token metadata" icon="bx-id-card">
                    <RowGroup items={[
                      { label: 'Graduations', value: int(m.graduations) },
                      { label: 'With creator', value: int(m.creators) },
                      { label: <>Token launches<Help text="Creates in pump_events, used for each creator's launch history." /></>, value: int(m.launches) },
                    ]} />
                    <div className="mt-3">
                      <ProgressList max={1} items={[
                        { label: 'Graduations with a known creator', value: m.graduations ? pct(m.creators / m.graduations, 1, false) : '—', frac: m.graduations ? m.creators / m.graduations : 0 },
                        { label: 'Launches that graduated', value: m.launches ? pct(m.graduations / m.launches, 1, false) : '—', frac: m.launches ? m.graduations / m.launches : 0 },
                      ]} />
                    </div>
                    <Caption>Graduation time, creation facts and each creator's track record, shared by training and live scoring.</Caption>
                  </Card>
                </div>
                <div className="col-12 col-xl-6">
                  <Card title="Token stats" sub="(Jupiter)" icon="query_stats">
                    <TokenStats />
                    <Caption>
                      Holders, organic score and top-holder share, recorded every 10 minutes for every token the selector can trade. Not a model input yet: Jupiter
                      serves only current values, so these need weeks of recorded history first.
                    </Caption>
                  </Card>
                </div>
              </div>
            </>
          )
        }}
      </Loaded>
    </>
  )
}
