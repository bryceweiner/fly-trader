// The fly decision inspector: a drawer any token, Kalshi market or search result opens (?inspect= in the URL). For a token:
// every minute the fly scored it against its teacher and its line, how those rows turned out, and what the books did with
// it; for a market: the visual fly's edges and the Kalshi books' orders, positions and fills (fly_trader/ui/api/inspector.py).
import { useEffect, useMemo, type ReactNode } from 'react'
import { useApp, type InspectTarget } from '../context'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { DonutChart, LineChart, RadialGauge, type Series } from '../components/Charts'
import { useFeedback } from '../components/feedback'
import { Callout, Caption, Card, Empty, Expander, Icon, KVList, Loaded, Md, Metric, Metrics, ProgressList, RowGroup, Segmented, Toggle, Widget, WidgetGrid,
  type WidgetColor } from '../components/ui'
import { useStored, useTopic } from '../lib/hooks'
import { ago, amount, dt, hm, int, num, pct, plain, shortAddr, tone, usd } from '../lib/format'

type Scored = { ts: string; strategy: string; side?: string; score: number | null; frozen_score: number | null; line: number | null; label: number | null; state: string; teacher_allow?: boolean | null }
type Latest = Scored & { hold_min?: number | null; frozen_line?: number | null; bootstrap_id?: number | null }
type Stat = {
  strategy: string | null; n: number; pending: number; resolved: number; cleared: number; frozen_cleared?: number; teacher_allowed?: number; teacher_known?: number
  label_cleared: number | null; n_label_cleared: number; label_not: number | null; n_label_not: number; label_allowed?: number | null; n_label_allowed?: number
  label_all?: number | null; mean_correction: number | null; first: string | null; last: string | null
}
type Decision = { id: number; ts: string; kind: string; side?: string | null; m_hat: number | null; size_sol: number | null; rail: string | null; reason: string | null
  book: string | null; strategy?: string | null; threshold?: number | null; action: string; actor: string }
type Position = { id: number; book: string; status: string; opened_at: string; closed_at: string | null; strategy: string | null; hold_s: number | null; cost_sol: number
  entry_price: number | null; last_mark_price: number | null; exit_price: number | null; realized_sol: number | null; fees_sol: number; forced_exit_kind: string | null
  return: number | null; held_min: number | null }
type Fill = { id: number; ts: string; book: string; side: string; amount: number; price_sol: number | null; fee: number; source: string | null; signature: string | null; tx_url: string | null }
type Pathway = { kc: number; mbon: number; delta: number; ratio: number; strategy: string; valence: string; kc_type: string | null; mbon_type: string | null }
type Pathways = { mode: string; items: Pathway[]; note: string | null; changed: number; reference: string | null; bootstrap: number | null; running: boolean
  n_pairs?: number; drift: Record<string, number> }
type Minute = { ts: string; price: number | null; volume: number | null; price_5m: number | null; volume_5m: number }
type TokenData = {
  chain: string; chain_name: string; unit: string; mint: string; label: string; symbol: string | null; name: string | null; graduated_at: string | null; creator: string | null
  explorer: string; books: { selector: string; fly: string; live: string }; minute?: Minute | null; fly_score: string; teacher_gate: boolean; learned_veto: number
  series: Scored[]; stats: Stat[]; total: Stat | null; latest: Latest[]; explain: string; decisions: Decision[]; positions: Position[]; fills: Fill[]
  pnl: { closed: number; realized: number; open: number; wins: number }; pathways: Pathways
}
type KOrder = { id: number; ts: string; book: string; status: string; side: string; action: string; price_cents: number; count: number; fill_count: number; tif: string
  post_only: boolean; expiration_ts: string | null; strategy: string | null; order_id: string | null; error: string | null }
type KPosition = { id: number; book: string; status: string; side: string; strategy: string | null; arm: string | null; contracts: number; cost_cents: number; fee_cents: number
  avg_price_cents: number | null; opened_at: string; closed_at: string | null; result: string | null; payout_cents: number | null; realized_cents: number | null
  score: number | null; line: number | null; last_mark_cents: number | null; return: number | null }
type KFill = { trade_id: string; ts: string; book: string | null; side: string; action: string | null; price_cents: number; count: number; fee_cents: number | null; is_taker: boolean | null; order_id: string | null }
type MarketData = {
  ticker: string; enabled: boolean; label: string; books: Record<string, string>
  market: { ticker: string; title: string | null; yes_sub_title: string | null; status: string | null; open_time: string | null; close_time: string | null; result: string | null
    settlement_ts: string | null; event_ticker: string | null; event_title: string | null; category: string | null; series_title: string | null } | null
  series: Scored[]; stats: Stat[]; total: Stat | null; latest: Latest[]; explain: string; decisions: Decision[]; orders: KOrder[]; positions: KPosition[]; fills: KFill[]
  pnl: { closed: number; realized_usd: number; open: number; wins: number }
}

type PMode = 'bootstrap' | '24h'
const STATE_COLOR: Record<string, string> = { pending: 'blue', resolved: 'green', unknown: 'gray', dropped: 'orange' }
const ACTION_COLOR: Record<string, string> = { bought: 'green', sold: 'magenta', 'not taken': 'gray', posted: 'blue', filled: 'green', settled: 'violet', cancelled: 'gray' }
const f4 = (v: number | null | undefined) => (v === null || v === undefined ? '—' : v.toFixed(4))
const share = (a?: number, b?: number) => (b ? `${int(a)} of ${int(b)} (${((100 * (a ?? 0)) / b).toFixed(0)}%)` : '—')
const frac = (a?: number, b?: number) => (b ? (a ?? 0) / b : 0)

function Copy({ value, show }: { value: string; show?: ReactNode }) {
  const fb = useFeedback()
  return (
    <span className="d-inline-flex align-items-center gap-1">
      <code className="mono" title={value} style={{ color: '#fff' }}>{show ?? value}</code>
      <button type="button" className="btn btn-sm btn-light py-0 px-1" title="Copy" aria-label="Copy"
        onClick={() => { void navigator.clipboard?.writeText(value); fb.toast('Copied', 'info', 1500) }}><Icon name="copy" /></button>
    </span>
  )
}

/** The score-over-time chart of one strategy (and side, for a Kalshi market): the fly's score, its teacher's or frozen score, the line. */
function ScoreChart({ rows, teacherName, scoreName, labels, yLabel }: { rows: Scored[]; teacherName: string; scoreName: string; labels: boolean; yLabel: string }) {
  const series = useMemo<Series[]>(() => {
    const s: Series[] = [
      { name: scoreName, data: rows.map((r) => [r.ts, r.score]) },
      { name: teacherName, data: rows.map((r) => [r.ts, r.frozen_score]) },
      { name: 'Buy line', data: rows.map((r) => [r.ts, r.line]) },
    ]
    if (labels) s.push({ name: 'Realized label', data: rows.filter((r) => r.label !== null).map((r) => [r.ts, r.label]) })
    return s
  }, [rows, teacherName, scoreName, labels])
  return <LineChart series={series} height={270} yLabel={yLabel} area={false} colors={['#ffffff', 'rgba(255,255,255,0.45)', '#ff2fb9', '#b6ff3b']}
    yFormat={(v) => (typeof v === 'number' && Number.isFinite(v) ? v.toFixed(3) : '')} />
}

function strategyKeys(rows: Scored[], withSide: boolean): string[] {
  const seen: string[] = []
  for (let i = rows.length - 1; i >= 0; i--) {
    const k = withSide ? `${rows[i].strategy} · ${String(rows[i].side).toUpperCase()}` : rows[i].strategy
    if (!seen.includes(k)) seen.push(k)
  }
  return seen
}

/** The newest scored minute's row of figures under the chart. */
function LatestFigures({ r, teacher }: { r: Scored | undefined; teacher: string }) {
  if (!r) return null
  const gap = r.score !== null && r.line !== null ? r.score - r.line : null
  return (
    <div style={{ margin: '0.75rem -1.1rem -1rem' }}>
      <RowGroup items={[
        { value: f4(r.score), label: `fly score · ${ago(r.ts)}` },
        { value: f4(r.frozen_score), label: teacher },
        { value: f4(r.line), label: 'buy line' },
        { value: <span className={tone(gap)}>{gap === null ? '—' : (gap >= 0 ? '+' : '') + gap.toFixed(4)}</span>, label: 'vs line' },
      ]} />
    </div>
  )
}

/** Cleared-vs-not as a donut with the outcome of each side underneath (Dashtreme's "Weekly sales"). */
function ClearedCard({ t, title = 'Cleared its line' }: { t: Stat | null; title?: string }) {
  const n = t?.n ?? 0, c = t?.cleared ?? 0
  return (
    <Card title={title} icon="percent" className="h-100">
      {n ? (
        <>
          <DonutChart labels={['At or above the line', 'Below the line']} values={[c, n - c]} height={220} center={`${((100 * c) / n).toFixed(1)}%`} centerLabel="cleared" />
          <KVList items={[
            { dot: '#ffffff', label: `Cleared · ${int(c)} minutes`, value: <span className={tone(t?.label_cleared)}>{pct(t?.label_cleared)} <span className="text-muted2">({int(t?.n_label_cleared)})</span></span> },
            { dot: 'rgba(255,255,255,0.6)', label: `Below · ${int(n - c)} minutes`, value: <span className={tone(t?.label_not)}>{pct(t?.label_not)} <span className="text-muted2">({int(t?.n_label_not)})</span></span> },
          ]} />
          <Caption>Mean realized label of each side, (resolved rows) in brackets.</Caption>
        </>
      ) : <Empty>Nothing scored yet.</Empty>}
    </Card>
  )
}

function StatsTable({ stats, kalshi }: { stats: Stat[]; kalshi?: boolean }) {
  const rows = stats.filter((s) => s.strategy !== null) as (Stat & Record<string, unknown>)[]
  const cols: Col<Stat & Record<string, unknown>>[] = [
    { key: 'strategy', label: 'strategy' },
    { key: 'n', label: 'rows', num: true, render: (r) => int(r.n) },
    { key: 'cleared', label: 'cleared line', num: true, render: (r) => share(r.cleared, r.n), help: 'Minutes its score was at or above its own line.' },
    ...(kalshi ? [] : [{ key: 'teacher_allowed', label: 'teacher buys', num: true, render: (r: Stat) => share(r.teacher_allowed, r.teacher_known),
      help: 'Minutes its teacher (the selector) would have bought, of those where its decision was recorded.' } as Col<Stat & Record<string, unknown>>]),
    { key: 'label_cleared', label: 'mean label · cleared', num: true, render: (r) => <span className={tone(r.label_cleared)}>{pct(r.label_cleared)} <span className="text-muted2">({int(r.n_label_cleared)})</span></span> },
    { key: 'label_not', label: 'mean label · not', num: true, render: (r) => <span className={tone(r.label_not)}>{pct(r.label_not)} <span className="text-muted2">({int(r.n_label_not)})</span></span> },
    { key: 'mean_correction', label: kalshi ? 'mean learned change' : 'mean learned correction', num: true, render: (r) => (r.mean_correction === null ? '—' : r.mean_correction.toFixed(5)) },
    { key: 'last', label: 'last scored', render: (r) => ago(r.last), value: (r) => r.last },
  ]
  return <DataTable rows={rows} cols={cols} filename="fly-scored-by-strategy" search={false} />
}

function StrategyCard({ stats, kalshi }: { stats: Stat[]; kalshi?: boolean }) {
  const rows = stats.filter((s) => s.strategy !== null)
  if (!rows.length) return null
  return (
    <Card title="By strategy" icon="category" sub="rows scored per strategy; the bar is its share of this token's minutes">
      <ProgressList items={rows.map((r) => ({
        label: <span><strong>{r.strategy}</strong> <span className="text-muted2" style={{ fontSize: 12.5 }}>· cleared {share(r.cleared, r.n)}{kalshi ? '' : ` · teacher buys ${share(r.teacher_allowed, r.teacher_known)}`}</span></span>,
        value: `${int(r.n)} rows`, frac: r.n,
      }))} />
      <Expander title="Strategy details" icon="bx-table"><StatsTable stats={stats} kalshi={kalshi} /></Expander>
    </Card>
  )
}

function DecisionTable({ rows, unit, kalshi }: { rows: Decision[]; unit: string; kalshi?: boolean }) {
  const cols: Col<Decision>[] = [
    { key: 'ts', label: 'time', render: (r) => dt(r.ts, true), value: (r) => r.ts },
    { key: 'book', label: 'book' },
    { key: 'action', label: 'action', render: (r) => <Pill color={ACTION_COLOR[r.action] ?? 'gray'}>{r.actor} {r.action}</Pill>, value: (r) => `${r.actor} ${r.action}` },
    ...(kalshi ? [{ key: 'side', label: 'side', render: (r: Decision) => String(r.side ?? '').toUpperCase() } as Col<Decision>]
      : [{ key: 'strategy', label: 'strategy' } as Col<Decision>]),
    { key: 'm_hat', label: kalshi ? 'edge' : 'score', num: true, render: (r) => f4(r.m_hat) },
    ...(kalshi ? [] : [{ key: 'threshold', label: 'line', num: true, render: (r: Decision) => f4(r.threshold ?? null) } as Col<Decision>]),
    { key: 'size_sol', label: kalshi ? 'dollars' : `size (${unit})`, num: true, render: (r) => (kalshi ? usd(r.size_sol) : num(r.size_sol, unit === 'SOL' ? 4 : 6)) },
    { key: 'why', label: 'why', value: (r) => (r.rail ? `${r.rail}: ` : '') + (r.reason ?? ''),
      render: (r) => { const w = (r.rail ? `${r.rail}: ` : '') + (r.reason ?? ''); return <span title={w}>{w}</span> } },
  ]
  return <DataTable rows={rows} cols={cols} filename="inspector-decisions" empty={kalshi ? 'No decisions on the Kalshi books.' : "No decisions on this chain's books."} />
}

/** The token's price and volume in the newest scored minute, each with an up/down arrow against the five minutes before. */
function MinuteWidgets({ m, unit, scored }: { m: Minute; unit: string; scored: boolean }) {
  const dp = m.price !== null && m.price_5m ? m.price / m.price_5m - 1 : null
  const dv = m.volume !== null && m.volume_5m > 0 ? m.volume / m.volume_5m - 1 : null
  const which = scored ? 'the minute the fly last scored it' : 'its newest traded minute (the fly has not scored it)'
  return (
    <Metrics min={320}>
      <Metric compact label={`Price · ${hm(m.ts)}`} value={m.price === null ? '—' : `${plain(m.price)} ${unit}`}
        delta={dp === null ? 'no price five minutes before' : `${pct(dp, 1)} over 5 min`} tone={dp ? 'normal' : 'off'} icon="candlestick"
        help={`The close of ${which}, ${dt(m.ts)} UTC, in ${unit} per token; the arrow is its change from the close five minutes earlier.`} />
      <Metric compact label={`Volume · ${hm(m.ts)}`} value={m.volume === null ? '—' : `${plain(m.volume)} ${unit}`}
        delta={dv !== null ? `${pct(dv, 0)} vs 5-min average` : m.volume ? 'up from none in the five minutes before' : 'no volume'} tone={dv || (dv === null && m.volume) ? 'normal' : 'off'}
        icon="chart" help={`Buys plus sells in ${which}, ${dt(m.ts)} UTC; the arrow compares it with the mean volume of the five minutes before (a minute without trades counts as none).`} />
    </Metrics>
  )
}

/** What the fly would do with the newest scored minute, as colour widgets (states, not text). */
function VerdictWidgets({ d }: { d: TokenData }) {
  const r = d.latest[0]
  if (!r) return <Callout kind="info">Not scored on {d.chain_name} in the last 35 days.</Callout>
  const corr = r.score !== null && r.frozen_score !== null ? r.score - r.frozen_score : null
  const gap = r.score !== null && r.line !== null ? r.score - r.line : null
  let verdict: [string, WidgetColor, string]
  if (d.teacher_gate) {
    verdict = r.teacher_allow ? (corr !== null && corr < -d.learned_veto ? ['Vetoes', 'warning', 'its learning moved the score below the teacher\'s'] : ['Would buy', 'success', 'its teacher buys and it does not veto'])
      : ['Passes', 'dark', 'its teacher rejects this row']
  } else {
    verdict = gap !== null && gap >= 0 ? ['Would buy', 'success', 'at or above its own line'] : ['Passes', 'dark', 'below its own line']
  }
  const outcome: [string, WidgetColor] = r.label !== null ? [pct(r.label), r.label >= 0 ? 'success' : 'danger'] : r.state === 'pending' ? ['Pending', 'info'] : [r.state, 'dark']
  return (
    <Metrics min={320}>
      <Metric label={`Fly now · ${r.strategy}`} value={verdict[0]} delta={verdict[2]} tone="off" icon="fly" color={verdict[1]}
        help={d.teacher_gate ? `Under the teacher gate the fly buys only rows its teacher buys and drops one only when its learning moved the score more than ${d.learned_veto} below the teacher's.` : 'Ungated, the fly buys at or above its own line.'} />
      <Metric label="Its teacher" value={r.teacher_allow === null || r.teacher_allow === undefined ? 'Unknown' : r.teacher_allow ? 'Would buy' : 'Would not'}
        delta="the selector that taught it" tone="off" icon="bx-check-shield" color={r.teacher_allow ? 'success' : 'dark'}
        help="Whether the selector stack that taught the fly would trade this row on this strategy: its trigger, line and every filter." />
      <Metric label="Score vs line" value={gap === null ? '—' : `${gap >= 0 ? '+' : ''}${gap.toFixed(4)}`} delta={`line ${f4(r.line)}`} tone="off" icon="target"
        color={gap !== null && gap >= 0 ? 'success' : 'violet'} help="The fly's score minus its own buy line at the newest scored minute." />
      <Metric label="Outcome" value={outcome[0]} delta={r.label !== null ? 'realized over the hold' : `scored ${ago(r.ts)}`} tone="off" icon="pnl" color={outcome[1]}
        help="The row's realized net return once its hold ran out (entry at the next minute's open, exit at the last close within the hold, the exit cost at both ends)." />
    </Metrics>
  )
}

// ---------------------------------------------------------------- a memecoin token
function TokenBody({ chain, mint, mode, setMode }: { chain: string; mint: string; mode: PMode; setMode: (m: PMode) => void }) {
  const [showLabels, setShowLabels] = useStored('insp_labels', false)
  const [pick, setPick] = useStored<string>('insp_strategy', '')
  const s = useTopic<TokenData>('inspector.token', { chain, mint, mode })
  return (
    <Loaded state={s} height={400}>
      {(d) => {
        const keys = strategyKeys(d.series, false)
        const strat = keys.includes(pick) ? pick : keys[0]
        const rows = d.series.filter((r) => r.strategy === strat)
        const t = d.total
        const teacher = d.fly_score === 'teacher'
        const u = d.unit
        const closed = [...d.positions].filter((p) => p.status !== 'open' && p.closed_at).sort((a, b) => Date.parse(a.closed_at!) - Date.parse(b.closed_at!))
        let cum = 0
        const pnlSpark = closed.map((p) => (cum += p.realized_sol ?? 0))
        const scoreSpark = rows.slice(-120).map((r) => r.score ?? 0)
        const coincide = teacher && rows.length > 0 && rows.every((r) => r.score === null || r.frozen_score === null || Math.abs(r.score - r.frozen_score) < 1e-6)
        return (
          <>
            <div className="d-flex flex-wrap gap-2 align-items-center mb-3">
              <Pill color="violet">{d.chain_name}</Pill>
              {d.name && <span style={{ color: '#fff' }}>{d.name}</span>}
              {d.graduated_at && <span className="caption m-0">· graduated {ago(d.graduated_at)}</span>}
              <span className="ms-auto d-flex gap-2 align-items-center">
                <Copy value={d.mint} show={shortAddr(d.mint, 8, 6)} />
                <a className="btn btn-sm btn-light" href={d.explorer} target="_blank" rel="noreferrer"><Icon name="external" />{chain === 'sol' ? 'Solscan' : 'Explorer'}</a>
              </span>
            </div>
            {d.minute && <><MinuteWidgets m={d.minute} unit={u} scored={d.latest.length > 0} /><div className="mt-3" /></>}
            <VerdictWidgets d={d} />
            <div className="mt-3" />
            <WidgetGrid min={320}>
              <Widget compact label="Minutes scored" value={int(t?.n)} delta={t?.pending ? `${int(t.pending)} pending` : `${int(t?.resolved)} resolved`} tone="off" icon="count"
                spark={scoreSpark.length > 1 ? scoreSpark : null} help="Rows in fly_scored for this token (35 days are kept): one per minute a strategy's trigger fired and the token was eligible. The sparkline is its newest scores." />
              <Widget compact label="Cleared its line" value={t?.n ? `${((100 * t.cleared) / t.n).toFixed(1)}%` : '—'} delta={t ? `${int(t.cleared)} rows` : null} tone="off" icon="target"
                progress={frac(t?.cleared, t?.n)} help="Minutes its score was at or above its own buy line." />
              <Widget compact label="Teacher would buy" value={t?.teacher_known ? `${((100 * (t.teacher_allowed ?? 0)) / t.teacher_known).toFixed(1)}%` : '—'}
                delta={t?.teacher_known ? `${int(t.teacher_allowed)} of ${int(t.teacher_known)}` : null} tone="off" icon="bx-check-shield" progress={frac(t?.teacher_allowed, t?.teacher_known)}
                help="Minutes the selector that taught it would have bought this token (its trigger, line and every filter): under the teacher gate the fly buys only those." />
              <Widget compact label="Realized on its books" value={amount(d.pnl.realized, u, u === 'SOL' ? 4 : 6, true)}
                delta={d.pnl.closed ? `${d.pnl.wins} of ${d.pnl.closed} won${d.pnl.open ? ` · ${d.pnl.open} open` : ''}` : d.pnl.open ? `${d.pnl.open} open` : 'no closed trade'} tone="off"
                icon="wealth" spark={pnlSpark.length > 1 ? pnlSpark : null} help={`Closed positions in ${d.books.selector}, ${d.books.fly} and ${d.books.live}; the sparkline is the running total.`} />
            </WidgetGrid>

            <Card title="Score over time" icon="score"
              sub={teacher ? 'fly score = its teacher\'s score + what its mushroom body learned; buy line = its own, recalibrated daily'
                : 'fly score = the plastic fly; frozen score = its bootstrap (D = 0); buy line = its own, recalibrated daily'}
              actions={keys.length > 0 ? <>
                {keys.length > 1 && <Segmented options={keys} value={strat} onChange={setPick} />}
                <Toggle label="Labels" checked={showLabels} onChange={setShowLabels} help="Each resolved minute's realized net return, on the same axis." />
              </> : undefined}>
              {keys.length > 0 ? (
                <>
                  <ScoreChart rows={rows} scoreName="Fly score" teacherName={teacher ? 'Teacher score' : 'Frozen score'} labels={showLabels} yLabel="score" />
                  <Caption>
                    The newest {d.series.length} scored minutes ({rows.length} on {strat}); minutes without a point were not scored (not eligible, or no trigger fired).
                    {coincide && ' Its mushroom body has learned nothing on these minutes, so the fly score lies exactly on the teacher score.'}
                  </Caption>
                  <Expander title={`Scored minutes (${rows.length})`} icon="history">
                    <DataTable rows={[...rows].reverse()} filename={`fly-scored-${mint.slice(0, 8)}`} cols={[
                      { key: 'ts', label: 'minute', render: (r) => dt(r.ts), value: (r) => r.ts },
                      { key: 'score', label: 'fly', num: true, render: (r) => f4(r.score) },
                      { key: 'frozen_score', label: teacher ? 'teacher' : 'frozen', num: true, render: (r) => f4(r.frozen_score) },
                      { key: 'line', label: 'line', num: true, render: (r) => f4(r.line) },
                      { key: 'gap', label: 'vs line', num: true, value: (r) => (r.score !== null && r.line !== null ? r.score - r.line : null),
                        render: (r) => (r.score !== null && r.line !== null ? <span className={tone(r.score - r.line)}>{(r.score - r.line).toFixed(4)}</span> : '—') },
                      { key: 'teacher_allow', label: 'teacher buys', render: (r) => (r.teacher_allow === null || r.teacher_allow === undefined ? '—' : r.teacher_allow ? 'yes' : 'no') },
                      { key: 'state', label: 'outcome', render: (r) => <Pill color={STATE_COLOR[r.state] ?? 'gray'}>{r.state}</Pill> },
                      { key: 'label', label: 'label', num: true, render: (r) => <span className={tone(r.label)}>{pct(r.label)}</span> },
                    ]} />
                  </Expander>
                  <LatestFigures r={rows[rows.length - 1]} teacher={teacher ? 'teacher score' : 'frozen score'} />
                </>
              ) : <Empty>Not scored on {d.chain_name} in the last 35 days.</Empty>}
            </Card>

            <div className="row g-3 mb-4">
              <div className="col-12 col-md-6"><ClearedCard t={t} /></div>
              <div className="col-12 col-md-6">
                <Card title="Agrees with its teacher" icon="bx-check-shield" className="h-100">
                  {t?.teacher_known ? (
                    <>
                      <RadialGauge value={frac(t.teacher_allowed, t.teacher_known)} label="teacher would buy" height={220} color="#ff2fb9" />
                      <KVList items={[
                        { label: 'Mean label · teacher buys', value: <span className={tone(t.label_allowed)}>{pct(t.label_allowed)} <span className="text-muted2">({int(t.n_label_allowed)})</span></span> },
                        { label: 'Mean label · every minute', value: <span className={tone(t.label_all)}>{pct(t.label_all)}</span> },
                        { label: teacher ? 'Mean learned correction' : 'Mean learned change', value: t.mean_correction === null ? '—' : t.mean_correction.toFixed(5) },
                      ]} />
                    </>
                  ) : <Empty>The teacher's decision was not recorded for these minutes.</Empty>}
                </Card>
              </div>
            </div>

            <Card title="Score now vs line" icon="bx-message-square-detail"><Md text={d.explain} /></Card>

            <StrategyCard stats={d.stats} />

            <Card title="Decisions" icon="trades" sub={`every buy, sell and pick not taken on ${d.books.selector} (the selector), ${d.books.fly} (the fly) and ${d.books.live} (the bot wallet)`}>
              <DecisionTable rows={d.decisions} unit={u} />
            </Card>

            <Card title="Positions" icon="wallet">
              <DataTable rows={d.positions} filename="inspector-positions" empty="No positions in this token." cols={[
                { key: 'book', label: 'book' },
                { key: 'status', label: 'status', render: (r) => <Pill color={r.status === 'open' ? 'blue' : 'gray'}>{r.status}</Pill> },
                { key: 'opened_at', label: 'opened', render: (r) => dt(r.opened_at), value: (r) => r.opened_at },
                { key: 'held_min', label: 'held (min)', num: true, render: (r) => num(r.held_min, 0) },
                { key: 'strategy', label: 'strategy' },
                { key: 'cost_sol', label: `cost (${u})`, num: true, render: (r) => num(r.cost_sol, u === 'SOL' ? 4 : 6) },
                { key: 'realized_sol', label: `P&L (${u})`, num: true, render: (r) => <span className={tone(r.realized_sol)}>{num(r.realized_sol, u === 'SOL' ? 4 : 6, true)}</span> },
                { key: 'return', label: 'return', num: true, render: (r) => <span className={tone(r.return)}>{pct(r.return)}{r.status === 'open' ? ' (gross, marked)' : ''}</span> },
                { key: 'forced_exit_kind', label: 'exit', render: (r) => (r.status === 'open' ? '—' : r.forced_exit_kind || 'time') },
              ]} />
            </Card>

            <Card title="Fills" icon="receipt">
              <DataTable rows={d.fills} filename="inspector-fills" empty="No fills in this token." cols={[
                { key: 'ts', label: 'time', render: (r) => dt(r.ts, true), value: (r) => r.ts },
                { key: 'book', label: 'book' },
                { key: 'side', label: 'side', render: (r) => <Pill color={r.side === 'buy' ? 'green' : 'magenta'}>{r.side}</Pill> },
                { key: 'amount', label: u, num: true, render: (r) => <span className={tone(r.amount)}>{num(r.amount, u === 'SOL' ? 4 : 6, true)}</span> },
                { key: 'price_sol', label: `price (${u})`, num: true, render: (r) => (r.price_sol === null ? '—' : r.price_sol.toExponential(3)) },
                { key: 'fee', label: `fee (${u})`, num: true, render: (r) => num(r.fee, 5) },
                { key: 'source', label: 'source' },
                { key: 'signature', label: 'signature', render: (r) => (r.signature ? (r.tx_url ? <a href={r.tx_url} target="_blank" rel="noreferrer">{shortAddr(r.signature, 8, 6)}</a> : shortAddr(r.signature, 8, 6)) : '—') },
              ]} />
            </Card>

            <PathwaysCard p={d.pathways} mode={mode} setMode={setMode} />
          </>
        )
      }}
    </Loaded>
  )
}

function PathwaysCard({ p, mode, setMode }: { p: Pathways; mode: PMode; setMode: (m: PMode) => void }) {
  const max = Math.max(1e-12, ...p.items.map((i) => Math.abs(i.ratio)))
  return (
    <Card title="What its mushroom body learned" icon="neurology"
      sub="the largest learned changes of Kenyon cell → output neuron (KC→MBON) synapses, relative to each synapse's bootstrap weight; dopamine teaches them from realized returns"
      actions={<Segmented options={['bootstrap', '24h'] as const} value={mode} onChange={setMode} labels={{ bootstrap: 'since bootstrap', '24h': 'last 24 h' }} />}>
      <Callout kind="info">These pathways are fly-wide: one mushroom body learns from every token on both chains, and per-token Kenyon-cell activity is not recorded, so this cannot say which of them this token lit.</Callout>
      <div className="row g-3 align-items-start">
        <div className="col-12 col-md-4">
          <KVList items={[
            { label: 'Fly bootstrap', value: p.bootstrap ? `#${p.bootstrap}` : '—' },
            { label: 'Synapses changed', value: `${int(p.changed)}${p.n_pairs ? ` of ${int(p.n_pairs)}` : ''}` },
            ...Object.entries(p.drift).map(([k, v]) => ({ label: `${k} drift`, value: `${(v * 100).toFixed(2)}%` })),
          ]} />
          <Caption>Approach MBONs push a score up, avoid MBONs push it down; changes are counted {p.reference ? `since ${p.reference}` : 'since bootstrap'}.</Caption>
        </div>
        <div className="col-12 col-md-8">
          {p.note && <Caption>{p.note}</Caption>}
          {p.items.length > 0 ? (
            <ProgressList max={max} items={p.items.map((r) => ({
              label: <span className="d-inline-flex align-items-center gap-2" style={{ flexWrap: 'wrap' }}>
                <Pill color={r.valence === 'approach' ? 'green' : r.valence === 'avoid' ? 'red' : 'gray'}>{r.valence}</Pill>
                <span>{r.strategy}</span><span className="text-muted2 mono" style={{ fontSize: 12 }}>{r.kc_type ?? 'KC'} #{r.kc} → {r.mbon_type ?? 'MBON'} #{r.mbon}</span>
              </span>,
              value: <span className={tone(r.ratio)}>{r.ratio >= 0 ? '+' : ''}{(r.ratio * 100).toFixed(3)}%</span>, frac: Math.abs(r.ratio),
            }))} />
          ) : !p.note && <Empty>No KC→MBON synapse has changed yet.</Empty>}
        </div>
      </div>
      {p.items.length > 0 && (
        <Expander title={`Pathway numbers (top ${p.items.length})`} icon="bx-table">
          <DataTable rows={p.items} filename="learned-pathways" search={false} cols={[
            { key: 'strategy', label: 'strategy' },
            { key: 'valence', label: 'valence', render: (r) => <Pill color={r.valence === 'approach' ? 'green' : r.valence === 'avoid' ? 'red' : 'gray'}>{r.valence}</Pill> },
            { key: 'kc', label: 'Kenyon cell', render: (r) => `${r.kc_type ?? 'KC'} #${r.kc}`, value: (r) => r.kc },
            { key: 'mbon', label: 'output neuron', render: (r) => `${r.mbon_type ?? 'MBON'} #${r.mbon}`, value: (r) => r.mbon },
            { key: 'delta', label: 'Δ weight', num: true, render: (r) => <span className={tone(r.delta)}>{r.delta.toExponential(2)}</span> },
            { key: 'ratio', label: 'Δ / bootstrap', num: true, render: (r) => <span className={tone(r.ratio)}>{(r.ratio * 100).toFixed(3)}%</span> },
          ]} />
        </Expander>
      )}
    </Card>
  )
}

// ---------------------------------------------------------------- a Kalshi market
function MarketBody({ ticker }: { ticker: string }) {
  const [pick, setPick] = useStored<string>('insp_kstrategy', '')
  const [showLabels, setShowLabels] = useStored('insp_labels', false)
  const s = useTopic<MarketData>('inspector.market', { ticker })
  return (
    <Loaded state={s} height={400}>
      {(d) => {
        const keys = strategyKeys(d.series, true)
        const k = keys.includes(pick) ? pick : keys[0]
        const rows = d.series.filter((r) => `${r.strategy} · ${String(r.side).toUpperCase()}` === k)
        const m = d.market, t = d.total
        const closeH = m?.close_time ? (Date.parse(m.close_time) - Date.now()) / 3.6e6 : null
        const closed = [...d.positions].filter((p) => p.status !== 'open' && p.closed_at).sort((a, b) => Date.parse(a.closed_at!) - Date.parse(b.closed_at!))
        let cum = 0
        const pnlSpark = closed.map((p) => (cum += (p.realized_cents ?? 0) / 100))
        const last = d.latest[0]
        const gap = last && last.score !== null && last.line !== null ? last.score - last.line : null
        return (
          <>
            {!d.enabled && <Callout kind="info">Kalshi is not set up on this install (no KALSHI_API_KEY_ID): showing what the database holds.</Callout>}
            {m ? (
              <div className="d-flex flex-wrap gap-2 align-items-center mb-3">
                {m.category && <Pill color="violet">{m.category}</Pill>}
                <span style={{ color: '#fff' }}>{m.event_title}{m.yes_sub_title ? ` — ${m.yes_sub_title}` : ''}</span>
                {m.series_title && <span className="caption m-0">· series {m.series_title}</span>}
                <span className="ms-auto"><Copy value={d.ticker} /></span>
              </div>
            ) : <Callout kind="warning">This ticker is not in kalshi_markets (never streamed or seeded on this install).</Callout>}
            <Metrics min={320}>
              <Metric label="Market" value={m?.status ?? 'unknown'} delta={m?.result ? `result: ${m.result.toUpperCase()}` : m?.settlement_ts ? `settled ${ago(m.settlement_ts)}` : 'not settled'} tone="off"
                icon="bx-store" color={m?.status === 'active' ? 'success' : m?.result ? 'info' : 'dark'} />
              <Metric label="Closes" value={closeH !== null && closeH > 0 ? `in ${closeH.toFixed(0)} h` : m?.close_time ? 'closed' : '—'} delta={m?.close_time ? `${dt(m.close_time)} UTC` : null} tone="off"
                icon="time" color={closeH !== null && closeH > 0 ? (closeH < 6 ? 'warning' : 'info') : 'dark'} />
              <Metric label={last ? `Edge now · ${last.strategy} ${String(last.side).toUpperCase()}` : 'Edge now'} value={last ? f4(last.score) : '—'}
                delta={gap !== null ? `${gap >= 0 ? '+' : ''}${gap.toFixed(4)} vs line` : 'not scored'} tone="off" icon="target" color={!last ? 'dark' : gap !== null && gap >= 0 ? 'success' : 'violet'}
                help="The visual fly's probability that the side wins minus the arm's fee-inclusive price, at the newest scored minute; it takes the row at or above its edge line." />
              <Metric label="Settled P&L" value={usd(d.pnl.realized_usd, 2, true)} delta={d.pnl.closed ? `${d.pnl.wins} of ${d.pnl.closed} won${d.pnl.open ? ` · ${d.pnl.open} open` : ''}` : d.pnl.open ? `${d.pnl.open} open` : 'no position'}
                tone="off" icon="wealth" color={d.pnl.closed ? (d.pnl.realized_usd >= 0 ? 'success' : 'danger') : 'dark'} spark={pnlSpark.length > 1 ? pnlSpark : null} />
            </Metrics>
            <div className="mt-3" />
            <Card title="Edge over time" icon="score" sub="edge = the fly's probability minus the fee-inclusive price; frozen edge = its bootstrap's (D = 0); line = its edge line, recalibrated daily"
              actions={keys.length > 0 ? <>
                {keys.length > 1 && <Segmented options={keys} value={k} onChange={setPick} />}
                <Toggle label="Labels" checked={showLabels} onChange={setShowLabels} />
              </> : undefined}>
              {keys.length > 0 ? (
                <>
                  <ScoreChart rows={rows} scoreName="Fly edge" teacherName="Frozen edge" labels={showLabels} yLabel="edge" />
                  <LatestFigures r={rows[rows.length - 1]} teacher="frozen edge" />
                </>
              ) : <Empty>The visual fly has not scored this market in the last 35 days.</Empty>}
            </Card>
            <div className="row g-3 mb-4">
              <div className="col-12 col-md-6"><ClearedCard t={t} title="Cleared its edge line" /></div>
              <div className="col-12 col-md-6"><Card title="Score now vs line" icon="bx-message-square-detail" className="h-100"><Md text={d.explain} /></Card></div>
            </div>
            <StrategyCard stats={d.stats} kalshi />
            <Card title="Decisions" icon="trades"><DecisionTable rows={d.decisions} unit="USD" kalshi /></Card>
            <Card title="Orders" icon="receipt">
              <DataTable rows={d.orders} filename="inspector-kalshi-orders" empty="No orders in this market." cols={[
                { key: 'ts', label: 'time', render: (r) => dt(r.ts, true), value: (r) => r.ts },
                { key: 'book', label: 'book', render: (r) => d.books[r.book] ?? r.book },
                { key: 'status', label: 'status', render: (r) => <Pill color={r.status === 'resting' ? 'blue' : r.status === 'executed' ? 'green' : 'gray'}>{r.status}</Pill> },
                { key: 'side', label: 'side', render: (r) => r.side.toUpperCase() },
                { key: 'price_cents', label: 'price (c)', num: true },
                { key: 'count', label: 'count', num: true, render: (r) => num(r.count, 0) },
                { key: 'fill_count', label: 'filled', num: true, render: (r) => num(r.fill_count, 0) },
                { key: 'tif', label: 'type', value: (r) => `${r.post_only ? 'post-only ' : ''}${r.tif}` },
                { key: 'expiration_ts', label: 'expires', render: (r) => dt(r.expiration_ts), value: (r) => r.expiration_ts },
                { key: 'strategy', label: 'strategy' },
                { key: 'error', label: 'error', wrap: true },
              ]} />
            </Card>
            <Card title="Positions" icon="wallet">
              <DataTable rows={d.positions} filename="inspector-kalshi-positions" empty="No positions in this market." cols={[
                { key: 'book', label: 'book', render: (r) => d.books[r.book] ?? r.book },
                { key: 'status', label: 'status', render: (r) => <Pill color={r.status === 'open' ? 'blue' : 'gray'}>{r.status}</Pill> },
                { key: 'side', label: 'side', render: (r) => r.side.toUpperCase() },
                { key: 'strategy', label: 'strategy' },
                { key: 'contracts', label: 'contracts', num: true, render: (r) => num(r.contracts, 0) },
                { key: 'avg_price_cents', label: 'paid (c)', num: true, render: (r) => num(r.avg_price_cents, 1) },
                { key: 'last_mark_cents', label: 'now (c)', num: true, render: (r) => num(r.last_mark_cents, 1) },
                { key: 'realized_cents', label: 'P&L ($)', num: true, render: (r) => <span className={tone(r.realized_cents)}>{r.realized_cents === null ? '—' : usd(r.realized_cents / 100, 2, true)}</span> },
                { key: 'return', label: 'return', num: true, render: (r) => <span className={tone(r.return)}>{pct(r.return)}</span> },
                { key: 'result', label: 'result' },
              ]} />
            </Card>
            <Card title="Fills" icon="receipt">
              <DataTable rows={d.fills} filename="inspector-kalshi-fills" empty="No exchange fills (paper books have none: their fills are the positions themselves)." cols={[
                { key: 'ts', label: 'time', render: (r) => dt(r.ts, true), value: (r) => r.ts },
                { key: 'book', label: 'book', render: (r) => (r.book ? d.books[r.book] ?? r.book : 'live') },
                { key: 'side', label: 'side', render: (r) => r.side.toUpperCase() },
                { key: 'action', label: 'action' },
                { key: 'price_cents', label: 'price (c)', num: true },
                { key: 'count', label: 'count', num: true, render: (r) => num(r.count, 0) },
                { key: 'fee_cents', label: 'fee ($)', num: true, render: (r) => usd((r.fee_cents ?? 0) / 100) },
                { key: 'is_taker', label: 'taker', render: (r) => (r.is_taker ? 'yes' : 'no') },
              ]} />
            </Card>
          </>
        )
      }}
    </Loaded>
  )
}

export function InspectorDrawer({ target, onClose }: { target: InspectTarget; onClose: () => void }) {
  const { meta } = useApp()
  const [mode, setMode] = useStored<PMode>('insp_pmode', 'bootstrap')
  useEffect(() => {
    const k = (e: KeyboardEvent) => e.key === 'Escape' && onClose()
    window.addEventListener('keydown', k)
    return () => window.removeEventListener('keydown', k)
  }, [onClose])
  const title = target.kind === 'token' ? <TokenTitle chain={target.chain} mint={target.mint} mode={mode} /> : <MarketTitle ticker={target.ticker} />
  const chainName = target.kind === 'token' ? meta?.markets.find((m) => m.chain === target.chain)?.name ?? target.chain : 'Kalshi'
  return (
    <>
      <div className="drawer-backdrop" onClick={onClose} />
      <aside className="drawer" role="dialog" aria-modal="true" aria-label="Fly decision inspector">
        <div className="head">
          <div style={{ width: 44, height: 44, borderRadius: 10, flexShrink: 0, display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: 23, color: '#fff',
            background: target.kind === 'token' ? 'linear-gradient(135deg, #ff49c4, #d4168f)' : 'linear-gradient(135deg, #2f8fff, #1559d6)' }}>
            <Icon name={target.kind === 'token' ? 'fly' : 'visibility'} />
          </div>
          <div className="d-flex flex-column" style={{ minWidth: 0 }}>
            <span className="fly-head" style={{ fontWeight: 600, fontSize: 17, whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}>{title}</span>
            <span style={{ fontSize: 12.5, color: 'rgb(255 255 255 / 65%)' }}>Fly decision inspector · {chainName}{target.kind === 'market' ? ` · ${target.ticker}` : ''}</span>
          </div>
          <button type="button" className="close" aria-label="Close" onClick={onClose}>×</button>
        </div>
        <div className="body">
          {target.kind === 'token' ? <TokenBody key={`${target.chain}:${target.mint}`} chain={target.chain} mint={target.mint} mode={mode} setMode={setMode} />
            : <MarketBody key={target.ticker} ticker={target.ticker} />}
        </div>
      </aside>
    </>
  )
}

/** The drawer's title: the token's label once the topic answered (the same subscription as the body's, shared). */
function TokenTitle({ chain, mint, mode }: { chain: string; mint: string; mode: PMode }) {
  const { data } = useTopic<TokenData>('inspector.token', { chain, mint, mode })
  return <>{data?.label ?? shortAddr(mint, 6, 4)}</>
}

function MarketTitle({ ticker }: { ticker: string }) {
  const { data } = useTopic<MarketData>('inspector.market', { ticker })
  return <>{data?.market?.title ?? ticker}</>
}
