// Model & training (memecoins) as a Dashtreme dashboard: the model at work (KPI widgets, its backtest by half, win rate,
// sizing bands, strategy stack), the automatic retraining pipeline (the selector every 7 days until the fly holds its seat;
// the fly bootstrapped once), the day-by-day backtest, the plastic fly and its replay, every saved model and the trainer's
// log (topics and actions in fly_trader/ui/api/model.py). Like the Streamlit page, it shows the Solana side.
import type { ApexOptions } from 'apexcharts'
import type { ReactNode } from 'react'
import { useNow, useTopic, type TopicState } from '../lib/hooks'
import { ago, dt, int, num, pct, pctPts } from '../lib/format'
import { Callout, Caption, Card, Empty, Expander, Help, KVList, Loaded, Md, Metric, Metrics, Progress, RowGroup, Widget, WidgetGrid, type WidgetColor } from '../components/ui'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { ApexChart, BarChart, RadialGauge } from '../components/Charts'
import { ActionButton } from '../components/feedback'
import { PageHead } from '../components/ChainPicker'

type Wf = { n: number | null; days: number | null; mean: number | null; win: number | null; pf: number | null; days_positive: number | null; weeks: number | null; weeks_pos: number | null }
type Band = { band: string; n: number | null; mean: number; win: number; bet: number }
type Sizing = { caption: string; bands: Band[]; bankroll: string | null }
type Component = {
  component: string | null; kept: boolean | null; why: string | null; win_sel: number | null; win_eval: number | null; pf_eval: number | null
  total_eval: number | null; n_eval: number | null; trials: number | null; best: string; params: string
}
type Strategy = { strategy: string; hold_min: number | null; line: number; trigger: string; n_eval: number | null; win_eval: number | null; pf_eval: number | null }
type Stack = { fallback: boolean; components: Component[]; strategies: Strategy[] }
type Half = { name: string; n: number | null; mean: number; random_mean: number | null; win: number | null; pf: number | null; weeks: number | null; weeks_pos: number | null; days: string[] | null }
type AtWork = {
  id: number; current: boolean; ts?: string; since?: string; first_day?: string | null; trained_through?: string | null; threshold?: number
  horizon_min?: number | null; min_age_h?: number | null; wf?: Wf; random_mean?: number | null; backtest_line?: string; sizing?: Sizing; stack?: Stack | null; halves?: Half[]
}
type LoadedData = { model: AtWork | null; latest: { id: number; ts: string; deployable: boolean; deploy_reason: string | null } | null }
type Retraining = {
  handover: boolean; interval_days: number; alive: boolean; running: boolean; stage: string | null; status: string; last_run_at: string | null; next_run: string
  latest: { id: number; deployable: boolean; backtest_line: string } | null; progress: { value: number; text: string } | null
  waiting: string | null; failed: string | null
}
type DayRow = { day: string | null; half: string | null; ic: number | null; n: number | null; model: number | null; random: number | null; winners: number | null; pf: number | null }
type Results = { hidden: boolean; rows: DayRow[] }
type PerStrategy = {
  strategy: string; ic: number | null; drift: number | null; mean_plastic: number | null; mean_frozen: number | null; n_plastic: number | null
  n_frozen: number | null; triggers: string | null; line: number | null; frozen_line: number | null
}
type FlyData = {
  running: boolean; frozen: boolean; runner_alive: boolean
  replay: {
    passed: boolean; reason: string | null; n: number | null; mean: number | null; random_mean: number | null; frozen_mean: number | null
    days_positive: number | null; days: number | null; alpha: string; half_life: string; caption: string
  } | null
  live: {
    line: number | null; frozen_line: number | null; drift: number | null; pending: number | null; ic: number | null; shadow_plastic: number | null
    shadow_frozen: number | null; device: string | null; drift_max: number; per_strategy: PerStrategy[]
  } | null
  detail: string | null
  updates: { hour: string; n: number | null; mean_abs_delta: number | null; drift: number | null; ic: number | null }[]
  rollbacks: { ts: string; reason: string | null; from_snapshot: number | null; to_snapshot: number | null }[]
  bootstrap: { id: number; ts: string; ok: boolean; text: string } | null
}
type Saved = {
  id: number; model: string; type: string; saved: string; data: string; trades: number | null; mean: number | null; random: number | null; pf: number | null
  deployed: boolean | null; in_use: boolean
}

const dash = (v: unknown) => (v === null || v === undefined ? '—' : String(v))
const yes = (v: boolean | null | undefined): ReactNode =>
  v === null || v === undefined ? null : v ? <i className="bx bx-check text-pos" aria-label="yes" /> : <i className="bx bx-x text-muted2" aria-label="no" />
const signed = (v: number | null | undefined, text: string) => <span className={v == null || v === 0 ? '' : v > 0 ? 'text-pos' : 'text-neg'}>{text}</span>
const bySign = (v: number | null | undefined, pos: WidgetColor = 'success'): WidgetColor => (v == null ? 'dark' : v > 0 ? pos : v < 0 ? 'danger' : 'dark')

/** A small static table (a few rows: no paging, search or export needed). */
function MiniTable<R>({ rows, cols }: { rows: R[]; cols: { label: string; help?: string; num?: boolean; cell: (r: R) => ReactNode }[] }) {
  return (
    <div className="ftable-wrap">
      <table className="ftable">
        <thead><tr>{cols.map((c) => <th key={c.label} className={c.num ? 'num' : ''}>{c.label}<Help text={c.help} /></th>)}</tr></thead>
        <tbody>{rows.map((r, i) => <tr key={i}>{cols.map((c) => <td key={c.label} className={c.num ? 'num' : ''}>{c.cell(r) ?? '—'}</td>)}</tr>)}</tbody>
      </table>
    </div>
  )
}

export default function Model() {
  const loaded = useTopic<LoadedData>('model.loaded')
  const results = useTopic<Results>('model.results')
  return (
    <>
      <PageHead title="Model & training" icon="bx-analyse" chain={false} />
      <ModelAtWork s={loaded} days={results.data} />
      <div className="row g-4">
        <div className="col-12 col-xl-7"><Retrain /></div>
        <div className="col-12 col-xl-5"><HowRetraining /></div>
      </div>
      <DayByDay s={results} />
      <PlasticFly />
      <History />
    </>
  )
}

// ---------------------------------------------------------------- Model at work
function ModelAtWork({ s, days }: { s: TopicState<LoadedData>; days?: Results }) {
  const now = useNow(10000)
  return (
    <Loaded state={s} height={130}>
      {({ model: m, latest }) => {
        const next = m && latest && latest.id > m.id && (latest.deployable
          ? <Callout kind="info" icon="upgrade">Model #{latest.id} qualified and replaces #{m.id} within 10 minutes, without a restart.</Callout>
          : <Caption>Model #{latest.id} ({ago(latest.ts, now)}) was not put to work: {latest.deploy_reason || 'it did not qualify'}.</Caption>)
        if (!m || !m.current) {
          return (
            <Card title="Model at work" icon="psychology">
              {m ? <Callout kind="warning">Model #{m.id} is in use but was trained on outdated data (before the data fixes of 2026-09-14). Its backtest is not valid and is hidden.</Callout>
                : !latest ? <Callout kind="info">No model has been trained on the current data yet, so nothing is trading. The automatic retraining below runs as soon as the data is ready.</Callout>
                  : <Callout kind="info" icon="block">No model is at work. The latest, #{latest.id}, was not put to work: {latest.deploy_reason || 'it did not qualify'}.</Callout>}
              {next}
            </Card>
          )
        }
        const wf: Partial<Wf> = m.wf ?? {}
        const byWeek = wf.days == null && wf.weeks != null          // the strategy-stack selectors judge their evaluation half by week
        const posN = byWeek ? wf.weeks_pos : wf.days_positive, posD = byWeek ? wf.weeks : wf.days
        const daySpark = (days?.rows ?? []).map((r) => r.model).filter((v): v is number => v !== null)
        return (
          <>
            <WidgetGrid min={190}>
              <Widget label="Model at work" value={`#${m.id}`} icon="psychology" color="violet" delta={`at work since ${ago(m.since, now)}`} tone="off"
                help={`Gradient-boosted selector · trained on ${m.first_day ?? '—'} → ${m.trained_through ?? '—'} · saved ${ago(m.ts, now)} · at work since ${ago(m.since, now)}`} />
              <Widget label="Buys when predicted return ≥" value={pct(m.threshold ?? 0, 1)} icon="target"
                help={`Each minute the model predicts every eligible token's net return over the hold after fees and price impact. It buys tokens at least ${dash(m.min_age_h)} h past graduation (or older than the archive) whose prediction clears this line.`} />
              <Widget label="Holds for" value={`${dash(m.horizon_min)} min`} icon="time" help="Each position is sold this long after the buy." />
              <Widget label="Test trades" value={wf.n == null ? '—' : int(wf.n)} icon="count"
                help={`Trades on the backtest's evaluation half (${byWeek ? `${dash(wf.weeks)} weeks` : `${dash(wf.days)} days`}), each traded by a model that never saw that day, at a buy line chosen on the earlier selection half.`} />
              <Widget label="Average per trade" value={pct(wf.mean)} icon="pnl" delta={m.random_mean != null ? `random ${pct(m.random_mean)}` : null} tone="off"
                color={bySign(wf.mean, 'teal')} spark={daySpark.length > 1 ? daySpark : null}
                help="Net return per trade on the test days, after fees and price impact. 'random' is the same number of random buys under the same rules." />
              <Widget label="Profit factor" value={wf.pf == null ? '∞' : num(wf.pf, 2)} icon="up" help="Total gains divided by total losses on the test trades. Above 1 makes money." />
            </WidgetGrid>
            <div className="row g-4">
              <div className="col-12 col-xl-8">
                <Card title="Backtest by half" icon="chart"
                  sub={`Model #${m.id} · gradient-boosted selector · trained on ${m.first_day ?? '—'} → ${m.trained_through ?? '—'} · saved ${ago(m.ts, now)} · at work since ${ago(m.since, now)}`}>
                  <Halves halves={m.halves ?? []} />
                  <Md text={m.backtest_line} className="mt-2" />
                  {next}
                </Card>
              </div>
              <div className="col-12 col-xl-4">
                <Card title="Winning trades" icon="target" sub="evaluation half">
                  <RadialGauge value={wf.win ?? 0} label="winning trades" display={pct(wf.win, 0, false)} height={250} />
                  <Caption>Share of test trades that made money.</Caption>
                  <RowGroup items={[
                    { label: byWeek ? 'profitable test weeks' : 'profitable test days', value: `${dash(posN)} of ${dash(posD)}` },
                    { label: 'random picks / trade', value: signed(m.random_mean, pct(m.random_mean)) },
                  ]} />
                  {posD ? <Progress value={(posN ?? 0) / posD} text={byWeek ? 'Test weeks whose trades made money after costs.' : 'Test days whose average trade made money after costs.'} /> : null}
                </Card>
              </div>
            </div>
            <div className="row g-4">
              <div className="col-12 col-xl-7">{m.sizing && <SizingCard sz={m.sizing} />}</div>
              <div className="col-12 col-xl-5">{m.stack && m.stack.strategies.length > 0 && <StrategiesCard st={m.stack} />}</div>
            </div>
            {m.stack && <ComponentsCard st={m.stack} />}
          </>
        )
      }}
    </Loaded>
  )
}

function Halves({ halves }: { halves: Half[] }) {
  if (!halves.length) return <Empty>No backtest halves recorded with this model.</Empty>
  const series = [{ name: 'model % / trade', data: halves.map((h) => h.mean * 100) }]
  if (halves.some((h) => h.random_mean != null)) series.push({ name: 'random % / trade', data: halves.map((h) => (h.random_mean == null ? null : h.random_mean * 100)) as number[] })
  return (
    <>
      <BarChart categories={halves.map((h) => h.name)} series={series} height={250} yLabel="% per trade" yFormat={(v) => `${v.toFixed(1)}%`} />
      <MiniTable rows={halves} cols={[
        { label: 'part', cell: (h) => h.name },
        { label: 'trades', num: true, cell: (h) => int(h.n) },
        { label: 'model / trade', num: true, cell: (h) => signed(h.mean, pct(h.mean)) },
        { label: 'random / trade', num: true, cell: (h) => signed(h.random_mean, pct(h.random_mean)) },
        { label: 'winners', num: true, cell: (h) => pct(h.win, 0, false) },
        { label: 'profit factor', num: true, cell: (h) => num(h.pf, 2) },
        { label: 'profitable weeks', num: true, cell: (h) => (h.weeks == null ? '—' : `${dash(h.weeks_pos)} of ${h.weeks}`) },
        { label: 'days', cell: (h) => (h.days ? h.days.map((x) => x.slice(5)).join(' → ') : '—') },
      ]} />
      <Caption>Selection chose the settings and the buy line; evaluation judged them on later weeks; the holdout's days were seen by no fit at all (overall and per chain).</Caption>
    </>
  )
}

function SizingCard({ sz }: { sz: Sizing }) {
  return (
    <Card title="Position sizing" icon="layers">
      <Caption>{sz.caption}</Caption>
      {sz.bands.length > 0 && (
        <>
          <BarChart categories={sz.bands.map((b) => b.band)} height={220} yFormat={(v) => `${v.toFixed(1)}%`}
            series={[{ name: 'bet (share of bankroll)', data: sz.bands.map((b) => b.bet) }, { name: 'average per trade', data: sz.bands.map((b) => b.mean) }]} />
          <MiniTable rows={sz.bands} cols={[
            { label: 'score above the line', cell: (b) => b.band },
            { label: 'test trades', num: true, cell: (b) => int(b.n) },
            { label: 'average per trade', num: true, cell: (b) => signed(b.mean, pctPts(b.mean, 2)) },
            { label: 'winning trades', num: true, cell: (b) => pctPts(b.win, 0, false) },
            { label: 'bet (share of bankroll)', num: true, help: '0 % means that band has no edge: no buy.', cell: (b) => (b.bet === 0 ? <span className="text-muted2">0.0% · no buy</span> : pctPts(b.bet, 1, false)) },
          ]} />
        </>
      )}
      {sz.bankroll && <Caption>{sz.bankroll}</Caption>}
    </Card>
  )
}

function StrategiesCard({ st }: { st: Stack }) {
  return (
    <Card title="Strategies" icon="category" sub="what the selector trades, judged on the evaluation half">
      <Metrics min={200}>
        {st.strategies.map((s) => (
          <Metric key={s.strategy} label={`${s.strategy} · buy line`} value={pctPts(s.line, 2)} icon="target" delta={`hold ${dash(s.hold_min)} min`} tone="off"
            help="The strategy's buy line; the bar is its share of winning trades on the evaluation half." progress={s.win_eval == null ? null : s.win_eval / 100} />
        ))}
      </Metrics>
      {st.strategies.map((s) => (
        <div key={s.strategy} className="mt-3">
          <div className="fw-semibold mb-1">{s.strategy}</div>
          <KVList items={[
            { label: 'hold (min)', value: dash(s.hold_min) },
            { label: 'trades (evaluation)', value: int(s.n_eval) },
            { label: 'winning (evaluation)', value: pctPts(s.win_eval, 0, false) },
            { label: 'profit factor (evaluation)', value: num(s.pf_eval, 2) },
            { label: 'trigger', value: <code style={{ whiteSpace: 'normal', wordBreak: 'break-all' }}>{s.trigger}</code> },
          ]} />
        </div>
      ))}
    </Card>
  )
}

function ComponentsCard({ st }: { st: Stack }) {
  const kept = st.components.filter((c) => c.kept).length
  const cols: Col<Component>[] = [
    { key: 'component', label: 'component' },
    { key: 'kept', label: 'kept', render: (r) => (r.kept == null ? null : <Pill color={r.kept ? 'green' : 'gray'}>{r.kept ? 'kept' : 'dropped'}</Pill>) },
    { key: 'why', label: 'why', wrap: true, render: (r) => <span style={{ display: 'inline-block', minWidth: 240 }}>{r.why}</span> },
    { key: 'win_sel', label: 'winning (selection)', num: true, render: (r) => pctPts(r.win_sel, 0, false) },
    { key: 'win_eval', label: 'winning (evaluation)', num: true, render: (r) => pctPts(r.win_eval, 0, false) },
    { key: 'pf_eval', label: 'profit factor (evaluation)', num: true, render: (r) => num(r.pf_eval, 2) },
    { key: 'total_eval', label: 'total (evaluation)', num: true, render: (r) => signed(r.total_eval, num(r.total_eval, 2, true)),
      help: "Sum of the net returns of the evaluation half's trades (1.00 = one position's cost)." },
    { key: 'n_eval', label: 'trades (evaluation)', num: true, render: (r) => int(r.n_eval) },
    { key: 'trials', label: 'settings tried', num: true, render: (r) => int(r.trials) },
    { key: 'best', label: 'best reached', wrap: true, help: 'The most profitable setting this component tried on the selection half, whether or not it cleared the bars.',
      render: (r) => <span style={{ display: 'inline-block', minWidth: 230 }}>{r.best}</span> },
    { key: 'params', label: 'parameters', wrap: true,
      render: (r) => <code style={{ display: 'block', minWidth: 380, whiteSpace: 'normal', wordBreak: 'break-word' }}>{r.params}</code> },
  ]
  return (
    <Card title="Components" icon="bx-git-branch" sub={`every component the selector tried: ${kept} kept, ${st.components.length - kept} dropped, and why`}>
      {st.fallback && (
        <Callout kind="warning">No setting reached 65 % winning trades on both halves: this model trades the most profitable system that passes profit factor 1.3 and the deploy rule instead.</Callout>
      )}
      <DataTable rows={st.components} cols={cols} filename="selector-components" />
    </Card>
  )
}

// ---------------------------------------------------------------- Automatic retraining
function statusColor(r: Retraining): WidgetColor {
  if (!r.alive) return 'dark'
  if (r.running) return 'info'
  if (r.stage === 'failed') return 'danger'
  if (r.stage === 'waiting for data') return 'warning'
  return 'success'
}

function Retrain() {
  const s = useTopic<Retraining>('model.retraining')
  const now = useNow(5000)
  const d = s.data
  const startNote = d && !d.alive ? ' The Trainer is not running, so this also starts it.' : ''
  return (
    <Card title={d?.handover ? 'Retraining' : 'Automatic retraining'} icon="autorenew"
      sub={d ? (d.handover ? 'on request only: the fly holds the seat' : `every ${d.interval_days} days`) : undefined}
      actions={d && (
        <>
          <ActionButton label="Retrain now" icon="play" name="model.retrain" variant="primary" disabled={d.running}
            help="Runs the whole pipeline now; the 7-day schedule restarts from this run."
            confirm={{ title: 'Retrain now?', confirmLabel: 'Retrain now',
              body: `The whole pipeline runs now: the selector retrains on every day of history and is backtested day by day against random picks; the ${d.interval_days}-day schedule restarts from this run.${startNote}` }} />
          {d.running && (
            <ActionButton label="Stop" icon="stop" name="model.train_stop" variant="danger" help="Stops the current run; the schedule resumes when the trainer is started again."
              confirm={{ title: 'Stop the training run?', body: 'The current run stops; the schedule resumes when the Trainer is started again.', confirmLabel: 'Stop', danger: true }} />
          )}
        </>
      )}>
      <Loaded state={s}>
        {(r) => (
          <>
            {!r.alive && <Callout kind="warning">Automatic retraining is off because the Trainer is not running. Click Retrain now, or start the Trainer on the Processes page.</Callout>}
            <Metrics min={320}>
              <Metric label="Status" value={r.status} icon="model_training" color={statusColor(r)} />
              <Metric label="Last run" value={r.last_run_at ? ago(r.last_run_at, now) : 'never'} icon="history" />
              <Metric label="Next run" value={r.next_run} icon="calendar" />
              <Metric label="Latest selector" value={r.latest ? `#${r.latest.id}` : '—'} delta={r.latest ? (r.latest.deployable ? 'at work' : 'not at work') : null} tone="off"
                icon="psychology" color={r.latest ? (r.latest.deployable ? 'violet' : 'warning') : 'dark'} help={r.latest?.backtest_line} />
            </Metrics>
            {r.progress && <Progress value={r.progress.value} text={r.progress.text} />}
            {r.waiting && <Callout kind="info" icon="time">{r.waiting}</Callout>}
            {r.failed && <Callout kind="error">{r.failed}</Callout>}
          </>
        )}
      </Loaded>
    </Card>
  )
}

const STEPS = [
  { icon: 'bx-analyse', text: 'The selector retrains on every day of history. Every day after the first 21 is backtested by a model that never saw it, against random picks, with real fees and price impact.' },
  { icon: 'bx-up-arrow-circle', text: 'It goes to work only if that backtest made money and beat random picks; the trading engine switches to it without a restart.' },
  { icon: 'bx-brain', text: "The fly is taught by the selector once — when none exists for the current definitions, or on request — and from then on learns from the market; weekly retraining stops when the fly takes the selector's seat." },
]

function HowRetraining() {
  return (
    <Card title="How retraining works" icon="bx-list-ol">
      <ol style={{ listStyle: 'none', padding: 0, margin: 0 }}>
        {STEPS.map((st, i) => (
          <li key={i} className="d-flex gap-3 align-items-start mb-3">
            <span className="d-inline-flex align-items-center justify-content-center flex-shrink-0"
              style={{ width: 40, height: 40, borderRadius: 10, background: 'rgb(255 255 255 / 14%)', color: '#fff', fontSize: 20 }}>
              <i className={`bx ${st.icon}`} />
            </span>
            <span><strong className="me-1">{i + 1} ·</strong><span className="text-white-50">{st.text}</span></span>
          </li>
        ))}
      </ol>
    </Card>
  )
}

// ---------------------------------------------------------------- Latest selector backtest, day by day
function DayByDay({ s }: { s: TopicState<Results> }) {
  const cols: Col<DayRow>[] = [
    { key: 'day', label: 'day' },
    { key: 'half', label: 'half', help: 'selection: the days that chose the buy line; evaluation: the days that judge it.',
      render: (r) => (r.half ? <Pill color={r.half === 'evaluation' ? 'violet' : 'gray'}>{r.half}</Pill> : null) },
    { key: 'ic', label: 'IC', num: true, render: (r) => num(r.ic, 3), help: 'Rank correlation of the predictions with the realised returns (0 = no skill).' },
    { key: 'n', label: 'trades', num: true, render: (r) => int(r.n) },
    { key: 'model', label: 'model % / trade', num: true, render: (r) => signed(r.model, num(r.model, 2, true)) },
    { key: 'random', label: 'random % / trade', num: true, render: (r) => signed(r.random, num(r.random, 2, true)), help: 'Random picks, same count, costs and fills.' },
    { key: 'winners', label: 'winners %', num: true, render: (r) => num(r.winners, 0) },
    { key: 'pf', label: 'profit factor', num: true, render: (r) => num(r.pf, 2) },
  ]
  return (
    <Card title="Latest selector backtest, day by day" icon="candlestick">
      <Loaded state={s}>
        {(d) => {
          if (d.hidden) return <Empty>The last backtest ran on outdated data and is hidden.</Empty>
          if (!d.rows.length) return <Empty>No selector backtest on the current data yet.</Empty>
          const series = (['model', 'random'] as const).filter((k) => d.rows.some((r) => r[k] !== null))
            .map((k) => ({ name: `${k} % / trade`, data: d.rows.map((r) => r[k]) }))
          return (
            <>
              <BarChart categories={d.rows.map((r) => r.day ?? '')} series={series} height={260} yLabel="% per trade" />
              <DataTable rows={d.rows} cols={cols} filename="selector-backtest-days" />
              <Caption>
                Each day is traded by a model trained only on days at least two days earlier, with the paper broker's fees and price impact. The buy line is chosen on the selection
                half and judged on the evaluation half.
              </Caption>
            </>
          )
        }}
      </Loaded>
    </Card>
  )
}

// ---------------------------------------------------------------- The plastic fly
function FlyActions({ d }: { d: FlyData }) {
  return (
    <>
      {d.frozen ? (
        <ActionButton label="Resume learning" icon="play" name="model.fly_command" body={{ cmd: 'resume' }} disabled={!d.running} variant="light"
          help="Pause: the fly keeps trading with what it has learned; its mushroom body stops changing."
          confirm={{ title: "Resume the fly's learning?", body: 'Its mushroom body learns from the realized returns again. Applied at the next minute.', confirmLabel: 'Resume learning' }} />
      ) : (
        <ActionButton label="Pause learning" icon="pause" name="model.fly_command" body={{ cmd: 'pause' }} disabled={!d.running} variant="light"
          help="Pause: the fly keeps trading with what it has learned; its mushroom body stops changing."
          confirm={{ title: "Pause the fly's learning?", body: 'The fly keeps trading with what it has learned; its mushroom body stops changing. Applied at the next minute.', confirmLabel: 'Pause learning' }} />
      )}
      <ActionButton label="Roll back" icon="undo" name="model.fly_command" body={{ cmd: 'rollback' }} disabled={!d.running} variant="light"
        help="Restore the newest good hourly snapshot at least a day old (or the bootstrap). Does not count toward the automatic re-bootstrap."
        confirm={{ title: 'Roll the fly back?', confirmLabel: 'Roll back', danger: true,
          body: 'The fly restores the newest good hourly snapshot at least a day old (or the bootstrap) and loses what it learned since. Does not count toward the automatic re-bootstrap. Applied at the next minute.' }} />
      <ActionButton label="Re-bootstrap" icon="refresh" name="model.fly_rebootstrap" variant="light" help="The selector retrains and teaches a new fly; it trades at once if it qualifies."
        confirm={{ title: 'Re-bootstrap the fly?', confirmLabel: 'Re-bootstrap', body: 'The selector retrains and teaches a new fly; it trades at once if it qualifies. If the Trainer is not running, this also starts it.' }} />
      <ActionButton label="Reset fly" icon="reset" name="model.fly_reset" variant="danger" disabled={d.runner_alive}
        help={d.runner_alive ? 'Stop the trading engine first.' : undefined}
        confirm={{ title: 'Reset the plastic fly?', confirmLabel: 'Reset', danger: true,
          body: "Archive and clear the fly's paper book and everything it learned? It restarts from its bootstrap and the race starts over." }} />
    </>
  )
}

function PlasticFly() {
  const s = useTopic<FlyData>('model.fly')
  const now = useNow(10000)
  const d = s.data
  return (
    <>
      <Card title="The plastic fly" icon="neurology" sub="taught once by the selector, then learning from the market" actions={d && <FlyActions d={d} />}>
        <Loaded state={s}>
          {(f) => (
            <>
              {f.live ? <FlyLive f={f} /> : (
                <div className="d-flex align-items-center gap-3 flex-wrap">
                  <div style={{ flex: '0 0 230px' }}><Metric label="The fly" value="Not trading" icon="neurology" color="dark" /></div>
                  <Caption>{f.detail}</Caption>
                </div>
              )}
              {f.bootstrap && (
                <div className={`callout ${f.bootstrap.ok ? 'success' : 'warning'} mt-3 mb-0`}>
                  <i className={`bx ${f.bootstrap.ok ? 'bxs-check-circle' : 'bx-error'}`} aria-hidden="true" />
                  <div>Bootstrap #{f.bootstrap.id} ({ago(f.bootstrap.ts, now)}): {f.bootstrap.text}</div>
                </div>
              )}
            </>
          )}
        </Loaded>
      </Card>
      {d && <Replay f={d} />}
    </>
  )
}

function FlyLive({ f }: { f: FlyData }) {
  const l = f.live!
  const rbCols: Col<FlyData['rollbacks'][number]>[] = [
    { key: 'ts', label: 'when', render: (r) => dt(r.ts) },
    { key: 'reason', label: 'reason', wrap: true },
    { key: 'from_snapshot', label: 'from_snapshot', num: true },
    { key: 'to_snapshot', label: 'to_snapshot', num: true },
  ]
  const share = l.drift == null ? 0 : l.drift / (l.drift_max || 0.5)
  return (
    <>
      <Metrics min={220}>
        <Metric label="Learning" value={f.frozen ? 'frozen' : 'on'} icon="neurology" color={f.frozen ? 'warning' : 'success'}
          delta={f.frozen ? 'trades with what it learned' : 'learning from realized returns'} tone="off"
          help="Pause: the fly keeps trading with what it has learned; its mushroom body stops changing." />
        <Metric label="Buy line" value={pct(l.line)} delta={`shadow ${pct(l.frozen_line)}`} tone="off" icon="target" color="violet"
          help="Recalibrated every day from its own scores and the realized returns of the last 7 days." />
        <Metric label="24 h IC" value={l.ic != null ? num(l.ic, 3) : '—'} icon="pulse" color={bySign(l.ic)}
          help="Rank correlation of its scores with the realized returns; rolled back below 0." />
        <Metric label="Picks vs shadow (3 d)" value={pct(l.shadow_plastic)} delta={`shadow ${pct(l.shadow_frozen)}`} tone="off" icon="race"
          color={l.shadow_plastic == null || l.shadow_frozen == null ? 'dark' : l.shadow_plastic >= l.shadow_frozen ? 'success' : 'danger'}
          help="Average realized return of its picks vs the frozen bootstrap's picks; rolled back when it trails by more than one standard error." />
        <Metric label="Labels pending" value={l.pending == null ? '—' : int(l.pending)} icon="time" help="Scored minutes waiting for their two-hour outcome." />
        <Metric label="Device" value={l.device || '—'} icon="bx-chip" help="Where the fly runs: DEVICE in .env (auto: CUDA, else Apple MPS, else CPU)." />
      </Metrics>
      {l.per_strategy.length > 0 && (
        <>
          <Caption>It trades several strategies, so its 24 h IC, its picks vs shadow and its buy line are checked per strategy:</Caption>
          <Metrics min={220}>
            {l.per_strategy.map((p) => (
              <Metric key={p.strategy} label={`${p.strategy} · 24 h IC`} value={num(p.ic, 3)} icon="pulse" color={bySign(p.ic)}
                delta={`picks ${pct(p.mean_plastic)} (${int(p.n_plastic)}) vs shadow ${pct(p.mean_frozen)} (${int(p.n_frozen)})`} tone="off"
                help={`Buy line ${pct(p.line)} (shadow ${pct(p.frozen_line)}); drift ${pct(p.drift, 2, false)}.${p.triggers ? ` Rollback checks failing now: ${p.triggers}.` : ' No rollback check failing.'}`} />
            ))}
          </Metrics>
        </>
      )}
      <div className="row g-4 mt-1">
        <div className="col-12 col-xl-8">
          <div className="fw-semibold mb-1">Drift and IC, hour by hour</div>
          {f.updates.length > 0 ? <DriftIc updates={f.updates} /> : <Empty>No hourly updates recorded yet.</Empty>}
        </div>
        <div className="col-12 col-xl-4">
          <RadialGauge value={share} label={`of the ${pct(l.drift_max, 0, false)} rollback limit`} display={pct(l.drift, 1, false)} height={250}
            color={share > 0.8 ? '#ff4d6d' : share > 0.5 ? '#ffc107' : '#ffffff'} />
          <Caption>Drift: size of the learned KC→MBON change relative to the bootstrap weights; rolled back above 50 %.</Caption>
        </div>
      </div>
      {f.rollbacks.length > 0 && (
        <Expander title={`Rollbacks (${f.rollbacks.length})`} icon="history">
          <DataTable rows={f.rollbacks} cols={rbCols} filename="fly-rollbacks" />
        </Expander>
      )}
    </>
  )
}

/** fly_updates: drift (left axis, % of the bootstrap weights) and the hourly IC (right axis) — drift is 100x smaller than the IC, so one axis would flatten it. */
function DriftIc({ updates }: { updates: FlyData['updates'] }) {
  const options: ApexOptions = {
    chart: { type: 'area', height: 250, background: 'transparent', foreColor: 'rgba(255,255,255,0.75)', toolbar: { show: false }, zoom: { enabled: false },
      animations: { enabled: false }, parentHeightOffset: 0, fontFamily: 'Inter Variable, Inter, Roboto, system-ui, sans-serif' },
    theme: { mode: 'dark' },
    colors: ['#ffffff', '#ff2fb9'],
    stroke: { width: [3, 2], curve: 'smooth' },
    fill: { type: 'gradient', gradient: { shadeIntensity: 0, opacityFrom: 0.4, opacityTo: 0.02, stops: [0, 95] } },
    dataLabels: { enabled: false },
    grid: { borderColor: 'rgba(221,221,221,0.08)', padding: { left: 8, right: 12, top: 0 } },
    xaxis: { type: 'datetime', labels: { datetimeUTC: true, style: { fontSize: '11px' } }, axisBorder: { show: false }, axisTicks: { show: false } },
    yaxis: [
      { seriesName: 'drift', title: { text: 'drift', style: { fontWeight: 400 } }, labels: { formatter: (v: number) => pct(v, 2, false) } },
      { seriesName: 'ic', opposite: true, title: { text: 'IC', style: { fontWeight: 400 } }, labels: { formatter: (v: number) => (Number.isFinite(v) ? v.toFixed(2) : '') } },
    ],
    legend: { position: 'top', horizontalAlign: 'right', labels: { colors: '#fff' } },
    tooltip: { theme: 'dark', shared: true, x: { format: 'MMM d HH:mm' } },
    markers: { size: 0, hover: { size: 4 } },
  }
  const series = [
    { name: 'drift', data: updates.map((u) => ({ x: Date.parse(u.hour), y: u.drift })) },
    { name: 'ic', data: updates.map((u) => ({ x: Date.parse(u.hour), y: u.ic })) },
  ]
  return <ApexChart options={options} series={series} height={250} />
}

function Replay({ f }: { f: FlyData }) {
  const r = f.replay
  if (!r) {
    return (
      <Card title="Replay" icon="bx-rewind">
        <Callout kind="info">
          <Md text="The replay has not run on the current definitions. It is the proof required before the fly trades: `fly-trader fly-replay` bootstraps a fly, lets it learn over months of history without retraining, and judges it." />
        </Callout>
      </Card>
    )
  }
  const bars = [{ k: 'plastic fly', v: r.mean }, { k: 'frozen fly', v: r.frozen_mean }, { k: 'random picks', v: r.random_mean }]
  return (
    <div className="row g-4">
      <div className="col-12 col-xl-8">
        <Card title="Replay" icon="bx-rewind" sub="the proof required before the fly trades">
          <Callout kind={r.passed ? 'success' : 'warning'} icon={r.passed ? 'bxs-check-circle' : 'block'}>Replay {r.passed ? 'passed' : 'failed'}: {r.reason ?? '—'}</Callout>
          <Metrics min={290}>
            <Metric label="Replay trades" value={r.n == null ? '—' : int(r.n)} icon="count" help="Trades on the evaluation half of the replay." />
            <Metric label="Average per trade" value={pct(r.mean)} delta={r.random_mean != null ? `random ${pct(r.random_mean)}` : null} tone="off" icon="pnl" color={bySign(r.mean, 'teal')} />
            <Metric label="Frozen fly" value={pct(r.frozen_mean)} icon="lock" help="The same bootstrap without plasticity, on the same days." />
            <Metric label="Profitable days" value={`${dash(r.days_positive)} of ${dash(r.days)}`} icon="calendar" progress={r.days ? (r.days_positive ?? 0) / r.days : null} />
            <Metric label="Learning rate α" value={r.alpha} icon="bx-slider-alt" />
            <Metric label="Forgetting half-life" value={r.half_life} icon="time" />
          </Metrics>
          <Caption>{r.caption}</Caption>
        </Card>
      </div>
      <div className="col-12 col-xl-4">
        <Card title="Replay vs random" icon="chart" sub="average net return per trade, evaluation half">
          <BarChart categories={bars.map((b) => b.k)} series={[{ name: '% per trade', data: bars.map((b) => (b.v == null ? null : b.v * 100)) }]} height={260}
            yFormat={(v) => `${v.toFixed(1)}%`} colors={['#ffffff']} />
          <KVList items={bars.map((b) => ({ label: b.k, value: signed(b.v, pct(b.v)) }))} />
        </Card>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------- Saved models, trainer log
function History() {
  const s = useTopic<{ rows: Saved[] }>('model.history')
  const log = useTopic<{ text: string }>('model.trainer_log')
  const cols: Col<Saved>[] = [
    { key: 'model', label: 'model', value: (r) => r.id, render: (r) => r.model },
    { key: 'type', label: 'type', render: (r) => <Pill color={r.type === 'selector' ? 'violet' : 'magenta'}>{r.type}</Pill> },
    { key: 'saved', label: 'saved', render: (r) => dt(r.saved) },
    { key: 'data', label: 'data', render: (r) => <Pill color={r.data === 'current' ? 'green' : 'gray'}>{r.data}</Pill> },
    { key: 'trades', label: 'trades', num: true, render: (r) => int(r.trades) },
    { key: 'mean', label: '% / trade', num: true, render: (r) => signed(r.mean, num(r.mean, 2, true)) },
    { key: 'random', label: 'random %', num: true, render: (r) => signed(r.random, num(r.random, 2, true)) },
    { key: 'pf', label: 'profit factor', num: true, render: (r) => num(r.pf, 2) },
    { key: 'deployed', label: 'put to work', render: (r) => yes(r.deployed) },
    { key: 'in_use', label: 'in use', render: (r) => (r.in_use ? <Pill color="green">in use</Pill> : null) },
  ]
  return (
    <>
      <Card title="Saved models" icon="folder" sub="the last 30 selectors and fly bootstraps">
        <Loaded state={s}>
          {(d) => d.rows.length ? (
            <>
              <DataTable rows={d.rows} cols={cols} filename="saved-models" />
              <Caption>Outdated models were trained before the data fixes of 2026-09-14; their numbers are not valid, are hidden, and they are never loaded.</Caption>
            </>
          ) : <Caption>No saved models.</Caption>}
        </Loaded>
      </Card>
      <Card>
        <Expander title="Trainer log" icon="terminal">
          <Loaded state={log}>{(d) => <pre className="log">{d.text}</pre>}</Loaded>
        </Expander>
      </Card>
    </>
  )
}
