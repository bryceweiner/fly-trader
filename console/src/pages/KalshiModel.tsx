// Kalshi model & training: the teacher, the trainer, the replay, the visual fly and its bootstrap as state widgets; the
// strategy stack with its edge lines, the automatic pipeline, the plastic visual fly's controls and health, and every
// saved Kalshi model (fly_trader/ui/api/kalshi_model.py).
import { PageHead } from '../components/ChainPicker'
import { BarChart, LineChart, RadialGauge } from '../components/Charts'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { ActionButton } from '../components/feedback'
import { Callout, Caption, Card, Expander, Help, KVList, Loaded, Metric, Metrics, Progress, ProgressList, RowGroup, Widget, WidgetGrid, type WidgetColor } from '../components/ui'
import { useApp } from '../context'
import { ago, dt, int, pct } from '../lib/format'
import { useNow, useTopic, type TopicState } from '../lib/hooks'
import { KalshiDormant } from './KalshiOverview'

type DormantData = { dormant: true }
const isDormant = (d: unknown): d is DormantData => !!d && (d as DormantData).dormant === true
const live = <T,>(s: TopicState<T | DormantData>): T | undefined => (s.data && !isDormant(s.data) ? s.data : undefined)

type Component = { component: string; kept: boolean; why: string; win: number | null; pf: number | null; trades: number | null; trials: number | null }
type Strategy = { strategy: string; arm: number | string | null; line: number | null; trigger: string; trades: number | null; win: number | null; pf: number | null }
type Stack = {
  selector: null | { id: number; ts: string | null; first_day: string | null; last_day: string | null; rows: number | null; deployable: boolean; deploy_reason: string }
  wf?: { n: number | null; mean: number | null; win: number | null; pf: number | null; days_positive: number | null; days: number | null }
  random_mean?: number | null
  components?: Component[]
  strategies?: Strategy[]
}
type Retraining = {
  interval_days: number; alive: boolean; running: boolean; status: string; last_run_at: string | null; next_run: string; selector_id: number | null
  bootstrap_id: number | null; may_trade: boolean | null; progress: { value: number; text: string } | null; waiting: string | null; failed: string | null
}
type FlyStrategy = { strategy: string; arm: string | null; line: number; shadow: number; drift: number; ic: number | null; checks: string }
type Update = { hour: string; n: number; mean_abs_delta: number | null; drift: number | null; ic: number | null }
type Rollback = { ts: string; reason: string | null; from_snapshot: number | null; to_snapshot: number | null }
type Replay = {
  passed: boolean; reason: string | null; n: number | null; mean: number | null; random_mean: number | null; frozen_mean: number | null; days_positive: number | null
  days: number | null; alpha: string; half_life: string; per_strategy: string | null
}
type VisualFly = {
  running: boolean; frozen: boolean; runner: boolean; replay: Replay | null; strategies: FlyStrategy[] | null; updates: Update[] | null; rollbacks: Rollback[] | null
  pending: number | null; learned: number; device: string | null; drift: number; drift_max: number; not_trading: string | null
  bootstrap: { id: number; ts: string | null; may_trade: boolean; text: string } | null
}
type Model = { id: number; model: string; type: string; saved: string; data: string; trades: number | null; per_trade: number | null; put_to_work: boolean | null }
type History = { models: Model[]; log: string }

const dash = (v: unknown) => (v === null || v === undefined ? '—' : String(v))
const fixed = (v: number | null | undefined, nd: number, signed = false) => (v === null || v === undefined ? null : `${signed && v >= 0 ? '+' : ''}${v.toFixed(nd)}`)
const yesNo = (v: boolean | null | undefined, yes = 'yes', no = 'no') => (v === null || v === undefined ? null : <Pill color={v ? 'green' : 'gray'}>{v ? yes : no}</Pill>)

export default function KalshiModel() {
  const { meta } = useApp()
  const off = !!meta && !meta.kalshi
  return (
    <>
      <PageHead title="Kalshi model & training" icon="bx-brain" chain={false} status={!off} />
      {off ? <KalshiDormant /> : <Body />}
    </>
  )
}

function Body() {
  const st = useTopic<Stack | DormantData>('kalshi_model.stack')
  const rt = useTopic<Retraining | DormantData>('kalshi_model.retraining')
  const vf = useTopic<VisualFly | DormantData>('kalshi_model.visual_fly')
  const hi = useTopic<History | DormantData>('kalshi_model.history')
  if ([st, rt, vf, hi].some((s) => isDormant(s.data))) return <KalshiDormant />
  return (
    <>
      <StateRow st={live(st)} rt={live(rt)} vf={live(vf)} />
      <StackRow s={st} />
      <RetrainingCard s={rt} />
      <VisualFlyRow s={vf} />
      <HistoryRow s={hi} />
    </>
  )
}

/** The page's state widgets: teacher, trainer, replay, the visual fly's learning and its bootstrap. */
function StateRow({ st, rt, vf }: { st?: Stack; rt?: Retraining; vf?: VisualFly }) {
  if (!st || !rt || !vf) return <div className="skeleton mb-4" style={{ height: 96 }} />
  const trainer: WidgetColor = !rt.alive ? 'dark' : rt.failed ? 'danger' : rt.running ? 'info' : rt.waiting ? 'warning' : 'success'
  return (
    <WidgetGrid min={200}>
      <Widget label="Teacher (deployable stack)" icon="psychology" color={st.selector ? 'violet' : 'dark'} value={st.selector ? `Selector #${st.selector.id}` : 'None put to work'}
        delta={st.selector ? `${pct(st.wf?.mean)} per evaluation trade` : 'no Kalshi selector put to work yet'} tone="off"
        help="The newest Kalshi selector on the current definitions that was put to work: the stack that teaches the visual fly." />
      <Widget label="Kalshi trainer" icon="model_training" color={trainer} value={rt.status} delta={rt.running ? rt.progress?.text || 'training' : `next run: ${rt.next_run}`} tone="off" />
      <Widget label="Replay" icon={vf.replay?.passed ? 'bx-badge-check' : 'block'} color={!vf.replay ? 'dark' : vf.replay.passed ? 'success' : 'warning'}
        value={!vf.replay ? 'Not run' : vf.replay.passed ? 'Passed' : 'Failed'} tone="off"
        delta={vf.replay ? `${vf.replay.n === null ? '—' : int(vf.replay.n)} trades · ${pct(vf.replay.mean)} per trade` : 'required before the fly trades'} />
      <Widget label="Visual fly learning" icon="visibility" color={!vf.running ? 'dark' : vf.frozen ? 'warning' : 'violet'} value={!vf.running ? 'Not trading' : vf.frozen ? 'Frozen' : 'On'}
        delta={vf.running ? `drift ${pct(vf.drift, 1, false)} · ${dash(vf.pending)} pending` : 'no settlements to learn from'} tone="off" />
      <Widget label="Latest bootstrap" icon="neurology" color={rt.bootstrap_id === null ? 'dark' : rt.may_trade ? 'success' : 'warning'}
        value={rt.bootstrap_id !== null ? `#${rt.bootstrap_id}` : '—'} delta={rt.bootstrap_id !== null ? (rt.may_trade ? 'may trade' : 'may not trade') : 'none for the current definitions'} tone="off" />
    </WidgetGrid>
  )
}

const COMP_COLS: Col<Component>[] = [
  { key: 'component', label: 'component' },
  { key: 'kept', label: 'kept', render: (r) => yesNo(r.kept, 'kept', 'dropped') },
  { key: 'why', label: 'why', wrap: true },
  { key: 'win', label: 'winning (evaluation)', num: true, render: (r) => (r.win === null ? null : `${r.win.toFixed(0)}%`) },
  { key: 'pf', label: 'profit factor', num: true, render: (r) => fixed(r.pf, 2) },
  { key: 'trades', label: 'trades', num: true, render: (r) => (r.trades === null ? null : int(r.trades)) },
  { key: 'trials', label: 'settings tried', num: true },
]
const STRAT_COLS: Col<Strategy>[] = [
  { key: 'strategy', label: 'strategy' },
  { key: 'arm', label: 'arm' },
  { key: 'line', label: 'edge line', num: true, help: 'Minimum p̂ − fee-inclusive price to trade.', render: (r) => fixed(r.line, 3) },
  { key: 'trigger', label: 'trigger', wrap: true, render: (r) => <code>{r.trigger}</code> },
  { key: 'trades', label: 'trades (evaluation)', num: true, render: (r) => (r.trades === null ? null : int(r.trades)) },
  { key: 'win', label: 'winning', num: true, render: (r) => (r.win === null ? null : `${r.win.toFixed(0)}%`) },
  { key: 'pf', label: 'profit factor', num: true, render: (r) => fixed(r.pf, 2) },
]

function StackRow({ s }: { s: TopicState<Stack | DormantData> }) {
  const now = useNow(15000)
  return (
    <Loaded state={s} height={200}>
      {(d) => isDormant(d) ? null : !d.selector ? (
        <Card title="The teacher: the Kalshi strategy stack" icon="psychology">
          <Callout kind="info">
            No Kalshi selector on the current definitions has been put to work yet: this panel shows the newest deployable stack, the one that teaches the visual
            fly (Saved Kalshi models below lists every fit). The pipeline below fits one as soon as the corpus is built.
          </Callout>
        </Card>
      ) : (
        <div className="row g-4">
          <div className="col-12 col-xl-8">
            <Card title="The teacher: the Kalshi strategy stack" icon="psychology"
              sub={<>Selector #{d.selector.id} (the newest put to work) · fitted walk-forward by settlement day on {d.selector.first_day || '—'} → {d.selector.last_day || '—'}{' '}
                ({d.selector.rows === null || d.selector.rows === undefined ? '—' : int(d.selector.rows)} rows) · saved {ago(d.selector.ts, now)} ·{' '}
                {d.selector.deployable ? 'put to work: ' : 'not put to work: '}{d.selector.deploy_reason}</>}>
              <RowGroup items={[
                { label: <>Evaluation trades<Help text="Positions on the evaluation half, each scored by a classifier that never saw that settlement day." /></>,
                  value: d.wf?.n === null || d.wf?.n === undefined ? '—' : int(d.wf.n) },
                { label: <>{d.random_mean !== null && d.random_mean !== undefined ? `Average per trade · random ${pct(d.random_mean)}` : 'Average per trade'}
                  <Help text="Net return per dollar of payout at Kalshi's fees; 'random' is the same number of random rows bought at the ask." /></>, value: pct(d.wf?.mean) },
                { label: 'Winning', value: pct(d.wf?.win, 0, false) },
                { label: 'Profit factor', value: d.wf?.pf ? d.wf.pf.toFixed(2) : '—' },
                { label: 'Profitable days', value: `${dash(d.wf?.days_positive)} of ${dash(d.wf?.days)}` },
              ]} />
              {!!d.components?.length && <div className="mt-4"><DataTable rows={d.components} cols={COMP_COLS} filename="kalshi-stack-components" search={false} /></div>}
              {!!d.strategies?.length && <div className="mt-4"><DataTable rows={d.strategies} cols={STRAT_COLS} filename="kalshi-stack-strategies" search={false} /></div>}
            </Card>
          </div>
          <div className="col-12 col-xl-4">
            <Card title="Evaluation" icon="target" sub="Winning share on the evaluation half">
              <RadialGauge value={d.wf?.win ?? 0} label="winning" display={pct(d.wf?.win, 0, false)} height={220} />
            </Card>
            <Card title="Strategy edge lines" icon="bx-slider-alt" sub="Minimum p̂ − fee-inclusive price each strategy trades at">
              {d.strategies?.length ? (
                <ProgressList items={d.strategies.map((x) => ({ label: `${x.strategy} · ${dash(x.arm)}`, value: fixed(x.line, 3) ?? '—', frac: Math.max(0, x.line ?? 0) }))} />
              ) : <Caption>No strategies recorded.</Caption>}
            </Card>
          </div>
        </div>
      )}
    </Loaded>
  )
}

function RetrainingCard({ s }: { s: TopicState<Retraining | DormantData> }) {
  const now = useNow(5000)
  const d = live(s)
  return (
    <Card title="Kalshi training pipeline" icon="autorenew" sub={d ? `every ${d.interval_days} days until the visual fly trades, then on request` : undefined}
      actions={d && (
        <>
          <ActionButton label="Retrain now" icon="play" name="kalshi_model.retrain" variant="primary" disabled={d.running}
            confirm={{
              title: 'Retrain the Kalshi stack now?', confirmLabel: 'Retrain now',
              body: <>The Kalshi training pipeline runs at once: the strategy stack refits walk-forward by settlement day on the whole corpus, and a visual fly is
                bootstrapped when none exists for the current definitions.{d.alive ? '' : ' The Kalshi trainer is not running, so it is started.'}</>,
            }} />
          {d.running && (
            <ActionButton label="Stop" icon="stop" name="kalshi_model.train_stop" variant="danger"
              confirm={{
                title: 'Stop Kalshi training?', confirmLabel: 'Stop', danger: true,
                body: 'The Kalshi trainer is asked to stop: the run in progress ends unfinished (stage "stopped") and the trainer thread exits. Retrain now or the Processes page starts it again.',
              }} />
          )}
        </>
      )}>
      <Loaded state={s}>
        {(r) => isDormant(r) ? null : (
          <>
            {!r.alive && <Callout kind="warning">The Kalshi trainer is not running: start it on the Processes page or click Retrain now.</Callout>}
            <Metrics min={170}>
              <Metric label="Status" icon="model_training" value={r.status} color={!r.alive ? 'dark' : r.failed ? 'danger' : r.running ? 'info' : 'success'} />
              <Metric label="Last run" icon="history" value={r.last_run_at ? ago(r.last_run_at, now) : 'never'} />
              <Metric label="Next run" icon="calendar" value={r.next_run} />
              <Metric label="Latest deployable selector" icon="psychology" value={r.selector_id !== null ? `#${r.selector_id}` : '—'}
                help="The newest Kalshi selector on the current definitions that was put to work; fits that were not are listed under Saved Kalshi models." />
              <Metric label="Latest bootstrap" icon="neurology" value={r.bootstrap_id !== null ? `#${r.bootstrap_id}` : '—'}
                delta={r.bootstrap_id !== null ? (r.may_trade ? 'may trade' : 'may not trade') : null} tone="off"
                color={r.bootstrap_id === null ? null : r.may_trade ? 'success' : 'warning'} />
            </Metrics>
            {r.progress && <Progress value={r.progress.value} text={r.progress.text} />}
            {r.waiting && <Callout kind="info">{r.waiting}</Callout>}
            {r.failed && <Callout kind="error">{r.failed}</Callout>}
            <Caption>
              1 · The Kalshi strategy stack (favorite, momentum, constraint, ev; taker and maker arms) is fitted walk-forward by settlement day at Kalshi's fees and
              judged against random picks.<br />
              2 · The visual fly is distilled from it once — when none exists for the current definitions, or on request — with its features entering the photoreceptors;
              then it learns from every settlement.<br />
              3 · The replay (<code>fly-trader kalshi-fly-replay</code>) must pass before it trades.
            </Caption>
          </>
        )}
      </Loaded>
    </Card>
  )
}

const FLY_COLS: Col<FlyStrategy>[] = [
  { key: 'strategy', label: 'strategy' },
  { key: 'arm', label: 'arm' },
  { key: 'line', label: 'edge line', num: true, render: (r) => fixed(r.line, 3) },
  { key: 'shadow', label: 'shadow line', num: true, render: (r) => fixed(r.shadow, 3) },
  { key: 'drift', label: 'drift', num: true, render: (r) => `${r.drift.toFixed(2)}%` },
  { key: 'ic', label: '24 h IC', num: true, render: (r) => fixed(r.ic, 3) },
  { key: 'checks', label: 'checks', wrap: true, render: (r) => <Pill color={r.checks === 'ok' ? 'green' : 'orange'}>{r.checks}</Pill> },
]
const ROLLBACK_COLS: Col<Rollback>[] = [
  { key: 'ts', label: 'ts', render: (r) => dt(r.ts, true) },
  { key: 'reason', label: 'reason', wrap: true },
  { key: 'from_snapshot', label: 'from_snapshot', num: true },
  { key: 'to_snapshot', label: 'to_snapshot', num: true },
]

function FlyActions({ d }: { d: VisualFly }) {
  return (
    <>
      <ActionButton label={d.frozen ? 'Resume learning' : 'Pause learning'} icon={d.frozen ? 'play' : 'pause'} name="kalshi_model.fly_command" variant="light"
        body={{ cmd: d.frozen ? 'resume' : 'pause' }} disabled={!d.running || !d.runner}
        confirm={d.frozen
          ? { title: "Resume the visual fly's learning?", confirmLabel: 'Resume learning',
            body: 'The visual fly learns from every settlement again (its rollback counters start over). Sent as a command it applies at its next minute.' }
          : { title: "Pause the visual fly's learning?", confirmLabel: 'Pause learning',
            body: 'The visual fly keeps trading but stops learning from settlements until you resume it. Sent as a command it applies at its next minute.' }} />
      <ActionButton label="Roll back" icon="undo" name="kalshi_model.fly_command" body={{ cmd: 'rollback' }} disabled={!d.running || !d.runner} variant="light"
        help="Restore the newest good hourly snapshot at least a day old (or the bootstrap)."
        confirm={{
          title: 'Roll the visual fly back?', confirmLabel: 'Roll back', danger: true,
          body: 'The visual fly restores the newest good hourly snapshot at least a day old (or its bootstrap), dropping what it learned since. Sent as a command it applies at its next minute.',
        }} />
      <ActionButton label="Re-bootstrap" icon="refresh" name="kalshi_model.fly_rebootstrap" help="The Kalshi stack refits and teaches a new visual fly." variant="light"
        confirm={{
          title: 'Re-bootstrap the visual fly?', confirmLabel: 'Re-bootstrap',
          body: 'The Kalshi stack refits and teaches a new visual fly: the Kalshi training pipeline runs with a fly bootstrap requested, and the Kalshi trainer is started if it is not running.',
        }} />
      <ActionButton label="Reset fly" icon="reset" name="kalshi_model.fly_reset" variant="danger" disabled={d.runner}
        help={d.runner ? 'Stop the Kalshi engine first.' : undefined}
        confirm={{
          title: 'Reset the visual fly?', confirmLabel: 'Reset', danger: true,
          body: (
            <>
              <p>Archive and clear the visual fly's paper books, orders and everything it learned?</p>
              <p className="mb-0 text-muted2">The Kalshi paper books' decisions, beats, wealth marks, positions, orders and fills, its plastic snapshots and sessions, the
                kalshi_fly_* tables, its book state and status, and its learned-state and activity folders are archived (reset_kalshi_fly_&lt;time&gt;) and cleared.
                The bootstrap snapshots and the live Kalshi books are kept.</p>
            </>
          ),
        }} />
    </>
  )
}

function VisualFlyRow({ s }: { s: TopicState<VisualFly | DormantData> }) {
  const now = useNow(15000)
  const d = live(s)
  return (
    <div className="row g-4">
      <div className="col-12 col-xl-8">
        <Card title="The visual fly" icon="visibility" sub="the optic lobes taught once by the Kalshi stack, then learning from every settlement"
          actions={d && <FlyActions d={d} />}>
          <Loaded state={s} height={200}>
            {(f) => isDormant(f) ? null : (
              <>
                {!f.replay ? (
                  <Callout kind="info">
                    The Kalshi replay has not run on the current definitions: <code>fly-trader kalshi-fly-replay</code> bootstraps a visual fly, lets it learn from months of
                    settlements without retraining, and judges it — the proof required before it trades.
                  </Callout>
                ) : (
                  <>
                    <Callout kind={f.replay.passed ? 'success' : 'warning'} icon={f.replay.passed ? 'bx-badge-check' : 'block'}>
                      Replay {f.replay.passed ? 'passed' : 'failed'}: {f.replay.reason}
                    </Callout>
                    <RowGroup items={[
                      { label: 'Replay trades', value: f.replay.n === null || f.replay.n === undefined ? '—' : int(f.replay.n) },
                      { label: f.replay.random_mean !== null && f.replay.random_mean !== undefined ? `Average per trade · random ${pct(f.replay.random_mean)}` : 'Average per trade',
                        value: pct(f.replay.mean) },
                      { label: <>Frozen fly<Help text="The same bootstrap without plasticity." /></>, value: pct(f.replay.frozen_mean) },
                      { label: 'Profitable days', value: `${dash(f.replay.days_positive)} of ${dash(f.replay.days)}` },
                      { label: 'Learning rate α', value: f.replay.alpha },
                      { label: 'Forgetting half-life', value: f.replay.half_life },
                    ]} />
                    {f.replay.per_strategy && <Caption>{f.replay.per_strategy}</Caption>}
                  </>
                )}
                {f.running ? (
                  <>
                    {!!f.strategies?.length && <div className="mt-3"><DataTable rows={f.strategies} cols={FLY_COLS} filename="kalshi-fly-strategies" search={false} /></div>}
                    {!!f.updates?.length && (
                      <div className="mt-3">
                        <h6 className="text-white mb-1">Hourly learning</h6>
                        <LineChart height={220} series={[
                          { name: 'drift', data: f.updates.map((u) => [u.hour, u.drift] as [string, number | null]) },
                          { name: 'ic', data: f.updates.map((u) => [u.hour, u.ic] as [string, number | null]) },
                        ]} />
                      </div>
                    )}
                    {!!f.rollbacks?.length && (
                      <Expander title={`Rollbacks (${f.rollbacks.length})`} icon="history">
                        <DataTable rows={f.rollbacks} cols={ROLLBACK_COLS} filename="kalshi-fly-rollbacks" search={false} />
                      </Expander>
                    )}
                  </>
                ) : <Callout kind="info" icon="bx-pause-circle">{f.not_trading}</Callout>}
                {f.bootstrap && <Caption>Bootstrap #{f.bootstrap.id} ({ago(f.bootstrap.ts, now)}): {f.bootstrap.text}</Caption>}
              </>
            )}
          </Loaded>
        </Card>
      </div>
      <div className="col-12 col-xl-4">
        <Card title="Drift" icon="bx-git-branch" sub={d ? `How far learning moved the fly from its bootstrap; a strategy past ${pct(d.drift_max, 0, false)} rolls back` : undefined}>
          <Loaded state={s} height={200}>
            {(f) => isDormant(f) ? null : (
              <>
                <RadialGauge value={f.drift_max ? f.drift / f.drift_max : 0} label={f.running ? 'drift' : 'not trading'} display={f.running ? pct(f.drift, 1, false) : '—'} height={220} />
                {f.running ? (
                  <KVList items={[
                    { label: <>Settlements pending<Help text="Scored rows waiting for their market to resolve." /></>, value: dash(f.pending) },
                    { label: 'Learned this minute', value: int(f.learned) },
                    { label: 'Learning', value: <Pill color={f.frozen ? 'orange' : 'violet'}>{f.frozen ? 'frozen' : 'on'}</Pill> },
                    { label: 'Device', value: f.device || '—' },
                  ]} />
                ) : <Caption>{f.not_trading}</Caption>}
              </>
            )}
          </Loaded>
        </Card>
        {d?.running && !!d.strategies?.length && (
          <Card title="Edge lines" icon="bx-slider-alt" sub="Each strategy's live edge line beside its shadow (the frozen bootstrap's) line">
            <BarChart categories={d.strategies.map((x) => x.strategy)} height={240} yFormat={(v) => v.toFixed(3)}
              series={[{ name: 'edge line', data: d.strategies.map((x) => x.line) }, { name: 'shadow line', data: d.strategies.map((x) => x.shadow) }]} />
          </Card>
        )}
      </div>
    </div>
  )
}

const MODEL_COLS: Col<Model>[] = [
  { key: 'model', label: 'model', value: (r) => r.id },
  { key: 'type', label: 'type', render: (r) => <Pill color={r.type === 'stack' ? 'blue' : 'violet'}>{r.type}</Pill> },
  { key: 'saved', label: 'saved', render: (r) => dt(r.saved) },
  { key: 'data', label: 'data', render: (r) => <Pill color={r.data === 'current' ? 'green' : 'gray'}>{r.data}</Pill> },
  { key: 'trades', label: 'trades', num: true, render: (r) => (r.trades === null ? null : int(r.trades)) },
  { key: 'per_trade', label: '% / trade', num: true, render: (r) => (r.per_trade === null ? null : <span className={r.per_trade >= 0 ? 'text-pos' : 'text-neg'}>{fixed(r.per_trade, 2, true)}</span>) },
  { key: 'put_to_work', label: 'put to work', render: (r) => yesNo(r.put_to_work) },
]

function HistoryRow({ s }: { s: TopicState<History | DormantData> }) {
  return (
    <Loaded state={s} height={120}>
      {(d) => isDormant(d) ? null : (
        <div className="row g-4">
          <div className="col-12 col-xl-7">
            <Card title="Saved Kalshi models" icon="folder" sub="Every Kalshi stack and visual-fly bootstrap, newest first">
              {d.models.length ? <DataTable rows={d.models} cols={MODEL_COLS} filename="kalshi-models" /> : <Caption>No saved Kalshi models.</Caption>}
            </Card>
          </div>
          <div className="col-12 col-xl-5">
            <Card title="Kalshi trainer log" icon="terminal" sub="The last 40 lines the Kalshi trainer logged">
              <pre className="log mb-0">{d.log || '(no log yet)'}</pre>
            </Card>
          </div>
        </div>
      )}
    </Loaded>
  )
}
