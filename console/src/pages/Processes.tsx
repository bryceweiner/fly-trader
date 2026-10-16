// Processes: every worker thread with what it does, its state, start/stop, autostart and recent activity; the event log
// (fly_trader/ui/api/processes.py). Dashtreme layout: a KPI row of colour widgets, one colour widget card per worker
// (its colour is its state), the event log as a DataTable beside the last 24 hours by level.
import { useState, type ReactNode } from 'react'
import { action } from '../lib/api'
import { useNow, useStored, useTopic } from '../lib/hooks'
import { ago, dt, hm, int } from '../lib/format'
import { Callout, Caption, Card, Empty, Icon, KVList, Loaded, Pills, SectionTitle, Sparkline, Toggle, Widget, WidgetGrid, type WidgetColor } from '../components/ui'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { DonutChart } from '../components/Charts'
import { ActionButton, Modal, useFeedback } from '../components/feedback'
import { PageHead } from '../components/ChainPicker'

type WorkerState = 'running' | 'stopping' | 'crashed' | 'stopped' | 'external'
type Worker = {
  name: string; title: string; icon: string; desc: string; role: string; role_label: string; group: string
  state: WorkerState; alive: boolean; stopping: boolean; started_at: string | null; stopped_at: string | null; error: string | null
  external: boolean; autostart: boolean; ext_pid: number | null; ext_started_at: string | null; start_note: string | null; stop_note: string | null
}
type External = { name: string; title: string; pid: number; started_at: string | null }
type Counts = { running: number; stopping: number; crashed: number; stopped: number; external: number; total: number; autostart: number; trade_needed: number; trade_up: number }
type Workers = { pid: number; observer: boolean; host_pid: number | null; external: External[]; groups: { key: string; title: string; workers: Worker[] }[]; counts: Counts }
type EventStats = { levels: Record<string, number>; hours: string[]; warning: number[]; error: number[] }
type LogLine = { time: string; level: string; what: string }
type Ev = { ts: string; level: string; source: string; message: string }

const STATE: Record<WorkerState, { color: WidgetColor; label: string; icon: string }> = {
  running: { color: 'success', label: 'Running', icon: 'bx-play-circle' },
  stopping: { color: 'warning', label: 'Stopping', icon: 'bx-loader-circle' },
  crashed: { color: 'danger', label: 'Crashed', icon: 'bx-error-circle' },
  stopped: { color: 'dark', label: 'Stopped', icon: 'bx-stop-circle' },
  external: { color: 'info', label: 'Running outside', icon: 'bx-terminal' },
}
const LEVELS = ['info', 'warning', 'error'] as const
const LEVEL_PILL: Record<string, string> = { info: 'gray', warning: 'orange', error: 'red', INFO: 'gray', WARNING: 'orange', ERROR: 'red', CRITICAL: 'red', DEBUG: 'gray' }
const LEVEL_DOT: Record<string, string> = { info: 'rgba(255,255,255,0.35)', warning: '#ffc107', error: '#f0435a' }

export default function Processes() {
  const s = useTopic<Workers>('processes.workers')
  const stats = useTopic<EventStats>('processes.event_stats')
  return (
    <>
      <PageHead title="Processes" icon="bx-server" chain={false} />
      <Loaded state={s} height={260}>
        {(d) => (
          <>
            <Kpis d={d} stats={stats.data} />
            <Caption>
              {d.observer
                ? `This console only observes (fly-trader ui --observe): the workers are threads inside the console that hosts them (pid ${d.host_pid ?? 'unknown'}); start and stop them there.`
                : `All workers are threads inside this console process (pid ${d.pid}); stopping one waits for it to finish its current step.`}
            </Caption>
            {d.external.length > 0 && <ExternalBanner ext={d.external} observer={d.observer} />}
            {d.groups.map((g) => {
              const up = g.workers.filter((w) => w.state === 'running' || w.state === 'external').length
              return (
                <section key={g.key} className="mt-4">
                  <SectionTitle>{g.title} · {up} of {g.workers.length} running</SectionTitle>
                  <div className="row g-4">
                    {g.workers.map((w) => <div key={w.name} className="col-12 col-md-6 col-xxl-4"><WorkerCard w={w} observer={d.observer} /></div>)}
                  </div>
                </section>
              )
            })}
          </>
        )}
      </Loaded>
      <section className="mt-4">
        <SectionTitle>Events</SectionTitle>
        <div className="row g-4">
          <div className="col-12 col-xl-8"><EventLog /></div>
          <div className="col-12 col-xl-4"><EventSummary s={stats.data} /></div>
        </div>
      </section>
    </>
  )
}

function Kpis({ d, stats }: { d: Workers; stats?: EventStats }) {
  const c = d.counts
  const errors = stats?.levels.error ?? null
  return (
    <WidgetGrid min={190}>
      <Widget label="Running" value={`${c.running} of ${c.total}`} icon="bx-play-circle" color={c.running ? 'success' : 'dark'} progress={c.total ? c.running / c.total : 0}
        delta={`${c.trade_up} of ${c.trade_needed} needed to trade`} tone="off" help="Worker threads running inside the console (CLI processes count under External)." />
      <Widget label="Stopped" value={int(c.stopped + c.stopping)} icon="bx-stop-circle" color="dark" delta={c.stopping ? `${c.stopping} stopping` : 'not running'} tone="off" />
      <Widget label="Crashed" value={int(c.crashed)} icon="bx-error-circle" color={c.crashed ? 'danger' : 'dark'} delta={c.crashed ? 'see the red cards' : 'no crashes'} tone="off"
        help="A worker whose thread ended with an error: its card shows the first line." />
      <Widget label="External" value={int(c.external)} icon="bx-terminal" color={c.external ? 'warning' : 'dark'} delta={c.external ? 'started from the command line' : 'none outside the console'} tone="off"
        help="Worker processes started with `fly-trader worker start`; the console refuses to run a second copy." />
      <Widget label="Autostart" value={`${c.autostart} of ${c.total}`} icon="bx-sync" color="info" progress={c.total ? c.autostart / c.total : 0} delta="start with the console" tone="off" />
      <Widget label="Errors · 24 h" value={errors === null ? '—' : int(errors)} icon="error" spark={stats?.error} sparkType="bar"
        delta={stats ? `${int(stats.levels.warning ?? 0)} warnings` : null} tone="off" help="Events at level error in the last 24 hours; the bars are per hour." />
    </WidgetGrid>
  )
}

function ExternalBanner({ ext, observer }: { ext: External[]; observer: boolean }) {
  const now = useNow(5000)
  return (
    <Callout kind="error" icon="warning">
      <div className="mb-2">Worker processes are running outside the console (started from the command line). Stop them to keep everything in one place.</div>
      {ext.map((a) => (
        <div key={a.name} className="d-flex align-items-center gap-3 flex-wrap py-1">
          <span>External <strong>{a.name}</strong> · pid {a.pid} · started {ago(a.started_at, now)}</span>
          <StopExternal name={a.name} title={a.title} pid={a.pid} observer={observer} />
        </div>
      ))}
    </Callout>
  )
}

function StopExternal({ name, title, pid, observer }: { name: string; title: string; pid: number | null; observer: boolean }) {
  return (
    <ActionButton label="Stop external" icon="stop" name="processes.stop_external" body={{ name }} disabled={observer} variant="light"
      confirm={{
        title: `Stop the external ${name} process?`, danger: true, confirmLabel: 'Stop external',
        body: <p className="mb-0">Sends SIGTERM to pid {pid ?? '?'}, waits up to 30 s for it to finish, then kills it (SIGKILL). The console can then run {title} as one of its threads.</p>,
      }} />
  )
}

/** A pill on a colour widget (role, autostart, thread). */
function WPill({ children, title }: { children: ReactNode; title?: string }) {
  return (
    <span title={title} style={{ background: 'rgb(255 255 255 / 16%)', border: '1px solid rgb(255 255 255 / 22%)', borderRadius: 999, padding: '0.12rem 0.6rem', fontSize: 12, whiteSpace: 'nowrap' }}>
      {children}
    </span>
  )
}

function WorkerCard({ w, observer }: { w: Worker; observer: boolean }) {
  const now = useNow(5000)
  const fb = useFeedback()
  const [activity, setActivity] = useState(false)
  const [busy, setBusy] = useState(false)
  const st = STATE[w.state]
  const ink = st.color === 'warning' ? '#241a00' : '#fff'
  const since = w.alive && w.started_at ? `running since ${ago(w.started_at, now)}` : w.stopped_at ? `stopped ${ago(w.stopped_at, now)}` : 'not started'
  const sub = w.state === 'external' ? `pid ${w.ext_pid ?? '?'} · started ${ago(w.ext_started_at, now)}` : since
  const setAutostart = async (on: boolean) => {
    const ok = await fb.confirm({
      title: on ? `Autostart ${w.title}?` : `Stop autostarting ${w.title}?`,
      body: <p className="mb-0">{on ? `${w.title} will start whenever the console starts.` : `${w.title} will no longer start with the console; this does not stop it now.`}</p>,
      confirmLabel: on ? 'Turn on' : 'Turn off',
    })
    if (!ok) return
    setBusy(true)
    try {
      const r = await action('processes.autostart', { name: w.name, on })
      fb.toast(r.message || 'Done', 'success')
    } catch (e) {
      fb.toast((e as Error).message, 'error', 10000)
    } finally {
      setBusy(false)
    }
  }
  return (
    <div className={`card radius-10 widget cw cw-${st.color} h-100 mb-0`} style={{ color: ink }}>
      <div className="w-body">
        <div className="w-text">
          <p className="w-label">{w.title}</p>
          <h4 className="w-value">{st.label}</h4>
          <p className="w-delta">{sub}</p>
        </div>
        <div className="w-icon"><Icon name={w.icon} /></div>
      </div>
      <div className="d-flex flex-column gap-2" style={{ padding: '0 1.1rem 1rem', flex: 1 }}>
        <div className="d-flex gap-2 flex-wrap align-items-center">
          <WPill>{w.role_label}</WPill>
          <WPill title="Starts whenever the console starts">{w.autostart ? 'autostart on' : 'autostart off'}</WPill>
          <WPill title="The thread's name (logs/<name>.log)">thread <code style={{ color: 'inherit' }}>{w.name}</code></WPill>
        </div>
        <div style={{ fontSize: 13, lineHeight: 1.45, opacity: 0.88 }}>{w.desc}</div>
        {w.error && (
          <div style={{ background: 'rgb(0 0 0 / 28%)', borderRadius: 8, padding: '0.45rem 0.65rem', fontFamily: 'var(--fly-mono)', fontSize: 12.5, wordBreak: 'break-word' }}>
            <i className="bx bx-error-circle me-1" />{w.error}
          </div>
        )}
        <div className="d-flex gap-2 align-items-center flex-wrap mt-auto pt-1">
          <ActionButton label="Start" icon="play" name="processes.start" body={{ name: w.name }} disabled={w.alive || w.external || observer} variant="light"
            help={w.external ? 'Running outside the console: stop the external process first.' : undefined}
            confirm={{
              title: `Start ${w.title}?`, confirmLabel: 'Start',
              body: (
                <>
                  <p>Starts {w.title} as a thread inside this console: {w.desc}</p>
                  {w.start_note && <Callout kind="warning">{w.start_note}</Callout>}
                </>
              ),
            }} />
          {w.state === 'external'
            ? <StopExternal name={w.name} title={w.title} pid={w.ext_pid} observer={observer} />
            : (
              <ActionButton label="Stop" icon="stop" name="processes.stop" body={{ name: w.name }} disabled={!w.alive || w.stopping || observer} variant="light"
                confirm={{
                  title: `Stop ${w.title}?`, confirmLabel: 'Stop', danger: w.role === 'trade' || w.role === 'vault',
                  body: (
                    <>
                      <p className={w.stop_note ? '' : 'mb-0'}>Asks the {w.title} thread to stop; it finishes its current step first, which can take a while.</p>
                      {w.stop_note && <p className="mb-0">{w.stop_note}</p>}
                    </>
                  ),
                }} />
            )}
          <button type="button" className="btn btn-sm btn-light" onClick={() => setActivity(true)}><Icon name="bx-list-ul" />Activity</button>
          <span className="ms-auto">
            <Toggle label="Autostart" checked={w.autostart} onChange={setAutostart} disabled={busy || observer} help="Start this worker whenever the console starts." />
          </span>
        </div>
      </div>
      {activity && <Activity name={w.name} title={w.title} onClose={() => setActivity(false)} />}
    </div>
  )
}

const LOG_COLS: Col<LogLine>[] = [
  { key: 'time', label: 'time', width: 90 },
  { key: 'level', label: 'level', width: 100, render: (r) => (r.level ? <Pill color={LEVEL_PILL[r.level] || 'gray'}>{r.level}</Pill> : '') },
  { key: 'what', label: 'what', wrap: true },
]

function Activity({ name, title, onClose }: { name: string; title: string; onClose: () => void }) {
  const s = useTopic<{ name: string; rows: LogLine[] }>('processes.activity', { worker: name })
  return (
    <Modal title={`${title} · activity`} icon="bx-list-ul" onClose={onClose} wide>
      <Loaded state={s}>
        {(d) => (d.rows.length
          ? <DataTable rows={d.rows} cols={LOG_COLS} height={480} filename={`${name}-activity`} search={false} pageSize={25} />
          : <Caption>Nothing logged yet.</Caption>)}
      </Loaded>
    </Modal>
  )
}

const EV_COLS: Col<Ev>[] = [
  { key: 'ts', label: 'time', width: 150, render: (r) => dt(r.ts, true) },
  { key: 'level', label: 'level', width: 100, render: (r) => <Pill color={LEVEL_PILL[r.level] || 'gray'}>{r.level}</Pill> },
  { key: 'source', label: 'source', width: 130 },
  { key: 'message', label: 'message', wrap: true },
]

function EventLog() {
  const [stored, setStored] = useStored<string[]>('ev_levels', ['warning', 'error'])
  const levels = LEVELS.filter((l) => stored.includes(l))
  const s = useTopic<{ levels: string[]; rows: Ev[] }>('processes.events', { levels })
  return (
    <Card title="Event log" icon="bx-history" className="h-100" actions={<Pills options={LEVELS} value={levels} onChange={(v) => setStored(v)} />}>
      <Loaded state={s}>
        {(d) => (d.rows.length ? <DataTable rows={d.rows} cols={EV_COLS} height={520} filename="events" /> : <Empty>Nothing at these levels.</Empty>)}
      </Loaded>
    </Card>
  )
}

function EventSummary({ s }: { s?: EventStats }) {
  if (!s) return <Card title="Last 24 hours" icon="bx-pie-chart-alt-2" className="h-100"><div className="skeleton" style={{ height: 300 }} /></Card>
  const total = Object.values(s.levels).reduce((a, b) => a + b, 0)
  const lv = LEVELS.map((l) => ({ l, n: s.levels[l] ?? 0 }))
  const hourly = s.warning.map((w, i) => w + (s.error[i] ?? 0))
  return (
    <Card title="Last 24 hours" icon="bx-pie-chart-alt-2" className="h-100">
      {total
        ? <DonutChart labels={lv.map((x) => x.l)} values={lv.map((x) => x.n)} center={int(total)} centerLabel="events" height={250} colors={lv.map((x) => LEVEL_DOT[x.l])} />
        : <Empty>No events in the last 24 hours.</Empty>}
      <KVList items={lv.map((x) => ({ label: x.l, value: `${int(x.n)}${total ? ` · ${((x.n / total) * 100).toFixed(0)}%` : ''}`, dot: LEVEL_DOT[x.l] }))} />
      <div className="mt-3">
        <div className="caption mb-1">
          Warnings and errors per hour, {hm(s.hours[0])}–{hm(s.hours[s.hours.length - 1])} UTC{hourly.some((v) => v > 0) ? ` · peak ${int(Math.max(...hourly))} in an hour` : ''}
        </div>
        {hourly.some((v) => v > 0) ? <Sparkline data={hourly} type="bar" height={64} /> : <Empty>None.</Empty>}
      </div>
    </Card>
  )
}
