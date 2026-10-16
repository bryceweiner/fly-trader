// Logs: the rotating JSON-lines file each worker writes (logs/<name>.log), tailed live every second
// (fly_trader/ui/api/logs.py). Lines are scrubbed of secrets by the logger before they reach the file.
import { useEffect, useMemo, useRef, useState } from 'react'
import { PageHead } from '../components/ChainPicker'
import { Callout, Caption, Card, Empty, Icon, Loaded, Pills, ProgressList, Segmented, Toggle, Widget, WidgetGrid } from '../components/ui'
import { useNow, useStored, useTopic } from '../lib/hooks'
import { ago, int } from '../lib/format'
import { download } from '../lib/csv'

type FileInfo = { name: string; exists: boolean; size: number; mtime: string | null }
type Tail = FileInfo & { n: number; lines: string[] }
type Line = { raw: string; i: number; ts?: string; level?: string; logger?: string; thread?: string; msg?: string; exc?: string }

const LEVELS = ['INFO', 'WARNING', 'ERROR'] as const
type Level = (typeof LEVELS)[number]
const LEVEL_COLOR: Record<string, string> = { DEBUG: 'gray', INFO: 'blue', WARNING: 'orange', ERROR: 'red', CRITICAL: 'red' }
const SIZES = ['200', '500', '1000', '2000', '5000'] as const
const BUCKETS = 24
const TITLES: Record<string, string> = {
  runner: 'Trading engine', pumpstream: 'Market feed', train: 'Trainer', replay: 'History archive', discover: 'Token stats', rh_stream: 'Robinhood Chain feed',
  vault: '$FLY vault', kalshi_runner: 'Kalshi trading engine', kalshi_stream: 'Kalshi market feed', kalshi_train: 'Kalshi trainer', kalshi_history: 'Kalshi history',
  console: 'Console', 'console-observer': 'Observer console',
}

/** WARNING for a WARNING line, ERROR for ERROR and CRITICAL, INFO for INFO and DEBUG (the three pills). */
function bucket(level?: string): Level | null {
  if (!level) return null
  return level === 'WARNING' ? 'WARNING' : level === 'ERROR' || level === 'CRITICAL' ? 'ERROR' : 'INFO'
}

function parse(raw: string, i: number): Line {
  if (!raw.startsWith('{')) return { raw, i }
  try {
    const o = JSON.parse(raw) as Record<string, unknown>
    return { raw, i, ts: String(o.ts ?? ''), level: String(o.level ?? ''), logger: String(o.logger ?? ''), thread: o.thread ? String(o.thread) : undefined,
      msg: String(o.msg ?? ''), exc: o.exc ? String(o.exc) : undefined }
  } catch {
    return { raw, i }
  }
}

function size(b: number): string {
  return b >= 1 << 20 ? `${(b / (1 << 20)).toFixed(1)} MB` : b >= 1024 ? `${(b / 1024).toFixed(0)} KB` : `${b} B`
}

/** Lines per level in ``BUCKETS`` equal slices of the tail's time span: the level widgets' bar sparklines. */
function histogram(lines: Line[]): Record<Level, number[]> {
  const out: Record<Level, number[]> = { INFO: Array(BUCKETS).fill(0), WARNING: Array(BUCKETS).fill(0), ERROR: Array(BUCKETS).fill(0) }
  const ts = lines.map((l) => (l.ts ? Date.parse(l.ts) : NaN))
  const ok = ts.filter((t) => Number.isFinite(t))
  if (ok.length < 2) return out
  const lo = Math.min(...ok), span = Math.max(...ok) - lo || 1
  lines.forEach((l, k) => {
    const b = bucket(l.level), t = ts[k]
    if (b && Number.isFinite(t)) out[b][Math.min(BUCKETS - 1, Math.floor(((t - lo) / span) * BUCKETS))]++
  })
  return out
}

function PrettyLine({ l, name }: { l: Line; name: string }) {
  if (l.level === undefined) return <div className="mono" style={{ color: 'var(--fly-muted)', padding: '0.2rem 0' }}>{l.raw}</div>
  const short = (l.logger ?? '').replace(/^fly_trader\./, '')
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '148px 84px minmax(0, 1fr)', gap: '0.65rem', padding: '0.32rem 0', borderBottom: '1px solid rgb(255 255 255 / 6%)', alignItems: 'start' }}>
      <span className="mono" style={{ color: 'rgb(255 255 255 / 60%)', whiteSpace: 'nowrap' }}>{(l.ts ?? '').replace('T', ' ')}</span>
      <span><span className={`fbadge ${LEVEL_COLOR[l.level] ?? 'gray'}`} style={{ padding: '0.2rem 0.5rem', fontSize: 11, minWidth: 70, justifyContent: 'center' }}>{l.level}</span></span>
      <div style={{ minWidth: 0 }}>
        <span style={{ color: '#d6c6ff' }}>{short}</span>
        {l.thread && l.thread !== name && <span style={{ color: 'var(--fly-muted)' }}> [{l.thread}]</span>}
        <span style={{ whiteSpace: 'pre-wrap', wordBreak: 'break-word', color: '#fff' }}> {l.msg}</span>
        {l.exc && (
          <details className="mt-1">
            <summary style={{ cursor: 'pointer', color: '#ff8da1' }}>traceback</summary>
            <pre className="log" style={{ maxHeight: 320, marginTop: 4 }}>{l.exc}</pre>
          </details>
        )}
      </div>
    </div>
  )
}

export default function Logs() {
  const files = useTopic<{ files: FileInfo[]; dir: string }>('logs.files')
  const [name, setName] = useStored<string>('logs_name', 'runner')
  const [n, setN] = useStored<(typeof SIZES)[number]>('logs_n', '500')
  const [levels, setLevels] = useStored<Level[]>('logs_levels', [...LEVELS])
  const [view, setView] = useStored<'pretty' | 'raw'>('logs_view', 'pretty')
  const [follow, setFollow] = useStored('logs_follow', true)
  const [text, setText] = useState('')
  const names = files.data?.files.map((f) => f.name) ?? []
  const pick = names.length && !names.includes(name) ? 'runner' : name
  const tail = useTopic<Tail>('logs.tail', { name: pick, n: Number(n) })
  // paused: hold the lines of this log as they were (switching log or length while paused reads the new one once)
  const [snap, setSnap] = useState<Tail | null>(null)
  const fits = (t: Tail | null | undefined): t is Tail => !!t && t.name === pick && t.n === Number(n)
  useEffect(() => {
    if (follow) setSnap(null)
    else if (!fits(snap) && fits(tail.data)) setSnap(tail.data)
  }, [follow, tail.data, snap]) // eslint-disable-line react-hooks/exhaustive-deps
  const data = !follow && fits(snap) ? snap : tail.data
  const state = { ...tail, data }
  const now = useNow(5000)
  const scroller = useRef<HTMLDivElement>(null)
  const lines = useMemo(() => (data?.lines ?? []).map(parse), [data])
  const needle = text.trim().toLowerCase()
  const shown = useMemo(() => lines.filter((l) => {
    const b = bucket(l.level)
    if (b && !levels.includes(b)) return false
    return !needle || l.raw.toLowerCase().includes(needle)
  }), [lines, levels, needle])
  const counts = useMemo(() => {
    const c: Record<string, number> = { INFO: 0, WARNING: 0, ERROR: 0 }
    for (const l of lines) { const b = bucket(l.level); if (b) c[b]++ }
    return c
  }, [lines])
  const hist = useMemo(() => histogram(lines), [lines])
  useEffect(() => {
    if (follow && scroller.current) scroller.current.scrollTop = scroller.current.scrollHeight
  }, [shown, follow, view])
  const fileList = files.data?.files ?? []
  const biggest = Math.max(1, ...fileList.map((f) => f.size))
  const ageS = data?.mtime ? (now - Date.parse(data.mtime)) / 1000 : null
  const title = TITLES[pick] ?? pick

  return (
    <>
      <PageHead title="Logs" icon="bx-terminal" chain={false}>
        <Segmented options={['pretty', 'raw'] as const} value={view} onChange={setView} labels={{ pretty: 'Pretty', raw: 'Raw' }} />
      </PageHead>
      <WidgetGrid min={200}>
        <Widget label="Errors" value={int(counts.ERROR)} icon="error" color={counts.ERROR ? 'danger' : 'dark'} delta={`in the last ${int(lines.length)} lines`} tone="off"
          spark={counts.ERROR ? hist.ERROR : null} progress={counts.ERROR ? null : 0} sparkType="bar" help="ERROR and CRITICAL lines in the tail shown, spread over its time span." />
        <Widget label="Warnings" value={int(counts.WARNING)} icon="warning" color={counts.WARNING ? 'warning' : 'dark'} delta={`in the last ${int(lines.length)} lines`} tone="off"
          spark={counts.WARNING ? hist.WARNING : null} progress={counts.WARNING ? null : 0} sparkType="bar" />
        <Widget label="Info" value={int(counts.INFO)} icon="info" color="info" delta={lines.length ? `${((100 * counts.INFO) / lines.length).toFixed(0)}% of the tail` : null} tone="off"
          spark={lines.length ? hist.INFO : null} sparkType="bar" />
        <Widget label="Last written" value={!data ? '—' : data.exists ? ago(data.mtime, now) : 'no file'} icon="time"
          color={!data?.exists ? 'dark' : ageS !== null && ageS < 600 ? 'success' : 'warning'} delta={data?.exists ? `${size(data.size)} · ${title}` : title} tone="off"
          help="When the worker last wrote this file. A running worker writes at least every few minutes; a stale file usually means it is stopped." />
      </WidgetGrid>
      <div className="row g-4">
        <div className="col-12 col-xl-3">
          <Card title="Log files" icon="folder" sub={files.data ? <code title={files.data.dir}>…/{files.data.dir.split('/').slice(-2).join('/')}</code> : undefined}>
            <Loaded state={files} height={300}>
              {() => (
                <ProgressList max={biggest} items={fileList.map((f) => ({
                  label: <span style={{ fontWeight: f.name === pick ? 700 : 400, opacity: f.exists ? 1 : 0.5 }}>
                    {f.name === pick && <Icon name="bx-chevron-right" />}{TITLES[f.name] ?? f.name}
                    <span className="mono" style={{ fontSize: 11.5, color: 'var(--fly-muted)', marginLeft: 6 }}>{f.name}</span>
                  </span>,
                  value: f.exists ? <span title={f.mtime ? `written ${ago(f.mtime, now)}` : undefined}>{size(f.size)}</span> : 'no file',
                  frac: f.size, onClick: () => setName(f.name),
                }))} />
              )}
            </Loaded>
          </Card>
        </div>
        <div className="col-12 col-xl-9">
          <Card title={`${title} log`} icon="logs" sub="the rotating file the worker writes, newest at the bottom"
            actions={<button type="button" className="btn btn-sm btn-light" title="Download the lines shown" disabled={!shown.length}
              onClick={() => download(`${pick}.log`, shown.map((l) => l.raw).join('\n') + '\n', 'text/plain')}><Icon name="download" />Download</button>}>
            <div className="toolbar">
              <div className="field">
                <label className="form-label" htmlFor="log-name">Log</label>
                <select id="log-name" className="form-select form-select-sm" style={{ minWidth: 220 }} value={pick} onChange={(e) => setName(e.target.value)}>
                  {(fileList.length ? fileList : [{ name: pick, exists: true, size: 0, mtime: null }]).map((f) => (
                    <option key={f.name} value={f.name}>{TITLES[f.name] ? `${TITLES[f.name]} (${f.name})` : f.name}{f.exists ? '' : ' — no file'}</option>
                  ))}
                </select>
              </div>
              <div className="field">
                <label className="form-label" htmlFor="log-n">Lines</label>
                <select id="log-n" className="form-select form-select-sm" value={n} onChange={(e) => setN(e.target.value as (typeof SIZES)[number])}>
                  {SIZES.map((s) => <option key={s} value={s}>last {s}</option>)}
                </select>
              </div>
              <div className="field">
                <span className="form-label">Levels</span>
                <Pills options={LEVELS} value={levels} onChange={setLevels} />
              </div>
              <div className="field" style={{ flex: '1 1 200px' }}>
                <label className="form-label" htmlFor="log-q">Search</label>
                <input id="log-q" className="form-control form-control-sm" placeholder="Filter lines by text…" value={text} onChange={(e) => setText(e.target.value)} />
              </div>
              <div className="field">
                <span className="form-label">&nbsp;</span>
                <Toggle label="Follow" checked={follow} onChange={setFollow} help="On: new lines arrive every second and the view stays at the bottom. Off: the lines hold still to read." />
              </div>
            </div>
            <Loaded state={state} height={300}>
              {(d) => !d.exists ? (
                <Callout kind="info">No log file for <code>{d.name}</code> yet: the worker has not run on this install (files live in <code>{files.data?.dir ?? 'logs/'}</code>).</Callout>
              ) : (
                <>
                  {view === 'raw' ? (
                    <div ref={scroller} style={{ maxHeight: 640, overflow: 'auto' }}>
                      {shown.length ? <pre className="log" style={{ maxHeight: 'none', overflow: 'visible' }}>{shown.map((l) => l.raw).join('\n')}</pre> : <Empty>No line matches.</Empty>}
                    </div>
                  ) : (
                    <div ref={scroller} className="mono" style={{ maxHeight: 640, overflow: 'auto', fontSize: 12, background: 'rgb(0 0 0 / 28%)', borderRadius: 8, padding: '0.5rem 0.85rem' }}>
                      {shown.length ? shown.map((l) => <PrettyLine key={l.i} l={l} name={d.name} />) : <Empty>No line matches.</Empty>}
                    </div>
                  )}
                  <Caption>
                    {shown.length === lines.length ? `${lines.length} lines` : `${shown.length} of ${lines.length} lines`} ·{' '}
                    <code>{(files.data?.dir ?? 'logs')}/{d.name}.log</code> · {size(d.size)} · written {d.mtime ? ago(d.mtime, now) : '—'}
                    {!follow && ' · paused'}. Times are the console host's local clock (the logger's own); rotated files (<code>.log.1</code>…) are not shown.
                  </Caption>
                </>
              )}
            </Loaded>
          </Card>
        </div>
      </div>
    </>
  )
}
