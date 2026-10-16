// Safety & wallet: both chains' bot wallets (balance in USD, withdrawals to your own wallets), the trading rails and their
// controls, the configured mode and limits, maintenance (fly_trader/ui/api/safety.py). Dashtreme layout: a KPI row of colour
// widgets for the rails, wallet cards with widget tiles, each circuit's rails as colour state tiles with their controls beside
// its live drawdown gauge, clean settings tables, a danger zone.
import { useEffect, useRef, useState, type ReactNode } from 'react'
import { action, ApiError } from '../lib/api'
import { useNow, useTopic, type TopicState } from '../lib/hooks'
import { ago, amount, dt, int, num, pct, shortAddr, usd } from '../lib/format'
import { Address, Callout, Caption, Card, Empty, Expander, Help, Icon, KVList, Loaded, Md, Metric, Metrics, ProgressList, SectionTitle, Sparkline, Tip, Toggle,
         Widget, WidgetGrid, type WidgetColor } from '../components/ui'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { RadialGauge } from '../components/Charts'
import { ActionButton, Modal, useFeedback } from '../components/feedback'
import { PageHead } from '../components/ChainPicker'

type Chain = 'sol' | 'rh'
type Summary = {
  chain: Chain; unit: string; address: string | null; native: number | null; error: string | null; n_open: number; positions: number
  marked_at: string | null; price_usd: number | null; price_at: string | null; native_usd: number | null; positions_usd: number | null; total_usd: number | null
}
type Hist = { when: string; amount: number | null; to: string | null; status: string | null; tx: string | null; tx_url: string | null }
type WalletEvent = { ts: string; kind: string; pubkey: string | null }
type WalletChain = {
  chain: Chain; name: string; unit: string; price_source: string; no_wallet: string; setting: string; summary: Summary
  explorer: string | null; address: string | null; blockers: string[]; history: Hist[]; events: WalletEvent[]
}
type Plan = {
  chain: Chain; unit: string; setting: string; destinations: string[]; blockers: string[]; step: number; to: string | null; error: string | null
  all?: boolean; amount?: number | null; max?: number; everything_label?: string; placeholder?: string; value?: number; value_usd?: number | null
  why?: string | null; show_why?: boolean; summary?: string | null; button?: string; send?: string
}
type ResultStatus = 'confirmed' | 'pending' | 'expired' | 'failed' | 'refused' | 'error'
type Result = { status: ResultStatus; error?: string | null; tx?: string | null; tx_url?: string | null; amount?: number | null; to?: string | null; to_short?: string; unit?: string }
type Circuit = {
  id: number; section: string; title: string; scope: string; kill_switch: boolean; kill_reason: string | null; tripped: boolean; fail_count: number
  entries_paused: boolean; halted: { book: string; reason: string | null }[]
}
type SettingRow = { setting: string; value: string; meaning: string }
type Settings = { tables: Record<string, { title: string; rows: SettingRow[] }>; caption: string }
type KAccount = {
  enabled: boolean; key?: string; subaccount?: number | null; wealth?: number | null; cash?: number | null; ts?: string | null
  open_n?: number; open_usd?: number; resting_n?: number; resting_usd?: number
}
type Maint = { runner_alive: boolean; kalshi: boolean; kalshi_runner_alive: boolean; archive_dir: string; resetting: boolean; observer: boolean }
type ApiErr = { ts: string; service: string; endpoint: string; status: number | null; error: string | null }

type LiveBook = {
  id: number; book: string; unit: string; limit: number; wealth: number | null; peak: number | null; drawdown: number | null; ts: string | null
  n_open: number | null; note: string | null; series: [string, number][]
}

const SETTINGS_KEY: Record<number, string> = { 1: 'sol', 3: 'rh', 2: 'kalshi' }

function newId(): string {
  try {
    return crypto.randomUUID()
  } catch {
    return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`
  }
}

export default function Safety() {
  const wal = useTopic<{ chains: WalletChain[] }>('safety.wallets')
  const rails = useTopic<{ circuits: Circuit[]; observer: boolean }>('safety.rails')
  const settings = useTopic<Settings>('safety.settings')
  const live = useTopic<{ books: LiveBook[] }>('safety.live_books')
  const [open, setOpen] = useState<{ chain: WalletChain; id: string } | null>(null)
  return (
    <>
      <PageHead title="Safety & wallet" icon="bx-shield-quarter" chain={false} />
      <SafetyKpis wallets={wal.data} circuits={rails.data?.circuits} />
      <SectionTitle>Wallets</SectionTitle>
      <Loaded state={wal} height={240}>
        {(d) => (
          <div className="row g-4">
            {d.chains.map((c) => (
              <div key={c.chain} className={d.chains.length > 1 ? 'col-12 col-xl-6' : 'col-12'}>
                <WalletCard c={c} onWithdraw={() => setOpen({ chain: c, id: newId() })} />
              </div>
            ))}
          </div>
        )}
      </Loaded>
      {open && <WithdrawDialog key={open.id} opening={open.id} wallet={open.chain} onClose={() => setOpen(null)} />}
      <Loaded state={rails} height={220}>
        {(d) => d.circuits.map((c) => (
          <section key={c.id} className="mt-4">
            <SectionTitle>{c.section}</SectionTitle>
            <div className="row g-4">
              <div className="col-12 col-xl-8"><RailsCard c={c} observer={d.observer} /></div>
              <div className="col-12 col-xl-4"><DrawdownCard b={live.data?.books.find((b) => b.id === c.id)} loading={!live.data} /></div>
              <div className={c.id === 2 ? 'col-12 col-xl-7' : 'col-12'}><SettingsCard state={settings} k={SETTINGS_KEY[c.id]} /></div>
              {c.id === 2 && <div className="col-12 col-xl-5"><KalshiAccount /></div>}
            </div>
          </section>
        ))}
      </Loaded>
      <section className="mt-4">
        <SectionTitle>Maintenance</SectionTitle>
        <Maintenance />
      </section>
    </>
  )
}

/** The page's KPI row: the bot wallets in USD, then every circuit's rails summed up as colour widgets. */
function SafetyKpis({ wallets, circuits }: { wallets?: { chains: WalletChain[] }; circuits?: Circuit[] }) {
  if (!circuits) return <div className="skeleton mb-4" style={{ height: 112 }} />
  const kill = circuits.filter((c) => c.kill_switch), trip = circuits.filter((c) => c.tripped), paused = circuits.filter((c) => c.entries_paused)
  const fails = circuits.reduce((a, c) => a + c.fail_count, 0), halted = circuits.flatMap((c) => c.halted)
  const totals = (wallets?.chains ?? []).map((c) => c.summary.total_usd).filter((v): v is number => v !== null && v !== undefined)
  const names = (xs: Circuit[]) => xs.map((c) => c.scope).join(', ')
  return (
    <WidgetGrid min={200}>
      <Widget label="Bot wallets" value={!wallets ? '…' : totals.length ? usd(totals.reduce((a, b) => a + b, 0)) : '—'} icon="wealth"
        delta={wallets ? wallets.chains.map((c) => `${c.name} ${usd(c.summary.total_usd)}`).join(' · ') : null} tone="off"
        help="Balance plus open live positions, in USD, across the bot wallets." />
      <Widget label="Kill switches" value={kill.length ? `${kill.length} tripped` : 'All armed'} icon="shield" color={kill.length ? 'danger' : 'success'}
        delta={kill.length ? names(kill) : `${circuits.length} circuit${circuits.length === 1 ? '' : 's'} watched`} tone="off"
        help="A kill switch trips when its live book falls too far below its peak wealth, and blocks new entries until cleared." />
      <Widget label="Circuit breakers" value={trip.length ? `${trip.length} tripped` : 'All ok'} icon="bx-pulse" color={trip.length ? 'danger' : fails ? 'warning' : 'success'}
        delta={trip.length ? names(trip) : `${fails} recorded failure${fails === 1 ? '' : 's'}`} tone="off"
        help="A circuit trips after repeated failed or unverifiable live orders, and blocks new live entries." />
      <Widget label="New entries" value={paused.length ? 'Paused' : 'Allowed'} icon={paused.length ? 'pause' : 'play'} color={paused.length ? 'warning' : 'success'}
        delta={paused.length ? names(paused) : 'on every circuit'} tone="off" help="The operator's switch: no new positions while paused; open positions still exit." />
      <Widget label="Halted paper books" value={halted.length ? int(halted.length) : 'None'} icon="block" color={halted.length ? 'danger' : 'success'}
        delta={halted.length ? halted.map((b) => b.book).join(', ') : 'every book trades'} tone="off"
        help="A paper book that fell too far below its own peak stops entering; the other books trade on." />
    </WidgetGrid>
  )
}

// ---------------------------------------------------------------- wallets
const histCols = (unit: string): Col<Hist>[] => [
  { key: 'when', label: 'when', render: (r) => dt(r.when, true) },
  { key: 'amount', label: unit, num: true, render: (r) => num(r.amount, 6) },
  { key: 'to', label: 'to', render: (r) => (r.to ? <code>{r.to}</code> : '—') },
  { key: 'status', label: 'status', render: (r) => <Pill color={r.status === 'confirmed' || r.status === 'mined_ok' ? 'green' : r.status === 'signed' || r.status === 'sent' || r.status === 'pending' ? 'blue' : 'red'}>{r.status ?? '—'}</Pill> },
  { key: 'tx', label: 'transaction', render: (r) => (r.tx_url ? <a href={r.tx_url} target="_blank" rel="noreferrer" title={r.tx ?? ''}><Icon name="external" /></a> : '—') },
]
const EVENT_COLS: Col<WalletEvent>[] = [
  { key: 'ts', label: 'ts', render: (r) => dt(r.ts, true) },
  { key: 'kind', label: 'kind', render: (r) => <Pill color="gray">{r.kind}</Pill> },
  { key: 'pubkey', label: 'pubkey', render: (r) => (r.pubkey ? <code>{r.pubkey}</code> : '—') },
]

function WalletCard({ c, onWithdraw }: { c: WalletChain; onWithdraw: () => void }) {
  const now = useNow(10000)
  const w = c.summary
  const blocked = c.blockers.length > 0
  const actions = (
    <>
      {c.explorer && <a className="btn btn-sm btn-light" href={c.explorer} target="_blank" rel="noreferrer"><Icon name="external" />Explorer</a>}
      {c.address && (
        <Tip text={blocked ? c.blockers.join('; ') : undefined}>
          <button type="button" className="btn btn-sm btn-primary" disabled={blocked} onClick={onWithdraw}><Icon name="bx-export" />Withdraw</button>
        </Tip>
      )}
    </>
  )
  return (
    <Card title={`${c.name} bot wallet`} icon="wallet" actions={actions} className="h-100">
      {!w.address
        ? <Md className="caption" text={c.no_wallet} />
        : (
          <>
            <Address value={w.address} />
            <div className="mt-3">
              <Metrics min={170}>
                <Metric label="Balance" value={usd(w.native_usd)} delta={amount(w.native, w.unit)} tone="off" icon="wallet"
                  help="What the wallet holds in its own coin; this is what can be withdrawn." />
                <Metric label="In live positions" value={usd(w.positions_usd)} delta={`${amount(w.positions, w.unit)} · ${w.n_open} open`} tone="off" icon="open"
                  help={"The live book's open positions at its last mark, net of exit cost" + (w.marked_at ? ` (marked ${ago(w.marked_at, now)}).` : '; at cost before the first mark.')} />
                <Metric label="Total" value={usd(w.total_usd)} delta={amount(w.native === null ? null : w.native + w.positions, w.unit)} tone="off" icon="wealth" />
              </Metrics>
            </div>
            <Caption>{w.price_usd ? `1 ${w.unit} = ${usd(w.price_usd)} · ${c.price_source}, ${ago(w.price_at, now)}` : `No ${w.unit}/USD price yet.`}</Caption>
            {w.error && <Callout kind="warning">Balance unavailable: {w.error}</Callout>}
          </>
        )}
      {c.address && (
        <>
          {blocked
            ? <Callout kind="warning" icon="lock">Withdraw: {c.blockers.join('; ')}.</Callout>
            : <Caption>Withdrawals go only to your own wallets ({c.setting} in .env).</Caption>}
          {c.history.length > 0 && (
            <Expander title={`Withdrawals (${c.history.length})`} icon="history">
              <DataTable rows={c.history} cols={histCols(c.unit)} height={320} filename={`${c.chain}-withdrawals`} search={false} />
            </Expander>
          )}
          {c.chain === 'sol' && c.events.length > 0 && (
            <Expander title="Wallet events" icon="wallet">
              <DataTable rows={c.events} cols={EVENT_COLS} height={320} filename="wallet-events" search={false} />
            </Expander>
          )}
        </>
      )}
    </Card>
  )
}

function useDebounced<T>(value: T, ms: number): T {
  const [v, setV] = useState(value)
  useEffect(() => {
    const h = setTimeout(() => setV(value), ms)
    return () => clearTimeout(h)
  }, [value, ms])
  return v
}

/** What the amount box holds, as the server will read it (6 decimals); null when empty or not a number ≥ 0. */
function parseAmount(text: string): number | null {
  if (!text.trim()) return null
  const v = Number(text)
  return Number.isFinite(v) && v >= 0 ? Math.round(v * 1e6) / 1e6 : null
}

type Frozen = { to: string; all: boolean; amount: number | null; send: string; usd: number | null | undefined }

/**
 * The withdraw dialog: one withdrawal per opening. The form reads `safety.withdraw_plan` (read-only: fresh limits, what would
 * go out, or why not); "Withdraw X" asks once more naming the amount and the full address; the send's outcome then replaces
 * the form for good. The server re-checks everything and refuses a second send from the same opening.
 */
function WithdrawDialog({ wallet, opening, onClose }: { wallet: WalletChain; opening: string; onClose: () => void }) {
  const fb = useFeedback()
  const [to, setTo] = useState<string | null>(null)
  const [all, setAll] = useState(false)
  const [text, setText] = useState('')
  const [stage, setStage] = useState<'form' | 'confirm' | 'sending'>('form')
  const [frozen, setFrozen] = useState<Frozen | null>(null)
  const [result, setResult] = useState<Result | null>(null)
  const [last, setLast] = useState<Plan | undefined>(undefined)
  const sent = useRef(false)
  const mounted = useRef(true)
  useEffect(() => {
    mounted.current = true
    return () => { mounted.current = false }
  }, [])
  const typed = parseAmount(text)
  const amt = useDebounced(typed, 300)
  const plan = useTopic<Plan>('safety.withdraw_plan', { chain: wallet.chain, to, all, amount: all ? null : amt }, result === null && stage !== 'sending')
  useEffect(() => {
    if (plan.data) setLast(plan.data)
  }, [plan.data])
  const p = plan.data ?? last
  const fresh = !!p && p.to !== null && p.to === (to ?? p.to) && p.all === all && (all || (p.amount ?? null) === typed)
  const ready = fresh && !p?.error && !p?.why && !(p?.blockers.length) && stage === 'form'

  const finish = (r: Result) => {
    if (mounted.current) {
      setResult(r)
      return
    }
    const kind = r.status === 'confirmed' ? 'success' : r.status === 'pending' || r.status === 'expired' ? 'warning' : 'error'
    fb.toast(`${wallet.name} withdrawal: ${resultText(r)}`, kind, 20000)
  }
  const send = async () => {
    if (sent.current || !frozen) return
    sent.current = true
    setStage('sending')
    try {
      const r = await action<{ result: Result }>('safety.withdraw', {
        chain: wallet.chain, to: frozen.to, amount: frozen.all ? null : frozen.amount, all: frozen.all, opening,
      })
      finish(r.result)
    } catch (e) {
      const status = e instanceof ApiError ? e.status : 0
      // 400/403/404/409: refused before anything was signed; anything else (500, a dropped connection): the outcome is unknown
      finish([400, 403, 404, 409].includes(status) ? { status: 'refused', error: (e as Error).message } : { status: 'error', error: (e as Error).message })
    }
  }
  const close = () => {
    if (stage === 'sending') fb.toast('The withdrawal is still running: its outcome will show as a notice here and in the wallet\'s withdrawals.', 'info', 10000)
    onClose()
  }
  const review = () => {
    if (!ready || !p || !p.to) return
    setFrozen({ to: p.to, all, amount: all ? null : typed, send: p.send ?? '', usd: p.value_usd })
    setStage('confirm')
  }

  let body: ReactNode
  let foot: ReactNode
  if (result) {
    body = <ResultView r={result} wallet={wallet} />
    foot = <button type="button" className="btn btn-light" onClick={onClose}>Close</button>
  } else if (!p) {
    body = plan.error ? <Callout kind="error">{plan.error}</Callout> : <div className="skeleton" style={{ height: 160 }} />
    foot = <button type="button" className="btn btn-light" onClick={close}>Cancel</button>
  } else if (!p.destinations.length) {
    body = <Callout kind="warning">None of your own wallets is listed: add yours to {p.setting} in .env and restart the console.</Callout>
    foot = <button type="button" className="btn btn-light" onClick={close}>Close</button>
  } else {
    const locked = stage !== 'form'
    body = (
      <>
        {p.blockers.length > 0 && <Callout kind="error">Withdraw: {p.blockers.join('; ')}.</Callout>}
        {plan.error && <Callout kind="error">{plan.error}</Callout>}
        <div className="mb-3">
          <label className="form-label d-flex align-items-center gap-1">To <Help text={`Only your own wallets: ${p.setting} in .env.`} /></label>
          <select className="form-select" value={to ?? p.to ?? ''} disabled={locked} onChange={(e) => setTo(e.target.value)}>
            {p.destinations.map((d) => <option key={d} value={d}>{d}</option>)}
          </select>
        </div>
        {p.error
          ? <Callout kind="error">{p.error}</Callout>
          : (
            <>
              <div className="mb-3">
                <Toggle label={p.everything_label ?? 'Everything withdrawable'} checked={all} disabled={locked} onChange={setAll} />
              </div>
              {!all && (
                <div className="mb-3">
                  <label className="form-label" htmlFor={`${wallet.chain}-wd-amt`}>Amount ({p.unit})</label>
                  <input id={`${wallet.chain}-wd-amt`} className="form-control num" type="number" inputMode="decimal" min={0} step={p.step}
                    placeholder={p.placeholder} value={text} disabled={locked} onChange={(e) => setText(e.target.value)}
                    onBlur={() => { const v = parseAmount(text); if (v !== null) setText(v.toFixed(6)) }} />
                </div>
              )}
              {fresh && p.show_why && p.why && <Callout kind="warning">{p.why}</Callout>}
              {fresh && !p.why && p.summary && <Md text={p.summary} />}
            </>
          )}
        <Caption>This waits for the network's confirmation, up to about a minute.</Caption>
        {stage === 'confirm' && frozen && (
          <Callout kind="warning" icon="payments">
            Send <strong>{frozen.send}</strong>{frozen.usd ? ` (${usd(frozen.usd)})` : ''} to <code style={{ wordBreak: 'break-all' }}>{frozen.to}</code>?
            {' '}This signs and broadcasts a real transaction from the bot wallet; it cannot be undone.
          </Callout>
        )}
        {stage === 'sending' && (
          <div className="d-flex align-items-center gap-2 caption"><span className="spinner-border spinner-border-sm" />Signing, sending, waiting for confirmation…</div>
        )}
      </>
    )
    foot = stage === 'form'
      ? (
        <>
          <button type="button" className="btn btn-light" onClick={close}>Cancel</button>
          <button type="button" className="btn btn-primary" disabled={!ready} onClick={review}>
            <Icon name="bx-send" />{ready && p.button ? p.button : 'Withdraw'}
          </button>
        </>
      )
      : stage === 'confirm'
        ? (
          <>
            <button type="button" className="btn btn-light" onClick={() => { setStage('form'); setFrozen(null) }}>Back</button>
            <button type="button" className="btn btn-danger" autoFocus onClick={send}><Icon name="bx-send" />Send {frozen?.send}</button>
          </>
        )
        : <button type="button" className="btn btn-primary" disabled><span className="spinner-border spinner-border-sm" /> Sending…</button>
  }
  return <Modal title={`Withdraw · ${wallet.name}`} icon="bx-export" onClose={result ? onClose : close} foot={foot}>{body}</Modal>
}

function resultText(r: Result): string {
  const what = `${amount(r.amount, r.unit, 6)} to ${r.to_short ?? shortAddr(r.to)}`
  switch (r.status) {
    case 'confirmed': return `sent ${what}`
    case 'pending': return `sent ${what}, no receipt yet: it is booked when it lands`
    case 'expired': return 'the transaction expired before it landed: nothing moved'
    case 'failed': return 'the transaction failed on chain: only the fee moved'
    case 'refused': return `nothing was sent: ${r.error}`
    default: return `${r.error}. Whether it went out is unknown: look at the wallet on the explorer before trying again`
  }
}

function ResultView({ r, wallet }: { r: Result; wallet: WalletChain }) {
  const link = r.tx_url ? <> · <a href={r.tx_url} target="_blank" rel="noreferrer">transaction</a></> : null
  const what = <>{amount(r.amount, r.unit ?? wallet.unit, 6)} to <code>{r.to_short ?? shortAddr(r.to)}</code></>
  switch (r.status) {
    case 'confirmed': return <Callout kind="success">Sent {what}{link}</Callout>
    case 'pending': return <Callout kind="info">Sent {what}, no receipt yet: it is booked when it lands. Look at the transaction before sending again{link}</Callout>
    case 'expired': return <Callout kind="warning">The transaction expired before it landed: nothing moved. You can try again.</Callout>
    case 'failed': return <Callout kind="error">The transaction failed on chain: only the fee moved{link}</Callout>
    case 'refused': return <Callout kind="error">Nothing was sent: {r.error}</Callout>
    default:
      return (
        <Callout kind="error">
          {r.error}. Whether it went out is unknown: look at the wallet on the {wallet.explorer ? <a href={wallet.explorer} target="_blank" rel="noreferrer">explorer</a> : 'explorer'} before trying again.
        </Callout>
      )
  }
}


// ---------------------------------------------------------------- rails
/** A colour static widget for one rail: its state, what it means, and its control. */
function StateTile({ label, value, delta, icon, color, text, control }: {
  label: string; value: string; delta?: string | null; icon: string; color: WidgetColor; text: ReactNode; control: ReactNode
}) {
  return (
    <div className={`widget cw cw-${color} h-100 d-flex flex-column`} style={{ borderRadius: 10, color: color === 'warning' ? '#241a00' : '#fff' }}>
      <div className="w-body">
        <div className="w-text">
          <p className="w-label">{label}</p>
          <h4 className="w-value">{value}</h4>
          {delta && <p className="w-delta">{delta}</p>}
        </div>
        <div className="w-icon"><Icon name={icon} /></div>
      </div>
      <div style={{ padding: '0 1.1rem 0.8rem', fontSize: 12.5, lineHeight: 1.45, opacity: 0.92, flex: 1 }}>{text}</div>
      <div style={{ padding: '0 1.1rem 1rem' }}>{control}</div>
    </div>
  )
}

const OBSERVER_HELP = 'This console only observes (fly-trader ui --observe): change the rails from the console that hosts the workers.'

function RailsCard({ c, observer }: { c: Circuit; observer: boolean }) {
  const settle = c.id === 2 ? 'open positions still run to settlement' : 'open positions still exit on schedule'
  return (
    <Card title={c.title} icon="shield" className="h-100">
      <div className="row g-3">
        <div className="col-12 col-md-4">
          <StateTile label="Kill switch" value={c.kill_switch ? 'Tripped' : 'Armed'} icon={c.kill_switch ? 'bx-shield-x' : 'shield'} color={c.kill_switch ? 'danger' : 'success'}
            text={<>Trips when the live book falls too far below its peak wealth; blocks new entries until cleared.{c.kill_reason ? ` Reason: ${c.kill_reason}.` : ''}</>}
            control={
              <ActionButton label="Clear kill switch" icon="reset" name="safety.kill_clear" body={{ circuit: c.id }} disabled={!c.kill_switch || observer} variant="light"
                help={observer ? OBSERVER_HELP : 'Clears the switch and re-bases the peak at current wealth.'}
                confirm={{
                  title: 'Clear the kill switch?', confirmLabel: 'Clear kill switch', danger: true,
                  body: (
                    <>
                      <p>Clearing the kill switch lets the engine open new {c.scope} positions again. The live book's peak re-bases at its current wealth, so a new drawdown is measured from now; this circuit's failure count is reset too.</p>
                      {c.kill_reason && <p className="mb-0">It tripped because: {c.kill_reason}.</p>}
                    </>
                  ),
                }} />
            } />
        </div>
        <div className="col-12 col-md-4">
          <StateTile label="Circuit breaker" value={c.tripped ? 'Tripped' : 'OK'} delta={`${c.fail_count} failures`} icon="bx-pulse"
            color={c.tripped ? 'danger' : c.fail_count ? 'warning' : 'success'}
            text="Trips after repeated failed or unverifiable live orders; blocks new live entries."
            control={
              <ActionButton label="Reset circuit" icon="refresh" name="safety.circuit_reset" body={{ circuit: c.id }} disabled={(!c.tripped && !c.fail_count) || observer} variant="light" help={observer ? OBSERVER_HELP : undefined}
                confirm={{
                  title: 'Reset the circuit breaker?', confirmLabel: 'Reset circuit', danger: true,
                  body: <p className="mb-0">Resetting clears its {c.fail_count} recorded failure{c.fail_count === 1 ? '' : 's'}{c.tripped ? ` and lets new live ${c.scope} entries through again` : ''}. It trips again after repeated failed or unverifiable live orders.</p>,
                }} />
            } />
        </div>
        <div className="col-12 col-md-4">
          <StateTile label="New entries" value={c.entries_paused ? 'Paused' : 'Allowed'} icon={c.entries_paused ? 'pause' : 'play'} color={c.entries_paused ? 'warning' : 'success'}
            text="Operator switch: no new positions while paused; open positions still exit on schedule."
            control={
              <ActionButton label={c.entries_paused ? 'Resume entries' : 'Pause entries'} icon={c.entries_paused ? 'play' : 'pause'} name="safety.entries" variant="light"
                body={{ circuit: c.id, paused: !c.entries_paused }} disabled={observer} help={observer ? OBSERVER_HELP : undefined}
                confirm={c.entries_paused
                  ? { title: 'Resume new entries?', confirmLabel: 'Resume entries', body: <p className="mb-0">The engine may open new {c.scope} positions again (the kill switch, the circuit and the book halts still apply).</p> }
                  : { title: 'Pause new entries?', confirmLabel: 'Pause entries', body: <p className="mb-0">No new {c.scope} positions open while paused; {settle}.</p> }} />
            } />
        </div>
      </div>
      <div className="mt-3">
        {c.halted.length === 0
          ? <Caption><Icon name="success" className="me-1" />No paper book of this circuit is halted.</Caption>
          : (
            <ul className="kv">
              {c.halted.map((b) => (
                <li key={b.book} className="d-flex align-items-center gap-3 flex-wrap">
                  <Pill color="red">{b.book} halted</Pill>
                  <span className="caption m-0 flex-grow-1" style={{ flexBasis: 240 }}>
                    The paper book fell too far below its own peak ({b.reason ?? '—'}); its entries are blocked, the other book trades on.
                  </span>
                  <ActionButton label="Clear" icon="reset" name="safety.clear_halt" body={{ circuit: c.id, book: b.book }} variant="light" disabled={observer} help={observer ? OBSERVER_HELP : undefined}
                    confirm={{
                      title: `Clear the halt on ${b.book}?`, confirmLabel: 'Clear halt',
                      body: <p className="mb-0">Clearing the halt lets {b.book} open new positions again; its peak re-bases from its next wealth marks. It halted because it fell too far below its own peak{b.reason ? ` (${b.reason})` : ''}.</p>,
                    }} />
                </li>
              ))}
            </ul>
          )}
      </div>
    </Card>
  )
}

/** The live book against its kill switch: drawdown as a share of the limit that trips it. */
function DrawdownCard({ b, loading }: { b?: LiveBook; loading: boolean }) {
  const now = useNow(10000)
  const money = (v: number | null | undefined) => (b?.unit === 'USD' ? usd(v) : amount(v, b?.unit ?? '', 4))
  const frac = b && b.drawdown !== null ? Math.max(0, b.drawdown) / (b.limit || 1) : 0
  return (
    <Card title="Live drawdown" icon="drawdown" className="h-100">
      {loading || !b
        ? <div className="skeleton" style={{ height: 260 }} />
        : (
          <>
            <RadialGauge value={b.drawdown === null ? 0 : frac} label={`of the ${pct(b.limit, 0, false)} kill limit`}
              display={b.drawdown === null ? '—' : pct(Math.max(0, b.drawdown), 1, false)} color={frac >= 0.8 ? '#f0435a' : frac >= 0.5 ? '#ffc107' : '#ffffff'} height={220} />
            {b.drawdown === null
              ? <Empty>No live wealth marks yet: the live book ({b.book}) is marked every minute once the fly trades it.</Empty>
              : (
                <>
                  <KVList items={[
                    { label: 'Live wealth', value: money(b.wealth) },
                    { label: <>Peak <Help text="The highest wealth mark the kill switch measures from (since the last clear or withdrawal)." /></>, value: money(b.peak) },
                    { label: 'Open positions', value: int(b.n_open) },
                    { label: 'Marked', value: ago(b.ts, now) },
                  ]} />
                  {b.series.length > 1 && <div className="mt-2"><Sparkline data={b.series.map((p) => p[1])} height={48} /></div>}
                </>
              )}
            {b.note && <Caption>{b.note}</Caption>}
          </>
        )}
    </Card>
  )
}

// ---------------------------------------------------------------- mode and limits, the Kalshi subaccount
function SettingsCard({ state, k }: { state: TopicState<Settings>; k: string }) {
  const t = state.data?.tables[k]
  if (!state.data) return <Loaded state={state}>{() => null}</Loaded>
  if (!t) return null
  return (
    <Card title={t.title} icon="bx-slider-alt" className="h-100">
      <div className="ftable-wrap">
        <table className="ftable">
          <thead><tr><th style={{ width: 190 }}>setting</th><th style={{ width: '28%' }}>value</th><th>meaning</th></tr></thead>
          <tbody>
            {t.rows.map((r) => (
              <tr key={r.setting}>
                <td style={{ fontWeight: 500 }}>{r.setting}</td>
                <td className="wrap" style={{ fontFamily: 'var(--fly-mono)', fontSize: 13 }}>{r.value}</td>
                <td className="wrap text-muted2">{r.meaning}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <Caption>{state.data.caption}</Caption>
    </Card>
  )
}

function KalshiAccount() {
  const s = useTopic<KAccount>('safety.kalshi_account')
  const now = useNow(10000)
  return (
    <Card title="Kalshi subaccount" icon="bx-building-house" className="h-100">
      <Loaded state={s}>
        {(d) => (!d.enabled
          ? <Caption>No Kalshi key configured (KALSHI_API_KEY_ID / KALSHI_PRIVATE_KEY_PATH).</Caption>
          : (
            <>
              <Md className="caption" text={`Key ${d.key} · subaccount ${d.subaccount ?? 'none'} · fund it with \`fly-trader kalshi-fund --usd N\`; \`fly-trader kalshi-status\` prints balances, orders and settlements.`} />
              <Metrics min={210}>
                <Metric label="Live wealth" value={usd(d.wealth)} icon="wealth" help={d.ts ? `as of ${ago(d.ts, now)}` : undefined} />
                <Metric label="Cash" value={usd(d.cash)} icon="bx-money" />
                <Metric label="Open live positions" value={int(d.open_n ?? 0)} delta={`${usd(d.open_usd ?? 0)} at cost`} tone="off" icon="open" />
                <Metric label="Resting live orders" value={int(d.resting_n ?? 0)} delta={`${usd(d.resting_usd ?? 0)} collateral`} tone="off" icon="time" />
              </Metrics>
            </>
          ))}
      </Loaded>
    </Card>
  )
}

// ---------------------------------------------------------------- maintenance
const ERR_COLS: Col<ApiErr>[] = [
  { key: 'ts', label: 'ts', width: 150, render: (r) => dt(r.ts, true) },
  { key: 'service', label: 'service', render: (r) => <Pill color="gray">{r.service}</Pill> },
  { key: 'endpoint', label: 'endpoint', wrap: true },
  { key: 'status', label: 'status', num: true, render: (r) => (r.status ? <Pill color={r.status >= 500 ? 'red' : 'orange'}>{r.status}</Pill> : '—') },
  { key: 'error', label: 'error', wrap: true },
]

function DangerRow({ title, text, control }: { title: string; text: ReactNode; control: ReactNode }) {
  return (
    <div className="d-flex align-items-center gap-3 flex-wrap py-3" style={{ borderBottom: '1px solid rgb(255 255 255 / 10%)' }}>
      <div className="flex-grow-1" style={{ flexBasis: 260 }}>
        <div style={{ color: '#fff', fontWeight: 600 }}>{title}</div>
        <div className="caption m-0">{text}</div>
      </div>
      {control}
    </div>
  )
}

function Maintenance() {
  const m = useTopic<Maint>('safety.maintenance')
  const errs = useTopic<{ rows: ApiErr[] }>('safety.api_errors')
  const [done, setDone] = useState<string | null>(null)
  const bySvc = (() => {
    const n: Record<string, number> = {}
    for (const r of errs.data?.rows ?? []) n[r.service] = (n[r.service] ?? 0) + 1
    return Object.entries(n).sort((a, b) => b[1] - a[1])
  })()
  return (
    <div className="row g-4">
      <div className="col-12 col-xl-5">
        <div className="h-100" style={{ borderRadius: 12, boxShadow: '0 0 0 1px rgb(240 67 90 / 65%), 0 0 24px rgb(240 67 90 / 12%)' }}>
        <Card title="Danger zone" icon="bx-error-alt" className="h-100 mb-0">
          <Loaded state={m}>
            {(d) => (
              <>
                <DangerRow title="Reset paper history"
                  text="Archive, then clear old books, their decisions and wealth marks — never the race books (selector, fly), the live book, the fly's learning or the models. Happens automatically when the trading engine starts."
                  control={
                    <ActionButton label="Reset paper history" icon="reset" name="safety.reset_training" variant="danger" disabled={d.runner_alive || d.resetting || d.observer}
                      help={d.observer ? 'This console only observes: reset from the console that hosts the workers.' : d.runner_alive ? 'Stop the trading engine first.' : d.resetting ? 'A reset is running.' : undefined}
                      onDone={(r) => setDone(r.message ?? null)}
                      confirm={{
                        title: 'Reset paper history?', confirmLabel: 'Reset', danger: true,
                        body: (
                          <>
                            <p>Archive and clear the old books now? The race and live books are kept.</p>
                            <p>It archives to <code>{d.archive_dir}/reset_&lt;time&gt;/</code>, then deletes the positions, orders, fills and wealth marks of every book but the race and live books; the decisions and beats of runs that are not race sessions, and those runs; brain snapshots that are not models; and the selector status (written again on the next minute).</p>
                            <p className="mb-0">Never touched: the race books (selector, fly) and the live books of every market with their runs, the fly's learning, the models, the circuits (kill switch, peak), wallet events, market data, API logs and events.</p>
                          </>
                        ),
                      }} />
                  } />
                {d.kalshi && (
                  <DangerRow title="Reset Kalshi fly"
                    text="Archive, then clear the visual fly's paper books, orders, scored rows and learning; it restarts from its bootstrap."
                    control={
                      <ActionButton label="Reset Kalshi fly" icon="reset" name="safety.reset_kalshi_fly" variant="danger" disabled={d.kalshi_runner_alive || d.resetting || d.observer}
                        help={d.observer ? 'This console only observes: reset from the console that hosts the workers.' : d.kalshi_runner_alive ? 'Stop the Kalshi engine first.' : d.resetting ? 'A reset is running.' : undefined}
                        onDone={(r) => setDone(r.message ?? null)}
                        confirm={{
                          title: 'Reset the Kalshi fly?', confirmLabel: 'Reset', danger: true,
                          body: (
                            <>
                              <p>Archive and clear the Kalshi fly's books and learning now?</p>
                              <p>It archives to <code>{d.archive_dir}/reset_kalshi_fly_&lt;time&gt;/</code>, then deletes the taker and maker paper books (positions, orders, fills, wealth marks and their halts), the Kalshi fly's sessions with their decisions and beats, its learned state (plastic snapshots, scored rows, calibrations, updates, rollbacks, its state directory) and its recorded activity. The fly restarts from its bootstrap.</p>
                              <p className="mb-0">Kept: the live Kalshi books and the bootstrap snapshots.</p>
                            </>
                          ),
                        }} />
                    } />
                )}
                <Caption>
                  {d.runner_alive || d.kalshi_runner_alive
                    ? `A reset runs only while its engine is stopped (${[d.runner_alive ? 'the trading engine' : null, d.kalshi_runner_alive ? 'the Kalshi engine' : null].filter(Boolean).join(' and ')} ${d.runner_alive && d.kalshi_runner_alive ? 'are' : 'is'} running): stop it on Processes first.`
                    : 'Both resets archive before they delete; the archive stays under data/pg_archive.'}
                </Caption>
                {done && <Callout kind="success">{done}</Callout>}
              </>
            )}
          </Loaded>
        </Card>
        </div>
      </div>
      <div className="col-12 col-xl-7">
        <Card title={`Recent API errors${errs.data ? ` (${errs.data.rows.length})` : ''}`} icon="error" className="h-100">
          <Loaded state={errs}>
            {(d) => (d.rows.length
              ? (
                <>
                  <ProgressList items={bySvc.slice(0, 4).map(([svc, n]) => ({ label: svc, value: `${n} of ${d.rows.length}`, frac: n / d.rows.length }))} max={1} />
                  <div className="mt-3"><DataTable rows={d.rows} cols={ERR_COLS} height={420} filename="api-errors" /></div>
                </>
              )
              : <Empty>No failed API calls recorded.</Empty>)}
          </Loaded>
        </Card>
      </div>
    </div>
  )
}
