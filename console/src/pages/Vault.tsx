// $FLY vault: the trading wallet's ledger (deposits, profit, claims), settlements, the claims queue, halts and the
// Robinhood Chain index (fly_trader/ui/api/vault.py). Only meaningful on the hosted vault fly (VAULT_ENABLED).
import { useState, type FormEvent, type ReactNode } from 'react'
import { PageHead } from '../components/ChainPicker'
import { BarChart, DonutChart } from '../components/Charts'
import { DataTable, Pill, type Col } from '../components/DataTable'
import { ActionButton, useFeedback } from '../components/feedback'
import { Callout, Caption, Card, KVList, Loaded, Md, Widget, WidgetGrid } from '../components/ui'
import { useApp } from '../context'
import { action } from '../lib/api'
import { dt, num, shortAddr } from '../lib/format'
import { useTopic, type TopicState } from '../lib/hooks'

const OFF = 'The vault is off on this machine (VAULT_ENABLED is not set). It runs on the hosted vault fly.'
type Off = { enabled: false; message: string }
const isOff = (d: { enabled: boolean }): d is Off => d.enabled === false
const off = (d: Off) => <Callout kind="info">{d.message || OFF}</Callout>

type Summary = {
  enabled: true; halted: { reasons: string[] } | null; nav: number; nav_at: string | null; nav_spark: number[]; deposits: number; withdrawn: number; profit: number
  allocated: number; paid: number; owed: number; scan_slot: number | null; rh_block: number | null; rh_updated_at: string | null; impl: string | null
}
type SummaryData = Off | Summary
type Row = Record<string, unknown>
type Tables = { enabled: true; tx_url: string; settlements: Row[]; claims: Row[]; flows: Row[]; holders: Row[] }
type TablesData = Off | Tables

const KINDS = ['deposit', 'profit', 'internal'] as const
type Kind = (typeof KINDS)[number]
const KIND_EFFECT: Record<Kind, string> = {
  deposit: 'It counts as principal you put in, never as profit.',
  profit: "It counts as trading profit, shared with the lockers at the week's settlement.",
  internal: 'It leaves the ledger (a move between our own wallets) and counts in neither direction.',
}
const PILL: Record<string, string> = {
  allocated: 'green', snapshotted: 'blue', failed: 'red', paid: 'green', rejected: 'red', waiting_liquidity: 'orange', received: 'blue', verified: 'blue',
  sending: 'violet', deposit: 'blue', profit: 'green', withdrawal: 'orange', claim: 'violet', unknown_outbound: 'red', anomaly: 'red', in: 'green', out: 'orange',
}
const FLOW_COLORS = ['#ffffff', '#b6ff3b', 'rgba(255,255,255,0.45)', '#ff2fb9']

const sol = (lamports: number) => (lamports / 1e9).toFixed(4)
const Addr = ({ v }: { v: unknown }) => (v ? <span className="mono" title={String(v)}>{shortAddr(String(v))}</span> : null)
const pill = (key: string): Col<Row> => ({ key, label: key, render: (r) => (r[key] ? <Pill color={PILL[String(r[key])] ?? 'gray'}>{String(r[key])}</Pill> : null) })
const solAmt = (key: string, label: string): Col<Row> => ({ key, label, num: true, render: (r) => num(r[key] as number | null, 6) })
const when = (key: string): Col<Row> => ({ key, label: key, render: (r) => dt(r[key] as string | null, true) })

function HaltBanner({ d }: { d: Summary }) {
  if (!d.halted) return null
  const reasons = d.halted.reasons
  return (
    <Callout kind="error">
      <div>Vault halted: {reasons.join('; ')}. Claims and entries are stopped.</div>
      <div className="mt-2">
        <ActionButton label="Resume vault (I checked the reason)" icon="play" name="vault.resume" body={{ reasons }} variant="danger"
          confirm={{
            title: 'Resume the vault?', confirmLabel: 'Resume vault', danger: true,
            body: (
              <>
                <p className="mb-2">The vault halted because:</p>
                <ul>{reasons.map((r, i) => <li key={i}>{r}</li>)}</ul>
                <p className="mb-2">Resuming clears the halt (and a /panic flag), so claims and settlement run again. Only resume once you know why it halted.</p>
                <p className="caption mb-0">New entries stay paused until you resume them on the Safety page. After a /panic, the kill switch and the signer's
                  panic stay until you clear them separately.</p>
              </>
            ),
          }} />
      </div>
    </Callout>
  )
}

function Kpis({ d }: { d: Summary }) {
  return (
    <WidgetGrid>
      <Widget label="Vault" value={d.halted ? 'halted' : 'running'} color={d.halted ? 'danger' : 'success'} icon={d.halted ? 'block' : 'savings'} tone="off"
        delta={d.halted ? 'claims and entries are stopped' : 'claims and entries run'} />
      <Widget label="Wallet NAV" value={`${sol(d.nav)} SOL`} icon="wealth" spark={d.nav_spark.length > 1 ? d.nav_spark.map((v) => v / 1e9) : null}
        delta={d.nav_at ? `marked ${dt(d.nav_at)}` : 'not marked yet'} tone="off" />
      <Widget label="Allocated to lockers" value={sol(d.allocated)} icon="bx-group" delta="SOL, settled weeks" tone="off" />
      <Widget label="Claims paid" value={sol(d.paid)} icon="receipt" progress={d.allocated ? d.paid / d.allocated : null}
        delta={d.allocated ? `${((d.paid / d.allocated) * 100).toFixed(0)}% of the allocated` : 'SOL'} tone="off" />
      <Widget label="Owed (reserved)" value={sol(d.owed)} icon="lock" delta="SOL kept for holders' claims" tone="off" />
    </WidgetGrid>
  )
}

function FlowsCard({ d }: { d: Summary }) {
  const parts = [{ label: 'Deposits', v: d.deposits }, { label: 'Profit received', v: d.profit }, { label: 'Withdrawn', v: d.withdrawn }, { label: 'Claims paid', v: d.paid }]
  const total = parts.reduce((x, p) => x + p.v, 0)
  return (
    <Card title="Wallet ledger" sub="SOL through the trading wallet" icon="swap">
      {total > 0
        ? <DonutChart labels={parts.map((p) => p.label)} values={parts.map((p) => p.v / 1e9)} center={sol(total)} centerLabel="SOL moved" colors={FLOW_COLORS} height={250} />
        : <div className="empty">No wallet flows yet.</div>}
      <KVList items={[
        ...parts.map((p, i) => ({ label: p.label, value: sol(p.v), dot: FLOW_COLORS[i] })),
        { label: 'Allocated to lockers', value: sol(d.allocated) },
        { label: 'Owed (reserved)', value: sol(d.owed) },
      ]} />
      <Caption>
        Wallet scan through slot {d.scan_slot ?? '—'} · Robinhood Chain indexed through block {d.rh_block ?? '—'} (updated {d.rh_updated_at ? dt(d.rh_updated_at, true) : '—'})
        {' '}· implementation {d.impl ?? '—'}
      </Caption>
    </Card>
  )
}

function SettlementsChart({ t }: { t: TopicState<TablesData> }) {
  return (
    <Card title="Weekly settlements" sub="realized profit, the lockers' pot, what was allocated and what carried, SOL" icon="bx-calendar-check">
      <Loaded state={t} height={280}>
        {(d) => {
          if (isOff(d)) return off(d)
          const rows = [...d.settlements].reverse()
          if (!rows.length) return <div className="empty">No settlements yet: the first week settles after it ends.</div>
          const v = (k: string) => rows.map((r) => (r[k] === null || r[k] === undefined ? null : Number(r[k])))
          return (
            <BarChart height={300} categories={rows.map((r) => dt(r.period_end as string).slice(0, 6))} yFormat={(x) => x.toFixed(3)}
              series={[{ name: 'Realized', data: v('realized_sol') }, { name: 'Pot', data: v('pot_sol') }, { name: 'Allocated', data: v('allocated_sol') }, { name: 'Carried', data: v('carried_sol') }]} />
          )
        }}
      </Loaded>
    </Card>
  )
}

function Reclassify({ signature, setSignature }: { signature: string; setSignature: (s: string) => void }) {
  const fb = useFeedback()
  const [kind, setKind] = useState<Kind>('deposit')
  const [busy, setBusy] = useState(false)
  const [result, setResult] = useState<{ ok: boolean; text: string } | null>(null)
  const submit = async (e: FormEvent) => {
    e.preventDefault()
    const sig = signature.trim()
    if (!sig) return
    const ok = await fb.confirm({
      title: 'Relabel this transaction?', confirmLabel: `Relabel as ${kind}`, danger: kind === 'internal',
      body: (
        <>
          <p className="mb-2">Transaction</p>
          <code className="addr mb-2">{sig}</code>
          <p className="mb-2">New label: <strong>{kind}</strong>. {KIND_EFFECT[kind]}</p>
          <p className="caption mb-0">A change after its week settled only moves future weeks.</p>
        </>
      ),
    })
    if (!ok) return
    setBusy(true); setResult(null)
    try {
      const r = await action<{ changed: number }>('vault.reclassify', { signature: sig, kind })
      setResult({ ok: true, text: r.message || `${r.changed} row(s) changed` })
    } catch (err) {
      setResult({ ok: false, text: (err as Error).message })
    } finally {
      setBusy(false)
    }
  }
  return (
    <form onSubmit={submit} className="mt-3 pt-3" style={{ borderTop: '1px solid rgb(255 255 255 / 12%)' }}>
      <h6 className="mb-1" style={{ color: '#fff' }}>Relabel a transaction</h6>
      <Caption>Relabel a transaction (e.g. a deposit you sent from a new address). A change after its week settled only moves future weeks.</Caption>
      <div className="toolbar mb-2">
        <label className="field flex-grow-1" style={{ minWidth: 260 }}>
          <span className="form-label mb-1">Transaction signature</span>
          <input className="form-control mono" value={signature} onChange={(e) => setSignature(e.target.value)} spellCheck={false} autoComplete="off" />
        </label>
        <label className="field">
          <span className="form-label mb-1">Label</span>
          <select className="form-select" value={kind} onChange={(e) => setKind(e.target.value as Kind)}>
            {KINDS.map((k) => <option key={k} value={k}>{k}</option>)}
          </select>
        </label>
        <button type="submit" className="btn btn-primary" disabled={busy || !signature.trim()}>
          {busy ? <span className="spinner-border spinner-border-sm me-1" /> : null}Reclassify
        </button>
      </div>
      {result && <Callout kind={result.ok ? 'success' : 'error'}>{result.text}</Callout>}
    </form>
  )
}

function TablesBlock({ t }: { t: TopicState<TablesData> }) {
  const [signature, setSignature] = useState('')
  const link = (base: string, v: unknown): ReactNode =>
    v ? <a href={`${base}${String(v)}`} target="_blank" rel="noreferrer" className="mono" title={String(v)} onClick={(e) => e.stopPropagation()}>{shortAddr(String(v), 10, 6)}</a> : null
  return (
    <Loaded state={t} height={300}>
      {(d) => {
        if (isOff(d)) return off(d)
        const settlements: Col<Row>[] = [
          { key: 'id', label: 'id', num: true }, when('period_end'), pill('status'),
          solAmt('realized_sol', 'realized_sol'), solAmt('pot_sol', 'pot_sol'), solAmt('allocated_sol', 'allocated_sol'), solAmt('carried_sol', 'carried_sol'),
          { key: 'earners', label: 'earners', num: true }, { key: 'inputs_sha256', label: 'inputs_sha256', render: (r) => <Addr v={r.inputs_sha256} /> },
        ]
        const claims: Col<Row>[] = [
          { key: 'id', label: 'id', num: true }, { key: 'relay_id', label: 'relay_id', num: true },
          { key: 'evm', label: 'evm', render: (r) => <Addr v={r.evm} /> }, { key: 'sol', label: 'sol', render: (r) => <Addr v={r.sol} /> },
          pill('status'), { key: 'reason', label: 'reason', wrap: true }, solAmt('sol_amount', 'sol_amount'),
          { key: 'tx_signature', label: 'tx_signature', render: (r) => link(d.tx_url, r.tx_signature) }, when('updated_at'),
        ]
        const flows: Col<Row>[] = [
          { key: 'id', label: 'id', num: true }, when('block_time'), pill('kind'), pill('direction'),
          { key: 'counterparty', label: 'counterparty', render: (r) => <Addr v={r.counterparty} /> }, solAmt('sol_amount', 'sol_amount'),
          { key: 'classified_by', label: 'classified_by' }, { key: 'signature', label: 'signature', render: (r) => link(d.tx_url, r.signature) },
          { key: 'note', label: 'note', wrap: true },
        ]
        const holders: Col<Row>[] = [
          { key: 'evm', label: 'evm', render: (r) => (r.evm ? <span className="mono" title={String(r.evm)}>{shortAddr(String(r.evm), 8, 6)}</span> : null) },
          solAmt('allocated_sol', 'allocated_sol'), solAmt('claimed_sol', 'claimed_sol'), solAmt('owed_sol', 'owed_sol'),
        ]
        return (
          <>
            <div className="row g-4">
              <div className="col-12 col-xl-7">
                <Card title="Settlements" sub="last 52 weeks" icon="bx-calendar-check">
                  <DataTable rows={d.settlements} cols={settlements} filename="vault-settlements" empty="No settlements yet." />
                </Card>
              </div>
              <div className="col-12 col-xl-5">
                <Card title="Holders" sub="allocated, claimed and owed per locker" icon="bx-group">
                  <DataTable rows={d.holders} cols={holders} filename="vault-holders" empty="No holders allocated yet." />
                </Card>
              </div>
            </div>
            <Card title="Claims" sub="last 100" icon="receipt">
              <DataTable rows={d.claims} cols={claims} filename="vault-claims" empty="No claims yet." />
            </Card>
            <Card title="Wallet flows" sub="last 200" icon="swap">
              <DataTable rows={d.flows} cols={flows} filename="vault-flows" empty="No wallet flows scanned yet."
                onRowClick={(r) => r.signature && setSignature(String(r.signature))} />
              {d.flows.length > 0 && <Caption>Click a flow to put its signature in the form below.</Caption>}
              <Reclassify signature={signature} setSignature={setSignature} />
            </Card>
          </>
        )
      }}
    </Loaded>
  )
}

function Page() {
  const s = useTopic<SummaryData>('vault.summary')
  const t = useTopic<TablesData>('vault.tables')
  return (
    <>
      <Loaded state={s} height={140}>
        {(d) => isOff(d) ? off(d) : (
          <>
            <HaltBanner d={d} />
            <Kpis d={d} />
            <div className="row g-4">
              <div className="col-12 col-xl-8"><SettlementsChart t={t} /></div>
              <div className="col-12 col-xl-4"><FlowsCard d={d} /></div>
            </div>
          </>
        )}
      </Loaded>
      <TablesBlock t={t} />
      <Callout kind="info" icon="bx-terminal">
        <Md text="Principal withdrawals are CLI-only: `fly-trader vault withdraw` prints how much principal you may take out of the treasury (your deposits net of any loss not yet earned back, never what holders are owed) and the Squads app steps to sign it from your own wallet." />
      </Callout>
    </>
  )
}

export default function Vault() {
  const { meta } = useApp()
  return (
    <>
      <PageHead title="$FLY vault" icon="bx-coin-stack" chain={false} />
      {!meta ? <div className="skeleton" style={{ height: 160 }} /> : !meta.vault ? <Callout kind="info">{OFF}</Callout> : <Page />}
    </>
  )
}
