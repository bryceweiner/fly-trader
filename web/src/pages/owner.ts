/** Treasury owner page: read the Squads multisig, create it and its spending limits L1/L2, change L2, revoke both.
 *  The connected Solana wallet (the owner) signs every transaction; nothing goes to the relay. */
import { Connection, Keypair, PublicKey, type VersionedTransaction } from '@solana/web3.js'
import { config, explorer } from '../config'
import type { Wallets, WalletState } from '../lib/appkit'
import * as sq from '../lib/squads'
import { $, busy, h, link, mountWalletBar, statusLine } from '../lib/ui'

const conn = new Connection(config.solana.rpcUrl, 'confirmed')
const status = statusLine('status')
const FIELDS = ['f-ms', 'f-trading', 'f-payout', 'f-l1', 'f-l2'] as const
const STORE = 'fly-owner-v1'
let wallets: Wallets | null = null
let owner: PublicKey | null = null

const input = (id: string) => $<HTMLInputElement>(id)
const val = (id: string) => input(id).value.trim()

function key(id: string, label: string): PublicKey {
  try {
    return new PublicKey(val(id))
  } catch {
    throw new Error(`${label} is not a Solana address.`)
  }
}

function lamports(id: string): bigint {
  const v = Number(val(id))
  if (!(v > 0) || v > 1000) throw new Error('Enter an amount between 0 and 1000 SOL.')
  return BigInt(Math.round(v * 1e9))
}

function save() {
  try {
    localStorage.setItem(STORE, JSON.stringify(Object.fromEntries(FIELDS.map((f) => [f, val(f)]))))
  } catch { /* private mode: fine */ }
}

function restore() {
  try {
    const s = JSON.parse(localStorage.getItem(STORE) || '{}') as Record<string, string>
    for (const f of FIELDS) if (s[f]) input(f).value = s[f]
  } catch { /* nothing stored */ }
}

async function confirm(sig: string) {
  for (let i = 0; i < 120; i++) {
    const st = (await conn.getSignatureStatuses([sig])).value[0]
    if (st?.err) throw new Error(`The transaction failed on chain: ${JSON.stringify(st.err)}`)
    if (st?.confirmationStatus === 'confirmed' || st?.confirmationStatus === 'finalized') return
    await new Promise((r) => setTimeout(r, 500))
  }
  throw new Error(`Not confirmed yet: ${sig}. Check it on an explorer before retrying.`)
}

/** Build, let the owner's wallet sign (plus any extra keypair), send, confirm. */
async function run(what: string, build: () => Promise<{ ixs: Parameters<typeof sq.toTransaction>[2]; extra?: Keypair[] }>) {
  if (!wallets || !owner) throw new Error('Connect the owner wallet first.')
  status(`${what}: preparing…`, 'busy')
  const { ixs, extra } = await build()
  const tx = await sq.toTransaction(conn, owner, ixs)
  if (extra?.length) tx.sign(extra)
  status(`${what}: approve it in your wallet…`, 'busy')
  const signed = await wallets.signSolTransaction<VersionedTransaction>(tx)
  const sig = await conn.sendTransaction(signed)
  status(`${what}: sent, waiting for confirmation…`, 'busy')
  await confirm(sig)
  status(h('span', {}, `${what}: done. `, link(explorer.solTx(sig), 'transaction')), 'ok')
}

function envLines() {
  const lines = [`VAULT_MULTISIG=${val('f-ms')}`, `VAULT_LIMIT_TRADING=${val('f-l1')}`, `VAULT_LIMIT_PAYOUT=${val('f-l2')}`]
  const out = $('env-out')
  out.textContent = `# /srv/fly/custody.env on the server, then restart the fly:\n${lines.join('\n')}`
  out.hidden = false
}

async function load() {
  save()
  const msText = val('f-ms')
  $('create-ms').hidden = !!msText
  if (!msText) {
    $('state-box').hidden = $('actions').hidden = $('revoke-box').hidden = true
    status('No multisig yet: connect your owner wallet and create one.', 'info')
    return
  }
  const ms = key('f-ms', 'Multisig')
  const t = await sq.readTreasury(conn, ms)
  const l1 = val('f-l1') ? await sq.readLimit(conn, key('f-l1', 'L1')) : null
  const l2 = val('f-l2') ? await sq.readLimit(conn, key('f-l2', 'L2')) : null
  const lim = (l: sq.Limit | null) => (l ? `${sq.sol(l.remaining)} of ${sq.sol(l.amount)} SOL left this ${l.period.toLowerCase()}` : 'none')
  const kv = (k: string, v: Node | string) => h('div', {}, h('span', { class: 'micro' }, k), h('div', { class: 'v' }, v))
  $('state').replaceChildren(
    kv('Treasury (vault 0)', link(explorer.solAccount(t.vault), t.vault)),
    kv('Balance', `${sq.sol(t.vaultLamports)} SOL`),
    kv('Members', t.members.join(', ')),
    kv('Threshold / time lock', `${t.threshold} / ${t.timeLock} s`),
    kv('L1 (float refill)', lim(l1)),
    kv('L2 (claims)', lim(l2)),
  )
  const probs = sq.problems(t, l1, l2, val('f-trading'), val('f-payout'))
  $('problems').replaceChildren(...(probs.length ? probs.map((p) => h('li', {}, p)) : [h('li', {}, 'Every check passes.')]))
  $('state-box').hidden = $('actions').hidden = false
  $('revoke-box').hidden = !(l1 || l2)
  if (owner && !t.members.includes(owner.toBase58())) status('The connected wallet is not a member of this multisig: it cannot change anything.', 'error')
  if (t.threshold > 1) status(`This multisig needs ${t.threshold} approvals: changes made here are proposed and approved by you; the other member executes them in the Squads app.`, 'info')
}

async function addLimit(which: 'L1' | 'L2') {
  const ms = key('f-ms', 'Multisig')
  const member = key(which === 'L1' ? 'f-trading' : 'f-payout', which === 'L1' ? 'Trading key' : 'Payout key')
  const createKey = Keypair.generate().publicKey
  const address = sq.limitAddress(ms, createKey)
  await run(`Create ${which}`, async () => ({
    ixs: (await sq.configChange(conn, ms, owner!, [sq.addLimit({
      createKey, lamports: lamports(which === 'L1' ? 'a-l1' : 'a-l2'), period: which === 'L1' ? 'Day' : 'Week', member,
      destinations: which === 'L1' ? [member] : [],
    })], [address])).ixs,
  }))
  input(which === 'L1' ? 'f-l1' : 'f-l2').value = address.toBase58()
  envLines()
  await load()
}

async function removeLimits(ids: ('f-l1' | 'f-l2')[], what: string) {
  const ms = key('f-ms', 'Multisig')
  const addrs = ids.filter((id) => val(id)).map((id) => key(id, id === 'f-l1' ? 'L1' : 'L2'))
  if (!addrs.length) throw new Error('No limit to remove.')
  await run(what, async () => ({ ixs: (await sq.configChange(conn, ms, owner!, addrs.map(sq.removeLimit), addrs)).ixs }))
  for (const id of ids) input(id).value = ''
  envLines()
  await load()
}

async function createMultisig() {
  const createKey = Keypair.generate()
  let ms: PublicKey | null = null
  await run('Create the treasury multisig', async () => {
    const r = await sq.createMultisig(conn, owner!, createKey)
    ms = r.multisig
    return { ixs: r.ixs, extra: [createKey] }
  })
  input('f-ms').value = ms!.toBase58()
  envLines()
  await load()
}

function wire(id: string, fn: () => Promise<void>) {
  const btn = $<HTMLButtonElement>(id)
  btn.addEventListener('click', () => void busy(btn, async () => {
    try {
      await fn()
    } catch (e) {
      status(e instanceof Error ? e.message : String(e), 'error')
    }
  }))
}

restore()
wire('load', load)
wire('create-ms', createMultisig)
wire('add-l1', () => addLimit('L1'))
wire('add-l2', () => addLimit('L2'))
wire('set-l2', async () => {
  if (val('f-l2')) await removeLimits(['f-l2'], 'Remove the old L2')
  await addLimit('L2')
})
wire('revoke', () => removeLimits(['f-l1', 'f-l2'], 'Revoke L1 and L2'))
void mountWalletBar({ solana: true, evm: false, onChange: (s: WalletState, w: Wallets) => {
  wallets = w
  owner = s.sol ? new PublicKey(s.sol) : null
} }).then(() => (val('f-ms') ? load().catch((e) => status(String(e), 'error')) : undefined))
