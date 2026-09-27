/**
 * Not a test: tools/vault_demo.py's treasury setup on its local validator, run through vitest (it compiles the
 * TypeScript; Node 20 cannot). Skipped unless SQUADS_SETUP names a JSON file:
 *   in:  {rpc, owner (base58 secret), trading, payout (pubkeys), l1_lamports, l2_lamports, fund_lamports}
 *   out: the same file plus {multisig, treasury, l1, l2}
 * It uses exactly what owner.html uses (./squads), with a local throwaway key standing in for Solflare.
 */
import { readFileSync, writeFileSync } from 'node:fs'
import { Connection, Keypair, PublicKey, SystemProgram } from '@solana/web3.js'
import bs58 from 'bs58'
import { describe, it } from 'vitest'
import * as sq from './squads'

const FILE = process.env.SQUADS_SETUP

async function send(conn: Connection, payer: Keypair, ixs: Parameters<typeof sq.toTransaction>[2], extra: Keypair[] = []) {
  const tx = await sq.toTransaction(conn, payer.publicKey, ixs)
  tx.sign([payer, ...extra])
  const sig = await conn.sendTransaction(tx)
  for (let i = 0; i < 120; i++) {
    const st = (await conn.getSignatureStatuses([sig])).value[0]
    if (st?.err) throw new Error(`transaction failed: ${JSON.stringify(st.err)}`)
    if (st?.confirmationStatus === 'confirmed' || st?.confirmationStatus === 'finalized') return sig
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`not confirmed: ${sig}`)
}

describe.skipIf(!FILE)('demo treasury setup', () => {
  it('creates the multisig, L1 and L2', async () => {
    const cfg = JSON.parse(readFileSync(FILE!, 'utf8'))
    const conn = new Connection(cfg.rpc, 'confirmed')
    const owner = Keypair.fromSecretKey(bs58.decode(cfg.owner))
    const trading = new PublicKey(cfg.trading), payout = new PublicKey(cfg.payout)
    const createKey = Keypair.generate()
    const { ixs, multisig } = await sq.createMultisig(conn, owner.publicKey, createKey)
    await send(conn, owner, ixs, [createKey])
    const vault = sq.vaultOf(multisig)
    if (cfg.fund_lamports) await send(conn, owner, [SystemProgram.transfer({ fromPubkey: owner.publicKey, toPubkey: vault, lamports: cfg.fund_lamports })])
    const k1 = Keypair.generate().publicKey, k2 = Keypair.generate().publicKey
    const l1 = sq.limitAddress(multisig, k1), l2 = sq.limitAddress(multisig, k2)
    await send(conn, owner, (await sq.configChange(conn, multisig, owner.publicKey, [
      sq.addLimit({ createKey: k1, lamports: BigInt(cfg.l1_lamports), period: 'Day', member: trading, destinations: [trading] }),
      sq.addLimit({ createKey: k2, lamports: BigInt(cfg.l2_lamports), period: 'Week', member: payout, destinations: [] }),
    ], [l1, l2])).ixs)
    writeFileSync(FILE!, JSON.stringify({ ...cfg, multisig: multisig.toBase58(), treasury: vault.toBase58(), l1: l1.toBase58(), l2: l2.toBase58() }, null, 1))
  }, 120_000)
})
