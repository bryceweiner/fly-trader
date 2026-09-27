/**
 * The owner page's Squads builders against the REAL Squads v4 program on a local validator (skipped unless
 * SQUADS_IT_RPC is set). tools/custody_e2e.sh starts that validator (program + its config account dumped from mainnet),
 * runs this, then tests/test_custody_e2e.py drives the Python signer against the multisig made here.
 */
import { writeFileSync } from 'node:fs'
import { Connection, Keypair, LAMPORTS_PER_SOL, PublicKey, SystemProgram } from '@solana/web3.js'
import bs58 from 'bs58'
import { describe, expect, it } from 'vitest'
import * as sq from './squads'

const RPC = process.env.SQUADS_IT_RPC
const OUT = process.env.SQUADS_IT_OUT

/** Confirmation by polling (no websocket: the test validator's ws port may be taken). */
async function confirm(conn: Connection, sig: string) {
  for (let i = 0; i < 120; i++) {
    const st = (await conn.getSignatureStatuses([sig])).value[0]
    if (st?.err) throw new Error(`transaction failed: ${JSON.stringify(st.err)}`)
    if (st && (st.confirmationStatus === 'confirmed' || st.confirmationStatus === 'finalized')) return
    await new Promise((r) => setTimeout(r, 250))
  }
  throw new Error(`not confirmed: ${sig}`)
}

async function send(conn: Connection, payer: Keypair, ixs: Parameters<typeof sq.toTransaction>[2], extra: Keypair[] = []) {
  const tx = await sq.toTransaction(conn, payer.publicKey, ixs)
  tx.sign([payer, ...extra])
  const sig = await conn.sendTransaction(tx)
  await confirm(conn, sig)
  return sig
}

describe.skipIf(!RPC)('Squads treasury on a local validator', () => {
  it('creates the treasury, its two limits, raises L2 and revokes', async () => {
    const conn = new Connection(RPC!, 'confirmed')
    const owner = Keypair.generate(), trading = Keypair.generate(), payout = Keypair.generate()
    for (const k of [owner, trading])
      await confirm(conn, await conn.requestAirdrop(k.publicKey, 20 * LAMPORTS_PER_SOL))

    const createKey = Keypair.generate()
    const { ixs, multisig } = await sq.createMultisig(conn, owner.publicKey, createKey)
    await send(conn, owner, ixs, [createKey])
    let t = await sq.readTreasury(conn, multisig)
    expect(t.members).toEqual([owner.publicKey.toBase58()])
    expect([t.threshold, t.timeLock]).toEqual([1, 0])

    // fund the treasury (the operator's principal)
    await send(conn, owner, [SystemProgram.transfer({ fromPubkey: owner.publicKey, toPubkey: new PublicKey(t.vault), lamports: 10 * LAMPORTS_PER_SOL })])

    const k1 = Keypair.generate().publicKey, k2 = Keypair.generate().publicKey
    const l1 = sq.limitAddress(multisig, k1), l2 = sq.limitAddress(multisig, k2)
    const add = await sq.configChange(conn, multisig, owner.publicKey, [
      sq.addLimit({ createKey: k1, lamports: 2n * sq.SOL_LAMPORTS, period: 'Day', member: trading.publicKey, destinations: [trading.publicKey] }),
      sq.addLimit({ createKey: k2, lamports: sq.SOL_LAMPORTS, period: 'Week', member: payout.publicKey, destinations: [] }),
    ], [l1, l2])
    expect(add.executes).toBe(true)
    await send(conn, owner, add.ixs)
    t = await sq.readTreasury(conn, multisig)
    const L1 = await sq.readLimit(conn, l1), L2 = await sq.readLimit(conn, l2)
    expect(sq.problems(t, L1, L2, trading.publicKey.toBase58(), payout.publicKey.toBase58())).toEqual([])
    expect(L1!.destinations).toEqual([trading.publicKey.toBase58()])
    expect(L2!.period).toBe('Week')

    if (OUT) {                                  // hand the chain to the Python signer test, then stop here
      writeFileSync(OUT, JSON.stringify({
        rpc: RPC, multisig: multisig.toBase58(), treasury: t.vault, l1: l1.toBase58(), l2: l2.toBase58(),
        owner: bs58.encode(owner.secretKey), trading: bs58.encode(trading.secretKey), payout: bs58.encode(payout.secretKey),
      }))
      return
    }

    // raise L2: remove the old limit, then add the new one (two changes: Squads refuses remove + add in one config
    // transaction -- 'sum of account balances before and after instruction do not match'), then revoke both
    await send(conn, owner, (await sq.configChange(conn, multisig, owner.publicKey, [sq.removeLimit(l2)], [l2])).ixs)
    const k3 = Keypair.generate().publicKey, l3 = sq.limitAddress(multisig, k3)
    await send(conn, owner, (await sq.configChange(conn, multisig, owner.publicKey, [
      sq.addLimit({ createKey: k3, lamports: 3n * sq.SOL_LAMPORTS, period: 'Week', member: payout.publicKey, destinations: [] }),
    ], [l3])).ixs)
    expect(await sq.readLimit(conn, l2)).toBeNull()
    expect((await sq.readLimit(conn, l3))!.amount).toBe(3n * sq.SOL_LAMPORTS)
    await send(conn, owner, (await sq.configChange(conn, multisig, owner.publicKey, [sq.removeLimit(l1), sq.removeLimit(l3)], [l1, l3])).ixs)
    expect(await sq.readLimit(conn, l1)).toBeNull()
    expect(await sq.readLimit(conn, l3)).toBeNull()
    // a stranger cannot change anything
    const stranger = Keypair.generate()
    await expect(sq.configChange(conn, multisig, stranger.publicKey, [sq.removeLimit(l1)], [l1])).rejects.toThrow(/not a member/)
  }, 120_000)
})
