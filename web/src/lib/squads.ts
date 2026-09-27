/**
 * The treasury's Squads v4 multisig, for its owner (owner.html): read it, create it, and change its two spending
 * limits (docs/vault/SPEC.md "Custody"). Pure builders over a Connection: the page signs with the connected wallet,
 * the integration test (squads.int.test.ts) with a plain keypair on a local validator running the real program.
 *
 *   L1  SOL, per day,  member = the server's trading key, ONLY destination = the trading wallet (the float's refill)
 *   L2  SOL, per week, member = the server's payout key,  any destination (holders' claims)
 *
 * Every change is one transaction: config transaction + proposal + the owner's approval + execution. That works while
 * the multisig has threshold 1 and no time lock (the owner alone); once the Ledger joins with threshold 2, the page
 * creates and approves, and the second member approves and executes in the Squads app.
 */
import * as multisig from '@sqds/multisig'
import {
  Connection,
  Keypair,
  PublicKey,
  TransactionMessage,
  VersionedTransaction,
  type TransactionInstruction,
} from '@solana/web3.js'

export const SQUADS = multisig.PROGRAM_ID
export const SOL = PublicKey.default
const DAY = multisig.types.Period.Day
const WEEK = multisig.types.Period.Week

export interface Limit {
  address: string
  amount: bigint
  remaining: bigint
  period: 'OneTime' | 'Day' | 'Week' | 'Month'
  members: string[]
  destinations: string[]
  mint: string
}

export interface Treasury {
  multisig: string
  vault: string
  threshold: number
  timeLock: number
  members: string[]
  configAuthority: string
  transactionIndex: bigint
  vaultLamports: number
}

const PERIODS = ['OneTime', 'Day', 'Week', 'Month'] as const

export function vaultOf(ms: PublicKey): PublicKey {
  return multisig.getVaultPda({ multisigPda: ms, index: 0 })[0]
}

export async function readTreasury(conn: Connection, ms: PublicKey): Promise<Treasury> {
  const m = await multisig.accounts.Multisig.fromAccountAddress(conn, ms)
  const vault = vaultOf(ms)
  return {
    multisig: ms.toBase58(),
    vault: vault.toBase58(),
    threshold: m.threshold,
    timeLock: m.timeLock,
    members: m.members.map((x) => x.key.toBase58()),
    configAuthority: m.configAuthority.toBase58(),
    transactionIndex: BigInt(m.transactionIndex.toString()),
    vaultLamports: await conn.getBalance(vault),
  }
}

export async function readLimit(conn: Connection, address: PublicKey): Promise<Limit | null> {
  const info = await conn.getAccountInfo(address)
  if (!info || !info.owner.equals(SQUADS)) return null
  const [l] = multisig.accounts.SpendingLimit.fromAccountInfo(info)
  return {
    address: address.toBase58(),
    amount: BigInt(l.amount.toString()),
    remaining: BigInt(l.remainingAmount.toString()),
    period: PERIODS[l.period],
    members: l.members.map((k) => k.toBase58()),
    destinations: l.destinations.map((k) => k.toBase58()),
    mint: l.mint.toBase58(),
  }
}

/** What is wrong with the setup for these server keys (mirrors fly_trader/vault/custody.problems). */
export function problems(t: Treasury, l1: Limit | null, l2: Limit | null, trading: string, payout: string): string[] {
  const out: string[] = []
  for (const [who, k] of [['trading', trading], ['payout', payout]] as const)
    if (k && t.members.includes(k)) out.push(`The ${who} key is a member of the multisig: remove it.`)
  if (t.configAuthority !== SOL.toBase58()) out.push('The multisig has a config authority: it can change members without a vote.')
  if (!l1) out.push('L1 is missing: the trading float cannot refill.')
  else {
    if (l1.mint !== SOL.toBase58()) out.push('L1 is not a SOL limit.')
    if (l1.members.join() !== trading) out.push('L1 must list only the trading key.')
    if (l1.destinations.join() !== trading) out.push('L1 must have the trading wallet as its ONLY destination.')
  }
  if (!l2) out.push('L2 is missing: claims will wait.')
  else {
    if (l2.mint !== SOL.toBase58()) out.push('L2 is not a SOL limit.')
    if (l2.members.join() !== payout) out.push('L2 must list only the payout key.')
  }
  return out
}

/** A new multisig whose only member is the owner (threshold 1, no time lock, no config authority). */
export async function createMultisig(conn: Connection, owner: PublicKey, createKey: Keypair): Promise<{ ixs: TransactionInstruction[]; multisig: PublicKey }> {
  const [ms] = multisig.getMultisigPda({ createKey: createKey.publicKey })
  const [cfg] = multisig.getProgramConfigPda({})
  const pc = await multisig.accounts.ProgramConfig.fromAccountAddress(conn, cfg)
  const ix = multisig.instructions.multisigCreateV2({
    treasury: pc.treasury,
    creator: owner,
    multisigPda: ms,
    configAuthority: null,
    threshold: 1,
    members: [{ key: owner, permissions: multisig.types.Permissions.all() }],
    timeLock: 0,
    createKey: createKey.publicKey,
    rentCollector: null,
  })
  return { ixs: [ix], multisig: ms }
}

type Action = multisig.types.ConfigAction

export function addLimit(o: { createKey: PublicKey; lamports: bigint; period: 'Day' | 'Week'; member: PublicKey; destinations: PublicKey[] }): Action {
  return {
    __kind: 'AddSpendingLimit',
    createKey: o.createKey,
    vaultIndex: 0,
    mint: SOL,
    amount: Number(o.lamports),          // beet's bignum: a number is exact up to 9e15 lamports (9 M SOL)
    period: o.period === 'Day' ? DAY : WEEK,
    members: [o.member],
    destinations: o.destinations,
  }
}

/** A limit's amount cannot be edited: remove it, then add a new one -- as TWO config changes (the program rejects a
 *  remove and an add in one config transaction). Removing first means the payout key never holds two limits at once;
 *  claims simply wait the few seconds in between. */
export function removeLimit(address: PublicKey): Action {
  return { __kind: 'RemoveSpendingLimit', spendingLimit: address }
}

export function limitAddress(ms: PublicKey, createKey: PublicKey): PublicKey {
  return multisig.getSpendingLimitPda({ multisigPda: ms, createKey })[0]
}

/** One config change: create, propose, approve and (when the owner alone may) execute. */
export async function configChange(conn: Connection, ms: PublicKey, owner: PublicKey, actions: Action[], touched: PublicKey[]): Promise<{ ixs: TransactionInstruction[]; executes: boolean }> {
  const t = await readTreasury(conn, ms)
  if (!t.members.includes(owner.toBase58())) throw new Error('The connected wallet is not a member of this multisig.')
  const transactionIndex = t.transactionIndex + 1n
  const ixs = [
    multisig.instructions.configTransactionCreate({ multisigPda: ms, transactionIndex, creator: owner, rentPayer: owner, actions }),
    multisig.instructions.proposalCreate({ multisigPda: ms, transactionIndex, creator: owner, rentPayer: owner }),
    multisig.instructions.proposalApprove({ multisigPda: ms, transactionIndex, member: owner }),
  ]
  const executes = t.threshold <= 1 && t.timeLock === 0
  if (executes)
    ixs.push(multisig.instructions.configTransactionExecute({ multisigPda: ms, transactionIndex, member: owner, rentPayer: owner, spendingLimits: touched }))
  return { ixs, executes }
}

export async function toTransaction(conn: Connection, payer: PublicKey, ixs: TransactionInstruction[]): Promise<VersionedTransaction> {
  const { blockhash } = await conn.getLatestBlockhash('confirmed')
  return new VersionedTransaction(new TransactionMessage({ payerKey: payer, recentBlockhash: blockhash, instructions: ixs }).compileToV0Message())
}

export const SOL_LAMPORTS = 1_000_000_000n

export function sol(lamports: bigint | number): string {
  return (Number(lamports) / 1e9).toLocaleString('en-US', { maximumFractionDigits: 4 })
}
