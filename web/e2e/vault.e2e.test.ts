/**
 * The Vault page's deposit and redemption paths, end to end, through the same functions the page calls
 * (src/lib/vault.ts, src/lib/evm.ts, src/lib/claim.ts, src/lib/api.ts) against the local dry run:
 * anvil (FlyVault + MockFLY), the relay and the vault fly paying claims on a local Solana validator.
 *
 *   .venv/bin/python tools/vault_demo.py up && web/e2e/run.sh vault
 *
 * Rerunnable: the lockers are fresh keys each run, and the claim section funds a new settlement (a "profit" airdrop
 * to the fly's wallet) and waits for it, so the fixed demo holders are owed SOL again.
 */
import { ed25519 } from '@noble/curves/ed25519'
import { randomBytes } from 'node:crypto'
import { readFileSync } from 'node:fs'
import bs58 from 'bs58'
import { erc20Abi, formatUnits, parseEther, type Address, type Hex } from 'viem'
import { beforeAll, describe, expect, it } from 'vitest'
import { config } from '../src/config'
import { api, ApiError, type ClaimStatus } from '../src/lib/api'
import { evmText, runClaim, solText, type ClaimFields, type ClaimStep } from '../src/lib/claim'
import { approveExact, humanError } from '../src/lib/evm'
import * as vault from '../src/lib/vault'
import { account, freshWallet, mockFlyAbi, pauseAbi, pub, rpc, wallet, walletOf, warp, type FreshWallet } from './harness'

const DEMO = new URL('../../.vault-demo/', import.meta.url)
const SOLANA_RPC = 'http://127.0.0.1:8899'
const VAULT = config.vault as Address
const FLY = config.fly.address as Address
const steps: string[] = []
const step = (m: string) => steps.push(m)
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

async function fails(run: () => Promise<unknown>): Promise<string> {
  try {
    await run()
  } catch (e) {
    return humanError(e)
  }
  throw new Error('expected the call to fail')
}

const flyBalance = (a: Address) => pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'balanceOf', args: [a] })
const allowance = (a: Address) => pub.readContract({ address: FLY, abi: erc20Abi, functionName: 'allowance', args: [a, VAULT] })

async function solBalance(pubkey: string): Promise<number> {
  const r = await rpc<{ value: number }>('getBalance', [pubkey], SOLANA_RPC)
  return r.value
}

describe('vault: lock, request, cancel, withdraw through the page libraries', () => {
  let u: FreshWallet
  let other: FreshWallet
  let delay: number
  const minted = parseEther('5000')

  beforeAll(async () => {
    expect(VAULT, 'VITE_VAULT_ADDRESS').toBeTruthy()
    expect(FLY, 'VITE_FLY_ADDRESS').toBeTruthy()
    u = await freshWallet()
    other = await freshWallet()
    // faucet: MockFLY.mint is public on the rehearsal token
    for (const w of [u, other]) {
      const h = await walletOf(w.account).writeContract({ address: FLY, abi: mockFlyAbi, functionName: 'mint', args: [w.address, minted] })
      await pub.waitForTransactionReceipt({ hash: h })
    }
  })

  it('reads the public totals the tiles show', async () => {
    const t = await vault.readTotals()
    delay = t.withdrawDelay
    expect(t.paused).toBe(false)
    expect(t.withdrawDelay).toBe(300) // the dry run's 5-minute delay
    expect(t.totalLocked).toBeGreaterThanOrEqual(parseEther('400000')) // the two demo holders
    expect(t.totalPending).toBeGreaterThanOrEqual(0n)
  })

  it('reads an empty position', async () => {
    const p = await vault.readPosition(u.address)
    expect(p.flyBalance).toBe(minted)
    expect(p.locked).toBe(0n)
    expect(p.pending).toBe(0n)
    expect(p.requests).toEqual([])
  })

  it('lock: approves exactly the amount, then locks (two wallet prompts)', async () => {
    steps.length = 0
    const before = await vault.readTotals()
    const hash = await vault.lock(u.cfg, u.address, parseEther('1000'), step)
    expect(hash).toMatch(/^0x[0-9a-f]{64}$/)
    expect(steps[0]).toMatch(/^1\/2/)
    expect(steps[1]).toMatch(/^2\/2/)
    const p = await vault.readPosition(u.address)
    expect(p.locked).toBe(parseEther('1000'))
    expect(p.flyBalance).toBe(minted - parseEther('1000'))
    expect(await allowance(u.address)).toBe(0n) // exact approval, fully consumed
    expect((await vault.readTotals()).totalLocked - before.totalLocked).toBe(parseEther('1000'))
  })

  it('lock: a second lock needs a fresh exact approval; a sufficient allowance is reused', async () => {
    await vault.lock(u.cfg, u.address, parseEther('500'), step)
    expect((await vault.readPosition(u.address)).locked).toBe(parseEther('1500'))
    // pre-approve more than needed: approveExact must not send another approval
    const h = await walletOf(u.account).writeContract({ address: FLY, abi: erc20Abi, functionName: 'approve', args: [VAULT, parseEther('10')] })
    await pub.waitForTransactionReceipt({ hash: h })
    expect(await approveExact(u.cfg, FLY, u.address, VAULT, parseEther('5'))).toBeNull()
    expect(await approveExact(u.cfg, FLY, u.address, VAULT, parseEther('11'))).toMatch(/^0x/)
    expect(await allowance(u.address)).toBe(parseEther('11'))
  })

  it('lock: refuses zero and more than the balance with the page wording', async () => {
    expect(await fails(() => vault.lock(u.cfg, u.address, 0n, step))).toBe('Enter an amount greater than zero.')
    expect(await fails(() => vault.lock(u.cfg, u.address, minted * 2n, step))).toBe('Your $FLY balance is too low for that amount.')
    expect((await vault.readPosition(u.address)).locked).toBe(parseEther('1500'))
  })

  let id1: bigint
  it('requestWithdrawal: stops earning at once, sets readyAt = now + delay', async () => {
    const t0 = Number((await pub.getBlock()).timestamp)
    await vault.requestWithdrawal(u.cfg, u.address, parseEther('600'), step)
    const p = await vault.readPosition(u.address)
    expect(p.locked).toBe(parseEther('900'))
    expect(p.pending).toBe(parseEther('600'))
    expect(p.requests).toHaveLength(1)
    const r = p.requests[0]
    id1 = r.id
    expect(r.state).toBe('pending')
    expect(r.amount).toBe(parseEther('600'))
    expect(r.readyAt).toBeGreaterThanOrEqual(t0 + delay)
    expect(r.readyAt).toBeLessThanOrEqual(t0 + delay + 30)
  })

  it('requestWithdrawal: refuses more than is locked, naming the locked amount', async () => {
    const msg = await fails(() => vault.requestWithdrawal(u.cfg, u.address, parseEther('901'), step))
    expect(msg).toMatch(/only have .*900.* \$FLY locked/)
    expect(await fails(() => vault.requestWithdrawal(u.cfg, u.address, 0n, step))).toBe('Enter an amount greater than zero.')
  })

  it('withdraw: not before readyAt', async () => {
    const msg = await fails(() => vault.withdraw(u.cfg, u.address, id1, step))
    expect(msg).toMatch(new RegExp(`request #${id1} is not ready until`))
  })

  it('cancelRequest: re-locks; a second cancel is refused', async () => {
    await vault.cancelRequest(u.cfg, u.address, id1, step)
    const p = await vault.readPosition(u.address)
    expect(p.locked).toBe(parseEther('1500'))
    expect(p.pending).toBe(0n)
    expect(p.requests[0].state).toBe('cancelled')
    expect(await fails(() => vault.cancelRequest(u.cfg, u.address, id1, step))).toBe(`Withdrawal request #${id1} was already cancelled or withdrawn.`)
    expect(await fails(() => vault.withdraw(u.cfg, u.address, id1, step))).toBe(`Withdrawal request #${id1} was already cancelled or withdrawn.`)
  })

  let id2: bigint
  it('another address cannot cancel or withdraw your request; unknown ids are named', async () => {
    await vault.requestWithdrawal(u.cfg, u.address, parseEther('700'), step)
    const p = await vault.readPosition(u.address)
    id2 = p.requests.find((r) => r.state === 'pending')!.id
    expect(await fails(() => vault.cancelRequest(other.cfg, other.address, id2, step))).toBe(`Withdrawal request #${id2} belongs to another address.`)
    expect(await fails(() => vault.withdraw(other.cfg, other.address, id2, step))).toBe(`Withdrawal request #${id2} belongs to another address.`)
    expect(await fails(() => vault.withdraw(u.cfg, u.address, 999_999n, step))).toBe('Withdrawal request #999999 does not exist.')
    expect(await fails(() => vault.cancelRequest(u.cfg, u.address, 0n, step))).toBe('Withdrawal request #0 does not exist.')
  })

  let id3: bigint
  it('pause: blocks lock and cancel with the page wording; request and withdraw keep working', async () => {
    const pauser = wallet(0) // the dry run's deployer is also the pauser
    let h = await pauser.writeContract({ address: VAULT, abi: pauseAbi, functionName: 'pause' })
    await pub.waitForTransactionReceipt({ hash: h })
    expect((await vault.readTotals()).paused).toBe(true)
    expect(await fails(() => vault.lock(u.cfg, u.address, parseEther('1'), step))).toMatch(/^The vault is paused/)
    expect(await fails(() => vault.cancelRequest(u.cfg, u.address, id2, step))).toMatch(/^The vault is paused/)
    await vault.requestWithdrawal(u.cfg, u.address, parseEther('100'), step)
    const p = await vault.readPosition(u.address)
    expect(p.locked).toBe(parseEther('700'))
    expect(p.pending).toBe(parseEther('800'))
    id3 = p.requests.filter((r) => r.state === 'pending').map((r) => r.id).sort((a, b) => (a < b ? 1 : -1))[0]
    expect(id3).not.toBe(id2)
    h = await pauser.writeContract({ address: VAULT, abi: pauseAbi, functionName: 'unpause' })
    await pub.waitForTransactionReceipt({ hash: h })
    expect((await vault.readTotals()).paused).toBe(false)
  })

  it('withdraw: pays out once readyAt is reached; a second withdraw is refused', async () => {
    await warp(delay + 1)
    const before = await flyBalance(u.address)
    const hash = await vault.withdraw(u.cfg, u.address, id2, step)
    expect(hash).toMatch(/^0x/)
    expect((await flyBalance(u.address)) - before).toBe(parseEther('700'))
    expect(await fails(() => vault.withdraw(u.cfg, u.address, id2, step))).toBe(`Withdrawal request #${id2} was already cancelled or withdrawn.`)
    await vault.withdraw(u.cfg, u.address, id3, step)
    const p = await vault.readPosition(u.address)
    expect(p.locked).toBe(parseEther('700'))
    expect(p.pending).toBe(0n)
    expect(p.requests.map((r) => r.state).sort()).toEqual(['cancelled', 'withdrawn', 'withdrawn'])
    // conservation: wallet + locked + pending == minted
    expect(p.flyBalance + p.locked + p.pending).toBe(minted)
  })

  it('the fly indexes the same vault the chain reports', async () => {
    const s = await api.stats()
    expect(s.vault.address.toLowerCase()).toBe(VAULT.toLowerCase())
    expect(s.vault.chain_id).toBe(config.evm.chainId)
    expect(BigInt(s.vault.total_locked)).toBeGreaterThanOrEqual(parseEther('400000'))
    expect(s.vault.paused).toBe(false)
  })
})

describe('claims: the redemption of SOL owed to a locker', () => {
  const holderKey = 1 // the dry run's first holder (anvil account 1, 300k of 400k mFLY locked; account 2 has 100k)
  const solKeys = JSON.parse(readFileSync(new URL('sol_keys.json', DEMO), 'utf8')) as { fly: { pubkey: string } }
  const HOST = 'localhost:5173' // the relay's configured domain in the dry run
  const holder = () => account(holderKey).address

  function solSigner() {
    const seed = randomBytes(32)
    const pubkey = bs58.encode(ed25519.getPublicKey(seed))
    return { pubkey, sign: async (msg: Uint8Array) => ed25519.sign(msg, seed) }
  }

  async function claimAs(i: number, sol = solSigner(), tamper?: (sig: Hex) => Hex) {
    const acct = account(i)
    const seen: ClaimStep['step'][] = []
    let texts: { evmText: string; solText: string } | null = null
    const final = await runClaim(
      {
        evm: acct.address,
        sol: sol.pubkey,
        signEvm: async (text) => {
          const sig = await acct.signMessage({ message: text })
          return tamper ? tamper(sig) : sig
        },
        signSol: sol.sign,
      },
      (s) => {
        seen.push(s.step)
        if (s.step === 'sign-evm') texts = { evmText: s.evmText, solText: s.solText }
      },
      { host: HOST },
    )
    return { final, seen, texts: texts!, sol }
  }

  async function owedAfterPush(evm: Address, until: (owed: number) => boolean, maxMs: number) {
    const t0 = Date.now()
    let a = await api.account(evm)
    while (!until(a.owed) && Date.now() - t0 < maxMs) {
      await sleep(5000)
      a = await api.account(evm)
    }
    return a
  }

  it('a profit lands in the fly wallet and the next settlement allocates it to the lockers', async () => {
    expect((await vault.readPosition(holder())).locked).toBe(parseEther('300000'))
    expect((await vault.readPosition(account(2).address)).locked).toBe(parseEther('100000'))
    // 0.2 SOL from nobody in FUNDING_ADDRESSES: the flow scanner classifies it as profit
    const sig = await rpc<string>('requestAirdrop', [solKeys.fly.pubkey, 200_000_000], SOLANA_RPC)
    expect(sig).toMatch(/^[1-9A-HJ-NP-Za-km-z]{80,90}$/)
    const before = (await api.account(holder())).owed
    // the dry run settles every 5 minutes and then waits for anvil's "finalized" block: allow two periods
    const a = await owedAfterPush(holder(), (owed) => owed >= config.claimMinLamports && owed > before, 11 * 60_000)
    expect(a.owed).toBeGreaterThanOrEqual(config.claimMinLamports)
    expect(a.allocated).toBeGreaterThan(0)
    expect(a.allocations.length).toBeGreaterThan(0)
  }, 12 * 60_000)

  it('challenge: the relay issues one for this host and network; another host is refused before signing', async () => {
    const c = await api.challenge(holder(), solSigner().pubkey)
    expect(c.domain).toBe(HOST)
    expect(c.chain_id).toBe(config.evm.chainId)
    expect(c.sol_chain).toBe('devnet')
    expect(c.request_id).toBe('fly-vault-claim-v1')
    expect(c.nonce).toMatch(/^[0-9a-f]{32}$/)
    expect(c.min_lamports).toBe(2_000_000)
    let signed = false
    await expect(
      runClaim(
        { evm: holder(), sol: solSigner().pubkey, signEvm: async () => ((signed = true), '0x'), signSol: async () => ((signed = true), new Uint8Array(64)) },
        () => {},
        { host: 'evil.example' },
      ),
    ).rejects.toThrow(/Open the Vault page at/)
    expect(signed).toBe(false)
  })

  it('the page refuses to submit an EVM signature that does not recover to the connected address', async () => {
    const owedBefore = (await api.account(holder())).owed
    await expect(claimAs(holderKey, solSigner(), (sig) => (sig.slice(0, -2) + (sig.endsWith('1b') ? '1c' : '1b')) as Hex)).rejects.toThrow(
      /EVM signature does not match/,
    )
    expect((await api.account(holder())).owed).toBe(owedBefore)
  })

  it('a tampered claim submitted past the page check is rejected by the fly, not paid, and owed is untouched', async () => {
    const owedBefore = (await api.account(holder())).owed
    const sol = solSigner()
    const c = await api.challenge(holder(), sol.pubkey)
    const f: ClaimFields = { domain: c.domain, uri: c.uri, chain_id: Number(c.chain_id), sol_chain: c.sol_chain, evm: holder(), sol: sol.pubkey, nonce: c.nonce, issued_at: c.issued_at, expires_at: c.expires_at }
    const good = await account(holderKey).signMessage({ message: evmText(f) })
    const bad = (good.slice(0, -2) + (good.endsWith('1b') ? '1c' : '1b')) as Hex // valid ECDSA, recovers to someone else
    const solSig = bs58.encode(await sol.sign(new TextEncoder().encode(solText(f))))
    const { id, status } = await api.submitClaim({ nonce: c.nonce, evm: holder(), sol: sol.pubkey, evm_sig: bad, sol_sig: solSig })
    expect(status).toBe('received') // the relay only checks syntax; the fly verifies
    let final = await api.claim(id)
    for (let i = 0; i < 30 && !['paid', 'rejected', 'failed'].includes(final.status); i++) {
      await sleep(3000)
      final = await api.claim(id)
    }
    expect(final.status).toBe('rejected')
    expect(final.reason).toMatch(/signature/)
    expect((await api.account(holder())).owed).toBe(owedBefore)
  })

  let paid: ClaimStatus
  it('a valid claim is paid in full to the Solana wallet', async () => {
    const owed = (await api.account(holder())).owed
    const sol = solSigner()
    const before = await solBalance(sol.pubkey)
    const r = await claimAs(holderKey, sol)
    paid = r.final
    expect(r.seen.slice(0, 4)).toEqual(['challenge', 'sign-evm', 'sign-sol', 'submit'])
    expect(r.texts.evmText).toContain(`Pay the SOL the $FLY vault owes this address to Solana wallet ${sol.pubkey}.`)
    expect(r.texts.solText).toContain(`Receive the SOL the $FLY vault owes Ethereum account ${holder()}.`)
    expect(paid.status).toBe('paid')
    expect(paid.lamports).toBe(owed)
    expect(paid.tx).toMatch(/^[1-9A-HJ-NP-Za-km-z]{80,90}$/)
    // the fly reports "paid" at confirmed commitment; getBalance answers at finalized, a few seconds later
    let after = await solBalance(sol.pubkey)
    for (let i = 0; i < 20 && after === before; i++) {
      await sleep(2000)
      after = await solBalance(sol.pubkey)
    }
    expect(after - before).toBe(owed)
  })

  it('after paying: the account shows it, a new claim is refused, a spent nonce cannot be replayed', async () => {
    // the relay learns about the payment from the fly's next push (every minute)
    const a = await owedAfterPush(holder(), (owed) => owed === 0, 120_000)
    expect(a.owed).toBe(0)
    expect(a.claimed).toBeGreaterThanOrEqual(paid.lamports!)
    expect(a.claims.some((c) => String(c.id) === String(paid.id) && c.status === 'paid' && c.tx === paid.tx)).toBe(true)
    await expect(claimAs(holderKey)).rejects.toMatchObject({ status: 400 })
    const c = await api.challenge(holder(), solSigner().pubkey)
    const junk = { nonce: c.nonce, evm: holder(), sol: solSigner().pubkey, evm_sig: ('0x' + '11'.repeat(65)) as Hex, sol_sig: bs58.encode(new Uint8Array(64)) }
    await expect(api.submitClaim(junk)).rejects.toMatchObject({ status: 400 }) // sol differs from the challenge's
    await expect(api.submitClaim({ ...junk, nonce: '0'.repeat(32) })).rejects.toMatchObject({ status: 400 }) // unknown nonce
    await expect(api.submitClaim({ ...junk, nonce: 'zz' })).rejects.toBeInstanceOf(ApiError)
  })

  it('the second holder (100k) gets a quarter of each period the first got three quarters of', async () => {
    const a1 = await api.account(holder())
    const a2 = await api.account(account(2).address)
    expect(a2.owed).toBeGreaterThan(0)
    const byPeriod = new Map(a1.allocations.map((x) => [x.period_end, x]))
    let compared = 0
    for (const x of a2.allocations) {
      const y = byPeriod.get(x.period_end)
      if (y == null) continue
      // weights are token-seconds: equal for full periods (3:1), not for the partial first period where the demo's
      // first holder locked a few seconds before the second. Lamports must follow the weights exactly (floored).
      const w1 = BigInt(y.weight)
      const w2 = BigInt(x.weight)
      if (w1 === 3n * w2) expect(Math.abs(y.lamports - 3 * x.lamports)).toBeLessThanOrEqual(3)
      const total = BigInt(y.lamports + x.lamports)
      const expected2 = (total * w2) / (w1 + w2) // within rounding, the split follows the weights
      expect(Math.abs(Number(expected2) - x.lamports)).toBeLessThanOrEqual(2)
      compared++
    }
    expect(compared).toBeGreaterThan(0)
    const r = await claimAs(2)
    expect(r.final.status).toBe('paid')
    expect(r.final.lamports).toBe(a2.owed)
  })

  it('the payment shows up as an outbound claim flow in the public history', async () => {
    let found = false
    for (let i = 0; i < 24 && !found; i++) {
      const page = await api.history('flows', null, 50)
      found = page.items.some((f) => f.kind === 'claim' && f.signature === paid.tx && f.direction === 'out' && f.lamports === paid.lamports)
      if (!found) await sleep(5000)
    }
    expect(found).toBe(true)
  })
})

describe('format sanity used by the page', () => {
  it('wei strings round-trip', () => {
    expect(formatUnits(parseEther('1234.5'), 18)).toBe('1234.5')
  })
})
