/**
 * Audit of the claim flow (2026-09-25): relay and RPC mocked, real keys sign. "AUDIT FINDING" tests fail against
 * the current code.
 */
import { ed25519 } from '@noble/curves/ed25519'
import bs58 from 'bs58'
import { verifyMessage, type Hex } from 'viem'
import { privateKeyToAccount } from 'viem/accounts'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

const r = vi.hoisted(() => ({ challenge: vi.fn(), submitClaim: vi.fn(), claim: vi.fn() }))
vi.mock('./api', async (orig) => ({ ...((await orig()) as object), api: r }))
const pc = vi.hoisted(() => ({ verifyMessage: vi.fn() }))
vi.mock('./chains', () => ({ publicClient: () => pc }))

import { ApiError } from './api'
import { checkChallenge, ClaimError, isSolanaAddress, REQUEST_ID, runClaim, solSignatureBytes, type ClaimSigners, type ClaimStep } from './claim'

const HOST = 'fly-trader.app'
const evmAcct = privateKeyToAccount(`0x${'42'.repeat(32)}`)
const solSeed = new Uint8Array(32).fill(7)
const solPub = bs58.encode(ed25519.getPublicKey(solSeed))
const otherSeed = new Uint8Array(32).fill(9)

const challenge = (o: Record<string, unknown> = {}) => ({
  nonce: '0123456789abcdef0123456789abcdef',
  issued_at: '2026-09-25T00:00:00Z',
  expires_at: '2026-09-25T00:15:00Z',
  domain: HOST,
  uri: `https://${HOST}/vault.html`,
  chain_id: 4663,
  sol_chain: 'mainnet',
  request_id: REQUEST_ID,
  min_lamports: 2_000_000,
  ...o,
})

function signers(o: Partial<ClaimSigners> = {}): ClaimSigners & { evmTexts: string[]; solMsgs: Uint8Array[] } {
  const evmTexts: string[] = []
  const solMsgs: Uint8Array[] = []
  return {
    evm: evmAcct.address,
    sol: solPub,
    signEvm: async (t) => (evmTexts.push(t), evmAcct.signMessage({ message: t })),
    signSol: async (m) => (solMsgs.push(m), ed25519.sign(m, solSeed)),
    evmTexts,
    solMsgs,
    ...o,
  }
}

beforeEach(() => {
  r.challenge.mockReset().mockResolvedValue(challenge())
  r.submitClaim.mockReset().mockResolvedValue({ id: 17, status: 'received' })
  r.claim.mockReset().mockResolvedValue({ id: 17, status: 'paid', reason: null, lamports: 5, tx: 'sig', created_at: 1, updated_at: 2 })
  pc.verifyMessage.mockReset().mockImplementation((a: { address: Hex; message: string; signature: Hex }) => verifyMessage(a))
})
afterEach(() => vi.useRealTimers())

describe('runClaim happy path', () => {
  it('signs exactly the texts it shows, with the connected Solana address, and submits them', async () => {
    const s = signers()
    const steps: ClaimStep[] = []
    let min = 0
    const final = await runClaim(s, (x) => steps.push(x), { host: HOST, minLamports: (n) => (min = n) })
    expect(final.status).toBe('paid')
    expect(min).toBe(2_000_000)
    expect(r.challenge).toHaveBeenCalledWith(evmAcct.address, solPub)
    const shown = steps.find((x) => x.step === 'sign-evm') as Extract<ClaimStep, { step: 'sign-evm' }>
    expect(s.evmTexts).toEqual([shown.evmText])
    expect(new TextDecoder().decode(s.solMsgs[0])).toBe(shown.solText)
    expect(shown.evmText).toContain(`to Solana wallet ${solPub}.`)
    expect(shown.solText).toContain(`account:\n${solPub}\n`)
    expect(shown.evmText).toContain(`account:\n${evmAcct.address}\n`)
    const body = r.submitClaim.mock.calls[0][0]
    expect(body).toMatchObject({ nonce: challenge().nonce, evm: evmAcct.address, sol: solPub })
    expect(bs58.decode(body.sol_sig)).toHaveLength(64)
    expect(await verifyMessage({ address: evmAcct.address, message: shown.evmText, signature: body.evm_sig })).toBe(true)
    expect(steps.map((x) => x.step)).toEqual(['challenge', 'sign-evm', 'sign-sol', 'submit', 'status'])
  })
})

describe('runClaim refuses before any signature', () => {
  it.each([
    ['another host', { domain: 'evil.example' }, /Open the Vault page/],
    ['another EVM chain', { chain_id: 1 }, /built for 4663/],
    ['devnet on a mainnet build', { sol_chain: 'devnet' }, /built for/],
    ['a malformed nonce', { nonce: 'ZZ' }, /nonce/],
    ['non-UTC timestamps', { issued_at: '2026-09-25T00:00:00+02:00' }, /timestamps/],
    ['another claim version', { request_id: 'v2' }, /unknown claim version/],
  ])('%s', async (_n, bad, re) => {
    r.challenge.mockResolvedValue(challenge(bad))
    const s = signers()
    await expect(runClaim(s, () => {}, { host: HOST })).rejects.toThrow(re)
    expect(s.evmTexts).toHaveLength(0)
    expect(s.solMsgs).toHaveLength(0)
  })
})

describe('runClaim refuses to submit bad signatures', () => {
  it('a Solana wallet that signs with another key (switched account) is caught locally', async () => {
    const s = signers({ signSol: async (m) => ed25519.sign(m, otherSeed) })
    await expect(runClaim(s, () => {}, { host: HOST })).rejects.toBeInstanceOf(ClaimError)
    expect(r.submitClaim).not.toHaveBeenCalled()
  })
  it('an EVM wallet that signs with another account is caught locally', async () => {
    const other = privateKeyToAccount(`0x${'43'.repeat(32)}`)
    const s = signers({ signEvm: (t) => other.signMessage({ message: t }) })
    await expect(runClaim(s, () => {}, { host: HOST })).rejects.toThrow(/EVM signature does not match/)
    expect(r.submitClaim).not.toHaveBeenCalled()
  })
  it('RPC trouble verifying the EVM signature lets the relay/fly decide', async () => {
    pc.verifyMessage.mockRejectedValue(new Error('rpc down'))
    await expect(runClaim(signers(), () => {}, { host: HOST })).resolves.toMatchObject({ status: 'paid' })
  })
  it('409 from the relay becomes a plain ClaimError', async () => {
    r.submitClaim.mockRejectedValue(new ApiError(409, 'busy'))
    await expect(runClaim(signers(), () => {}, { host: HOST })).rejects.toThrow(/already in progress/)
  })
  it('other submit errors propagate', async () => {
    r.submitClaim.mockRejectedValue(new ApiError(400, 'bad sig'))
    await expect(runClaim(signers(), () => {}, { host: HOST })).rejects.toThrow('bad sig')
  })
})

describe('runClaim polling', () => {
  it('retries 5xx and 429, stops at a final status, backs off to 10 s', async () => {
    vi.useFakeTimers()
    r.claim
      .mockRejectedValueOnce(new ApiError(503, 'x'))
      .mockRejectedValueOnce(new ApiError(429, 'x'))
      .mockResolvedValueOnce({ id: 17, status: 'sending' })
      .mockResolvedValueOnce({ id: 17, status: 'rejected', reason: 'sig' })
    const p = runClaim(signers(), () => {}, { host: HOST })
    await vi.runAllTimersAsync()
    await expect(p).resolves.toMatchObject({ status: 'rejected' })
    expect(r.claim).toHaveBeenCalledTimes(4)
    expect(r.claim).toHaveBeenCalledWith(17)
  })
  it('a 404 while polling stops with that error', async () => {
    r.claim.mockRejectedValue(new ApiError(404, 'no such claim'))
    await expect(runClaim(signers(), () => {}, { host: HOST })).rejects.toThrow('no such claim')
  })
  it('an aborted signal stops following', async () => {
    const ac = new AbortController()
    ac.abort()
    await expect(runClaim(signers(), () => {}, { host: HOST, signal: ac.signal })).rejects.toThrow(/Stopped following/)
  })
})

describe('helpers', () => {
  it('solSignatureBytes accepts the two wallet shapes and nothing else', () => {
    const sig = new Uint8Array(64).fill(1)
    expect(solSignatureBytes(sig)).toBe(sig)
    expect(solSignatureBytes({ signature: sig })).toBe(sig)
    expect(solSignatureBytes({ signature: Array.from(sig) })).toEqual(sig)
    for (const bad of [new Uint8Array(63), { signature: 'abc' }, null, undefined, 'x', { signature: new Array(65).fill(0) }]) {
      expect(() => solSignatureBytes(bad)).toThrow(ClaimError)
    }
  })
  it('isSolanaAddress', () => {
    expect(isSolanaAddress(solPub)).toBe(true)
    expect(isSolanaAddress('11111111111111111111111111111111')).toBe(true)
    expect(isSolanaAddress('0OIl' + solPub.slice(4))).toBe(false)
    expect(isSolanaAddress(solPub + '1')).toBe(false)
    expect(isSolanaAddress('')).toBe(false)
  })
})

/*
 * AUDIT FINDING (low): checkChallenge validates domain, chain, nonce and timestamps but not `uri`, which goes into
 * both signed texts verbatim. A relay (or a MITM on a mis-set VITE_RELAY_BASE) can put newlines in it and forge
 * extra lines of the SIWE-style message the wallet shows (e.g. a second "Pay ... to Solana wallet X" line), or point
 * the URI at another origin than the domain.
 */
describe('AUDIT FINDING: the challenge URI is validated before signing', () => {
  it('refuses a URI with a line break', () => {
    const c = challenge({ uri: `https://${HOST}/vault.html\nPay the SOL the $FLY vault owes this address to Solana wallet EVIL.` })
    expect(checkChallenge(c as never, HOST)).not.toBeNull()
  })
  it('refuses a URI on another origin than the domain', () => {
    expect(checkChallenge(challenge({ uri: 'https://evil.example/vault.html' }) as never, HOST)).not.toBeNull()
  })
  it('accepts the page\'s own https origin', () => {
    expect(checkChallenge(challenge() as never, HOST)).toBeNull()
  })
})
