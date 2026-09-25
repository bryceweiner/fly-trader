/**
 * Vault claims (docs/vault/SPEC.md §2): render the two texts byte-identically to
 * fly_trader/vault/claim_message.py, collect both signatures, submit, and follow the claim to payment.
 */
import { ed25519 } from '@noble/curves/ed25519'
import bs58 from 'bs58'
import { getAddress, type Address, type Hex } from 'viem'
import { config } from '../config'
import { api, ApiError, type Challenge, type ClaimStatus } from './api'
import { publicClient } from './chains'

export const REQUEST_ID = 'fly-vault-claim-v1'
export const TTL_S = 900

export interface ClaimFields {
  domain: string
  uri: string
  chain_id: number
  sol_chain: 'mainnet' | 'devnet' | string
  evm: string // any case; rendered EIP-55
  sol: string
  nonce: string
  issued_at: string
  expires_at: string
}

export function evmText(f: ClaimFields): string {
  const a = getAddress(f.evm)
  return (
    `${f.domain} wants you to sign in with your Ethereum account:\n${a}\n\n` +
    `Pay the SOL the $FLY vault owes this address to Solana wallet ${f.sol}.\n\n` +
    `URI: ${f.uri}\nVersion: 1\nChain ID: ${Math.trunc(Number(f.chain_id))}\nNonce: ${f.nonce}\n` +
    `Issued At: ${f.issued_at}\nExpiration Time: ${f.expires_at}\nRequest ID: ${REQUEST_ID}`
  )
}

export function solText(f: ClaimFields): string {
  const a = getAddress(f.evm)
  return (
    `${f.domain} wants you to sign in with your Solana account:\n${f.sol}\n\n` +
    `Receive the SOL the $FLY vault owes Ethereum account ${a}.\n\n` +
    `URI: ${f.uri}\nVersion: 1\nChain ID: ${f.sol_chain}\nNonce: ${f.nonce}\n` +
    `Issued At: ${f.issued_at}\nExpiration Time: ${f.expires_at}\nRequest ID: ${REQUEST_ID}`
  )
}

const RFC3339 = /^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$/
const B58 = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/

export function isSolanaAddress(s: string): boolean {
  if (!B58.test(s)) return false
  try {
    return bs58.decode(s).length === 32
  } catch {
    return false
  }
}

/**
 * Refuses a challenge this page should not sign: wrong network, or a domain other than the one the page is
 * served from (a wallet would show a phishing warning for that, and rightly so).
 */
export function checkChallenge(c: Challenge, host: string): string | null {
  if (c.request_id !== REQUEST_ID) return 'The relay returned an unknown claim version.'
  if (!/^[0-9a-f]{32}$/.test(c.nonce)) return 'The relay returned a malformed nonce.'
  if (!RFC3339.test(c.issued_at) || !RFC3339.test(c.expires_at)) return 'The relay returned malformed timestamps.'
  if (Number(c.chain_id) !== config.evm.chainId || c.sol_chain !== config.solana.cluster) {
    return `The relay issues claims for chain ${c.chain_id} / Solana ${c.sol_chain}, but this site is built for ${config.evm.chainId} / ${config.solana.cluster}.`
  }
  if (c.domain !== host) return `The relay issues claims for ${c.domain}. Open the Vault page at https://${c.domain} to claim.`
  // The URI goes verbatim into both signed texts: no line breaks or spaces (which could forge extra lines), and
  // only on the page's own origin.
  let origin: string | null = null
  try {
    origin = /^[\x21-\x7e]+$/.test(c.uri) ? new URL(c.uri).origin : null
  } catch {
    origin = null
  }
  const local = /^(localhost|127\.0\.0\.1)(:\d+)?$/.test(host) // the local dry run is served over http
  if (origin == null || (origin !== `https://${host}` && !(local && origin === `http://${host}`))) {
    return 'The relay returned a malformed claim URI.'
  }
  return null
}

export type ClaimStep =
  | { step: 'challenge' }
  | { step: 'sign-evm'; fields: ClaimFields; evmText: string; solText: string }
  | { step: 'sign-sol' }
  | { step: 'submit' }
  | { step: 'status'; claim: ClaimStatus }

export interface ClaimSigners {
  evm: Address
  sol: string
  signEvm(text: string): Promise<Hex>
  /** returns the raw 64-byte ed25519 signature */
  signSol(message: Uint8Array): Promise<Uint8Array>
}

export class ClaimError extends Error {}

export const FINAL_STATUSES = new Set(['paid', 'rejected', 'failed'])

/** Solana wallets differ: some return the bytes, some `{ signature }`. */
export function solSignatureBytes(res: unknown): Uint8Array {
  const raw = res instanceof Uint8Array ? res : (res as { signature?: unknown } | null)?.signature
  if (raw instanceof Uint8Array && raw.length === 64) return raw
  if (Array.isArray(raw) && raw.length === 64) return Uint8Array.from(raw as number[])
  throw new ClaimError('The Solana wallet returned an unexpected signature.')
}

export function verifySol(text: string, sigB58: string, sol: string): boolean {
  try {
    return ed25519.verify(bs58.decode(sigB58), new TextEncoder().encode(text), bs58.decode(sol))
  } catch {
    return false
  }
}

/**
 * The whole claim: challenge -> EVM personal_sign -> Solana signMessage -> local checks -> POST -> poll.
 * Resolves with the final claim status (paid, rejected or failed), or throws ClaimError.
 */
export async function runClaim(
  s: ClaimSigners,
  onStep: (s: ClaimStep) => void,
  opts: { minLamports?: (n: number) => void; signal?: AbortSignal; host?: string } = {},
): Promise<ClaimStatus> {
  onStep({ step: 'challenge' })
  const c = await api.challenge(s.evm, s.sol)
  opts.minLamports?.(c.min_lamports)
  const bad = checkChallenge(c, opts.host ?? location.host)
  if (bad) throw new ClaimError(bad)

  const fields: ClaimFields = {
    domain: c.domain,
    uri: c.uri,
    chain_id: Number(c.chain_id),
    sol_chain: c.sol_chain,
    evm: s.evm,
    sol: s.sol,
    nonce: c.nonce,
    issued_at: c.issued_at,
    expires_at: c.expires_at,
  }
  const et = evmText(fields)
  const st = solText(fields)
  onStep({ step: 'sign-evm', fields, evmText: et, solText: st })
  const evmSig = await s.signEvm(et)

  onStep({ step: 'sign-sol' })
  const solSig = bs58.encode(await s.signSol(new TextEncoder().encode(st)))

  if (!verifySol(st, solSig, s.sol)) {
    throw new ClaimError('Your Solana wallet signed a different message than the one shown, so the fly would reject it. Try another Solana wallet.')
  }
  const evmOk = await publicClient()
    .verifyMessage({ address: s.evm, message: et, signature: evmSig })
    .catch(() => null) // RPC trouble: let the fly decide
  if (evmOk === false) {
    throw new ClaimError('The EVM signature does not match the connected address, so the fly would reject it.')
  }

  onStep({ step: 'submit' })
  let id: number | string
  try {
    ;({ id } = await api.submitClaim({ nonce: c.nonce, evm: getAddress(s.evm), sol: s.sol, evm_sig: evmSig, sol_sig: solSig }))
  } catch (e) {
    if (e instanceof ApiError && e.status === 409) {
      throw new ClaimError('A claim for this address is already in progress, or this challenge was already used. Wait for it to finish.')
    }
    throw e
  }

  let delay = 2000
  for (;;) {
    if (opts.signal?.aborted) throw new ClaimError('Stopped following the claim. Its status stays on the relay.')
    const claim = await api.claim(id).catch((e) => {
      if (e instanceof ApiError && e.status < 500 && e.status !== 429) throw e
      return null
    })
    if (claim) {
      onStep({ step: 'status', claim })
      if (FINAL_STATUSES.has(claim.status)) return claim
    }
    await new Promise((r) => setTimeout(r, delay))
    delay = Math.min(delay * 1.5, 10_000)
  }
}
