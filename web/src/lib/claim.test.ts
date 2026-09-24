import { readFileSync } from 'node:fs'
import { ed25519 } from '@noble/curves/ed25519'
import bs58 from 'bs58'
import { hexToBytes, verifyMessage, type Hex } from 'viem'
import { privateKeyToAccount } from 'viem/accounts'
import { describe, expect, it } from 'vitest'
import { checkChallenge, evmText, solText, verifySol, type ClaimFields } from './claim'

interface Vector {
  name: string
  fields: ClaimFields
  evm_checksummed: string
  evm_text: string
  sol_text: string
  evm_sig: Hex
  sol_sig: string
  test_keys: { evm_private_key: Hex; sol_seed: string }
}

// The same file pins fly_trader/vault/claim_message.py (docs/vault/SPEC.md §2).
const vectors: Vector[] = JSON.parse(
  readFileSync(new URL('../../../tests/vectors/claim_v1.json', import.meta.url), 'utf8'),
).vectors

describe('claim texts match the Python renderer byte for byte', () => {
  it('has vectors', () => expect(vectors.length).toBeGreaterThanOrEqual(2))

  for (const v of vectors) {
    it(`${v.name}: evm_text`, () => {
      expect(evmText(v.fields)).toBe(v.evm_text)
      expect(Buffer.from(evmText(v.fields), 'utf8').equals(Buffer.from(v.evm_text, 'utf8'))).toBe(true)
    })
    it(`${v.name}: sol_text`, () => {
      expect(Buffer.from(solText(v.fields), 'utf8').equals(Buffer.from(v.sol_text, 'utf8'))).toBe(true)
    })
    it(`${v.name}: renders the address EIP-55 whatever the input case`, () => {
      const upper = { ...v.fields, evm: '0x' + v.fields.evm.slice(2).toUpperCase() }
      expect(evmText(upper)).toBe(v.evm_text)
      expect(v.evm_text).toContain(v.evm_checksummed)
    })
    it(`${v.name}: EVM signature verifies (personal_sign)`, async () => {
      expect(await verifyMessage({ address: v.evm_checksummed as Hex, message: v.evm_text, signature: v.evm_sig })).toBe(true)
      expect(privateKeyToAccount(v.test_keys.evm_private_key).address).toBe(v.evm_checksummed)
    })
    it(`${v.name}: Solana signature verifies (ed25519, base58)`, () => {
      expect(bs58.decode(v.sol_sig).length).toBe(64)
      expect(bs58.encode(ed25519.getPublicKey(hexToBytes(`0x${v.test_keys.sol_seed}`)))).toBe(v.fields.sol)
      expect(ed25519.verify(bs58.decode(v.sol_sig), new TextEncoder().encode(v.sol_text), bs58.decode(v.fields.sol))).toBe(true)
      expect(verifySol(v.sol_text, v.sol_sig, v.fields.sol)).toBe(true)
      expect(verifySol(v.sol_text + ' ', v.sol_sig, v.fields.sol)).toBe(false)
    })
  }
})

describe('checkChallenge', () => {
  const base = {
    nonce: '0123456789abcdef0123456789abcdef',
    issued_at: '2026-09-28T00:00:00Z',
    expires_at: '2026-09-28T00:15:00Z',
    domain: 'fly-trader.app',
    uri: 'https://fly-trader.app/vault.html',
    chain_id: 4663,
    sol_chain: 'mainnet',
    request_id: 'fly-vault-claim-v1',
    min_lamports: 2_000_000,
  }
  it('accepts a challenge for this host and network', () => expect(checkChallenge(base, 'fly-trader.app')).toBeNull())
  it('refuses another domain', () => expect(checkChallenge(base, 'evil.example')).toMatch(/fly-trader\.app/))
  it('refuses another chain', () => expect(checkChallenge({ ...base, chain_id: 1 }, 'fly-trader.app')).toMatch(/chain/))
  it('refuses a malformed nonce', () => expect(checkChallenge({ ...base, nonce: 'XYZ' }, 'fly-trader.app')).toMatch(/nonce/))
})
