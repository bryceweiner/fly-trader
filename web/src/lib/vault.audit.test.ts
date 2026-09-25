/**
 * Audit of the vault write/read paths and the shared EVM helpers (2026-09-25), with wagmi and the RPC client mocked.
 * "AUDIT FINDING" tests were failing before the 2026-09-25 fixes and now guard them.
 */
import {
  BaseError,
  ContractFunctionExecutionError,
  ContractFunctionRevertedError,
  encodeErrorResult,
  UserRejectedRequestError,
  type Address,
} from 'viem'
import { beforeEach, describe, expect, it, vi } from 'vitest'

const VAULT = '0x00000000000000000000000000000000000000aa' as Address
const FLYA = '0x2fc7f9e2911f20b2c4660d2aef808aa91bddb3d3' as Address
const USER = '0x1111111111111111111111111111111111111111' as Address
const CHAIN = 4663

vi.mock('../config', async (orig) => {
  const m = (await orig()) as typeof import('../config')
  return { ...m, config: { ...m.config, vault: '0x00000000000000000000000000000000000000aa' } }
})

const w = vi.hoisted(() => ({
  getAccount: vi.fn(),
  switchChain: vi.fn(),
  readContract: vi.fn(),
  writeContract: vi.fn(),
  simulateContract: vi.fn(),
  waitForTransactionReceipt: vi.fn(),
}))
vi.mock('@wagmi/core', () => w)

const client = vi.hoisted(() => ({ multicall: vi.fn(), readContract: vi.fn() }))
vi.mock('./chains', () => ({ publicClient: () => client }))

import { approveExact, ensureChain, humanError, isUserRejection, waitOk } from './evm'
import * as vault from './vault'

const cfg = {} as never
const H1 = `0x${'1'.repeat(64)}` as const
const H2 = `0x${'2'.repeat(64)}` as const

beforeEach(() => {
  for (const f of Object.values(w)) f.mockReset()
  client.multicall.mockReset()
  client.readContract.mockReset()
  w.getAccount.mockReturnValue({ chainId: CHAIN, address: USER })
  w.simulateContract.mockImplementation(async (_c, p) => ({ request: { ...p, __simulated: true } }))
  w.writeContract.mockResolvedValue(H1)
  w.waitForTransactionReceipt.mockImplementation(async (_c, { hash }) => ({ status: 'success', transactionHash: hash }))
})

const revert = (errorName: string, args: readonly unknown[] = []) =>
  new ContractFunctionExecutionError(
    new ContractFunctionRevertedError({
      abi: vault.vaultAbi,
      functionName: 'lock',
      data: encodeErrorResult({ abi: vault.vaultAbi, errorName: errorName as never, args: args as never }),
    }),
    { abi: vault.vaultAbi, functionName: 'lock', args: [1n], contractAddress: VAULT },
  )

describe('ensureChain', () => {
  it('does nothing on the right chain', async () => {
    await ensureChain(cfg)
    expect(w.switchChain).not.toHaveBeenCalled()
  })
  it('switches a wallet on another chain', async () => {
    w.getAccount.mockReturnValue({ chainId: 1, address: USER })
    await ensureChain(cfg)
    expect(w.switchChain).toHaveBeenCalledWith(cfg, { chainId: CHAIN })
  })
  it('refuses (no tx) when the person declines the switch', async () => {
    w.getAccount.mockReturnValue({ chainId: 1, address: USER })
    w.switchChain.mockRejectedValue(new UserRejectedRequestError(new Error('no')))
    await expect(ensureChain(cfg)).rejects.toThrow(/declined the network switch/)
  })
  it('explains a wallet that cannot add the chain', async () => {
    w.getAccount.mockReturnValue({ chainId: undefined, address: USER })
    w.switchChain.mockRejectedValue(new Error('Unrecognized chain ID'))
    await expect(ensureChain(cfg)).rejects.toThrow(/could not switch/)
  })
  it('a declined switch stops lock before any approval prompt', async () => {
    w.getAccount.mockReturnValue({ chainId: 1, address: USER })
    w.switchChain.mockRejectedValue({ code: 4001 })
    await expect(vault.lock(cfg, USER, 5n, () => {})).rejects.toThrow(/declined/)
    expect(w.readContract).not.toHaveBeenCalled()
    expect(w.writeContract).not.toHaveBeenCalled()
  })
})

describe('approveExact', () => {
  it('skips the prompt when the allowance already covers the amount', async () => {
    w.readContract.mockResolvedValue(10n)
    expect(await approveExact(cfg, FLYA, USER, VAULT, 10n)).toBeNull()
    expect(w.writeContract).not.toHaveBeenCalled()
    expect(w.readContract.mock.calls[0][1]).toMatchObject({ functionName: 'allowance', args: [USER, VAULT], chainId: CHAIN })
  })
  it('approves exactly the amount (never unlimited) to the given spender and waits for it', async () => {
    w.readContract.mockResolvedValue(3n)
    expect(await approveExact(cfg, FLYA, USER, VAULT, 10n)).toBe(H1)
    expect(w.writeContract.mock.calls[0][1]).toMatchObject({ address: FLYA, functionName: 'approve', args: [VAULT, 10n], chainId: CHAIN })
    expect(w.waitForTransactionReceipt).toHaveBeenCalledWith(cfg, expect.objectContaining({ hash: H1, chainId: CHAIN }))
  })
  it('a reverted approval throws, so nothing after it runs', async () => {
    w.readContract.mockResolvedValue(0n)
    w.waitForTransactionReceipt.mockResolvedValue({ status: 'reverted', transactionHash: H1 })
    await expect(approveExact(cfg, FLYA, USER, VAULT, 10n)).rejects.toThrow(/reverted/)
  })
})

describe('vault.lock', () => {
  it('approve (exact, to the vault) -> receipt -> simulate lock(amount) -> send -> receipt, in that order', async () => {
    const order: string[] = []
    w.readContract.mockImplementation(async () => (order.push('allowance'), 0n))
    w.writeContract.mockImplementation(async (_c, p) => (order.push(`write:${p.functionName}`), p.functionName === 'approve' ? H1 : H2))
    w.waitForTransactionReceipt.mockImplementation(async (_c, { hash }) => (order.push(`wait:${hash === H1 ? 'approve' : 'lock'}`), { status: 'success', transactionHash: hash }))
    w.simulateContract.mockImplementation(async (_c, p) => (order.push(`sim:${p.functionName}`), { request: p }))
    const steps: string[] = []
    expect(await vault.lock(cfg, USER, 7n, (m) => steps.push(m))).toBe(H2)
    expect(order).toEqual(['allowance', 'write:approve', 'wait:approve', 'sim:lock', 'write:lock', 'wait:lock'])
    expect(w.writeContract.mock.calls[0][1].args).toEqual([VAULT, 7n])
    expect(w.simulateContract.mock.calls[0][1]).toMatchObject({ account: USER, address: VAULT, functionName: 'lock', args: [7n], chainId: CHAIN })
    expect(steps).toHaveLength(2)
  })
  it('an existing sufficient allowance is reused: one prompt only', async () => {
    w.readContract.mockResolvedValue(100n)
    await vault.lock(cfg, USER, 7n, () => {})
    expect(w.writeContract).toHaveBeenCalledTimes(1)
    expect(w.writeContract.mock.calls[0][1].functionName).toBe('lock')
  })
  it('a lock that would revert (paused) never reaches the wallet, and reads as the paused message', async () => {
    w.readContract.mockResolvedValue(100n)
    w.simulateContract.mockRejectedValue(revert('EnforcedPause'))
    const e = await vault.lock(cfg, USER, 7n, () => {}).catch((x) => x)
    expect(w.writeContract).not.toHaveBeenCalled()
    expect(humanError(e)).toMatch(/paused/)
  })
})

describe('request / cancel / withdraw send exactly one call with the right id or amount', () => {
  it.each([
    ['requestWithdrawal', vault.requestWithdrawal, 5n * 10n ** 18n],
    ['cancelRequest', vault.cancelRequest, 3n],
    ['withdraw', vault.withdraw, 0n],
  ] as const)('%s', async (fn, call, arg) => {
    const hash = await call(cfg, USER, arg, () => {})
    expect(hash).toBe(H1)
    expect(w.simulateContract).toHaveBeenCalledTimes(1)
    expect(w.simulateContract.mock.calls[0][1]).toMatchObject({ account: USER, address: VAULT, functionName: fn, args: [arg], chainId: CHAIN })
    expect(w.writeContract.mock.calls[0][1]).toMatchObject({ __simulated: true, functionName: fn })
    expect(w.readContract).not.toHaveBeenCalled() // no approvals outside lock
  })
  it('a mined-but-reverted tx is reported as a failure, not "Confirmed"', async () => {
    w.waitForTransactionReceipt.mockResolvedValue({ status: 'reverted', transactionHash: H1 })
    await expect(vault.withdraw(cfg, USER, 1n, () => {})).rejects.toThrow(/reverted/)
  })
})

describe('humanError names every vault revert in plain words', () => {
  it.each([
    ['EnforcedPause', [], /paused/],
    ['ZeroAmount', [], /greater than zero/],
    ['InsufficientLocked', [10n, 2n * 10n ** 18n], /only have 2 \$FLY locked/],
    ['UnknownRequest', [9n], /#9 does not exist/],
    ['NotRequestOwner', [9n], /#9 belongs to another address/],
    ['RequestNotPending', [9n], /#9 was already/],
    ['WithdrawalNotReady', [9n, 1_790_000_000n], /#9 is not ready until .*2026/],
    ['ERC20InsufficientBalance', [USER, 0n, 1n], /balance is too low/],
    ['ERC20InsufficientAllowance', [VAULT, 0n, 1n], /Approve again/],
    ['ExpectedPause', [], /refused: ExpectedPause/],
  ] as const)('%s', (name, args, re) => expect(humanError(revert(name, args))).toMatch(re))
  it('user rejection in all its shapes', () => {
    expect(isUserRejection(new UserRejectedRequestError(new Error('x')))).toBe(true)
    expect(isUserRejection({ code: 4001 })).toBe(true)
    expect(isUserRejection(new Error('User denied transaction signature'))).toBe(true)
    expect(isUserRejection(new Error('insufficient funds'))).toBe(false)
    expect(humanError(new BaseError('insufficient funds for gas * price + value'))).toMatch(/Not enough ETH/)
    expect(humanError('plain')).toBe('plain')
  })
})

describe('readPosition', () => {
  it('reads only the connected address and maps request states by the contract enum order', async () => {
    client.multicall
      .mockResolvedValueOnce([10n, 4n, [3n, 7n, 8n]])
      .mockResolvedValueOnce([
        [USER, 1n, 100n, 1],
        [USER, 2n, 200n, 2],
        [USER, 3n, 300n, 3],
      ])
    client.readContract.mockResolvedValue(55n)
    const p = await vault.readPosition(USER)
    expect(client.multicall.mock.calls[0][0].contracts.map((c: { args: unknown[] }) => c.args)).toEqual([[USER], [USER], [USER]])
    expect(client.multicall.mock.calls[1][0].contracts.map((c: { args: unknown[] }) => c.args)).toEqual([[3n], [7n], [8n]])
    expect(p).toEqual({
      flyBalance: 55n,
      locked: 10n,
      pending: 4n,
      requests: [
        { id: 3n, amount: 1n, readyAt: 100, state: 'pending' },
        { id: 7n, amount: 2n, readyAt: 200, state: 'cancelled' },
        { id: 8n, amount: 3n, readyAt: 300, state: 'withdrawn' },
      ],
    })
  })
  it('no requests: a single multicall', async () => {
    client.multicall.mockResolvedValueOnce([0n, 0n, []])
    client.readContract.mockResolvedValue(0n)
    expect((await vault.readPosition(USER)).requests).toEqual([])
    expect(client.multicall).toHaveBeenCalledTimes(1)
  })
  it('readTotals', async () => {
    client.multicall.mockResolvedValueOnce([5n, 1n, true, 604800n])
    expect(await vault.readTotals()).toEqual({ totalLocked: 5n, totalPending: 1n, paused: true, withdrawDelay: 604800 })
  })
})

/*
 * AUDIT FINDING (low): waitOk trusts any successful receipt. wagmi/viem's waitForTransactionReceipt resolves with
 * the REPLACEMENT's receipt when the wallet speeds up or cancels the tx (reason 'cancelled' = a 0-value self
 * transfer, status 'success'). The page then shows "Confirmed" with the original hash for a lock / withdraw that
 * never happened.
 */
describe('AUDIT FINDING: a cancelled (replaced) transaction is not reported as confirmed', () => {
  it('waitOk throws when the receipt is for a different transaction', async () => {
    w.waitForTransactionReceipt.mockResolvedValue({ status: 'success', transactionHash: H2 })
    await expect(waitOk(cfg, H1)).rejects.toThrow()
  })
})
