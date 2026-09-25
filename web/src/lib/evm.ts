/** Wallet-side EVM helpers shared by the Trading and Vault pages: chain switching, exact approvals, errors. */
import {
  getAccount,
  readContract,
  switchChain,
  waitForTransactionReceipt,
  writeContract,
  type Config,
} from '@wagmi/core'
import { BaseError, ContractFunctionRevertedError, erc20Abi, UserRejectedRequestError, type Address, type Hex } from 'viem'
import { config } from '../config'
import { token } from './format'

export async function ensureChain(cfg: Config): Promise<void> {
  const { chainId } = getAccount(cfg)
  if (chainId === config.evm.chainId) return
  try {
    await switchChain(cfg, { chainId: config.evm.chainId })
  } catch (e) {
    if (isUserRejection(e)) throw new Error('You declined the network switch in your wallet.')
    throw new Error(
      `Your wallet could not switch to ${config.evm.name} (chain ${config.evm.chainId}). Add it in your wallet ` +
        `(RPC ${config.evm.rpcUrl}) or use a wallet that can add custom networks.`,
    )
  }
}

export async function waitOk(cfg: Config, hash: Hex): Promise<void> {
  const r = await waitForTransactionReceipt(cfg, { hash, chainId: config.evm.chainId })
  if (r.status !== 'success') throw new Error(`Transaction ${hash} reverted.`)
}

/** Approves exactly `amount` (never unlimited) if the current allowance is short. Returns the tx hash, if any. */
export async function approveExact(
  cfg: Config,
  tokenAddr: Address,
  owner: Address,
  spender: Address,
  amount: bigint,
): Promise<Hex | null> {
  const allowance = await readContract(cfg, {
    address: tokenAddr,
    abi: erc20Abi,
    functionName: 'allowance',
    args: [owner, spender],
    chainId: config.evm.chainId,
  })
  if (allowance >= amount) return null
  const hash = await writeContract(cfg, {
    address: tokenAddr,
    abi: erc20Abi,
    functionName: 'approve',
    args: [spender, amount],
    chainId: config.evm.chainId,
  })
  await waitOk(cfg, hash)
  return hash
}

export function isUserRejection(e: unknown): boolean {
  if (e instanceof BaseError && e.walk((x) => x instanceof UserRejectedRequestError)) return true
  const code = (e as { code?: number })?.code
  const msg = String((e as Error)?.message || '')
  return code === 4001 || /user (rejected|denied)|rejected the request|request rejected/i.test(msg)
}

/** Custom errors of FlyVault (contracts/src/FlyVault.sol) and the OpenZeppelin ones it can bubble up. */
const REVERTS: Record<string, (args: readonly unknown[]) => string> = {
  EnforcedPause: () => 'The vault is paused: new locks and cancellations are blocked for now. Withdrawals still work.',
  ZeroAmount: () => 'Enter an amount greater than zero.',
  ZeroAddress: () => 'Zero address.',
  TransferAmountMismatch: () => 'The vault received a different amount than requested, so it refused the lock.',
  InsufficientLocked: (a) => `You only have ${token(a[1] as bigint)} $FLY locked.`,
  UnknownRequest: (a) => `Withdrawal request #${a[0]} does not exist.`,
  NotRequestOwner: (a) => `Withdrawal request #${a[0]} belongs to another address.`,
  RequestNotPending: (a) => `Withdrawal request #${a[0]} was already cancelled or withdrawn.`,
  WithdrawalNotReady: (a) =>
    `Withdrawal request #${a[0]} is not ready until ${new Date(Number(a[1]) * 1000).toUTCString()}.`,
  ERC20InsufficientBalance: () => 'Your $FLY balance is too low for that amount.',
  ERC20InsufficientAllowance: () => 'The approval is lower than the amount. Approve again.',
  ReentrancyGuardReentrantCall: () => 'Reentrant call refused.',
}

export function humanError(e: unknown): string {
  if (isUserRejection(e)) return 'You rejected the request in your wallet.'
  if (e instanceof BaseError) {
    const revert = e.walk((x) => x instanceof ContractFunctionRevertedError) as ContractFunctionRevertedError | null
    const name = revert?.data?.errorName
    if (name && REVERTS[name]) return REVERTS[name](revert?.data?.args ?? [])
    if (name) return `The contract refused: ${name}.`
    if (/insufficient funds/i.test(e.message)) return `Not enough ETH on ${config.evm.name} to pay for gas.`
    if (/Return amount is not enough/i.test(e.message)) {
      return 'The pools would return less than your slippage tolerance allows, so the router refuses the swap. Raise the slippage and try again.'
    }
    return e.shortMessage || e.message
  }
  return e instanceof Error ? e.message : String(e)
}
