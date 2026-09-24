/** FlyVault reads (one multicall) and writes (docs/vault/SPEC.md §1, contracts/src/FlyVault.sol). */
import { simulateContract, writeContract, type Config } from '@wagmi/core'
import { erc20Abi, parseAbi, type Address, type Hex } from 'viem'
import { config } from '../config'
import { publicClient } from './chains'
import { approveExact, ensureChain, waitOk } from './evm'

export const vaultAbi = parseAbi([
  'function lock(uint256 amount)',
  'function requestWithdrawal(uint256 amount) returns (uint256 id)',
  'function cancelRequest(uint256 id)',
  'function withdraw(uint256 id)',
  'function token() view returns (address)',
  'function locked(address user) view returns (uint256)',
  'function pendingOf(address user) view returns (uint256)',
  'function totalLocked() view returns (uint256)',
  'function totalPending() view returns (uint256)',
  'function request(uint256 id) view returns (address user, uint256 amount, uint64 readyAt, uint8 state)',
  'function requestsOf(address user) view returns (uint256[] ids)',
  'function paused() view returns (bool)',
  'function WITHDRAW_DELAY() view returns (uint256)',
  'event Locked(address indexed user, uint256 amount, uint256 lockedAfter)',
  'event WithdrawRequested(address indexed user, uint256 indexed id, uint256 amount, uint64 readyAt, uint256 lockedAfter)',
  'event RequestCancelled(address indexed user, uint256 indexed id, uint256 amount, uint256 lockedAfter)',
  'event Withdrawn(address indexed user, uint256 indexed id, uint256 amount)',
  'error ZeroAddress()',
  'error ZeroAmount()',
  'error TransferAmountMismatch(uint256 expected, uint256 received)',
  'error InsufficientLocked(uint256 requested, uint256 available)',
  'error UnknownRequest(uint256 id)',
  'error NotRequestOwner(uint256 id)',
  'error RequestNotPending(uint256 id)',
  'error WithdrawalNotReady(uint256 id, uint64 readyAt)',
  'error EnforcedPause()',
  'error ExpectedPause()',
  'error ReentrancyGuardReentrantCall()',
  'error SafeERC20FailedOperation(address token)',
  'error ERC20InsufficientBalance(address sender, uint256 balance, uint256 needed)',
  'error ERC20InsufficientAllowance(address spender, uint256 allowance, uint256 needed)',
])

export const REQUEST_STATE = ['none', 'pending', 'cancelled', 'withdrawn'] as const

export interface WithdrawRequest {
  id: bigint
  amount: bigint
  readyAt: number
  state: (typeof REQUEST_STATE)[number]
}

export interface VaultTotals {
  totalLocked: bigint
  totalPending: bigint
  paused: boolean
  withdrawDelay: number
}

export interface Position {
  flyBalance: bigint
  locked: bigint
  pending: bigint
  requests: WithdrawRequest[]
}

function vaultAddr(): Address {
  if (!config.vault) throw new Error('The vault is not deployed yet.')
  return config.vault
}

export async function readTotals(): Promise<VaultTotals> {
  const address = vaultAddr()
  const [totalLocked, totalPending, paused, delay] = await publicClient().multicall({
    allowFailure: false,
    contracts: [
      { address, abi: vaultAbi, functionName: 'totalLocked' },
      { address, abi: vaultAbi, functionName: 'totalPending' },
      { address, abi: vaultAbi, functionName: 'paused' },
      { address, abi: vaultAbi, functionName: 'WITHDRAW_DELAY' },
    ],
  })
  return { totalLocked, totalPending, paused, withdrawDelay: Number(delay) }
}

export async function readPosition(user: Address): Promise<Position> {
  const address = vaultAddr()
  const fly = config.fly.address
  const client = publicClient()
  const [[locked, pending, ids], flyBalance] = await Promise.all([
    client.multicall({
      allowFailure: false,
      contracts: [
        { address, abi: vaultAbi, functionName: 'locked', args: [user] },
        { address, abi: vaultAbi, functionName: 'pendingOf', args: [user] },
        { address, abi: vaultAbi, functionName: 'requestsOf', args: [user] },
      ],
    }),
    fly ? client.readContract({ address: fly, abi: erc20Abi, functionName: 'balanceOf', args: [user] }) : Promise.resolve(0n),
  ])
  const reqs = ids.length
    ? await client.multicall({
        allowFailure: false,
        contracts: ids.map((id) => ({ address, abi: vaultAbi, functionName: 'request' as const, args: [id] as const })),
      })
    : []
  const requests = reqs.map(([, amount, readyAt, state], i) => ({
    id: ids[i],
    amount,
    readyAt: Number(readyAt),
    state: REQUEST_STATE[state] ?? 'none',
  }))
  return { flyBalance, locked, pending, requests }
}

type Step = (msg: string) => void

async function send(cfg: Config, account: Address, functionName: 'lock' | 'requestWithdrawal' | 'cancelRequest' | 'withdraw', arg: bigint): Promise<Hex> {
  const { request } = await simulateContract(cfg, {
    account,
    address: vaultAddr(),
    abi: vaultAbi,
    functionName,
    args: [arg],
    chainId: config.evm.chainId,
  })
  const hash = await writeContract(cfg, request)
  await waitOk(cfg, hash)
  return hash
}

/** Approve exactly `amount` (if needed), then lock it. */
export async function lock(cfg: Config, account: Address, amount: bigint, step: Step): Promise<Hex> {
  if (!config.fly.address) throw new Error('$FLY address is not configured for this build.')
  await ensureChain(cfg)
  step('1/2 · Approve exactly this amount for the vault in your wallet…')
  await approveExact(cfg, config.fly.address, account, vaultAddr(), amount)
  step('2/2 · Confirm the lock in your wallet…')
  return send(cfg, account, 'lock', amount)
}

export async function requestWithdrawal(cfg: Config, account: Address, amount: bigint, step: Step): Promise<Hex> {
  await ensureChain(cfg)
  step('Confirm the withdrawal request in your wallet…')
  return send(cfg, account, 'requestWithdrawal', amount)
}

export async function cancelRequest(cfg: Config, account: Address, id: bigint, step: Step): Promise<Hex> {
  await ensureChain(cfg)
  step(`Confirm cancelling request #${id} in your wallet…`)
  return send(cfg, account, 'cancelRequest', id)
}

export async function withdraw(cfg: Config, account: Address, id: bigint, step: Step): Promise<Hex> {
  await ensureChain(cfg)
  step(`Confirm withdrawing request #${id} in your wallet…`)
  return send(cfg, account, 'withdraw', id)
}
