/** Shared plumbing for the e2e tests: a wagmi Config whose "wallet" is an anvil account, plus raw RPC helpers. */
import { connect, createConfig, http as wagmiHttp, mock, type Config } from '@wagmi/core'
import { createPublicClient, createWalletClient, defineChain, http, parseEther, type Address, type Chain, type Hex } from 'viem'
import { generatePrivateKey, privateKeyToAccount } from 'viem/accounts'
import { config } from '../src/config'

/** anvil's well-known development keys (never fund these on a real chain). Indexes 0–2 belong to tools/vault_demo.py. */
export const ANVIL_KEYS: Hex[] = [
  '0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80',
  '0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d',
  '0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a',
  '0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6',
  '0x47e179ec197488593b187f80a00eb0da91f1b9d0b13f8733639f19c30a34926a',
  '0x8b3a350cf5c34c9194ca85829a2df0ec3153be0318b5e2d3348e872092edffba',
  '0x92db14e403b83dfe3df233f83dfa3a0d7096f21ca9b0d6d6b8d88b2b4ec1564e',
]

export const chain: Chain = defineChain({
  id: config.evm.chainId,
  name: config.evm.name,
  nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
  rpcUrls: { default: { http: [config.evm.rpcUrl] } },
  contracts: { multicall3: { address: config.multicall3 } },
})

export const account = (i: number) => privateKeyToAccount(ANVIL_KEYS[i])

/** What the page gets from AppKit: a connected wagmi Config. The mock connector signs through anvil's unlocked accounts. */
export async function walletFor(i: number): Promise<{ cfg: Config; address: Address }> {
  const address = account(i).address
  const cfg = createConfig({
    chains: [chain],
    connectors: [mock({ accounts: [address] })],
    transports: { [chain.id]: wagmiHttp(config.evm.rpcUrl, { timeout: 180_000 }) },
  })
  await connect(cfg, { connector: cfg.connectors[0], chainId: chain.id })
  return { cfg, address }
}

export interface FreshWallet {
  cfg: Config
  address: Address
  account: ReturnType<typeof privateKeyToAccount>
}

/** A brand-new key, funded with ETH and impersonated on anvil so the mock connector can send from it: every run of
 *  the suite starts from an empty position, whatever the chain already holds. */
export async function freshWallet(): Promise<FreshWallet> {
  const account = privateKeyToAccount(generatePrivateKey())
  await rpc('anvil_setBalance', [account.address, '0x' + parseEther('10').toString(16)])
  await rpc('anvil_impersonateAccount', [account.address])
  const cfg = createConfig({
    chains: [chain],
    connectors: [mock({ accounts: [account.address] })],
    transports: { [chain.id]: wagmiHttp(config.evm.rpcUrl, { timeout: 180_000 }) },
  })
  await connect(cfg, { connector: cfg.connectors[0], chainId: chain.id })
  return { cfg, address: account.address, account }
}

/** Direct clients for test setup (minting, pausing, funding) outside the page code under test. */
/** Long timeout: a mainnet fork pulls a swap's storage from the remote node on first touch. */
export const pub = createPublicClient({ chain, transport: http(config.evm.rpcUrl, { timeout: 180_000 }) })
export const wallet = (i: number) => createWalletClient({ chain, transport: http(config.evm.rpcUrl), account: account(i) })
export const walletOf = (acct: ReturnType<typeof privateKeyToAccount>) => createWalletClient({ chain, transport: http(config.evm.rpcUrl), account: acct })

export async function rpc<T = unknown>(method: string, params: unknown[] = [], url = config.evm.rpcUrl): Promise<T> {
  const res = await fetch(url, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ jsonrpc: '2.0', id: 1, method, params }) })
  const body = (await res.json()) as { result?: T; error?: { message: string } }
  if (body.error) throw new Error(`${method}: ${body.error.message}`)
  return body.result as T
}

/** Moves anvil's clock forward and mines a block so the new timestamp is observable. */
export async function warp(seconds: number): Promise<void> {
  await rpc('evm_increaseTime', [seconds])
  await rpc('evm_mine', [])
}

export const mockFlyAbi = [
  { type: 'function', name: 'mint', stateMutability: 'nonpayable', inputs: [{ name: 'to', type: 'address' }, { name: 'amount', type: 'uint256' }], outputs: [] },
] as const

export const pauseAbi = [
  { type: 'function', name: 'pause', stateMutability: 'nonpayable', inputs: [], outputs: [] },
  { type: 'function', name: 'unpause', stateMutability: 'nonpayable', inputs: [], outputs: [] },
] as const
