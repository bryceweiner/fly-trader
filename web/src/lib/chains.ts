/** Robinhood Chain definitions for viem. Multicall3 has code at the canonical address on both chains
 *  (checked with eth_getCode on 2026-09-23). */
import { createPublicClient, defineChain, http, type Chain } from 'viem'
import { config } from '../config'

const multicall3 = { address: '0xcA11bde05977b3631167028862bE2a173976CA11' } as const
const eth = { name: 'Ether', symbol: 'ETH', decimals: 18 } as const

export const robinhood = defineChain({
  id: 4663,
  name: 'Robinhood Chain',
  nativeCurrency: eth,
  rpcUrls: { default: { http: ['https://rpc.mainnet.chain.robinhood.com'] } },
  blockExplorers: { default: { name: 'Blockscout', url: 'https://robinhoodchain.blockscout.com' } },
  contracts: { multicall3 },
})

export const robinhoodTestnet = defineChain({
  id: 46630,
  name: 'Robinhood Chain Testnet',
  nativeCurrency: eth,
  rpcUrls: { default: { http: ['https://rpc.testnet.chain.robinhood.com'] } },
  blockExplorers: { default: { name: 'Explorer', url: 'https://explorer.testnet.chain.robinhood.com' } },
  contracts: { multicall3 },
  testnet: true,
})

export const evmChain: Chain = config.network === 'mainnet' ? robinhood : robinhoodTestnet

let client: ReturnType<typeof createPublicClient> | null = null

/** Read-only client; works without a wallet. */
export function publicClient() {
  client ??= createPublicClient({ chain: evmChain, transport: http(config.evm.rpcUrl, { batch: true }) })
  return client
}
