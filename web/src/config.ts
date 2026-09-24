/** Build-time network configuration. Everything here comes from docs/vault/SPEC.md §5 unless overridden by env. */
import type { Address } from 'viem'

export type NetworkName = 'mainnet' | 'testnet'

const env = import.meta.env
export const NETWORK: NetworkName = env.VITE_NETWORK === 'testnet' ? 'testnet' : 'mainnet'

/** The real $FLY on Robinhood Chain mainnet; the index page always shows this one. */
export const FLY_MAINNET = '0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3' as Address

/** Env addresses are lowercased so a mistyped checksum cannot make viem reject them; pages display them EIP-55.
 *  (No viem import here: this module is on every page, including the index.) */
function addr(v: string | undefined): Address | null {
  const s = (v || '').trim()
  return /^0x[0-9a-fA-F]{40}$/.test(s) ? (s.toLowerCase() as Address) : null
}

interface EvmNetwork {
  chainId: number
  name: string
  rpcUrl: string
  explorer: string
}

const EVM: Record<NetworkName, EvmNetwork> = {
  mainnet: {
    chainId: 4663,
    name: 'Robinhood Chain',
    rpcUrl: 'https://rpc.mainnet.chain.robinhood.com',
    explorer: 'https://robinhoodchain.blockscout.com',
  },
  testnet: {
    chainId: 46630,
    name: 'Robinhood Chain Testnet',
    rpcUrl: 'https://rpc.testnet.chain.robinhood.com',
    explorer: 'https://explorer.testnet.chain.robinhood.com',
  },
}

const mainnet = NETWORK === 'mainnet'

/** VITE_EVM_RPC points the build at another node, e.g. a local anvil for the dry run (the chain id must match). */
const evm: EvmNetwork = { ...EVM[NETWORK], rpcUrl: (env.VITE_EVM_RPC || '').trim() || EVM[NETWORK].rpcUrl }

export const config = {
  network: NETWORK,
  evm,
  solana: {
    cluster: (mainnet ? 'mainnet' : 'devnet') as 'mainnet' | 'devnet',
    /** appended to Solscan links */
    solscanSuffix: mainnet ? '' : '?cluster=devnet',
  },
  fly: {
    address: mainnet ? addr(env.VITE_FLY_ADDRESS) ?? FLY_MAINNET : addr(env.VITE_FLY_ADDRESS),
    decimals: 18,
    symbol: mainnet ? 'FLY' : 'MockFLY',
  },
  usdg: mainnet ? { address: '0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168' as Address, decimals: 6, symbol: 'USDG' } : null,
  vault: addr(env.VITE_VAULT_ADDRESS),
  timelock: addr(env.VITE_TIMELOCK_ADDRESS),
  multicall3: '0xcA11bde05977b3631167028862bE2a173976CA11' as Address,
  kyber: {
    enabled: mainnet,
    api: 'https://aggregator-api.kyberswap.com/robinhood/api/v1',
    router: '0x6131B5fae19EA4f9D964eAc0408E4408b66337b5' as Address,
    clientId: 'fly-trader',
  },
  market: {
    /** testnet builds show bundled fixture candles (SPEC §5: "Chart: fixtures") */
    live: mainnet,
    geckoApi: 'https://api.geckoterminal.com/api/v2',
    geckoNetwork: 'robinhood',
    pool: '0xe6925f7bdedf22c2714c749d0ff208ce9e4bc1540938fa2b9ab2379486322edd',
  },
  relayBase: (env.VITE_RELAY_BASE || '/api').replace(/\/+$/, ''),
  reownProjectId: (env.VITE_REOWN_PROJECT_ID || '').trim(),
  claimMinLamports: 2_000_000,
  withdrawDelayS: 7 * 86400,
} as const

export const LINKS = {
  github: 'https://github.com/bryceweiner/fly-trader',
  x: 'https://x.com/i/communities/2038854012578173030',
  venue: 'https://ponsfamily.com/coin/', // the CA is appended
  flyCa: FLY_MAINNET,
}

export const explorer = {
  evmAddress: (a: string) => `${config.evm.explorer}/address/${a}`,
  evmTx: (h: string) => `${config.evm.explorer}/tx/${h}`,
  solAccount: (a: string) => `https://solscan.io/account/${a}${config.solana.solscanSuffix}`,
  solTx: (s: string) => `https://solscan.io/tx/${s}${config.solana.solscanSuffix}`,
  solToken: (m: string) => `https://solscan.io/token/${m}${config.solana.solscanSuffix}`,
}
