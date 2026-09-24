/**
 * Reown AppKit with an EVM (wagmi) and a Solana namespace, both connectable at the same time.
 * Loaded with a dynamic import from the Trading and Vault pages only.
 */
import { createAppKit } from '@reown/appkit'
import { defineChain, solana, solanaDevnet } from '@reown/appkit/networks'
import { SolanaAdapter } from '@reown/appkit-adapter-solana'
import { WagmiAdapter } from '@reown/appkit-adapter-wagmi'
import { getAccount, signMessage, watchAccount, type Config } from '@wagmi/core'
import { getAddress, type Address, type Hex } from 'viem'
import { config } from '../config'

export interface WalletState {
  evm: Address | null
  evmChainId: number | null
  sol: string | null
}

export interface Wallets {
  wagmiConfig: Config
  state(): WalletState
  subscribe(cb: (s: WalletState) => void): () => void
  connect(ns: 'eip155' | 'solana'): Promise<void>
  disconnect(ns: 'eip155' | 'solana'): Promise<void>
  signEvm(text: string): Promise<Hex>
  signSol(message: Uint8Array): Promise<unknown>
}

interface SolanaProvider {
  signMessage(message: Uint8Array): Promise<Uint8Array>
}

let instance: Wallets | null = null

export function initWallets(): Wallets {
  if (instance) return instance
  const projectId = config.reownProjectId
  if (!projectId) throw new Error('Wallet connection is not configured.')

  const evmNetwork = defineChain({
    id: config.evm.chainId,
    caipNetworkId: `eip155:${config.evm.chainId}`,
    chainNamespace: 'eip155',
    name: config.evm.name,
    nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
    rpcUrls: { default: { http: [config.evm.rpcUrl] } },
    blockExplorers: { default: { name: 'Explorer', url: config.evm.explorer } },
    contracts: { multicall3: { address: config.multicall3 } },
    testnet: config.network === 'testnet',
  })
  const solNetwork = config.solana.cluster === 'mainnet' ? solana : solanaDevnet

  const wagmiAdapter = new WagmiAdapter({ projectId, networks: [evmNetwork], ssr: false })
  const modal = createAppKit({
    adapters: [wagmiAdapter, new SolanaAdapter()],
    networks: [evmNetwork, solNetwork],
    defaultNetwork: evmNetwork,
    projectId,
    metadata: {
      name: 'fly-trader',
      description: 'The $FLY vault and trading pages',
      url: location.origin,
      icons: [`${location.origin}/favicon.svg`],
    },
    features: { email: false, socials: false, analytics: false, swaps: false, onramp: false, send: false, history: false },
    customRpcUrls: { [`eip155:${config.evm.chainId}`]: [{ url: config.evm.rpcUrl }] },
    // Wallets that cannot add chain 4663 stay connected: they can still sign claims.
    allowUnsupportedChain: true,
    enableCoinbase: false,
    enableBaseAccount: false,
    themeMode: 'dark',
    themeVariables: {
      '--w3m-accent': '#ff2fb9',
      '--w3m-border-radius-master': '0px',
      '--w3m-font-family': 'Inter, system-ui, sans-serif',
      '--w3m-z-index': 1000,
    },
    termsConditionsUrl: `${location.origin}/terms.html`,
    privacyPolicyUrl: `${location.origin}/privacy.html`,
  })
  const wagmiConfig = wagmiAdapter.wagmiConfig

  const state = (): WalletState => {
    const a = getAccount(wagmiConfig)
    const s = modal.getAccount('solana')
    return {
      evm: a.address ? getAddress(a.address) : null,
      evmChainId: a.chainId ?? null,
      sol: s?.isConnected && s.address ? s.address : null,
    }
  }

  instance = {
    wagmiConfig,
    state,
    subscribe(cb) {
      let last = ''
      const emit = () => {
        const s = state()
        const key = JSON.stringify(s)
        if (key !== last) {
          last = key
          cb(s)
        }
      }
      const offEvm = watchAccount(wagmiConfig, { onChange: emit })
      const offSol = modal.subscribeAccount(emit, 'solana')
      emit()
      return () => {
        offEvm()
        offSol()
      }
    },
    async connect(ns) {
      await modal.open({ view: 'Connect', namespace: ns })
    },
    async disconnect(ns) {
      await modal.disconnect(ns)
    },
    async signEvm(text) {
      const account = getAccount(wagmiConfig).address
      if (!account) throw new Error('Connect an EVM wallet first.')
      return signMessage(wagmiConfig, { account, message: text })
    },
    async signSol(message) {
      const provider = modal.getProvider<SolanaProvider>('solana')
      if (!provider) throw new Error('Connect a Solana wallet first.')
      return provider.signMessage(message)
    },
  }
  return instance
}
