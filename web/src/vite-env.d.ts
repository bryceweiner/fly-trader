/// <reference types="vite/client" />

interface ImportMetaEnv {
  readonly VITE_NETWORK?: 'mainnet' | 'testnet'
  readonly VITE_REOWN_PROJECT_ID?: string
  readonly VITE_VAULT_ADDRESS?: string
  readonly VITE_TIMELOCK_ADDRESS?: string
  readonly VITE_FLY_ADDRESS?: string
  readonly VITE_RELAY_BASE?: string
  readonly VITE_RELAY_PROXY?: string
  readonly VITE_EVM_RPC?: string
  readonly VITE_SOLANA_RPC?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
