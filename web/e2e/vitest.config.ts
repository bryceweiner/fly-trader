/// <reference types="vitest/config" />
/**
 * Integration tests that drive the site's real page libraries (src/lib/vault.ts, evm.ts, claim.ts, kyber.ts) against
 * running chains: the local dry run from tools/vault_demo.py and an anvil fork of Robinhood Chain mainnet.
 * They are not part of `npm test`; run them with e2e/run.sh.
 */
import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    include: ['e2e/**/*.e2e.test.ts'],
    environment: 'node',
    testTimeout: 240_000,
    hookTimeout: 240_000,
    fileParallelism: false,
    sequence: { concurrent: false },
  },
})
