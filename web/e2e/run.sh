#!/usr/bin/env bash
# Drives the site's contract and relay interactions against real chains.
#
#   e2e/run.sh vault     needs the local dry run:  .venv/bin/python tools/vault_demo.py up   (anvil :8545, relay :8612)
#   e2e/run.sh trading   needs an anvil fork of mainnet: anvil --fork-url https://rpc.mainnet.chain.robinhood.com --port 8546
#                        (FORK_RPC overrides the URL) and reaches the real KyberSwap API for quotes.
set -euo pipefail
cd "$(dirname "$0")/.."
DEP=../.vault-demo/deployment.json
case "${1:-vault}" in
  vault)
    [ -f "$DEP" ] || { echo "no $DEP: start the dry run first (tools/vault_demo.py up)"; exit 1; }
    # the relay allows 10 claim posts per hour per IP; back-to-back runs from one machine would trip that
    command -v sqlite3 >/dev/null && sqlite3 ../.vault-demo/relay/relay.sqlite3 "DELETE FROM buckets;" 2>/dev/null || true
    VITE_NETWORK=testnet VITE_EVM_RPC=http://127.0.0.1:8545 VITE_RELAY_BASE=http://127.0.0.1:8612/api \
      VITE_VAULT_ADDRESS="$(node -p "require('$DEP').vault")" VITE_TIMELOCK_ADDRESS="$(node -p "require('$DEP').timelock")" \
      VITE_FLY_ADDRESS="$(node -p "require('$DEP').fly")" \
      npx vitest run --config e2e/vitest.config.ts --reporter=verbose e2e/vault.e2e.test.ts
    ;;
  trading)
    VITE_NETWORK=mainnet VITE_EVM_RPC="${FORK_RPC:-http://127.0.0.1:8546}" \
      npx vitest run --config e2e/vitest.config.ts --reporter=verbose e2e/trading.e2e.test.ts
    ;;
  *)
    echo "usage: e2e/run.sh vault|trading"; exit 2
    ;;
esac
