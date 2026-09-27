#!/usr/bin/env bash
# The treasury custody end to end against the REAL Squads v4 program, on a throwaway local validator:
#   1. dump the program and its config account from mainnet (read-only)
#   2. start solana-test-validator on 11xxx ports (it never touches the vault demo's validator on 8899)
#   3. web/src/lib/squads.int.test.ts: the owner page's builders create the treasury, L1 and L2, change L2, revoke
#   4. tests/test_custody_e2e.py: the signer refills through L1, pays a claim through L2 up to the cap, and a stolen
#      key cannot redirect L1 or use the other key's limit (refused on chain)
# Usage: tools/custody_e2e.sh <mainnet RPC URL, e.g. your Helius URL>
set -euo pipefail
RPC_MAIN="${1:?usage: tools/custody_e2e.sh <mainnet RPC URL>}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
W="$(mktemp -d "${TMPDIR:-/tmp}/custody-e2e.XXXX")"
SQUADS=SQDS4ep65T869zMMBKyuUq6aD6EgTu8psMjkvj52pCf
CONFIG=BSTq9w3kZwNwpBXJEvTZz2G9ZTNyKBvoSeXMvwb4cNZr      # the program's config PDA (seeds "multisig", "program_config")
RPC=http://127.0.0.1:11899
trap 'kill "${V:-0}" 2>/dev/null || true; rm -rf "$W"' EXIT

solana program dump -u "$RPC_MAIN" "$SQUADS" "$W/squads.so" >/dev/null
solana account -u "$RPC_MAIN" "$CONFIG" --output json -o "$W/config.json" >/dev/null
# the RPC's websocket is rpc+1 (11900): keep the faucet and the dynamic range off it and off the demo's 8000-10000
solana-test-validator --ledger "$W/l" --reset --quiet --rpc-port 11899 --faucet-port 11950 --gossip-port 11100 \
  --dynamic-port-range 11000-11090 --bpf-program "$SQUADS" "$W/squads.so" --account "$CONFIG" "$W/config.json" &
V=$!
until curl -s -m 3 -X POST -H 'content-type: application/json' -d '{"jsonrpc":"2.0","id":1,"method":"getSlot"}' "$RPC" \
      | grep -qE '"result":([2-9][0-9]|[0-9]{3,})'; do sleep 2; done

cd "$REPO/web"
SQUADS_IT_RPC="$RPC" npx vitest run src/lib/squads.int.test.ts
SQUADS_IT_RPC="$RPC" SQUADS_IT_OUT="$W/chain.json" npx vitest run src/lib/squads.int.test.ts
cd "$REPO"
CUSTODY_E2E="$W/chain.json" uv run pytest -q tests/test_custody_e2e.py
echo "custody end to end: all passed"
