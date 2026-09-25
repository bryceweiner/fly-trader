# FlyVault contracts

Holders lock $FLY in `FlyVault` on Robinhood Chain. Off-chain, the vault fly pays locked holders a weekly share of
profit, weighted over time by each holder's earning balance, which it rebuilds from the vault's events. Leaving
takes two steps. `requestWithdrawal` stops the tokens earning at once, and `withdraw` works `WITHDRAW_DELAY`
(7 days) later. `cancelRequest` locks them again. The interface is frozen in `../docs/vault/SPEC.md` §1.

| | |
|---|---|
| `src/FlyVault.sol` | UUPS implementation. ERC-7201 storage in `fly.storage.FlyVault`. Roles: DEFAULT_ADMIN + UPGRADER go to the timelock, PAUSER to the pauser. |
| `src/testnet/MockFLY.sol` | 18-decimal "Mock FLY"/"mFLY" with a public faucet `mint(to, amount)`, capped at 1M per call. |
| `script/FlyVaultStack.sol` | The one code path that deploys and checks the stack. The scripts and tests all use it. |
| `script/Deploy.s.sol` | Mainnet or any chain. Deploys TimelockController, the implementation, and an ERC1967Proxy that is initialized in its constructor. |
| `script/DeployTestnet.s.sol` | Deploys MockFLY plus the same stack with short delays. Runs only on chain 46630 or anvil. |
| `script/ScheduleUpgrade.s.sol`, `script/ExecuteUpgrade.s.sol` | Upgrades through the timelock. |
| `abi/FlyVault.json`, `abi/TimelockController.json` | ABIs for the indexer and the site. |
| `storage-layout.json`, `storage-layout.namespaced.json` | Storage layouts to diff before upgrades (see below). |
| `deployments/<chainid>.json` | Address summary written by the deploy scripts. |

The pauser can block only the calls that start or resume earning (`lock`, `cancelRequest`). `requestWithdrawal`
and `withdraw` never pause. The contract has no function to rescue or sweep the token, no receipt token, and
positions cannot be transferred.

## Build and test

Requires Foundry (tested with forge 1.5.1). Dependencies are pinned in `soldeer.lock` and never committed:

```sh
cd contracts
forge soldeer install          # forge-std 1.16.2, @openzeppelin-contracts(-upgradeable) 5.7.0
forge build
forge test -vv                 # unit, upgrade, invariant, red-team, script tests; fork tests are skipped
FORK=1 forge test --match-contract ForkTest -vv   # forks RH mainnet and uses the real $FLY (RH_RPC_URL overrides the RPC)
forge snapshot                 # refreshes .gas-snapshot
```

`test/RedTeam.t.sol` holds the adversarial cases: reentrancy through the unguarded functions, false-returning and
no-return tokens, donations, forged or replayed request ids, a permanent pause, a malicious implementation behind
the timelock (holders who request within a day of the schedule are out before it executes), a smuggled
re-`initialize`, an `UPGRADER_ROLE` grant to an EOA (public for 8 days, then no delay: treat it like an upgrade), a
timelock admin trying to skip the delay, and a check that no plain storage slot is ever written.
`test/RedTeamExtra.t.sol` (2026-09-25) adds ETH sends, allowance and request-id hijacking, request spam (another holder's
gas is unchanged), a pauser who pauses and then loses its key, batched `updateDelay(0)` + upgrade, predecessors,
cancelled operations, cross-vault ids, extreme amounts, and the other side of the exit window: someone who locks more
than `TIMELOCK_DELAY - WITHDRAW_DELAY` after a malicious upgrade was scheduled cannot leave in time (the Vault page
warns above the Lock form). `test/InvariantExit.t.sol` fuzzes with upgrades mid-run and, after every run, makes every
actor leave (paused or not) and checks each gets back exactly what they locked. `test/Symbolic.t.sol` holds Halmos
proofs: `halmos --contract SymbolicVaultTest` (run `forge clean` first if forge built without ASTs). `forge lint` and
Slither (`slither . --filter-paths "dependencies|test|script"`) report nothing beyond informational items.

Compiler settings: solc 0.8.36, optimizer 10,000 runs, `evm_version = "shanghai"`. Cancun opcodes are not
verified on this chain, so the contracts use the storage-based `ReentrancyGuard` and never the transient one.

## Deploy (mainnet, chain 4663)

The owner is the hardware wallet. It becomes the timelock's proposer and canceller, and by default also the
vault's pauser. Set `PAUSER` to use a different account. The script refuses to run on chain 4663 unless
`FLY_TOKEN` is `0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3`, `TIMELOCK_DELAY` is 8 days and `WITHDRAW_DELAY` is
7 days. Leave the two delays unset, since the defaults are already 8 and 7 days.

```sh
export FLY_TOKEN=0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3
export OWNER=0x...your-ledger-address
# 1. dry run (simulates against the live chain; writes deployments/4663.dry-run.json, sends nothing)
forge script script/Deploy.s.sol --rpc-url robinhood --sender $OWNER
# 2. real deploy, signed on the Ledger (3 transactions), verified on Blockscout
forge script script/Deploy.s.sol --rpc-url robinhood --ledger --sender $OWNER \
  --mnemonic-indexes 0 --broadcast \
  --verify --verifier blockscout --verifier-url https://robinhoodchain.blockscout.com/api/
```

After deploying:
- The script checks every role, both delays, the token and the proxy's implementation slot. The deployer ends up
  with no role on either contract.
- It writes `deployments/4663.json` with `vault` (the proxy, the address everyone uses), `implementation`,
  `timelock`, `token`, `owner`, `pauser`, both delays, and `blockNumber`/`timestamp`. Those two come from the
  simulated block, so they are a safe lower bound for where the indexer should start.
- The JSON is written before forge sends anything. If the broadcast fails, delete the file. The receipts in
  `broadcast/Deploy.s.sol/4663/run-latest.json` are authoritative.
- Commit `deployments/4663.json`.

### Verify on Blockscout manually

Use this if `--verify` fails. Plain `curl` to `robinhoodchain.blockscout.com/api` currently gets a Cloudflare 403,
so forge may hit it too.

```sh
V=https://robinhoodchain.blockscout.com/api/     # testnet: https://explorer.testnet.chain.robinhood.com/api/
forge verify-contract <implementation> src/FlyVault.sol:FlyVault --chain 4663 \
  --verifier blockscout --verifier-url $V --constructor-args $(cast abi-encode "f(uint256)" 604800) --watch
forge verify-contract <timelock> dependencies/@openzeppelin-contracts-5.7.0/governance/TimelockController.sol:TimelockController \
  --chain 4663 --verifier blockscout --verifier-url $V --watch \
  --constructor-args $(cast abi-encode "f(uint256,address[],address[],address)" 691200 "[$OWNER]" "[0x0000000000000000000000000000000000000000]" 0x0000000000000000000000000000000000000000)
forge verify-contract <vault> dependencies/@openzeppelin-contracts-5.7.0/proxy/ERC1967/ERC1967Proxy.sol:ERC1967Proxy \
  --chain 4663 --verifier blockscout --verifier-url $V --watch \
  --constructor-args $(cast abi-encode "f(address,bytes)" <implementation> $(cast calldata "initialize(address,address,address)" $FLY_TOKEN <pauser> <timelock>))
```

If the API stays blocked, run `forge verify-contract ... --show-standard-json-input > input.json` and upload that
file in the explorer's "Verify & publish" page, choosing "Standard JSON input" and compiler 0.8.36.

## Testnet rehearsal (chain 46630)

```sh
export OWNER=0x...   # optional; defaults to the broadcasting account
forge script script/DeployTestnet.s.sol --rpc-url robinhood_testnet --ledger --sender $OWNER --broadcast \
  --verify --verifier blockscout --verifier-url https://explorer.testnet.chain.robinhood.com/api/
# optional env: TIMELOCK_DELAY (default 600 s), WITHDRAW_DELAY (default 300 s), PAUSER
cast send <mFLY> "mint(address,uint256)" <you> 1000000ether --rpc-url robinhood_testnet --ledger   # faucet
```

## Upgrade runbook

An upgrade deploys a new implementation and calls `vault.upgradeToAndCall(newImpl, data)` through the timelock:
the owner schedules, 8 days pass, then anyone executes. The owner can `cancel(id)` on the timelock at any point
before execution.

1. **Diff storage first.** Solc's storage-layout output does not see ERC-7201 namespaces. For this vault,
   `storage-layout.json` must stay empty: any entry means someone added a plain state variable. The fields of
   the namespaced struct come from a probe contract:
   ```sh
   forge inspect FlyVault storageLayout --json > /tmp/new.json && diff storage-layout.json /tmp/new.json
   forge inspect StorageLayoutProbe storageLayout --json > /tmp/ns.json && diff storage-layout.namespaced.json /tmp/ns.json
   ```
   The rules:
   - Existing `FlyVaultStorage` fields keep their order, types and slots.
   - New fields go only at the end of the struct, or in a new namespace.
   - The `Request` struct only grows at the end.
   - Keep the namespace id and `FLY_VAULT_STORAGE_LOCATION` unchanged.

   Commit the regenerated files with the upgrade.
2. **Edit** `src/FlyVault.sol`. Any one-time migration goes in a `reinitializer(2)` function. Then run the full
   test suite and the fork test.
3. **Schedule**, sending as the owner. The script deploys the implementation and schedules the upgrade with the
   timelock's minimum delay. Before anything is sent, it rehearses locally: it warps past the delay, executes,
   checks that the state survived, and rolls back. A broken upgrade therefore never starts the 8-day clock.
   ```sh
   forge script script/ScheduleUpgrade.s.sol --rpc-url robinhood --ledger --sender $OWNER --mnemonic-indexes 0 --broadcast \
     --verify --verifier blockscout --verifier-url https://robinhoodchain.blockscout.com/api/
   ```
   Optional env:
   - `VAULT`, `TIMELOCK`: default to `deployments/<chainid>.json`.
   - `WITHDRAW_DELAY`: defaults to the live value. Changing it is exactly this kind of upgrade. Existing requests
     keep their `readyAt`.
   - `UPGRADE_CALL`: hex calldata for the reinitializer. Defaults to empty.
   - `SALT`, `PREDECESSOR`: bytes32, both default to 0.

   The script prints the operation id and the exact env to execute with. Salt and predecessor rules:
   - The operation id is `hashOperation(vault, 0, upgradeToAndCall(newImpl, UPGRADE_CALL), PREDECESSOR, SALT)`.
   - Every upgrade deploys a new implementation address, so the id is unique even with `SALT=0`.
   - A new `SALT` is needed only to re-propose a call byte-identical to one already executed. A cancelled
     operation can be scheduled again with the same salt.
   - `PREDECESSOR` forces this operation to wait until another operation id has executed. Leave it at 0 unless
     you are chaining operations, e.g. a role change that must land first.
4. **Wait** 8 days. Watch for `CallScheduled`/`Cancelled` on the timelock. The site shows `vault.upgrade_scheduled`.
5. **Execute**, sent from any account, with the same values the schedule used:
   ```sh
   NEW_IMPLEMENTATION=0x... SALT=0x00...00 forge script script/ExecuteUpgrade.s.sol --rpc-url robinhood \
     --ledger --sender $OWNER --broadcast
   ```
   It refuses early and when the operation is unknown or already done. Afterwards it confirms the proxy's
   implementation slot.

Other timelock-governed actions work the same way via `cast` (`schedule` → wait 8 days → `execute`). Examples:
rotating the pauser (`vault.grantRole/revokeRole(PAUSER_ROLE, …)`) and changing the delay
(`timelock.updateDelay`, which itself waits the current delay).

## Pause

```sh
cast send <vault> "pause()" --rpc-url robinhood --ledger     # PAUSER_ROLE; stops lock + cancelRequest only
cast send <vault> "unpause()" --rpc-url robinhood --ledger
```

## Gas (from `forge test --gas-report`, median)

| call | gas |
|---|---|
| `lock` | ~110k (first lock by a user; later ones are cheaper) |
| `requestWithdrawal` | ~169k (writes a new request record and appends to `requestsOf`) |
| `cancelRequest` | ~33k |
| `withdraw` | ~60k |

FlyVault's runtime bytecode is 9,981 bytes; the ERC1967Proxy's is 176 bytes.
