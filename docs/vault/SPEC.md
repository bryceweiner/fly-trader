# $FLY vault — interface spec v1

The contract between the parts of the vault system. Change it here first, then in code.

- **FlyVault** (`contracts/`, Robinhood Chain): holds locked $FLY.
- **The vault fly** (`fly_trader/vault/`, the hosted server): trades, keeps the ledger, settles weekly and pays claims.
- **The signer** (`fly_trader/signer/`, its own container on the server): the only holder of the two hot keys (§6).
- **The relay** (`web/relay/`, on the site's Gandi app): passes stats and claims between the fly and browsers.
- **The site** (`web/`): the Trading, Vault and (unlisted) Owner pages.

Units, everywhere:
- SOL amounts are integer **lamports**.
- $FLY amounts are **wei as decimal strings**, because they exceed 2^53.
- Timestamps are **unix seconds** (int) in JSON, and RFC 3339 UTC with a trailing `Z` inside signed messages.
- Prices are floats.

## 1. FlyVault (Solidity)

Solidity `0.8.24+` on OpenZeppelin Contracts Upgradeable v5, using UUPS and ERC-7201 namespaced storage
(`fly.storage.FlyVault`). The constructor calls `_disableInitializers()`.

```solidity
uint256 public immutable WITHDRAW_DELAY;   // constructor arg of the implementation: 7 days on mainnet, 300 s by default
                                            // on the rehearsal (DeployTestnet.s.sol). Changing it = an upgrade.
bytes32 public constant PAUSER_ROLE   = keccak256("PAUSER_ROLE");
bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

function initialize(address fly, address pauser, address timelock) external initializer;   // reverts on a zero address
// DEFAULT_ADMIN_ROLE + UPGRADER_ROLE -> timelock ; PAUSER_ROLE -> pauser

function lock(uint256 amount) external;                          // whenNotPaused, nonReentrant; reverts unless received == amount
function requestWithdrawal(uint256 amount) external returns (uint256 id);   // never paused
function cancelRequest(uint256 id) external;                     // whenNotPaused (it re-locks)
function withdraw(uint256 id) external;                          // never paused, nonReentrant; block.timestamp >= readyAt
function pause() external;   function unpause() external;        // PAUSER_ROLE

function token() external view returns (address);
function locked(address user) external view returns (uint256);        // earning balance (excludes pending)
function pendingOf(address user) external view returns (uint256);
function totalLocked() external view returns (uint256);
function totalPending() external view returns (uint256);
function request(uint256 id) external view returns (address user, uint256 amount, uint64 readyAt, uint8 state);
  // state: 0 none, 1 pending, 2 cancelled, 3 withdrawn
function requestsOf(address user) external view returns (uint256[] memory ids);

event Locked(address indexed user, uint256 amount, uint256 lockedAfter);
event WithdrawRequested(address indexed user, uint256 indexed id, uint256 amount, uint64 readyAt, uint256 lockedAfter);
event RequestCancelled(address indexed user, uint256 indexed id, uint256 amount, uint256 lockedAfter);
event Withdrawn(address indexed user, uint256 indexed id, uint256 amount);
// plus OZ: Paused(address), Unpaused(address), Upgraded(address indexed implementation), RoleGranted/RoleRevoked

error ZeroAddress(); error ZeroAmount(); error TransferAmountMismatch(uint256 expected, uint256 received);
error InsufficientLocked(uint256 requested, uint256 available); error UnknownRequest(uint256 id);
error NotRequestOwner(uint256 id); error RequestNotPending(uint256 id); error WithdrawalNotReady(uint256 id, uint64 readyAt);
```

- **Request ids** start at 1 and are global.
- **`lockedAfter`** is the user's earning balance after the event, so the indexer can resync to an absolute value.
- **Timelock.** Upgrades and role changes go through an OZ `TimelockController`:
  - minDelay = 8 days on mainnet (600 s by default on the rehearsal)
  - proposer = canceller = the owner (MetaMask until the Ledger takes over; `OWNER` in the deploy)
  - executor = anyone (`address(0)`)
  - admin = `address(0)`: the timelock administers itself, so role changes also wait the full delay
- **Mainnet deploy rules.** On chain 4663 the deploy (`FlyVaultStack.enforceChainRules`) refuses anything but the real
  $FLY, an 8-day timelock and a 7-day withdraw delay. `PAUSER` defaults to `OWNER`.
- **No rescue function for FLY.** The token is set only in `initialize`.
- **The fly indexes** the vault's events and the timelock's `CallScheduled`, `CallExecuted` and `Cancelled` for
  operations aimed at the vault or the timelock (an alert and `vault.upgrade_scheduled`, §4.1).
- **Scripts.**
  - `HandOver.s.sol`: one timelock batch that grants PROPOSER, CANCELLER and PAUSER to a new owner and revokes them
    from the old one. `MODE=schedule|execute`, or `MODE=calldata` to print the calldata for a browser wallet.
  - `ScheduleUpgrade.s.sol` rehearses an upgrade locally and schedules `upgradeToAndCall`; `ExecuteUpgrade.s.sol` runs
    it after the delay.

## 2. Claim messages (v1)

A claim is two signatures over two texts, rendered from the same fields. Lines end with `\n` only, with no trailing
newline. Vectors: `tests/vectors/claim_v1.json`. Python (`fly_trader/vault/claim_message.py`) and TypeScript
(`web/src/lib/claim.ts`) must render them byte-identically.

Fields:
- `domain`: e.g. `fly-trader.app`, or `localhost:5173` on staging
- `uri`: e.g. `https://fly-trader.app/vault.html`, or `http://localhost:5173/vault.html` on staging; https (http only
  for localhost/127.0.0.1), no spaces or line breaks
- `chain_id`: `4663`, or `46630` on testnet
- `sol_chain`: `mainnet` or `devnet`
- `evm`: EIP-55 checksummed
- `sol`: base58 pubkey
- `nonce`: 32 lowercase hex characters, issued by the relay
- `issued_at`, `expires_at`: RFC 3339 `YYYY-MM-DDTHH:MM:SSZ`; expiry = issue + 900 s. The fly accepts no other TTL, so
  the relay's `claim_ttl_s` must stay 900.
- `request_id`: always `fly-vault-claim-v1`

EVM text, signed with `personal_sign` (EIP-191); the signature is `0x…` hex, 65 bytes for an EOA and up to 1024 bytes
for a contract wallet (e.g. a Safe's concatenated owner signatures):
```
{domain} wants you to sign in with your Ethereum account:
{evm}

Pay the SOL the $FLY vault owes this address to Solana wallet {sol}.

URI: {uri}
Version: 1
Chain ID: {chain_id}
Nonce: {nonce}
Issued At: {issued_at}
Expiration Time: {expires_at}
Request ID: fly-vault-claim-v1
```
Solana text, signed with wallet `signMessage` over its UTF-8 bytes; the signature is 64 bytes, **base58**:
```
{domain} wants you to sign in with your Solana account:
{sol}

Receive the SOL the $FLY vault owes Ethereum account {evm}.

URI: {uri}
Version: 1
Chain ID: {sol_chain}
Nonce: {nonce}
Issued At: {issued_at}
Expiration Time: {expires_at}
Request ID: fly-vault-claim-v1
```

What the fly checks:
- **The texts.** Both are re-rendered from the fields; relay-supplied text is never used. `domain`/`uri` must equal
  `VAULT_SITE_DOMAIN`/`VAULT_SITE_URI`, `chain_id` must equal `RH_CHAIN_ID`, and `sol_chain` must match `VAULT_CLUSTER`.
- **Freshness.** A claim is rejected if the fly first stored it more than 3600 s after `expires_at`, or if `issued_at`
  is more than 300 s in the future. The relay already refuses expired nonces at submission.
- **The EVM signature.**
  - ecrecover first: exactly 65 bytes, 0 < r < n, low-s, v ∈ {27, 28} (0/1 are normalized).
  - Only if that does not recover the address and the address has code: EIP-1271 `isValidSignature(hashMessage(text), sig)`
    (65..1024 bytes, 200,000 gas cap) must return exactly `0x1626ba7e`.
- **The Solana signature.** 64 bytes, ed25519 over the UTF-8 text; `sol` must be an on-curve pubkey.
- **The nonce.** Unique per EVM address.
- **One in flight.** A claim is rejected while another claim for the address is `verified | waiting_liquidity |
  sending`.
- **The amount.** Owed ≥ `CLAIM_MIN_LAMPORTS` (2,000,000), read when the claim moves to `sending`; the amount is fixed
  then. The relay's `min_lamports` and the site's `claimMinLamports` must hold the same value.
- **Halts.** Nothing is verified or paid while the vault is halted (claims are still pulled from the relay).

A valid claim pays **all** SOL owed to the EVM address, sent to `sol` from the **treasury** (Squads vault 0) under
spending limit L2 (§6): the payout key signs `spendingLimitUse` and the trading key pays the fee. Nothing is ever paid
from the trading wallet's own SOL. Neither the treasury, the trading wallet nor any payout signature is published.

**Statuses:** `received → verified → (waiting_liquidity) → sending → paid | rejected | failed`.
- `waiting_liquidity`: L2 has too little left this week, the treasury is short, or the signer is in panic. The claim
  retries every loop and alerts the operator; it never fails for these reasons.
- `failed`: four unlanded attempts, a failure on chain, or a signer policy refusal.
- The payment's signature and last valid block height are stored before broadcast; a claim is re-signed only once its
  blockhash has expired unlanded, and the signer keeps its own once-per-claim ledger (§6).

## 3. Relay HTTP API (v1)

The base is `{origin}/api`. Every response is JSON with `Cache-Control: no-store`. Errors look like `{"error": "<text>"}`
with status 400, 401, 404, 405 (with `Allow`), 409, 413, 429 (with `Retry-After`), 500 or 503. 503 means either
`relay not configured` (relay.json missing or invalid) or `storage unavailable` (SQLite error, with `Retry-After: 5`).

### Public

- **`GET /api/stats`** returns the latest snapshot (§4), or 503 before the first push.
- **`GET /api/history?kind=nav|trades|flows|settlements&before=<cursor>&limit=<1..500, default 100>`** returns
  `{"kind", "items": [...], "next": <cursor|null>}`, newest first by (time, id). The time is nav `ts`, trades
  `closed_at`, flows `ts`, settlements `period_end`. The cursor is opaque; send it back as `before`. An unknown `kind`,
  a bad `limit` or a bad cursor gets 400.
- **`GET /api/account?evm=0x…`** returns the account (§4.3). `evm` is accepted in any case. An unknown address gets all
  zeros; the fly pushes only addresses with at least one allocation.
- **`GET /api/claim/challenge?evm=0x…&sol=<base58>`**:
  - Returns `{"nonce", "issued_at", "expires_at", "domain", "uri", "chain_id", "sol_chain", "request_id", "min_lamports"}`.
  - The relay stores the challenge; a bad `evm` or `sol` gets 400 (checked before the rate limit).
- **`POST /api/claim`** with body `{"nonce", "evm", "sol", "evm_sig", "sol_sig"}` (≤ 4 KB):
  - Returns `202 {"id", "status": "received"}`.
  - Checks, all cheap and none cryptographic:
    - the nonce is known, unexpired, unused, and issued for this `evm` + `sol`
    - the signatures have valid syntax: `evm_sig` matches `^0x([0-9a-fA-F]{2}){65,1024}$`; `sol_sig` is base58 of 64 bytes
    - the published account's owed ≥ `min_lamports`
    - no claim for this EVM address that the fly has verified is in flight (`verified | waiting_liquidity |
      sending`). An unverified `received` claim does not block, since it may be a stranger's junk; the fly pays
      each address at most once whatever the relay accepts.
  - 400 for bad syntax, an unknown, expired or mismatched nonce, or nothing to claim; 409 if the nonce was already used
    or a verified claim is in flight; 429 per the limits below.
- **`GET /api/claim/<id>`** (the relay's id, 1-18 digits) returns `{"id", "status", "reason", "lamports", "tx",
  "created_at", "updated_at"}`, or 404.
  - Status is one of `received | verified | waiting_liquidity | sending | paid | rejected | failed`.
  - `tx` is always null: the fly never reports the payout signature.
- **Rate limits (token buckets; a 429 carries `Retry-After`):**
  - challenge: 30 per hour per IP
  - claim: 10 per hour per IP, charged after the syntax checks; and 5 per hour per EVM address and IP, charged only
    after every other check passes (so junk from one client cannot lock a holder out)
  - reads (stats, history, account, claim/<id>): 600 per hour per IP
  - IPv6 clients are bucketed per /64. Behind a proxy, `client_ip_header` names the header to read, and the relay takes
    its rightmost entry (the one the proxy appended). The fly's HMAC routes are not rate-limited.

### Fly only (HMAC)

- **`POST /api/fly/push`**: upserts.
  - Body: `{"stats": {…§4…}?, "history": {"nav": [...], "trades": [...], "flows": [...], "settlements": [...]}?, "accounts": [ …§4.3… ]?}`.
  - Unknown top-level keys get 400. History items upsert by `id` (nav by `ts`); each needs an integer time field (nav
    `ts`; trades `closed_at` or `opened_at`; settlements `period_end` or `period_start`; flows `ts`) and, except nav, an
    `id` (an integer ≥ 0 or a string ≤ 200 characters). Accounts upsert by `evm` and need an integer `owed`. `stats` is
    stored as is; NaN and Infinity become null.
  - Returns `{"ok": true}`. Body ≤ 2 MB.
- **`GET /api/fly/claims?after=<id, default 0>&limit=<1..50, default 50>`**:
  - Returns `{"claims": [{"id", "nonce", "evm", "sol", "evm_sig", "sol_sig", "domain", "uri", "chain_id", "sol_chain", "issued_at", "expires_at", "created_at"}]}`.
  - Only claims in status `received`, in id order. `evm` is lowercase; `created_at` is unix seconds.
- **`POST /api/fly/claims/result`**: body `{"results": [{"id", "status", "reason"?, "lamports"?, "tx"?}]}` returns
  `{"ok": true}`. `status` must be one of the seven statuses; `reason` is cut to 500 characters; omitted fields keep
  their value; unknown ids are ignored.
- **`GET /api/fly/probe`** returns `{"python", "db_path", "writable", "wal", "pid", "remote_addr", "headers": {...},
  "sqlite", "time", "client_ip", "multiprocess", "multithread"}` (headers: every `HTTP_*` except Cookie).

HMAC headers:
- `X-Fly-Key`: the key id
- `X-Fly-Ts`: unix seconds (1-15 digits)
- `X-Fly-Nonce`: 32 hex characters
- `X-Fly-Sig`: hex HMAC-SHA256(key = the UTF-8 bytes of the configured secret string, canonical)

The canonical string:
```
FLY-RELAY-1\n{METHOD}\n{PATH}\n{QUERY}\n{TS}\n{NONCE}\n{sha256hex(body)}
```
- `PATH` starts at `/api/…` and excludes the query string.
- `QUERY` is `&`.join(`quote(k,safe='')=quote(v,safe='')` for (k, v) in sorted(parse_qsl(qs, keep_blank_values=True))).
- The body is the raw bytes; an empty body hashes as sha256 of `b""`.
- The timestamp must be within ±300 s. Nonces are remembered for 15 minutes (case-insensitively), and a reused nonce
  gets 401. The body is verified before the nonce is recorded.

Relay config: `relay.json`, kept next to `wsgi.py` and never inside `site/`, or wherever `FLY_RELAY_CONFIG` points. It
is re-read when it changes; a relative `db_path` is resolved against its folder. Secrets must be at least 32 characters
(64 hex recommended).
```json
{"keys": {"k1": "<64 hex>"}, "db_path": "<writable path>", "domain": "fly-trader.app", "uri": "https://fly-trader.app/vault.html",
 "chain_id": 4663, "sol_chain": "mainnet", "claim_ttl_s": 900, "min_lamports": 2000000, "client_ip_header": null}
```

## 4. Payloads

### 4.1 Stats snapshot (`stats`, pushed every vault-worker loop, about every 10 s)
```json
{
 "v": 1, "ts": 0, "book": "live", "cluster": "mainnet",
 "fly": {"state": "starting|paper|live|halted", "handover": false, "kill_switch": false,
         "entries_paused": false, "model": {"fly": 62, "selector": 59, "release": 3}},
 "wallet": {"native": 0, "nav": 0},
 "ledger": {"deposits": 0, "withdrawals": 0, "claims_paid": 0, "realized": 0, "booked_realized": 0,
            "allocated": 0, "reserved": 0, "pot": 0},
 "prices": {"sol_usd": 0.0, "fly_usd": 0.0, "ts": 0},
 "vault": {"address": "0x…", "chain_id": 4663, "total_locked": "0", "total_pending": "0", "earners": 0,
           "finalized_block": 0, "paused": false, "impl": "0x…", "upgrade_scheduled": null},
 "settlement": {"last": null}
}
```
- **Nothing that helps trade ahead of the fly.** `ts` is rounded down to the hour; `wallet.native` and `wallet.nav`
  come from the last 5-minute mark and are rounded to 0.1 SOL (NAV history points likewise), so one buy cannot be
  matched to an on-chain trade; `ledger.realized` is R at that same mark. Open positions, the performance index and
  the next settlement time are not published; a trade appears in `trades` history once it closes. No address of the
  fly's (trading wallet, treasury, keys) is published, nor anything that leads to one: no transaction signatures and no
  deposit senders.
- `book` is `VAULT_BOOK` (`live`, or a paper book such as `paper_fly` on a dry run). `cluster` is `mainnet` or
  `VAULT_CLUSTER` as is.
- `fly.state`: `halted` if the vault is halted or the kill switch is on; `live` after the handover once the live wallet
  reports; `paper` once the fly reports status; else `starting`. `fly.model` = `{fly, selector, release}`, each may be
  null.
- `wallet.native` = trading wallet + treasury lamports; `wallet.nav` = native + open positions' value − exit cost.
- `ledger`: `realized` is R (§7); `pot` = max(0, R − allocated); `reserved` = max(0, allocated − claims_paid);
  `booked_realized` is the book's summed closed-trade profit; `deposits`, `withdrawals` (fees included) and
  `claims_paid` are running totals. On a paper book `realized` is closed-trade profit since the vault started and
  `deposits` is `CAPITAL_SOL`.
- `prices`: `sol_usd` from Jupiter Price v3, `fly_usd` from the GeckoTerminal pool, cached 60 s (a failed fetch keeps
  the last value); `ts` is the older of the two fetch times, or 0.
- `vault`: `total_locked`, `total_pending`, `earners` and `paused` come from events in finalized blocks (about 18 minutes
  behind the head), up to `finalized_block`. `address` is null until `VAULT_ADDRESS` is set.
- `settlement.last` is the newest `allocated` settlement (§4.2), or null.
- `vault.upgrade_scheduled` is `{"eta": ts, "id": "0x…"}` or null: the soonest pending timelock operation whose target is the
  vault (upgrade, role change) or the timelock itself (`updateDelay`, proposer changes). `eta` is the block time of
  `CallScheduled` plus its `delay`.

### 4.2 History items
- `nav`: `{"ts", "nav", "sol_usd"}`: one point per 5-minute boundary (`ts`), from the newest consistent mark at or before
  it, nav rounded to 0.1 SOL; `sol_usd` is the price at push time.
- `trades`: `{"id", "mint", "symbol", "opened_at", "closed_at", "cost", "proceeds", "realized", "exit_kind"}`: closed
  positions of `VAULT_BOOK` only; `proceeds` = cost + realized; `exit_kind` is the forced-exit kind or `hold`.
- `flows`: `{"id", "ts", "direction": "in|out", "kind": "deposit|profit|withdrawal|claim", "lamports"}`. Internal moves
  (`sweep`, `topup`) and `unknown_outbound`/`anomaly` rows are never published.
- `settlements`: `{"id", "period_start", "period_end", "realized", "pot", "allocated", "carried", "earners",
  "total_weight": "<str>", "status"}`: `status` is always `allocated`; `carried` = pot − allocated; `total_weight` is
  the sum of wei·seconds; `realized` is R at the snapshot.

### 4.3 Account
```json
{"evm": "0x…", "allocated": 0, "claimed": 0, "in_flight": 0, "owed": 0,
 "allocations": [{"period_end": 0, "lamports": 0, "weight": "<str>", "share": 0.0}],
 "claims": [{"id": 0, "status": "", "lamports": 0, "sol": "", "tx": "", "created_at": 0, "updated_at": 0}]}
```
- `owed` = allocated − claimed − in_flight. `in_flight` counts claims from `sending` on (a `verified` claim has no amount
  yet).
- `allocations`: the 52 most recent; `share` = weight / that settlement's total weight.
- `claims`: the 20 most recent; `id` is the relay's claim id (as in `GET /api/claim/<id>`); `tx` is always `""`.

## 5. Networks

| | mainnet | rehearsal |
|---|---|---|
| Site build | `VITE_NETWORK=mainnet` (default) | `VITE_NETWORK=testnet` |
| EVM chain | Robinhood Chain 4663, `https://rpc.mainnet.chain.robinhood.com` | RH testnet 46630, `https://rpc.testnet.chain.robinhood.com` |
| Explorer | `https://robinhoodchain.blockscout.com` | `https://explorer.testnet.chain.robinhood.com` |
| $FLY | `0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3` | `MockFLY` (deployed for the rehearsal) |
| USDG | `0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168` (6 decimals) | — |
| Multicall3 | `0xcA11bde05977b3631167028862bE2a173976CA11` | same |
| Solana | mainnet-beta | devnet |
| Squads v4 | `SQDS4ep65T869zMMBKyuUq6aD6EgTu8psMjkvj52pCf` | same |
| Settlement period | 7 days (`VAULT_PERIOD_S=604800`) | 900 s |
| Kyber | `https://aggregator-api.kyberswap.com/robinhood/api/v1/`, router `0x6131B5fae19EA4f9D964eAc0408E4408b66337b5` | — |
| Chart | GeckoTerminal network `robinhood`, pool `0xe6925f7bdedf22c2714c749d0ff208ce9e4bc1540938fa2b9ab2379486322edd` | fixtures |

- The fly's defaults are mainnet. A rehearsal sets `RH_CHAIN_ID=46630`, `RH_RPC_URL`, `VAULT_CLUSTER=devnet`,
  `VAULT_SOLANA_RPC_URL`, `VAULT_PERIOD_S=900` and `VAULT_SITE_DOMAIN`/`VAULT_SITE_URI`.
- The site's Solana RPC (the Owner page) is WalletConnect's unless `VITE_SOLANA_RPC` is set.

## 6. Custody (Solana)

The fly's SOL is one book in two accounts:
- **The treasury** is vault 0 of a Squads v4 multisig. Its members are the owner's wallets only (Solflare until the
  Ledger takes over); threshold 1 and no time lock while there is one member.
- **The trading wallet** is a hot key that keeps a float of about `TREASURY_FLOAT_SOL` (default 2 SOL).

| Spending limit | Member | Destinations | Period | Used for |
|---|---|---|---|---|
| L1 | trading key | the trading wallet only | Day | refilling the float (`topup` flows, memo `fly-topup`) |
| L2 | payout key | any | Week | claims (`spendingLimitUse` from the treasury, memo `fly-vault-claim:<fly claim id>:<nonce>`) |

- **Rebalancing** (every minute, `vault/payout.py`): first the treasury must hold everything owed plus its 890,880-lamport
  rent minimum, sweeping from the trading wallet if it is short; then the trading wallet is brought to the float, by a
  top-up under L1 when low or a sweep when high. Moves under `TOPUP_MIN_SOL` (0.1) are skipped, except a sweep the
  treasury needs. A sweep never takes the trading wallet below `GAS_RESERVE_SOL` + fee. Under `/panic` the float target
  is 0.
- **Flows.** A trading → treasury transfer is a `sweep`, a treasury → trading one a `topup`: both internal, excluded
  from R, the index and published history. SOL or wSOL arriving at either account from `FUNDING_ADDRESSES` is a
  `deposit`; from anyone else it is `profit` (a gift). A treasury outflow signed by a multisig member is a `withdrawal`
  (the owner's principal; alerted, louder when the destination is not a funding address). A transaction one of our
  keys signed that no table knows is `unknown_outbound`; lamports leaving without our or an owner's signature are an
  `anomaly`. Both halt the vault (§7), unless the second RPC provider sees the transaction differently.
- **Accounting.** R, NAV and the performance index count both accounts together. Positions are sized on the bankroll
  = trading free SOL + open positions at cost + (treasury − owed); buys spend only the trading wallet's own SOL, less
  any owed SOL the treasury does not hold.
- **The signer** (`fly_trader/signer`) is the only holder of the two hot keys. It builds and signs:
  - Jupiter swaps: SOL ↔ one token only; Metis routes only (`excludeRouters=jupiterz,dflow,okx`); static rules on every
    instruction (programs, the trading wallet's roles, a priority-fee cap of `SIGNER_MAX_PRIORITY_SOL`, 0.005); a
    simulation, through its own RPC, of the trading wallet's SOL and every token account it owns (a buy may spend at
    most the ordered SOL plus fees and must deliver the token to the trading wallet; a sell may spend at most the
    ordered tokens and must return SOL; no other holding may shrink); a value floor of `SIGNER_MIN_VALUE_RATIO` (0.6)
    at the signer's own Jupiter prices. Buys are capped per trade (`SIGNER_MAX_BUY_SOL`, default `MAX_POSITION_SOL`) and
    per day (`SIGNER_DAILY_BUY_SOL`, 10); all swaps per minute (`SIGNER_SWAPS_PER_MIN`, 6).
  - top-ups under L1 (at most `TREASURY_FLOAT_SOL` each; L1's destinations must be exactly the trading wallet) and
    sweeps (a plain transfer to the treasury)
  - claims under L2, once per claim id, with the same destination and amount; re-signed only after the earlier
    signature failed or its blockhash expired
  - closes of empty token accounts of the trading wallet (batches of 10)
  - the hot keys, age-encrypted to `VAULT_BACKUP_RECIPIENT`, for the off-box backup
  - after `panic`: no buys, top-ups or claims; `clear-panic` is a CLI command inside its container only

  On the server the brain reaches it over a Unix socket (one JSON request per line, 256 KB max, one at a time) and never
  sees a key. The signer keeps its own SQLite ledger. Without `SIGNER_SOCKET` (self-hosting, tests) the same signer runs
  inside the brain's process.
- **Custody checks** (every minute, `vault/custody.py`), alerted: neither server key is a multisig member; no config
  authority; L1 is a SOL limit with member = the trading key and sole destination = the trading wallet; L2 is a SOL
  limit with member = the payout key. Any change to members, threshold, time lock or the limits is reported.
- **The owner** changes the limits from `owner.html` (`web/src/lib/squads.ts`). Every new limit has a fresh address:
  the server's `VAULT_LIMIT_TRADING` / `VAULT_LIMIT_PAYOUT` must be updated to it (the page prints the lines). A
  limit's amount is changed by removing it and adding a new one, in two config transactions: the program rejects both
  in one.

## 7. Settlement, halts, kill switch

- **Schedule.** Period ends fall at `VAULT_EPOCH + k·VAULT_PERIOD_S` (345600 = Monday 1970-01-05 00:00 UTC; 7 days).
  The first settlement covers the period before the first end after the vault started; missed periods are settled as
  one window. Settlements advance one step per minute, in period order.
- **Two phases.**
  1. Snapshot at the period end T1: both accounts at a finalized slot that contains every transaction the fly sent,
     read under the wallet lock. The second RPC provider (`SOLANA_CHECK_RPC_URL`) must report the same balance, or the
     settlement pauses, alerts and retries. Flows are scanned up to that slot.
  2. Allocate once Robinhood Chain is finalized past T1, and only if the vault's implementation code hash is in
     `VAULT_IMPL_CODEHASHES` (else the vault halts).

  Statuses: `snapshotted | allocated | failed`.
- **Formulas.**
  ```
  R   = N + K + C − D + Wd + P       N: both accounts' SOL; K: wSOL and empty token-account rent; C: open positions at
                                     cost; D: deposits; Wd: withdrawals incl. fees; P: claims paid
  pot = max(0, R − A)                A: everything ever allocated
  out = min(pot, N − (A − P) − GAS_RESERVE_SOL)
  weight_i = ∫ lockedAfter_i dt over [T0, T1)   (wei·s, from FlyVault events)
  alloc_i  = floor(out · weight_i / Σ weight); the remainder, dust and empty weeks carry forward (A grows only by alloc)
  ```
- **Halts.** The vault halts on `unknown_outbound`, `anomaly`, an unknown implementation code hash, or Telegram
  `/panic` (the operator's user only). A halt stops claim processing, settlement and new entries. `/panic` also trips
  the kill switch with liquidation, sets the float target to 0 and puts the signer in panic. Resume is SSH-only:
  `fly-trader vault resume` (clears halt and panic), `fly-trader reset-circuit --kill`, `fly-trader resume-entries`,
  and the signer's `clear-panic`.
- **Kill switch.** A flow-adjusted index `I_t = I_{t−1}·(W_t − F_t)/W_{t−1}` over consistent NAV marks of both
  accounts, where F is every deposit/profit in and withdrawal/claim out since the previous mark. Its peak uses marks at
  least 600 s old, over 30 days. It trips at `KILL_SWITCH_DRAWDOWN` (default 0.30; the vault server sets 0.20) and, with
  `KILL_SWITCH_LIQUIDATE`, sells every open position.
- **Principal withdrawal.** Only the owner, from the Squads app. The cap `fly-trader vault withdraw` prints is
  min(D − Wd + min(0, R − A), treasury − owed − 890,880).
