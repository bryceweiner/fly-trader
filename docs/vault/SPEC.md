# $FLY vault — interface spec v1

The contract between the four parts of the vault system. Change it here first, then in code.

- **FlyVault** (`contracts/`, Robinhood Chain): holds locked $FLY.
- **The vault fly** (`fly_trader/vault/`, the hosted server): trades, keeps the ledger, settles weekly and pays claims.
- **The relay** (`web/relay/`, on the site's Gandi app): passes stats and claims between the fly and browsers.
- **The site** (`web/`): the Trading and Vault pages.

Units, everywhere:
- SOL amounts are integer **lamports**.
- $FLY amounts are **wei as decimal strings**, because they exceed 2^53.
- Timestamps are **unix seconds** (int) in JSON, and RFC 3339 UTC with a trailing `Z` inside signed messages.
- Prices are floats.

## 1. FlyVault (Solidity)

Solidity `0.8.24+` on OpenZeppelin Contracts Upgradeable v5, using UUPS and ERC-7201 namespaced storage
(`fly.storage.FlyVault`). The constructor calls `_disableInitializers()`.

```solidity
uint256 public immutable WITHDRAW_DELAY;   // constructor arg of the implementation: 7 days on mainnet (Deploy.s.sol refuses
                                            // anything else on chain 4663); shorter on the testnet rehearsal. Changing it = an upgrade.
bytes32 public constant PAUSER_ROLE   = keccak256("PAUSER_ROLE");
bytes32 public constant UPGRADER_ROLE = keccak256("UPGRADER_ROLE");

function initialize(address fly, address pauser, address timelock) external initializer;
// DEFAULT_ADMIN_ROLE + UPGRADER_ROLE -> timelock ; PAUSER_ROLE -> pauser

function lock(uint256 amount) external;                          // whenNotPaused; reverts unless received == amount
function requestWithdrawal(uint256 amount) external returns (uint256 id);   // never paused
function cancelRequest(uint256 id) external;                     // whenNotPaused (it re-locks)
function withdraw(uint256 id) external;                          // never paused; block.timestamp >= readyAt
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
```

- **Request ids** start at 1 and are global.
- **`lockedAfter`** is the user's earning balance after the event, so the indexer can resync to an absolute value.
- **Timelock.** Upgrades go through an OZ `TimelockController`:
  - minDelay = 8 days
  - proposer = canceller = the operator's hardware account
  - executor = anyone (`address(0)`)
  - admin = `address(0)`
- **No rescue function for FLY.** The token is set only in `initialize`.

## 2. Claim messages (v1)

A claim is two signatures over two texts, rendered from the same fields. Lines end with `\n` only, with no trailing
newline. Vectors: `tests/vectors/claim_v1.json`. Python (`fly_trader/vault/claim_message.py`) and TypeScript
(`web/src/lib/claim.ts`) must render them byte-identically.

Fields:
- `domain`: e.g. `fly-trader.app`, or `localhost:5173` on staging
- `uri`: e.g. `https://fly-trader.app/vault.html`
- `chain_id`: `4663`, or `46630` on testnet
- `sol_chain`: `mainnet` or `devnet`
- `evm`: EIP-55 checksummed
- `sol`: base58 pubkey
- `nonce`: 32 lowercase hex characters, issued by the relay
- `issued_at`, `expires_at`: RFC 3339 `YYYY-MM-DDTHH:MM:SSZ`; expiry = issue + 900 s
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
- **The texts.** It re-renders both texts from the fields and never uses text supplied by the relay.
- **The EVM signature.**
  - Low-s, and v ∈ {27, 28} (0/1 are normalized).
  - Recovered by ecrecover. If the address has code, it instead calls EIP-1271 `isValidSignature(hashMessage(text), sig) == 0x1626ba7e`.
- **The Solana signature.** ed25519 over the text.
- **The nonce.** Unique per EVM address and not expired.
- **The amount.** Owed ≥ `CLAIM_MIN_LAMPORTS` (2,000,000).

A valid claim pays **all** SOL owed to the EVM address, sent to `sol`.

## 3. Relay HTTP API (v1)

The base is `{origin}/api`. Every response is JSON with `Cache-Control: no-store`. Errors look like `{"error": "<text>"}`
with status 400, 401, 404, 409, 413, 429 or 503.

### Public

- **`GET /api/stats`** returns the latest snapshot (§4), or 503 before the first push.
- **`GET /api/history?kind=nav|trades|flows|settlements&before=<cursor>&limit=<1..500, default 100>`** returns
  `{"kind", "items": [...], "next": <cursor|null>}`, newest first. The cursor is opaque; send it back as `before`.
- **`GET /api/account?evm=0x…`** returns the account (§4.3). An unknown address gets all zeros.
- **`GET /api/claim/challenge?evm=0x…&sol=<base58>`**:
  - Returns `{"nonce", "issued_at", "expires_at", "domain", "uri", "chain_id", "sol_chain", "request_id", "min_lamports"}`.
  - The relay stores the challenge and checks the syntax of `evm` and `sol`.
- **`POST /api/claim`** with body `{"nonce", "evm", "sol", "evm_sig", "sol_sig"}` (≤ 4 KB):
  - Returns `202 {"id", "status": "received"}`.
  - Checks, all cheap and none cryptographic:
    - the nonce is known, unexpired, unused, and issued for this `evm` + `sol`
    - the signatures have valid syntax: `evm_sig` matches `^0x([0-9a-fA-F]{2}){65,1024}$`; `sol_sig` is base58 of 64 bytes
    - the published account's owed ≥ min
    - no claim for this EVM address is in flight
  - 409 if the nonce was already used or a claim is in flight.
- **`GET /api/claim/<id>`** returns `{"id", "status", "reason", "lamports", "tx", "created_at", "updated_at"}`.
  - Status is one of `received | verified | waiting_liquidity | sending | paid | rejected | failed`.
- **Rate limits (token buckets):**
  - challenge: 30 per hour per IP
  - claim: 10 per hour per IP and 5 per hour per EVM address
  - reads: 600 per hour per IP.

### Fly only (HMAC)

- **`POST /api/fly/push`**: upserts.
  - Body: `{"stats": {…§4…}?, "history": {"nav": [...], "trades": [...], "flows": [...], "settlements": [...]}?, "accounts": [ …§4.3… ]?}`.
  - History items upsert by `id` (nav by `ts`); accounts upsert by `evm`.
  - Returns `{"ok": true}`. Body ≤ 2 MB.
- **`GET /api/fly/claims?after=<id>&limit=<1..50>`**:
  - Returns `{"claims": [{"id", "nonce", "evm", "sol", "evm_sig", "sol_sig", "domain", "uri", "chain_id", "sol_chain", "issued_at", "expires_at", "created_at"}]}`.
  - Only claims in status `received`, in id order.
- **`POST /api/fly/claims/result`**: body `{"results": [{"id", "status", "reason"?, "lamports"?, "tx"?}]}` returns `{"ok": true}`.
- **`GET /api/fly/probe`** returns `{"python", "db_path", "writable", "wal", "pid", "remote_addr", "headers": {...}}`.

HMAC headers:
- `X-Fly-Key`: the key id
- `X-Fly-Ts`: unix seconds
- `X-Fly-Nonce`: 32 hex characters
- `X-Fly-Sig`: hex HMAC-SHA256(key = the UTF-8 bytes of the configured secret string, canonical)

The canonical string:
```
FLY-RELAY-1\n{METHOD}\n{PATH}\n{QUERY}\n{TS}\n{NONCE}\n{sha256hex(body)}
```
- `PATH` excludes the query string.
- `QUERY` is `&`.join(`quote(k,safe='')=quote(v,safe='')` for (k, v) in sorted(parse_qsl(qs, keep_blank_values=True))).
- The body is the raw bytes; an empty body hashes as sha256 of `b""`.
- The timestamp must be within ±300 s. Nonces are remembered for 15 minutes, and a reused nonce gets 401.

Relay config: `relay.json`, kept next to `wsgi.py` and never inside `site/`:
```json
{"keys": {"k1": "<64 hex>"}, "db_path": "<writable path>", "domain": "fly-trader.app", "uri": "https://fly-trader.app/vault.html",
 "chain_id": 4663, "sol_chain": "mainnet", "claim_ttl_s": 900, "min_lamports": 2000000, "client_ip_header": null}
```

## 4. Payloads

### 4.1 Stats snapshot (`stats`, pushed every 60 s)
```json
{
 "v": 1, "ts": 0, "book": "live", "cluster": "mainnet",
 "fly": {"state": "starting|paper|live|halted", "wallet": "<base58>", "handover": false, "kill_switch": false,
         "entries_paused": false, "model": {"fly": 62, "selector": 59, "release": 3}},
 "wallet": {"native": 0, "token_accounts": 0, "open_cost": 0, "positions_value": 0, "exit_cost": 0, "nav": 0},
 "ledger": {"deposits": 0, "withdrawals": 0, "claims_paid": 0, "realized": 0, "booked_realized": 0,
            "allocated": 0, "reserved": 0, "pot": 0},
 "prices": {"sol_usd": 0.0, "fly_usd": 0.0, "ts": 0},
 "vault": {"address": "0x…", "chain_id": 4663, "total_locked": "0", "total_pending": "0", "earners": 0,
           "finalized_block": 0, "paused": false, "impl": "0x…", "upgrade_scheduled": null},
 "settlement": {"next_at": 0, "last": null},
 "positions": [{"mint": "", "symbol": "", "opened_at": 0, "cost": 0, "value": 0, "entry_price": 0.0, "mark_price": 0.0, "hold_min": 0}],
 "index": {"value": 1.0, "peak": 1.0, "drawdown": 0.0}
}
```
- `wallet.nav` = native + token_accounts + positions_value − exit_cost.
- `ledger.realized` is R. `pot` = max(0, R − allocated). `reserved` = allocated − claims_paid.
- `settlement.last` is a settlements item (§4.2) or null.
- `vault.upgrade_scheduled` is `{"eta": ts, "id": "0x…"}` or null: the soonest pending timelock operation whose target is the
  vault (upgrade, role change) or the timelock itself (`updateDelay`, proposer changes). `eta` is the block time of
  `CallScheduled` plus its `delay`.

### 4.2 History items
- `nav`: `{"ts", "nav", "index", "sol_usd"}`
- `trades`: `{"id", "mint", "symbol", "opened_at", "closed_at", "cost", "proceeds", "realized", "exit_kind"}`
- `flows`: `{"id", "ts", "signature", "direction": "in|out", "kind": "deposit|profit|withdrawal|claim", "counterparty", "lamports"}`
- `settlements`: `{"id", "period_start", "period_end", "realized", "pot", "allocated", "carried", "earners", "total_weight": "<str>", "status"}`

### 4.3 Account
```json
{"evm": "0x…", "allocated": 0, "claimed": 0, "in_flight": 0, "owed": 0,
 "allocations": [{"period_end": 0, "lamports": 0, "weight": "<str>", "share": 0.0}],
 "claims": [{"id": 0, "status": "", "lamports": 0, "sol": "", "tx": "", "created_at": 0, "updated_at": 0}]}
```
`owed` = allocated − claimed − in_flight, where in_flight counts claims in `verified | waiting_liquidity | sending`.

## 5. Networks

| | mainnet | rehearsal |
|---|---|---|
| EVM chain | Robinhood Chain 4663, `https://rpc.mainnet.chain.robinhood.com` | RH testnet 46630, `https://rpc.testnet.chain.robinhood.com` |
| Explorer | `https://robinhoodchain.blockscout.com` | `https://explorer.testnet.chain.robinhood.com` |
| $FLY | `0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3` | `MockFLY` (deployed for the rehearsal) |
| Solana | mainnet-beta | devnet |
| Kyber | `https://aggregator-api.kyberswap.com/robinhood/api/v1/`, router `0x6131B5fae19EA4f9D964eAc0408E4408b66337b5` | — |
| Chart | GeckoTerminal network `robinhood`, pool `0xe6925f7bdedf22c2714c749d0ff208ce9e4bc1540938fa2b9ab2379486322edd` | fixtures |
