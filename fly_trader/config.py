"""Typed configuration from the environment (.env is loaded once, never overriding real env vars).

Every knob lives here. Secret values are never printed; ``summary()`` returns the non-secret
configuration for persistence in ``runs.config``. All time handling is UTC.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(REPO_ROOT / ".env", override=False)

SECRET_ENV_NAMES = ("HELIUS_API_KEY", "JUPITER_API_KEY", "BOT_PRIVATE_KEY", "KALSHI_API_KEY_ID", "KALSHI_PRIVATE_KEY_PATH",
                    "RELAY_SECRET", "TELEGRAM_BOT_TOKEN", "VAULT_BACKUP_HF_TOKEN", "VAULT_SOLANA_RPC_URL", "SIGNER_RPC_URL",
                    "SOLANA_CHECK_RPC_URL", "HEALTHCHECK_URL", "RH_BOT_PRIVATE_KEY", "RH_RPC_URL_LOGS", "ENVIO_API_TOKEN", "QUICKNODE_API_KEY")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def env_str(name: str, default: str | None = None) -> str | None:
    v = os.environ.get(name)
    if v is None or v.strip() == "":
        return default
    return v.strip()


def env_int(name: str, default: int) -> int:
    v = env_str(name)
    return int(v) if v is not None else default


def env_float(name: str, default: float) -> float:
    v = env_str(name)
    return float(v) if v is not None else default


def env_bool(name: str, default: bool = False) -> bool:
    v = env_str(name)
    if v is None:
        return default
    return v.lower() in ("1", "true", "yes", "on")


def env_list(name: str, default: list[str]) -> list[str]:
    v = env_str(name)
    if v is None:
        return list(default)
    return [x.strip() for x in v.split(",") if x.strip()]

# ---- secrets (names only ever appear in logs) ----
HELIUS_API_KEY = env_str("HELIUS_API_KEY")
JUPITER_API_KEY = env_str("JUPITER_API_KEY")

# ---- paths ----
DATA_DIR = (REPO_ROOT / env_str("DATA_DIR", "data")).resolve()
LOG_DIR = (REPO_ROOT / env_str("LOG_DIR", "logs")).resolve()
BRAIN_DIR = DATA_DIR / "brain"
PG_ARCHIVE_DIR = DATA_DIR / "pg_archive"

# ---- database ----
DATABASE_URL = env_str("DATABASE_URL", "postgresql:///fly_trader")

# ---- chain / execution (default-deny) ----
SOLANA_CLUSTER = env_str("SOLANA_CLUSTER", "mainnet-beta")
LIVE_ENABLED = env_bool("LIVE_ENABLED", False)
CAPITAL_SOL = env_float("CAPITAL_SOL", 0.0)
MAX_POSITION_SOL = env_float("MAX_POSITION_SOL", 0.0)
GAS_RESERVE_SOL = env_float("GAS_RESERVE_SOL", 0.30)
# position sizing (agent/sizing.py): a fraction of the growth-optimal (Kelly) bet measured per score band in the backtest
KELLY_FRACTION = env_float("KELLY_FRACTION", 0.25)                 # quarter Kelly: the band estimates are noisy
MAX_POSITION_FRACTION = env_float("MAX_POSITION_FRACTION", 0.10)   # of the deployable bankroll (wealth minus the gas reserve)
MAX_POOL_SHARE = env_float("MAX_POOL_SHARE", 0.02)                 # of the pool's quote reserve (bounds price impact)
# The training label charges what an exit really costs at the size the book takes. market/features.py prices
# exit_cost_0p1 for a fixed 0.1 SOL, but impact = value/(value + reserves) grows with size. Labels are priced at
# min(LABEL_SIZE_SOL, MAX_POOL_SHARE of the pool) -- the "label size" -- and no position is ever bigger (agent/sizing.py):
# every edge the models show was measured at that size. A fixed reference, never a share of the bankroll: derived from
# CAPITAL_SOL it made the labels, and the sizes, grow with the bankroll, and in the 2026-09-26 replay a 500 SOL book
# sized past it lost 426 SOL over 60 days where one capped at it made the same +13 SOL as a 5 or 50 SOL book.
LABEL_SIZE_SOL = env_float("LABEL_SIZE_SOL", 0.5)
MIN_POSITION_SOL = env_float("MIN_POSITION_SOL", 0.02)             # smaller sizes are skipped
KILL_SWITCH_DRAWDOWN = env_float("KILL_SWITCH_DRAWDOWN", 0.30)
KILL_SWITCH_LIQUIDATE = env_bool("KILL_SWITCH_LIQUIDATE", False)   # a tripped kill switch also sells every open live position (agent/fly_live.py)
CIRCUIT_THRESHOLD = env_int("CIRCUIT_THRESHOLD", 3)
EXECUTABILITY_MAX_IMPACT = env_float("EXECUTABILITY_MAX_IMPACT", 0.30)
SLIPPAGE_ENTRY_BPS = env_int("SLIPPAGE_ENTRY_BPS", 150)
SLIPPAGE_EXIT_BPS = env_int("SLIPPAGE_EXIT_BPS", 300)
SLIPPAGE_FORCED_BPS = env_int("SLIPPAGE_FORCED_BPS", 500)
SLIPPAGE_STEP_BPS = env_int("SLIPPAGE_STEP_BPS", 100)
MAX_SLIPPAGE_ENTRY_BPS = env_int("MAX_SLIPPAGE_ENTRY_BPS", 300)

HELIUS_HTTP = "https://mainnet.helius-rpc.com/"
JUPITER_SWAP_BASE = "https://api.jup.ag/swap/v2"
JUPITER_TOKENS_BASE = "https://api.jup.ag/tokens/v2"
JUPITER_RPS = env_float("JUPITER_RPS", 8.0)

WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
LAMPORTS_PER_SOL = 1_000_000_000

# ---- token stats (Jupiter) ----
STATS_REFRESH_S = env_float("STATS_REFRESH_S", 600.0)

# ---- connectome (the fly) ----
# How the fly picks its buy line (train/fly_calibrate.py): total | selector | match. match = its teacher's selectivity:
# in the 2026-09-26 replays it took 92 trades at +10.1 % (58 % winners, PF 1.92) where total took 1,607 at +2.0 %.
FLY_CALIB_RULE = env_str("FLY_CALIB_RULE", "match")
# How the fly sizes a buy (agent/sizing.py): flat = MAX_POSITION_FRACTION of the deployable bankroll on every buy | kelly
# = its calibrated bands. A selective fly resolves too few trades a week for bands: they flipped between Kelly 0.97, 0 and
# 0.15, and flat sizing beat them in 98 % of resampled replays.
FLY_SIZING = env_str("FLY_SIZING", "flat")
DEVICE = env_str("DEVICE", "auto")          # auto (CUDA, else MPS, else CPU) | cpu | cuda | cuda:N | mps  — brain/device.py
NT_SIGN_MODE = env_str("NT_SIGN_MODE", "shiu")  # shiu | flybrain
CELL_TYPES_SOURCE = env_str("CELL_TYPES_SOURCE", "annotations")  # annotations | flybrain

# ---- costs ----
LAMBDA_IMPACT_DLMM = env_float("LAMBDA_IMPACT_DLMM", 0.02)

# ---- training lifecycle ----
RESET_ON_START = env_bool("RESET_ON_START", True)  # wipe the paper book and its stats when the runner starts

# ---- the fly's handover (agent/fly_session.py): when its paper race earns it the live seat ----
HANDOVER_DAYS = env_int("HANDOVER_DAYS", 14)                        # days of racing first; 0 = as soon as the trades are in
HANDOVER_MIN_TRADES = env_int("HANDOVER_MIN_TRADES", 30)            # closed paper_fly trades required
HANDOVER_BEAT_SELECTOR = env_bool("HANDOVER_BEAT_SELECTOR", True)   # and realized P&L at least the paper selector's over the window


def helius_http_url() -> str:
    if not HELIUS_API_KEY:
        raise RuntimeError("HELIUS_API_KEY is not set")
    return f"{HELIUS_HTTP}?api-key={HELIUS_API_KEY}"


def summary() -> dict:
    """Non-secret configuration snapshot for ``runs.config``."""
    out = {}
    for k, v in globals().items():
        if k.isupper() and k not in SECRET_ENV_NAMES and not k.startswith("_"):
            if isinstance(v, (str, int, float, bool, list, tuple)):
                out[k] = v
            elif isinstance(v, Path):
                out[k] = str(v)
            elif isinstance(v, dict):
                out[k] = v
    if "DATABASE_URL" in out:                      # may carry a password (URL or key=value form); the database name is enough
        try:
            from psycopg.conninfo import conninfo_to_dict
            db = conninfo_to_dict(str(out["DATABASE_URL"])).get("dbname") or "?"
        except Exception:
            db = "?"
        out["DATABASE_URL"] = f"postgresql:///{db}"
    return out


def live_prerequisites_missing() -> list[str]:
    missing = []
    if SOLANA_CLUSTER != "mainnet-beta":
        missing.append("SOLANA_CLUSTER must be mainnet-beta for live memecoin trading")
    if not LIVE_ENABLED:
        missing.append("LIVE_ENABLED=1")
    if CAPITAL_SOL <= 0:
        missing.append("CAPITAL_SOL")
    if MAX_POSITION_SOL <= 0:
        missing.append("MAX_POSITION_SOL")
    if GAS_RESERVE_SOL <= 0:
        missing.append("GAS_RESERVE_SOL")
    if not SIGNER_SOCKET and not env_str("BOT_PRIVATE_KEY") and not env_str("BOT_PRIVATE_KEY_FILE"):
        missing.append("BOT_PRIVATE_KEY")                          # the vault server's brain has no key: its signer does
    if not JUPITER_API_KEY:
        missing.append("JUPITER_API_KEY")
    if not HELIUS_API_KEY:
        missing.append("HELIUS_API_KEY")
    return missing

# ---- the $FLY vault (fly_trader/vault, docs/vault/SPEC.md): only the hosted vault fly sets VAULT_ENABLED ----
VAULT_ENABLED = env_bool("VAULT_ENABLED", False)
PAYOUT_PRIVATE_KEY_FILE = env_str("PAYOUT_PRIVATE_KEY_FILE")      # the key that pays claims from the treasury (spending limit L2); read by the signer only
VAULT_BOOK = env_str("VAULT_BOOK", "live")                         # the book the vault shares: live; a paper book (e.g. paper_fly) for a dry run on real decisions without real SOL
VAULT_CLUSTER = env_str("VAULT_CLUSTER", "mainnet-beta")           # devnet only for the rehearsal (trading off, claims on devnet)
VAULT_SOLANA_RPC_URL = env_str("VAULT_SOLANA_RPC_URL")             # overrides Helius for the vault's reads/claims (the devnet rehearsal)
FUNDING_ADDRESSES = env_list("FUNDING_ADDRESSES", [])              # SOL from these is a deposit; from anyone else, profit
RH_CHAIN_ID = env_int("RH_CHAIN_ID", 4663)
RH_RPC_URL = env_str("RH_RPC_URL", "https://rpc.mainnet.chain.robinhood.com")
VAULT_ADDRESS = env_str("VAULT_ADDRESS")                            # the FlyVault proxy
VAULT_TIMELOCK = env_str("VAULT_TIMELOCK")                          # its TimelockController (upgrade announcements)
VAULT_START_BLOCK = env_int("VAULT_START_BLOCK", 0)                 # the proxy's deployment block: the indexer starts there
VAULT_IMPL_CODEHASHES = env_list("VAULT_IMPL_CODEHASHES", [])       # implementation code hashes a settlement accepts
VAULT_PERIOD_S = env_int("VAULT_PERIOD_S", 7 * 86400)               # one settlement period; the rehearsal uses 900
VAULT_EPOCH = env_int("VAULT_EPOCH", 345_600)                       # boundaries at EPOCH + k*PERIOD; 345600 = Monday 1970-01-05 00:00 UTC
CLAIM_MIN_LAMPORTS = env_int("CLAIM_MIN_LAMPORTS", 2_000_000)       # 0.002 SOL: above the rent-exempt minimum of a new wallet
CLAIM_MIN_WEI = env_int("CLAIM_MIN_WEI", 100_000_000_000_000)          # 0.0001 ETH: smaller ETH owed waits for the next claim
VAULT_RH_GAS_RESERVE_WEI = env_int("VAULT_RH_GAS_RESERVE_WEI", 200_000_000_000_000)   # ETH the settlement never allocates (the RH wallet pays payout gas)
RELAY_URL = env_str("RELAY_URL")                                    # the site origin; the fly calls {RELAY_URL}/api/...
RELAY_KEY_ID = env_str("RELAY_KEY_ID", "k1")
RELAY_SECRET = env_str("RELAY_SECRET")
VAULT_SITE_DOMAIN = env_str("VAULT_SITE_DOMAIN", "fly-trader.app")  # the domain and URI claim texts must carry
VAULT_SITE_URI = env_str("VAULT_SITE_URI", "https://fly-trader.app/vault.html")
FLY_TOKEN_ADDRESS = env_str("FLY_TOKEN_ADDRESS", "0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3")
FLY_POOL = env_str("FLY_POOL", "0xe6925f7bdedf22c2714c749d0ff208ce9e4bc1540938fa2b9ab2379486322edd")
GECKO_NETWORK = env_str("GECKO_NETWORK", "robinhood")
TELEGRAM_BOT_TOKEN = env_str("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = env_str("TELEGRAM_CHAT_ID")
VAULT_BACKUP_RECIPIENT = env_str("VAULT_BACKUP_RECIPIENT")          # age / ssh-ed25519 public key the backups are encrypted to
VAULT_BACKUP_BUCKET = env_str("VAULT_BACKUP_BUCKET")                # private HF storage bucket, e.g. bryceweiner/fly-vault-backups
VAULT_BACKUP_HF_TOKEN = env_str("VAULT_BACKUP_HF_TOKEN")            # scoped to that bucket only
# custody (docs/vault/SPEC.md "Custody"): a Squads v4 treasury the hot keys draw on only through its spending limits
VAULT_MULTISIG = env_str("VAULT_MULTISIG")                          # the Squads multisig; the treasury is its vault 0
VAULT_LIMIT_TRADING = env_str("VAULT_LIMIT_TRADING")                # L1: trading key -> the trading wallet only, per day
VAULT_LIMIT_PAYOUT = env_str("VAULT_LIMIT_PAYOUT")                  # L2: payout key -> any holder, per week
TREASURY_FLOAT_SOL = env_float("TREASURY_FLOAT_SOL", 2.0)           # the most the trading wallet keeps idle; the rest goes back
TOPUP_MIN_SOL = env_float("TOPUP_MIN_SOL", 0.1)                     # smaller refills and returns are skipped
SIGNER_SOCKET = env_str("SIGNER_SOCKET")                            # the signer container's socket; unset: an in-process signer
SIGNER_RPC_URL = env_str("SIGNER_RPC_URL")                          # the signer's own RPC (a second provider)
SOLANA_CHECK_RPC_URL = env_str("SOLANA_CHECK_RPC_URL")              # a second provider money-moving reads must agree with
TELEGRAM_ADMIN_USER_ID = env_str("TELEGRAM_ADMIN_USER_ID")          # the only Telegram user whose /panic and /status count
HEALTHCHECK_URL = env_str("HEALTHCHECK_URL")                        # dead-man ping (healthchecks.io) every 5 min


def vault_solana_rpc_url() -> str:
    return VAULT_SOLANA_RPC_URL or helius_http_url()


def signer_env() -> dict:
    """What an in-process signer reads (fly_trader/signer; the signer container has its own environment)."""
    return {k: str(v) for k, v in {"VAULT_MULTISIG": VAULT_MULTISIG, "VAULT_LIMIT_TRADING": VAULT_LIMIT_TRADING,
                                   "VAULT_LIMIT_PAYOUT": VAULT_LIMIT_PAYOUT,
                                   "TREASURY_FLOAT_SOL": TREASURY_FLOAT_SOL, "VAULT_BACKUP_RECIPIENT": VAULT_BACKUP_RECIPIENT}.items() if v}

# ---- Robinhood Chain memecoins (fly_trader/rh; Pons launches graduated into Uniswap v4; branch rh-memecoins) ----
# The same selector and fly as Solana, trained on both chains' rows (a chain input tells them apart); RH books keep their
# money in native ETH. Amounts on RH rows are ETH (non-ETH-quoted pools converted at the minute's base/ETH price).
RH_ENABLED = env_bool("RH_ENABLED", True)                           # index RH and paper-trade it
RH_LIVE_ENABLED = env_bool("RH_LIVE_ENABLED", False)                # sign and send real RH transactions (live_rh)
RH_TESTNET = env_bool("RH_TESTNET", False)                          # chain 46630 (the rehearsal); the guard refuses a mismatch
RH_EXPECTED_CHAIN_ID = 46630 if RH_TESTNET else RH_CHAIN_ID
RH_RPC_URL_LOGS = env_str("RH_RPC_URL_LOGS")                        # optional better endpoint for eth_getLogs / tx lookups (else RH_RPC_URL)
# block headers for timing the index (rh/blocktime.py): PublicNode serves header batches fast but refuses eth_getLogs (403),
# the official RPC serves logs but rate-limits at ~20 requests/min (measured 2026-09-28) — so each does what it does well
RH_RPC_URL_BLOCKS = env_str("RH_RPC_URL_BLOCKS", "https://robinhood-rpc.publicnode.com")
RH_HYPERSYNC_URL = env_str("RH_HYPERSYNC_URL", "https://robinhood.hypersync.xyz")   # log history in bulk when ENVIO_API_TOKEN is set (rh/hypersync.py)
RH_HYPERSYNC_RPM = env_float("RH_HYPERSYNC_RPM", 40)                 # HyperSync queries per minute (free tier: fair use, refused ~6 % at 60; Starter plan: 100)
RH_BOT_ADDRESS = env_str("RH_BOT_ADDRESS")                          # the address RH_BOT_PRIVATE_KEY must derive to (a guard, not a secret)
RH_CONFIRMATIONS = env_int("RH_CONFIRMATIONS", 3)                   # blocks before a live receipt or a live log is applied
RH_START_BLOCK = env_int("RH_START_BLOCK", 0)                       # the indexer's first block (0: the Pons V2 factory's deployment block)
RH_START_DAY = env_str("RH_START_DAY", "2026-08-04")                # Pons V2 went live: the first RH corpus day
# Fixed at build time by tools/rh_constants.py (2026-09-28 13:12 UTC: SOL/USD 119.51, ETH/USD 2682.05): the Solana sizes in ETH.
# Never derived live: they set the size every RH label is priced at (see LABEL_SIZE_SOL).
RH_ETH_PER_SOL = env_float("RH_ETH_PER_SOL", 0.0445583)                   # K: converts fixed-unit thresholds (eligibility, triggers) for RH rows
RH_CAPITAL_ETH = env_float("RH_CAPITAL_ETH", 0.222791)                   # the RH paper books' bankroll (5 SOL at build time)
RH_LABEL_SIZE_ETH = env_float("RH_LABEL_SIZE_ETH", 0.0222791)             # 0.5 SOL at build time: RH labels are priced at it; no RH buy is bigger
RH_MIN_POSITION_ETH = env_float("RH_MIN_POSITION_ETH", 0.000891166)         # 0.02 SOL at build time
RH_GAS_RESERVE_ETH = env_float("RH_GAS_RESERVE_ETH", 0.0029)        # 200 round trips at 2x the gas rh-probe measured (356k gas, 0.02 gwei, 2026-09-28)
RH_MAX_FEE_GWEI = env_float("RH_MAX_FEE_GWEI", 5.0)                  # a transaction never offers more per gas
RH_TX_FEE_ETH = env_float("RH_TX_FEE_ETH", 7.2e-6)                 # one swap's gas in ETH (rh-probe 2026-09-28: 356k gas × 0.02 gwei); fixed like the SOL fee
RH_HOOK_FEE = env_float("RH_HOOK_FEE", 0.0099)                      # Pons hook fee + creator tax per side when a minute has none: the median of 52k indexed swaps (1 % of the net = 0.99 % of gross; high-tax launches reach 5.7 %)
RH_KILL_SWITCH_DRAWDOWN = env_float("RH_KILL_SWITCH_DRAWDOWN", 0.30)
RH_MIN_SWEEP_ETH = env_float("RH_MIN_SWEEP_ETH", 0.0002)           # base-asset or dead-bag balances worth less are dust
RH_STREAM_POLL_S = env_float("RH_STREAM_POLL_S", 2.0)
RH_TRADE_LAG_S = env_float("RH_TRADE_LAG_S", 240.0)                 # an RH minute this old (s) is still traded: its stream lands ~2–3 min late
RH_MINUTES_KEEP_DAYS = env_int("RH_MINUTES_KEEP_DAYS", 14)
RH_DIR_NAME = "rh"                                                  # data/corpus/rh/... (the RH corpus beside the Solana one)
# contracts on Robinhood Chain (mainnet; verified to carry code 2026-09-28)
PONS_FACTORY = env_str("PONS_FACTORY", "0x7ed598bcef8bd9edd8c97a195c6d13f40801ec7e")
PONS_ROUTER = env_str("PONS_ROUTER", "0xe33e9e479df8802cb0866d5d05258bec4cf62948")
PONS_HOOK = env_str("PONS_HOOK", "0xe5e702641ea86f4ae6cc3cdaed2b886f976be044")
PONS_LOCKER = env_str("PONS_LOCKER", "0x267444d099b10fb5ed7c3cc7b7c767adca574952")
PONS_GRAD_EXECUTOR = env_str("PONS_GRAD_EXECUTOR", "0xc7819b64a1daecd7ec19856d026cb14efbd89046")
V4_POOL_MANAGER = env_str("V4_POOL_MANAGER", "0x8366a39cc670b4001a1121b8f6a443a643e40951")
V4_QUOTER = env_str("V4_QUOTER", "0x8dc178efb8111bb0973dd9d722ebeff267c98f94")
V4_STATE_VIEW = env_str("V4_STATE_VIEW", "0xf3334192d15450cdd385c8b70e03f9a6bd9e673b")
UNIVERSAL_ROUTER = env_str("UNIVERSAL_ROUTER", "0x8876789976decbfcbbbe364623c63652db8c0904")
PERMIT2 = env_str("PERMIT2", "0x000000000022D473030F116dDEE9F6B43aC78BA3")
V3_FACTORY = env_str("V3_FACTORY", "0x1f7d7550b1b028f7571e69a784071f0205fd2efa")
V3_QUOTER_V2 = env_str("V3_QUOTER_V2", "0x33e885ed0ec9bf04ecfb19341582aadcb4c8a9e7")
RH_WETH = env_str("RH_WETH", "0x0bd7d308f8e1639fab988df18a8011f41eacad73")      # SwapRouter02.WETH9()
RH_USDG = env_str("RH_USDG", "0x5fc5360D0400a0Fd4f2af552ADD042D716F1d168")
KYBER_API = env_str("KYBER_API", "https://aggregator-api.kyberswap.com/robinhood/api/v1")
KYBER_ROUTER = env_str("KYBER_ROUTER", "0x6131B5fae19EA4f9D964eAc0408E4408b66337b5")  # MetaAggregationRouterV2: the only router a built Kyber tx may call
KYBER_CLIENT_ID = env_str("KYBER_CLIENT_ID", "fly-trader")


def rh_live_prerequisites_missing() -> list[str]:
    missing = []
    if not RH_LIVE_ENABLED:
        missing.append("RH_LIVE_ENABLED=1")
    if not env_str("RH_BOT_PRIVATE_KEY"):
        missing.append("RH_BOT_PRIVATE_KEY")
    if not RH_BOT_ADDRESS:
        missing.append("RH_BOT_ADDRESS")
    for name in ("RH_CAPITAL_ETH", "RH_LABEL_SIZE_ETH", "RH_MIN_POSITION_ETH", "RH_ETH_PER_SOL", "RH_GAS_RESERVE_ETH"):
        if globals()[name] <= 0:
            missing.append(name)
    return missing


# ---- Kalshi prediction markets (fly_trader/kalshi; the same key names as better_bot, whose client is vendored) ----
KALSHI_API_KEY_ID = env_str("KALSHI_API_KEY_ID")
KALSHI_PRIVATE_KEY_PATH = env_str("KALSHI_PRIVATE_KEY_PATH")
KALSHI_API_BASE_URL = env_str("KALSHI_API_BASE_URL", "https://api.elections.kalshi.com/trade-api/v2")
KALSHI_WS_URL = env_str("KALSHI_WS_URL", "wss://api.elections.kalshi.com/trade-api/ws/v2")
KALSHI_SUBACCOUNT = env_int("KALSHI_SUBACCOUNT", 0)                # 0 = the primary account: only a dedicated subaccount may trade live
KALSHI_FEE_FACTOR = env_float("KALSHI_FEE_FACTOR", 0.07)            # taker fee per contract = factor x multiplier x P(1-P)
KALSHI_MAKER_FEE_FACTOR = env_float("KALSHI_MAKER_FEE_FACTOR", 0.0175)
KALSHI_CAPITAL_USD = env_float("KALSHI_CAPITAL_USD", 100.0)         # the cap both Kalshi arms share (paper start; live never sizes above it)
KALSHI_CASH_FLOOR_USD = env_float("KALSHI_CASH_FLOOR_USD", 5.0)     # the gas reserve's analogue: never spent
KALSHI_MIN_PAYOUT_CENTS = env_int("KALSHI_MIN_PAYOUT_CENTS", 25)    # a fill whose winning profit is below this is skipped
KALSHI_LIVE_ENABLED = env_bool("KALSHI_LIVE_ENABLED", False)
KALSHI_TAKER_LIVE = env_bool("KALSHI_TAKER_LIVE", True)
KALSHI_MAKER_LIVE = env_bool("KALSHI_MAKER_LIVE", True)
KALSHI_KILL_SWITCH_DRAWDOWN = env_float("KALSHI_KILL_SWITCH_DRAWDOWN", 0.30)
KALSHI_MAX_SLATE_FRACTION = env_float("KALSHI_MAX_SLATE_FRACTION", 0.50)   # of the cap deployed at once across every open position
KALSHI_MAX_DAYS_TO_CLOSE = env_float("KALSHI_MAX_DAYS_TO_CLOSE", 10.0)     # the evidence base (Whelan 2026) covers the last 10 days
KALSHI_MIN_MINUTES_TO_CLOSE = env_float("KALSHI_MIN_MINUTES_TO_CLOSE", 5.0)
KALSHI_MAX_SPREAD_CENTS = env_int("KALSHI_MAX_SPREAD_CENTS", 10)
KALSHI_MIN_OPEN_INTEREST = env_float("KALSHI_MIN_OPEN_INTEREST", 100.0)
KALSHI_MIN_VOLUME_24H = env_float("KALSHI_MIN_VOLUME_24H", 50.0)
KALSHI_MAKER_QUIET_MIN = env_float("KALSHI_MAKER_QUIET_MIN", 30.0)         # no resting orders this close to the market's close
KALSHI_MAKER_TTL_H = env_float("KALSHI_MAKER_TTL_H", 6.0)
KALSHI_MAKER_MAX_RESTING = env_int("KALSHI_MAKER_MAX_RESTING", 40)
KALSHI_HISTORY_START = env_str("KALSHI_HISTORY_START", "2025-01-01")
KALSHI_DATASET_DIR = env_str("KALSHI_DATASET_DIR")                        # jon-becker/prediction-market-analysis checkout (data/kalshi/{markets,trades})
KALSHI_MINUTES_KEEP_DAYS = env_int("KALSHI_MINUTES_KEEP_DAYS", 7)
KALSHI_MARKETS_PER_DAY = env_int("KALSHI_MARKETS_PER_DAY", 150)           # the event sample's target: about this many markets per category per day (kalshi/history.sample_rates);
                                                                           # events are kept by md5(event ticker) < the category's rate, blind to outcome and volume
KALSHI_DIR = DATA_DIR / "kalshi"
KALSHI_TRAIN_MAX_ROWS = env_int("KALSHI_TRAIN_MAX_ROWS", 8_000_000)        # training rows per walk-forward block / final fit (uniform random subset)
KALSHI_FILL_THREADS = env_int("KALSHI_FILL_THREADS", 8)                   # markets filled at once by the history worker, all under the one KALSHI_RPS bucket
KALSHI_RPS = env_float("KALSHI_RPS", 16.0)                                 # per process; the account's Advanced grant reads 200 tokens/s at 10 per call = 20 calls/s, the feed uses ~3


def kalshi_live_prerequisites_missing() -> list[str]:
    missing = []
    if not KALSHI_LIVE_ENABLED:
        missing.append("KALSHI_LIVE_ENABLED=1")
    if not KALSHI_API_KEY_ID:
        missing.append("KALSHI_API_KEY_ID")
    if not KALSHI_PRIVATE_KEY_PATH or not Path(KALSHI_PRIVATE_KEY_PATH).expanduser().exists():
        missing.append("KALSHI_PRIVATE_KEY_PATH (readable PEM file)")
    if KALSHI_SUBACCOUNT <= 0:
        missing.append("KALSHI_SUBACCOUNT (a dedicated subaccount, never the primary)")
    if KALSHI_CAPITAL_USD <= 0:
        missing.append("KALSHI_CAPITAL_USD")
    return missing

# ---- historical corpus (every pump.fun graduation, pulled from the migration wallet + pump.fun's swap-api) ----
CORPUS_DIR = DATA_DIR / "corpus"
CORPUS_DAYS = env_int("CORPUS_DAYS", 240)                          # enumerate graduations this far back (must reach before CORPUS_PULL_BEFORE)
CORPUS_MIGRATION_AUTHORITY = env_str("CORPUS_MIGRATION_AUTHORITY", "39azUYFWPz3VHgKCf3VChUwbpURdCHRxjWVowf5jUJjg")
CORPUS_SWAP_API = env_str("CORPUS_SWAP_API", "https://swap-api.pump.fun")
CORPUS_PACE_S = env_float("CORPUS_PACE_S", 2.2)                    # seconds between swap-api requests (~30/min bucket; 1/s throttles after ~20)
CORPUS_PACE_MIN_S = env_float("CORPUS_PACE_MIN_S", 2.2)
CORPUS_CANDLE_1M_H = env_float("CORPUS_CANDLE_1M_H", 12.0)          # 1-minute candles for this long after graduation
CORPUS_CANDLE_5M_D = env_float("CORPUS_CANDLE_5M_D", 7.0)          # then 5-minute candles until this many days
CORPUS_MIN_LIFE_H = env_float("CORPUS_MIN_LIFE_H", 1.0)             # per-trade rows only for tokens that traded at least this long
CORPUS_TRADES_H = env_float("CORPUS_TRADES_H", 3.0)                 # per-trade rows for the first hours after graduation
CORPUS_TRADES_DAYS = env_int("CORPUS_TRADES_DAYS", 400)             # ... and only for graduations this recent (the 10 % sample bounds the cost)
CORPUS_TRADES_MAX_PAGES = env_int("CORPUS_TRADES_MAX_PAGES", 200)   # 100 trades per page
CORPUS_TRADES_SAMPLE = env_float("CORPUS_TRADES_SAMPLE", 0.10)      # share of surviving tokens that get per-trade rows (deterministic by mint)
CORPUS_MIN_AGE_H = env_float("CORPUS_MIN_AGE_H", 6.0)                # pull a token only once its first hours are complete
CORPUS_RQ0_SOL = env_float("CORPUS_RQ0_SOL", 79.0)                  # PumpSwap quote reserve at migration (85 SOL curve − 6 SOL fee); R_q(t) ≈ RQ0·sqrt(p/p_grad)
CORPUS_FEATURES_DIR = CORPUS_DIR / "features"

# ---- PumpAPI historical replay (free hourly archive of decoded pump.fun / PumpSwap events since 2026-04-18) ----
REPLAY_URL = env_str("REPLAY_URL", "https://replay.pumpapi.io")
REPLAY_START = env_str("REPLAY_START", "2026-04-18")
REPLAY_PARALLEL = env_int("REPLAY_PARALLEL", 4)                     # concurrent hour downloads (~2-4 MB/s each)
REPLAY_ASSEMBLE = env_bool("REPLAY_ASSEMBLE", True)                  # 0: download and parse only (no assembly / corpus_meta / mature rounds)
REPLAY_DIR = CORPUS_DIR / "replay"
CORPUS_PULL_BEFORE = env_str("CORPUS_PULL_BEFORE", REPLAY_START)     # swap-api puller only handles graduations before the replay archive begins

# ---- PumpAPI live stream (free firehose, same events as the replay archive; the selector's parity feed) ----
PUMPSTREAM_URL = env_str("PUMPSTREAM_URL", "wss://stream.pumpapi.io")
PUMP_MINUTES_KEEP_DAYS = env_int("PUMP_MINUTES_KEEP_DAYS", 7)       # older minute rows are archived to Parquet, then removed from the table
