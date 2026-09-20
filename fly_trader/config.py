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

SECRET_ENV_NAMES = ("HELIUS_API_KEY", "JUPITER_API_KEY", "BOT_PRIVATE_KEY")


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
# exit_cost_0p1 for a fixed 0.1 SOL, but positions are Kelly-sized up to MAX_POSITION_FRACTION of the bankroll, and
# impact = value/(value + reserves) grows with size. 0 = the sizing rule's own ceiling (train/decisions.py).
LABEL_SIZE_SOL = env_float("LABEL_SIZE_SOL", 0.0)
MIN_POSITION_SOL = env_float("MIN_POSITION_SOL", 0.02)             # smaller sizes are skipped
KILL_SWITCH_DRAWDOWN = env_float("KILL_SWITCH_DRAWDOWN", 0.30)
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
DEVICE = env_str("DEVICE", "mps")
NT_SIGN_MODE = env_str("NT_SIGN_MODE", "shiu")  # shiu | flybrain
CELL_TYPES_SOURCE = env_str("CELL_TYPES_SOURCE", "annotations")  # annotations | flybrain

# ---- costs ----
LAMBDA_IMPACT_DLMM = env_float("LAMBDA_IMPACT_DLMM", 0.02)

# ---- training lifecycle ----
RESET_ON_START = env_bool("RESET_ON_START", True)  # wipe the paper book and its stats when the runner starts


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
    if not env_str("BOT_PRIVATE_KEY"):
        missing.append("BOT_PRIVATE_KEY")
    if not JUPITER_API_KEY:
        missing.append("JUPITER_API_KEY")
    if not HELIUS_API_KEY:
        missing.append("HELIUS_API_KEY")
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
