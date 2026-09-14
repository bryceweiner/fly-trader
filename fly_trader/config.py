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
CAPTURE_DIR = DATA_DIR / "capture"
BRAIN_DIR = DATA_DIR / "brain"
ENCODER_DIR = DATA_DIR / "encoders"
PG_ARCHIVE_DIR = DATA_DIR / "pg_archive"
VOC_CORPUS_DIR = Path(env_str("VOC_CORPUS_DIR", "/Users/bryce/Documents/code/VOC/data/live_meteora"))

# ---- database ----
DATABASE_URL = env_str("DATABASE_URL", "postgresql:///fly_trader")

# ---- chain / execution (default-deny) ----
SOLANA_CLUSTER = env_str("SOLANA_CLUSTER", "mainnet-beta")
LIVE_ENABLED = env_bool("LIVE_ENABLED", False)
CAPITAL_SOL = env_float("CAPITAL_SOL", 0.0)
MAX_POSITION_SOL = env_float("MAX_POSITION_SOL", 0.0)
GAS_RESERVE_SOL = env_float("GAS_RESERVE_SOL", 0.30)
NOTIONAL_CAP_SOL_24H = env_float("NOTIONAL_CAP_SOL_24H", 10.0)
KILL_SWITCH_DRAWDOWN = env_float("KILL_SWITCH_DRAWDOWN", 0.30)
HARD_STOP_FRAC = env_float("HARD_STOP_FRAC", 0.50)
DEAD_BAG_HOURS = env_float("DEAD_BAG_HOURS", 3.0)
# reflex exits (sharp, operator design 2026-09-13): rejection below -EXIT_LOSS, a winner that turns negative, satiety
EXIT_LOSS = env_float("EXIT_LOSS", 0.02)          # unrealized return <= -2% -> exit ("bitter"); -1% sat inside tape noise
TURN_MARGIN = env_float("TURN_MARGIN", 0.01)      # was >= +1% at its peak and is now <= 0 -> exit ("turned")
SATIETY_TARGET = env_float("SATIETY_TARGET", 10.0) # exit when sum over held beats of max(u,0) x dt reaches this (e.g. +5% for 200 s)
FEE_ROUND_TRIP = env_float("FEE_ROUND_TRIP", 0.011)  # satiety may only realise a gain that clears the round-trip cost
REWARD_UNIT = env_float("REWARD_UNIT", 0.02)      # 1 dopamine unit = 2% unrealized return; clipped at +-DELTA_CLIP
REALIZED_GAIN = env_float("REALIZED_GAIN", 1.0)   # realized log return of the whole trade, delivered as a reward pulse at exit
CIRCUIT_THRESHOLD = env_int("CIRCUIT_THRESHOLD", 3)
STALE_FEED_S = env_float("STALE_FEED_S", 30.0)
EXECUTABILITY_MAX_IMPACT = env_float("EXECUTABILITY_MAX_IMPACT", 0.30)
SLIPPAGE_ENTRY_BPS = env_int("SLIPPAGE_ENTRY_BPS", 150)
SLIPPAGE_EXIT_BPS = env_int("SLIPPAGE_EXIT_BPS", 300)
SLIPPAGE_FORCED_BPS = env_int("SLIPPAGE_FORCED_BPS", 500)
SLIPPAGE_STEP_BPS = env_int("SLIPPAGE_STEP_BPS", 100)
MAX_SLIPPAGE_ENTRY_BPS = env_int("MAX_SLIPPAGE_ENTRY_BPS", 300)
MAX_SLIPPAGE_FORCED_BPS = env_int("MAX_SLIPPAGE_FORCED_BPS", 800)
EXEC_TIMEOUT_S = env_float("EXEC_TIMEOUT_S", 90.0)  # per order; covers blockhash expiry (~60-90 s)

HELIUS_HTTP = "https://mainnet.helius-rpc.com/"
HELIUS_WS = "wss://mainnet.helius-rpc.com/"
JUPITER_SWAP_BASE = "https://api.jup.ag/swap/v2"
JUPITER_TOKENS_BASE = "https://api.jup.ag/tokens/v2"
JUPITER_PRICE_BASE = "https://api.jup.ag/price/v3"
JUPITER_RPS = env_float("JUPITER_RPS", 8.0)
DEXSCREENER_BASE = "https://api.dexscreener.com"

WSOL_MINT = "So11111111111111111111111111111111111111112"
USDC_MINT = "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v"
USDT_MINT = "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB"
QUOTE_MINTS = {WSOL_MINT: 9, USDC_MINT: 6, USDT_MINT: 6}
LAMPORTS_PER_SOL = 1_000_000_000

# ---- universe ----
LAUNCHPADS = env_list("LAUNCHPADS", ["pump.fun"])
ACTIVITY_FLOOR_SOL_3H = env_float("ACTIVITY_FLOOR_SOL_3H", 1.0)
WATCH_DAYS_AFTER_GRADUATION = env_float("WATCH_DAYS_AFTER_GRADUATION", 7.0)
DISCOVER_INTERVAL_S = env_float("DISCOVER_INTERVAL_S", 60.0)
STATS_REFRESH_S = env_float("STATS_REFRESH_S", 600.0)

# ---- brain ----
DEVICE = env_str("DEVICE", "mps")
BEAT_S = env_float("BEAT_S", 1.0)
BEAT_TICKS = env_int("BEAT_TICKS", 50)
SLOTS = env_int("SLOTS", 128)
READOUT_WINDOW = env_int("READOUT_WINDOW", 30)
LEAK = env_float("LEAK", 0.95)
THETA = env_float("THETA", 1.0)
REFRACTORY = env_int("REFRACTORY", 3)
KWTA_ENABLED = env_bool("KWTA_ENABLED", True)
KC_KWTA_FRAC = env_float("KC_KWTA_FRAC", 0.05)
KWTA_TAU_TICKS = env_float("KWTA_TAU_TICKS", 10.0)  # 0 = instantaneous V (flybrain-style)
KWTA_MODE = env_str("KWTA_MODE", "external")  # external (encoder hash winners) | drive (low-passed input)
DAN_MODE = env_str("DAN_MODE", "injected")  # injected (Bennett RPE drives DANs) | measured (DAN rates read back)
NT_SIGN_MODE = env_str("NT_SIGN_MODE", "shiu")  # shiu | flybrain
CELL_TYPES_SOURCE = env_str("CELL_TYPES_SOURCE", "annotations")  # annotations | flybrain
MEMBRANE_NOISE = env_float("MEMBRANE_NOISE", 0.0)

# ---- encoders ----
K_ODOR = env_int("K_ODOR", 8)
I_ORN_MAX = env_float("I_ORN_MAX", 0.15)
I_DANGER_MAX = env_float("I_DANGER_MAX", 0.15)
I_KC = env_float("I_KC", 0.15)  # sustained current on winning KCs (hash code)
I_CX_TONIC = env_float("I_CX_TONIC", 0.02)
I_HUNGER_MAX = env_float("I_HUNGER_MAX", 0.05)
HUNGER_BEATS = env_int("HUNGER_BEATS", 400)
G_TASTE = env_float("G_TASTE", 0.15)
I_JO_MAX = env_float("I_JO_MAX", 0.15)
I_THERMO_MAX = env_float("I_THERMO_MAX", 0.15)
IDENTITY_DIM = 16
IDENTITY_SCALE = env_float("IDENTITY_SCALE", 2.0)

# ---- plasticity ----
ETA = env_float("ETA", 0.02)
ETA_MAX = env_float("ETA_MAX", 0.05)
BETA_POT = env_float("BETA_POT", 0.3)
TAU_E_BEATS = env_float("TAU_E_BEATS", 3.0)
DW_FROB_FRAC = env_float("DW_FROB_FRAC", 0.02)
KAPPA_HOURS = env_float("KAPPA_HOURS", 4.0)
W_MAX_MULT = env_float("W_MAX_MULT", 3.0)
GAMMA = env_float("GAMMA", 0.0)  # 0 = Bennett's d = r - m (no TD bootstrapping)
DELTA_CLIP = env_float("DELTA_CLIP", 3.0)
SNAPSHOT_EVERY_BEATS = env_int("SNAPSHOT_EVERY_BEATS", 2000)
PLASTIC_MASK = env_str("PLASTIC_MASK", "connectome")  # connectome | dense

# ---- decision ----
Z_BUY = env_float("Z_BUY", 2.0)            # enter when valence is >= Z_BUY spreads above the slot mean (2.0: turnover cut) ...
Z_HUNGER = env_float("Z_HUNGER", 0.5)      # ... minus Z_HUNGER x hunger (hunger = undeployed capital fraction)
Z_SELL = env_float("Z_SELL", -0.5)         # exit when valence falls Z_SELL spreads below the mean
Z_SIZE_RANGE = env_float("Z_SIZE_RANGE", 2.0)  # full size at z_buy_eff + Z_SIZE_RANGE
Z_SIGMA_FLOOR = env_float("Z_SIGMA_FLOOR", 0.005)
CONFIRM_BEATS = env_int("CONFIRM_BEATS", 2)
SIZE_MIN_SOL = env_float("SIZE_MIN_SOL", 0.01)
SOFTMAX_BETA = env_float("SOFTMAX_BETA", 1.0)  # per valence-spread unit
DWELL_BEATS = env_int("DWELL_BEATS", 20)

# ---- capture / tape ----
CAPTURE_FLUSH_ROWS = env_int("CAPTURE_FLUSH_ROWS", 5000)
CAPTURE_FLUSH_S = env_float("CAPTURE_FLUSH_S", 30.0)
TAPE_BATCH_MS = env_int("TAPE_BATCH_MS", 250)
TAPE_HOT_HOURS = env_int("TAPE_HOT_HOURS", 72)
PG_ARCHIVE_DAYS = env_int("PG_ARCHIVE_DAYS", 14)
POOL_RELOAD_S = env_float("POOL_RELOAD_S", 30.0)

# ---- context / features ----
CONTEXT_WINDOW_S = env_float("CONTEXT_WINDOW_S", 10800.0)
CONTEXT_GAP_S = env_float("CONTEXT_GAP_S", 600.0)
CONTEXT_READY_COVERAGE = env_float("CONTEXT_READY_COVERAGE", 0.9)
LAMBDA_IMPACT_DLMM = env_float("LAMBDA_IMPACT_DLMM", 0.02)

# ---- corpus replay inside the live brain ----
BEAT_S_SIM = env_float("BEAT_S_SIM", 5.0)           # simulated seconds per beat for replay columns
REPLAY_SLOTS = env_int("REPLAY_SLOTS", 0)            # replay columns removed from the runner (operator decision 2026-09-13)
REPLAY_CORPUS = env_str("REPLAY_CORPUS", "meteora")  # meteora | capture
REPLAY_BOOK = "replay_meteora"

# ---- sniff before feeding ----
WARMUP_S = env_float("WARMUP_S", 120.0)         # no entries in the first WARMUP_S after the runner starts
SNIFF_BEATS = env_int("SNIFF_BEATS", 120)       # a token must have been in a slot this many beats before it can be entered
# hard eligibility (measured 2026-09-13 on 472k slot observations: each condition raised the 5/15-min forward return;
# graduates younger than 6 h averaged -0.7 % and are refused)
ELIG_MIN_AGE_H = env_float("ELIG_MIN_AGE_H", 6.0)
ELIG_MIN_IMB = env_float("ELIG_MIN_IMB", 0.2)          # buy/sell imbalance over 5 m AND 15 m
ELIG_MAX_DD_1H = env_float("ELIG_MAX_DD_1H", -0.05)     # within 5 % of the 1-hour high
ELIG_REQUIRE_RISING = env_bool("ELIG_REQUIRE_RISING", True)  # ret_1m > 0 and ret_5m > 0

# ---- innate valence and realized edge ----
INNATE_GAIN = env_float("INNATE_GAIN", 0.05)       # valence units per std of the innate score (learned m̂ spread is ~0.03-0.13)
EDGE_HORIZON_S = env_float("EDGE_HORIZON_S", 60.0)
EDGE_WINDOW_S = env_float("EDGE_WINDOW_S", 600.0)
EDGE_MIN_SAMPLES = env_int("EDGE_MIN_SAMPLES", 500)
EDGE_MIN = env_float("EDGE_MIN", 0.0)              # realized edge <= this -> no live/mirror entries ("no edge smells bad")
NO_EDGE_BITTER = env_float("NO_EDGE_BITTER", 1.0)  # bitter input (dopamine units) per unit of negative edge

# ---- brain mode ----
BRAIN_MODE = env_str("BRAIN_MODE", "selector")   # selector (the strategy, 2026-09-14) | policy (FlyGM-style trained connectome) | lif (frozen spiking + KC->MBON plasticity)
POLICY_MIN_TRADE_FRAC = env_float("POLICY_MIN_TRADE_FRAC", 0.25)

# ---- training lifecycle ----
RESET_ON_START = env_bool("RESET_ON_START", True)  # wipe paper/replay training state when the runner starts

# ---- books ----
BOOKS = ("live", "paper_free", "paper_mirror")


def helius_http_url() -> str:
    if not HELIUS_API_KEY:
        raise RuntimeError("HELIUS_API_KEY is not set")
    return f"{HELIUS_HTTP}?api-key={HELIUS_API_KEY}"


def helius_ws_url() -> str:
    if not HELIUS_API_KEY:
        raise RuntimeError("HELIUS_API_KEY is not set")
    return f"{HELIUS_WS}?api-key={HELIUS_API_KEY}"


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
    if "DATABASE_URL" in out:                      # may carry a password; the database name is enough
        out["DATABASE_URL"] = "postgresql:///" + str(out["DATABASE_URL"]).rsplit("/", 1)[-1].split("?")[0]
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
REPLAY_DIR = CORPUS_DIR / "replay"
CORPUS_PULL_BEFORE = env_str("CORPUS_PULL_BEFORE", REPLAY_START)     # swap-api puller only handles graduations before the replay archive begins

# ---- PumpAPI live stream (free firehose, same events as the replay archive; the selector's parity feed) ----
PUMPSTREAM_URL = env_str("PUMPSTREAM_URL", "wss://stream.pumpapi.io")
PUMP_MINUTES_KEEP_DAYS = env_int("PUMP_MINUTES_KEEP_DAYS", 7)       # older minute rows are archived to Parquet, then removed from the table
SELECTOR_SOURCE = env_str("SELECTOR_SOURCE", "stream")               # stream (pump_minutes from PumpAPI) | tape (Helius swap_tape, watched pools only)
