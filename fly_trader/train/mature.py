"""The mature-token universe: every PumpSwap token that traded on a day, at any age, as 1-minute rich candles and
as feature rows — the population the live bot actually meets on Jupiter's lists, as opposed to the first 12 hours
after graduation covered by ``replay_assemble``.

Stage 1 (``aggregate_day``): all ``pump-amm`` trade legs of pump.fun-origin mints (suffix ``pump``, SOL-quoted) of a UTC day,
except legs in pools created directly rather than by a pump.fun migration (``pump_pools``: their owner can pull the unburned
liquidity, and the withdrawal is not in the trade data, so the price just stops after a pump),
with causal filters only (the live stream applies the same ones): quote reserve 0.001–100,000 SOL, a leg more than 50× away
from the median of the mint's previous 200 legs that day is dropped, and each (mint, minute) keeps the pool with the most
legs → per (mint, minute): open/high/low/close,
SOL volume split by side, buy/sell counts, distinct traders in the minute, and the pool's quote reserve at the
minute's last trade → ``data/corpus/mature/<day>.parquet`` (registry ``mature_days``).

Stage 2 (``build_day``): feature rows for day D using D−1 as lookback. Each minute feeds ``TokenState`` as two
synthetic trades (the minute's buy volume and sell volume at the close), so imbalance and volume features are
exact and price features match the live engine; trade counts come from the candle counts (``logn_*``),
signer counts are replaced by per-minute-distinct trader sums (``logsigners_*``, an upper bound, flagged in the
mask as absent), Jupiter stats stay masked. Age and price-vs-graduation come from ``corpus_meta`` when the
graduation is inside the archive (``corpus_meta.graduated_at``, the source live uses too); older tokens carry NaN age.
Output: ``data/corpus/features_mature/<day>/part.parquet``. Both outputs carry a version in their Parquet metadata
(``AGG_VERSION``, ``market.features.FEATURE_VERSION``); ``loop_once`` rebuilds any file from another version and moves
the old one to ``data/corpus/_mature_stale/``. Feature parts also record how many of their mints had a known graduation
(``fly_known``); a part is rebuilt once ``corpus_meta`` dates more of them (older days are backfilled after newer ones).
"""
from __future__ import annotations

import fcntl
import logging
import math
import os
import shutil
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config, logging_setup
from ..db.connection import transaction
from ..market.exit_cost import PUMP_SUPPLY
from ..market.features import FEATURE_VERSION, FIDX, TokenMeta, TokenState
from .corpus_features import SCHEMA as FEAT_SCHEMA, PRE_COLS, _epoch_s, _row
from .corpus_meta import blocked_pool_ids
from .flow import FLOW_COLS, MARKET_COLS, N_SKILL, SKILL_COLS, UNKNOWN, FlowWindow, MarketWindow

log = logging.getLogger(__name__)
MATURE_DIR = config.CORPUS_DIR / "mature"
MATURE_FEAT_DIR = config.CORPUS_DIR / "features_mature"
STALE_DIR = config.CORPUS_DIR / "_mature_stale"
AGG_VERSION = 4          # 2: causal price band and per-minute dominant pool; 3: directly created (custom) PumpSwap pools excluded; 4: wallet flow columns (train/flow.py)
KNOWN_REBUILD_FRAC, KNOWN_REBUILD_MIN = 0.01, 25     # a part is rebuilt once corpus_meta dates this many more of its mints
MASKED = ["logsigners_15m", "logsigners_1h", "hawkes", "log_since_last", "vpin_15m"]
EXTRA = ["traders_15m", "traders_1h", "n_trades_1m"] + FLOW_COLS + SKILL_COLS + MARKET_COLS
SCHEMA = FEAT_SCHEMA
for _c in EXTRA:
    SCHEMA = SCHEMA.append(pa.field(_c, pa.float32()))


def part_version(path) -> int:
    md = pq.read_schema(path).metadata or {}
    try:
        return int(md.get(b"fly_version", b"0"))
    except ValueError:
        return 0


def agg_skill(path) -> str:
    """The wallet-skill version an aggregate's skill buckets came from ('none': no table for its day)."""
    return (pq.read_schema(path).metadata or {}).get(b"fly_skill", b"none").decode()


def _skill_version() -> str:
    from .wallet_skill import skill_version
    return skill_version()


def agg_stale(path) -> bool:
    """An aggregate to rebuild: another aggregation version, or skill buckets from another skill version while its day now
    has a table of the version in force (days without a table keep 'none')."""
    p = Path(path)
    if part_version(p) != AGG_VERSION:
        return True
    want = _skill_version()
    return want != "none" and (SKILL_DIR / f"{p.stem}.parquet").exists() and agg_skill(p) != want


def part_agg(path) -> int:
    """The aggregation version a feature part was built from (parts from before the key: 0 = unknown, rebuilt)."""
    md = pq.read_schema(path).metadata or {}
    try:
        return int(md.get(b"fly_agg", b"0"))
    except ValueError:
        return 0


def part_current(path) -> bool:
    p = Path(path)
    return p.exists() and part_version(p) == FEATURE_VERSION and part_agg(p) == AGG_VERSION


def build_complete() -> tuple[bool, str]:
    """Is the training data complete: every aggregate from the current version, no ingested day left unaggregated, and
    a current feature part for every non-empty aggregate day that has a previous day? (reason when not)"""
    aggs = sorted(MATURE_DIR.glob("*.parquet"))
    if not aggs:
        return False, "no aggregated days yet"
    old = [f for f in aggs if agg_stale(f)]
    if old:
        return False, f"{len(old)} day(s) still to re-aggregate"
    pending = days_ready()
    if pending:
        return False, f"{len(pending)} ingested day(s) not aggregated yet"
    no_pid = days_missing_pool_id()
    if no_pid:
        return False, f"{len(no_pid)} day(s) await re-downloaded pool ids (fly-trader refetch-pool-ids; e.g. {no_pid[0]})"
    missing = [f.stem for f in aggs[1:] if pq.read_metadata(f).num_rows and not part_current(MATURE_FEAT_DIR / f.stem / "part.parquet")]
    if missing:
        return False, f"{len(missing)} day(s) of features still to build (e.g. {missing[0]})"
    return True, f"{len(aggs)} days ready"


def part_known(path) -> int:
    """Mints of a feature part whose graduation time was known when it was built (parts from before the key: from age_h)."""
    md = pq.read_schema(path).metadata or {}
    if b"fly_known" in md:
        return int(md[b"fly_known"])
    t = pq.read_table(path, columns=["mint", "age_h"]).to_pandas()
    return int(t.loc[t["age_h"].notna(), "mint"].nunique())


def write_part(table: pa.Table, path, version: int, extra: dict[str, str] | None = None) -> None:
    """Atomic write with the version (and any ``extra`` keys) in the Parquet metadata."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    md = {**(table.schema.metadata or {}), b"fly_version": str(version).encode(), **{k.encode(): str(v).encode() for k, v in (extra or {}).items()}}
    table = table.replace_schema_metadata(md)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    pq.write_table(table, tmp, compression="zstd"); tmp.replace(path)


def _retire(path, tag: str) -> None:
    """Move a stale output aside (data is never deleted)."""
    path = Path(path)
    if path.exists():
        STALE_DIR.mkdir(parents=True, exist_ok=True)
        path.replace(STALE_DIR / f"{tag}_v{part_version(path)}_{int(time.time())}.parquet")


def _archive(path, tag: str) -> None:
    """Keep a copy of an output about to be replaced (data is never deleted); unlike ``_retire`` the original stays in place,
    so a concurrent reader (training) never sees the day missing."""
    path = Path(path)
    if path.exists():
        STALE_DIR.mkdir(parents=True, exist_ok=True)
        dst = STALE_DIR / f"{tag}_v{part_version(path)}_{int(time.time())}.parquet"
        try:
            os.link(path, dst)                   # the atomic replace gives the part a new inode; the link keeps the old one
        except OSError:
            shutil.copy2(path, dst)


def _hour_files(d: date) -> list[str]:
    day = config.REPLAY_DIR / d.isoformat()
    return [str(day / f"{h:02d}_trades.parquet") for h in range(24) if (day / f"{h:02d}_trades.parquet").exists()]


def _lacks_pool_id(files: list[str]) -> list[str]:
    return [f for f in files if "pool_id" not in pq.read_schema(f).names]


def days_missing_pool_id() -> list[str]:
    """Days with an hour file written before the parser kept ``pool_id``: aggregating them would skip the blocked-pool
    filter and the dominant-pool choice the live stream applies, so they wait for the re-download."""
    out = []
    for day in sorted(p for p in config.REPLAY_DIR.glob("*") if p.is_dir()):
        try:
            d = date.fromisoformat(day.name)
        except ValueError:
            continue
        if _lacks_pool_id(_hour_files(d)):
            out.append(day.name)
    return out


def days_ready() -> list[date]:
    """Days with all 24 replay hours ingested and no mature aggregate yet, newest first."""
    with transaction() as conn:
        done = {r["hour"].astimezone(timezone.utc) for r in conn.execute("SELECT hour FROM replay_hours WHERE status IN ('done','missing')").fetchall()}
        built = {r["day"] for r in conn.execute("SELECT day FROM mature_days").fetchall()}
    out = []
    for d in sorted({h.date() for h in done}, reverse=True):
        if d in built:
            continue
        if all(datetime(d.year, d.month, d.day, h, tzinfo=timezone.utc) in done for h in range(24)):
            out.append(d)
    return out


def aggregate_table(files: list[str], skill, blocked: list[str] | None = None, insiders: dict | None = None) -> pa.Table:
    """A day's minute aggregates from its hour files, with ``skill`` (wallet → decile table, or None) as the day's skill
    table — ``aggregate_day``'s SQL, also used to rebuild the skill inputs under a candidate table (train/wallet_skill.fit).
    ``blocked`` (pool ids) and ``insiders`` (mint -> wallets) are read from Postgres when not given; a caller that aggregates
    many days hands them in once, rather than opening two connections per day."""
    con = duckdb.connect()
    cols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [files]).description]
    quote_filter = "AND (quote_mint IS NULL OR quote_mint = 'So11111111111111111111111111111111111111112')" if "quote_mint" in cols else ""
    pool_sel = "pool_id" if "pool_id" in cols else "NULL AS pool_id"
    if blocked is None:
        with transaction() as conn:
            blocked = blocked_pool_ids(conn)
    con.register("blocked", pa.table({"pool_id": pa.array(blocked, pa.string())}))
    pool_filter = "AND (pool_id IS NULL OR pool_id NOT IN (SELECT pool_id FROM blocked))" if "pool_id" in cols else ""
    # causal filters only (nothing later in the day decides which earlier trades exist); the live stream applies the same:
    # no directly created pool, reserve band, a leg within 50x of the median of the mint's previous 200 legs, and per (mint, minute) the pool with the most legs
    con.execute(f"""CREATE TEMP TABLE raw AS SELECT mint, ts, slot, trader, side, sol, price, quote_in_pool, {pool_sel}
                    FROM read_parquet(?, union_by_name = true)
                    WHERE pool = 'pump-amm' AND price > 0 AND mint LIKE '%pump' AND quote_in_pool BETWEEN 0.001 AND 100000 {quote_filter} {pool_filter}""", [files])
    con.execute("""CREATE TEMP TABLE banded AS SELECT * EXCLUDE (pref) FROM (
                      SELECT *, median(price) OVER (PARTITION BY mint ORDER BY ts, slot ROWS BETWEEN 200 PRECEDING AND 1 PRECEDING) AS pref FROM raw)
                    WHERE pref IS NULL OR price BETWEEN pref / 50 AND pref * 50""")
    con.execute("""CREATE TEMP TABLE dom AS SELECT mint, mb, arg_max(pool_id, n) AS pid FROM (
                      SELECT mint, time_bucket(INTERVAL 1 MINUTE, ts) AS mb, pool_id, count(*) AS n FROM banded GROUP BY ALL) GROUP BY mint, mb""")
    con.execute("""CREATE TEMP TABLE clean AS SELECT b.* FROM banded b JOIN dom d ON d.mint = b.mint AND d.mb = time_bucket(INTERVAL 1 MINUTE, b.ts)
                    WHERE b.pool_id IS NULL OR d.pid IS NULL OR b.pool_id = d.pid""")
    con.execute("""CREATE TEMP TABLE cand AS
        SELECT mint, time_bucket(INTERVAL 1 MINUTE, ts) AS ts,
               first(price ORDER BY ts, slot) AS open, max(price) AS high, min(price) AS low, last(price ORDER BY ts, slot) AS close,
               sum(CASE WHEN side = 1 THEN sol ELSE 0 END) AS buy_sol, sum(CASE WHEN side = -1 THEN sol ELSE 0 END) AS sell_sol,
               count(*) FILTER (WHERE side = 1) AS n_buys, count(*) FILTER (WHERE side = -1) AS n_sells,
               count(DISTINCT trader) AS n_traders, last(quote_in_pool ORDER BY ts, slot) AS resq_sol
        FROM clean GROUP BY mint, time_bucket(INTERVAL 1 MINUTE, ts)""")
    # wallet flow (train/flow.py, the live stream's minute_wallet_cols): per (mint, minute, trader) bought and sold SOL
    con.execute("""CREATE TEMP TABLE w AS SELECT mint, time_bucket(INTERVAL 1 MINUTE, ts) AS mb, trader,
                      sum(CASE WHEN side = 1 THEN sol ELSE 0 END) AS b, sum(CASE WHEN side = -1 THEN sol ELSE 0 END) AS s
                    FROM clean WHERE trader IS NOT NULL GROUP BY ALL""")
    mints = [r[0] for r in con.execute("SELECT DISTINCT mint FROM w").fetchall()]
    if insiders is None:
        with transaction() as conn:
            rows = conn.execute("SELECT mint, wallet FROM token_insiders WHERE mint = ANY(%s)", (mints,)).fetchall() if mints else []
        ins = [(r["mint"], r["wallet"]) for r in rows]
    else:
        ins = [(m, w) for m in mints for w in insiders.get(m, ())]
    con.register("ins", pa.table({"mint": pa.array([m for m, _ in ins], pa.string()), "wallet": pa.array([w for _, w in ins], pa.string())}))
    con.register("skill", skill if skill is not None else pa.table({"wallet": pa.array([], pa.string()), "bucket": pa.array([], pa.int8())}))
    sk_expr = ("[" + ", ".join(f"sum(CASE WHEN coalesce(k.bucket, {UNKNOWN}) = {q} THEN w.b ELSE 0 END)" for q in range(N_SKILL)) + "]") if skill is not None else "NULL::DOUBLE[]"
    con.execute(f"""CREATE TEMP TABLE wc AS SELECT w.mint, w.mb AS ts, count(*) FILTER (WHERE w.b > 0) AS n_buyers,
                      sum(CASE WHEN w.b > 0 AND w.s > 0 THEN w.b + w.s ELSE 0 END) AS wash_sol,
                      sum(CASE WHEN w.b > 0 AND w.s > 0 THEN w.b ELSE 0 END) AS wash_buy_sol, max(w.s) AS top_sell_sol,
                      sum(CASE WHEN i.wallet IS NOT NULL THEN w.s ELSE 0 END) AS insider_sell_sol, {sk_expr} AS skill_buy
                    FROM w LEFT JOIN ins i ON i.mint = w.mint AND i.wallet = w.trader LEFT JOIN skill k ON k.wallet = w.trader
                    GROUP BY w.mint, w.mb""")
    tab = con.execute("""SELECT c.*, coalesce(wc.n_buyers, 0) AS n_buyers, coalesce(wc.wash_sol, 0) AS wash_sol, coalesce(wc.wash_buy_sol, 0) AS wash_buy_sol,
                             coalesce(wc.top_sell_sol, 0) AS top_sell_sol, coalesce(wc.insider_sell_sol, 0) AS insider_sell_sol, wc.skill_buy
                          FROM cand c LEFT JOIN wc USING (mint, ts) ORDER BY c.mint, c.ts""").fetch_arrow_table()
    con.close()
    return tab


def aggregate_day(d: date) -> int:
    t0 = time.time(); files = _hour_files(d)
    if not files:
        return 0
    if _lacks_pool_id(files):                   # exact parity with the live stream needs every leg's pool (refetch-pool-ids)
        log.warning("mature %s: %d hour file(s) lack pool_id; not aggregated until they are downloaded again", d, len(_lacks_pool_id(files)))
        return 0
    MATURE_DIR.mkdir(parents=True, exist_ok=True)
    skill = skill_table_for(d)
    tab = aggregate_table(files, skill)
    write_part(tab, MATURE_DIR / f"{d.isoformat()}.parquet", AGG_VERSION, {"fly_skill": _skill_version() if skill is not None else "none"})
    n_mints = len(pa.compute.unique(tab["mint"]))
    with transaction() as conn:
        conn.execute("INSERT INTO mature_days (day, mints, rows, took_s) VALUES (%s,%s,%s,%s) ON CONFLICT (day) DO UPDATE SET mints = EXCLUDED.mints, rows = EXCLUDED.rows, took_s = EXCLUDED.took_s, built_at = now()",
                     (d, n_mints, tab.num_rows, time.time() - t0))
    log.info("mature %s: %d mints, %d minute rows in %.0fs", d, n_mints, tab.num_rows, time.time() - t0)
    return tab.num_rows


SKILL_DIR = config.CORPUS_DIR / "wallet_skill"


def skill_table_for(d: date):
    """The wallet-skill table in force on day D (``train/wallet_skill.py``: wallet → decile, from earlier days only),
    as a pyarrow table, or None when the day has none (skill buckets are then NULL, as live without a table)."""
    f = SKILL_DIR / f"{d.isoformat()}.parquet"
    return pq.read_table(f, columns=["wallet", "bucket"]) if f.exists() else None


def _graduations() -> dict:
    with transaction() as conn:
        return {r["mint"]: r["graduated_at"] for r in conn.execute("SELECT mint, graduated_at FROM corpus_meta WHERE graduated_at IS NOT NULL").fetchall()}


def _supplies() -> dict:
    """mint → token supply from the create event (mayhem tokens: 2 billion); tokens not in corpus_meta use PUMP_SUPPLY."""
    with transaction() as conn:
        return {r["mint"]: float(r["supply"]) for r in conn.execute("SELECT mint, supply FROM corpus_meta WHERE supply > 0").fetchall()}


def build_day(d: date, lookback_days: int = 1, grads: dict | None = None, supplies: dict | None = None) -> int:
    """Feature rows for every minute of day D, warmed up on the previous day's candles. ``grads``: mint → graduated_at
    (default: read from ``corpus_meta``); the part records how many of its mints had one (metadata ``fly_known``)."""
    t0 = time.time()
    frames = []
    for k in range(lookback_days, -1, -1):
        f = MATURE_DIR / f"{(d - timedelta(days=k)).isoformat()}.parquet"
        if f.exists():
            frames.append(pq.read_table(f).to_pandas())
    if not frames:
        return 0
    cd = pd.concat(frames, ignore_index=True).sort_values(["mint", "ts"])
    day_start = datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp(); day_end = day_start + 86400
    grads = _graduations() if grads is None else grads
    supplies = _supplies() if supplies is None else supplies
    mkt = _market_windows(cd)
    out: list[dict] = []; n_mints = 0; n_known = 0
    for mint, x in cd.groupby("mint", sort=False):
        ts_s = _epoch_s(x["ts"])
        if not (ts_s >= day_start).any():
            continue
        n_mints += 1
        g = grads[mint].timestamp() if grads.get(mint) else None
        n_known += g is not None
        st = TokenState(mint); tm = TokenMeta(mint=mint, program_label="Pump.fun Amm", graduated_at=g, supply=supplies.get(mint) or PUMP_SUPPLY)
        pre = {k: float("nan") for k in PRE_COLS}
        o, h, l, c = x["open"].to_numpy(), x["high"].to_numpy(), x["low"].to_numpy(), x["close"].to_numpy()
        bs, ss, nb, ns, nt, rq = x["buy_sol"].to_numpy(), x["sell_sol"].to_numpy(), x["n_buys"].to_numpy(), x["n_sells"].to_numpy(), x["n_traders"].to_numpy(), x["resq_sol"].to_numpy()
        hist_ts, hist_n, hist_tr = [], [], []
        fw = FlowWindow(); wcols = [x[c].to_numpy() if c in x else np.zeros(len(x)) for c in ("n_buyers", "wash_sol", "wash_buy_sol", "top_sell_sol", "insider_sell_sol")]
        skb = x["skill_buy"].to_numpy() if "skill_buy" in x else np.full(len(x), None, dtype=object)
        for i in range(len(ts_s)):
            t_end = float(ts_s[i]) + 60.0; price = float(c[i]); resq = float(rq[i]) if np.isfinite(rq[i]) and rq[i] > 0 else None
            if bs[i] > 0:
                st.append(t_end - 2e-3, price, float(bs[i]), True, None, resq)
            if ss[i] > 0 or bs[i] <= 0:
                st.append(t_end - 1e-3, price, float(ss[i]), False, None, resq)
            hist_ts.append(t_end); hist_n.append(float(nb[i] + ns[i])); hist_tr.append(float(nt[i]))
            fw.append(t_end, bs[i], ss[i], wcols[0][i], wcols[1][i], wcols[2][i], wcols[3][i], wcols[4][i], None if skb[i] is None else list(skb[i]))
            if t_end <= day_start:
                continue
            f, mask = st.features(t_end, tm)
            ht = np.asarray(hist_ts); hn = np.asarray(hist_n); htr = np.asarray(hist_tr)
            for k, w in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)):
                sel = ht > t_end - w
                f[FIDX[f"logn_{k}"]] = math.log1p(float(hn[sel].sum()))
            for name in MASKED:
                f[FIDX[name]] = 0.0; mask &= ~(1 << FIDX[name])
            if g is None:
                f[FIDX["log_age_h"]] = 0.0; mask &= ~(1 << FIDX["log_age_h"])
            row = _row(mint, float(ts_s[i]), g if g is not None else float(ts_s[i]) - 1e9, float(o[i]), float(h[i]), float(l[i]), price, float(bs[i] + ss[i]), resq, False, f, mask, pre)
            row["traders_15m"] = float(htr[ht > t_end - 900].sum()); row["traders_1h"] = float(htr[ht > t_end - 3600].sum()); row["n_trades_1m"] = float(nb[i] + ns[i])
            row.update(fw.features(t_end)); row["mkt_vol_1h"] = mkt.get(t_end, 0.0)
            if g is None:
                row["age_h"] = float("nan"); row["t_rel_min"] = -1; row["phase"] = "amm"
            out.append(row)
    if not out:
        return 0
    write_part(pa.Table.from_pylist(out, schema=SCHEMA), MATURE_FEAT_DIR / d.isoformat() / "part.parquet", FEATURE_VERSION, {"fly_known": n_known, "fly_agg": AGG_VERSION})
    log.info("mature features %s: %d mints (%d with graduation), %d rows in %.0fs", d, n_mints, n_known, len(out), time.time() - t0)
    return len(out)


def _market_windows(cd: pd.DataFrame) -> dict:
    """minute end → ``mkt_vol_1h`` over every mint of the frame (the live engine's MarketWindow over its minute rows)."""
    tot = (cd["buy_sol"].fillna(0.0) + cd["sell_sol"].fillna(0.0)).groupby(_epoch_s(cd["ts"])).sum().sort_index()
    mw = MarketWindow(); out = {}
    for t, v in tot.items():
        t_end = float(t) + 60.0; mw.add(t_end, float(v)); out[t_end] = mw.value(t_end)
    return out


def _knows_more(part: Path, grads: dict) -> bool:
    """corpus_meta now dates enough more of the part's mints than when it was built (≥ KNOWN_REBUILD_FRAC of them, at least
    KNOWN_REBUILD_MIN): old tokens keep trading for months, so any lower bar rebuilds every part on each backfilled day."""
    mints = pa.compute.unique(pq.read_table(part, columns=["mint"])["mint"]).to_pylist()
    grew = sum(1 for m in mints if m in grads) - part_known(part)
    return grew >= max(KNOWN_REBUILD_MIN, KNOWN_REBUILD_FRAC * len(mints))


def loop_once() -> int:
    """One build round. A file lock serialises rounds across processes (the console's replay worker and the CLI)."""
    MATURE_DIR.mkdir(parents=True, exist_ok=True)
    with open(MATURE_DIR / ".build.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return _loop_once()


def _loop_once() -> int:
    n = 0
    from . import wallet_skill
    n += wallet_skill.daily()       # wallet days and the tables in force first: a new table re-aggregates its day below
    # aggregates from another version are rebuilt; their day's and the next day's features depend on them
    for f in sorted(MATURE_DIR.glob("*.parquet"), reverse=True):
        if agg_stale(f) and _hour_files(date.fromisoformat(f.stem)):
            d = date.fromisoformat(f.stem)
            _retire(f, f"agg_{d}")
            for dd in (d, d + timedelta(days=1)):
                _retire(MATURE_FEAT_DIR / dd.isoformat() / "part.parquet", f"feat_{dd}")
            with transaction() as conn:
                conn.execute("DELETE FROM mature_days WHERE day = %s", (d,))
    for d in days_ready():
        aggregate_day(d); n += 1
    grads = _graduations()          # corpus_meta.rebuild reads the day's events before every round, so its graduations are known
    for f in sorted(MATURE_DIR.glob("*.parquet"), reverse=True):
        d = date.fromisoformat(f.stem)
        part = MATURE_FEAT_DIR / d.isoformat() / "part.parquet"
        # a current part is rebuilt when graduations assembled since (the backfill runs newest-first) date more of its mints
        if part_current(part) and not _knows_more(part, grads):
            continue
        if not (MATURE_DIR / f"{(d - timedelta(days=1)).isoformat()}.parquet").exists():
            continue                    # the previous day is the lookback
        _archive(part, f"feat_{d}")             # the old part stays readable until the new one replaces it atomically
        if not build_day(d, grads=grads) and part.exists():
            part.unlink()                        # nothing to write: the (archived) old part must not linger
        n += 1
    return n


def main() -> None:
    logging_setup.setup("replay")
    print(f"processed {loop_once()} day steps")
