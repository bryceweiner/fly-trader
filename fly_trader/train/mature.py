"""The mature-token universe: every PumpSwap token that traded on a day, at any age, as 1-minute rich candles and
as feature rows — the population the live bot actually meets on Jupiter's lists, as opposed to the first 12 hours
after graduation covered by ``replay_assemble``.

Stage 1 (``aggregate_day``): all ``pump-amm`` trade legs of pump.fun-origin mints (suffix ``pump``, SOL-quoted) of a UTC day → per (mint, minute): open/high/low/close,
SOL volume split by side, buy/sell counts, distinct traders in the minute, and the pool's quote reserve at the
minute's last trade → ``data/corpus/mature/<day>.parquet`` (registry ``mature_days``).

Stage 2 (``build_day``): feature rows for day D using D−1 as lookback. Each minute feeds ``TokenState`` as two
synthetic trades (the minute's buy volume and sell volume at the close), so imbalance and volume features are
exact and price features match the live engine; trade counts come from the candle counts (``logn_*``),
signer counts are replaced by per-minute-distinct trader sums (``logsigners_*``, an upper bound, flagged in the
mask as absent), Jupiter stats stay masked. Age and price-vs-graduation come from ``corpus_meta`` when the
graduation is inside the archive; older tokens carry NaN age. Output: ``data/corpus/features_mature/<day>/part.parquet``.
"""
from __future__ import annotations

import logging
import math
import time
from datetime import date, datetime, timedelta, timezone

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..market.features import D, FEATURES, FIDX, TokenMeta, TokenState
from .corpus_features import SCHEMA as FEAT_SCHEMA, PRE_COLS, _row

log = logging.getLogger(__name__)
MATURE_DIR = config.CORPUS_DIR / "mature"
MATURE_FEAT_DIR = config.CORPUS_DIR / "features_mature"
MASKED = ["logsigners_15m", "logsigners_1h", "hawkes", "log_since_last", "vpin_15m"]
EXTRA = ["traders_15m", "traders_1h", "n_trades_1m"]
SCHEMA = FEAT_SCHEMA.append(pa.field("traders_15m", pa.float32())).append(pa.field("traders_1h", pa.float32())).append(pa.field("n_trades_1m", pa.float32()))


def _hour_files(d: date) -> list[str]:
    day = config.REPLAY_DIR / d.isoformat()
    return [str(day / f"{h:02d}_trades.parquet") for h in range(24) if (day / f"{h:02d}_trades.parquet").exists()]


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


def aggregate_day(d: date) -> int:
    t0 = time.time(); files = _hour_files(d)
    if not files:
        return 0
    MATURE_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    cols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [files]).description]
    quote_filter = "AND (quote_mint IS NULL OR quote_mint = 'So11111111111111111111111111111111111111112')" if "quote_mint" in cols else ""
    pool_sel = "pool_id" if "pool_id" in cols else "NULL AS pool_id"
    # sane SOL pools only (reserve band), one price scale per mint per day (within 50x of the day's median), dominant pool when known
    con.execute(f"""CREATE TEMP TABLE raw AS SELECT mint, ts, slot, trader, side, sol, price, quote_in_pool, {pool_sel}
                    FROM read_parquet(?, union_by_name = true)
                    WHERE pool = 'pump-amm' AND price > 0 AND mint LIKE '%pump' AND quote_in_pool BETWEEN 0.001 AND 100000 {quote_filter}""", [files])
    con.execute("""CREATE TEMP TABLE med AS SELECT mint, median(price) AS pmed FROM raw GROUP BY mint""")
    if "pool_id" in cols:
        con.execute("""CREATE TEMP TABLE dom AS SELECT mint, arg_max(pool_id, n) AS pool_id FROM (SELECT mint, pool_id, count(*) AS n FROM raw GROUP BY mint, pool_id) GROUP BY mint""")
        con.execute("""CREATE TEMP TABLE clean AS SELECT r.* FROM raw r JOIN med USING (mint) JOIN dom USING (mint) WHERE r.price BETWEEN med.pmed / 50 AND med.pmed * 50 AND (r.pool_id IS NULL OR r.pool_id = dom.pool_id)""")
    else:
        con.execute("""CREATE TEMP TABLE clean AS SELECT r.* FROM raw r JOIN med USING (mint) WHERE r.price BETWEEN med.pmed / 50 AND med.pmed * 50""")
    tab = con.execute("""
        SELECT mint, time_bucket(INTERVAL 1 MINUTE, ts) AS ts,
               first(price ORDER BY ts, slot) AS open, max(price) AS high, min(price) AS low, last(price ORDER BY ts, slot) AS close,
               sum(CASE WHEN side = 1 THEN sol ELSE 0 END) AS buy_sol, sum(CASE WHEN side = -1 THEN sol ELSE 0 END) AS sell_sol,
               count(*) FILTER (WHERE side = 1) AS n_buys, count(*) FILTER (WHERE side = -1) AS n_sells,
               count(DISTINCT trader) AS n_traders, last(quote_in_pool ORDER BY ts, slot) AS resq_sol
        FROM clean GROUP BY mint, time_bucket(INTERVAL 1 MINUTE, ts) ORDER BY mint, ts""").fetch_arrow_table()
    con.close()
    pq.write_table(tab, MATURE_DIR / f"{d.isoformat()}.parquet", compression="zstd")
    n_mints = len(pa.compute.unique(tab["mint"]))
    with transaction() as conn:
        conn.execute("INSERT INTO mature_days (day, mints, rows, took_s) VALUES (%s,%s,%s,%s) ON CONFLICT (day) DO UPDATE SET mints = EXCLUDED.mints, rows = EXCLUDED.rows, took_s = EXCLUDED.took_s, built_at = now()",
                     (d, n_mints, tab.num_rows, time.time() - t0))
    log.info("mature %s: %d mints, %d minute rows in %.0fs", d, n_mints, tab.num_rows, time.time() - t0)
    return tab.num_rows


def _epoch_s(col: pd.Series):
    return ((col - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def build_day(d: date, lookback_days: int = 1) -> int:
    """Feature rows for every minute of day D, warmed up on the previous day's candles."""
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
    with transaction() as conn:
        meta = {r["mint"]: r for r in conn.execute("SELECT m.mint, t.graduated_at, m.create_ts FROM corpus_meta m JOIN corpus_tokens t USING (mint)").fetchall()}
    out: list[dict] = []; n_mints = 0
    for mint, x in cd.groupby("mint", sort=False):
        ts_s = _epoch_s(x["ts"]); 
        if not (ts_s >= day_start).any():
            continue
        n_mints += 1
        g = meta[mint]["graduated_at"].timestamp() if mint in meta and meta[mint]["graduated_at"] else None
        st = TokenState(mint); tm = TokenMeta(mint=mint, program_label="Pump.fun Amm", graduated_at=g)
        pre = {k: float("nan") for k in PRE_COLS}
        o, h, l, c = x["open"].to_numpy(), x["high"].to_numpy(), x["low"].to_numpy(), x["close"].to_numpy()
        bs, ss, nb, ns, nt, rq = x["buy_sol"].to_numpy(), x["sell_sol"].to_numpy(), x["n_buys"].to_numpy(), x["n_sells"].to_numpy(), x["n_traders"].to_numpy(), x["resq_sol"].to_numpy()
        hist_ts, hist_n, hist_tr = [], [], []
        for i in range(len(ts_s)):
            t_end = float(ts_s[i]) + 60.0; price = float(c[i]); resq = float(rq[i]) if np.isfinite(rq[i]) and rq[i] > 0 else None
            if bs[i] > 0:
                st.append(t_end - 2e-3, price, float(bs[i]), True, None, resq)
            if ss[i] > 0 or bs[i] <= 0:
                st.append(t_end - 1e-3, price, float(ss[i]), False, None, resq)
            hist_ts.append(t_end); hist_n.append(float(nb[i] + ns[i])); hist_tr.append(float(nt[i]))
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
            if g is None:
                row["age_h"] = float("nan"); row["t_rel_min"] = -1; row["phase"] = "amm"
            out.append(row)
    if not out:
        return 0
    dd = MATURE_FEAT_DIR / d.isoformat(); dd.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(out, schema=SCHEMA), dd / "part.parquet", compression="zstd")
    log.info("mature features %s: %d mints, %d rows in %.0fs", d, n_mints, len(out), time.time() - t0)
    return len(out)


def loop_once() -> int:
    n = 0
    for d in days_ready():
        aggregate_day(d); n += 1
    # features for days whose aggregate and previous day's aggregate exist and no feature part yet
    for f in sorted(MATURE_DIR.glob("*.parquet"), reverse=True):
        d = date.fromisoformat(f.stem)
        if (MATURE_FEAT_DIR / d.isoformat() / "part.parquet").exists() or not (MATURE_DIR / f"{(d - timedelta(days=1)).isoformat()}.parquet").exists():
            continue
        build_day(d); n += 1
    return n


def main() -> None:
    from ..logging_setup import setup
    setup("replay")
    print(f"processed {loop_once()} day steps")
