"""Wallet skill on Robinhood Chain: train/wallet_skill's definitions over RH wallets only (a Solana wallet and an RH wallet
are different people; the tables never mix), under the same frozen (h, L) — so ``skill_top*_15m`` means the same thing on
both chains' rows.

``build_wallet_day(D)``: each Pons swap leg of day D (trader, ETH amount, ETH price) marked out over every hold like the
Solana legs (the close of the last minute at or before t + h, after the one-side exit cost at both ends, clipped ±100 %)
→ data/corpus/rh/wallet_day/<D>.parquet. ``write_table(D)`` → data/corpus/rh/wallet_skill/<D>.parquet from the wallet days
ending GAP_DAYS before D (train/wallet_skill.skill_table with the RH directories).
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..train import wallet_skill as ws

log = logging.getLogger(__name__)
ROOT = config.CORPUS_DIR / config.RH_DIR_NAME
DAY_DIR = ROOT / "wallet_day"
SKILL_DIR = ROOT / "wallet_skill"
MATURE_DIR = ROOT / "mature"
FEAT_DIR = ROOT / "features_mature"


def skill_version() -> str:
    return "rh-" + ws.skill_version()


def legs(conn, d: date) -> pd.DataFrame:
    t0 = datetime(d.year, d.month, d.day, tzinfo=timezone.utc); t1 = t0 + timedelta(days=1)
    rows = conn.execute(
        "SELECT s.token AS mint, s.trader, s.ts, s.side, s.quote_raw, a.decimals AS qdec, s.price_q, s.pool_id, p.quote_asset "
        "FROM rh_swaps s JOIN rh_pools p ON p.pool_id = s.pool_id JOIN rh_assets a ON a.asset = p.quote_asset "
        "WHERE s.ts >= %s AND s.ts < %s AND s.trader IS NOT NULL AND s.price_q > 0", (t0, t1)).fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    from . import prices
    base: dict[tuple, float | None] = {}
    def b(asset, ts):
        k = (asset, int(ts.timestamp() // 60))
        if k not in base:
            base[k] = prices.base_eth(conn, asset, ts.replace(second=59, microsecond=0))[0]
        return base[k]
    df["base"] = [b(a, t) for a, t in zip(df["quote_asset"], df["ts"])]
    df = df[df["base"].notna()].copy()
    df["sol"] = df["quote_raw"].astype(float) / (10.0 ** df["qdec"].astype(float)) * df["base"]
    df["price"] = df["price_q"].astype(float) * df["base"]
    return df[["mint", "trader", "ts", "side", "sol", "price"]]


def build_wallet_day(d: date) -> int:
    nxt = d + timedelta(days=1)
    aggs = [MATURE_DIR / f"{x.isoformat()}.parquet" for x in (d, nxt)]; parts = [FEAT_DIR / x.isoformat() / "part.parquet" for x in (d, nxt)]
    if not all(p.exists() for p in aggs + parts):
        return 0
    with transaction() as conn:
        lg = legs(conn, d)
    if lg.empty:
        return 0
    con = duckdb.connect(); con.register("legs_df", lg)
    con.execute("CREATE TEMP TABLE legs AS SELECT * FROM legs_df")
    con.execute("CREATE TEMP TABLE closes AS SELECT mint, ts AS mts, close FROM read_parquet(?) ORDER BY mint, mts", [[str(p) for p in aggs]])
    con.execute("CREATE TEMP TABLE ec AS SELECT mint, ts AS mts, exit_cost_0p1 AS ec FROM read_parquet(?)", [[str(p) for p in parts]])
    out = []
    for h in ws.HOLDS_MIN:
        out.append(con.execute(f"""
            SELECT l.trader AS wallet, {h} AS h, count(*) AS n, sum(l.sol) AS sol,
                   sum(l.sol * greatest(-1.0, least(1.0, CASE WHEN l.side = 1
                        THEN c.close / l.price * (1 - coalesce(e0.ec, 0)) * (1 - coalesce(e1.ec, 0)) - 1
                        ELSE -(c.close / l.price - 1) END))) AS sol_m
            FROM legs l ASOF JOIN closes c ON c.mint = l.mint AND c.mts <= l.ts + INTERVAL {h} MINUTE
            LEFT JOIN ec e0 ON e0.mint = l.mint AND e0.mts = time_bucket(INTERVAL 1 MINUTE, l.ts)
            LEFT JOIN ec e1 ON e1.mint = c.mint AND e1.mts = c.mts
            GROUP BY l.trader""").fetch_arrow_table())
    con.close()
    tab = pa.concat_tables(out); DAY_DIR.mkdir(parents=True, exist_ok=True)
    tmp = DAY_DIR / f"{d.isoformat()}.parquet.tmp"; pq.write_table(tab, tmp, compression="zstd"); tmp.replace(DAY_DIR / f"{d.isoformat()}.parquet")
    return tab.num_rows


def write_table(d: date) -> bool:
    c = ws.skill_config()
    if not c:
        return False
    return ws.write_table(d, int(c["h"]), int(c["L"]), out_dir=SKILL_DIR, day_dir=DAY_DIR)


def table_for(d: date) -> dict | None:
    f = SKILL_DIR / f"{d.isoformat()}.parquet"
    if not f.exists():
        return None
    t = pq.read_table(f, columns=["wallet", "bucket"])
    return dict(zip(t.column("wallet").to_pylist(), t.column("bucket").to_pylist()))


_TODAY: dict = {"key": None, "table": None}


def load_today() -> dict | None:
    """Live: the RH table in force today (wallet → decile), read once per file. Like the Solana feed
    (ingest/pumpstream.load_skill), a missing table for today falls back to the newest one on disk: without one every
    minute's skill inputs read as zero and the selector fails closed on every RH row (2026-10-05..07: no RH trade)."""
    today = datetime.now(timezone.utc).date()
    f = SKILL_DIR / f"{today.isoformat()}.parquet"
    if not f.exists():
        have = sorted(SKILL_DIR.glob("????-??-??.parquet")) if SKILL_DIR.exists() else []
        if not have:
            return None
        f = have[-1]
        log.warning("no RH wallet skill table for %s: using the newest on disk, %s", today, f.stem)
    key = (str(f), f.stat().st_mtime)
    if _TODAY["key"] != key:
        t = pq.read_table(f, columns=["wallet", "bucket"])
        _TODAY["table"] = dict(zip(t.column("wallet").to_pylist(), t.column("bucket").to_pylist())); _TODAY["key"] = key
        log.info("RH wallet skill table %s loaded: %d wallets", f.stem, len(_TODAY["table"]))
    return _TODAY["table"]


def window_ready(d: date, L: int) -> bool:
    """Every RH wallet day of D's window exists (days before the first RH wallet day do not count)."""
    have = sorted(DAY_DIR.glob("????-??-??.parquet")) if DAY_DIR.exists() else []
    if not have:
        return False
    first = date.fromisoformat(have[0].stem)
    need = [x for x in ws.window_days(d, L) if x >= first]
    return bool(need) and all((DAY_DIR / f"{x.isoformat()}.parquet").exists() for x in need)


def daily(through: date | None = None) -> int:
    """The table of every day from the first RH wallet day through tomorrow (UTC) once its window exists — written before
    the day begins, as Solana's (train/wallet_skill.daily), so the live minutes and the corpus read the same file. The
    window ends GAP_DAYS before the day, so writing early and writing at corpus time give the same table."""
    c = ws.skill_config()
    have = sorted(DAY_DIR.glob("????-??-??.parquet")) if DAY_DIR.exists() else []
    if not c or not have:
        return 0
    v = f"h{int(c['h'])}-L{int(c['L'])}"; n = 0
    d = date.fromisoformat(have[0].stem) + timedelta(days=1)
    last = through or (datetime.now(timezone.utc).date() + timedelta(days=1))
    while d <= last:
        f = SKILL_DIR / f"{d.isoformat()}.parquet"
        if (not f.exists() or ws.table_version(f) != v) and window_ready(d, int(c["L"])) and write_table(d):
            n += 1
        d += timedelta(days=1)
    return n


def path_of(d: date) -> Path:
    return SKILL_DIR / f"{d.isoformat()}.parquet"
