"""Creation-time facts and point-in-time creator history for every graduated token (table ``corpus_meta``).

From the replay lifecycle rows: the ``create`` event (creator wallet, the creator's own first buy in SOL and tokens,
supply, mayhem flag, metadata URI) and the ``migrate`` event (graduation time, pool id, real quote reserve at
migration). Derived: ``ttg_min`` (minutes from creation to graduation; instant graduations are bundled launches),
``dev_share`` (creator's first buy as a share of supply).

Creator history is strictly point-in-time as of the token's own graduation: ``prior_launches`` (the creator's
earlier creates), ``prior_grads`` (the creator's earlier graduations), and among those with a known outcome,
``prior_rug_share`` (fell 90 % within 60 min of graduation) and ``prior_moon_share`` (doubled within 60 min).
The token's own outcome columns (``own_*``) exist only to feed other tokens' history; never use them as features.

``creator_history()`` is the same definition for one token, used live by ``ingest/pumpstream.py`` at graduation; it
reads the creator's creates from ``pump_events`` (the archive's creates are backfilled there by ``rebuild``, the stream
adds new ones) and earlier graduations from this table (``graduated_at``, outcomes known 60 min after graduation —
the stream fills ``own_dd60``/``own_max60`` from its own minutes at that time, the archive rebuild later replaces them).

``rebuild()`` recomputes the whole table from the archive (events are small) and fills outcomes incrementally from
the assembled candle files. Runs inside the ``replay`` worker after each assembly round and as
``fly-trader build-corpus-meta``.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction

log = logging.getLogger(__name__)
FEATURE_COLS = ["ttg_min", "dev_sol", "dev_share", "mayhem", "rq0", "prior_launches", "prior_grads", "prior_known", "prior_rug_share", "prior_moon_share"]


def _outcomes(rows: list[dict]) -> list[tuple]:
    """(mint, dd60, max60, alive6h) from a token's candle file (post-graduation 1m candles)."""
    out = []
    for r in rows:
        try:
            t = pq.read_table(r["candle_path"], columns=["ts", "interval", "close"]).to_pandas()
        except Exception:
            continue
        g = r["graduated_at"]; t = t[t["ts"] >= g].sort_values("ts")
        if t.empty:
            out.append((r["mint"], None, None, False)); continue
        p0 = float(t["close"].iloc[0]); w = t[t["ts"] < g + pd.Timedelta(minutes=60)]
        dd = float(w["close"].min() / p0 - 1) if len(w) else None; mx = float(w["close"].max() / p0 - 1) if len(w) else None
        alive = bool((t["ts"].max() - g) >= pd.Timedelta(hours=6))
        out.append((r["mint"], dd, mx, alive))
    return out


def rebuild(outcome_batch: int = 20000) -> int:
    t0 = time.time(); con = duckdb.connect(); ev = str(config.REPLAY_DIR / "*" / "*_events.parquet")
    if not list(config.REPLAY_DIR.glob("*/*_events.parquet")):
        return 0
    creates = con.execute(f"""SELECT mint, min(ts) AS create_ts, arg_min(signer, ts) AS creator, arg_min(quote_amount, ts) AS dev_sol,
                              arg_min(initial_buy, ts) AS dev_tokens, arg_min(supply, ts) AS supply, arg_min(mayhem, ts) AS mayhem, arg_min(uri, ts) AS uri,
                              arg_min(name, ts) AS name, arg_min(symbol, ts) AS symbol, arg_min(sig, ts) AS sig FROM read_parquet('{ev}') WHERE action = 'create' AND pool = 'pump' GROUP BY mint""").df()
    migr = con.execute(f"""SELECT mint, min(ts) AS g, arg_min(quote_in_pool, ts) AS rq0, arg_min(pool_id, ts) AS pool_id
                           FROM read_parquet('{ev}') WHERE action = 'migrate' AND pool = 'pump-amm' GROUP BY mint""").df()
    con.close()
    meta = migr.merge(creates, on="mint", how="left")
    meta["ttg_min"] = (meta["g"] - meta["create_ts"]).dt.total_seconds() / 60.0
    meta["dev_share"] = meta["dev_tokens"] / meta["supply"]
    # outcomes: existing ones from the table, new ones from candle files
    with transaction() as conn:
        known = pd.DataFrame(conn.execute("SELECT mint, own_dd60, own_max60, own_alive6h FROM corpus_meta WHERE own_alive6h IS NOT NULL OR own_dd60 IS NOT NULL").fetchall())
        todo = conn.execute("SELECT t.mint, t.graduated_at, t.candle_path FROM corpus_tokens t LEFT JOIN corpus_meta m USING (mint) "
                            "WHERE t.candle_path IS NOT NULL AND t.status = 'done' AND (m.mint IS NULL OR m.own_alive6h IS NULL) ORDER BY t.graduated_at DESC LIMIT %s", (outcome_batch,)).fetchall()
    new = pd.DataFrame(_outcomes(todo), columns=["mint", "own_dd60", "own_max60", "own_alive6h"]) if todo else pd.DataFrame(columns=["mint", "own_dd60", "own_max60", "own_alive6h"])
    # candle-file outcomes win over stream-computed ones; stream values survive until the day is assembled
    outcomes = pd.concat([new, known], ignore_index=True).drop_duplicates("mint") if len(known) or len(new) else new
    meta = meta.merge(outcomes, on="mint", how="left")
    # point-in-time creator history (as of this token's graduation)
    cr = creates.sort_values("create_ts").copy(); cr["prior_launches"] = cr.groupby("creator").cumcount()
    meta = meta.merge(cr[["mint", "prior_launches"]], on="mint", how="left")
    hist = meta.dropna(subset=["creator"]).sort_values("g").copy()
    hist["rug"] = (hist["own_dd60"] <= -0.9).astype(float).where(hist["own_dd60"].notna())
    hist["moon"] = (hist["own_max60"] >= 1.0).astype(float).where(hist["own_max60"].notna())
    hist["prior_grads"] = hist.groupby("creator").cumcount()
    # outcomes of an earlier graduation A are known 60 min after g_A; B may only count A when g_A + 60 min <= g_B (no look-ahead)
    known = np.zeros(len(hist), int); rugs = np.zeros(len(hist)); moons = np.zeros(len(hist))
    g_s = (hist["g"] - pd.Timestamp(0, tz="UTC")).dt.total_seconds().to_numpy(); rug_v = hist["rug"].to_numpy(); moon_v = hist["moon"].to_numpy()
    for _, idx in hist.groupby("creator").indices.items():
        idx = np.asarray(idx); gs = g_s[idx]; order = np.argsort(gs, kind="stable"); idx = idx[order]; gs = gs[order]
        j = 0; k_cnt = 0; r_sum = 0.0; m_sum = 0.0
        for b in range(len(idx)):
            while j < b and gs[j] + 3600.0 <= gs[b]:
                if not np.isnan(rug_v[idx[j]]):
                    k_cnt += 1; r_sum += rug_v[idx[j]]; m_sum += moon_v[idx[j]]
                j += 1
            known[idx[b]] = k_cnt; rugs[idx[b]] = r_sum; moons[idx[b]] = m_sum
    hist["prior_known"] = known
    hist["prior_rug_share"] = np.where(known > 0, rugs / np.maximum(known, 1), np.nan)
    hist["prior_moon_share"] = np.where(known > 0, moons / np.maximum(known, 1), np.nan)
    meta = meta.merge(hist[["mint", "prior_grads", "prior_known", "prior_rug_share", "prior_moon_share"]], on="mint", how="left")
    meta["graduated_at"] = meta["g"]
    cols = ["mint", "graduated_at", "create_ts", "creator", "dev_sol", "dev_tokens", "dev_share", "supply", "mayhem", "uri", "name", "symbol", "ttg_min", "rq0", "pool_id",
            "prior_launches", "prior_grads", "prior_known", "prior_rug_share", "prior_moon_share", "own_dd60", "own_max60", "own_alive6h"]
    def _v(x):
        if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NaT:
            return None
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.bool_,)):
            return bool(x)
        if isinstance(x, pd.Timestamp):
            return x.to_pydatetime()
        return x
    rows = [tuple(_v(v) for v in rec) for rec in meta[cols].itertuples(index=False, name=None)]
    with transaction() as conn:
        conn.cursor().executemany(
            "INSERT INTO corpus_meta (" + ",".join(cols) + ") VALUES (" + ",".join(["%s"] * len(cols)) + ") ON CONFLICT (mint) DO UPDATE SET " +
            ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "mint") + ", updated_at = now()", rows)
    n_cr = _backfill_creates(creates, _v)
    log.info("corpus_meta rebuilt: %d graduated tokens (%d with creator, %d with outcomes), %d archive creates added to pump_events, in %.0fs",
             len(meta), int(meta["creator"].notna().sum()), int(meta["own_dd60"].notna().sum()), n_cr, time.time() - t0)
    return len(meta)


def _backfill_creates(creates: pd.DataFrame, _v) -> int:
    """Copy the archive's creates into ``pump_events`` (newer than the last backfill), so live creator lookups count the
    same launches training counts."""
    import json
    from datetime import datetime
    with transaction() as conn:
        r = conn.execute("SELECT value->>'through' AS t FROM ui_settings WHERE key = 'pump_events_backfill'").fetchone()
    wm = pd.Timestamp(r["t"]) if r and r["t"] else None
    new = creates[creates["create_ts"] > wm] if wm is not None else creates
    new = new[new["sig"].notna()]
    if new.empty:
        return 0
    cols = ["sig", "create_ts", "mint", "creator", "dev_sol", "dev_tokens", "supply", "mayhem", "name", "symbol", "uri"]
    with transaction() as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE TEMP TABLE _cr (sig text, ts timestamptz, mint text, signer text, dev_sol float8, dev_tokens float8, supply float8, "
                        "mayhem boolean, name text, symbol text, uri text) ON COMMIT DROP")
            with cur.copy("COPY _cr FROM STDIN") as cp:
                for rec in new[cols].itertuples(index=False, name=None):
                    cp.write_row(tuple(_v(x) for x in rec))
            cur.execute("INSERT INTO pump_events (sig, ts, action, pool, mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri) "
                        "SELECT sig, ts, 'create', 'pump', mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri FROM _cr ON CONFLICT (sig) DO NOTHING")
            n = cur.rowcount
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('pump_events_backfill', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps({"through": pd.Timestamp(new["create_ts"].max()).isoformat(), "at": datetime.now().isoformat()}),))
    return int(n)


def creator_history(conn, creator: str | None, create_ts, graduated_at) -> dict:
    """Point-in-time creator history for one token, the same definitions ``rebuild`` computes in bulk:
    prior_launches = the creator's creates before this token's creation; prior_grads = the creator's graduations before
    this graduation; prior_known / rug / moon share = among those, the ones whose 60-minute outcome was known by now."""
    out = {"prior_launches": None, "prior_grads": None, "prior_known": None, "prior_rug_share": None, "prior_moon_share": None}
    if not creator:
        return out
    if create_ts is not None:
        out["prior_launches"] = int(conn.execute("SELECT count(*) AS n FROM pump_events WHERE signer = %s AND action = 'create' AND ts < %s",
                                                 (creator, create_ts)).fetchone()["n"])
    r = conn.execute("""SELECT count(*) AS grads,
                                count(*) FILTER (WHERE own_dd60 IS NOT NULL AND graduated_at + interval '60 minutes' <= %(g)s) AS known,
                                avg(CASE WHEN own_dd60 <= -0.9 THEN 1.0 ELSE 0.0 END) FILTER (WHERE own_dd60 IS NOT NULL AND graduated_at + interval '60 minutes' <= %(g)s) AS rug,
                                avg(CASE WHEN own_max60 >= 1.0 THEN 1.0 ELSE 0.0 END) FILTER (WHERE own_dd60 IS NOT NULL AND graduated_at + interval '60 minutes' <= %(g)s) AS moon
                         FROM corpus_meta WHERE creator = %(c)s AND graduated_at < %(g)s""", {"c": creator, "g": graduated_at}).fetchone()
    out["prior_grads"] = int(r["grads"]); out["prior_known"] = int(r["known"])
    out["prior_rug_share"] = float(r["rug"]) if r["rug"] is not None else None
    out["prior_moon_share"] = float(r["moon"]) if r["moon"] is not None else None
    return out


def load_features() -> pd.DataFrame:
    with transaction() as conn:
        rows = conn.execute("SELECT mint, " + ", ".join(FEATURE_COLS) + " FROM corpus_meta").fetchall()
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=["mint"] + FEATURE_COLS)
    df["mayhem"] = df["mayhem"].map({True: 1.0, False: 0.0})
    return df


def main() -> None:
    from ..logging_setup import setup
    setup("replay")
    prev = None
    while True:
        n = rebuild()
        with transaction() as conn:
            left = conn.execute("SELECT count(*) AS n FROM corpus_tokens t LEFT JOIN corpus_meta m USING (mint) WHERE t.candle_path IS NOT NULL AND t.status = 'done' AND (m.mint IS NULL OR m.own_alive6h IS NULL)").fetchone()["n"]
        print(f"corpus_meta: {n} tokens; outcomes still missing for {left}")
        if not left or left == prev:
            break
        prev = left
