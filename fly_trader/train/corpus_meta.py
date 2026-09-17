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
``fly-trader build-corpus-meta``. It first copies the archive's directly created PumpSwap pools into ``pump_pools``
(``backfill_pump_pools``), which the stream and ``train/mature.py`` exclude from the universe (``blocked_pool_ids``).
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
CURVE_COLS = ["bundle_share", "dev_hold_share", "dev_sold_frac", "grad_hhi"]
FEATURE_COLS = ["ttg_min", "dev_sol", "dev_share", "mayhem", "rq0", "prior_launches", "prior_grads", "prior_known", "prior_rug_share", "prior_moon_share",
                *CURVE_COLS, "curve_known"]


def curve_facts(create_slot: int | None, creator: str | None, supply: float | None, initial_buy: float | None, wallets: dict) -> tuple[dict, list[tuple[str, str]]]:
    """Graduation-time rug facts from a token's bonding-curve trading, one definition for the archive (``update_curve_facts``)
    and the live stream (``ingest/pumpstream.py``). ``wallets``: wallet → [first buy slot or None, tokens bought, tokens sold]
    over the curve legs up to the migration. The creator's first buy rides on the create event (``initial_buy``, not a
    trade leg — measured 2026-09-15: a creator leg in the create slot occurs only when the create carried no initial buy).
    - bundle_share: tokens held at migration by non-creator wallets whose first buy is in the create slot (same block as
      the creation: bought together with it) / supply;
    - dev_hold_share: the creator's held tokens (initial buy + curve buys − sells) / supply;
    - dev_sold_frac: the creator's sold tokens / its bought tokens (initial buy included);
    - grad_hhi: Herfindahl concentration of holdings at migration (Σ of squared shares of all positive holdings).
    Returns (facts, insider wallets [(wallet, kind)]: the creator and bundle wallets)."""
    sup = float(supply) if supply else 0.0; ib = float(initial_buy or 0.0)
    held, bundle, insiders = [], 0.0, []
    dev_b, dev_s = ib, 0.0
    for w, (first, bought, sold) in wallets.items():
        net = float(bought) - float(sold)
        if w == creator:
            dev_b += float(bought); dev_s += float(sold)
            continue
        if net > 0:
            held.append(net)
        if first is not None and create_slot is not None and int(first) == int(create_slot):
            bundle += max(net, 0.0); insiders.append((w, "bundle"))
    dev_net = dev_b - dev_s
    if creator:
        insiders.insert(0, (creator, "creator"))
        if dev_net > 0:
            held.append(dev_net)
    tot = sum(held)
    facts = {"bundle_share": bundle / sup if sup > 0 else None, "dev_hold_share": max(dev_net, 0.0) / sup if sup > 0 else None,
             "dev_sold_frac": dev_s / dev_b if dev_b > 0 else 0.0, "grad_hhi": sum((h / tot) ** 2 for h in held) if tot > 0 else None}
    return facts, insiders


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
    n_pp = backfill_pump_pools()               # before mature.loop_once aggregates the new days
    creates =con.execute(f"""SELECT mint, min(ts) AS create_ts, arg_min(signer, ts) AS creator, arg_min(quote_amount, ts) AS dev_sol,
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
    rows = [tuple(_pg(v) for v in rec) for rec in meta[cols].itertuples(index=False, name=None)]
    with transaction() as conn:
        conn.cursor().executemany(
            "INSERT INTO corpus_meta (" + ",".join(cols) + ") VALUES (" + ",".join(["%s"] * len(cols)) + ") ON CONFLICT (mint) DO UPDATE SET " +
            ", ".join(f"{c} = EXCLUDED.{c}" for c in cols if c != "mint") + ", updated_at = now()", rows)
    n_cr = _backfill_creates(creates)
    n_curve = update_curve_facts()
    log.info("corpus_meta rebuilt: %d graduated tokens (%d with creator, %d with outcomes), %d archive creates added to pump_events, %d custom pools to pump_pools, "
             "curve facts for %d, in %.0fs", len(meta), int(meta["creator"].notna().sum()), int(meta["own_dd60"].notna().sum()), n_cr, n_pp, n_curve, time.time() - t0)
    return len(meta)


def update_curve_facts() -> int:
    """Curve facts and insiders (``curve_facts``) for graduated tokens whose whole curve life (create hour .. graduation
    hour) is in the archive and that have none yet; the archive's values replace the stream's (COALESCEd at graduation)."""
    from datetime import timedelta
    with transaction() as conn:
        todo = conn.execute("SELECT mint, create_ts, graduated_at, creator, supply, dev_tokens FROM corpus_meta "
                            "WHERE curve_known IS NOT TRUE AND create_ts IS NOT NULL AND graduated_at IS NOT NULL").fetchall()
        done = {r["hour"].astimezone(__import__("datetime").timezone.utc) for r in conn.execute("SELECT hour FROM replay_hours WHERE status IN ('done','missing')").fetchall()}
    ready = []
    for r in todo:
        h = r["create_ts"].replace(minute=0, second=0, microsecond=0); g = r["graduated_at"]
        ok = True
        while h <= g:
            if h not in done:
                ok = False; break
            h += timedelta(hours=1)
        if ok:
            ready.append(r)
    if not ready:
        return 0
    days = sorted({(r["create_ts"] + timedelta(hours=k)).date() for r in ready for k in range(0, int((r["graduated_at"] - r["create_ts"]).total_seconds() // 3600) + 2)})
    trade_files = [str(f) for d in days for f in sorted((config.REPLAY_DIR / d.isoformat()).glob("*_trades.parquet"))]
    event_files = [str(f) for d in days for f in sorted((config.REPLAY_DIR / d.isoformat()).glob("*_events.parquet"))]
    if not trade_files:
        return 0
    import pyarrow as pa
    con = duckdb.connect()
    con.register("todo", pa.table({"mint": pa.array([r["mint"] for r in ready], pa.string()),
                                   "g": pa.array([r["graduated_at"] for r in ready], pa.timestamp("us", tz="UTC"))}))
    slots = dict(con.execute("SELECT e.mint, arg_min(e.slot, e.ts) FROM read_parquet(?, union_by_name = true) e JOIN todo USING (mint) "
                             "WHERE e.action = 'create' AND e.pool = 'pump' GROUP BY e.mint", [event_files]).fetchall()) if event_files else {}
    legs = con.execute("""SELECT t.mint, t.trader, min(t.slot) FILTER (WHERE t.side = 1) AS first_buy,
                                 coalesce(sum(t.tokens) FILTER (WHERE t.side = 1), 0) AS bought, coalesce(sum(t.tokens) FILTER (WHERE t.side = -1), 0) AS sold
                          FROM read_parquet(?, union_by_name = true) t JOIN todo d ON d.mint = t.mint
                          WHERE t.pool = 'pump' AND t.ts <= d.g AND t.trader IS NOT NULL GROUP BY t.mint, t.trader""", [trade_files]).df()
    con.close()
    by_mint = {m: g for m, g in legs.groupby("mint")} if len(legs) else {}
    upd, ins = [], []
    for r in ready:
        g = by_mint.get(r["mint"])
        wallets = {} if g is None else {w: [None if pd.isna(f) else int(f), float(b), float(s_)] for w, f, b, s_ in zip(g["trader"], g["first_buy"], g["bought"], g["sold"])}
        facts, insiders = curve_facts(slots.get(r["mint"]), r["creator"], r["supply"], r["dev_tokens"], wallets)
        upd.append((*[_pg(facts[c]) for c in CURVE_COLS], r["mint"])); ins.extend((r["mint"], w, k) for w, k in insiders)
    with transaction() as conn:
        with conn.cursor() as cur:
            cur.executemany("UPDATE corpus_meta SET " + ", ".join(f"{c} = %s" for c in CURVE_COLS) + ", curve_known = true, updated_at = now() WHERE mint = %s", upd)
            cur.execute("CREATE TEMP TABLE _ti (mint text, wallet text, kind text) ON COMMIT DROP")
            with cur.copy("COPY _ti FROM STDIN") as cp:
                for rec in ins:
                    cp.write_row(rec)
            cur.execute("INSERT INTO token_insiders (mint, wallet, kind) SELECT DISTINCT ON (mint, wallet) mint, wallet, kind FROM _ti ON CONFLICT (mint, wallet) DO NOTHING")
    log.info("curve facts: %d tokens, %d insider wallets", len(upd), len(ins))
    return len(upd)


def _pg(x):
    """A pandas/numpy value as a Postgres parameter (text cannot hold NUL bytes; token names sometimes do)."""
    if x is None or (isinstance(x, float) and np.isnan(x)) or x is pd.NaT:
        return None
    if isinstance(x, str):
        return x.replace("\x00", "")
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, pd.Timestamp):
        return x.to_pydatetime()
    return x


def _complete_days(conn) -> set[str]:
    """UTC days whose 24 replay hours are all ingested (their creates can no longer change)."""
    rows = conn.execute("SELECT (hour AT TIME ZONE 'UTC')::date AS d, count(*) AS n FROM replay_hours WHERE status IN ('done','missing') GROUP BY 1").fetchall()
    return {r["d"].isoformat() for r in rows if r["n"] >= 24}


def _backfill_creates(creates: pd.DataFrame) -> int:
    """Copy the archive's creates into ``pump_events``, so live creator lookups count the same launches training counts.
    Replay ingests hours newest first, so a time watermark would skip older hours that arrive later; instead every
    create day not yet recorded as complete is sent (duplicates are no-ops) and complete days are recorded."""
    import json
    from datetime import datetime
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'pump_events_backfill'").fetchone()
        complete = _complete_days(conn)
    done = set(((r or {}).get("value") or {}).get("days") or [])
    day = pd.to_datetime(creates["create_ts"], utc=True).dt.strftime("%Y-%m-%d")
    new = creates[~day.isin(done) & creates["sig"].notna()]
    n = 0
    if not new.empty:
        cols = ["sig", "create_ts", "mint", "creator", "dev_sol", "dev_tokens", "supply", "mayhem", "name", "symbol", "uri"]
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE TEMP TABLE _cr (sig text, ts timestamptz, mint text, signer text, dev_sol float8, dev_tokens float8, supply float8, "
                            "mayhem boolean, name text, symbol text, uri text) ON COMMIT DROP")
                with cur.copy("COPY _cr FROM STDIN") as cp:
                    for rec in new[cols].itertuples(index=False, name=None):
                        cp.write_row(tuple(_pg(x) for x in rec))
                cur.execute("INSERT INTO pump_events (sig, ts, action, pool, mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri) "
                            "SELECT sig, ts, 'create', 'pump', mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri FROM _cr ON CONFLICT (sig) DO NOTHING")
                n = cur.rowcount
    done |= set(day.unique()) & complete
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('pump_events_backfill', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps({"days": sorted(done), "at": datetime.now().isoformat()}),))
    return int(n)


def backfill_pump_pools() -> int:
    """Copy the archive's directly created PumpSwap pools (``createPool`` on ``pump-amm`` not by a pump.fun migration;
    missing ``pool_created_by`` counts as custom) into ``pump_pools``. Only day directories not yet recorded as complete
    are read (duplicates are no-ops), so a re-run costs the newest day's files."""
    import json
    from datetime import datetime
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'pump_pools_backfill'").fetchone()
        complete = _complete_days(conn)
    done = set(((r or {}).get("value") or {}).get("days") or [])
    days = {p.name: sorted(str(f) for f in p.glob("*_events.parquet")) for p in config.REPLAY_DIR.glob("*") if p.is_dir() and p.name not in done}
    days = {d: fs for d, fs in days.items() if fs}
    if not days:
        return 0
    files = [f for fs in days.values() for f in fs]; con = duckdb.connect()
    cols = [c[0] for c in con.execute("SELECT * FROM read_parquet(?, union_by_name = true) LIMIT 0", [files]).description]
    by = "pool_created_by" if "pool_created_by" in cols else "NULL::VARCHAR"
    pools = con.execute(f"""SELECT pool_id, arg_min(mint, ts) AS mint, arg_min({by}, ts) AS created_by, min(ts) AS ts FROM read_parquet(?, union_by_name = true)
                            WHERE action = 'createPool' AND pool = 'pump-amm' AND coalesce({by}, 'custom') <> 'pump' AND pool_id IS NOT NULL GROUP BY pool_id""", [files]).df()
    con.close(); n = 0
    if not pools.empty:
        with transaction() as conn:
            with conn.cursor() as cur:
                cur.execute("CREATE TEMP TABLE _pp (pool_id text, mint text, created_by text, ts timestamptz) ON COMMIT DROP")
                with cur.copy("COPY _pp FROM STDIN") as cp:
                    for rec in pools[["pool_id", "mint", "created_by", "ts"]].itertuples(index=False, name=None):
                        cp.write_row(tuple(_pg(x) for x in rec))
                cur.execute("INSERT INTO pump_pools (pool_id, mint, created_by, ts, source) SELECT pool_id, mint, created_by, ts, 'archive' FROM _pp ON CONFLICT (pool_id) DO NOTHING")
                n = cur.rowcount
    done |= set(days) & complete
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('pump_pools_backfill', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps({"days": sorted(done), "at": datetime.now().isoformat()}),))
    return int(n)


def blocked_pool_ids(conn) -> list[str]:
    """Blocked pools that can carry universe legs. The stream and ``train/mature.py`` both keep only pump.fun-origin mints
    (suffix ``pump``) before this check, so pools of other mints never match either way."""
    return [r["pool_id"] for r in conn.execute("SELECT pool_id FROM pump_pools WHERE mint IS NULL OR right(mint, 4) = 'pump'").fetchall()]


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
    df["curve_known"] = df["curve_known"].map({True: 1.0, False: 0.0})
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
