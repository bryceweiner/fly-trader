"""Wallet skill: how well each wallet's past PumpSwap trades turned out, as a daily decile table the archive
aggregation (``mature.aggregate_day``) and the live stream (``pumpstream.load_skill``) both join, so the selector's
``skill_top*_15m`` inputs (``train/flow.py``: the share of the last 15 minutes' buying from wallets in the top deciles)
are the same in training and live. Copying wallets does not work (2026-09-13); skill is used as an input only.

``build_wallet_day(D)``: for every universe leg of day D (SOL-quoted PumpSwap legs of pump.fun-origin mints) and every
hold h of ``HOLDS_MIN``, the leg's markout — a buy: the net return of holding it h minutes (the close of the last traded
minute at or before t + h over the leg's price, after the one-side exit cost at both ends, ``exit_cost_0p1``); a sell:
minus the gross move it stepped aside from — summed per wallet, SOL-weighted, clipped to ±100 % like the selector's label
→ ``data/corpus/wallet_day/<D>.parquet`` (wallet, h, n, sol, sol_m). Needs D's and D+1's aggregates and feature parts.

``skill_table(D, h, L)``: the L wallet days ending ``GAP_DAYS`` before D (the exits of those days' holds are known once
the following day is aggregated, so the table for D exists before D begins — live never waits for it), score =
Σ sol·m / (Σ sol + k0) with k0 the median Σ sol of the window's wallets (empirical-Bayes shrinkage: a wallet with little
history is pulled toward 0), wallets cut into score deciles → ``wallet_skill/<D>.parquet`` (wallet, bucket 0–9).
The (h, L) in force is fitted by the selector's profit objective (``train/selector.py``) and frozen in
``wallet_skill/config.json``; ``skill_version()`` names it (part of the selector's data version and of every aggregate).
"""
from __future__ import annotations

import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from . import flow, mature

log = logging.getLogger(__name__)
HOLDS_MIN = (10, 20, 30, 45, 60, 90, 120, 180, 240)
LOOKBACKS_D = (1, 3, 7, 14, 30)
GAP_DAYS = 3
DAY_DIR = config.CORPUS_DIR / "wallet_day"
CONFIG_FILE = mature.SKILL_DIR / "config.json"
FIT_FILE = mature.SKILL_DIR / "fit.json"


def skill_config() -> dict | None:
    try:
        return json.loads(CONFIG_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return None


def skill_version() -> str:
    c = skill_config()
    return f"h{c['h']}-L{c['L']}" if c else "none"


def _day_inputs(d: date) -> tuple[list[str], list[str], list[str]] | None:
    nxt = d + timedelta(days=1)
    aggs = [mature.MATURE_DIR / f"{x.isoformat()}.parquet" for x in (d, nxt)]
    parts = [mature.MATURE_FEAT_DIR / x.isoformat() / "part.parquet" for x in (d, nxt)]
    trades = mature._hour_files(d)
    if not trades or not all(p.exists() for p in aggs + parts):
        return None
    return trades, [str(p) for p in aggs], [str(p) for p in parts]


def build_wallet_day(d: date, out_dir: Path | None = None) -> int:
    """Per-wallet markouts of day D for every hold (None inputs → 0 rows written)."""
    t0 = time.time(); inp = _day_inputs(d)
    if inp is None:
        return 0
    trades, aggs, parts = inp
    con = duckdb.connect()
    con.execute("""CREATE TEMP TABLE legs AS SELECT mint, trader, ts, side, sol, price FROM read_parquet(?, union_by_name = true)
                   WHERE pool = 'pump-amm' AND mint LIKE '%pump' AND price > 0 AND sol > 0 AND trader IS NOT NULL""", [trades])
    con.execute("CREATE TEMP TABLE closes AS SELECT mint, ts AS mts, close FROM read_parquet(?) ORDER BY mint, mts", [aggs])
    con.execute("CREATE TEMP TABLE ec AS SELECT mint, ts AS mts, exit_cost_0p1 AS ec FROM read_parquet(?)", [parts])
    out = []
    for h in HOLDS_MIN:
        r = con.execute(f"""
            SELECT l.trader AS wallet, {h} AS h, count(*) AS n, sum(l.sol) AS sol,
                   sum(l.sol * greatest(-1.0, least(1.0, CASE WHEN l.side = 1
                        THEN c.close / l.price * (1 - coalesce(e0.ec, 0)) * (1 - coalesce(e1.ec, 0)) - 1
                        ELSE -(c.close / l.price - 1) END))) AS sol_m
            FROM legs l ASOF JOIN closes c ON c.mint = l.mint AND c.mts <= l.ts + INTERVAL {h} MINUTE
            LEFT JOIN ec e0 ON e0.mint = l.mint AND e0.mts = time_bucket(INTERVAL 1 MINUTE, l.ts)
            LEFT JOIN ec e1 ON e1.mint = c.mint AND e1.mts = c.mts
            GROUP BY l.trader""").fetch_arrow_table()
        out.append(r)
    con.close()
    tab = pa.concat_tables(out)
    root = out_dir or DAY_DIR; root.mkdir(parents=True, exist_ok=True)
    tmp = root / f"{d.isoformat()}.parquet.tmp"; pq.write_table(tab, tmp, compression="zstd"); tmp.replace(root / f"{d.isoformat()}.parquet")
    log.info("wallet day %s: %d wallet-hold rows in %.0fs", d, tab.num_rows, time.time() - t0)
    return tab.num_rows


def window_days(d: date, L: int) -> list[date]:
    """The L wallet days a table for day D may use: D − GAP_DAYS − L + 1 .. D − GAP_DAYS."""
    end = d - timedelta(days=GAP_DAYS)
    return [end - timedelta(days=k) for k in range(L - 1, -1, -1)]


def skill_table(d: date, h: int, L: int, day_dir: Path | None = None) -> pa.Table | None:
    """wallet → decile 0–9 of shrunk skill over the window, or None when the window has no wallet day."""
    root = day_dir or DAY_DIR
    files = [str(root / f"{x.isoformat()}.parquet") for x in window_days(d, L) if (root / f"{x.isoformat()}.parquet").exists()]
    if not files:
        return None
    con = duckdb.connect()
    w = con.execute(f"SELECT wallet, sum(sol) AS sol, sum(sol_m) AS sol_m FROM read_parquet(?) WHERE h = {int(h)} GROUP BY wallet HAVING sum(sol) > 0", [files]).df()
    con.close()
    if w.empty:
        return None
    k0 = float(np.median(w["sol"])); score = w["sol_m"] / (w["sol"] + k0)
    rank = score.rank(method="first") - 1
    bucket = np.minimum((rank * 10 // len(w)).astype(int), 9).astype(np.int8)
    return pa.table({"wallet": pa.array(w["wallet"].astype(str).tolist(), pa.string()), "bucket": pa.array(bucket.tolist(), pa.int8())})


def write_table(d: date, h: int, L: int, out_dir: Path | None = None, day_dir: Path | None = None) -> bool:
    t = skill_table(d, h, L, day_dir)
    if t is None:
        return False
    root = out_dir or mature.SKILL_DIR; root.mkdir(parents=True, exist_ok=True)
    t = t.replace_schema_metadata({b"fly_skill": f"h{int(h)}-L{int(L)}".encode()})
    tmp = root / f"{d.isoformat()}.parquet.tmp"; pq.write_table(t, tmp, compression="zstd"); tmp.replace(root / f"{d.isoformat()}.parquet")
    return True


def freeze(h: int, L: int, fit: dict | None = None) -> None:
    """Record the fitted (h, L); aggregates stamped with another skill version are rebuilt (train/mature.py)."""
    mature.SKILL_DIR.mkdir(parents=True, exist_ok=True)
    CONFIG_FILE.write_text(json.dumps({"h": int(h), "L": int(L), "fit": fit or {}, "frozen_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}))


def table_version(path) -> str:
    """The (h, L) a skill table was written with ('none': before tables were stamped)."""
    return (pq.read_schema(path).metadata or {}).get(b"fly_skill", b"none").decode()


def build_wallet_days() -> int:
    """Wallet days for every aggregated day that has none yet (a day whose next day is not aggregated waits)."""
    n = 0
    for f in sorted(mature.MATURE_DIR.glob("*.parquet")):
        d = date.fromisoformat(f.stem)
        if not (DAY_DIR / f"{d.isoformat()}.parquet").exists() and build_wallet_day(d):
            n += 1
    return n


def _window_ready(d: date, L: int) -> bool:
    """Every wallet day of D's window exists (days before the first wallet day do not count: the corpus starts there)."""
    have = sorted(DAY_DIR.glob("*.parquet")) if DAY_DIR.exists() else []
    if not have:
        return False
    first = date.fromisoformat(have[0].stem.split(".")[0])
    need = [x for x in window_days(d, L) if x >= first]
    return bool(need) and all((DAY_DIR / f"{x.isoformat()}.parquet").exists() for x in need)


def daily() -> int:
    """Run by mature's build loop: wallet days for the aggregated days; with a frozen (h, L), the table of that version for
    every day from the first aggregate through tomorrow (UTC) once its whole window exists — so a day's table is there
    before the day begins, and archive and live read the same file. A rewritten table re-aggregates its day."""
    n = build_wallet_days()
    c = skill_config()
    aggs = sorted(mature.MATURE_DIR.glob("*.parquet"))
    if not c or not aggs:
        return n
    v = skill_version(); d = date.fromisoformat(aggs[0].stem); last = datetime.now(timezone.utc).date() + timedelta(days=1)
    while d <= last:
        f = mature.SKILL_DIR / f"{d.isoformat()}.parquet"
        if (not f.exists() or table_version(f) != v) and _window_ready(d, c["L"]) and write_table(d, c["h"], c["L"]):
            n += 1
        d += timedelta(days=1)
    return n


def _day_skill(d: date, h: int, L: int, day_dir: Path | None = None):
    """(mint, minute start s, skill_buy [n, N_SKILL]) of day D's aggregate rows as the archive builds them under candidate
    (h, L)'s table (``mature.aggregate_table``: the aggregation's own SQL); None without usable hour files."""
    files = mature._hour_files(d)
    if not files or mature._lacks_pool_id(files):
        return None
    tab = mature.aggregate_table(files, skill_table(d, h, L, day_dir))
    sb = np.zeros((tab.num_rows, flow.N_SKILL))
    col = tab["skill_buy"].combine_chunks()
    if col.null_count < len(col):
        valid = col.is_valid().to_numpy(zero_copy_only=False)
        sb[valid] = np.asarray(col.flatten().to_numpy(zero_copy_only=False), dtype=np.float64).reshape(-1, flow.N_SKILL)
    return tab["mint"].to_numpy(zero_copy_only=False), mature._epoch_s(tab["ts"].to_pandas()), sb


def candidate_skill_cols(ds, h: int, L: int, day_dir: Path | None = None, stop=None) -> np.ndarray:
    """[rows of ``ds``, SKILL_COLS]: the skill inputs every decision row would have had under candidate (h, L)'s tables —
    each day re-aggregated with its candidate table, the previous day as the lookback (``mature.build_day``), FlowWindow's
    15-minute window (``flow.skill_features``)."""
    from . import progress as prog
    out = np.zeros((len(ds.y), len(flow.SKILL_COLS)), np.float32)
    groups = pd.Series(np.arange(len(ds.y))).groupby(ds.day).indices
    days = sorted(groups); prev_d, prev = None, None
    for i, d in enumerate(days):
        if stop is not None and stop.is_set():
            break
        before = prev if prev_d == d - timedelta(days=1) else _day_skill(d - timedelta(days=1), h, L, day_dir)
        cur = _day_skill(d, h, L, day_dir); prev_d, prev = d, cur
        if cur is None:
            continue
        parts = [x for x in (before, cur) if x is not None]
        m = np.concatenate([x[0] for x in parts]); t = np.concatenate([x[1] for x in parts]); sb = np.vstack([x[2] for x in parts])
        f = flow.skill_features(m, t, sb)
        keep = np.flatnonzero(t >= datetime(d.year, d.month, d.day, tzinfo=timezone.utc).timestamp())
        idx = groups[d]
        j = pd.DataFrame({"mint": ds.mint[idx], "ts": np.asarray(ds.ts[idx], dtype=np.float64), "i": idx}).merge(
            pd.DataFrame({"mint": m[keep], "ts": np.asarray(t[keep], dtype=np.float64), "k": keep}), on=["mint", "ts"], how="left")
        hit = j["k"].notna().to_numpy()
        out[j["i"].to_numpy()[hit]] = f[j["k"].to_numpy()[hit].astype(np.int64)]
        prog.update(f"wallet skill: inputs under h{h} L{L}", i + 1, len(days))
    return out


def _read_fit() -> dict:
    try:
        return json.loads(FIT_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {}


def _write_fit(state: dict) -> None:
    FIT_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FIT_FILE.with_suffix(".tmp"); tmp.write_text(json.dumps(state, default=str)); tmp.replace(FIT_FILE)


def fit(days: int | None = None, stop=None) -> dict:
    """The (h, L) the skill tables use: for every hold of ``HOLDS_MIN`` × lookback of ``LOOKBACKS_D``, the ev strategy fitted
    on the legacy inputs + the skill inputs those tables give (``strategies.fit_strategy``: the selector's objective on the
    selection half of the stack's walk-forward). The best by that objective is frozen; its evaluation half is read once,
    for the record (the stack's input step judges the skill group again). Resumable: finished candidates are kept in
    ``wallet_skill/fit.json``. Then the tables of that version are written (``daily``) and the build loop re-aggregates."""
    from . import progress as prog, strategies
    from .decisions import LEGACY_COLS, build
    from .selector import in_universe
    prog.update("wallet skill: wallet days", 0, 1, force=True)
    build_wallet_days()
    ds = build(days=days, horizon_min=strategies.EV_HOLD, holds=strategies.HOLDS_MIN)   # ev fits its hold, like the other strategies
    ds.X = np.require(ds.X, requirements=["W"])            # pandas hands back a read-only array; the skill columns are rewritten per candidate
    uni = in_universe(ds.X, ds.cols)
    tdays = sorted({d for blk in strategies.wf_blocks(ds.days) for d in blk})
    half = tdays[len(tdays) // 2]; day0 = tdays[0]
    tested = np.isin(ds.day, tdays); sel = tested & (ds.day < half); ev = tested & (ds.day >= half)
    sk = [ds.cols.index(c) for c in flow.SKILL_COLS]; cols = list(LEGACY_COLS) + list(flow.SKILL_COLS)
    key = f"{ds.days[0]}..{ds.days[-1]}|{strategies.OBJECTIVE}"     # candidates fitted under another objective are re-run, not reused
    state = _read_fit()
    if state.get("key") != key:
        state = {"key": key, "candidates": {}}
    if "baseline" not in state:
        b = strategies.fit_strategy(ds, "ev", list(LEGACY_COLS), uni, sel, day0, stop, holds=(strategies.EV_HOLD,))
        if stop is not None and stop.is_set():
            return {"stopped": True}
        state["baseline"] = b.selection if b else None; _write_fit(state)
    grid = [(h, L) for h in HOLDS_MIN for L in LOOKBACKS_D]
    for i, (h, L) in enumerate(grid):
        name = f"h{h}-L{L}"
        if name in state["candidates"]:
            continue
        prog.update("wallet skill: fitting (hold, lookback)", i, len(grid), force=True, candidate=name)
        ds.X[:, sk] = candidate_skill_cols(ds, h, L, stop=stop)
        f = strategies.fit_strategy(ds, "ev", cols, uni, sel, day0, stop, holds=(strategies.EV_HOLD,), cache_tag=f"|{name}")
        if stop is not None and stop.is_set():
            return {"stopped": True, "candidates": len(state["candidates"])}
        state["candidates"][name] = {"h": h, "L": L, "selection": f.selection if f else None, "trials": f.trials if f else 0,
                                     "best_seen": f.best_seen if f else None}
        _write_fit(state)
        log.info("wallet skill %s: selection %s", name, state["candidates"][name]["selection"])

    def score(s):
        return strategies.Score(**{k: v for k, v in s.items() if k in strategies.Score.__dataclass_fields__}) if s else None
    best = None
    for c in state["candidates"].values():
        sc = score(c["selection"])
        if sc is not None and strategies.better(sc, score(best["selection"]) if best else None):
            best = c
    if best is None:
        raise RuntimeError("no (hold, lookback) gave the ev strategy a setting that passes the objective on the selection half")
    ds.X[:, sk] = candidate_skill_cols(ds, best["h"], best["L"], stop=stop)
    f = strategies.fit_strategy(ds, "ev", cols, uni, sel, day0, stop, holds=(strategies.EV_HOLD,), cache_tag=f"|h{best['h']}-L{best['L']}")
    evs = strategies.score_pick(ds, f.pick(ds, ev, uni), f.hold_min * 60.0, ds.fwd_h.get(f.hold_min, ds.fwd_pess), day0) if f else strategies.Score()
    rec = {"h": best["h"], "L": best["L"], "selection": best["selection"], "evaluation": evs.dict(), "baseline": state.get("baseline"),
           "beats_baseline": bool(strategies.better(score(best["selection"]), score(state.get("baseline")))), "candidates": len(state["candidates"]),
           "trials": int(sum(c["trials"] for c in state["candidates"].values())), "days": key}
    freeze(best["h"], best["L"], rec)
    rec["tables"] = daily()
    log.info("wallet skill frozen at h%d L%d (beats the inputs without skill: %s); %d tables written", best["h"], best["L"], rec["beats_baseline"], rec["tables"])
    prog.update("wallet skill: done", 1, 1, force=True)
    return rec
