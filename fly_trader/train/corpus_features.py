"""Corpus → observation rows: the same 45 features the runner computes (``market/features.py``), one row per
minute in which the token traded, for every pulled token.

* candle path (every token): each 1-minute candle is fed to ``TokenState`` as one synthetic trade at the close
  with the candle's SOL volume; features that need per-trade detail (imbalance, counts, signers, Hawkes,
  VPIN, since-last) are zeroed and unmasked. Realised vol is candle-based here and trade-based live.
* trade path (the sampled tokens with per-trade rows): the real trades replay through ``TokenState`` exactly
  as the live tape does; holders, top-10 concentration, net buyers and holder change are reconstructed from
  wallet flows and written into the Jupiter-stat slots (proxies: traders only, not all token accounts).
* liquidity: PumpSwap quote reserve ≈ ``CORPUS_RQ0_SOL·sqrt(p/p_grad)`` (tape check 2026-09-13: per-token
  correlation of log reserve with ½·log price = 0.997). Only post-graduation minutes become rows; the bonding-curve phase is summarised in pre_* columns.

Output: ``data/corpus/features/<graduation day>/part-<epoch>.parquet`` (columns: mint, ts, phase, t_rel_min,
open/high/low/close, volume_sol, resq, age_h, has_trades, mask, and one column per feature name); registry
``corpus_features``. Runs as a thread inside the ``corpus`` worker and as ``fly-trader build-corpus-features``.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.connection import transaction
from ..market.features import D, FEATURES, FIDX, TokenMeta, TokenState

log = logging.getLogger(__name__)
CANDLE_ONLY_MASKED = ["imb_1m", "imb_5m", "imb_15m", "imb_1h", "logn_1m", "logn_5m", "logn_15m", "logn_1h",
                      "logsigners_15m", "logsigners_1h", "hawkes", "log_since_last", "vpin_15m"]
PROXY_STATS = ["log_holders", "top_holders_pct", "net_buyers_1h", "holder_change_1h"]
PRE_COLS = ["pre_minutes", "pre_trades", "pre_buyers", "pre_vol_sol", "pre_top10_pct"]   # bonding-curve summary at graduation (MELT-style)
BASE_COLS = ["mint", "ts", "phase", "t_rel_min", "open", "high", "low", "close", "volume_sol", "resq", "age_h", "has_trades", "mask"] + PRE_COLS
SCHEMA = pa.schema([("mint", pa.string()), ("ts", pa.timestamp("ms", tz="UTC")), ("phase", pa.string()), ("t_rel_min", pa.int32()),
                    ("open", pa.float64()), ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64()), ("volume_sol", pa.float64()),
                    ("resq", pa.float64()), ("age_h", pa.float32()), ("has_trades", pa.bool_()), ("mask", pa.int64())] +
                   [(name, pa.float32()) for name in PRE_COLS] + [(name, pa.float32()) for name in FEATURES])


def _epoch_s(col: pd.Series):
    """tz-aware datetime column → float seconds since the epoch, whatever the stored unit (ms/ns)."""
    return ((col - pd.Timestamp(0, tz="UTC")) / pd.Timedelta(seconds=1)).to_numpy(dtype="float64")


def _rq(price: float, p_grad: float) -> float:
    return config.CORPUS_RQ0_SOL * math.sqrt(max(price, 1e-30) / p_grad)


def _row(mint: str, ts_s: float, g: float, o: float, h: float, l: float, c: float, vol: float, resq: float | None,
         has_trades: bool, f: list[float], mask: int, pre: dict) -> dict:
    r = {"mint": mint, **{k: float(pre.get(k, float("nan"))) for k in PRE_COLS}, "ts": datetime.fromtimestamp(ts_s, timezone.utc),
         "phase": "amm" if ts_s + 60.0 > g else "curve",          # the minute containing the graduation second counts as AMM
         "t_rel_min": max(0, int(math.floor((ts_s - g) / 60.0))), "open": o, "high": h, "low": l, "close": c, "volume_sol": vol,
         "resq": resq if resq is not None else float("nan"), "age_h": max(0.0, (ts_s - g) / 3600.0), "has_trades": has_trades, "mask": mask}
    for i, name in enumerate(FEATURES):
        r[name] = float(f[i])
    return r


def candle_rows(mint: str, g: float, candle_path: str) -> list[dict]:
    """Rows for every post-graduation 1-minute candle; the bonding-curve candles only feed the pre-migration summary."""
    cd = pq.read_table(candle_path).to_pandas()
    cd = cd[cd["interval"] == "1m"].sort_values("ts")
    if cd.empty:
        return []
    ts_s = _epoch_s(cd["ts"]); amm = ts_s >= g
    if not amm.any():
        return []
    pre_vol = float(cd["volume_sol"].to_numpy()[~amm].sum()); pre_min = float((g - ts_s[~amm].min()) / 60.0) if (~amm).any() else 0.0
    pre = {"pre_minutes": pre_min, "pre_trades": float("nan"), "pre_buyers": float("nan"), "pre_vol_sol": pre_vol, "pre_top10_pct": float("nan")}
    cd = cd[amm]; ts_s = ts_s[amm]
    p_grad = float(cd["close"].iloc[0])
    real = cd["resq_sol"].to_numpy(dtype="float64") if "resq_sol" in cd.columns else None
    st = TokenState(mint); meta = TokenMeta(mint=mint, program_label="Pump.fun Amm", graduated_at=g)
    out = []
    for i, (t0, o, h, l, c, v) in enumerate(zip(ts_s, cd["open"], cd["high"], cd["low"], cd["close"], cd["volume_sol"])):
        t_end = float(t0) + 60.0; price = float(c)
        resq = float(real[i]) if real is not None and math.isfinite(real[i]) and real[i] > 0 else _rq(price, p_grad)
        st.append(t_end - 1e-3, price, float(v), True, None, resq)
        f, mask = st.features(t_end, meta)
        for name in CANDLE_ONLY_MASKED:
            f[FIDX[name]] = 0.0; mask &= ~(1 << FIDX[name])
        out.append(_row(mint, float(t0), g, float(o), float(h), float(l), price, float(v), resq, False, f, mask, pre))
    return out


def trade_rows(mint: str, g: float, trade_path: str) -> list[dict]:
    """One row per post-graduation minute with trades. Curve-phase trades only build wallet balances and the trailing-hour
    buyer/seller sets (their prices are on the curve's own scale); the 85 SOL migration transfer is dropped."""
    td = pq.read_table(trade_path).to_pandas().sort_values("ts")
    if td.empty:
        return []
    ts_s = _epoch_s(td["ts"])
    keep = ~((td["program"].to_numpy() == "pump") & (ts_s >= g - 5.0) & (td["sol"].to_numpy() >= 50.0))
    td = td[keep]; ts_s = ts_s[keep]
    prices = td["price_sol"].to_numpy(); sols = td["sol"].to_numpy(); sides = td["side"].to_numpy(); toks = td["tokens"].to_numpy(); wallets = td["wallet"].to_numpy()
    amm = (ts_s >= g) & (prices > 0)
    if not amm.any():
        return []
    p_grad = float(prices[amm][0])
    st = TokenState(mint); meta = TokenMeta(mint=mint, program_label="Pump.fun Amm", graduated_at=g)
    bal: dict = defaultdict(float)
    recent: deque = deque()            # (ts, wallet, side) for the trailing hour
    buyers_1h: Counter = Counter(); sellers_1h: Counter = Counter()
    holders_hist: deque = deque()      # (ts, holders)
    pre_buyers: set = set(); pre_trades = 0; pre_vol = 0.0; pre_min = float((g - ts_s.min()) / 60.0) if ts_s.min() < g else 0.0

    def book(t: float, w, side: int, tok: float) -> None:
        if side == 1:
            bal[w] += tok; buyers_1h[w] += 1
        else:
            bal[w] = max(0.0, bal[w] - tok); sellers_1h[w] += 1
        recent.append((t, w, side))

    def expire(t_end: float) -> None:
        while recent and recent[0][0] < t_end - 3600.0:
            _, w, side = recent.popleft(); cnt = buyers_1h if side == 1 else sellers_1h
            cnt[w] -= 1
            if cnt[w] <= 0:
                del cnt[w]

    out = []; n = len(ts_s); i = 0
    while i < n and ts_s[i] < g:                     # curve phase
        t, side, tok, w, s_ = float(ts_s[i]), int(sides[i]), float(toks[i]), wallets[i], float(sols[i]); i += 1
        book(t, w, side, tok); pre_trades += 1; pre_vol += s_
        if side == 1:
            pre_buyers.add(w)
    pos0 = [v for v in bal.values() if v > 0]
    pre = {"pre_minutes": pre_min, "pre_trades": float(pre_trades), "pre_buyers": float(len(pre_buyers)), "pre_vol_sol": pre_vol,
           "pre_top10_pct": (100.0 * sum(sorted(pos0)[-10:]) / sum(pos0)) if pos0 else float("nan")}
    while i < n:                                     # AMM phase, minute by minute
        minute = math.floor(ts_s[i] / 60.0) * 60.0
        o = h = l = c = None; vol = 0.0
        while i < n and ts_s[i] < minute + 60.0:
            t, p, s_, side, tok, w = float(ts_s[i]), float(prices[i]), float(sols[i]), int(sides[i]), float(toks[i]), wallets[i]
            i += 1
            if not (p > 0 and math.isfinite(p)):
                continue
            st.append(t, p, s_, side == 1, w, _rq(p, p_grad))
            o = p if o is None else o; h = p if h is None else max(h, p); l = p if l is None else min(l, p); c = p; vol += s_
            book(t, w, side, tok)
        if c is None:
            continue
        t_end = minute + 60.0; expire(t_end)
        f, mask = st.features(t_end, meta)
        pos = [v for v in bal.values() if v > 0]
        holders = len(pos); total = sum(pos)
        top10 = sum(sorted(pos)[-10:]) / total if total > 0 else 0.0
        while holders_hist and holders_hist[0][0] < t_end - 3660.0:
            holders_hist.popleft()
        h_ago = holders_hist[0][1] if holders_hist else None
        holders_hist.append((t_end, holders))
        f[FIDX["log_holders"]] = math.log1p(holders); f[FIDX["top_holders_pct"]] = 100.0 * top10
        f[FIDX["net_buyers_1h"]] = float(len(buyers_1h) - len(sellers_1h))
        f[FIDX["holder_change_1h"]] = (holders / h_ago - 1.0) if h_ago else 0.0
        for name in PROXY_STATS:
            mask |= 1 << FIDX[name]
        out.append(_row(mint, minute, g, o, h, l, c, vol, _rq(c, p_grad), True, f, mask, pre))
    return out


def build_token(mint: str, graduated_at: datetime, candle_path: str | None, trade_path: str | None) -> tuple[list[dict], int]:
    g = graduated_at.timestamp(); rows: list[dict] = []; n_trade = 0
    if candle_path and Path(candle_path).exists():
        rows += candle_rows(mint, g, candle_path)
    if trade_path and Path(trade_path).exists():
        tr = trade_rows(mint, g, trade_path); n_trade = len(tr); rows += tr
    return rows, n_trade


def _pending(limit: int) -> list[dict]:
    with transaction() as conn:
        return conn.execute("SELECT t.mint, t.graduated_at, t.candle_path, t.trade_path FROM corpus_tokens t LEFT JOIN corpus_features f USING (mint) "
                            "WHERE t.status = 'done' AND f.mint IS NULL ORDER BY t.graduated_at DESC LIMIT %s", (limit,)).fetchall()


def _write_parts(rows_by_day: dict[str, list[dict]]) -> dict[str, str]:
    paths = {}
    for day, rows in rows_by_day.items():
        d = config.CORPUS_FEATURES_DIR / day; d.mkdir(parents=True, exist_ok=True)
        path = d / f"part-{int(time.time() * 1000)}.parquet"
        pq.write_table(pa.Table.from_pylist(rows, schema=SCHEMA), path, compression="zstd")
        paths[day] = str(path)
    return paths


def _status(**kv) -> None:
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('corpus_features_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (json.dumps({**kv, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str),))
    except Exception:
        log.debug("features status write failed", exc_info=True)


def build_batch(limit: int = 200) -> int:
    """Build features for up to ``limit`` pulled tokens without rows yet. Returns tokens built."""
    todo = _pending(limit)
    if not todo:
        return 0
    rows_by_day: dict[str, list[dict]] = defaultdict(list); reg: list[tuple] = []; t0 = time.time()
    for t in todo:
        try:
            rows, n_trade = build_token(t["mint"], t["graduated_at"], t["candle_path"], t["trade_path"])
        except Exception as e:
            log.warning("features %s failed: %s", t["mint"], e); rows, n_trade = [], 0
        day = t["graduated_at"].astimezone(timezone.utc).date().isoformat()
        rows_by_day[day].extend(rows)
        reg.append((t["mint"], len(rows), n_trade, n_trade > 0, day))
    paths = _write_parts(rows_by_day)
    with transaction() as conn:
        conn.cursor().executemany("INSERT INTO corpus_features (mint, rows, trade_rows, has_trades, path) VALUES (%s,%s,%s,%s,%s) ON CONFLICT (mint) DO NOTHING",
                                  [(m, r, tr, ht, paths.get(day)) for m, r, tr, ht, day in reg])
    n_rows = sum(len(v) for v in rows_by_day.values())
    log.info("features: %d tokens, %d rows in %.1fs", len(todo), n_rows, time.time() - t0)
    return len(todo)


def build_loop(stop_event: threading.Event | None = None, idle_s: float = 20.0) -> None:
    built = 0; rows_total = 0
    while not (stop_event is not None and stop_event.is_set()):
        try:
            n = build_batch()
        except Exception as e:
            log.exception("feature build failed"); _status(stage="error", last_error=str(e)[:200]); n = 0
        if n:
            built += n
            with transaction() as conn:
                c = conn.execute("SELECT count(*) AS n, coalesce(sum(rows),0) AS rows, count(*) FILTER (WHERE has_trades) AS with_trades FROM corpus_features").fetchone()
            _status(stage="building", built_this_run=built, tokens=int(c["n"]), rows=int(c["rows"]), with_trades=int(c["with_trades"]))
            continue
        _status(stage="idle", built_this_run=built)
        end = time.time() + idle_s
        while time.time() < end and not (stop_event is not None and stop_event.is_set()):
            time.sleep(1.0)


def main() -> None:
    """CLI: build everything pending, then exit."""
    from ..logging_setup import setup
    setup("corpus")
    total = 0
    while True:
        n = build_batch(500)
        if not n:
            break
        total += n
    print(f"built features for {total} tokens")
