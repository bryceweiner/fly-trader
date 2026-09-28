"""RH minutes (``rh_minutes``): the twin of ``pump_minutes``, in ETH, built from the indexed swaps.

The same rules as the Solana stream (ingest/pumpstream.Aggregator) and the archive (train/mature.aggregate_table):
- a swap counts only while its pool's reserve is plausible (``RESQ_BAND`` scaled by K = ETH per SOL) and its price is
  within ``PRICE_BAND`` (50×) of the median of the mint's previous ``REF_LEGS`` (200) prices that UTC day;
- open/high/low/close = the pool price after each swap (sqrtPriceX96 of the Swap event), in ETH per token: the quote price
  times the quote asset's ETH value that minute (rh/prices.py; ETH-quoted pools: 1). A minute whose non-ETH quote has no
  mark yet is not written (it cannot be expressed in ETH);
- buy/sell volume in ETH; resq = the pool's virtual quote reserve in ETH (full-range locked liquidity: constant product);
- fee_rate = the volume-weighted Pons hook fee + tax fraction (the analogue of pump_minutes.fee_rate, the pool fee charged);
- the wallet columns (buyers, wash, top seller, insider sells, buys per skill decile) by train/flow.minute_wallet_cols.

Minutes are recomputed from the swaps, never accumulated, so a live pass and a backfill write identical rows.
"""
from __future__ import annotations

import json
import statistics
from collections import deque
from datetime import datetime, timezone

from .. import config
from ..db.connection import transaction
from ..train.flow import minute_wallet_cols
from . import prices

RESQ_BAND_SOL = (0.001, 100000.0)          # ingest/pumpstream.RESQ_BAND
PRICE_BAND = 50.0
REF_LEGS = 200
GRACE_S = 20                               # a minute is complete once the index is this far past its end


class _Row:
    __slots__ = ("pool", "asset", "cls", "open", "high", "low", "close", "close_q", "base", "buy", "sell", "nb", "ns", "traders", "wallets",
                 "resq_eth", "resq_q", "fee_w", "fee_v")

    def __init__(self, pool, asset, cls, base):
        self.pool, self.asset, self.cls, self.base = pool, asset, cls, base
        self.open = None; self.high = -1e308; self.low = 1e308; self.close = None; self.close_q = None
        self.buy = self.sell = 0.0; self.nb = self.ns = 0; self.traders = set(); self.wallets: dict[str, list[float]] = {}
        self.resq_eth = self.resq_q = None; self.fee_w = 0.0; self.fee_v = 0.0


def aggregate(conn, t0: int, t1: int, skill: dict | None = None) -> list[tuple]:
    """rh_minutes rows for every minute in [t0, t1) (epoch s, minute-aligned)."""
    k = config.RH_ETH_PER_SOL; lo_band, hi_band = RESQ_BAND_SOL[0] * k, RESQ_BAND_SOL[1] * k
    day0 = (t0 // 86400) * 86400
    swaps = conn.execute(
        "SELECT s.ts, s.token, s.pool_id, s.trader, s.side, s.quote_raw, s.price_q, s.resq_q, s.fee_frac, p.quote_asset, a.decimals AS qdec, a.class AS cls "
        "FROM rh_swaps s JOIN rh_pools p ON p.pool_id = s.pool_id JOIN rh_assets a ON a.asset = p.quote_asset "
        "WHERE s.ts >= to_timestamp(%s) AND s.ts < to_timestamp(%s) ORDER BY s.ts, s.block, s.log_index", (day0, t1)).fetchall()
    insiders: dict[str, frozenset] = {}
    base_cache: dict[tuple, tuple] = {}
    ref: dict[str, deque] = {}; ref_day = None
    rows: dict[tuple, _Row] = {}
    for s in swaps:
        ts = s["ts"].timestamp(); day = int(ts // 86400)
        if ref_day is None or day != ref_day:
            ref = {}; ref_day = day
        p = float(s["price_q"] or 0.0)
        if p <= 0:
            continue
        dq = ref.setdefault(s["token"], deque(maxlen=REF_LEGS))
        ok = not dq or (statistics.median(dq) / PRICE_BAND <= p <= statistics.median(dq) * PRICE_BAND)
        dq.append(p)
        if not ok or ts < t0:
            continue                                               # before t0: only the median's history
        m = int(ts // 60) * 60
        key_b = (s["quote_asset"], m)
        if key_b not in base_cache:
            base_cache[key_b] = prices.base_eth(conn, s["quote_asset"], datetime.fromtimestamp(m + 59, timezone.utc))
        base, _stale = base_cache[key_b]
        if base is None:
            continue
        resq_eth = float(s["resq_q"] or 0.0) * base
        if not (lo_band <= resq_eth <= hi_band):
            continue
        r = rows.get((s["token"], m))
        if r is None:
            r = rows[(s["token"], m)] = _Row(s["pool_id"], s["quote_asset"], s["cls"], base)
        pe = p * base
        r.open = pe if r.open is None else r.open; r.high = max(r.high, pe); r.low = min(r.low, pe); r.close = pe; r.close_q = p
        vol = float(s["quote_raw"] or 0) / 10 ** int(s["qdec"]) * base
        if s["side"] > 0:
            r.buy += vol; r.nb += 1
        else:
            r.sell += vol; r.ns += 1
        if s["trader"]:
            r.traders.add(s["trader"]); r.wallets.setdefault(s["trader"], [0.0, 0.0])[0 if s["side"] > 0 else 1] += vol
        r.resq_eth, r.resq_q = resq_eth, float(s["resq_q"] or 0.0)
        if s["fee_frac"] is not None:
            r.fee_w += float(s["fee_frac"]) * vol; r.fee_v += vol
    out = []
    for (mint, m), r in sorted(rows.items(), key=lambda kv: (kv[0][1], kv[0][0])):
        if mint not in insiders:
            insiders[mint] = frozenset(x["wallet"] for x in conn.execute("SELECT wallet FROM rh_insiders WHERE mint = %s", (mint,)).fetchall())
        wc = minute_wallet_cols(r.wallets, insiders[mint], skill)
        out.append((mint, datetime.fromtimestamp(m, timezone.utc), r.pool, r.asset, r.cls, r.open, r.high, r.low, r.close, r.close_q, r.base, r.buy, r.sell,
                    r.nb, r.ns, len(r.traders), r.resq_eth, r.resq_q, (r.fee_w / r.fee_v) if r.fee_v > 0 else None,
                    wc["n_buyers"], wc["wash_sol"], wc["wash_buy_sol"], wc["top_sell_sol"], wc["insider_sell_sol"], wc["skill_buy"]))
    return out


def write(conn, rows: list[tuple]) -> int:
    if rows:
        conn.cursor().executemany(
            "INSERT INTO rh_minutes (mint, ts, pool_id, quote_asset, quote_class, open, high, low, close, close_q, base_eth, buy_eth, sell_eth, n_buys, n_sells, "
            "n_traders, resq_eth, resq_q, fee_rate, n_buyers, wash_eth, wash_buy_eth, top_sell_eth, insider_sell_eth, skill_buy) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (mint, ts) DO UPDATE SET "
            "pool_id = EXCLUDED.pool_id, open = EXCLUDED.open, high = EXCLUDED.high, low = EXCLUDED.low, close = EXCLUDED.close, close_q = EXCLUDED.close_q, "
            "base_eth = EXCLUDED.base_eth, buy_eth = EXCLUDED.buy_eth, sell_eth = EXCLUDED.sell_eth, n_buys = EXCLUDED.n_buys, n_sells = EXCLUDED.n_sells, "
            "n_traders = EXCLUDED.n_traders, resq_eth = EXCLUDED.resq_eth, resq_q = EXCLUDED.resq_q, fee_rate = EXCLUDED.fee_rate, n_buyers = EXCLUDED.n_buyers, "
            "wash_eth = EXCLUDED.wash_eth, wash_buy_eth = EXCLUDED.wash_buy_eth, top_sell_eth = EXCLUDED.top_sell_eth, insider_sell_eth = EXCLUDED.insider_sell_eth, "
            "skill_buy = EXCLUDED.skill_buy, revised = rh_minutes.revised OR (rh_minutes.close IS DISTINCT FROM EXCLUDED.close)", rows)
    return len(rows)


def indexed_through_s(conn, rpc) -> float | None:
    """The chain time the index has reached (the cursor block's timestamp)."""
    r = conn.execute("SELECT block FROM rh_scan WHERE name = 'rh'").fetchone()
    if not r or int(r["block"]) <= 0:
        return None
    b = rpc.block(int(r["block"]))
    return float(int(b["timestamp"], 16)) if b else None


def run_once(rpc, skill: dict | None = None, max_minutes: int = 240) -> dict:
    """Write every complete minute since the last flush (up to ``max_minutes``), set rh_stream_status.flushed_through."""
    with transaction() as conn:
        top = indexed_through_s(conn, rpc)
        st = conn.execute("SELECT detail FROM rh_scan WHERE name = 'rh_minutes'").fetchone()
        if top is None:
            return {"minutes": 0}
        bound = int((top - GRACE_S) // 60) * 60
        done = int(((st["detail"] or {}) if st else {}).get("through") or 0)
        if not done:
            first = conn.execute("SELECT min(ts) AS t FROM rh_swaps").fetchone()["t"]
            if first is None:
                return {"minutes": 0}
            done = int(first.timestamp() // 60) * 60
        t1 = min(bound, done + max_minutes * 60)
        if t1 <= done:
            return {"minutes": 0, "through": done}
        n = write(conn, aggregate(conn, done, t1, skill))
        conn.execute("INSERT INTO rh_scan (name, block, detail) VALUES ('rh_minutes', 0, %s) ON CONFLICT (name) DO UPDATE SET detail = EXCLUDED.detail, updated_at = now()",
                     (json.dumps({"through": t1}),))
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('rh_stream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps({"flushed_through": datetime.fromtimestamp(t1 - 60, timezone.utc).isoformat(), "rows": n}),))
    return {"minutes": (t1 - done) // 60, "rows": n, "through": t1}
