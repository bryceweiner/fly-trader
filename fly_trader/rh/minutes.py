"""RH minutes (``rh_minutes``): the twin of ``pump_minutes``, in ETH, built from the indexed swaps.

The same rules as the Solana stream (ingest/pumpstream.Aggregator) and the archive (train/mature.aggregate_table):
- a swap counts only while its pool's reserve is plausible (``RESQ_BAND`` scaled by K = ETH per SOL) and its price is
  within ``PRICE_BAND`` (50×) of the median of the pool's previous ``REF_LEGS`` (200) prices that UTC day (a Pons coin has
  one Pons pool: until 2026-10-23 the median was keyed by the mint, the same prices);
- open/high/low/close = the pool price after each swap (sqrtPriceX96 of the Swap event), in ETH per token: the quote price
  times the quote asset's ETH value that minute (rh/prices.py; ETH-quoted pools: 1). A minute whose non-ETH quote has no
  mark yet is not written (it cannot be expressed in ETH);
- buy/sell volume in ETH; resq = the pool's virtual quote reserve in ETH (full-range locked liquidity: constant product);
- fee_rate = the volume-weighted Pons hook fee + tax fraction, or another v4 pool's LP fee (the analogue of
  pump_minutes.fee_rate, the pool fee charged);
- a coin with several indexed pools (a Pons coin's own pools elsewhere, a plain v4 memecoin's pools: rh/v4_universe.py)
  takes each minute from the pool with the most swaps, as the Solana stream and archive take the pool with the most legs;
- the wallet columns (buyers, wash, top seller, insider sells, buys per skill decile) by train/flow.minute_wallet_cols.

Minutes are recomputed from the swaps, never accumulated, so a live pass and a backfill write identical rows. Only the
median needs swaps before the first minute: each pool's last ``REF_LEGS`` prices that day are read for it (until 2026-10-23
every swap since 00:00 UTC was re-read for each live minute: ~2.3 s per minute by 23:59, in front of the RH live entries).
Only the Pons pools and the other v4 pools that went live (``indexed``) count, unless ``tokens`` names the coins.
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
GRACE_S = 2                                # a minute is complete once the index is this far past its end: block times never
#   decrease, so every swap before the minute's end is indexed once a later block is (ingest/pumpstream.FLUSH_GRACE_S; 20 until 2026-10-09)


class _Row:
    __slots__ = ("pool", "asset", "cls", "open", "high", "low", "close", "close_q", "base", "buy", "sell", "nb", "ns", "traders", "wallets",
                 "resq_eth", "resq_q", "fee_w", "fee_v")

    def __init__(self, pool, asset, cls, base):
        self.pool, self.asset, self.cls, self.base = pool, asset, cls, base
        self.open = None; self.high = -1e308; self.low = 1e308; self.close = None; self.close_q = None
        self.buy = self.sell = 0.0; self.nb = self.ns = 0; self.traders = set(); self.wallets: dict[str, list[float]] = {}
        self.resq_eth = self.resq_q = None; self.fee_w = 0.0; self.fee_v = 0.0


def aggregate(conn, t0: int, t1: int, skill: dict | None = None, quote_asset: str | None = None, tokens: list[str] | None = None) -> list[tuple]:
    """rh_minutes rows for every minute in [t0, t1) (epoch s, minute-aligned). ``quote_asset``: only the pools quoted in
    it (a repair after that asset's ETH marks changed); ``tokens``: only these coins, every pool of theirs indexed so far
    (a backfill of their history; otherwise only live pools count)."""
    k = config.RH_ETH_PER_SOL; lo_band, hi_band = RESQ_BAND_SOL[0] * k, RESQ_BAND_SOL[1] * k
    day0 = (t0 // 86400) * 86400
    swaps = conn.execute(
        "SELECT s.ts, s.token, s.pool_id, s.trader, s.side, s.quote_raw, s.price_q, s.resq_q, "
        "CASE WHEN s.fee_v = 2 OR s.fee_frac IS NULL OR s.fee_frac >= 1 THEN s.fee_frac ELSE s.fee_frac / (1 - s.fee_frac) END AS fee_frac, "   # rh/index.FEE_FRAC
        "p.quote_asset, a.decimals AS qdec, a.class AS cls "
        "FROM rh_swaps s JOIN rh_pools p ON p.pool_id = s.pool_id JOIN rh_assets a ON a.asset = p.quote_asset "
        "WHERE s.ts >= to_timestamp(%(t0)s) AND s.ts < to_timestamp(%(t1)s) AND (%(qa)s::text IS NULL OR p.quote_asset = %(qa)s) "
        "AND (CASE WHEN %(toks)s::text[] IS NULL THEN p.is_pons OR p.indexed ELSE s.token = ANY(%(toks)s::text[]) END) ORDER BY s.ts, s.block, s.log_index",
        {"t0": t0, "t1": t1, "qa": quote_asset, "toks": tokens}).fetchall()
    insiders: dict[str, frozenset] = {}
    base_cache: dict[tuple, tuple] = {}
    ref: dict[str, deque] = {}; ref_day = day0 // 86400
    first_day = sorted({(s["token"], s["pool_id"]) for s in swaps if s["ts"].timestamp() < day0 + 86400})
    if first_day and t0 > day0:                                    # the median's state at t0: each pool's last prices that day
        for h in conn.execute(
                "SELECT u.pool_id, h.price_q FROM unnest(%s::text[], %s::text[]) AS u(token, pool_id) CROSS JOIN LATERAL ("
                "SELECT s.price_q, s.ts, s.block, s.log_index FROM rh_swaps s WHERE s.token = u.token AND s.pool_id = u.pool_id "
                "AND s.ts >= to_timestamp(%s) AND s.ts < to_timestamp(%s) AND s.price_q > 0 "
                "ORDER BY s.ts DESC, s.block DESC, s.log_index DESC LIMIT %s) h ORDER BY u.pool_id, h.ts, h.block, h.log_index",
                ([t for t, _ in first_day], [p for _, p in first_day], day0, t0, REF_LEGS)).fetchall():
            ref.setdefault(h["pool_id"], deque(maxlen=REF_LEGS)).append(float(h["price_q"]))
    rows: dict[tuple, _Row] = {}
    for s in swaps:
        ts = s["ts"].timestamp(); day = int(ts // 86400)
        if day != ref_day:
            ref = {}; ref_day = day
        p = float(s["price_q"] or 0.0)
        if p <= 0:
            continue
        dq = ref.setdefault(s["pool_id"], deque(maxlen=REF_LEGS))
        ok = not dq or (statistics.median(dq) / PRICE_BAND <= p <= statistics.median(dq) * PRICE_BAND)
        dq.append(p)
        if not ok:
            continue
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
        r = rows.get((s["token"], m, s["pool_id"]))
        if r is None:
            r = rows[(s["token"], m, s["pool_id"])] = _Row(s["pool_id"], s["quote_asset"], s["cls"], base)
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
    best: dict[tuple, _Row] = {}                                   # per (coin, minute): the pool with the most swaps
    for (mint, m, _pool), r in rows.items():
        b = best.get((mint, m))
        if b is None or (r.nb + r.ns, r.buy + r.sell) > (b.nb + b.ns, b.buy + b.sell):
            best[(mint, m)] = r
    out = []
    for (mint, m), r in sorted(best.items(), key=lambda kv: (kv[0][1], kv[0][0])):
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
            "n_traders, resq_eth, resq_q, fee_rate, n_buyers, wash_eth, wash_buy_eth, top_sell_eth, insider_sell_eth, skill_buy, fee_v) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,2) ON CONFLICT (mint, ts) DO UPDATE SET fee_v = 2, "
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
    from ..agent import live_first
    live_first.minute_published()                       # committed: the engine's loop wakes now, not at its next poll
    return {"minutes": (t1 - done) // 60, "rows": n, "through": t1}
