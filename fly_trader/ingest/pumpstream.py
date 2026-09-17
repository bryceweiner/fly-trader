"""PumpAPI live stream → per-minute rich candles (Supervisor thread ``pumpstream``).

``wss://stream.pumpapi.io`` is a free, keyless firehose (~400–800 events/s, one connection per IP) whose events are
identical to the replay archive. This worker keeps only what the selector needs: PumpSwap (``pump-amm``, SOL-quoted)
buys and sells of pump.fun-origin mints, aggregated per (mint, minute) into the same fields ``train/mature.py``
derives from the archive, with the same filters: legs in a pool created directly rather than by a pump.fun migration
(``createPool`` with ``poolCreatedBy`` other than ``pump``; the owner can pull its unburned liquidity) are dropped — such pools
go to ``pump_pools`` as they are created, and the blocked set is reloaded from that table (archive backfill included) at
start and every 5 minutes —, the pool's quote reserve must lie in 0.001–100,000 SOL, a leg more
than 50× away from the median of the mint's previous 200 legs that UTC day is dropped, and each minute keeps the
pool with the most legs. Minutes are written to ``pump_minutes`` once complete: an event stamped ≥ 2 s after the
minute's end has arrived (event-time watermark) or, once no trade has arrived for 15 s, by wall clock; a leg for a minute
already written is dropped (never merged into it). ``pumpstream_status.flushed_through``
publishes the newest complete minute for the selector.

Lifecycle: ``create`` events go to ``pump_events`` as they arrive; at ``migrate`` a ``corpus_meta`` row is written
with the creation facts and the creator's point-in-time history (``train/corpus_meta.creator_history``, the training
definition); 61 minutes after a graduation its own 60-minute outcome is filled from the stream's minutes so later
graduations by the same creator see it, as in training. Minute rows older than ``PUMP_MINUTES_KEEP_DAYS`` are archived
to ``data/corpus/stream/<day>.parquet`` before removal. Database failures keep the data in memory for the next attempt.
"""
from __future__ import annotations

import asyncio
import json
import logging
import math
import ssl
import statistics
import threading
import time
from collections import defaultdict, deque
from datetime import datetime, timedelta, timezone

import certifi
import orjson
import pyarrow as pa
import pyarrow.parquet as pq
import websockets

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from ..train.corpus_meta import CURVE_COLS, blocked_pool_ids, creator_history, curve_facts
from ..train.flow import minute_wallet_cols

log = logging.getLogger(__name__)
WSOL = "So11111111111111111111111111111111111111112"
RESQ_BAND = (0.001, 100000.0)     # train/mature.py: quote_in_pool BETWEEN 0.001 AND 100000
PRICE_BAND = 50.0                  # train/mature.py: within 50x of the trailing median of the mint's previous 200 legs
REF_LEGS = 200
FLUSH_GRACE_S = 2.0
FLUSH_FALLBACK_S = 15.0
MAX_PENDING_CREATES = 200_000


def _txt(v):
    """Postgres text cannot hold NUL bytes; one in a token name would otherwise wedge the create queue."""
    return v.replace("\x00", "") if isinstance(v, str) else v


class Minute:
    __slots__ = ("open", "high", "low", "close", "buy", "sell", "nb", "ns", "traders", "wallets", "resq", "pool_id", "fee_rate")

    def __init__(self, pool_id: str | None = None):
        self.open = None; self.high = -math.inf; self.low = math.inf; self.close = None; self.buy = 0.0; self.sell = 0.0
        self.nb = 0; self.ns = 0; self.traders = set(); self.resq = None; self.pool_id = pool_id; self.fee_rate = None   # the pool fee charged (poolFeeRate)
        self.wallets: dict[str, list[float]] = {}                  # trader → [bought SOL, sold SOL] (train/flow.py)


class Aggregator:
    def __init__(self):
        self.minutes: dict[int, dict[str, dict[str, Minute]]] = defaultdict(lambda: defaultdict(dict))   # minute → mint → pool → Minute
        self.ref: dict[str, deque] = {}; self.ref_day: int | None = None                                  # trailing legs per mint, reset each UTC day
        self.max_event_s = 0.0; self.last_event_wall = 0.0                # newest event time; wall clock of the last trade received
        self.creates: dict[str, dict] = {}                                 # mint → create facts (24 h)
        self.curve: dict[str, dict] = {}                                   # mint → bonding-curve wallets since its create (corpus_meta.curve_facts)
        self.pending_creates: list[tuple[str, dict]] = []                 # creates not yet written to pump_events
        self.blocked: set[str] = set()                                     # pool ids of directly created (custom) PumpSwap pools
        self.pending_pools: list[tuple] = []                               # custom pools not yet written to pump_pools
        self.flushed_through: datetime | None = None                       # start of the newest complete minute written
        self.insiders: dict[str, frozenset] = {}                           # mint → creator/bundle/early wallets (token_insiders)
        self.skill: dict | None = None; self.skill_day: str | None = None  # wallet → skill decile for the current UTC day (train/wallet_skill.py)
        self.stats = {"events": 0, "trades": 0, "dropped_band": 0, "dropped_late": 0, "dropped_custom_pool": 0, "flushed_minutes": 0, "flushed_rows": 0, "creates": 0, "migrates": 0,
                      "custom_pools": 0, "blocked_pools": 0, "reconnects": 0, "outcomes_filled": 0, "started_at": datetime.now(timezone.utc).isoformat()}

    def row(self, minute: int, mint: str) -> Minute | None:
        """The minute's candle for a mint: the pool with the most legs (as in training)."""
        pools = self.minutes.get(minute, {}).get(mint)
        return max(pools.values(), key=lambda r: (r.nb + r.ns, r.buy + r.sell)) if pools else None

    def ingest(self, e: dict) -> None:
        self.stats["events"] += 1
        a = e.get("action"); pool = e.get("pool"); mint = e.get("mint"); ts = e.get("timestamp")
        if a == "createPool" and pool == "pump-amm" and (e.get("poolCreatedBy") or "custom") != "pump":
            pid = e.get("poolId")                   # not a pump.fun migration: the owner can pull the liquidity unseen
            if pid:
                self.block([pid]); self.stats["custom_pools"] += 1
                self.pending_pools.append((pid, mint, e.get("poolCreatedBy"), datetime.fromtimestamp(ts / 1000, timezone.utc) if ts is not None else None))
            return
        if ts is None or not mint:
            return
        if a in ("buy", "sell") and pool == "pump":
            self._curve(e)
            return
        if a in ("buy", "sell"):
            if pool != "pump-amm" or e.get("quoteMint") != WSOL or not str(mint).endswith("pump"):
                return
            if e.get("poolId") in self.blocked:
                self.stats["dropped_custom_pool"] += 1
                return
            price = e.get("price"); q = e.get("quoteInPool")
            if not price or price <= 0 or q is None or not (RESQ_BAND[0] <= float(q) <= RESQ_BAND[1]):
                return
            ts_s = ts / 1000.0; self.max_event_s = max(self.max_event_s, ts_s); self.last_event_wall = time.time()
            day = int(ts_s // 86400)
            if self.ref_day is None or day > self.ref_day:      # a late event from the previous day never wipes today's references
                self.ref = {}; self.ref_day = day
            p = float(price)
            legs = e.get("breakdown") or [{"action": a, "trader": e.get("txSigner"), "quoteAmount": e.get("quoteAmount")}]
            dq = self.ref.get(mint)
            if dq is None:
                dq = self.ref[mint] = deque(maxlen=REF_LEGS)
            ok = True
            if dq:
                pref = statistics.median(dq)
                ok = pref / PRICE_BAND <= p <= pref * PRICE_BAND
            for _ in legs:
                dq.append(p)
            if not ok:
                self.stats["dropped_band"] += 1
                return
            m = int(ts_s // 60) * 60; pid = e.get("poolId") or ""
            if self.flushed_through is not None and m <= self.flushed_through.timestamp():
                self.stats["dropped_late"] += 1           # its minute is written and consumed; a partial re-write would corrupt it
                return
            row = self.minutes[m][mint].get(pid)
            if row is None:
                row = self.minutes[m][mint][pid] = Minute(pid or None)
            row.open = p if row.open is None else row.open; row.high = max(row.high, p); row.low = min(row.low, p); row.close = p
            for b in legs:
                sol = float(b.get("quoteAmount") or 0.0)
                if b.get("action") == "buy":
                    row.buy += sol; row.nb += 1
                else:
                    row.sell += sol; row.ns += 1
                trader = b.get("trader") or e.get("txSigner")          # as replay_pull writes the archive's trader column
                if trader:
                    row.traders.add(trader)
                    row.wallets.setdefault(trader, [0.0, 0.0])[0 if b.get("action") == "buy" else 1] += sol
            row.resq = float(q)
            if e.get("poolFeeRate") is not None:
                row.fee_rate = float(e["poolFeeRate"])
            self.stats["trades"] += 1
        elif a == "create" and pool == "pump":
            c = {"ts": ts, "creator": e.get("txSigner"), "dev_sol": e.get("quoteAmount"), "dev_tokens": e.get("initialBuy"), "supply": e.get("supply"),
                 "mayhem": e.get("mayhemMode"), "name": _txt(e.get("name")), "symbol": _txt(e.get("symbol")), "uri": _txt(e.get("uri")), "sig": e.get("signature")}
            self.creates[mint] = c; self.pending_creates.append((mint, c))
            self.curve[mint] = {"slot": int(e.get("block") or 0), "creator": e.get("txSigner"), "supply": e.get("supply"), "initial_buy": e.get("initialBuy"), "ts": ts, "wallets": {}}
            self.stats["creates"] += 1
            if len(self.creates) > 200_000:
                cutoff = ts - 86_400_000
                self.creates = {k: v for k, v in self.creates.items() if v["ts"] >= cutoff}
                self.curve = {k: v for k, v in self.curve.items() if v["ts"] >= cutoff}
        elif a == "migrate" and pool == "pump-amm":
            self.stats["migrates"] += 1
            self.write_creates()                    # the creator's own earlier creates must be visible to the history query
            self._graduation(e)

    def _curve(self, e: dict) -> None:
        """A bonding-curve buy/sell of a token created while the stream ran: per wallet its first buy slot and tokens bought/sold."""
        st = self.curve.get(e.get("mint"))
        if st is None:
            return                                   # created before this process started: the archive fills its facts (fail closed until then)
        slot = int(e.get("block") or 0)
        for b in e.get("breakdown") or [{"action": e.get("action"), "trader": e.get("txSigner"), "tokenAmount": e.get("tokenAmount")}]:
            w = b.get("trader") or e.get("txSigner")
            if not w:
                continue
            rec = st["wallets"].setdefault(w, [None, 0.0, 0.0]); tok = float(b.get("tokenAmount") or 0.0)
            if (b.get("action") or e.get("action")) == "buy":
                rec[1] += tok; rec[0] = slot if rec[0] is None else min(rec[0], slot)
            else:
                rec[2] += tok

    def write_creates(self) -> int:
        """Persist creates as they arrive (keyed by signature); on failure they stay queued for the next attempt."""
        rows, self.pending_creates = self.pending_creates, []
        if not rows:
            return 0
        try:
            with transaction() as conn:
                conn.cursor().executemany("INSERT INTO pump_events (sig, ts, action, pool, mint, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri) "
                                          "VALUES (%s,%s,'create','pump',%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sig) DO NOTHING",
                                          [(c["sig"], datetime.fromtimestamp(c["ts"] / 1000, timezone.utc), m, c["creator"], c["dev_sol"], c["dev_tokens"], c["supply"],
                                            c["mayhem"], c["name"], c["symbol"], c["uri"]) for m, c in rows if c.get("sig")])
        except Exception:
            self.pending_creates = (rows + self.pending_creates)[-MAX_PENDING_CREATES:]
            log.exception("create rows failed; %d queued for the next attempt", len(self.pending_creates))
            return 0
        return len(rows)

    def block(self, pool_ids) -> None:
        self.blocked.update(pool_ids); self.stats["blocked_pools"] = len(self.blocked)

    def write_pools(self) -> int:
        """Persist custom pools seen on the stream (training excludes the same pools); on failure they stay queued."""
        rows, self.pending_pools = self.pending_pools, []
        if not rows:
            return 0
        try:
            with transaction() as conn:
                conn.cursor().executemany("INSERT INTO pump_pools (pool_id, mint, created_by, ts, source) VALUES (%s,%s,%s,%s,'stream') ON CONFLICT (pool_id) DO NOTHING", rows)
        except Exception:
            self.pending_pools = (rows + self.pending_pools)[-MAX_PENDING_CREATES:]
            log.exception("custom pool rows failed; %d queued for the next attempt", len(self.pending_pools))
            return 0
        return len(rows)

    def _graduation(self, e: dict) -> None:
        mint = e["mint"]; g = datetime.fromtimestamp(e["timestamp"] / 1000, timezone.utc); c = self.creates.get(mint)
        try:
            with transaction() as conn:
                if c is None:                                       # created before this process started: look it up
                    r = conn.execute("SELECT sig, ts, signer, dev_sol, dev_tokens, supply, mayhem, name, symbol, uri FROM pump_events WHERE mint = %s AND action = 'create' ORDER BY ts LIMIT 1", (mint,)).fetchone()
                    if r:
                        c = {"ts": r["ts"].timestamp() * 1000, "creator": r["signer"], "dev_sol": r["dev_sol"], "dev_tokens": r["dev_tokens"], "supply": r["supply"],
                             "mayhem": r["mayhem"], "name": r["name"], "symbol": r["symbol"], "uri": r["uri"], "sig": r["sig"]}
                conn.execute("INSERT INTO pump_events (sig, ts, action, pool, mint, pool_id, signer, quote_in_pool) VALUES (%s,%s,'migrate','pump-amm',%s,%s,%s,%s) ON CONFLICT (sig) DO NOTHING",
                             (e.get("signature"), g, mint, e.get("poolId"), e.get("txSigner"), e.get("quoteInPool")))
                ct = datetime.fromtimestamp(c["ts"] / 1000, timezone.utc) if c else None
                h = creator_history(conn, c.get("creator") if c else None, ct, g)
                conn.execute("""INSERT INTO corpus_meta (mint, graduated_at, create_ts, creator, dev_sol, dev_tokens, dev_share, supply, mayhem, uri, name, symbol, ttg_min, rq0, pool_id,
                                                         prior_launches, prior_grads, prior_known, prior_rug_share, prior_moon_share)
                                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (mint) DO UPDATE SET
                                  graduated_at = COALESCE(corpus_meta.graduated_at, EXCLUDED.graduated_at),
                                  create_ts = COALESCE(corpus_meta.create_ts, EXCLUDED.create_ts), creator = COALESCE(corpus_meta.creator, EXCLUDED.creator),
                                  dev_sol = COALESCE(corpus_meta.dev_sol, EXCLUDED.dev_sol), dev_tokens = COALESCE(corpus_meta.dev_tokens, EXCLUDED.dev_tokens),
                                  dev_share = COALESCE(corpus_meta.dev_share, EXCLUDED.dev_share), supply = COALESCE(corpus_meta.supply, EXCLUDED.supply),
                                  mayhem = COALESCE(corpus_meta.mayhem, EXCLUDED.mayhem), uri = COALESCE(corpus_meta.uri, EXCLUDED.uri),
                                  name = COALESCE(corpus_meta.name, EXCLUDED.name), symbol = COALESCE(corpus_meta.symbol, EXCLUDED.symbol),
                                  ttg_min = COALESCE(corpus_meta.ttg_min, EXCLUDED.ttg_min), rq0 = COALESCE(corpus_meta.rq0, EXCLUDED.rq0), pool_id = COALESCE(corpus_meta.pool_id, EXCLUDED.pool_id),
                                  prior_launches = COALESCE(corpus_meta.prior_launches, EXCLUDED.prior_launches), prior_grads = COALESCE(corpus_meta.prior_grads, EXCLUDED.prior_grads),
                                  prior_known = COALESCE(corpus_meta.prior_known, EXCLUDED.prior_known), prior_rug_share = COALESCE(corpus_meta.prior_rug_share, EXCLUDED.prior_rug_share),
                                  prior_moon_share = COALESCE(corpus_meta.prior_moon_share, EXCLUDED.prior_moon_share), updated_at = now()""",
                             (mint, g, ct, c.get("creator") if c else None, c.get("dev_sol") if c else None, c.get("dev_tokens") if c else None,
                              (float(c["dev_tokens"]) / float(c["supply"])) if c and c.get("dev_tokens") and c.get("supply") else None,
                              c.get("supply") if c else None, c.get("mayhem") if c else None, c.get("uri") if c else None, c.get("name") if c else None, c.get("symbol") if c else None,
                              ((e["timestamp"] - c["ts"]) / 60000.0) if c else None, e.get("quoteInPool"), e.get("poolId"),
                              h["prior_launches"], h["prior_grads"], h["prior_known"], h["prior_rug_share"], h["prior_moon_share"]))
                conn.execute("INSERT INTO corpus_tokens (mint, graduated_at, grad_slot, source, status) VALUES (%s,%s,%s,'stream','pending') ON CONFLICT (mint) DO NOTHING",
                             (mint, g, int(e.get("block") or 0)))
                st = self.curve.pop(mint, None)
                if st is not None:                           # the token's whole curve life was seen: the archive's definition (curve_facts)
                    facts, insiders = curve_facts(st["slot"], st["creator"], st["supply"], st["initial_buy"], st["wallets"])
                    conn.execute("UPDATE corpus_meta SET " + ", ".join(f"{c} = COALESCE({c}, %s)" for c in CURVE_COLS) + ", curve_known = COALESCE(curve_known, true) WHERE mint = %s",
                                 (*[facts[c] for c in CURVE_COLS], mint))
                    conn.cursor().executemany("INSERT INTO token_insiders (mint, wallet, kind) VALUES (%s,%s,%s) ON CONFLICT (mint, wallet) DO NOTHING",
                                              [(mint, w, k) for w, k in insiders])
                    self.insiders[mint] = frozenset(w for w, _ in insiders)
        except Exception:
            log.exception("graduation upsert failed for %s", mint)

    def flush(self, before_minute: int) -> int:
        """Write every minute strictly older than ``before_minute`` (seconds). Rows leave memory only after the commit."""
        done = sorted(m for m in self.minutes if m < before_minute)
        if not done:
            return 0
        rows = []
        for m in done:
            ts = datetime.fromtimestamp(m, timezone.utc)
            for mint in self.minutes[m]:
                r = self.row(m, mint)
                if r is None or r.close is None:
                    continue
                wc = minute_wallet_cols(r.wallets, self.insiders.get(mint, frozenset()), self.skill)
                rows.append((mint, ts, r.pool_id, r.open, r.high, r.low, r.close, r.buy, r.sell, r.nb, r.ns, len(r.traders), r.resq, r.fee_rate,
                             wc["n_buyers"], wc["wash_sol"], wc["wash_buy_sol"], wc["top_sell_sol"], wc["insider_sell_sol"], wc["skill_buy"]))
        if rows:
            with transaction() as conn:
                conn.cursor().executemany("INSERT INTO pump_minutes (mint, ts, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol, fee_rate, "
                                          "n_buyers, wash_sol, wash_buy_sol, top_sell_sol, insider_sell_sol, skill_buy) "
                                          "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (mint, ts) DO UPDATE SET close = EXCLUDED.close, high = greatest(pump_minutes.high, EXCLUDED.high), "
                                          "low = least(pump_minutes.low, EXCLUDED.low), buy_sol = pump_minutes.buy_sol + EXCLUDED.buy_sol, sell_sol = pump_minutes.sell_sol + EXCLUDED.sell_sol, "
                                          "n_buys = pump_minutes.n_buys + EXCLUDED.n_buys, n_sells = pump_minutes.n_sells + EXCLUDED.n_sells, n_traders = greatest(pump_minutes.n_traders, EXCLUDED.n_traders), "
                                          "resq_sol = EXCLUDED.resq_sol, fee_rate = COALESCE(EXCLUDED.fee_rate, pump_minutes.fee_rate), n_buyers = EXCLUDED.n_buyers, wash_sol = EXCLUDED.wash_sol, "
                                          "wash_buy_sol = EXCLUDED.wash_buy_sol, top_sell_sol = EXCLUDED.top_sell_sol, insider_sell_sol = EXCLUDED.insider_sell_sol, skill_buy = EXCLUDED.skill_buy", rows)
        for m in done:
            self.minutes.pop(m, None)
        self.stats["flushed_minutes"] += len(done); self.stats["flushed_rows"] += len(rows)
        newest = datetime.fromtimestamp(done[-1], timezone.utc)
        if self.flushed_through is None or newest > self.flushed_through:          # a late event for an old minute never rewinds it
            self.flushed_through = newest
        return len(rows)

    def flush_bound(self, now: float) -> int:
        """Minutes strictly before this start are complete: event-time watermark, or the wall-clock fallback once no trade
        has arrived for ``FLUSH_FALLBACK_S`` (a lagging stream still delivers the minute's events, so it waits for them)."""
        cur = int(now // 60) * 60
        wm = int((self.max_event_s - FLUSH_GRACE_S) // 60) * 60
        if now - self.last_event_wall >= FLUSH_FALLBACK_S:
            wm = max(wm, int((now - FLUSH_FALLBACK_S) // 60) * 60)
        return min(cur, wm)


def fill_outcomes() -> int:
    """Own 60-minute outcome for stream graduations 61+ minutes old, from the stream's minutes (candle files replace it later)."""
    with transaction() as conn:
        todo = conn.execute("""SELECT mint, graduated_at FROM corpus_meta WHERE own_dd60 IS NULL AND graduated_at IS NOT NULL
                               AND graduated_at BETWEEN now() - interval '6 hours' AND now() - interval '61 minutes'""").fetchall()
        n = 0
        for t in todo:
            rows = conn.execute("SELECT close FROM pump_minutes WHERE mint = %s AND ts >= %s AND ts < %s ORDER BY ts",
                                (t["mint"], t["graduated_at"], t["graduated_at"] + timedelta(minutes=60))).fetchall()
            if not rows:
                continue
            closes = [float(r["close"]) for r in rows]; p0 = closes[0]
            if p0 <= 0:
                continue
            conn.execute("UPDATE corpus_meta SET own_dd60 = %s, own_max60 = %s, updated_at = now() WHERE mint = %s AND own_dd60 IS NULL",
                         (min(closes) / p0 - 1.0, max(closes) / p0 - 1.0, t["mint"]))
            n += 1
    return n


def archive_old(keep_days: int) -> int:
    """Archive minute rows older than ``keep_days`` to Parquet, then remove them from the table."""
    cutoff = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=keep_days)
    with transaction() as conn:
        days = [r["d"] for r in conn.execute("SELECT DISTINCT date_trunc('day', ts) AS d FROM pump_minutes WHERE ts < %s", (cutoff,)).fetchall()]
    n = 0
    for d in days:
        with transaction() as conn:
            rows = conn.execute("SELECT * FROM pump_minutes WHERE ts >= %s AND ts < %s ORDER BY mint, ts", (d, d + timedelta(days=1))).fetchall()
            if rows:
                out = config.CORPUS_DIR / "stream"; out.mkdir(parents=True, exist_ok=True)
                path = out / f"{d:%Y-%m-%d}.parquet"; tmp = path.with_name(path.name + ".tmp")
                pq.write_table(pa.Table.from_pylist([dict(r) for r in rows]), tmp, compression="zstd"); tmp.replace(path)
                conn.execute("DELETE FROM pump_minutes WHERE ts >= %s AND ts < %s", (d, d + timedelta(days=1)))
                n += len(rows)
    return n


def _status(agg: Aggregator, extra: dict) -> None:
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('pumpstream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                         (json.dumps({**agg.stats, **extra, "updated_at": datetime.now(timezone.utc).isoformat()}, default=str),))
    except Exception:
        log.debug("pumpstream status write failed", exc_info=True)


def _logged(name: str, fn, *args):
    try:
        n = fn(*args)
        if n:
            log.info("%s: %d", name, n)
        return n
    except Exception:
        log.exception("%s failed", name)
        return 0


def _load_insiders(agg: Aggregator) -> None:
    """Insider wallets (creator, bundle, early buyers: token_insiders) of every mint traded today not loaded yet; a token's
    set is fixed at its graduation, so loaded sets are kept."""
    todo = [m for m in agg.ref if m not in agg.insiders]
    if not todo:
        return
    try:
        with transaction() as conn:
            rows = conn.execute("SELECT mint, wallet FROM token_insiders WHERE mint = ANY(%s)", (todo,)).fetchall()
    except Exception:
        log.exception("insider load failed; retried in 1 min")
        return
    got: dict[str, set] = {}
    for r in rows:
        got.setdefault(r["mint"], set()).add(r["wallet"])
    for m in todo:
        agg.insiders[m] = frozenset(got.get(m, ()))


def load_skill(agg: Aggregator, now: float | None = None) -> None:
    """The wallet-skill table of the current UTC day (the file training joins for that day), once per day."""
    from ..train.mature import SKILL_DIR
    day = datetime.fromtimestamp(now or time.time(), timezone.utc).date().isoformat()
    if agg.skill_day == day and agg.skill is not None:
        return
    f = SKILL_DIR / f"{day}.parquet"
    if not f.exists():
        agg.skill = None; agg.skill_day = day
        return
    t = pq.read_table(f, columns=["wallet", "bucket"])
    agg.skill = dict(zip(t["wallet"].to_pylist(), t["bucket"].to_pylist())); agg.skill_day = day
    log.info("wallet skill table for %s loaded: %d wallets", day, len(agg.skill))


def _load_blocked(agg: Aggregator) -> None:
    """Union the custom pools in ``pump_pools`` (stream rows and the archive backfill) into the blocked set."""
    try:
        with transaction() as conn:
            agg.block(blocked_pool_ids(conn))
    except Exception:
        log.exception("custom pool load failed; retried in 5 min")


async def _run(stop: threading.Event | None, agg: Aggregator) -> None:
    ctx = ssl.create_default_context(cafile=certifi.where()); backoff = 1.0
    _load_blocked(agg); last_pools = time.time(); last_ins = 0.0
    try:
        load_skill(agg)
    except Exception:
        log.exception("wallet skill load failed; retried in 5 min")
    last_status = last_flush = last_outcomes = 0.0; last_archive = time.time() - 3000; last_msg = 0.0; rate_n = 0; rate_t = time.time()
    bg: dict[str, asyncio.Task | None] = {"archive": None, "outcomes": None}
    while not (stop is not None and stop.is_set()):
        try:
            async with websockets.connect(config.PUMPSTREAM_URL, ssl=ctx, max_size=8_000_000, ping_interval=20, ping_timeout=20) as ws:
                log.info("pumpstream connected"); backoff = 1.0
                while not (stop is not None and stop.is_set()):
                    try:
                        msg = await asyncio.wait_for(ws.recv(), timeout=30)
                    except asyncio.TimeoutError:
                        log.warning("pumpstream: no events for 30 s; reconnecting"); break
                    last_msg = time.time(); rate_n += 1
                    try:
                        e = orjson.loads(msg)
                    except orjson.JSONDecodeError:
                        continue
                    if isinstance(e, dict):
                        agg.ingest(e)
                    now = time.time()
                    if now - last_flush >= 1.0:
                        try:
                            agg.flush(agg.flush_bound(now))
                        except Exception:
                            log.exception("minute flush failed; rows kept for the next attempt")
                        last_flush = now
                    if now - last_pools >= 300:
                        _load_blocked(agg); last_pools = now
                        try:
                            load_skill(agg, now)
                        except Exception:
                            log.exception("wallet skill load failed; retried in 5 min")
                    if now - last_ins >= 60:
                        _load_insiders(agg); last_ins = now
                    if now - last_status >= 5.0:
                        agg.write_creates(); agg.write_pools()
                        _status(agg, {"events_per_s": rate_n / max(now - rate_t, 1e-9), "pending_minutes": len(agg.minutes), "lag_s": now - last_msg, "connected": True,
                                      "flushed_through": agg.flushed_through.isoformat() if agg.flushed_through else None,
                                      "event_watermark": datetime.fromtimestamp(agg.max_event_s, timezone.utc).isoformat() if agg.max_event_s else None})
                        rate_n = 0; rate_t = now; last_status = now
                    if now - last_outcomes >= 60 and (bg["outcomes"] is None or bg["outcomes"].done()):
                        bg["outcomes"] = asyncio.create_task(asyncio.to_thread(_logged, "stream outcomes filled", fill_outcomes)); last_outcomes = now
                    if now - last_archive >= 3600 and (bg["archive"] is None or bg["archive"].done()):
                        bg["archive"] = asyncio.create_task(asyncio.to_thread(_logged, "archived old minute rows", archive_old, config.PUMP_MINUTES_KEEP_DAYS)); last_archive = now
        except Exception as e:
            agg.stats["reconnects"] += 1; log.warning("pumpstream: %s; reconnecting in %.0fs", e, backoff)
            _status(agg, {"connected": False, "last_error": str(e)[:200]})
            await asyncio.sleep(backoff); backoff = min(60.0, backoff * 2)
    for t in bg.values():
        if t is not None:
            try:
                await t
            except Exception:
                pass
    try:
        agg.flush(int(time.time() // 60) * 60 + 60)
    except Exception:
        log.exception("final minute flush failed")
    agg.write_creates(); agg.write_pools()


def main(stop_event: threading.Event | None = None) -> None:
    setup("pumpstream"); agg = Aggregator()
    record_event("info", "pumpstream", "pumpstream started", {"url": config.PUMPSTREAM_URL})
    try:
        asyncio.run(_run(stop_event, agg))
    finally:
        _status(agg, {"connected": False, "stage": "stopped"})
        record_event("info", "pumpstream", "pumpstream stopped", {k: agg.stats.get(k) for k in ("events", "trades", "dropped_band", "dropped_custom_pool", "flushed_rows", "creates",
                                                                                                  "custom_pools", "migrates", "reconnects")})
