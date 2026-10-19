"""Our own Solana feed for the live minutes: pump.fun and PumpSwap events from Helius's standard websockets, decoded into
the event dicts PumpAPI's stream sends, so ingest/pumpstream.Aggregator builds the same minutes training reads from
PumpAPI's archive (checked field by field against it on 2026-10-19).

- Transport: ``logsSubscribe`` (processed) on each program, one connection each. A transaction's logs carry its events
  as ``Program data:`` lines, but Solana cuts logs at 10 KB: every migration and the wash bots' 4-swap transactions lose
  events that way. A cut-off transaction is fetched whole (``getTransaction``, confirmed) and decoded from its inner
  instructions (the events' self-CPI copies) instead; its logs are never used.
- PumpSwap trades name their pool, not their mints: ``Pools`` maps pool -> (base, quote) mint from the stream's recent
  minutes, the pool creations seen, and the pool accounts themselves (``getMultipleAccounts``) for a pool first seen.
- PumpAPI's definitions (archive): a trade's SOL is the pool-side quote amount; ``quoteInPool`` and the price are the
  pool's state after the transaction's last swap in that pool, the price counting the pool's virtual quote reserve;
  the trader is the event's user; the timestamp is ``TS_OFFSET_S`` after the block's second (PumpAPI stamps on receipt);
  a create's dev buy is its creator's buys in the same transaction and a mayhem-mode coin's supply counts the agent's
  billion; a migration's ``quoteInPool`` is the pool's real quote after its boost (``InitBoostEvent``).
- ``pending``: per minute, transactions still being fetched or waiting for their pool: the minute's flush waits for them
  (``Aggregator.flush_bound``), at most ``pumpstream.HOLD_MAX_S``.
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import logging
import ssl
import time
from collections import Counter

import base58
import certifi
import orjson
import websockets

from .. import config

log = logging.getLogger(__name__)
PUMP = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"          # pump.fun bonding curves
AMM = "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA"           # PumpSwap
WSOL = "So11111111111111111111111111111111111111112"
TS_OFFSET_S = 0.9            # PumpAPI's stamp after the block's second (archive ts - blockTime: p05 0.79, p50 0.90, p95 1.12 s)
EVENT_IX_TAG = bytes.fromhex("e445a52e51cb9a1d")              # Anchor's self-CPI event instruction
EVENTS = ("BuyEvent", "SellEvent", "CreatePoolEvent", "InitBoostEvent", "TradeEvent", "CreateEvent", "CompletePumpAmmMigrationEvent")
DISC = {hashlib.sha256(f"event:{n}".encode()).digest()[:8]: n for n in EVENTS}
BACKFILL_RETRY_S, BACKFILL_GIVE_UP_S, BACKFILL_BATCH = 0.3, 8.0, 25
RESOLVE_BATCH, STALE_S = 100, 5.0


def _u64(b: bytes, o: int) -> int:
    return int.from_bytes(b[o:o + 8], "little")


def _pk(b: bytes, o: int) -> str:
    return base58.b58encode(b[o:o + 32]).decode()


def _str(b: bytes, o: int) -> tuple[str, int]:
    n = int.from_bytes(b[o:o + 4], "little")
    return b[o + 4:o + 4 + n].decode("utf-8", "replace").replace("\x00", ""), o + 4 + n


def log_events(logs: list[str]) -> list[tuple[str, bytes]]:
    """(event name, payload) of every known event in a transaction's logs, in order."""
    out = []
    for line in logs or ():
        if line.startswith("Program data: "):
            try:
                b = base64.b64decode(line[14:])
            except ValueError:
                continue
            n = DISC.get(b[:8])
            if n:
                out.append((n, b[8:]))
    return out


def cpi_events(tx: dict) -> list[tuple[str, bytes]]:
    """The same from a whole transaction (getTransaction, json): the events' self-CPI copies in its inner instructions,
    which the 10 KB log limit never cuts."""
    meta = tx.get("meta") or {}
    la = meta.get("loadedAddresses") or {}
    keys = list(tx["transaction"]["message"]["accountKeys"]) + list(la.get("writable") or []) + list(la.get("readonly") or [])
    out = []
    for grp in sorted(meta.get("innerInstructions") or [], key=lambda g: g.get("index", 0)):
        for ix in grp.get("instructions") or []:
            if keys[ix["programIdIndex"]] not in (PUMP, AMM):
                continue
            d = base58.b58decode(ix["data"])
            if d[:8] == EVENT_IX_TAG:
                n = DISC.get(d[8:16])
                if n:
                    out.append((n, d[16:]))
    return out


def amm_trade(name: str, p: bytes) -> dict:
    """A PumpSwap Buy/SellEvent: the fields PumpAPI's trade needs (raw units). The virtual quote reserve sits after the
    buy's variable-length instruction name; a sale has none."""
    if name == "SellEvent":
        vq_at = 384
    else:
        vq_at = 397 + int.from_bytes(p[393:397], "little") + 32
    vq = int.from_bytes(p[vq_at:vq_at + 16], "little", signed=True) if len(p) >= vq_at + 16 else 0
    return {"side": "buy" if name == "BuyEvent" else "sell", "ts": _u64(p, 0), "base": _u64(p, 8), "pbr": _u64(p, 40), "pqr": _u64(p, 48),
            "q": _u64(p, 56), "fee_bps": _u64(p, 64) + _u64(p, 80) + _u64(p, 336), "pool": _pk(p, 112), "user": _pk(p, 144), "vq": max(vq, 0)}


def curve_trade(p: bytes) -> dict:
    return {"mint": _pk(p, 0), "sol": _u64(p, 32), "token": _u64(p, 40), "buy": bool(p[48]), "user": _pk(p, 49), "ts": _u64(p, 81)}


def create(p: bytes) -> dict:
    name, o = _str(p, 0); symbol, o = _str(p, o); uri, o = _str(p, o)
    return {"name": name, "symbol": symbol, "uri": uri, "mint": _pk(p, o), "user": _pk(p, o + 64), "creator": _pk(p, o + 96),
            "ts": _u64(p, o + 128), "supply": _u64(p, o + 160), "mayhem": bool(p[o + 200]) if len(p) > o + 200 else False}


def create_pool(p: bytes) -> dict:
    return {"ts": _u64(p, 0), "creator": _pk(p, 10), "base": _pk(p, 42), "quote": _pk(p, 74), "pool_quote": _u64(p, 132), "pool": _pk(p, 165)}


def migration(p: bytes) -> dict:
    return {"user": _pk(p, 0), "mint": _pk(p, 32), "sol": _u64(p, 72), "ts": _u64(p, 120), "pool": _pk(p, 128)}


def init_boost(p: bytes) -> dict:
    return {"ts": _u64(p, 0), "mint": _pk(p, 8), "pool": _pk(p, 72), "real_quote_after": _u64(p, 120)}


def _ms(ts: int) -> float:
    return (ts + TS_OFFSET_S) * 1000.0


def build(sig: str, slot: int, evs: list[tuple[str, bytes]], pools: dict, payer: str | None = None) -> tuple[list[dict], set]:
    """PumpAPI-shaped events of one transaction, and the PumpSwap pools whose mints are not known yet (then nothing is
    built: the caller resolves them and builds again)."""
    trades, curve, creates, cpools = [], [], [], []
    mig = boost = None
    for n, p in evs:
        try:
            if n in ("BuyEvent", "SellEvent"):
                trades.append(amm_trade(n, p))
            elif n == "TradeEvent":
                curve.append(curve_trade(p))
            elif n == "CreateEvent":
                creates.append(create(p))
            elif n == "CreatePoolEvent":
                cpools.append(create_pool(p))
            elif n == "CompletePumpAmmMigrationEvent":
                mig = migration(p)
            elif n == "InitBoostEvent":
                boost = init_boost(p)
        except (IndexError, ValueError):
            log.warning("undecodable %s in %s", n, sig[:12])
    for c in cpools:                                     # a pool created here names its mints for the trades after it
        pools[c["pool"]] = (c["base"], c["quote"])
    unknown = {t["pool"] for t in trades if t["pool"] not in pools}
    if unknown:
        return [], unknown
    out: list[dict] = []
    for c in creates:                                    # the create first: the curve state its dev buy lands in
        dev = [t for t in curve if t["mint"] == c["mint"] and t["buy"] and t["user"] in (c["user"], c["creator"])]
        out.append({"action": "create", "pool": "pump", "mint": c["mint"], "timestamp": _ms(c["ts"]), "txSigner": c["user"],
                    "quoteAmount": sum(t["sol"] for t in dev) / 1e9, "initialBuy": sum(t["token"] for t in dev) / 1e6,
                    "supply": c["supply"] / 1e6 * (2 if c["mayhem"] else 1), "mayhemMode": c["mayhem"], "name": c["name"], "symbol": c["symbol"],
                    "uri": c["uri"], "signature": sig, "block": slot})
    by_mint: dict[str, list] = {}
    for t in curve:
        by_mint.setdefault(t["mint"], []).append(t)
    for mint, legs in by_mint.items():
        out.append({"action": "buy" if legs[0]["buy"] else "sell", "pool": "pump", "mint": mint, "timestamp": _ms(legs[0]["ts"]), "block": slot,
                    "txSigner": legs[0]["user"], "tokenAmount": legs[0]["token"] / 1e6, "signature": sig,
                    "breakdown": [{"action": "buy" if t["buy"] else "sell", "trader": t["user"], "tokenAmount": t["token"] / 1e6} for t in legs]})
    by_pool: dict[str, list] = {}
    for t in trades:
        by_pool.setdefault(t["pool"], []).append(t)
    for pool, legs in by_pool.items():
        base, quote = pools[pool]
        last = legs[-1]; sgn = 1 if last["side"] == "buy" else -1
        post_q, post_b = last["pqr"] + sgn * last["q"], last["pbr"] - sgn * last["base"]
        dec = 6 if base.endswith("pump") else None       # only pump.fun coins (6 decimals) are priced: the minutes keep no other
        out.append({"action": legs[0]["side"], "pool": "pump-amm", "mint": base, "quoteMint": quote, "poolId": pool, "timestamp": _ms(legs[0]["ts"]),
                    "price": ((post_q + last["vq"]) / 1e9) / (post_b / 10 ** dec) if dec is not None and post_b > 0 else None,
                    "quoteInPool": post_q / 1e9, "quoteAmount": legs[0]["q"] / 1e9, "txSigner": legs[0]["user"], "poolFeeRate": last["fee_bps"] / 1e4,
                    "signature": sig, "block": slot,
                    "breakdown": [{"action": t["side"], "trader": t["user"], "quoteAmount": t["q"] / 1e9} for t in legs]})
    for c in cpools:
        if mig is not None and mig["pool"] == c["pool"]:
            out.append({"action": "migrate", "pool": "pump-amm", "mint": mig["mint"], "timestamp": _ms(mig["ts"]), "signature": sig, "poolId": c["pool"],
                        "txSigner": payer or mig["user"], "block": slot,
                        "quoteInPool": (boost["real_quote_after"] if boost and boost["pool"] == c["pool"] else c["pool_quote"]) / 1e9})
        else:                                            # created directly, not by a pump.fun migration: the owner can pull it
            out.append({"action": "createPool", "pool": "pump-amm", "poolId": c["pool"], "mint": c["base"], "poolCreatedBy": c["creator"],
                        "timestamp": _ms(c["ts"]), "signature": sig})
    return out, set()


def minute_of(evs: list[tuple[str, bytes]]) -> int | None:
    """The minute (start, epoch s) a transaction's events are stamped in, from the first event carrying a timestamp."""
    for n, p in evs:
        try:
            ts = {"BuyEvent": 0, "SellEvent": 0, "CreatePoolEvent": 0, "InitBoostEvent": 0}.get(n)
            t = _u64(p, ts) if ts is not None else curve_trade(p)["ts"] if n == "TradeEvent" else None
        except (IndexError, ValueError):
            t = None
        if t:
            return int((t + TS_OFFSET_S) // 60) * 60
    return None


class Pools:
    """pool -> (base mint, quote mint) of PumpSwap pools: seeded from the stream's recent minutes, learned from pool
    creations, else read from the pool account (``base_mint`` at byte 43, ``quote_mint`` at 75)."""

    def __init__(self):
        self.map: dict[str, tuple[str, str]] = {}

    def seed(self) -> int:
        from ..db.connection import transaction
        try:
            with transaction() as conn:
                rows = conn.execute("SELECT DISTINCT ON (pool_id) pool_id, mint FROM pump_minutes WHERE pool_id IS NOT NULL AND ts > now() - interval '3 days'").fetchall()
        except Exception:
            log.exception("pool map seed failed; pools are read as they come")
            return 0
        for r in rows:
            self.map.setdefault(r["pool_id"], (r["mint"], WSOL))
        return len(rows)

    def resolve(self, rpc, pools: list[str]) -> int:
        """Read unknown pool accounts (``rpc``: chain/rpc.HttpSolanaRpc); a pool that is not a PumpSwap pool maps to
        ('', '') so its trades are dropped."""
        n = 0
        for i in range(0, len(pools), RESOLVE_BATCH):
            chunk = pools[i:i + RESOLVE_BATCH]
            v = (rpc.call("getMultipleAccounts", [chunk, {"encoding": "base64", "commitment": "processed"}]) or {}).get("value") or []
            for pool, acc in zip(chunk, v):
                if acc and acc.get("owner") == AMM:
                    d = base64.b64decode(acc["data"][0])
                    self.map[pool] = (_pk(d, 43), _pk(d, 75)) if len(d) >= 107 else ("", "")
                else:
                    self.map[pool] = ("", "")
                n += 1
        return n


class HeliusFeed:
    """Both programs' logs on their own connections; decoded transactions go to ``emit(events, source='helius')``, each
    once (a migration comes on both). ``wm``: the newest PumpSwap event stamp (epoch s) — the minutes' clock while the
    feed is up (``up``), ``grace_s`` back."""
    source = "helius"
    grace_s = 0.5                # its stamps are its blocks' seconds (+TS_OFFSET_S) in slot order: a minute is whole at the next second's first event

    def __init__(self, emit, rpc=None):
        self.emit, self.pools = emit, Pools()
        self._rpc = rpc
        self.delivered: dict[str, float] = {}                   # signature -> when it went out
        self.wm = 0.0; self.last_amm = 0.0; self.connected: dict[str, bool] = {AMM: False, PUMP: False}
        self.pending: Counter = Counter()                       # minute start -> transactions of it still on their way
        self.fetching: dict[str, dict] = {}                     # signature -> {slot, minute, first, n}: cut-off transactions to fetch
        self.waiting: list[dict] = []                           # transactions waiting for their pools' mints
        self.stats = Counter()

    @property
    def rpc(self):
        if self._rpc is None:
            from ..chain.rpc import HttpSolanaRpc
            self._rpc = HttpSolanaRpc()
        return self._rpc

    def up(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.connected.get(AMM, False) and now - self.last_amm < STALE_S

    # ---- one transaction ----
    def _deliver(self, sig: str, slot: int, evs: list, payer: str | None, minute: int | None) -> bool:
        """Build and emit, once per transaction; False (kept back) while a pool's mints are unknown."""
        if sig in self.delivered:
            return True
        out, unknown = build(sig, slot, evs, self.pools.map, payer)
        if unknown:
            return False
        now = time.time(); self.delivered[sig] = now
        if len(self.delivered) > 200_000:
            self.delivered = {s: t for s, t in self.delivered.items() if now - t < 600}
        if out:
            self.emit(out, "helius")
            for e in out:
                if e["pool"] == "pump-amm" and e["action"] in ("buy", "sell"):
                    self.wm = max(self.wm, e["timestamp"] / 1000.0)
        return True

    def on_logs(self, program: str, msg: dict) -> None:
        r = msg["params"]["result"]; v = r["value"]; slot = int(r["context"]["slot"])
        if program == AMM:
            self.last_amm = time.time()
        self.stats["msgs"] += 1
        if v.get("err") is not None:
            return
        logs = v.get("logs") or []; sig = v["signature"]
        evs = log_events(logs)
        if any("Log truncated" in line for line in logs):         # its events are fetched whole: the logs lost some
            if sig not in self.fetching:
                m = minute_of(evs)
                self.fetching[sig] = {"slot": slot, "minute": m, "first": time.time(), "n": 0}
                if m is not None:
                    self.pending[m] += 1
                self.stats["truncated"] += 1
            return
        if not evs:
            return
        m = minute_of(evs)
        if not self._deliver(sig, slot, evs, None, m):
            self.waiting.append({"sig": sig, "slot": slot, "evs": evs, "payer": None, "minute": m})
            if m is not None:
                self.pending[m] += 1

    # ---- the workers ----
    async def _fetch_loop(self, stop) -> None:
        while not (stop is not None and stop.is_set()):
            await asyncio.sleep(BACKFILL_RETRY_S)
            await self._fetch_round()

    async def _fetch_round(self) -> None:
        """Cut-off transactions, fetched whole in batches once confirmed; given up ``BACKFILL_GIVE_UP_S`` after first seen
        (a processed transaction whose fork was dropped never confirms)."""
        todo = list(self.fetching.items())[:BACKFILL_BATCH]
        if not todo:
            return
        calls = [("getTransaction", [s, {"encoding": "json", "maxSupportedTransactionVersion": 1, "commitment": "confirmed"}]) for s, _ in todo]
        try:
            res = await asyncio.to_thread(self.rpc.batch, calls)
        except Exception as e:
            self.stats["fetch_errors"] += 1
            log.warning("helius backfill batch failed: %s", type(e).__name__)
            res = [None] * len(todo)
        now = time.time()
        for (sig, f), tx in zip(todo, res):
            if tx:
                self.fetching.pop(sig, None)
                evs = cpi_events(tx); payer = tx["transaction"]["message"]["accountKeys"][0]
                m = f["minute"] if f["minute"] is not None else minute_of(evs)
                if f["minute"] is not None:
                    self.pending[f["minute"]] -= 1
                if evs and not self._deliver(sig, int(tx.get("slot") or f["slot"]), evs, payer, m):
                    self.waiting.append({"sig": sig, "slot": int(tx.get("slot") or f["slot"]), "evs": evs, "payer": payer, "minute": m})
                    if m is not None:
                        self.pending[m] += 1
                self.stats["backfilled"] += 1
            elif now - f["first"] > BACKFILL_GIVE_UP_S:
                self.fetching.pop(sig, None)
                if f["minute"] is not None:
                    self.pending[f["minute"]] -= 1
                self.stats["backfill_lost"] += 1
        self._tidy()

    async def _resolve_loop(self, stop) -> None:
        while not (stop is not None and stop.is_set()):
            await asyncio.sleep(0.05)
            await self._resolve_round()

    async def _resolve_round(self) -> None:
        """Transactions waiting for their pools' mints: the unknown pools read in one batch, then built again."""
        if not self.waiting:
            return
        need = sorted({t["pool"] for w in self.waiting for n, p in w["evs"] if n in ("BuyEvent", "SellEvent")
                       for t in (amm_trade(n, p),) if t["pool"] not in self.pools.map})
        if need:
            try:
                self.stats["pools_read"] += await asyncio.to_thread(self.pools.resolve, self.rpc, need)
            except Exception as e:
                self.stats["resolve_errors"] += 1
                log.warning("helius pool read failed: %s", type(e).__name__)
                return
        still = []
        for w in self.waiting:
            if self._deliver(w["sig"], w["slot"], w["evs"], w["payer"], w["minute"]):
                if w["minute"] is not None:
                    self.pending[w["minute"]] -= 1
            else:
                still.append(w)
        self.waiting = still
        self._tidy()

    def _tidy(self) -> None:
        for m in [m for m, n in self.pending.items() if n <= 0]:
            del self.pending[m]

    async def _connection(self, program: str, stop) -> None:
        ctx = ssl.create_default_context(cafile=certifi.where()); backoff = 0.5
        while not (stop is not None and stop.is_set()):
            try:
                async with websockets.connect(config.helius_wss_url(), ssl=ctx, max_size=16_000_000, ping_interval=20, ping_timeout=20) as ws:
                    await ws.send(json.dumps({"jsonrpc": "2.0", "id": 1, "method": "logsSubscribe",
                                              "params": [{"mentions": [program]}, {"commitment": config.HELIUS_FEED_COMMITMENT}]}))
                    log.info("helius feed: subscribed to %s logs", program[:4]); backoff = 0.5
                    while not (stop is not None and stop.is_set()):
                        raw = await asyncio.wait_for(ws.recv(), timeout=30)
                        self.stats["bytes"] += len(raw)
                        m = orjson.loads(raw)
                        if m.get("method") == "logsNotification":
                            self.connected[program] = True
                            try:
                                self.on_logs(program, m)
                            except Exception:
                                self.stats["decode_errors"] += 1
                                log.exception("helius feed: a %s notification failed to decode", program[:4])
                        elif "error" in m:
                            raise RuntimeError(f"logsSubscribe refused: {m['error']}")
            except Exception as e:
                self.stats["reconnects"] += 1
                log.warning("helius feed (%s): %s; reconnecting in %.1fs", program[:4], str(e)[:200], backoff)
            finally:
                self.connected[program] = False
            if stop is not None and stop.is_set():
                break
            await asyncio.sleep(backoff); backoff = min(10.0, backoff * 2)

    async def run(self, stop) -> None:
        n = await asyncio.to_thread(self.pools.seed)
        log.info("helius feed: %d pools known from the stream's minutes", n)
        await asyncio.gather(self._connection(AMM, stop), self._connection(PUMP, stop), self._fetch_loop(stop), self._resolve_loop(stop))

    def status(self) -> dict:
        return {**{k: int(v) for k, v in self.stats.items()}, "up": self.up(), "connected": {k[:4]: v for k, v in self.connected.items()},
                "fetching": len(self.fetching), "waiting": len(self.waiting), "pools": len(self.pools.map), "watermark": self.wm,
                "lag_s": round(time.time() - self.wm, 2) if self.wm else None}
