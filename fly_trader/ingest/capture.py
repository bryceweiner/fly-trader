"""Capture worker: the single chain ingest (plan, Architecture and Algorithms §11 "Capture").

One Helius ``transactionSubscribe`` websocket over the active ``watch_pools``; every notification is
decoded from the pool vaults' pre/post token balances (``swap_decoder``), written to Parquet
(``data/capture/swaps`` and ``data/capture/lp`` for side 0) and inserted into ``swap_tape`` in
``TAPE_BATCH_MS`` batches. Pools are reloaded every ``POOL_RELOAD_S``; on a change a NEW subscription
is sent first, its id awaited, then the old one is unsubscribed (no gap; the overlap is deduplicated
by signature). Vaults of pools without learned vaults are learned from their first notification and
persisted to ``watch_pools``. Reconnects use exponential backoff (1 s → 60 s). ``capture_status`` and
``data/capture/_status.json`` are refreshed every 30 s; start/stop/reconnect/errors go to ``events``.
SIGINT/SIGTERM flush everything and exit 0. The API key is never logged (websocket errors are passed
through the logging scrubber, and the URL itself is never formatted into a message).

Patterns lifted from VOC capture_meteora_live.py (ws loop, backoff, _status.json, signal flush).
"""
from __future__ import annotations

import asyncio
import json
import logging
import signal
import ssl
import time
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import psycopg
import pyarrow as pa
import websockets
from psycopg.types.json import Jsonb

from .. import config, logging_setup
from ..db import schema
from ..db.connection import connect
from . import swap_decoder, tape
from .parquet_writer import Writer, write_status
from .swap_decoder import PoolVaults

log = logging.getLogger("fly_trader.capture")

# ---- local constants (not in config.py; listed in the delivery report) ----
STATUS_EVERY_S = 30.0          # capture_status upsert + _status.json cadence
PING_EVERY_S = 30.0            # websocket ping frames (Helius drops idle sockets after 10 min)
TAPE_BATCH_ROWS = 500          # insert early when this many rows are pending
BACKOFF_MIN_S = 1.0
BACKOFF_MAX_S = 60.0
ACK_TIMEOUT_S = 30.0           # subscription ack wait before the socket is considered dead
DEDUPE_SIGS = 20000            # recent signatures remembered across the subscription overlap
PENDING_TAPE_MAX = 200_000     # rows kept in memory while the database is unreachable
ERROR_LOG_EVERY_S = 10.0       # rate limit for repeated decode/db error logs

PARQUET_SCHEMA = pa.schema([
    ("ts", pa.timestamp("us", tz="UTC")), ("slot", pa.int64()), ("sig", pa.string()), ("tx_index", pa.int32()),
    ("pool", pa.string()), ("mint", pa.string()), ("quote_mint", pa.string()), ("side", pa.int8()),
    ("amount_base", pa.uint64()), ("amount_quote", pa.int64()), ("price_sol", pa.float64()),
    ("price_quote", pa.float64()), ("signer", pa.string()), ("res_base", pa.uint64()), ("res_quote", pa.int64()),
    ("program_label", pa.string()),
])


def _ssl_context() -> ssl.SSLContext:
    """The python.org framework build ships no CA bundle; use certifi's like VOC does."""
    try:
        import certifi
        return ssl.create_default_context(cafile=certifi.where())
    except Exception:  # pragma: no cover
        return ssl.create_default_context()


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Capture:
    def __init__(self):
        self.stop = asyncio.Event()
        self.pools_changed = asyncio.Event()
        self.db_lock = asyncio.Lock()
        self.conn: psycopg.Connection | None = None
        self.started_at = _utcnow()
        # watch list
        self.learned: dict[str, PoolVaults] = {}      # pool -> vaults (decodable)
        self.unlearned: dict[str, dict] = {}          # pool -> watch_pools row lacking vaults
        self.vault_index: dict[str, PoolVaults] = {}
        self.pool_addresses: list[str] = []
        self.subscribed_addresses: list[str] = []
        self.logged_empty = False
        # websocket
        self.ws = None
        self.sub_id: int | None = None
        self.active_sub_ids: set[int] = set()
        self.req_id = 0
        self.pending_acks: dict[int, asyncio.Future] = {}
        self.pending_sub_reqs: set[int] = set()
        self.ws_connected = False
        self.connected_at: float | None = None
        self.last_message_at: float | None = None
        # counters
        self.reconnects = 0
        self.decode_failures = 0
        self.db_failures = 0
        self.rows_total = 0
        self.swaps_total = 0
        self.lp_total = 0
        self.notifications = 0
        self.duplicates = 0
        self.last_swap_ts: datetime | None = None
        self.last_error: str | None = None
        self._last_error_log = 0.0
        self.recent_sigs: OrderedDict[str, None] = OrderedDict()
        # sinks
        self.pending_tape: list[tape.TapeRow] = []
        self.writer_swaps = Writer(config.CAPTURE_DIR, "swaps", config.CAPTURE_FLUSH_ROWS,
                                   config.CAPTURE_FLUSH_S, schema=PARQUET_SCHEMA)
        self.writer_lp = Writer(config.CAPTURE_DIR, "lp", config.CAPTURE_FLUSH_ROWS,
                                config.CAPTURE_FLUSH_S, schema=PARQUET_SCHEMA)

    # ---- database helpers (sync; always called through _db) ----
    def _connect(self) -> psycopg.Connection:
        if self.conn is None or self.conn.closed:
            self.conn = connect()
            schema.ensure_partitions(self.conn)
            self.conn.commit()
        return self.conn

    async def _db(self, fn, *args):
        """Run a sync DB function on a worker thread, serialized on one connection."""
        async with self.db_lock:
            return await asyncio.to_thread(self._db_call, fn, *args)

    def _db_call(self, fn, *args):
        conn = self._connect()
        try:
            return fn(conn, *args)
        except Exception:
            try:
                conn.rollback()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            self.conn = None
            raise

    def _event(self, conn, level: str, message: str, detail: dict | None = None) -> None:
        with conn.cursor() as cur:
            cur.execute("INSERT INTO events (level, source, message, detail) VALUES (%s, 'capture', %s, %s)",
                        (level, message, Jsonb(detail or {})))
        conn.commit()

    async def event(self, level: str, message: str, detail: dict | None = None) -> None:
        try:
            await self._db(self._event, level, message, detail)
        except Exception as e:  # the event log must never take the daemon down
            log.warning("event insert failed: %s", type(e).__name__)

    def _note_error(self, where: str, exc: BaseException) -> None:
        self.last_error = f"{where}: {type(exc).__name__}: {str(exc)[:200]}"
        now = time.time()
        if now - self._last_error_log >= ERROR_LOG_EVERY_S:
            self._last_error_log = now
            log.warning("%s", self.last_error)

    # ---- watch list ----
    @staticmethod
    def _load_pools(conn) -> list[dict]:
        with conn.cursor() as cur:
            cur.execute("SELECT pool, mint, quote_mint, program_label, program_id, base_vault, quote_vault, base_decimals, "
                        "quote_decimals FROM watch_pools WHERE active ORDER BY pool")
            rows = cur.fetchall()
        conn.commit()
        return rows

    def _apply_pools(self, rows: list[dict]) -> bool:
        learned: dict[str, PoolVaults] = {}
        unlearned: dict[str, dict] = {}
        for r in rows:
            if r["base_vault"] and r["quote_vault"] and r["base_decimals"] is not None and r["quote_decimals"] is not None:
                learned[r["pool"]] = PoolVaults(
                    pool=r["pool"], mint=r["mint"], quote_mint=r["quote_mint"] or "", base_vault=r["base_vault"],
                    quote_vault=r["quote_vault"], base_decimals=int(r["base_decimals"]),
                    quote_decimals=int(r["quote_decimals"]), program_label=r["program_label"])
            else:
                unlearned[r["pool"]] = r
        addresses = sorted(learned) + sorted(unlearned)
        addresses = sorted(set(addresses))
        changed = addresses != self.pool_addresses
        self.learned, self.unlearned = learned, unlearned
        self.vault_index = swap_decoder.build_vault_index(learned)
        self.pool_addresses = addresses
        return changed

    async def pool_reloader(self) -> None:
        while not self.stop.is_set():
            try:
                rows = await self._db(self._load_pools)
                changed = self._apply_pools(rows)
                if not rows:
                    if not self.logged_empty:
                        log.info("watch_pools is empty; polling every %.0fs", config.POOL_RELOAD_S)
                        self.logged_empty = True
                else:
                    self.logged_empty = False
                if changed:
                    log.info("watch list changed: %d pools (%d with vaults, %d to learn)",
                             len(self.pool_addresses), len(self.learned), len(self.unlearned))
                    self.pools_changed.set()
            except Exception as e:
                self.db_failures += 1
                self._note_error("pool reload", e)
            await self._sleep(config.POOL_RELOAD_S)

    async def _sleep(self, seconds: float) -> None:
        try:
            await asyncio.wait_for(self.stop.wait(), timeout=seconds)
        except asyncio.TimeoutError:
            pass

    # ---- vault learning ----
    @staticmethod
    def _store_vaults(conn, pv: PoolVaults) -> str | None:
        """Persist learned vaults. The chain is authoritative for the quote mint: a differing
        watch_pools.quote_mint is overwritten and a non-SOL quote makes the pool untradable (v1 trades
        SOL-quoted pools only). Returns the previous quote_mint when it was changed."""
        with conn.cursor() as cur:
            cur.execute("SELECT quote_mint FROM watch_pools WHERE pool = %s", (pv.pool,))
            row = cur.fetchone()
            previous = (row["quote_mint"] if isinstance(row, dict) else row[0]) if row else None
            cur.execute(
                "UPDATE watch_pools SET base_vault = %s, quote_vault = %s, base_decimals = %s, quote_decimals = %s, "
                "quote_mint = %s, tradable = CASE WHEN %s = %s THEN tradable ELSE false END WHERE pool = %s",
                (pv.base_vault, pv.quote_vault, pv.base_decimals, pv.quote_decimals, pv.quote_mint,
                 pv.quote_mint, swap_decoder.WSOL_MINT, pv.pool))
        conn.commit()
        return previous if previous and previous != pv.quote_mint else None

    async def _maybe_learn(self, tx: dict, keys: set[str]) -> None:
        for pool in [p for p in self.unlearned if p in keys]:
            row = self.unlearned[pool]
            quotes = dict(swap_decoder.QUOTE_MINTS)
            if row.get("quote_mint") and row["quote_mint"] not in quotes:
                quotes[row["quote_mint"]] = int(row.get("quote_decimals") or 0)
            program_id = row.get("program_id") or swap_decoder.PROGRAM_IDS_BY_LABEL.get(row.get("program_label") or "")
            pv = swap_decoder.learn_vaults(tx, pool, quotes, mint=row["mint"], program_label=row.get("program_label"),
                                           program_id=program_id)
            if pv is None:
                continue
            self.unlearned.pop(pool, None)
            self.learned[pool] = pv
            self.vault_index = swap_decoder.build_vault_index(self.learned)
            log.info("learned vaults for %s (%s): base=%s quote=%s dec=%d/%d quote_mint=%s", pool,
                     pv.program_label, pv.base_vault, pv.quote_vault, pv.base_decimals, pv.quote_decimals,
                     pv.quote_mint)
            previous = None
            try:
                previous = await self._db(self._store_vaults, pv)
            except Exception as e:
                self.db_failures += 1
                self._note_error("store vaults", e)
            await self.event("info", "vaults learned", {"pool": pool, "base_vault": pv.base_vault,
                                                        "quote_vault": pv.quote_vault, "quote_mint": pv.quote_mint,
                                                        "base_decimals": pv.base_decimals,
                                                        "quote_decimals": pv.quote_decimals})
            if previous:
                log.warning("pool %s quote mint is %s on chain, watch_pools said %s; corrected%s", pool,
                            pv.quote_mint, previous,
                            "" if pv.quote_mint == swap_decoder.WSOL_MINT else " and marked untradable")
                await self.event("warning", "quote mint corrected", {"pool": pool, "was": previous,
                                                                     "now": pv.quote_mint})

    # ---- notification handling ----
    def _remember(self, sig: str) -> bool:
        """True if ``sig`` was already seen (subscription overlap or replay)."""
        if sig in self.recent_sigs:
            return True
        self.recent_sigs[sig] = None
        while len(self.recent_sigs) > DEDUPE_SIGS:
            self.recent_sigs.popitem(last=False)
        return False

    async def _on_notification(self, msg: dict) -> None:
        params = msg.get("params") or {}
        result = params.get("result") or {}
        sub = params.get("subscription")
        if self.active_sub_ids and sub not in self.active_sub_ids:
            return
        self.notifications += 1
        sig = result.get("signature") or ""
        if sig and self._remember(sig):
            self.duplicates += 1
            return
        now = _utcnow()
        try:
            txn, meta, _s, _slot, _i, _bt = swap_decoder.unwrap(result)
            if meta.get("err") is not None:
                return
            if self.unlearned:
                keys, _flags = swap_decoder.account_keys(txn, meta)
                await self._maybe_learn(result, set(keys))
            rows = swap_decoder.decode_transaction(result, self.learned, vault_index=self.vault_index, ts=now)
        except Exception as e:
            self.decode_failures += 1
            self._note_error("decode", e)
            return
        for row in rows:
            d = row.as_dict()
            if row.side == 0:
                self.writer_lp.add(d)
                self.lp_total += 1
            else:
                self.writer_swaps.add(d)
                self.swaps_total += 1
                self.last_swap_ts = row.ts
            self.pending_tape.append(row)
            self.rows_total += 1
        if len(self.pending_tape) > PENDING_TAPE_MAX:
            dropped = len(self.pending_tape) - PENDING_TAPE_MAX
            del self.pending_tape[:dropped]
            self._note_error("tape backlog", RuntimeError(f"dropped {dropped} oldest pending rows"))

    # ---- websocket ----
    def _subscribe_request(self, addresses: list[str]) -> tuple[int, str]:
        self.req_id += 1
        req = {"jsonrpc": "2.0", "id": self.req_id, "method": "transactionSubscribe", "params": [
            {"accountInclude": addresses, "failed": False, "vote": False},
            {"commitment": "confirmed", "encoding": "jsonParsed", "transactionDetails": "full",
             "showRewards": False, "maxSupportedTransactionVersion": 1},
        ]}
        return self.req_id, json.dumps(req)

    def _unsubscribe_request(self, sub_id: int) -> tuple[int, str]:
        self.req_id += 1
        return self.req_id, json.dumps({"jsonrpc": "2.0", "id": self.req_id, "method": "transactionUnsubscribe",
                                        "params": [sub_id]})

    async def _send_and_wait(self, ws, req_id: int, payload: str):
        fut = asyncio.get_running_loop().create_future()
        self.pending_acks[req_id] = fut
        try:
            await ws.send(payload)
            return await asyncio.wait_for(fut, timeout=ACK_TIMEOUT_S)
        finally:
            self.pending_acks.pop(req_id, None)

    def _handle_ack(self, msg: dict) -> None:
        req_id = msg.get("id")
        fut = self.pending_acks.get(req_id)
        if fut is None or fut.done():
            return
        if "error" in msg:
            fut.set_exception(RuntimeError(f"rpc error: {json.dumps(msg['error'])[:200]}"))
        else:
            result = msg.get("result")
            if req_id in self.pending_sub_reqs and isinstance(result, int):
                # accept the new subscription's notifications immediately, before _resubscribe resumes
                self.active_sub_ids.add(result)
            fut.set_result(result)

    async def _reader(self, ws) -> None:
        """Dispatch incoming frames to ack futures / notification handler until the socket closes."""
        async for raw in ws:
            self.last_message_at = time.time()
            try:
                msg = json.loads(raw)
            except ValueError:
                continue
            if "id" in msg and msg.get("method") is None:
                self._handle_ack(msg)
            elif msg.get("method") == "transactionNotification":
                await self._on_notification(msg)

    async def _subscribe(self, ws, addresses: list[str]) -> int:
        req_id, payload = self._subscribe_request(addresses)
        self.pending_sub_reqs.add(req_id)
        try:
            sub_id = await self._send_and_wait(ws, req_id, payload)
        finally:
            self.pending_sub_reqs.discard(req_id)
        if not isinstance(sub_id, int):
            raise RuntimeError(f"unexpected subscribe ack: {sub_id!r}")
        return sub_id

    async def _resubscribe(self, ws) -> None:
        """New subscription first, then unsubscribe the old id — the runner never sees a gap."""
        addresses = list(self.pool_addresses)
        if not addresses:
            return
        old = self.sub_id
        new = await self._subscribe(ws, addresses)
        self.active_sub_ids.add(new)
        self.sub_id = new
        self.subscribed_addresses = addresses
        log.info("subscribed %d pools (subscription %s, replacing %s)", len(addresses), new, old)
        if old is not None and old != new:
            try:
                req_id, payload = self._unsubscribe_request(old)
                await self._send_and_wait(ws, req_id, payload)
            except Exception as e:
                self._note_error("unsubscribe", e)
            finally:
                self.active_sub_ids.discard(old)

    async def _subscription_manager(self, ws) -> None:
        """Re-subscribes when the reloader flags a change, without interrupting the reader."""
        while not self.stop.is_set():
            await self.pools_changed.wait()
            self.pools_changed.clear()
            if self.stop.is_set():
                return
            if list(self.pool_addresses) == self.subscribed_addresses:
                continue
            if not self.pool_addresses:
                log.info("watch list empty; keeping the socket open with the previous subscription")
                continue
            await self._resubscribe(ws)

    async def ws_loop(self) -> None:
        backoff = BACKOFF_MIN_S
        ctx = _ssl_context()
        first = True
        while not self.stop.is_set():
            if not self.pool_addresses:
                await self._sleep(1.0)
                continue
            try:
                async with websockets.connect(config.helius_ws_url(), ssl=ctx, ping_interval=PING_EVERY_S,
                                              ping_timeout=PING_EVERY_S, max_size=None, max_queue=None,
                                              open_timeout=30) as ws:
                    self.ws = ws
                    self.ws_connected = True
                    self.connected_at = time.time()
                    self.sub_id = None
                    self.active_sub_ids = set()
                    self.subscribed_addresses = []
                    if not first:
                        self.reconnects += 1
                        await self.event("warning", "reconnected", {"reconnects": self.reconnects})
                    first = False
                    reader = asyncio.create_task(self._reader(ws))
                    try:
                        await self._resubscribe(ws)
                        backoff = BACKOFF_MIN_S
                        self.pools_changed.clear()
                        manager = asyncio.create_task(self._subscription_manager(ws))
                        stopper = asyncio.create_task(self.stop.wait())
                        done, _pending = await asyncio.wait({reader, manager, stopper},
                                                            return_when=asyncio.FIRST_COMPLETED)
                        for t in (manager, stopper):
                            t.cancel()
                        for t in done:
                            if t is reader or t is manager:
                                t.result()  # re-raise ConnectionClosed etc.
                    finally:
                        reader.cancel()
                        for fut in self.pending_acks.values():
                            if not fut.done():
                                fut.cancel()
                        self.ws_connected = False
                        self.ws = None
            except asyncio.CancelledError:
                raise
            except Exception as e:
                if self.stop.is_set():
                    break
                self.ws_connected = False
                self._note_error("websocket", e)
                log.warning("websocket down (%s) - reconnecting in %.0fs", type(e).__name__, backoff)
                await self.event("warning", "websocket disconnect", {"error": type(e).__name__,
                                                                     "detail": logging_setup.scrub(str(e))[:300],
                                                                     "backoff_s": backoff})
                await self._sleep(backoff)
                backoff = min(backoff * 2, BACKOFF_MAX_S)

    # ---- tape / status ----
    @staticmethod
    def _insert(conn, rows: list[tape.TapeRow]) -> int:
        return tape.insert_rows(conn, rows)

    async def flush_tape(self) -> int:
        if not self.pending_tape:
            return 0
        batch, self.pending_tape = self.pending_tape, []
        try:
            n = await asyncio.shield(self._db(self._insert, batch))     # a cancel at shutdown must not lose the batch in flight
        except asyncio.CancelledError:
            self.pending_tape = batch + self.pending_tape
            raise
        except Exception as e:
            self.db_failures += 1
            self._note_error("tape insert", e)
            self.pending_tape = batch + self.pending_tape
            return 0
        return n

    async def tape_flusher(self) -> None:
        interval = max(config.TAPE_BATCH_MS, 10) / 1000.0
        while not self.stop.is_set():
            deadline = time.time() + interval
            while time.time() < deadline and len(self.pending_tape) < TAPE_BATCH_ROWS and not self.stop.is_set():
                await asyncio.sleep(min(0.02, max(0.0, deadline - time.time())))
            await self.flush_tape()
            for w in (self.writer_swaps, self.writer_lp):
                try:
                    w.maybe_flush()
                except Exception as e:
                    self._note_error("parquet flush", e)

    def status_payload(self) -> dict:
        now = time.time()
        return {
            "pid": __import__("os").getpid(),
            "started_at": self.started_at.isoformat(),
            "uptime_s": round(now - self.started_at.timestamp(), 1),
            "ws_connected": self.ws_connected,
            "subscription_id": self.sub_id,
            "pools_subscribed": len(self.subscribed_addresses),
            "pools_active": len(self.pool_addresses),
            "pools_with_vaults": len(self.learned),
            "pools_to_learn": len(self.unlearned),
            "notifications": self.notifications,
            "duplicates": self.duplicates,
            "rows_total": self.rows_total,
            "swaps_total": self.swaps_total,
            "lp_total": self.lp_total,
            "pending_tape": len(self.pending_tape),
            "last_swap_ts": self.last_swap_ts.isoformat() if self.last_swap_ts else None,
            "last_message_age_s": round(now - self.last_message_at, 1) if self.last_message_at else None,
            "reconnects": self.reconnects,
            "decode_failures": self.decode_failures,
            "db_failures": self.db_failures,
            "last_error": self.last_error,
        }

    def _upsert_status(self, conn) -> None:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO capture_status (singleton, updated_at, last_swap_ts, rows_total, pools_subscribed, "
                "subscription_id, reconnects, decode_failures, last_error) VALUES (true, now(), %s, %s, %s, %s, %s, %s, %s) "
                "ON CONFLICT (singleton) DO UPDATE SET updated_at = now(), last_swap_ts = EXCLUDED.last_swap_ts, "
                "rows_total = EXCLUDED.rows_total, pools_subscribed = EXCLUDED.pools_subscribed, "
                "subscription_id = EXCLUDED.subscription_id, reconnects = EXCLUDED.reconnects, "
                "decode_failures = EXCLUDED.decode_failures, last_error = EXCLUDED.last_error",
                (self.last_swap_ts, self.rows_total, len(self.subscribed_addresses), self.sub_id, self.reconnects,
                 self.decode_failures, self.last_error))
        conn.commit()

    async def write_status(self) -> None:
        payload = self.status_payload()
        try:
            write_status(config.CAPTURE_DIR, "capture", payload)
        except Exception as e:
            self._note_error("status file", e)
        try:
            await self._db(self._upsert_status)
        except Exception as e:
            self.db_failures += 1
            self._note_error("status upsert", e)

    async def status_loop(self) -> None:
        while not self.stop.is_set():
            await self._sleep(STATUS_EVERY_S)
            await self.write_status()
            log.info("status: rows=%d swaps=%d lp=%d pools=%d/%d (learn %d) notif=%d dup=%d reconnects=%d "
                     "decode_fail=%d pending=%d", self.rows_total, self.swaps_total, self.lp_total,
                     len(self.subscribed_addresses), len(self.pool_addresses), len(self.unlearned),
                     self.notifications, self.duplicates, self.reconnects, self.decode_failures,
                     len(self.pending_tape))

    # ---- lifecycle ----
    async def _watch_stop_event(self, ev) -> None:
        while not ev.is_set():
            await asyncio.sleep(0.5)
        self.stop.set()

    async def run(self, stop_event=None) -> int:
        loop = asyncio.get_running_loop()
        import threading as _th
        if stop_event is not None:
            asyncio.create_task(self._watch_stop_event(stop_event), name="stop-watch")
        elif _th.current_thread() is _th.main_thread():
            for s in (signal.SIGINT, signal.SIGTERM):
                try:
                    loop.add_signal_handler(s, self.stop.set)
                except (NotImplementedError, RuntimeError):  # pragma: no cover
                    pass
        config.CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
        log.info("capture starting: parquet -> %s, tape batches every %d ms, pool reload every %.0fs",
                 config.CAPTURE_DIR, config.TAPE_BATCH_MS, config.POOL_RELOAD_S)
        await self.event("info", "capture start", {"pid": __import__("os").getpid(),
                                                   "capture_dir": str(config.CAPTURE_DIR)})
        tasks = [asyncio.create_task(self.pool_reloader(), name="pools"),
                 asyncio.create_task(self.ws_loop(), name="ws"),
                 asyncio.create_task(self.tape_flusher(), name="tape"),
                 asyncio.create_task(self.status_loop(), name="status")]
        await self.stop.wait()
        log.info("stop requested; flushing")
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.flush_tape()
        if self.pending_tape:
            await self.flush_tape()
        for w in (self.writer_swaps, self.writer_lp):
            try:
                w.close()
            except Exception as e:
                self._note_error("parquet close", e)
        await self.write_status()
        await self.event("info", "capture stop", {"rows_total": self.rows_total, "swaps": self.swaps_total,
                                                  "lp": self.lp_total, "reconnects": self.reconnects,
                                                  "decode_failures": self.decode_failures,
                                                  "pending_unwritten": len(self.pending_tape)})
        if self.conn is not None:
            try:
                self.conn.close()
            except Exception:
                pass
        log.info("capture stopped: rows=%d swaps=%d lp=%d reconnects=%d decode_failures=%d",
                 self.rows_total, self.swaps_total, self.lp_total, self.reconnects, self.decode_failures)
        return 0


def main(stop_event=None) -> int:
    logging_setup.setup("capture")
    if not config.HELIUS_API_KEY:
        log.error("HELIUS_API_KEY is not set")
        return 2
    cap = Capture()
    try:
        return asyncio.run(cap.run(stop_event))
    except KeyboardInterrupt:  # pragma: no cover
        return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
