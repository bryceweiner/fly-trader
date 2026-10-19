#!/usr/bin/env python3
"""fly-trader's stream relay: PumpAPI's firehose, held near its source, forwarded to fly-trader over one compressed,
resumable connection (setup: stream_relay/README.md).

From Asia, PumpAPI's stream (Frankfurt) falls behind and reconnects about once a minute, losing ~40 % of the legs
(2026-10-19): the long hop cannot carry the whole firehose. This relay holds the firehose where it is fast and sends
only what fly-trader's aggregator (fly_trader/ingest/pumpstream.py) reads:

- the events: PumpSwap buys and sells of pump.fun coins quoted in SOL, bonding-curve buys and sells, creates,
  migrations and pool creations (``keep``) -- everything else (transfers, other venues, other quotes) is dropped;
- the fields: only those the aggregator reads (``FIELDS``, ``LEG_FIELDS``; tests/test_stream_relay.py checks them
  against pumpstream.py), unchanged.

Each event gets a sequence number and stays in a memory ring (``RELAY_BUFFER`` events): a client that reconnects
resumes where it left off, losing nothing within the ring (else it is told the gap). Batches go out every
``RELAY_BATCH_S`` and a heartbeat every second, each with the relay's watermark (its newest PumpSwap trade time up to
that batch) and whether PumpAPI is connected. Clients connect over TLS (``RELAY_TLS_CERT``/``RELAY_TLS_KEY``) with
``Authorization: Bearer <RELAY_TOKEN>``; ``GET /health`` answers without it. Needs websockets>=14 (orjson optional).
"""
from __future__ import annotations

import asyncio
import hmac
import json
import logging
import os
import ssl
import time
from http import HTTPStatus

import websockets

try:
    import orjson
    loads, dumps = orjson.loads, orjson.dumps
except ImportError:                                       # slower, same output
    loads = json.loads

    def dumps(o) -> bytes:
        return json.dumps(o, separators=(",", ":"), ensure_ascii=False).encode()

VERSION = 1
UPSTREAM = os.environ.get("RELAY_UPSTREAM", "wss://stream.pumpapi.io")
HOST = os.environ.get("RELAY_HOST", "0.0.0.0")
PORT = int(os.environ.get("RELAY_PORT", "8443"))
TOKEN = os.environ.get("RELAY_TOKEN", "")
TLS_CERT, TLS_KEY = os.environ.get("RELAY_TLS_CERT"), os.environ.get("RELAY_TLS_KEY")
ALLOW_PLAIN = os.environ.get("RELAY_ALLOW_PLAIN") == "1"   # no TLS: tests on loopback only
BUFFER = int(os.environ.get("RELAY_BUFFER", "300000"))
BATCH_S = float(os.environ.get("RELAY_BATCH_S", "0.05"))
HEARTBEAT_S, MAX_BATCH, STALE_S = 1.0, 5000, 10.0
WSOL = "So11111111111111111111111111111111111111112"
FIELDS = frozenset({"action", "pool", "mint", "timestamp", "signature", "block", "txSigner", "poolId", "quoteMint", "price", "quoteInPool",
                    "quoteAmount", "tokenAmount", "poolFeeRate", "poolCreatedBy", "breakdown", "supply", "initialBuy", "mayhemMode", "name",
                    "symbol", "uri"})
LEG_FIELDS = frozenset({"action", "trader", "quoteAmount", "tokenAmount"})
log = logging.getLogger("stream_relay")


def keep(e: dict) -> bool:
    """The events the aggregator uses (pumpstream.Aggregator.ingest): PumpSwap trades of pump.fun coins in SOL,
    pool creations and migrations; bonding-curve trades and creates."""
    a, pool = e.get("action"), e.get("pool")
    if pool == "pump-amm":
        if a in ("buy", "sell"):
            return e.get("quoteMint") == WSOL and str(e.get("mint") or "").endswith("pump")
        return a in ("migrate", "createPool")
    return pool == "pump" and a in ("buy", "sell", "create")


def trim(e: dict) -> dict:
    out = {k: v for k, v in e.items() if k in FIELDS}
    if isinstance(out.get("breakdown"), list):
        out["breakdown"] = [{k: v for k, v in b.items() if k in LEG_FIELDS} if isinstance(b, dict) else b for b in out["breakdown"]]
    return out


class Relay:
    """The ring of forwarded events (seq -> serialized event, and the watermark as of it) and PumpAPI's state."""

    def __init__(self, size: int = BUFFER):
        self.size = size; self.ring: list = [None] * size; self.ring_wm: list = [0.0] * size
        self.next_seq = 1; self.wm = 0.0; self.connected = False; self.last_upstream = 0.0
        self.stats = {"upstream_events": 0, "forwarded": 0, "reconnects": 0, "clients": 0, "bytes_sent": 0, "started": time.time()}

    def oldest(self) -> int:
        return max(1, self.next_seq - self.size)

    def up(self) -> bool:
        return self.connected and time.time() - self.last_upstream < STALE_S

    def add(self, e: dict) -> None:
        """An upstream event: forwarded when the aggregator uses it."""
        self.stats["upstream_events"] += 1; self.last_upstream = time.time()
        if not isinstance(e, dict) or not keep(e):
            return
        if e.get("pool") == "pump-amm" and e.get("action") in ("buy", "sell") and isinstance(e.get("timestamp"), (int, float)):
            self.wm = max(self.wm, e["timestamp"] / 1000.0)
        slot = self.next_seq % self.size
        self.ring[slot] = dumps(trim(e)); self.ring_wm[slot] = self.wm
        self.next_seq += 1; self.stats["forwarded"] += 1

    def batch(self, start: int) -> tuple[bytes, int]:
        """(message, next start) of the events from ``start`` (at most MAX_BATCH), or a heartbeat when there are none.
        ``wm`` is the watermark as of the batch's last event: a client never flushes a minute ahead of what it holds."""
        upto = min(self.next_seq, start + MAX_BATCH)
        wm = self.ring_wm[(upto - 1) % self.size] if upto > start else (self.ring_wm[(start - 1) % self.size] if start > 1 else 0.0)
        head = b'{"from":%d,"wm":%s,"up":%s,"t":%s,"events":[' % (start, repr(float(wm)).encode(), b"true" if self.up() else b"false", repr(time.time()).encode())
        return head + b",".join(self.ring[s % self.size] for s in range(start, upto)) + b"]}", upto

    async def upstream(self, stop: asyncio.Event) -> None:
        backoff = 1.0
        while not stop.is_set():
            try:
                async with websockets.connect(UPSTREAM, ssl=ssl.create_default_context() if UPSTREAM.startswith("wss") else None, max_size=8_000_000,
                                              ping_interval=20, ping_timeout=20) as ws:
                    log.info("connected to %s", UPSTREAM); self.connected = True; backoff = 1.0
                    async for msg in ws:
                        try:
                            self.add(loads(msg))
                        except ValueError:
                            continue
                log.warning("upstream closed; reconnecting in %.0fs", backoff)
            except Exception as e:
                log.warning("upstream: %s; reconnecting in %.0fs", str(e)[:200], backoff)
            finally:
                self.connected = False
            if not stop.is_set():
                self.stats["reconnects"] += 1
                await asyncio.sleep(backoff); backoff = min(30.0, backoff * 2)

    async def serve_client(self, ws) -> None:
        """One client: its first message names where to resume ({"resume": seq or null}); then batches from there."""
        self.stats["clients"] += 1
        peer = ws.remote_address[0] if ws.remote_address else "?"
        try:
            first = loads(await asyncio.wait_for(ws.recv(), timeout=10))
            want = first.get("resume") if isinstance(first, dict) else None
            nxt = self.next_seq if want is None else int(want)
            if nxt < self.oldest():
                await ws.send(dumps({"gap": [nxt, self.oldest()]})); nxt = self.oldest()
            nxt = min(nxt, self.next_seq)
            await ws.send(dumps({"hello": {"version": VERSION, "next": nxt, "oldest": self.oldest()}}))
            log.info("client %s from seq %d", peer, nxt)
            last_sent = 0.0
            while True:
                if nxt < self.oldest():                   # fell behind the ring
                    await ws.send(dumps({"gap": [nxt, self.oldest()]})); nxt = self.oldest()
                if self.next_seq > nxt or time.time() - last_sent >= HEARTBEAT_S:
                    msg, nxt = self.batch(nxt)
                    await ws.send(msg); last_sent = time.time(); self.stats["bytes_sent"] += len(msg)
                await asyncio.sleep(BATCH_S)
        except websockets.ConnectionClosed:
            pass
        finally:
            self.stats["clients"] -= 1
            log.info("client %s gone", peer)

    def health(self) -> dict:
        return {"ok": self.up(), "upstream_connected": self.connected, "watermark_lag_s": round(time.time() - self.wm, 1) if self.wm else None,
                "next_seq": self.next_seq, "oldest": self.oldest(), **{k: (round(v) if k == "started" else v) for k, v in self.stats.items()}}


def authorized(header: str | None) -> bool:
    return bool(TOKEN) and header is not None and hmac.compare_digest(header.encode(), f"Bearer {TOKEN}".encode())


async def main(stop: asyncio.Event | None = None, relay: Relay | None = None, port: int | None = None, upstream: bool = True) -> None:
    relay = relay or Relay(); stop = stop or asyncio.Event()
    if not TOKEN:
        raise SystemExit("RELAY_TOKEN is not set")
    if not (TLS_CERT and TLS_KEY) and not ALLOW_PLAIN:
        raise SystemExit("RELAY_TLS_CERT and RELAY_TLS_KEY are required (the token travels in the clear without TLS)")
    tls = None
    if TLS_CERT and TLS_KEY:
        tls = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); tls.load_cert_chain(TLS_CERT, TLS_KEY)

    def process_request(connection, request):
        if request.path == "/health":
            return connection.respond(HTTPStatus.OK, json.dumps(relay.health()) + "\n")
        if not authorized(request.headers.get("Authorization")):
            return connection.respond(HTTPStatus.UNAUTHORIZED, "unauthorized\n")
        return None

    async with websockets.serve(relay.serve_client, HOST, port or PORT, ssl=tls, process_request=process_request, compression="deflate",
                                max_size=1_000_000, ping_interval=20, ping_timeout=20):
        log.info("serving on %s:%d (%s)", HOST, port or PORT, "TLS" if tls else "PLAIN")
        jobs = [asyncio.create_task(relay.upstream(stop))] if upstream else []
        try:
            while not stop.is_set():
                try:
                    await asyncio.wait_for(stop.wait(), timeout=60)
                except asyncio.TimeoutError:
                    log.info("%s", relay.health())
        finally:
            stop.set()
            for j in jobs:
                j.cancel()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    asyncio.run(main())
