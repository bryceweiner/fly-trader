"""The stream relay's client: PumpAPI's events, filtered and trimmed by stream_relay/relay.py on a server near PumpAPI,
over one compressed connection that resumes where it left off (nothing lost within the relay's ring).

With ``SOL_FEED=relay`` it is the minutes' clock (ingest/pumpstream.Aggregator.flush_bound: the relay's watermark as of
the events it has sent, ``grace_s`` back, as PumpAPI's own stream) and this machine's own PumpAPI connection is its
backup; each transaction is taken once (``Aggregator.ingest_from``). ``check`` connects without switching anything and
reports what arrives (``fly-trader relay-check``)."""
from __future__ import annotations

import asyncio
import json
import logging
import ssl
import time
from collections import Counter

import certifi
import orjson
import websockets

from .. import config

log = logging.getLogger(__name__)
GRACE_S = 2.0                # the events carry PumpAPI's stamps (its receive times): the grace of its own stream (pumpstream.FLUSH_GRACE_S)
STALE_S = 5.0                # the relay heartbeats every second: silence this long means it is down


class RelayFeed:
    source = "relay"
    grace_s = GRACE_S

    def __init__(self, emit):
        self.emit = emit
        self.next_seq: int | None = None                    # the next event wanted: a reconnect resumes there
        self.wm = 0.0; self.last_msg = 0.0; self.connected = False; self.upstream_up = False
        self.pending: Counter = Counter()                   # nothing arrives later than its batch: no minute waits on the relay
        self.stats = Counter()

    def up(self, now: float | None = None) -> bool:
        now = time.time() if now is None else now
        return self.connected and self.upstream_up and now - self.last_msg < STALE_S

    def on_message(self, m: dict) -> None:
        self.last_msg = time.time()
        if "gap" in m:                                      # the relay's ring no longer holds what this client missed
            a, b = m["gap"]; self.stats["gaps"] += 1; self.stats["lost"] += max(0, b - a); self.next_seq = b
            log.warning("stream relay: %d events lost while disconnected (beyond its buffer)", b - a)
            return
        if "hello" in m:
            self.next_seq = m["hello"]["next"]
            return
        evs = m.get("events") or []
        if self.next_seq is not None and m["from"] != self.next_seq:
            log.warning("stream relay: batch from %s, expected %s", m["from"], self.next_seq)
        if evs:
            self.emit(evs, self.source); self.stats["events"] += len(evs)
        self.next_seq = m["from"] + len(evs)
        self.wm = max(self.wm, float(m.get("wm") or 0.0)); self.upstream_up = bool(m.get("up"))

    async def run(self, stop) -> None:
        if not (config.STREAM_RELAY_URL and config.STREAM_RELAY_TOKEN):
            log.error("SOL_FEED=relay needs STREAM_RELAY_URL and STREAM_RELAY_TOKEN (stream_relay/README.md): PumpAPI's stream alone meanwhile")
            return
        tls = ssl.create_default_context(cafile=config.STREAM_RELAY_CA_FILE or certifi.where()) if config.STREAM_RELAY_URL.startswith("wss") else None
        backoff = 0.5
        while not (stop is not None and stop.is_set()):
            try:
                async with websockets.connect(config.STREAM_RELAY_URL, ssl=tls, additional_headers={"Authorization": f"Bearer {config.STREAM_RELAY_TOKEN}"},
                                              compression="deflate", max_size=64_000_000, ping_interval=10, ping_timeout=10, open_timeout=10) as ws:
                    await ws.send(json.dumps({"resume": self.next_seq}))
                    self.connected = True; backoff = 0.5
                    log.info("stream relay connected (resuming at %s)", self.next_seq)
                    while not (stop is not None and stop.is_set()):
                        self.on_message(orjson.loads(await asyncio.wait_for(ws.recv(), timeout=STALE_S * 2)))
            except Exception as e:
                self.stats["reconnects"] += 1
                log.warning("stream relay: %s; reconnecting in %.1fs", str(e)[:200], backoff)
            finally:
                self.connected = False
            if stop is not None and stop.is_set():
                break
            await asyncio.sleep(backoff); backoff = min(10.0, backoff * 2)

    def status(self) -> dict:
        return {**{k: int(v) for k, v in self.stats.items()}, "up": self.up(), "connected": self.connected, "upstream_up": self.upstream_up,
                "next_seq": self.next_seq, "watermark": self.wm, "lag_s": round(time.time() - self.wm, 2) if self.wm else None}


def check(seconds: float = 30.0) -> dict:
    """Connect to the relay for ``seconds`` without touching the live stream and report what arrived."""
    if not (config.STREAM_RELAY_URL and config.STREAM_RELAY_TOKEN):
        return {"error": "STREAM_RELAY_URL and STREAM_RELAY_TOKEN are not set in .env (stream_relay/README.md)"}
    kinds: Counter = Counter(); lags: list[float] = []
    feed = RelayFeed(lambda evs, src: kinds.update((e.get("pool"), e.get("action")) for e in evs))

    async def go():
        stop = asyncio.Event(); task = asyncio.create_task(feed.run(stop)); t0 = time.time()
        while time.time() - t0 < seconds:
            await asyncio.sleep(1.0)
            if feed.wm:
                lags.append(time.time() - feed.wm)
        stop.set()
        await asyncio.sleep(1.5); task.cancel()

    asyncio.run(go())
    lags.sort()
    out = {"seconds": seconds, "events_per_s": round(sum(kinds.values()) / seconds, 1), "by_kind": {f"{p}/{a}": n for (p, a), n in kinds.most_common()},
           "lag_s_p50": round(lags[len(lags) // 2], 2) if lags else None, "lag_s_max": round(lags[-1], 2) if lags else None, **feed.status()}
    return out
