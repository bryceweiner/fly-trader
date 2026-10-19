"""The stream relay (stream_relay/relay.py) and its client (ingest/relay_feed.py): the relay keeps every event and field
the aggregator uses, so the minutes built from its stream equal those built from PumpAPI's; a client that reconnects
resumes losing nothing within the relay's ring (else it is told the gap); only the token's holder is served; with
SOL_FEED=relay the relay is the minutes' clock and the direct stream's copies are dropped."""
import asyncio
import importlib.util
import re
import socket
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
import websockets

from fly_trader import config
from fly_trader.ingest import pumpstream, relay_feed

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("stream_relay_relay", ROOT / "stream_relay" / "relay.py")
relay = importlib.util.module_from_spec(spec); spec.loader.exec_module(relay)
WSOL = relay.WSOL
M = 1_800_000_000


def test_the_relay_keeps_every_field_the_aggregator_reads():
    src = (ROOT / "fly_trader" / "ingest" / "pumpstream.py").read_text()
    event = set(re.findall(r'\be\.get\("(\w+)"\)|\be\["(\w+)"\]', src)); event = {a or b for a, b in event}
    legs = {k for k in re.findall(r'\bb\.get\("(\w+)"\)', src)}
    assert event <= relay.FIELDS, event - relay.FIELDS
    assert legs <= relay.LEG_FIELDS, legs - relay.LEG_FIELDS


def _events():
    junk = {"tradersInvolved": ["x", "y"], "tokensInPool": 123.0, "extra": {"deep": [1, 2]}}
    leg = lambda a, t, q, tok: {"action": a, "trader": t, "quoteAmount": q, "tokenAmount": tok, "junk": 1}   # noqa: E731
    return [
        {"action": "create", "pool": "pump", "mint": "Cpump", "timestamp": (M + 1) * 1000, "txSigner": "dev", "quoteAmount": 1.5, "initialBuy": 1e6,
         "supply": 1e9, "mayhemMode": False, "name": "C", "symbol": "CC", "uri": "u", "signature": "s1", "block": 5, **junk},
        {"action": "buy", "pool": "pump", "mint": "Cpump", "timestamp": (M + 2) * 1000, "block": 6, "txSigner": "w1", "tokenAmount": 5e5,
         "signature": "s2", "breakdown": [leg("buy", "w1", 0.5, 5e5), leg("buy", "w2", 0.2, 2e5)], **junk},
        {"action": "buy", "pool": "pump-amm", "mint": "Apump", "quoteMint": WSOL, "poolId": "P1", "timestamp": (M + 3) * 1000, "price": 1e-6,
         "quoteInPool": 80.0, "txSigner": "t1", "poolFeeRate": 0.0125, "signature": "s3", "breakdown": [leg("buy", "t1", 2.0, 1e6)], **junk},
        {"action": "sell", "pool": "pump-amm", "mint": "Apump", "quoteMint": WSOL, "poolId": "P1", "timestamp": (M + 30) * 1000, "price": 0.9e-6,
         "quoteInPool": 79.0, "txSigner": "t2", "signature": "s4", "breakdown": [leg("sell", "t2", 1.0, 9e5), leg("buy", "t3", 0.5, 4e5)], **junk},
        {"action": "buy", "pool": "pump-amm", "mint": "Upump", "quoteMint": "USDC", "poolId": "P2", "timestamp": (M + 4) * 1000, "price": 1.0,
         "quoteInPool": 10.0, "signature": "s5", "breakdown": [leg("buy", "t9", 1.0, 1.0)]},
        {"action": "sell", "pool": "pump-amm", "mint": "Notpumpcoin", "quoteMint": WSOL, "poolId": "P3", "timestamp": (M + 5) * 1000, "price": 1.0,
         "quoteInPool": 10.0, "signature": "s6"},
        {"action": "createPool", "pool": "pump-amm", "poolId": "PX", "mint": "Xpump", "poolCreatedBy": "someWallet", "timestamp": (M + 6) * 1000, **junk},
        {"action": "buy", "pool": "pump-amm", "mint": "Xpump", "quoteMint": WSOL, "poolId": "PX", "timestamp": (M + 7) * 1000, "price": 1e-6,
         "quoteInPool": 50.0, "signature": "s7", "breakdown": [leg("buy", "t4", 1.0, 1e6)]},
        {"action": "migrate", "pool": "pump-amm", "mint": "Cpump", "timestamp": (M + 8) * 1000, "signature": "s8", "poolId": "PM", "txSigner": "mig",
         "quoteInPool": 67.4, "block": 9, **junk},
        {"action": "transfer", "pool": None, "mint": "Apump", "timestamp": (M + 9) * 1000, "signature": "s9"},
        {"action": "buy", "pool": "raydium-v4", "mint": "Apump", "quoteMint": WSOL, "timestamp": (M + 10) * 1000, "price": 1.0, "quoteInPool": 10.0},
    ]


def _aggregate(events, monkeypatch):
    agg = pumpstream.Aggregator(); grads = []
    monkeypatch.setattr(agg, "_graduation", lambda e: grads.append({k: e.get(k) for k in ("mint", "timestamp", "signature", "poolId", "txSigner", "quoteInPool", "block")}))
    monkeypatch.setattr(agg, "write_creates", lambda: 0)
    for e in events:
        agg.ingest(e)
    rows = {(m, mint): (r.open, r.high, r.low, r.close, r.buy, r.sell, r.nb, r.ns, sorted(r.traders), r.wallets, r.resq, r.pool_id, r.fee_rate)
            for m in agg.minutes for mint in agg.minutes[m] for r in (agg.row(m, mint),)}
    return rows, agg.curve, agg.creates, agg.blocked, [p[:3] for p in agg.pending_pools], grads


def test_the_minutes_from_the_relay_equal_the_minutes_from_pumpapi(monkeypatch):
    full = _events()
    relayed = [relay.trim(e) for e in full if relay.keep(e)]
    assert len(relayed) == 7 and all("junk" not in b for e in relayed for b in e.get("breakdown") or [])
    assert all(k in relay.FIELDS for e in relayed for k in e)
    a, b = _aggregate(full, monkeypatch), _aggregate(relayed, monkeypatch)
    assert a == b and a[0] and a[1] and a[3] == {"PX"} and len(a[5]) == 1      # trades, curve wallets, the custom pool blocked, one migration


def _ev(i, mint="Apump"):
    return {"action": "buy", "pool": "pump-amm", "mint": mint, "quoteMint": WSOL, "poolId": "P1", "timestamp": (M + i) * 1000, "price": 1e-6,
            "quoteInPool": 80.0, "signature": f"sig{i}", "breakdown": [{"action": "buy", "trader": f"t{i}", "quoteAmount": 0.1}]}


def test_a_batch_carries_the_watermark_as_of_its_own_last_event_and_an_old_resume_is_a_gap():
    r = relay.Relay(size=8); r.connected = True; r.last_upstream = time.time()
    for i in range(5):
        r.add(_ev(i))
    r.add({"action": "transfer", "pool": None})                                      # not forwarded
    relay.MAX_BATCH, old = 3, relay.MAX_BATCH
    try:
        import orjson
        msg, nxt = r.batch(1); m = orjson.loads(msg)
        assert (m["from"], len(m["events"]), nxt, m["wm"], m["up"]) == (1, 3, 4, M + 2, True)   # the watermark of event 3, not of the newest
        msg, nxt = r.batch(nxt); m = orjson.loads(msg)
        assert (m["from"], [e["signature"] for e in m["events"]], nxt, m["wm"]) == (4, ["sig3", "sig4"], 6, M + 4)
        msg, nxt = r.batch(6); m = orjson.loads(msg)
        assert m["events"] == [] and m["wm"] == M + 4 and nxt == 6                    # a heartbeat
    finally:
        relay.MAX_BATCH = old
    for i in range(5, 15):
        r.add(_ev(i))
    assert r.oldest() == 16 - 8 and r.next_seq == 16 and r.stats["forwarded"] == 15


def _free_port():
    s = socket.socket(); s.bind(("127.0.0.1", 0)); p = s.getsockname()[1]; s.close()
    return p


def test_the_client_resumes_where_it_left_off_and_only_the_tokens_holder_is_served(monkeypatch):
    port = _free_port()
    monkeypatch.setattr(relay, "TOKEN", "relay-test-token-123"); monkeypatch.setattr(relay, "ALLOW_PLAIN", True); monkeypatch.setattr(relay, "HOST", "127.0.0.1")
    monkeypatch.setattr(relay, "BATCH_S", 0.02)
    monkeypatch.setattr(config, "STREAM_RELAY_URL", f"ws://127.0.0.1:{port}/stream"); monkeypatch.setattr(config, "STREAM_RELAY_TOKEN", "relay-test-token-123")

    async def scenario():
        stop = asyncio.Event(); r = relay.Relay(size=1000); r.connected = True; r.last_upstream = time.time()
        server = asyncio.create_task(relay.main(stop=stop, relay=r, port=port, upstream=False))
        await asyncio.sleep(0.3)
        got = []; feed = relay_feed.RelayFeed(lambda evs, src: got.extend((src, e["signature"]) for e in evs))

        async def until(cond, timeout=5.0):
            t0 = time.time()
            while not cond() and time.time() - t0 < timeout:
                await asyncio.sleep(0.02)
            return cond()

        cstop = asyncio.Event(); client = asyncio.create_task(feed.run(cstop))
        assert await until(lambda: feed.connected and feed.next_seq is not None)
        for i in range(3):
            r.add(_ev(i))
        assert await until(lambda: len(got) == 3) and feed.up() and feed.wm == M + 2
        cstop.set(); await asyncio.wait_for(client, 5)                                # disconnected
        for i in range(3, 6):
            r.add(_ev(i))                                                              # sent while it was away
        cstop = asyncio.Event(); client = asyncio.create_task(feed.run(cstop))
        assert await until(lambda: len(got) == 6)
        assert got == [("relay", f"sig{i}") for i in range(6)] and not feed.stats["gaps"]   # resumed: nothing lost, nothing twice
        cstop.set(); await asyncio.wait_for(client, 5)
        with pytest.raises(websockets.InvalidStatus) as bad:
            async with websockets.connect(f"ws://127.0.0.1:{port}/stream", additional_headers={"Authorization": "Bearer wrong"}):
                pass
        health = await asyncio.to_thread(httpx.get, f"http://127.0.0.1:{port}/health", timeout=5)
        stop.set(); await asyncio.wait_for(server, 5)
        return bad.value.response.status_code, health.status_code, health.json()

    status, hcode, h = asyncio.run(scenario())
    assert status == 401 and hcode == 200 and h["forwarded"] == 6 and h["ok"] is True


def test_a_resume_beyond_the_ring_is_reported_as_a_gap():
    feed = relay_feed.RelayFeed(lambda evs, src: None)
    feed.on_message({"gap": [10, 25]}); feed.on_message({"hello": {"version": 1, "next": 25, "oldest": 25}})
    assert feed.stats["gaps"] == 1 and feed.stats["lost"] == 15 and feed.next_seq == 25


def test_with_the_relay_up_it_is_the_clock_and_the_direct_streams_copies_are_dropped():
    feed = SimpleNamespace(source="relay", grace_s=relay_feed.GRACE_S, up=lambda now=None: True, wm=M + 62.5, pending=Counter())
    agg = pumpstream.Aggregator(); agg.feed = feed; agg.last_event_wall = M + 63
    assert agg.flush_bound(M + 63) == M + 60                                         # PumpAPI's stamps: its 2 s grace
    feed.wm = M + 61.5
    assert agg.flush_bound(M + 63) == M
    e = _ev(1); agg.ingest_from(dict(e), "relay"); agg.ingest_from(dict(e), "pumpapi")
    nosig = {k: v for k, v in _ev(2).items() if k != "signature"}
    agg.ingest_from(dict(nosig), "pumpapi")                                           # unmatched: the backup's copy waits for the relay to be down
    assert agg.row(M, "Apump").nb == 1 and agg.stats["duplicates"] == 2
    feed.up = lambda now=None: False
    agg.ingest_from(dict(nosig), "pumpapi")
    assert agg.row(M, "Apump").nb == 2


def test_the_relay_takes_pumpapis_stream_and_forwards_only_what_is_kept(monkeypatch):
    """The relay's own side: it holds the upstream stream (a stand-in for PumpAPI here), keeps the events and fields the
    aggregator uses, and reconnects when the upstream drops."""
    import orjson
    up_port = _free_port(); sent = {"conns": 0}

    async def fake_pumpapi(ws):
        sent["conns"] += 1
        for e in _events():
            await ws.send(orjson.dumps(e).decode())
        if sent["conns"] == 1:
            await ws.close()                                                          # the upstream drops once
        else:
            await ws.wait_closed()

    async def scenario():
        monkeypatch.setattr(relay, "UPSTREAM", f"ws://127.0.0.1:{up_port}")
        r = relay.Relay(size=100); stop = asyncio.Event()
        async with websockets.serve(fake_pumpapi, "127.0.0.1", up_port):
            task = asyncio.create_task(r.upstream(stop))
            t0 = time.time()
            while r.stats["forwarded"] < 14 and time.time() - t0 < 10:
                await asyncio.sleep(0.05)
            stop.set(); task.cancel()
        return r

    r = asyncio.run(scenario())
    assert sent["conns"] == 2 and r.stats["reconnects"] == 1 and r.stats["upstream_events"] == 22 and r.stats["forwarded"] == 14
    kept = [orjson.loads(r.ring[s % r.size]) for s in range(1, r.next_seq)]
    assert {(e["pool"], e["action"]) for e in kept} == {("pump", "create"), ("pump", "buy"), ("pump-amm", "buy"), ("pump-amm", "sell"), ("pump-amm", "createPool"), ("pump-amm", "migrate")}
    assert all(set(e) <= relay.FIELDS for e in kept) and r.wm == M + 30
