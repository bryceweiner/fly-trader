"""HyperSync logs come back in eth_getLogs' shape, with block times, across paginated responses; blocks HyperSync has
not reached are read from the RPC."""
import httpx
import pytest

from fly_trader import config
from fly_trader.rh import hypersync as H


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr(config, "RH_HYPERSYNC_RPM", 1e6)


seen_paths = []


def _handler(pages):
    seen = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen_paths.append(request.url.path)
        if request.url.path == "/height":
            return httpx.Response(200, json={"height": 1000})
        import json
        body = json.loads(request.content); seen.append(body)
        lo, hi = body["from_block"], body["to_block"]
        mid = min(hi, lo + pages, 1001)                                       # the archive height is 1000
        logs = [{"block_number": b, "log_index": 0, "transaction_hash": f"0x{b:064x}", "block_hash": "0xbb", "address": "0xAA",
                 "data": "0x01", "topic0": "0xt0", "topic1": None} for b in range(lo, mid, 10)]
        blocks = [{"number": b, "timestamp": hex(1_790_000_000 + b)} for b in range(lo, mid, 10)]
        return httpx.Response(200, json={"data": [{"logs": logs, "blocks": blocks}], "next_block": max(mid, lo), "archive_height": 1000})
    return handle, seen


def test_paginated_logs_in_rpc_shape_with_times():
    handle, seen = _handler(pages=100)
    hs = H.HyperSync("tok", url="https://hs.test", client=httpx.Client(transport=httpx.MockTransport(handle)))
    logs = hs.get_logs(["0xAA"], 0, 349, [["0xT0"], None])
    assert len(seen) == 4 and seen[0]["logs"][0] == {"address": ["0xaa"], "topics": [["0xt0"], []]}
    assert [int(g["blockNumber"], 16) for g in logs] == list(range(0, 350, 10))
    g = logs[3]
    assert g["address"] == "0xaa" and g["topics"] == ["0xt0"] and g["logIndex"] == "0x0" and int(g["blockTimestamp"], 16) == 1_790_000_030


def test_blocks_past_the_archive_height_come_from_the_rpc(monkeypatch):
    handle, _ = _handler(pages=10_000)
    hs = H.HyperSync("tok", url="https://hs.test", client=httpx.Client(transport=httpx.MockTransport(handle)))
    rpc = H.HyperRpc(hs, url="https://rpc.test", chain_id=4663)
    asked = []
    monkeypatch.setattr(H.RhRpc, "get_logs_multi", lambda self, a, lo, hi, t=None: asked.append((lo, hi)) or [])
    logs = rpc.get_logs_multi(["0xaa"], 900, 1100)
    assert asked == [(1001, 1100)] and max(int(g["blockNumber"], 16) for g in logs) <= 1000
    assert not any(r == "/height" for r in seen_paths)                           # no /height call on the hot path
