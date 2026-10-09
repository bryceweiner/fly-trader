"""A live RH pass reads its logs once (rh/prefetch.py): eth_getLogs coverage and matching, the index step and the price
scan answered from memory (only the swappers' Transfers go out), HyperSync's multi-selection query, and the next pass
timed to the moment a minute can be written."""
import json
from pathlib import Path

import httpx
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.rh import hypersync as H, index, prefetch, prices, stream, v4

LOGS = json.loads((Path(__file__).parent / "fixtures" / "rh_logs.json").read_text())
ETH = "0x" + "00" * 20


@pytest.fixture(autouse=True)
def _no_pacing(monkeypatch):
    monkeypatch.setattr(config, "RH_HYPERSYNC_RPM", 1e6)


def _log(addr, topics, block, li=0, data="0x"):
    return {"address": addr, "topics": topics, "data": data, "blockNumber": hex(block), "logIndex": hex(li), "transactionHash": f"0x{block:064x}",
            "blockHash": "0xbb", "blockTimestamp": hex(1_790_000_000 + block)}


def test_coverage_follows_eth_getlogs_semantics():
    sels = [{"address": ["0xaa", "0xbb"], "topics": [["0xt1", "0xt2"]]}, {"address": ["0xcc"], "topics": [[], ["0xid1", "0xid2"]]}]
    assert prefetch.covered(["0xAA"], [["0xT1"]], sels)                      # a narrower request is covered
    assert prefetch.covered(["0xaa"], ["0xt2", "0xanything"], sels)          # a constraint the selection lacks is applied locally
    assert not prefetch.covered(["0xaa"], [["0xt3"]], sels)                  # a topic the selection did not fetch
    assert not prefetch.covered(["0xaa"], None, sels)                        # any topic0: more than was fetched
    assert not prefetch.covered(["0xaa", "0xdd"], [["0xt1"]], sels)          # an address outside the fetch
    assert not prefetch.covered(None, [["0xt1"]], sels)                      # any address: never from memory
    assert prefetch.covered(["0xcc"], [None, ["0xid2"]], sels) and not prefetch.covered(["0xcc"], [None, ["0xid3"]], sels)
    g = _log("0xAA", ["0xT1", "0xX"], 5)
    assert prefetch.matches(g, {"0xaa"}, prefetch._sets([["0xt1"], "0xx"])) and not prefetch.matches(g, {"0xaa"}, prefetch._sets([None, "0xy"]))


def test_hypersync_one_query_for_many_selections_each_log_once():
    bodies = []

    def handle(request):
        body = json.loads(request.content); bodies.append(body)
        lg = {"block_number": 10, "log_index": 1, "transaction_hash": "0x1", "block_hash": "0xb", "address": "0xAA", "data": "0x", "topic0": "0xt0"}
        return httpx.Response(200, json={"data": [{"logs": [lg, dict(lg)], "blocks": [{"number": b, "timestamp": hex(1000 + b)} for b in range(10, 21)]}],
                                         "next_block": 21, "archive_height": 20})
    hs = H.HyperSync("tok", url="https://hs.test", client=httpx.Client(transport=httpx.MockTransport(handle)))
    logs, through, ts = hs.get_logs_sel([H.selection(["0xAA"], [["0xT0"]]), H.selection(["0xaa"], [None, "0xID"])], 10, 30, upto=True, tries=1, all_blocks=True)
    assert len(bodies) == 1 and len(bodies[0]["logs"]) == 2 and bodies[0]["include_all_blocks"] is True
    assert bodies[0]["logs"][1] == {"address": ["0xaa"], "topics": [[], ["0xid"]]}
    assert len(logs) == 1 and through == 20 and ts[20] == 1020                 # the duplicate once; covered to the archive height


class FakeRpc:
    def __init__(self, hs=None):
        self.hs = hs; self.calls = []; self.url = "https://rpc.test"

    def get_logs_multi(self, addresses, lo, hi, topics=None):
        self.calls.append((tuple(addresses or ()), lo, hi, topics)); return []

    def block(self, tag, full=False):
        self.calls.append(("block", tag)); return {"number": hex(tag), "timestamp": hex(1)}


def test_a_live_index_step_reads_only_the_transfers_beyond_memory(monkeypatch):
    sw = dict(LOGS["v4_Swap"]); d = v4.decode(sw); b = d["block"]
    sw["blockTimestamp"] = hex(1_790_000_000)
    meme = "0x" + "a7" * 20
    with transaction() as conn:
        for t in ("rh_swaps", "rh_liquidity", "rh_pools"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals) VALUES (%s, 'ETH', 'eth', 18) ON CONFLICT (asset) DO NOTHING", (ETH,))
        conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, quote_asset, is_pons) VALUES (%s, %s, %s, %s, false, %s, true)",
                     (d["pool_id"], meme, ETH, meme, ETH))
    rpc = FakeRpc(hs=object())
    sels = [H.selection([config.PONS_FACTORY, config.PONS_ROUTER, config.PONS_HOOK], index.PONS_TOPICS),
            H.selection([config.V4_POOL_MANAGER], [index.V4_TOPICS0]), H.selection([config.PONS_HOOK], [[index.FEE_TOPIC]])]
    pre = prefetch.Prefetch(rpc, b - 5, b + 5, [sw], {b + 5: 1_790_000_009}, sels, "hypersync")
    try:
        with transaction() as conn:
            n = index.step(conn, pre, b - 5, b + 5)
            row = conn.execute("SELECT pool_id, side FROM rh_swaps WHERE block = %s", (b,)).fetchone()
        assert n["swaps"] == 1 and row["pool_id"] == d["pool_id"]
        assert pre.served == 3 and pre.passed == 1                                # Pons, PoolManager and fee reads from memory
        assert rpc.calls == [((meme,), b, b, [index.TRANSFER])]                   # only the swappers' Transfers went out
        assert pre.block(b + 5)["timestamp"] == hex(1_790_000_009) and len(rpc.calls) == 1    # the last block's time from the fetch
    finally:
        with transaction() as conn:
            for t in ("rh_swaps", "rh_pools"):
                conn.execute(f"DELETE FROM {t}")


def test_the_price_scan_reads_from_memory(monkeypatch):
    plain = {"pool": "0x" + "b1" * 20, "emitter": "0x" + "b1" * 20, "token0": ETH, "token1": "0x" + "c1" * 20, "dec0": 18, "dec1": 18, "token_in": ETH}
    pid = "0x" + "d1" * 32
    mgr = {"pool": pid, "emitter": "0x" + "e1" * 20, "token0": ETH, "token1": "0x" + "c2" * 20, "dec0": 18, "dec1": 18, "token_in": ETH}
    with transaction() as conn:
        conn.execute("DELETE FROM rh_pools")
        conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals, ref_path) VALUES (%s, 'SPY', 'stock', 18, %s) "
                     "ON CONFLICT (asset) DO UPDATE SET ref_path = EXCLUDED.ref_path, class = 'stock'", ("0x" + "c1" * 20, json.dumps([plain, mgr])))
        conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, quote_asset, is_pons) VALUES ('px', '0xt', %s, '0xt', %s, true)",
                     ("0x" + "c1" * 20, "0x" + "c1" * 20))
        sels = prefetch.selections(conn)
    try:
        logs = [_log(plain["emitter"], [prices.SWAP_V3], 100), _log(mgr["emitter"], ["0xswapsig", pid], 101), _log(mgr["emitter"], ["0xswapsig", "0x" + "ee" * 32], 102)]
        rpc = FakeRpc(hs=object())
        pre = prefetch.Prefetch(rpc, 90, 110, logs, {}, sels, "hypersync")
        got = prices.fetch_pool_logs(pre, {prices._hop_key(plain): plain, prices._hop_key(mgr): mgr}, 95, 110)
        assert pre.passed == 0 and rpc.calls == []
        assert len(got[prices._hop_key(plain)]) == 1 and len(got[prices._hop_key(mgr)]) == 1     # the other pool's swap is not this hop's
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM rh_pools"); conn.execute("UPDATE rh_assets SET ref_path = NULL WHERE asset = %s", ("0x" + "c1" * 20,))


def test_the_next_pass_starts_when_a_minute_can_be_written(monkeypatch):
    monkeypatch.setattr(config, "RH_STREAM_POLL_S", 2.0)
    m = 1_790_000_040 - (1_790_000_040 % 60)
    assert stream._wait_s(m - 60, now=m + 1.0) == pytest.approx(1.5)            # the minute that just ended: at its grace
    assert stream._wait_s(m - 60, now=m + 30.0) == pytest.approx(0.05)          # overdue: at once
    assert stream._wait_s(m, now=m + 30.0) == pytest.approx(2.0)                # written: the poll, at most
    assert stream._wait_s(m, now=m + 59.0) == pytest.approx(2.0)
