"""The price scan moves every quote asset together: a reference pool two assets share is fetched once per range, each
asset only writes minutes past its own cursor, and the laggard sets the range."""
import json

from fly_trader.db.connection import transaction
from fly_trader.rh import prices

USDG, SPY, NVDA = "0x" + "a1" * 20, "0x" + "a2" * 20, "0x" + "a3" * 20
P_US, P_SPY, P_NV = "0x" + "b1" * 20, "0x" + "b2" * 20, "0x" + "b3" * 20
WETH = "0x" + "ff" * 20


def _hop(pool, tin, tout):
    t0, t1 = sorted([tin, tout], key=lambda a: int(a, 16))
    return {"pool": pool, "emitter": pool, "token0": t0, "token1": t1, "dec0": 18, "dec1": 18, "token_in": tin}


def test_shared_pool_fetched_once_and_cursors_move_together(db_conn, monkeypatch):
    paths = {SPY: [_hop(P_SPY, SPY, USDG), _hop(P_US, USDG, WETH)], NVDA: [_hop(P_NV, NVDA, USDG), _hop(P_US, USDG, WETH)]}
    with transaction() as conn:
        for t in ("rh_base_prices", "rh_pools", "rh_assets", "rh_scan"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("INSERT INTO rh_scan (name, block, range_blocks, detail) VALUES ('rh', 5000, 1000, %s)", (json.dumps({"start_block": 1000}),))
        for a, pool in ((SPY, "0xp1"), (NVDA, "0xp2")):
            conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals, ref_path) VALUES (%s, 'X', 'stock', 18, %s)", (a, json.dumps(paths[a])))
            conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, quote_asset, is_pons) VALUES (%s, '0xt', %s, '0xt', %s, true)", (pool, a, a))
        conn.execute("INSERT INTO rh_scan (name, block, range_blocks) VALUES (%s, 2999, 1000), (%s, 1999, 1000)", (f"price:{SPY}", f"price:{NVDA}"))
    fetched = []
    sq = 2 ** 96

    def swap_logs(rpc, h, lo, hi):
        fetched.append((h["pool"], lo, hi))
        return [{"blockNumber": hex(b), "logIndex": "0x0", "data": "0x" + "00" * 64 + format(sq, "064x")} for b in range(lo, hi + 1, 100)]
    monkeypatch.setattr(prices, "_swap_logs", swap_logs)
    monkeypatch.setattr(prices, "logs_rpc", lambda: None)
    monkeypatch.setattr("fly_trader.rh.blocktime.times", lambda rpc, lo, hi, bs: {b: 1_790_000_000 + b for b in bs})
    prices.run_once(max_ranges=10)
    rounds = sorted({(lo, hi) for _, lo, hi in fetched})
    assert rounds[0][0] == 2000                                                        # the laggard (NVDA) sets the first range
    for lo, hi in rounds:
        assert sum(1 for p, a, b in fetched if p == P_US and (a, b) == (lo, hi)) == 1  # the shared USDG pool once per range
    with transaction() as conn:
        cur = {r["name"]: int(r["block"]) for r in conn.execute("SELECT name, block FROM rh_scan WHERE name LIKE 'price:%%'").fetchall()}
        first_spy = conn.execute("SELECT min(ts) AS t FROM rh_base_prices WHERE asset = %s", (SPY,)).fetchone()["t"]
    assert cur == {f"price:{SPY}": 5000, f"price:{NVDA}": 5000}
    assert int(first_spy.timestamp()) >= (1_790_000_000 + 3000) // 60 * 60            # SPY wrote nothing before its own cursor
