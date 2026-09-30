"""The Robinhood Chain index: Pons and v4 decoders on real logs (tests/fixtures/rh_logs.json, pinned 2026-09-28), pool
math in both orientations, traders from the memecoin's own transfers, the 10,000-log cap split, reference-pool prices,
and the minute aggregation (ETH conversion, the 50x band, fee weighting, wallet columns, recompute-identical)."""
import json
from pathlib import Path

import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.rh import index, minutes, pons, prices, univ4 as U, v4

LOGS = json.loads((Path(__file__).parent / "fixtures" / "rh_logs.json").read_text())


def test_pons_decoders_on_real_logs():
    tl = pons.decode(LOGS["TokenLaunched"])
    assert tl["event"] == "TokenLaunched" and tl["token"] != tl["curve"] != tl["deployer"] and tl["threshold_raw"] > 0
    buy, sell = pons.decode(LOGS["CurveBuy"]), pons.decode(LOGS["CurveSell"])
    assert buy["side"] == 1 and sell["side"] == -1
    assert abs(buy["fee_raw"] / buy["quote_raw"] - 0.01) < 1e-9 and abs(sell["fee_raw"] / sell["quote_raw"] - 0.01) < 0.002   # the curve's 1 % fee
    reg = pons.decode(LOGS["PoolRegistered"])
    key = U.pool_key(*sorted([reg["quote_asset"], reg["token"]], key=lambda a: int(a, 16)))
    assert U.pool_id(key) == reg["pool_id"]                                              # the hook's id is keccak(PoolKey)
    grad = pons.decode(LOGS["PoolGraduated"]); fee = pons.decode(LOGS["HookFeeCollected"])
    assert grad["pair_amount_raw"] > 0 and fee["fee_raw"] >= 0 and fee["pool_id"].startswith("0x")


def test_v4_decoders_on_real_logs():
    ini = v4.decode(LOGS["v4_Initialize"]); ml = v4.decode(LOGS["v4_ModifyLiquidity"]); sw = v4.decode(LOGS["v4_Swap"])
    assert ini["fee"] == 0 and ini["tick_spacing"] == 200 and ini["hooks"] == config.PONS_HOOK.lower()
    assert v4.is_full_range(ml["tick_lower"], ml["tick_upper"], 200)                      # the locked graduation position
    assert (sw["amount0"] > 0) != (sw["amount1"] > 0) and v4.sqrt_price_word(LOGS["v4_Swap"]) == sw["sqrt_price_x96"]


def test_pool_math_both_orientations():
    sp = int((2.0 ** 0.5) * v4.Q96)                                   # P = 2 currency1 per currency0 (raw)
    assert v4.price1_per_0(sp, 18, 18) == pytest.approx(2.0)
    assert v4.token_price_in_quote(sp, token_is_0=True, dec_token=18, dec_quote=18) == pytest.approx(2.0)    # quote is currency1
    assert v4.token_price_in_quote(sp, token_is_0=False, dec_token=18, dec_quote=18) == pytest.approx(0.5)
    x, y = v4.virtual_reserves(sp, 10 ** 18)
    assert x * y == pytest.approx(10 ** 36) and y / x == pytest.approx(2.0)
    assert v4.quote_reserve(sp, 10 ** 18, token_is_0=False, dec_quote=18) == pytest.approx(x / 1e18)
    usdg = v4.token_price_in_quote(int(((1e-6 * 1e6 / 1e18) ** 0.5) * v4.Q96), token_is_0=True, dec_token=18, dec_quote=6)   # raw P = 1e-18: 1e-6 USDG / token
    assert usdg == pytest.approx(1e-6, rel=1e-9)


def _tr(tok, a, b, v, tx):
    return {"address": tok, "transactionHash": tx, "topics": [index.TRANSFER, "0x" + a[2:].rjust(64, "0"), "0x" + b[2:].rjust(64, "0")], "data": hex(v)}


def test_traders_come_from_the_memecoins_own_transfers():
    tok, pid, pm, ur, kyb = "0x" + "a1" * 20, "0x" + "11" * 32, config.V4_POOL_MANAGER.lower(), config.UNIVERSAL_ROUTER.lower(), "0x" + "e7" * 20
    alice, bob = "0x" + "aa" * 20, "0x" + "bb" * 20
    transfers = [_tr(tok, pm, alice, 100, "0x1"),                                   # UR buy: taken straight to the buyer
                 _tr(tok, pm, kyb, 50, "0x2"), _tr(tok, kyb, bob, 50, "0x2"),      # Kyber buy: through its executor
                 _tr(tok, alice, ur, 70, "0x3"), _tr(tok, ur, pm, 70, "0x3")]       # a sell through the router
    swaps = [{"tx_hash": "0x1", "pool_id": pid, "side": 1}, {"tx_hash": "0x2", "pool_id": pid, "side": 1}, {"tx_hash": "0x3", "pool_id": pid, "side": -1}]
    got = index.traders_from_transfers(transfers, swaps, {pid: tok})
    assert got == {("0x1", pid): alice, ("0x2", pid): bob, ("0x3", pid): alice}


def test_log_queries_split_under_the_cap():
    class R:
        calls = 0

        def get_logs_multi(self, addrs, lo, hi, topics):
            R.calls += 1
            if hi - lo > 100:
                raise RuntimeError("evm rpc eth_getLogs error -32000: logs matched by query exceeds limit of 10000")
            return [{"blockNumber": hex(lo)}]
    import fly_trader.rh.scan as scan
    old = scan.PAUSE_S; scan.PAUSE_S = 0.0
    try:
        out = index._get_logs(R(), ["0x0"], 0, 799, None)
    finally:
        scan.PAUSE_S = old
    assert len(out) == 8 and sorted(int(x["blockNumber"], 16) for x in out) == [0, 100, 200, 300, 400, 500, 600, 700]


def test_reference_hop_prices_invert_for_the_other_direction():
    sp = int((2.0 ** 0.5) * v4.Q96)
    h = {"token0": "0x01", "token1": "0x02", "dec0": 18, "dec1": 18, "token_in": "0x01"}
    assert prices.hop_price(h, sp) == pytest.approx(2.0) and prices.hop_price({**h, "token_in": "0x02"}, sp) == pytest.approx(0.5)


ETH, USDG, MEME1, MEME2 = "0x" + "00" * 20, "0x" + "5f" * 20, "0x" + "a1" * 20, "0x" + "a2" * 20


@pytest.fixture
def minute_db():
    with transaction() as conn:
        for t in ("rh_swaps", "rh_pools", "rh_base_prices", "rh_insiders", "rh_minutes"):
            conn.execute(f"DELETE FROM {t}")
        conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals) VALUES (%s, 'ETH', 'eth', 18), (%s, 'USDG', 'stable', 6) "
                     "ON CONFLICT (asset) DO UPDATE SET decimals = EXCLUDED.decimals, class = EXCLUDED.class", (ETH, USDG))
        conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, quote_asset, is_pons) VALUES ('p1', %s, %s, %s, false, %s, true), "
                     "('p2', %s, %s, %s, true, %s, true)", (MEME1, ETH, MEME1, ETH, MEME2, MEME2, USDG, USDG))
        conn.execute("INSERT INTO rh_insiders (mint, wallet, kind) VALUES (%s, 'dev', 'dev')", (MEME1,))
    yield
    with transaction() as conn:
        for t in ("rh_swaps", "rh_pools", "rh_base_prices", "rh_insiders", "rh_minutes"):
            conn.execute(f"DELETE FROM {t}")


def _swap(conn, i, t, tok, pool, trader, side, quote_raw, price_q, resq_q, fee=0.029):
    conn.execute("INSERT INTO rh_swaps (block, log_index, tx_hash, ts, pool_id, token, trader, side, token_raw, quote_raw, price_q, resq_q, fee_frac) "
                 "VALUES (%s, 0, %s, to_timestamp(%s), %s, %s, %s, %s, 1, %s, %s, %s, %s)", (i, f"0x{i}", t, pool, tok, trader, side, quote_raw, price_q, resq_q, fee))


def test_minutes_in_eth_with_bands_fees_and_wallets(minute_db, monkeypatch):
    monkeypatch.setattr(config, "RH_ETH_PER_SOL", 0.05)
    t0 = 1_790_000_040 - (1_790_000_040 % 60)
    with transaction() as conn:
        _swap(conn, 1, t0 + 5, MEME1, "p1", "alice", 1, 10 ** 17, 1e-8, 4.0)          # 0.1 ETH buy
        _swap(conn, 2, t0 + 20, MEME1, "p1", "dev", -1, 3 * 10 ** 16, 1.1e-8, 3.9, fee=0.03)   # an insider sells 0.03 ETH
        _swap(conn, 3, t0 + 30, MEME1, "p1", "alice", -1, 10 ** 16, 1e-6, 3.9)        # 100x the median: dropped
        _swap(conn, 4, t0 + 70, MEME2, "p2", "bob", 1, 50 * 10 ** 6, 2e-6, 9000.0)     # 50 USDG buy in the next minute
        conn.execute("INSERT INTO rh_base_prices (asset, ts, price_eth) VALUES (%s, to_timestamp(%s), 0.0004)", (USDG, t0 + 60))
        rows = minutes.aggregate(conn, t0, t0 + 120)
        by = {(r[0], int(r[1].timestamp())): r for r in rows}
        r1 = by[(MEME1, t0)]; r2 = by[(MEME2, t0 + 60)]
        # (mint, ts, pool, asset, cls, open, high, low, close, close_q, base, buy, sell, nb, ns, n_traders, resq_eth, resq_q, fee_rate, n_buyers, wash, wash_buy, top_sell, insider_sell, skill)
        assert r1[5] == pytest.approx(1e-8) and r1[8] == pytest.approx(1.1e-8) and r1[6] == pytest.approx(1.1e-8)       # the 100x print never entered
        assert r1[11] == pytest.approx(0.1) and r1[12] == pytest.approx(0.03) and (r1[13], r1[14], r1[15]) == (1, 1, 2)
        assert r1[18] == pytest.approx((0.029 * 0.1 + 0.03 * 0.03) / 0.13) and r1[23] == pytest.approx(0.03)            # volume-weighted fee; insider sell
        assert r2[10] == pytest.approx(0.0004) and r2[8] == pytest.approx(2e-6 * 0.0004) and r2[11] == pytest.approx(50 * 0.0004)
        assert r2[16] == pytest.approx(9000 * 0.0004)                                                                    # reserve in ETH
        n1 = minutes.write(conn, rows); first = [dict(x) for x in conn.execute("SELECT * FROM rh_minutes ORDER BY mint, ts").fetchall()]
        minutes.write(conn, minutes.aggregate(conn, t0, t0 + 120))
        again = [dict(x) for x in conn.execute("SELECT * FROM rh_minutes ORDER BY mint, ts").fetchall()]
        assert n1 == 2 and first == again and not any(r["revised"] for r in again)                                     # a recompute is identical


def test_a_non_eth_minute_without_a_mark_is_not_written(minute_db, monkeypatch):
    monkeypatch.setattr(config, "RH_ETH_PER_SOL", 0.05)
    t0 = 1_790_001_000 - (1_790_001_000 % 60)
    with transaction() as conn:
        _swap(conn, 9, t0 + 5, MEME2, "p2", "bob", 1, 50 * 10 ** 6, 2e-6, 9000.0)
        assert minutes.aggregate(conn, t0, t0 + 60) == []


def test_a_query_kind_that_hit_the_log_cap_is_asked_in_spans_that_work(monkeypatch):
    from fly_trader.rh import index, scan
    monkeypatch.setattr(scan, "PAUSE_S", 0.0); monkeypatch.setattr(index, "_SPAN", {})

    class Rpc:
        calls = refused = 0

        def get_logs_multi(self, addresses, lo, hi, topics):
            self.calls += 1
            if hi - lo + 1 > 3000:                                           # the busy period: > 10k logs above 3000 blocks
                self.refused += 1
                raise RuntimeError("logs matched by query exceeds limit of 10000")
            return [{"blockNumber": hex(b)} for b in range(lo, hi + 1, 1000)]
    r = Rpc()
    first = index._get_logs(r, ["0xa"], 0, 19_999, ["0xt"])
    r1 = r.refused
    again = index._get_logs(r, ["0xa"], 20_000, 39_999, ["0xt"])
    assert [int(x["blockNumber"], 16) for x in first] == list(range(0, 20_000, 1000)) or len(first) >= 20
    assert len(again) >= 20 and r.refused - r1 <= 2                            # the second range barely pays for refusals
