"""Robinhood Chain's other Uniswap v4 pools (2026-10-23): a token's contract check, which pools qualify, a pool's activity
against the floors, the slot's live-first gate, a coin's minute taken from its busiest live pool, the price filter's
same-day history, and a plain v4 coin held back until its feature windows are full."""

from types import SimpleNamespace

from fly_trader import config
from fly_trader.rh import abi, token_check as T, v4_universe as U

ZERO = "0x" + "00" * 20
WETH = config.RH_WETH.lower()
USDG = config.RH_USDG.lower()
TOK = "0x" + "ab" * 20
PONS = "0x" + "cd" * 20


def _dispatch(*sigs) -> str:
    """Bytecode with a dispatcher entry (DUP1 PUSH4 sel EQ) per signature."""
    body = b"".join(b"\x80\x63" + abi.selector(s) + b"\x14\x61\x00\x00\x57" for s in sigs)
    return "0x" + body.hex()


def test_selectors_and_minimal_proxies():
    code = _dispatch("transfer(address,uint256)", "mint(address,uint256)")
    assert T.selectors(code) == {abi.selector("transfer(address,uint256)").hex(), abi.selector("mint(address,uint256)").hex()}
    impl = "11" * 20
    for prefix in T.CLONE_PREFIXES:
        assert T.clone_target("0x" + prefix + impl + "5af43d") == "0x" + impl
    assert T.clone_target(code) is None


def test_contract_check_decisions():
    sel = lambda *s: {abi.selector(x).hex() for x in s}        # noqa: E731
    ok = dict(owner_live=False, upgradeable=False, decimals=18, symbol="UHI", name="Unhinged")
    assert T.judge(sel("transfer(address,uint256)"), **ok)[0]                                        # no powers
    assert T.judge(sel("mint(address,uint256)"), **ok)[0]                                            # a power, owner renounced
    assert not T.judge(sel("mint(address,uint256)"), **{**ok, "owner_live": True})[0]                 # a live owner can mint
    assert not T.judge(sel("pause()", "hasRole(bytes32,address)"), **ok)[0]                          # an admin role can pause
    assert not T.judge(sel("buyTax()"), **ok)[0]                                                     # a built-in tax
    assert not T.judge(set(), **{**ok, "upgradeable": True})[0]
    assert not T.judge(set(), **{**ok, "decimals": 9})[0]
    assert not T.judge(set(), **{**ok, "symbol": "USDC"})[0]
    assert not T.judge(set(), **{**ok, "name": "NVIDIA • Robinhood Token"})[0]


def _init(c0, c1, hooks=ZERO, fee=3000):
    a, b = sorted([c0, c1], key=lambda x: int(x, 16))
    return {"pool_id": "0x" + "99" * 32, "currency0": a, "currency1": b, "hooks": hooks, "fee": fee, "tick_spacing": 60, "block": 1, "tx_hash": "0xt"}


def test_which_pools_qualify(monkeypatch):
    monkeypatch.setattr(config, "RH_V4_QUOTES", (USDG,)); monkeypatch.setattr(config, "RH_V4_HOOKS", ())
    assets = {USDG: {"class": "stable", "priced": True}, ZERO: {"class": "eth", "priced": True}, "0x" + "ee" * 20: {"class": "stock", "priced": False}}
    assert U.classify(_init(TOK, WETH), assets, {}) == {"token": TOK, "quote": WETH, "kind": "v4"}
    assert U.classify(_init(TOK, USDG), assets, {})["kind"] == "v4"
    assert U.classify(_init(PONS, USDG), assets, {PONS: "graduated"}) == {"token": PONS, "quote": USDG, "kind": "pons_other"}
    assert U.classify(_init(PONS, USDG), assets, {PONS: "curve"}) is None                    # a Pons coin before its graduation: never
    assert U.classify(_init(TOK, WETH, hooks="0x" + "77" * 20), assets, {}) is None          # a hook may take fees the Swap event hides
    assert U.classify(_init(TOK, WETH, hooks=config.PONS_HOOK.lower()), assets, {}) is None   # a Pons pool registers itself
    assert U.classify(_init(WETH, USDG), assets, {}) is None                                  # two quote assets
    assert U.classify(_init("0x" + "ee" * 20, WETH), assets, {}) is None                     # a stock token is no memecoin
    monkeypatch.setattr(config, "RH_V4_HOOKS", ("0x" + "77" * 20,))
    assert U.classify(_init(TOK, WETH, hooks="0x" + "77" * 20), assets, {})["kind"] == "v4"   # an allowed hook


def test_activity_against_the_floors():
    b = 1000
    a = [b, 3, 0, 0, 4, 0, 0, 0, 0, 0, 0.0]                        # 3 swaps in this bucket, 4 in the one before
    assert U.activity(a, b)[0] == 7
    assert U.activity(a, b + 1)[0] == 3                              # the bucket before has left the half hour
    assert U.activity(a, b + 2)[0] == 0
    min_resq, min_vol = U.floors()
    q96 = 2 ** 96                                                    # price 1: equal virtual reserves of L / 1e18 each
    liq = int(2 * min_resq * 1e18)
    rich = [b, 10, int(2 * min_vol * 1e18), int(2 * min_vol * 1e18), 0, 0, 0, q96, liq, 0, 0.0]
    assert U.clears(rich, b, True, 18, 1.0)[0]
    assert not U.clears(rich, b, True, 18, 0.1)[0]                   # the same pool quoted in an asset worth 0.1 ETH
    thin = list(rich); thin[3] = 0                                   # no quote-side volume (the token is currency0)
    assert not U.clears(thin, b, True, 18, 1.0)[0]


def test_the_slot_waits_for_live_orders_and_the_minute(monkeypatch):
    from fly_trader.rh import prefetch, stream
    ran = []
    monkeypatch.setattr(U, "slot", lambda deadline, now=None: ran.append(deadline) or {"ran": True})
    monkeypatch.setattr(stream, "_status", lambda **kv: None)
    monkeypatch.setattr(prefetch, "hypersync_down", lambda: False)
    monkeypatch.setattr(stream, "_orders_pending", lambda: False)
    m0 = 1_900_000_020 - 1_900_000_020 % 60
    written = {"minutes": {"through": m0}}
    stream._V4["minute"] = None
    assert stream.v4_slot(written, now=m0 + 5) == {}                  # before mid-minute
    assert stream.v4_slot({"minutes": {"through": m0 - 60}}, now=m0 + 25) == {}   # the newest minute is not written yet
    monkeypatch.setattr(stream, "_orders_pending", lambda: True)
    assert stream.v4_slot(written, now=m0 + 25) == {}                 # a live order is queued or in flight
    monkeypatch.setattr(stream, "_orders_pending", lambda: False)
    assert stream.v4_slot(written, now=m0 + 25) == {"ran": True} and ran == [m0 + 40]
    assert stream.v4_slot(written, now=m0 + 30) == {}                 # once a minute
    assert stream.v4_slot({"minutes": {"through": m0 + 60}}, now=m0 + 60 + 45) == {}   # past the window


def test_a_coins_minute_comes_from_its_busiest_pool(db_conn):
    from fly_trader.rh import minutes
    t0 = 1_900_000_000 - 1_900_000_000 % 86400 + 3600; tok = "0x" + "a1" * 20; p1, p2 = "0x" + "b1" * 32, "0x" + "b2" * 32
    db_conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals) VALUES (%s, 'ETH', 'eth', 18) ON CONFLICT DO NOTHING", (ZERO,))
    for pid in (p1, p2):
        db_conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, quote_asset, is_pons, kind, indexed) VALUES (%s,%s,%s,%s,false,%s,false,'v4',true) "
                        "ON CONFLICT DO NOTHING", (pid, tok, ZERO, tok, ZERO))
    rows = [(p1, 1, 1.0), (p2, 2, 1.1), (p2, 3, 1.2)]                   # p2 has two swaps in the minute, p1 one
    for i, (pid, k, px) in enumerate(rows):
        db_conn.execute("INSERT INTO rh_swaps (block, log_index, tx_hash, ts, pool_id, token, trader, side, token_raw, quote_raw, price_q, resq_q, fee_frac, fee_v) "
                        "VALUES (%s, %s, 'tx', to_timestamp(%s), %s, %s, 'w', 1, 1, %s, %s, 5, 0.003, 2) ON CONFLICT DO NOTHING",
                        (10_000_000 + i, k, t0 + 5 + i, pid, tok, 10 ** 17, px))
    db_conn.commit()
    try:
        out = minutes.aggregate(db_conn, t0, t0 + 60, None, tokens=[tok])
        assert len(out) == 1 and out[0][2] == p2 and out[0][8] == 1.2 and out[0][13] == 2                # pool, close, buys of p2 only
        assert abs(out[0][18] - 0.003) < 1e-12                                                          # its LP fee
        assert minutes.aggregate(db_conn, t0, t0 + 60, None) == out                                      # live: both pools are indexed
        db_conn.execute("UPDATE rh_pools SET indexed = false WHERE pool_id = %s", (p2,)); db_conn.commit()
        live = minutes.aggregate(db_conn, t0, t0 + 60, None)
        assert len(live) == 1 and live[0][2] == p1                                                       # a pool not yet live never counts live
    finally:
        db_conn.execute("DELETE FROM rh_swaps WHERE token = %s", (tok,)); db_conn.execute("DELETE FROM rh_pools WHERE token = %s", (tok,)); db_conn.commit()


def test_the_price_filter_remembers_the_day_before_the_first_minute(db_conn):
    from fly_trader.rh import minutes
    day0 = 1_900_000_000 - 1_900_000_000 % 86400; t0 = day0 + 7200; tok = "0x" + "a2" * 20; pid = "0x" + "b3" * 32
    db_conn.execute("INSERT INTO rh_assets (asset, symbol, class, decimals) VALUES (%s, 'ETH', 'eth', 18) ON CONFLICT DO NOTHING", (ZERO,))
    db_conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, quote_asset, is_pons, kind, indexed) VALUES (%s,%s,%s,%s,false,%s,true,'pons',false) "
                    "ON CONFLICT DO NOTHING", (pid, tok, ZERO, tok, ZERO))
    rows = [(day0 - 600, 500.0), (day0 + 60, 1.0), (day0 + 120, 1.0), (day0 + 180, 1.0), (t0 + 5, 100.0), (t0 + 10, 1.2)]
    for i, (ts, px) in enumerate(rows):                              # yesterday's 500 is another day's median: forgotten
        db_conn.execute("INSERT INTO rh_swaps (block, log_index, tx_hash, ts, pool_id, token, trader, side, token_raw, quote_raw, price_q, resq_q, fee_frac, fee_v) "
                        "VALUES (%s, 0, 'tx', to_timestamp(%s), %s, %s, 'w', 1, 1, %s, %s, 5, 0.01, 2) ON CONFLICT DO NOTHING", (20_000_000 + i, ts, pid, tok, 10 ** 17, px))
    db_conn.commit()
    try:
        out = minutes.aggregate(db_conn, t0, t0 + 60, None)
        assert len(out) == 1 and out[0][5] == 1.2 and out[0][13] == 1                                    # 100 is 100x the day's median: dropped
    finally:
        db_conn.execute("DELETE FROM rh_swaps WHERE token = %s", (tok,)); db_conn.execute("DELETE FROM rh_pools WHERE token = %s", (tok,)); db_conn.commit()


def test_a_plain_v4_coin_waits_for_full_feature_windows():
    from fly_trader.agent.minute_engine import MinuteEngine, V4_WARM_H
    eng = SimpleNamespace(rh=True, spec=SimpleNamespace(min_resq=lambda: 0.1, min_vol_15m=lambda: 0.1))
    info = {"broken": False, "resq": 10.0, "logvol_15m": 5.0, "source": "v4", "seen_h": V4_WARM_H - 0.1}
    assert not MinuteEngine.eligible(eng, info)
    assert MinuteEngine.eligible(eng, {**info, "seen_h": V4_WARM_H})
    assert MinuteEngine.eligible(eng, {**info, "source": "pons", "seen_h": 0.0})                     # a Pons coin: as before
