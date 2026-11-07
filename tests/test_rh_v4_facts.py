"""Live launch facts of plain v4 coins (rh/v4_facts.py): the launch block, who holds what at its end, the row the creator rules read."""
import json
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config
from fly_trader.rh import index, stream, v4, v4_facts as F

ZERO = F.ZERO; PM = config.V4_POOL_MANAGER.lower(); TOK = "0x" + "c0" * 20; DEV = "0x" + "d0" * 20; POOL = "0x" + "e0" * 20; LOCK = "0x" + "e1" * 20
A, B, C, D = ("0x" + x * 20 for x in ("a1", "a2", "a3", "a4")); SUPPLY = 10 ** 27


class Rpc:
    """Code sizes and the mint transaction; the Transfer and Swap logs of a coin."""

    def __init__(self, sizes=None, transfers=None, swaps=None, sender=DEV):
        self.sizes, self.transfers, self.swaps, self.sender = sizes or {}, transfers or [], swaps or [], sender
        self.read: list[str] = []; self.ranges: list[tuple] = []; self.blocks = self

    def batch(self, calls):
        self.read += [c[1][0] for c in calls]
        return ["0x" + "00" * self.sizes.get(c[1][0], 0) for c in calls]

    def tx_by_hash(self, h):
        return {"from": self.sender}

    def call(self, method, params):
        assert method == "eth_getBlockByNumber"
        return {"timestamp": hex(1_900_000_000 + int(params[0], 16))}

    def get_logs_multi(self, addresses, lo, hi, topics):
        self.ranges.append((lo, hi))
        src = self.swaps if addresses == [config.V4_POOL_MANAGER] else self.transfers
        return [x for x in src if lo <= int(x["blockNumber"], 16) <= hi]

    served_last = True


def row(b, s, d, v, tx="0xmint"):
    return (b, tx, s, d, v)


def log(b, i, s, d, v, tx="0xmint"):
    return {"blockNumber": hex(b), "logIndex": hex(i), "transactionHash": tx, "topics": [index.TRANSFER, "0x" + "00" * 12 + s[2:], "0x" + "00" * 12 + d[2:]], "data": hex(v)}


def test_a_coin_seeded_by_hand_into_a_v4_pool():
    rows = [row(10, ZERO, DEV, SUPPLY), row(11, DEV, PM, SUPPLY * 80 // 100), row(12, DEV, A, SUPPLY * 5 // 100), row(50, PM, B, SUPPLY // 100), row(60, PM, C, SUPPLY // 100)]
    f = F.launch(TOK, rows, F.Codes(Rpc()))
    assert f["L"] == 50 and f["mint_block"] == 10 and f["mint_tx"] == "0xmint"
    assert f["owned"] == {DEV: pytest.approx(0.15), A: pytest.approx(0.05), B: pytest.approx(0.01)}              # the pool's share is nobody's; C bought after L


def test_a_launch_on_a_factory_pool_counts_the_pool_as_a_market():
    rows = [row(10, ZERO, DEV, SUPPLY), row(10, DEV, POOL, SUPPLY * 9 // 10), row(10, POOL, A, SUPPLY // 100), row(11, POOL, B, SUPPLY // 100), row(12, POOL, C, SUPPLY // 100)]
    f = F.launch(TOK, rows, F.Codes(Rpc({POOL: 5000})))
    assert f["L"] == 10 and set(f["owned"]) == {DEV, A} and f["owned"][DEV] == pytest.approx(0.1)                 # three payees in the window: a market
    assert f["hhi"] == pytest.approx((0.1 ** 2 + 0.01 ** 2) / 0.11 ** 2)


def test_a_contract_that_pays_nobody_is_owned_it_fails_closed():
    rows = [row(10, ZERO, DEV, SUPPLY), row(11, DEV, PM, SUPPLY // 4), row(12, DEV, LOCK, SUPPLY * 70 // 100), row(50, PM, A, SUPPLY // 100)]
    f = F.launch(TOK, rows, F.Codes(Rpc({LOCK: 9000})))
    assert f["L"] == 50 and f["owned"][LOCK] == pytest.approx(0.7) and sum(f["owned"].values()) == pytest.approx(0.76)


def test_a_wallet_paying_out_is_no_launch():
    delegated = "0x" + "b7" * 20                       # an EIP-7702 wallet: 23 bytes of code
    rows = [row(10, ZERO, delegated, SUPPLY), row(11, delegated, A, SUPPLY // 10), row(12, delegated, B, SUPPLY // 10), row(13, delegated, C, SUPPLY // 10)]
    assert F.launch(TOK, rows, F.Codes(Rpc({delegated: 23})))["L"] is None


def test_burns_are_not_buyers_and_nothing_minted_is_no_coin():
    rows = [row(10, ZERO, DEV, SUPPLY), row(10, DEV, POOL, SUPPLY), row(11, POOL, "0x000000000000000000000000000000000000dead", SUPPLY // 2), row(12, POOL, TOK, SUPPLY // 100), row(13, POOL, A, SUPPLY // 100)]
    assert F.launch(TOK, rows, F.Codes(Rpc({POOL: 5000})))["L"] == 13
    assert F.launch(TOK, [row(10, DEV, A, 5)], F.Codes(Rpc())) is None


def test_a_later_launch_widens_the_window_then_stops_past_L():
    logs = [log(100, 0, ZERO, DEV, SUPPLY), log(101, 0, DEV, PM, SUPPLY // 2), log(5_000, 0, PM, A, SUPPLY // 100), log(5_100, 0, PM, B, SUPPLY // 100), log(9_000, 0, PM, C, SUPPLY // 100)]
    rpc = Rpc(transfers=logs); f = F.find_launch(rpc, F.Codes(rpc), TOK, 100, 1_000_000)
    assert f["L"] == 5_000 and rpc.ranges[0] == (0, 700) and rpc.ranges[1] == (701, 36_100)
    assert f["owned"][A] == pytest.approx(0.01) and B not in f["owned"]


def test_the_window_reaches_past_a_launch_near_its_end():
    logs = [log(100, 0, ZERO, DEV, SUPPLY), log(101, 0, DEV, PM, SUPPLY // 2), log(690, 0, PM, A, SUPPLY // 100), log(800, 0, PM, B, SUPPLY // 100)]
    rpc = Rpc(transfers=logs); F.find_launch(rpc, F.Codes(rpc), TOK, 100, 1_000_000)
    assert rpc.ranges == [(0, 700), (701, 990)]


def test_no_launch_up_to_the_head_is_said_and_costs_no_row():
    rpc = Rpc(transfers=[log(100, 0, ZERO, DEV, SUPPLY), log(101, 0, DEV, A, SUPPLY // 10)])
    assert F.facts(rpc, F.Codes(rpc), {"token": TOK, "first_block": 100, "pools": []}, 2_000) == {"why": "no launch found"}
    rpc = Rpc(transfers=[log(100, 0, ZERO, DEV, SUPPLY), log(101, 0, DEV, A, SUPPLY // 10)])
    assert F.facts(rpc, F.Codes(rpc), {"token": TOK, "first_block": 100, "pools": []}, 100_000_000) == {"why": "no launch found"} and rpc.ranges[-1][1] == 3_600_100
    assert F.facts(Rpc(), F.Codes(Rpc()), {"token": TOK, "first_block": 100, "pools": []}, 2_000) == {"why": "nothing minted in its Transfers"}


def test_the_first_swap_is_looked_for_from_the_registered_pool_and_never_up_to_the_head():
    """A coin whose registered pool opened long after its first pool: the search starts at that pool. HOOD's busy pool cost a 76M-block pull when it did not."""
    pool = "0x" + "f1" * 32
    rpc = Rpc(transfers=[log(100, 0, ZERO, DEV, SUPPLY), log(101, 0, DEV, PM, SUPPLY // 2), log(200, 0, PM, A, SUPPLY // 100)], swaps=[_swap_log(50_000_100, pool)])
    f = F.facts(rpc, F.Codes(rpc), {"token": TOK, "first_block": 100, "pools": [{"pool_id": pool, "init_block": 50_000_000, "quote_asset": ZERO, "token_is_0": True, "qdec": 18}]}, 100_000_000)
    assert f["first_swap"]["at"] == datetime.fromtimestamp(1_900_000_000 + 50_000_100, timezone.utc) and rpc.ranges[-1] == (50_000_000, 50_036_000)
    rpc = Rpc(swaps=[_swap_log(9_000_000, pool)])
    assert F.first_swap(rpc, [pool], 100, 100_000_000) is None and rpc.ranges[-1][1] == 3_600_100          # past the last window: rq0 stays unknown


def _swap_log(block, pool, sqrt=2 ** 96, liq=10 ** 18):
    words = [(-5 % 2 ** 256), 7, sqrt, liq, 0, 3000]
    return {"blockNumber": hex(block), "logIndex": "0x0", "transactionHash": "0xs", "topics": [v4.TOPIC["Swap"], pool, "0x" + "00" * 12 + A[2:]], "data": "0x" + "".join(f"{w:064x}" for w in words)}


def test_the_facts_of_a_coin():
    pool = "0x" + "f1" * 32
    rpc = Rpc(transfers=[log(100, 0, ZERO, DEV, SUPPLY), log(101, 0, DEV, PM, SUPPLY * 90 // 100), log(200, 0, PM, A, SUPPLY // 100)], swaps=[_swap_log(200, pool), _swap_log(300, pool)])
    f = F.facts(rpc, F.Codes(rpc), {"token": TOK, "first_block": 100, "pools": [{"pool_id": pool, "init_block": 100, "quote_asset": ZERO, "token_is_0": True, "qdec": 18}]}, 2_000)
    assert f["creator"] == DEV and f["dev_share"] == pytest.approx(0.1) and f["dev_hold_share"] == pytest.approx(0.1) and f["bundle_share"] == pytest.approx(0.01)
    assert f["ttg_min"] == pytest.approx(100 / 60) and f["curve_known"] == 1.0 and f["supply"] == pytest.approx(1e9)
    assert f["first_swap"]["reserve"] == pytest.approx(1.0) and f["first_swap"]["at"] == datetime.fromtimestamp(1_900_000_200, timezone.utc)


def test_a_failed_read_raises_so_the_coin_is_tried_again():
    class Bad(Rpc):
        def batch(self, calls):
            return [RuntimeError("rate limited") for _ in calls]
    with pytest.raises(RuntimeError):
        F.Codes(Bad()).need([A])


@pytest.fixture
def coin(db_conn):
    now = datetime.now(timezone.utc); old = now - timedelta(hours=30); pool = "0x" + "f2" * 32
    tables = ("rh_insiders", "rh_meta", "rh_pools", "rh_minutes", "rh_v4_tokens")

    def clear():
        for t in tables:
            c = {"rh_v4_tokens": "token", "rh_pools": "token", "rh_minutes": "mint", "rh_meta": "mint", "rh_insiders": "mint"}[t]
            db_conn.execute(f"DELETE FROM {t} WHERE {c} LIKE '0xc0%%' OR {c} LIKE '0xc1%%'")
        db_conn.commit()
    clear()

    def add(tok, g=old, resq=5.0, vol=3.0, detail=None, ttg=None):
        db_conn.execute("INSERT INTO rh_v4_tokens (token, status, first_block, first_ts, first_pool, detail) VALUES (%s, 'ok', 100, %s, %s, %s)", (tok, g, pool, detail))
        db_conn.execute("INSERT INTO rh_meta (mint, graduated_at, create_ts, creator, curve_known, source, ttg_min) VALUES (%s, %s, %s, %s, 0, 'v4', %s)", (tok, g, g, A, ttg))
        db_conn.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, token_is_0, quote_asset, is_pons, kind, indexed) VALUES (%s,%s,%s,%s,true,%s,false,'v4',true)",
                        (tok[:60] + "f2", tok, tok, ZERO, ZERO))
        db_conn.execute("INSERT INTO rh_minutes (mint, ts, pool_id, buy_eth, sell_eth, resq_eth, close) VALUES (%s, %s, %s, %s, 0, %s, 1.0)",
                        (tok, now.replace(second=0, microsecond=0) - timedelta(minutes=2), tok[:60] + "f2", vol, resq))
        db_conn.commit()
    yield add
    clear()


def test_due_takes_old_busy_coins_without_facts_busiest_first(db_conn, coin):
    young = datetime.now(timezone.utc) - timedelta(hours=2)
    a, b = "0xc0" + "01" * 19, "0xc0" + "02" * 19
    coin(a, vol=1.0); coin(b, vol=9.0)
    coin("0xc0" + "03" * 19, g=young)                                  # too young
    coin("0xc0" + "04" * 19, resq=0.01)                                # under the reserve floor
    coin("0xc0" + "05" * 19, vol=0.01)                                 # under the volume floor
    coin("0xc0" + "06" * 19, ttg=3.0)                                  # has its facts
    db_conn.commit()
    got = F.due(db_conn)
    assert [c["token"] for c in got] == [b, a] and got[0]["first_block"] == 100 and got[0]["pools"][0]["qdec"] == 18


def test_a_coin_without_a_launch_waits_a_day(db_conn, coin):
    t = "0xc0" + "07" * 19; coin(t)
    F.note_failure(db_conn, t, "no launch found"); db_conn.commit()
    assert F.due(db_conn) == []
    assert [c["token"] for c in F.due(db_conn, now=datetime.now(timezone.utc) + timedelta(hours=F.RETRY_H + 1))] == []   # its minutes are old by then; below, with a fresh one
    db_conn.execute("UPDATE rh_v4_tokens SET detail = jsonb_set(detail, '{facts,at}', to_jsonb((now() - interval '25 hours')::text)) WHERE token = %s", (t,)); db_conn.commit()
    assert [c["token"] for c in F.due(db_conn)] == [t]


def test_the_row_is_written_once_and_the_universe_reads_it(db_conn, coin):
    from fly_trader.train.decisions import X_COLS
    from fly_trader.train.selector import in_universe
    import numpy as np
    t = "0xc0" + "08" * 19; coin(t); g = datetime.now(timezone.utc) - timedelta(hours=30); wallet = "0xc1" + "09" * 19
    f = {"graduated_at": g, "create_ts": g - timedelta(minutes=3), "creator": DEV, "supply": 1e9, "ttg_min": 3.0, "dev_sol": 0.0, "dev_tokens": 1e8, "dev_share": 0.1, "bundle_share": 0.05, "dev_hold_share": 0.1,
         "dev_sold_frac": 0.0, "grad_hhi": 0.2, "curve_known": 1.0, "owned": {DEV: 0.1, wallet: 0.05}, "rq0": None, "first_swap": {"at": g, "quote_asset": ZERO, "reserve": 2.5}}
    assert F.write(db_conn, t, f) is True
    r = db_conn.execute("SELECT * FROM rh_meta WHERE mint = %s", (t,)).fetchone()
    assert r["ttg_min"] == 3.0 and r["creator"] == DEV and r["rq0"] == 2.5 and r["curve_known"] == 1.0 and r["dev_share"] == pytest.approx(0.1) and r["source"] == "v4"
    assert r["prior_known"] == 0 and r["prior_grads"] == 0
    assert dict(db_conn.execute("SELECT wallet, kind FROM rh_insiders WHERE mint = %s AND wallet = %s", (t, DEV)).fetchone())["kind"] == "dev"
    assert db_conn.execute("SELECT kind FROM rh_insiders WHERE mint = %s AND wallet = %s", (t, wallet)).fetchone()["kind"] == "bundle"
    assert F.write(db_conn, t, {**f, "dev_share": 0.9}) is False                                           # a row that has its facts stays as it is
    assert db_conn.execute("SELECT dev_share FROM rh_meta WHERE mint = %s", (t,)).fetchone()["dev_share"] == pytest.approx(0.1)
    x = np.zeros((2, len(X_COLS)), np.float32); ix = {c: i for i, c in enumerate(X_COLS)}
    x[:, [ix["meta_known"], ix["curve_known"]]] = 1.0; x[:, ix["log_age_h"]] = np.log1p(30.0)
    x[0, [ix["dev_share"], ix["dev_hold_share"], ix["bundle_share"]]] = 0.1, 0.1, 0.05                       # this coin
    x[1, [ix["dev_share"], ix["dev_hold_share"], ix["bundle_share"]]] = 0.1, 0.1, 0.65                       # over the 0.7 line
    assert in_universe(x, X_COLS).tolist() == [True, False]


def test_a_runs_hypersync_calls_take_the_slot_behind_live(monkeypatch):
    m0 = 1_900_000_020 - 1_900_000_020 % 60; t = [m0 + 2.0]
    clock, sleep = (lambda: t[0]), (lambda s: t.__setitem__(0, t[0] + s))
    monkeypatch.setitem(F._SLOT, "minute", None)
    monkeypatch.setattr(F, "_minute_written", lambda end: t[0] >= end + 5.0)          # the stream writes each minute 5 s in
    at = []
    for _ in range(3):
        F._slot(clock, sleep); at.append(t[0] - m0)
    assert at == [8.0, 8.0, 68.0]                                                     # two a minute, from 8 s in
    monkeypatch.setattr(F, "_minute_written", lambda end: t[0] >= end + 12.0)         # a late minute: after it
    t[0] = m0 + 120 + 2.0; F._slot(clock, sleep); assert t[0] - m0 - 120 == 12.0
    t[0] = m0 + 180 + 25.0; F._slot(clock, sleep); assert t[0] - m0 - 240 == 12.0     # past the slot: the next minute's
    monkeypatch.setattr(F, "_minute_written", lambda end: False)                       # the stream stopped
    with pytest.raises(TimeoutError):
        F._slot(clock, sleep)


def test_a_run_waits_for_live_trouble_not_for_the_429_cap(db_conn, monkeypatch):
    from fly_trader.rh import hypersync
    old = db_conn.execute("SELECT value FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()

    def status(**kv):
        db_conn.execute("INSERT INTO ui_settings (key, value) VALUES ('rh_stream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                        (json.dumps({"mode": "live", "behind_blocks": 11, "updated_at": datetime.now(timezone.utc).isoformat(), **kv}),))
        db_conn.commit()

    def call(status_code, ok):
        db_conn.execute("INSERT INTO api_calls (ts, service, endpoint, method, status, latency_ms, ok) VALUES (now(), 'rh_hypersync', '/query', 'POST', %s, 100, %s)", (status_code, ok))
        db_conn.commit()
    try:
        db_conn.execute("DELETE FROM api_calls WHERE service = 'rh_hypersync'"); db_conn.commit()
        status(); call(200, True); call(429, False)
        assert F.live_trouble(db_conn) is None                                        # the cap's 429s
        status(behind_blocks=F.LAG_BLOCKS + 1); assert "behind" in F.live_trouble(db_conn)
        status(mode="backfill"); assert "behind" in F.live_trouble(db_conn)
        status(updated_at=(datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()); assert F.live_trouble(db_conn) == "the stream is silent"
        m0 = int(datetime.now(timezone.utc).timestamp()) // 60 * 60
        status(flushed_through=datetime.fromtimestamp(m0 - 60, timezone.utc).isoformat())  # the minute ending at m0
        assert F._minute_written(m0) and not F._minute_written(m0 + 60)
        status(); call(None, False); assert "failures" in F.live_trouble(db_conn)      # an outage
        monkeypatch.setattr(hypersync, "_pace", hypersync._pace)
        assert F.run_once()["paused"] and hypersync._pace is F._slot                   # its calls take the slot
    finally:
        db_conn.execute("DELETE FROM api_calls WHERE service = 'rh_hypersync'")
        if old:
            db_conn.execute("UPDATE ui_settings SET value = %s WHERE key = 'rh_stream_status'", (json.dumps(old["value"]) if not isinstance(old["value"], str) else old["value"],))
        else:
            db_conn.execute("DELETE FROM ui_settings WHERE key = 'rh_stream_status'")
        db_conn.commit()


def test_the_stream_starts_one_child_at_most_every_ten_minutes(monkeypatch):
    started = []

    class P:
        pid = 7; returncode = 0
        alive = True

        def poll(self):
            return None if self.alive else 0

    def popen(args, **k):
        started.append(args); return P()
    monkeypatch.setattr(stream.subprocess, "Popen", popen); monkeypatch.setitem(stream._FACTS, "proc", None); monkeypatch.setitem(stream._FACTS, "at", -1e9)
    t = [1000.0]; monkeypatch.setattr(stream.time, "monotonic", lambda: t[0])
    assert stream.facts_apart() == {"started": 7} and started[0][-1] == "rh-v4-facts" and started[0][:3] == ["/usr/bin/nice", "-n", "19"]
    assert stream.facts_apart() == {"running": 7}                                                         # one at a time
    stream._FACTS["proc"].alive = False; assert stream.facts_apart() == {}                                 # it ended: not again yet
    t[0] += stream.FACTS_EVERY_S + 1; assert stream.facts_apart() == {"started": 7} and len(started) == 2
