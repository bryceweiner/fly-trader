"""The take-profit (config.TAKE_PROFIT_X / TAKE_PROFIT_DIP, 2026-10-28): a buy of a coin trading far below its high since
graduation carries a line at twice its entry price on its decision; paper sells it at the first minute that closes at or
above the line (a 'take_profit' exit), never on a minutes-old minute, and does not buy the token again until its hold
would have ended; the live mirror sells a live position of the same decision at once, never one in flight or due by its
hold, and only while the chain's wallet trades; the engine's high counts the stream's minutes before the ones it fed."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import fly_live, live_first, paper_trading
from fly_trader.agent.minute_engine import MinuteEngine, MintState
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker

BOOK = "replay_take_profit_test"
TABLE = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.7, "kelly": 0.4}]
M0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _ctx(conn, m1, prices, high=None, lag_s=0.0):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), agg={m: {} for m in prices}, prices=dict(prices), resqs={m: 1000.0 for m in prices},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6, fee=lambda m: None, cost_label=lambda m, label: label, lag_s=lag_s,
                           high=lambda m: (high or {}).get(m))


def _info(m, price):
    return {"resq": 1000.0, "pool": "P" + m, "age_h": 10.0, "price": price, "mcap": 1e6, "decimals": 6, "program_label": None, "fee_rate": None}


def _trade(conn, broker, run, beat, m1, prices, buy=(), high=None, lag_s=0.0):
    buy = list(buy)
    return paper_trading.trade_minute(_ctx(conn, m1, prices, high, lag_s), book=BOOK, run_id=run, beat_no=beat, broker=broker, kind="selector", mints=buy,
                                      infos=[_info(m, prices[m]) for m in buy], scores=np.full(len(buy), 0.05), threshold=0.01, table=TABLE, horizon_s=14400.0)


@pytest.fixture()
def book(db_conn, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0)

    def wipe():
        with transaction() as conn:
            for t in ("positions", "wealth_marks", "fills"):
                conn.execute(f"DELETE FROM {t} WHERE book = %s", (BOOK,))
            conn.execute("DELETE FROM decisions WHERE detail->>'book' = %s", (BOOK,))
            conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false WHERE id = 1")
    wipe(); yield PaperBroker(BOOK); wipe()


@pytest.fixture()
def clean_live():
    def wipe():
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint LIKE 'LIVE_TP%%'"); conn.execute("DELETE FROM decisions WHERE mint LIKE 'LIVE_TP%%'")
    wipe(); yield; wipe()


def test_a_buy_far_below_its_high_carries_a_line_at_twice_its_entry(book):
    with transaction() as conn:
        _trade(conn, book, str(uuid.uuid4()), 1, M0, {"TPDIPpump": 1.0, "TPNEARpump": 1.0}, buy=["TPDIPpump", "TPNEARpump"],
               high={"TPDIPpump": 2.0, "TPNEARpump": 1.9})
        lines = {p["mint"]: p["take_profit_x"] for p in ledger.open_positions(conn, BOOK)}
    assert lines == {"TPDIPpump": 2.0, "TPNEARpump": None}


def test_paper_sells_at_the_first_fresh_minute_that_closes_at_its_line(book):
    run = str(uuid.uuid4())
    with transaction() as conn:
        _trade(conn, book, run, 1, M0, {"TPDIPpump": 1.0, "TPNEARpump": 1.0}, buy=["TPDIPpump", "TPNEARpump"], high={"TPDIPpump": 2.0, "TPNEARpump": 1.9})
        entry = {p["mint"]: float(p["entry_price"]) for p in ledger.open_positions(conn, BOOK)}
        _trade(conn, book, run, 2, M0 + timedelta(minutes=5), {m: 1.99 * e for m, e in entry.items()})                        # just under the line
        under = {p["mint"] for p in ledger.open_positions(conn, BOOK)}
        at = {m: 2.0 * e for m, e in entry.items()}
        _trade(conn, book, run, 3, M0 + timedelta(minutes=6), at, lag_s=config.ENTRY_MAX_LAG_S + 1)                          # a minutes-old minute
        stale = {p["mint"] for p in ledger.open_positions(conn, BOOK)}
        st = _trade(conn, book, run, 4, M0 + timedelta(minutes=7), at)
        left = {p["mint"] for p in ledger.open_positions(conn, BOOK)}
        sold = conn.execute("SELECT forced_exit_kind, realized_sol, cost_sol FROM positions WHERE book = %s AND mint = 'TPDIPpump'", (BOOK,)).fetchone()
        why = conn.execute("SELECT reason FROM decisions WHERE run_id = %s AND kind = 'selector_exit'", (run,)).fetchall()
    assert under == stale == {"TPDIPpump", "TPNEARpump"} and left == {"TPNEARpump"} and st["exited"] == 1
    assert sold["forced_exit_kind"] == "take_profit" and sold["realized_sol"] > 0.8 * sold["cost_sol"]
    assert [r["reason"] for r in why] == ["took profit: +100% over its entry (its ×2 line)"]


def test_a_token_sold_at_its_line_is_not_bought_again_before_its_hold_ends(book):
    run = str(uuid.uuid4())
    with transaction() as conn:
        _trade(conn, book, run, 1, M0, {"TPDIPpump": 1.0}, buy=["TPDIPpump"], high={"TPDIPpump": 2.0})
        entry = float(ledger.open_positions(conn, BOOK)[0]["entry_price"])
        same = _trade(conn, book, run, 2, M0 + timedelta(minutes=30), {"TPDIPpump": 2.0 * entry}, buy=["TPDIPpump"], high={"TPDIPpump": 5.0})   # sold, still a pick
        again = _trade(conn, book, run, 3, M0 + timedelta(minutes=31), {"TPDIPpump": 1.0}, buy=["TPDIPpump"], high={"TPDIPpump": 2.5})
        rails = [r["rail"] for r in conn.execute("SELECT rail FROM decisions WHERE run_id = %s AND kind = 'blocked' ORDER BY id", (run,)).fetchall()]
        later = _trade(conn, book, run, 4, M0 + timedelta(minutes=241), {"TPDIPpump": 1.0}, buy=["TPDIPpump"], high={"TPDIPpump": 2.5})
    assert (same["exited"], same["entered"], again["entered"]) == (1, 0, 0) and rails == ["took profit", "took profit"] and later["entered"] == 1


def test_no_line_where_the_chain_takes_no_profit_or_the_rule_is_off(book, monkeypatch):
    monkeypatch.setattr(config, "TAKE_PROFIT_CHAINS", ["rh"])
    with transaction() as conn:
        _trade(conn, book, str(uuid.uuid4()), 1, M0, {"TPDIPpump": 1.0}, buy=["TPDIPpump"], high={"TPDIPpump": 2.0})
        off_chain = ledger.open_positions(conn, BOOK)[0]["take_profit_x"]
    monkeypatch.setattr(config, "TAKE_PROFIT_CHAINS", ["sol"]); monkeypatch.setattr(config, "TAKE_PROFIT_X", 0.0)
    assert off_chain is None and paper_trading.take_profit_x(_ctx(None, M0, {}, {"X": 2.0}), SimpleNamespace(chain="sol"), "X", 1.0) is None


class _Worker:
    """execution/worker.ExecutionWorker's queue rules, nothing executed (tests/test_fly_live.py)."""

    def __init__(self):
        self.submitted, self.inflight = [], {}

    def submit(self, req):
        if self.inflight.get(req.mint):
            return False
        self.submitted.append(req); self.inflight.setdefault(req.mint, []).append(req.side); return True

    def pending(self):
        return set(self.inflight)


def test_the_live_mirror_sells_at_the_line_at_once_and_only_there(db_conn, clean_live):
    w = _Worker(); mirror = fly_live.LiveMirror(14400.0, broker=SimpleNamespace(rpc=None, pubkey="PK"), worker=w)
    m1 = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc); run = str(uuid.uuid4())
    with transaction() as conn:
        for mint, line, age in (("LIVE_TP1", 2.0, 600), ("LIVE_TP2", None, 600), ("LIVE_TP3", 2.0, 600), ("LIVE_TP4", 2.0, 14400 + 60), ("LIVE_TP5", 2.0, 600)):
            did = conn.execute("INSERT INTO decisions (run_id, ts, mint, kind, detail) VALUES (%s, %s, %s, 'fly_enter', %s) RETURNING id",
                               (run, m1 - timedelta(seconds=age), mint, json.dumps({"book": "paper_fly", "take_profit_x": line}))).fetchone()["id"]
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, status) "
                         "VALUES ('live', %s, 'P', %s, %s, 1000000, 0.2, 0.0000002, 'open')", (mint, m1 - timedelta(seconds=age), did))
        w.inflight["LIVE_TP3"] = ["sell"]                                     # an order of it in flight
        prices = {"LIVE_TP1": 0.0000004, "LIVE_TP2": 0.0000004, "LIVE_TP3": 0.0000004, "LIVE_TP4": 0.0000004, "LIVE_TP5": 0.00000039}
        n = mirror.take_profits(conn, m1=m1, prices=prices, run_id=run)
        d = conn.execute("SELECT reason, forced, run_id FROM decisions WHERE mint = 'LIVE_TP1' AND kind = 'fly_exit'").fetchone()
    req = w.submitted[-1]
    assert n == 1 and [r.mint for r in w.submitted] == ["LIVE_TP1"]               # no line, in flight, due by its hold, under its line: not sold
    assert (req.side, req.forced_kind, req.slippage_bps, req.amount_in) == ("sell", "take_profit", config.SLIPPAGE_EXIT_BPS, 1000000)
    assert (d["reason"], d["forced"], str(d["run_id"])) == ("took profit: +100% over its entry (its ×2 line)", False, run)


def test_take_profit_sales_go_out_only_while_the_chains_wallet_trades(db_conn, monkeypatch):
    calls = []
    stub = SimpleNamespace(take_profits=lambda conn, m1, prices, run_id: calls.append((m1, prices, run_id)) or 1)
    monkeypatch.setattr(live_first, "mirror_of", lambda chain: stub)
    monkeypatch.setattr(live_first, "signing_refused", lambda chain: None); monkeypatch.setattr(live_first, "trades_live", lambda conn, chain, live: True)
    assert live_first.chain_take_profits("sol", 1_800_000_000.0, {"M": 1.0}, True, "R") == 1
    assert calls == [(datetime.fromtimestamp(1_800_000_000.0, timezone.utc), {"M": 1.0}, "R")]
    assert live_first.chain_take_profits("sol", 1_800_000_060.0, {}, False) == 0                   # the runner may not sign
    monkeypatch.setattr(live_first, "trades_live", lambda conn, chain, live: False)
    assert live_first.chain_take_profits("sol", 1_800_000_060.0, {}, True) == 0                    # no seat, no LIVE_WITHOUT_HANDOVER
    monkeypatch.setattr(live_first, "trades_live", lambda conn, chain, live: True); monkeypatch.setattr(live_first, "mirror_of", lambda chain: None)
    assert live_first.chain_take_profits("rh", 1_800_000_060.0, {}, True) == 0 and len(calls) == 1  # no mirror yet


def test_the_engines_high_counts_the_streams_minutes_before_the_ones_it_fed(db_conn):
    e = MinuteEngine(); t1 = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc).timestamp()
    s = MintState("TPHIGHpump", 6, "P", None, None, None); s.first_seen = t1; s.hi = 1.5; e.states["TPHIGHpump"] = s
    with transaction() as conn:
        conn.execute("DELETE FROM pump_minutes WHERE mint = 'TPHIGHpump'")
        conn.execute("INSERT INTO pump_minutes (mint, ts, close) VALUES ('TPHIGHpump', %s, 3.0), ('TPHIGHpump', %s, 9.0)",
                     (datetime.fromtimestamp(t1 - 3600, timezone.utc), datetime.fromtimestamp(t1 - 60, timezone.utc)))   # the second is the first minute fed
        first = e._high(conn, "TPHIGHpump"); s.hi = 4.0; fed = e._high(conn, "TPHIGHpump")
        conn.execute("DELETE FROM pump_minutes WHERE mint = 'TPHIGHpump'")
    assert first == 3.0 and fed == 4.0 and s.hi_before == 3.0 and e._high(None, "UNSEENpump") is None


def test_the_live_take_profits_go_out_before_the_minutes_features(db_conn, monkeypatch):
    """Live money first: the wallet's take-profit sales need the minute's closes alone, so the live phase sends them the
    moment it reads the minute, before the features and the fly's scores; never on a minute traded long after its end."""
    import time
    from fly_trader.agent import minute_engine
    calls = []
    e = MinuteEngine(live=True)
    book = SimpleNamespace(name="fly", LIVE=True, chain="sol", run_id="R", on_bars=lambda t, bars: None, on_minute=lambda ctx: calls.append("fly") or {})
    e.books = [book]
    monkeypatch.setattr(e, "_aggregate", lambda conn, m0, m1: {"TPpump": {"close": 2.0, "resq": 100.0, "fee_rate": None}})
    monkeypatch.setattr(e, "_minute_rows", lambda conn, agg, m1: calls.append("features") or (np.zeros((0, 1), np.float32), [], [], {}))
    monkeypatch.setattr(e, "stream_fresh", lambda m0: True)
    monkeypatch.setattr(minute_engine.live_first, "prefetch", lambda chain, live, m1: None)
    monkeypatch.setattr(minute_engine.live_first, "chain_take_profits", lambda chain, m1, prices, live, run_id: calls.append(("tp", chain, prices, live, run_id)) or 0)
    now = float(int(time.time() // 60) * 60)
    e.live_phase(now)
    e.live_phase(now - config.ENTRY_MAX_LAG_S - 60)
    assert calls == [("tp", "sol", {"TPpump": 2.0}, True, "R"), "features", "fly", "features", "fly"]
