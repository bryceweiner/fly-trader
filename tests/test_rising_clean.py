"""Rising + clean (agent/rising_clean.py): the live breadth equals the study's series, a qualifying pick takes twice its stake
under the same caps on ``paper_rising`` and nothing else changes there, the go-live is judged once from the doubled trades and
moves from the console, and the wallet doubles only after a go-live passed, with no query in front of its orders."""
import json
import math
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from fly_trader import config, markets
from fly_trader.agent import fly_live, fly_session, minute_engine as me, paper_trading, rising_clean as rc, selector_session, sizing
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker
from fly_trader.train.flow import BreadthWindow
from test_fly_live import _Rpc, _Worker

M0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
TABLE = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.7, "kelly": 0.4}]
START = datetime(2026, 9, 1, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def eighth_kelly(monkeypatch):
    """The live stake (since 2026-10-31): at the code's quarter Kelly the bankroll cap would take most of the doubling."""
    monkeypatch.setattr(config, "KELLY_FRACTION", 0.125)


@pytest.fixture()
def test_row(db_conn):
    def wipe():
        with transaction() as conn:
            conn.execute("DELETE FROM ui_settings WHERE key IN (%s, %s)", (rc.KEY, rc.STATUS_KEY))
            for t in ("positions", "wealth_marks", "fills"):
                conn.execute(f"DELETE FROM {t} WHERE book = %s", (rc.BOOK,))
            conn.execute("DELETE FROM decisions WHERE detail->>'book' = %s OR kind LIKE 'rising%%'", (rc.BOOK,))
            conn.execute("DELETE FROM book_state WHERE book = %s", (rc.BOOK,))
            conn.execute("DELETE FROM beats WHERE run_id IN (SELECT run_id FROM runs WHERE kind = %s)", (rc.KIND,))
            conn.execute("DELETE FROM runs WHERE kind = %s", (rc.KIND,))
            conn.execute("DELETE FROM events WHERE source = %s OR message = 'selector session started'", (rc.KIND,))
            conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false WHERE id = 1")
    wipe(); yield; wipe()


# ---- the measure
def study_recipe(rows: pd.DataFrame) -> pd.Series:
    """data/scratch/winners_20261031/w_market.py on (ts, resq, ret_1h): the study's up_chg_6h per 5-minute bucket."""
    d = rows[rows.resq >= 10.0].copy(); d["b"] = (d.ts // 300).astype(int)
    m = d.groupby("b").ret_1h.apply(lambda x: (x > 0).mean()).sort_index()
    m = m.reindex(range(m.index.min(), m.index.max() + 1)).ffill(limit=12)
    return m - m.shift(72)


def test_breadth_is_the_studys_series_through_gaps_thin_pools_and_flat_coins():
    rng = np.random.default_rng(3); rows = []; t0 = 1_800_000_000 // 300 * 300
    for k in range(400):                                                # a day and more of buckets, with holes of 1..20 buckets
        if 150 <= k < 155 or 260 <= k < 280 or k % 37 == 0:
            continue
        for i in range(rng.integers(1, 6)):
            for _ in range(rng.integers(0, 8)):
                rows.append((t0 + k * 300 + i * 60, float(rng.choice([5.0, 9.99, 10.0, 50.0])), float(rng.choice([-0.1, 0.0, 0.2]))))
    df = pd.DataFrame(rows, columns=["ts", "resq", "ret_1h"]).sort_values("ts", kind="stable"); study = study_recipe(df)
    w = BreadthWindow(); checked = known = 0; nxt = 0; ts = df.ts.to_numpy(); rq = df.resq.to_numpy(); r = df.ret_1h.to_numpy()
    for t in range(t0, t0 + 400 * 300, 60):                             # as a minute sees it: everything before it fed, nothing of it yet
        while nxt < len(ts) and ts[nxt] < t:
            w.add(float(ts[nxt]), float(rq[nxt]), float(r[nxt])); nxt += 1
        got = w.change(float(t)); want = study.get(t // 300 - 1, float("nan"))
        checked += 1
        if math.isnan(want):
            assert got is None
        else:
            known += 1; assert got == pytest.approx(want, abs=1e-12)
    assert checked == 2000 and known > 1000
    assert len(w.b) <= 72 + 12 + 3                                      # it keeps only what the change can read


def test_breadth_counts_the_unknown_pool_as_absent_and_a_nan_return_as_not_up():
    w = BreadthWindow(); w.add(0.0, None, 1.0); assert not w.b
    w.add(0.0, 10.0, float("nan")); w.add(60.0, 10.0, 0.5); assert w.b[0] == [2, 1]


def test_the_engine_feeds_breadth_on_solana_warm_up_or_not_and_never_on_rh(db_conn):
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 6.0, "sell": 0.0, "nb": 2, "ns": 0, "n_traders": 2, "resq": 60.0, "pool": "P", "program_label": "Pump.fun Amm"}
    s, h = me.MinuteEngine(), me.MinuteEngine(market=markets.RH); t = 1_800_000_000.0 // 300 * 300 + 60
    s._features(db_conn, "RCpump", a, t); h._features(db_conn, "RCpump", a, t)
    assert sum(c[0] for c in s.breadth.b.values()) == 1 and not h.breadth.b


def test_a_row_carries_its_coins_concentration_and_none_when_unknown(db_conn):
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 6.0, "sell": 0.0, "nb": 2, "ns": 0, "n_traders": 2, "resq": 60.0, "pool": "P", "program_label": "Pump.fun Amm"}
    s = me.MinuteEngine(); t = 1_800_000_000.0
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint IN ('RCHpump', 'RCNpump')")
        cur.execute("INSERT INTO corpus_meta (mint, graduated_at, grad_hhi) VALUES ('RCHpump', now(), 0.0125), ('RCNpump', now(), NULL)")
    db_conn.commit()
    try:
        assert s._features(db_conn, "RCHpump", a, t)[1]["grad_hhi"] == pytest.approx(0.0125)
        assert s._features(db_conn, "RCNpump", a, t)[1]["grad_hhi"] is None and s._features(db_conn, "RCMISSINGpump", a, t)[1]["grad_hhi"] is None
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM corpus_meta WHERE mint IN ('RCHpump', 'RCNpump')")
        db_conn.commit()


# ---- who qualifies
@pytest.mark.parametrize("chg,hhi,ok", [(rc.RISE_6H, rc.CLEAN_HHI, True), (rc.RISE_6H - 1e-9, rc.CLEAN_HHI, False), (rc.RISE_6H, rc.CLEAN_HHI + 1e-9, False),
                                        (None, 0.01, False), (0.1, None, False), (None, None, False), (0.5, 0.0, True)])
def test_both_cuts_must_hold_and_unknown_is_not_clean(chg, hhi, ok):
    assert rc.qualifies(chg, hhi) is ok
    assert rc.mark(chg, {"grad_hhi": hhi})["stake_x"] == (2.0 if ok else 1.0)


def test_the_cuts_are_the_studys_frozen_ones():
    assert (rc.RISE_6H, rc.CLEAN_HHI, rc.STAKE_X) == (0.027572083, 0.019701237, 2.0)


# ---- sizing
def test_the_multiple_is_on_the_stake_and_every_cap_still_binds():
    base, why = sizing.size_position(0.9, 0.8, TABLE, 5.3, 5.3, 1000.0)
    x2, why2 = sizing.size_position(0.9, 0.8, TABLE, 5.3, 5.3, 1000.0, stake_x=2.0)
    assert x2 == pytest.approx(2 * base) and "× 2" in why2 and "×" not in why.replace("× 0", "")
    deployable = 5.3 - config.GAS_RESERVE_SOL
    big = [{**TABLE[0], "kelly": 0.99}]
    assert sizing.size_position(0.9, 0.8, big, 5.3, 5.3, 1000.0, stake_x=50.0)[0] == pytest.approx(config.MAX_POSITION_FRACTION * deployable)   # bankroll cap
    assert sizing.size_position(0.9, 0.8, TABLE, 5.3, 5.3, 10.0, stake_x=2.0)[0] == pytest.approx(0.02 * 10.0)                                  # label size
    assert sizing.size_position(0.9, 0.8, TABLE, 5.3, config.GAS_RESERVE_SOL + 0.05, 1000.0, stake_x=2.0)[0] == pytest.approx(0.05)             # free cash
    assert sizing.size_position(0.9, 0.8, TABLE, 5.3, 5.3, 1000.0, creator_heavy=True, stake_x=2.0)[0] == pytest.approx(config.CREATOR_HEAVY_CAP_FRACTION * 5.3)


# ---- the book
def _ctx(conn, m1, prices, chg, engine=None):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), t_start=m1.timestamp() - 60.0, agg={m: {} for m in prices}, prices=dict(prices), resqs={m: 1000.0 for m in prices},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6, fee=lambda m: None, cost_label=lambda m, label: label, lag_s=0.0, high=lambda m: None,
                           trade=True, fresh=True, mints=list(prices), X=np.zeros((len(prices), 1)), engine=engine or SimpleNamespace(breadth=SimpleNamespace(change=lambda t: chg), has_book=lambda n: False))


def _info(m, price, hhi):
    return {"resq": 1000.0, "pool": "P" + m, "age_h": 10.0, "price": price, "mcap": 1e6, "decimals": 6, "program_label": None, "fee_rate": None, "grad_hhi": hhi}


class _Model:
    """A selector whose picks are every row, one strategy, one table."""
    metrics = {"groups": []}; threshold = 0.01; horizon_min = 240; sizing = TABLE; cols = []
    stack = {"strategies": {"ev": {}}}

    def decide(self, X, cols, t):
        n = len(X)
        return {"score": np.full(n, 0.05), "threshold": np.full(n, 0.01), "allow": np.ones(n, bool), "reason": [""] * n, "hold_s": np.full(n, 14400.0),
                "strategy": ["ev"] * n, "tables": [TABLE] * n}


def _book(monkeypatch, cls=rc.RisingBook):
    from fly_trader.train import selector as sel
    monkeypatch.setattr(selector_session, "pinned_snapshot", lambda: None); monkeypatch.setattr(sel, "load_latest", lambda: _Model())
    monkeypatch.setattr(sel, "latest_current", lambda conn: {"id": None})
    return cls()


def test_the_book_doubles_only_what_qualifies_and_records_why(test_row, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0); b = _book(monkeypatch)
    ctx_infos = [_info("RCCLEANpump", 1.0, 0.01), _info("RCDIRTYpump", 1.0, 0.5), _info("RCNONEpump", 1.0, None)]
    with transaction() as conn:
        ctx = _ctx(conn, M0, {"RCCLEANpump": 1.0, "RCDIRTYpump": 1.0, "RCNONEpump": 1.0}, 0.05); ctx.infos = ctx_infos
        out = b.on_minute(ctx); pos = {p["mint"]: float(p["cost_sol"]) for p in ledger.open_positions(conn, rc.BOOK)}
        d = {r["mint"]: r["detail"] for r in conn.execute("SELECT mint, detail FROM decisions WHERE kind = 'rising_enter'").fetchall()}
    assert out["entered"] == 3 and pos["RCCLEANpump"] == pytest.approx(2 * pos["RCDIRTYpump"]) and pos["RCDIRTYpump"] == pytest.approx(pos["RCNONEpump"])
    assert d["RCCLEANpump"]["stake_x"] == 2.0 and d["RCCLEANpump"]["breadth_chg"] == 0.05 and d["RCCLEANpump"]["grad_hhi"] == 0.01 and "× 2" in d["RCCLEANpump"]["sizing"]
    assert d["RCDIRTYpump"]["stake_x"] == 1.0 and d["RCNONEpump"]["grad_hhi"] is None


def test_in_a_flat_market_the_book_is_the_selectors_to_the_lamport(test_row, monkeypatch):
    """The same minute on both books, nothing qualifying: the same stakes, as sizing, cash and costs are shared."""
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0); b = _book(monkeypatch); plain = "replay_rising_plain"; run = str(uuid.uuid4())
    infos = [_info("RCAAApump", 1.0, 0.01), _info("RCBBBpump", 2.0, 0.5)]; prices = {"RCAAApump": 1.0, "RCBBBpump": 2.0}
    try:
        with transaction() as conn:
            ctx = _ctx(conn, M0, prices, 0.0); ctx.infos = infos
            b.on_minute(ctx)
            paper_trading.trade_minute(ctx, book=plain, run_id=run, beat_no=1, broker=PaperBroker(plain), kind="selector", mints=list(prices), infos=infos, scores=np.full(2, 0.05),
                                       threshold=0.01, table=TABLE, horizon_s=14400.0, holds=np.full(2, 14400.0), strategies=["ev", "ev"], tables=[TABLE, TABLE])
            got = {b_: {p["mint"]: (float(p["cost_sol"]), int(p["qty"])) for p in ledger.open_positions(conn, b_)} for b_ in (rc.BOOK, plain)}
        assert got[rc.BOOK] == got[plain]
    finally:
        with transaction() as conn:
            for t in ("positions", "wealth_marks", "fills"):
                conn.execute(f"DELETE FROM {t} WHERE book = %s", (plain,))
            conn.execute("DELETE FROM decisions WHERE detail->>'book' = %s", (plain,)); conn.execute("DELETE FROM beats WHERE run_id = %s", (run,))


def test_a_handover_does_not_stop_the_book(test_row, monkeypatch):
    from fly_trader.agent import handover
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0); b = _book(monkeypatch)
    with transaction() as conn:
        conn.execute("DELETE FROM ui_settings WHERE key = 'handover'"); handover.record(conn, {"why": "test"}, "sol")
    try:
        with transaction() as conn:
            ctx = _ctx(conn, M0, {"RCHOpump": 1.0}, 0.05); ctx.engine.has_book = lambda n: True; ctx.infos = [_info("RCHOpump", 1.0, 0.01)]
            assert b.on_minute(ctx)["entered"] == 1 and b.done is False
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM ui_settings WHERE key = 'handover'")


def test_the_engine_admits_the_book_only_while_a_test_exists(test_row, monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(fly_session, "try_start", lambda live, chain="sol", brain=None: (NS(name="fly", chain=chain), ""))
    monkeypatch.setattr(selector_session, "SelectorBook", lambda chain="sol": NS(name="selector", chain=chain))
    monkeypatch.setattr(rc, "RisingBook", lambda: NS(name=rc.KIND, chain="sol"))
    monkeypatch.setattr(config, "RH_ENABLED", False)
    assert rc.KIND not in [b.name for b in me.admit_books(me.MinuteEngine(), live=False)]
    with transaction() as conn:
        rc.start(conn)
    assert [b.name for b in me.admit_books(me.MinuteEngine(), live=False)][-1] == rc.KIND
    e = me.MinuteEngine([NS(name=rc.KIND, chain="sol")])
    assert rc.KIND not in [b.name for b in me.admit_books(e, live=False)]


def test_a_book_that_cannot_start_never_stops_the_runner(test_row, monkeypatch):
    from types import SimpleNamespace as NS
    monkeypatch.setattr(fly_session, "try_start", lambda live, chain="sol", brain=None: (NS(name="fly", chain=chain), ""))
    monkeypatch.setattr(selector_session, "SelectorBook", lambda chain="sol": NS(name="selector", chain=chain))
    monkeypatch.setattr(config, "RH_ENABLED", False)

    def boom():
        raise RuntimeError("no")
    monkeypatch.setattr(rc, "RisingBook", boom)
    with transaction() as conn:
        rc.start(conn)
    assert [b.name for b in me.admit_books(me.MinuteEngine(), live=False)] == ["fly", "selector"]


def test_every_restart_keeps_the_book_its_run_and_its_decisions(test_row, tmp_path, monkeypatch):
    from fly_trader.ops.reset import reset_training_state
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path / "arch"); run = str(uuid.uuid4())
    with transaction() as conn:
        conn.execute("INSERT INTO runs (run_id, kind) VALUES (%s, %s)", (run, rc.KIND))
        beat = conn.execute("INSERT INTO beats (run_id, beat_no) VALUES (%s, 1) RETURNING id", (run,)).fetchone()["id"]
        did = conn.execute("INSERT INTO decisions (run_id, beat_id, kind, mint, detail) VALUES (%s, %s, 'rising_enter', 'RCKEEPpump', %s) RETURNING id", (run, beat, json.dumps({"book": rc.BOOK}))).fetchone()["id"]
        conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, status, entry_decision_id) VALUES (%s, 'RCKEEPpump', 1, 0.1, 'open', %s)", (rc.BOOK, did))
        conn.execute("INSERT INTO wealth_marks (beat_id, book, wealth) VALUES (%s, %s, 5.0)", (beat, rc.BOOK))
    reset_training_state(reason="test", archive=False)
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM positions WHERE book = %s", (rc.BOOK,)).fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM wealth_marks WHERE book = %s", (rc.BOOK,)).fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM decisions WHERE id = %s", (did,)).fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM beats WHERE run_id = %s", (run,)).fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM runs WHERE run_id = %s", (run,)).fetchone()["n"] == 1
        conn.execute("DELETE FROM beats WHERE run_id = %s", (run,)); conn.execute("DELETE FROM runs WHERE run_id = %s", (run,))


def test_only_solana_has_the_book():
    assert rc.BOOK in markets.SOL.paper_books and rc.BOOK not in markets.RH.paper_books and rc.BOOK not in markets.SOL.books
    assert markets.for_book(rc.BOOK) is markets.SOL


# ---- the go-live
def _trade(conn, pnl, stake_x=2.0, closed=True, opened=None):
    did = conn.execute("INSERT INTO decisions (kind, mint, detail) VALUES ('rising_enter', 'RCTpump', %s) RETURNING id", (json.dumps({"book": rc.BOOK, "stake_x": stake_x}),)).fetchone()["id"]
    conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, status, realized_sol, opened_at, entry_decision_id) VALUES (%s, 'RCTpump', 1, 0.4, %s, %s, %s, %s)",
                 (rc.BOOK, "closed" if closed else "open", pnl if closed else None, opened or (START + timedelta(days=1)), did))


def test_nothing_is_judged_before_the_time(test_row):
    with transaction() as conn:
        s = rc.start(conn, START); _trade(conn, 0.5)
        assert s["go_live_at"] == (START + timedelta(days=14)).isoformat()
        assert rc.decide_go_live(conn, START + timedelta(days=13, hours=23)) is None and not rc.wallet_doubles(conn)


def test_a_profitable_test_goes_live_once(test_row):
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.5); _trade(conn, -0.1)
        at = START + timedelta(days=14)
        assert rc.decide_go_live(conn, at) == "live" and rc.wallet_doubles(conn)
        assert rc.state(conn)["live_since"] == at.isoformat()
        _trade(conn, -9.0)                                             # judged once: later losses do not take it back
        assert rc.decide_go_live(conn, at + timedelta(minutes=1)) is None and rc.wallet_doubles(conn)


@pytest.mark.parametrize("trades", [[], [-0.1], [0.1, -0.1]], ids=["no trades", "a loss", "flat"])
def test_a_test_that_did_not_make_money_stays_on_paper_and_clears_the_date(test_row, trades):
    with transaction() as conn:
        rc.start(conn, START)
        for x in trades:
            _trade(conn, x)
        assert rc.decide_go_live(conn, START + timedelta(days=14)) == "held" and not rc.wallet_doubles(conn)
        s = rc.state(conn); assert s["go_live_at"] is None and s["held"]["trades"] == len(trades)
        assert rc.decide_go_live(conn, START + timedelta(days=15)) is None             # no date: nothing to judge
        ev = conn.execute("SELECT level, message FROM events WHERE source = %s ORDER BY id DESC LIMIT 1", (rc.KIND,)).fetchone()
    assert ev["level"] == "warning" and "stays on paper" in ev["message"]


def test_only_closed_doubled_trades_of_this_test_count(test_row):
    with transaction() as conn:
        rc.start(conn, START)
        _trade(conn, 5.0, stake_x=1.0)                                 # a plain trade
        _trade(conn, 5.0, opened=START - timedelta(days=1))            # before the test
        _trade(conn, None, closed=False)                               # still open
        assert rc.doubled_result(conn, rc.state(conn)["started_at"]) == (0, 0.0)
        assert rc.decide_go_live(conn, START + timedelta(days=14)) == "held"


def test_moving_or_clearing_the_date_takes_the_wallet_back_first(test_row):
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.5); at = START + timedelta(days=14)
        rc.decide_go_live(conn, at); assert rc.wallet_doubles(conn)
        rc.set_go_live(conn, at + timedelta(days=7)); assert not rc.wallet_doubles(conn) and rc.state(conn)["go_live_at"] == (at + timedelta(days=7)).isoformat()
        assert rc.decide_go_live(conn, at + timedelta(days=7)) == "live" and rc.wallet_doubles(conn)
        rc.set_go_live(conn, None); assert not rc.wallet_doubles(conn) and rc.state(conn)["go_live_at"] is None
        assert rc.decide_go_live(conn, at + timedelta(days=30)) is None


def test_the_console_and_the_judgement_take_turns_on_the_row(test_row):
    """A date moved while the go-live is being judged is not overwritten: the second to come waits for the row and reads what the first wrote."""
    import threading
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.5)
    later = START + timedelta(days=30); done = []
    with transaction() as conn:
        s = rc.state(conn, lock=True)                                  # the book holds the row, as if judging

        def console():
            with transaction() as c2:
                rc.set_go_live(c2, later)
            done.append(True)
        t = threading.Thread(target=console); t.start(); t.join(0.3)
        assert not done                                                 # the console waits for the book's commit
        rc._write(conn, {**s, "live_since": (START + timedelta(days=14)).isoformat()})
    t.join(5.0)
    with transaction() as conn:
        v = rc.state(conn)
    assert done and v["go_live_at"] == later.isoformat() and v["live_since"] is None      # the operator's move stands, the wallet goes back


def test_a_date_moved_just_before_the_lock_is_seen_by_the_judgement(test_row, monkeypatch):
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.5)
        real = rc.state; calls = []

        def racing(c, lock=False):
            calls.append(lock)
            if lock:                                                    # between the first read and the lock the console moved the date
                _s = real(c); rc._write(c, {**_s, "go_live_at": (START + timedelta(days=30)).isoformat()})
            return real(c, lock)
        monkeypatch.setattr(rc, "state", racing)
        assert rc.decide_go_live(conn, START + timedelta(days=14)) is None and calls == [False, True]
        monkeypatch.setattr(rc, "state", real)
        assert not rc.wallet_doubles(conn)


def test_a_new_date_after_a_held_go_live_clears_the_hold(test_row):
    with transaction() as conn:
        rc.start(conn, START); rc.decide_go_live(conn, START + timedelta(days=14)); assert rc.state(conn)["held"]
        rc.set_go_live(conn, START + timedelta(days=21)); assert rc.state(conn)["held"] is None


def test_the_book_judges_the_go_live_only_on_a_trading_minute(test_row, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0); b = _book(monkeypatch)
    with transaction() as conn:
        rc.start(conn, M0 - timedelta(days=14)); _trade(conn, 0.5, opened=M0 - timedelta(days=2))
        ctx = _ctx(conn, M0, {}, 0.0); ctx.infos = []; ctx.trade = False
        b.on_minute(ctx); assert not rc.wallet_doubles(conn)            # catching up: an old minute
        ctx.trade = True
        b.on_minute(ctx); assert rc.wallet_doubles(conn)


# ---- the wallet
def _entry(mint, hhi, size=0.2, price=1.0, resq=1000.0):
    return {"mint": mint, "decision_id": None, "score": 0.03, "size_sol": size, "threshold": 0.01, "table": TABLE, "ref_price": 1.01, "info": {**_info(mint, price, hhi), "resq": resq}}


def test_only_qualifying_entries_double_and_their_price_limit_follows_the_size(db_conn):
    c, d = _entry("LIVE_RC1", 0.01), _entry("LIVE_RC2", 0.5); broker = PaperBroker("paper_fly")
    assert rc.for_wallet(_ctx(None, M0, {}, 0.05), [c, d], broker) == 1
    q1, q2 = (broker.quote_buy(size_sol=x, price=1.0, res_quote_sol=1000.0, mcap_sol=1e6, decimals=6, program_label=None) for x in (0.2, 0.4))
    assert c["stake_x"] == 2.0 and "stake_x" not in d and d["ref_price"] == 1.01
    assert c["ref_price"] == pytest.approx(0.4 / (q2.token_delta / 1e6)) and c["ref_price"] > 0.2 / (q1.token_delta / 1e6)    # twice the size pays more impact: the limit moves with it


def test_a_flat_market_or_a_size_paper_would_refuse_leaves_the_entry_as_it_was(db_conn):
    c = _entry("LIVE_RC1", 0.01); assert rc.for_wallet(_ctx(None, M0, {}, 0.0), [c], PaperBroker("paper_fly")) == 0 and "stake_x" not in c
    c = _entry("LIVE_RC1", 0.01); assert rc.for_wallet(_ctx(None, M0, {}, None), [c], PaperBroker("paper_fly")) == 0 and "stake_x" not in c
    c = _entry("LIVE_RC1", 0.01, price=0.0); assert rc.for_wallet(_ctx(None, M0, {}, 0.05), [c], PaperBroker("paper_fly")) == 0 and "stake_x" not in c and c["ref_price"] == 1.01


def test_the_caps_bound_the_doubled_size_and_its_price_limit(db_conn):
    broker = PaperBroker("paper_fly"); pool = 15.0; cap = 0.02 * pool
    c = _entry("LIVE_RC1", 0.01, size=0.2, resq=pool); assert rc.for_wallet(_ctx(None, M0, {}, 0.05), [c], broker) == 1
    q = broker.quote_buy(size_sol=cap, price=1.0, res_quote_sol=pool, mcap_sol=1e6, decimals=6, program_label=None)
    assert c["stake_x"] == 2.0 and c["ref_price"] == pytest.approx(cap / (q.token_delta / 1e6))                     # at what the wallet can buy, not twice the stake
    at_cap = _entry("LIVE_RC1", 0.01, size=cap, resq=pool); assert rc.for_wallet(_ctx(None, M0, {}, 0.05), [at_cap], broker) == 0 and "stake_x" not in at_cap   # no room: nothing doubled
    heavy = _entry("LIVE_RC1", 0.01); heavy["info"]["creator_heavy"] = True
    assert rc.for_wallet(_ctx(None, M0, {}, 0.05), [heavy], broker) == 0 and "stake_x" not in heavy                 # its own cap leaves no room either


@pytest.fixture()
def clean_live():
    def wipe():
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint LIKE 'LIVE_RC%%'"); conn.execute("DELETE FROM decisions WHERE mint LIKE 'LIVE_RC%%'")
            conn.execute("UPDATE circuit_state SET kill_switch = false, tripped = false, entries_paused = false, fail_count = 0 WHERE id = 1")
    wipe(); yield; wipe()


def test_the_wallet_sizes_a_doubled_entry_twice_and_records_it_a_plain_one_as_ever(db_conn, clean_live, monkeypatch):
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3); w = _Worker()
    mirror = fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    with transaction() as conn:
        d1 = conn.execute("INSERT INTO decisions (kind, mint, detail) VALUES ('fly_enter', 'LIVE_RC1', '{}') RETURNING id").fetchone()["id"]
        d2 = conn.execute("INSERT INTO decisions (kind, mint, detail) VALUES ('fly_enter', 'LIVE_RC2', '{}') RETURNING id").fetchone()["id"]
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        e1 = {**_entry("LIVE_RC1", 0.01), "decision_id": d1, "stake_x": 2.0}; e2 = {**_entry("LIVE_RC2", 0.5), "decision_id": d2}
        out = mirror.minute(SimpleNamespace(conn=conn, m1=M0, m1_epoch=M0.timestamp(), prices={}, resqs={}, fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: None, fee=lambda m: None, lag_s=0.0),
                            run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[e1, e2])
        det = {r["id"]: r for r in conn.execute("SELECT id, detail, book_targets FROM decisions WHERE id = ANY(%s)", ([d1, d2],)).fetchall()}
    buys = {r.mint: r.amount_in for r in w.submitted}
    assert out["entered"] == 2 and buys["LIVE_RC2"] > 0 and buys["LIVE_RC1"] == pytest.approx(2 * buys["LIVE_RC2"], abs=2)
    assert det[d1]["detail"]["live_stake_x"] == 2.0 and "live_stake_x" not in det[d2]["detail"] and "live" in det[d1]["book_targets"] and "live" in det[d2]["book_targets"]


def _fly(rising_live, doubled=None):
    return SimpleNamespace(rising_live=rising_live, name="fly", chain="sol", broker=PaperBroker("paper_fly"), _double=doubled)


def test_a_failure_of_the_stakes_leaves_the_minutes_orders_at_plain_stakes(db_conn, monkeypatch):
    def boom(ctx, entries, broker):
        ctx.conn.execute("SELECT no_such_column FROM ui_settings")     # a failed query: without a savepoint it would poison the minute's transaction
    monkeypatch.setattr(rc, "for_wallet", boom); monkeypatch.setattr(rc, "wallet_doubles", lambda conn: True)
    e = _entry("LIVE_RC1", 0.01)
    with transaction() as conn:
        failed = fly_session.FlyBook._double(_fly(True), SimpleNamespace(conn=conn, m1=M0), [e])
        assert conn.execute("SELECT 1 AS ok").fetchone()["ok"] == 1     # the transaction the orders run in is still good
    assert "stake_x" not in e and "UndefinedColumn" in failed


def test_a_date_cleared_after_the_fly_last_read_it_binds_the_next_buy(test_row):
    """The fly's memory still says doubling; the row says otherwise: the row wins, with no minute's lag."""
    e, broker = _entry("LIVE_RC1", 0.01), PaperBroker("paper_fly")
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.5); rc.decide_go_live(conn, START + timedelta(days=14)); assert rc.wallet_doubles(conn)
        rc.set_go_live(conn, None)
        ctx = _ctx(conn, M0, {}, 0.05); ctx.conn = conn
        assert fly_session.FlyBook._double(_fly(True), ctx, [e]) is None and "stake_x" not in e


def test_a_live_row_doubles_through_the_flys_real_path(test_row):
    e = _entry("LIVE_RC1", 0.01)
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.5); rc.decide_go_live(conn, START + timedelta(days=14))
        assert fly_session.FlyBook._double(_fly(True), _ctx(conn, M0, {}, 0.05), [e]) is None and e["stake_x"] == 2.0


def test_a_failure_is_recorded_only_after_the_minutes_orders(monkeypatch):
    calls = []
    monkeypatch.setattr(fly_session.paper_trading, "plan_minute", lambda ctx, **kw: {"beat_id": 7, "entries": ["e"], "rebuy": {}})
    monkeypatch.setattr(fly_session, "record_event", lambda *a, **k: calls.append("event"))
    d = {"allow": np.ones(1, bool), "score": np.ones(1), "threshold": np.zeros(1), "hold_s": np.ones(1), "strategy": ["ev"], "tables": [[]], "reason": [""]}
    me_ = SimpleNamespace(_commands=lambda conn, t: None, _fast=lambda: True, _decide=lambda c: {"decision": d, "picks": 1}, race_started_at=0.0, beat_no=0, BOOK="paper_fly", run_id="run", chain="sol",
                          broker=None, KIND="fly", names=["ev"], holds={"ev": 14400.0}, has_seat=False, trades_live=lambda conn: True, rising_live=True,
                          _double=lambda c, e: calls.append("double") or "boom", _go_live=lambda c, beat, e, rebuy: calls.append("orders") or {})
    fly_session.FlyBook.on_minute(me_, SimpleNamespace(conn=None, m1=M0, m1_epoch=M0.timestamp(), lag_s=0.0, trade=True, fresh=True, mints=["M"], infos=[{}]))
    assert calls == ["double", "orders", "event"]


def test_a_retried_entry_goes_at_its_plain_stake(db_conn, clean_live, monkeypatch):
    """A retry's price limit is paper's fill at the plain stake, and the date may have moved since: it never doubles."""
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3); w = _Worker()
    mirror = fly_live.LiveMirror(7200.0, broker=SimpleNamespace(rpc=_Rpc(5 * config.LAMPORTS_PER_SOL), pubkey="PK"), worker=w)
    sizes = {}
    with transaction() as conn:
        for mint, x in (("LIVE_RC1", {"stake_x": 2.0}), ("LIVE_RC2", {})):
            did = conn.execute("INSERT INTO decisions (kind, mint, detail) VALUES ('fly_enter', %s, '{}') RETURNING id", (mint,)).fetchone()["id"]
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status, entry_decision_id) VALUES ('paper_fly', %s, 'P', %s, 1000000, 0.2, 0.0000002, 'open', %s)", (mint, M0, did))
            mirror.failed[did] = {"e": {**_entry(mint, 0.01), "decision_id": did, **x}, "ts": M0, "tries": 1, "error": "no route"}
        beat = conn.execute("INSERT INTO beats (beat_no) VALUES (1) RETURNING id").fetchone()["id"]
        rebuy = {m: {"score": 0.03, "threshold": 0.01, "table": TABLE, "ref_price": 1.0} for m in ("LIVE_RC1", "LIVE_RC2")}
        out = mirror.minute(SimpleNamespace(conn=conn, m1=M0 + timedelta(minutes=1), m1_epoch=M0.timestamp() + 60, prices={}, resqs={}, fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: None, fee=lambda m: None, lag_s=0.0),
                            run_id=str(uuid.uuid4()), beat_id=beat, line=0.01, table=TABLE, entries=[], rebuy=rebuy)
        conn.execute("DELETE FROM positions WHERE book = 'paper_fly' AND mint LIKE 'LIVE_RC%%'")
    sizes = {r.mint: r.amount_in for r in w.submitted}
    assert out["retried"] == 2 and sizes["LIVE_RC1"] == sizes["LIVE_RC2"] > 0


@pytest.mark.parametrize("live,entries,doubled", [(False, ["e"], 0), (True, [], 0), (True, ["e"], 1)], ids=["before the go-live", "nothing to buy", "live"])
def test_the_fly_doubles_after_its_plan_and_before_the_wallets_orders(monkeypatch, live, entries, doubled):
    """Through the fly's real live minute: the stakes change between the paper plan and the wallet's orders, and only when live."""
    calls = []
    monkeypatch.setattr(fly_session.paper_trading, "plan_minute", lambda ctx, **kw: calls.append("plan") or {"beat_id": 7, "entries": entries, "rebuy": {}})
    d = {"allow": np.ones(1, bool), "score": np.ones(1), "threshold": np.zeros(1), "hold_s": np.ones(1), "strategy": ["ev"], "tables": [[]], "reason": [""]}
    me_ = SimpleNamespace(_commands=lambda conn, t: None, _fast=lambda: True, _decide=lambda c: {"decision": d, "picks": 1}, race_started_at=0.0, beat_no=0, BOOK="paper_fly", run_id="run",
                          broker=None, KIND="fly", names=["ev"], holds={"ev": 14400.0}, has_seat=False, trades_live=lambda conn: True, rising_live=live,
                          _double=lambda c, e: calls.append("double"), _go_live=lambda c, beat, e, rebuy: calls.append("orders") or {})
    ctx = SimpleNamespace(conn=None, m1=M0, m1_epoch=M0.timestamp(), lag_s=0.0, trade=True, fresh=True, mints=["M"], infos=[{}])
    fly_session.FlyBook.on_minute(me_, ctx)
    assert calls == (["plan", "double", "orders"] if doubled else ["plan", "orders"])


def test_before_a_go_live_no_query_stands_in_front_of_the_wallet(monkeypatch):
    """The fly's live part asks its memory first; only a fly that has gone live reads the row, in ``_double``, on a minute with a buy."""
    import inspect
    live = inspect.getsource(fly_session.FlyBook.on_minute); paper = inspect.getsource(fly_session.FlyBook.after_minute)
    assert "wallet_doubles" not in live and "self.rising_live and plan" in live and "wallet_doubles" in paper


# ---- the console
def test_the_console_runs_the_test_and_moves_the_go_live(console, test_row):
    d = console.topic("overview.rising", chain="sol").json()
    assert d["running"] is False and d["hidden"] is False and d["days"] == 14
    assert console.topic("overview.rising", chain="rh").json().get("hidden", True) is True or not config.RH_ENABLED
    assert console.act("overview.rising_go_live", {"at": None}).status_code == 409             # no test yet
    assert console.act("overview.rising_start").status_code == 200
    assert console.act("overview.rising_start").status_code == 409                             # one test
    d = console.topic("overview.rising", chain="sol").json(); assert d["running"] is True and d["live_since"] is None and d["go_live_at"]
    assert (d["book"]["closed"], d["doubled"]["closed"], d["rest"]["closed"]) == (0, 0, 0)
    soon = datetime.now(timezone.utc) + timedelta(days=3)
    assert console.act("overview.rising_go_live", {"at": soon.isoformat()}).status_code == 200
    assert console.topic("overview.rising", chain="sol").json()["go_live_at"] == soon.isoformat()
    assert console.act("overview.rising_go_live", {"at": None}).status_code == 200
    assert console.topic("overview.rising", chain="sol").json()["go_live_at"] is None


@pytest.mark.parametrize("at,code", [("not a time", 400), (5, 400), ("2030-01-01T00:00:00", 400), ("2020-01-01T00:00:00+00:00", 409)])
def test_the_console_refuses_a_bad_time(console, test_row, at, code):
    console.act("overview.rising_start")
    assert console.act("overview.rising_go_live", {"at": at}).status_code == code


def test_the_console_sets_results_apart_by_stake(console, test_row):
    since = datetime.now(timezone.utc) - timedelta(days=1)
    with transaction() as conn:
        rc.start(conn, since); _trade(conn, 0.4, opened=since + timedelta(hours=1)); _trade(conn, -0.1, stake_x=1.0, opened=since + timedelta(hours=2))
    d = console.topic("overview.rising", chain="sol").json()
    assert d["doubled"]["closed"] == 1 and d["doubled"]["pnl"] == pytest.approx(0.4) and d["doubled"]["mean"] == pytest.approx(1.0)
    assert d["rest"]["closed"] == 1 and d["rest"]["pnl"] == pytest.approx(-0.1) and d["book"]["closed"] == 2 and d["book"]["pnl"] == pytest.approx(0.3)


def test_a_browser_time_ending_in_z_is_taken(console, test_row):
    console.act("overview.rising_start")
    at = (datetime.now(timezone.utc) + timedelta(days=2)).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    assert console.act("overview.rising_go_live", {"at": at}).status_code == 200


def test_a_time_just_past_is_now_for_someone_typing_it(console, test_row):
    console.act("overview.rising_start")
    assert console.act("overview.rising_go_live", {"at": (datetime.now(timezone.utc) - timedelta(seconds=30)).isoformat()}).status_code == 200


def test_an_observer_console_cannot_change_the_test(console, test_row, monkeypatch):
    from fly_trader.ui import runtime
    monkeypatch.setattr(runtime, "sup", lambda: runtime.Observer())
    assert console.act("overview.rising_start").status_code == 409
    with transaction() as conn:
        assert rc.state(conn) is None
    assert console.topic("overview.rising", chain="sol").json()["observer"] is True


def test_the_trades_page_opens_the_book_and_names_its_decisions(console, test_row):
    from fly_trader.ui.api import trades
    with transaction() as conn:
        rc.start(conn, START); _trade(conn, 0.4)
    assert rc.BOOK in console.topic("trades.books", chain="sol").json()["books"]
    assert console.topic("trades.closed", chain="sol", book=rc.BOOK).json()["book"] == rc.BOOK
    assert trades.ACTION["rising_enter"] == "Bought" and trades.ACTION["rising_exit"] == "Sold"


def test_a_halted_book_is_on_the_solana_circuit(db_conn):
    from fly_trader.agent import rails
    from fly_trader.ui.api import safety
    assert safety._in_circuit(rc.BOOK, rails.SOLANA_CIRCUIT) and not safety._in_circuit(rc.BOOK, rails.RH_CIRCUIT)
