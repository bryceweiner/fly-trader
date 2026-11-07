"""The crash exit (agent/crash_exit.py, Bryce 2026-11-07): a held Solana coin whose price falls 25 % under its high of the
last 5 minutes is sold at once from the stream's prices, the wallet's position by the live mirror, paper's at the crash's
price in the minute's bookkeeping; a slower fall, prices before the entry and coins nobody holds do not count."""
import json
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import crash_exit, fly_live, paper_trading
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker

T0 = 1_800_000_000.0


class _Pool:
    def __init__(self):
        self.calls = []

    def submit(self, fn, *a):
        self.calls.append((fn.__name__, a))


@pytest.fixture()
def watch(monkeypatch):
    w = crash_exit.Watch(); pool = _Pool()
    monkeypatch.setattr(crash_exit, "_POOL", pool)
    w.held = {"COIN": (T0, 1.0)}; w.refreshed = T0 + 10 ** 6                 # no re-read in these tests
    return w, pool


def _sales(pool):
    return [a for name, a in pool.calls if name == "_sell"]


def test_a_25_percent_fall_within_5_minutes_sells_at_once(watch):
    w, pool = watch
    for dt, px in ((30, 1.1), (60, 1.2), (200, 1.0)):
        w.observe("COIN", T0 + dt, px, now=T0 + dt)
    assert not _sales(pool)
    w.observe("COIN", T0 + 230, 0.89, now=T0 + 230)                        # 26 % under 1.2, 170 s after it
    assert _sales(pool) == [("COIN", 0.89, 1.2, True)] and w.crashed("COIN", datetime.fromtimestamp(T0, timezone.utc)) == (T0 + 230, 0.89)


def test_a_slower_fall_prices_before_the_entry_and_coins_not_held_do_not_count(watch):
    w, pool = watch
    w.observe("COIN", T0 - 60, 5.0, now=T0)                                 # before the entry
    for k in range(1, 15):
        w.observe("COIN", T0 + 60 * k, 0.97 ** k, now=T0 + 60 * k)          # -3 % a minute: -14 % over any 5 minutes, -35 % in all
    w.observe("OTHER", T0 + 10, 1.0, now=T0 + 10); w.observe("OTHER", T0 + 20, 0.1, now=T0 + 20)
    assert not _sales(pool) and not w.crashes


def test_the_entry_price_counts_inside_the_window(watch):
    w, pool = watch
    w.observe("COIN", T0 + 10, 0.74, now=T0 + 10)                           # the first price seen, 26 % under the entry
    assert _sales(pool) == [("COIN", 0.74, 1.0, True)]
    w2 = crash_exit.Watch(); w2.held = {"COIN": (T0, 1.0)}; w2.refreshed = T0 + 10 ** 6
    w2.observe("COIN", T0 + 400, 0.74, now=T0 + 400)                        # the entry is out of the window by then
    assert len(_sales(pool)) == 1


def test_paper_takes_the_first_price_5_seconds_after_the_crash(watch):
    w, pool = watch
    since = datetime.fromtimestamp(T0, timezone.utc)
    w.observe("COIN", T0 + 10, 1.2, now=T0 + 10); w.observe("COIN", T0 + 20, 0.85, now=T0 + 20)
    assert w.crashed("COIN", since) == (T0 + 20, 0.85)                      # no later price yet: the crash's own
    w.observe("COIN", T0 + 23, 0.8, now=T0 + 23)
    assert w.crashed("COIN", since) == (T0 + 20, 0.85)
    w.observe("COIN", T0 + 26, 0.7, now=T0 + 26); w.observe("COIN", T0 + 30, 0.6, now=T0 + 30)
    assert w.crashed("COIN", since) == (T0 + 20, 0.7)


def test_the_live_sale_is_tried_again_while_the_crash_stands(watch):
    w, pool = watch
    w.observe("COIN", T0 + 10, 1.2, now=T0 + 10); w.observe("COIN", T0 + 20, 0.8, now=T0 + 20)
    w.observe("COIN", T0 + 25, 0.85, now=T0 + 25)                           # 5 s later: not yet
    w.observe("COIN", T0 + 40, 0.95, now=T0 + 40)                           # 20 s later: again, the crash's own price
    assert _sales(pool) == [("COIN", 0.8, 1.2, True), ("COIN", 0.8, 1.2, False)]


def test_the_refresh_watches_every_open_solana_position(db_conn, monkeypatch):
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM positions WHERE mint LIKE 'CRASH%%'")
        cur.execute("INSERT INTO positions (book, mint, opened_at, qty, cost_sol, entry_price, status, chain) VALUES "
                    "('paper_fly', 'CRASH1', now() - interval '10 minutes', 1, 0.1, 2.0, 'open', 'sol'), "
                    "('live', 'CRASH1', now() - interval '9 minutes', 1, 0.1, 2.1, 'open', 'sol'), "
                    "('paper_rh_fly', 'CRASH2', now(), 1, 0.1, 1.0, 'open', 'rh'), ('paper_fly', 'CRASH3', now(), 1, 0.1, 1.0, 'closed', 'sol')")
    db_conn.commit()
    try:
        w = crash_exit.Watch(); w._refresh()
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM positions WHERE mint LIKE 'CRASH%%'")
        db_conn.commit()
    assert "CRASH1" in w.held and w.held["CRASH1"][1] == 2.0 and "CRASH2" not in w.held and "CRASH3" not in w.held


class _Worker:
    def __init__(self):
        self.submitted, self.inflight = [], {}

    def submit(self, req):
        if self.inflight.get(req.mint):
            return False
        self.submitted.append(req); self.inflight.setdefault(req.mint, []).append(req.side); return True

    def pending(self):
        return set(self.inflight)


def test_the_live_mirror_sells_a_crashed_coin_at_the_forced_slippage(db_conn):
    w = _Worker(); mirror = fly_live.LiveMirror(14400.0, broker=SimpleNamespace(rpc=None, pubkey="PK"), worker=w)
    with transaction() as conn:
        conn.execute("DELETE FROM positions WHERE mint LIKE 'LIVE_CR%%'"); conn.execute("DELETE FROM decisions WHERE mint LIKE 'LIVE_CR%%'")
        for m in ("LIVE_CR1", "LIVE_CR2"):
            conn.execute("INSERT INTO positions (book, mint, pool, opened_at, qty, cost_sol, entry_price, status) VALUES ('live', %s, 'P', now(), 1000, 0.2, 0.0002, 'open')", (m,))
        w.inflight["LIVE_CR2"] = ["sell"]
        n1 = mirror.crash_exit(conn, "LIVE_CR1", "crash: 30% under its 5-minute high")
        n2 = mirror.crash_exit(conn, "LIVE_CR2", "crash")
        n3 = mirror.crash_exit(conn, "LIVE_CR9", "crash")
        d = conn.execute("SELECT reason, forced FROM decisions WHERE mint = 'LIVE_CR1' AND kind = 'fly_exit'").fetchone()
        conn.execute("DELETE FROM positions WHERE mint LIKE 'LIVE_CR%%'"); conn.execute("DELETE FROM decisions WHERE mint LIKE 'LIVE_CR%%'")
    req = w.submitted[-1]
    assert (n1, n2, n3) == (1, 0, 0) and (req.mint, req.side, req.forced_kind, req.slippage_bps, req.amount_in) == ("LIVE_CR1", "sell", "crash", config.SLIPPAGE_FORCED_BPS, 1000)
    assert (d["reason"], d["forced"]) == ("crash: 30% under its 5-minute high", True)


BOOK = "replay_crash_test"
TABLE = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.7, "kelly": 0.4}]
M0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)


def _ctx(conn, m1, prices):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), agg={m: {} for m in prices}, prices=dict(prices), resqs={m: 1000.0 for m in prices},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6, fee=lambda m: None, cost_label=lambda m, label: label, lag_s=0.0,
                           high=lambda m: None)


def _trade(conn, broker, run, beat, m1, prices, buy=()):
    buy = list(buy)
    return paper_trading.trade_minute(_ctx(conn, m1, prices), book=BOOK, run_id=run, beat_no=beat, broker=broker, kind="selector", mints=buy,
                                      infos=[{"resq": 1000.0, "pool": "P" + m, "age_h": 10.0, "price": prices[m], "mcap": 1e6, "decimals": 6, "program_label": None,
                                              "fee_rate": None} for m in buy], scores=np.full(len(buy), 0.05), threshold=0.01, table=TABLE, horizon_s=14400.0)


def test_paper_sells_a_crashed_coin_at_the_crashs_price_and_may_buy_it_again(db_conn, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0)
    w = crash_exit.Watch(); monkeypatch.setattr(crash_exit, "WATCH", w)

    def wipe():
        with transaction() as conn:
            for t in ("positions", "wealth_marks", "fills"):
                conn.execute(f"DELETE FROM {t} WHERE book = %s", (BOOK,))
            conn.execute("DELETE FROM decisions WHERE detail->>'book' = %s", (BOOK,))
    wipe(); broker = PaperBroker(BOOK); run = str(uuid.uuid4())
    try:
        with transaction() as conn:
            _trade(conn, broker, run, 1, M0, {"CRASHpump": 1.0, "CALMpump": 1.0}, buy=["CRASHpump", "CALMpump"])
            entry = float(ledger.open_positions(conn, BOOK)[0]["entry_price"])
            w.crashes["CRASHpump"] = [(M0 + timedelta(minutes=3, seconds=20)).timestamp(), 0.75 * entry, 1.0 * entry, 0.7 * entry]   # its fill 5 s on
            w.crashes["CALMpump"] = [(M0 - timedelta(minutes=30)).timestamp(), 0.5, 1.0, None]     # before its entry: not its crash
            st = _trade(conn, broker, run, 2, M0 + timedelta(minutes=4), {"CRASHpump": 0.4 * entry, "CALMpump": 1.0})
            sold = conn.execute("SELECT forced_exit_kind, exit_price FROM positions WHERE book = %s AND mint = 'CRASHpump'", (BOOK,)).fetchone()
            why = [r["reason"] for r in conn.execute("SELECT reason FROM decisions WHERE run_id = %s AND kind = 'selector_exit'", (run,)).fetchall()]
            again = _trade(conn, broker, run, 3, M0 + timedelta(minutes=5), {"CRASHpump": 0.5, "CALMpump": 1.0}, buy=["CRASHpump"])
            left = {p["mint"] for p in ledger.open_positions(conn, BOOK)}
    finally:
        wipe()
    assert st["exited"] == 1 and sold["forced_exit_kind"] == "crash" and sold["exit_price"] == pytest.approx(0.7 * entry, rel=0.05)
    assert why == ["crash: sold at -30% from its entry, 25% under its 5-minute high"]
    assert again["entered"] == 1 and left == {"CRASHpump", "CALMpump"}
