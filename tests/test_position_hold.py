"""Per-strategy holds in the book: a 10-minute and a 240-minute position exit at their own times; a row a filter
blocks is recorded with the filter's reason; a call without the new arguments behaves as before."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import paper_trading
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker

BOOK = "replay_hold_test"


def _ctx(conn, m1, mints, price=1.0, lag_s=0.0):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), agg={m: {} for m in mints}, prices={m: price for m in mints}, resqs={m: 1000.0 for m in mints},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6, fee=lambda m: None, cost_label=lambda m, label: label, lag_s=lag_s)


def _info(m):
    return {"resq": 1000.0, "pool": "P" + m, "age_h": 10.0, "price": 1.0, "mcap": 1e6, "decimals": 6, "program_label": None, "fee_rate": None}


def test_per_position_holds_and_filter_reasons(db_conn, monkeypatch):
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0)
    table = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.7, "kelly": 0.4}]
    run = str(uuid.uuid4()); broker = PaperBroker(BOOK); m0 = datetime(2026, 9, 10, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        conn.execute("DELETE FROM positions WHERE book = %s", (BOOK,)); conn.execute("DELETE FROM wealth_marks WHERE book = %s", (BOOK,))
        conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false WHERE id = 1")
        mints = ["SHORTpump", "LONGpump", "BLOCKpump"]
        st = paper_trading.trade_minute(_ctx(conn, m0, mints), book=BOOK, run_id=run, beat_no=1, broker=broker, kind="selector", mints=mints,
                                        infos=[_info(m) for m in mints], scores=np.array([0.05, 0.05, 0.05]), threshold=np.array([0.01, 0.01, 0.01]), table=None,
                                        horizon_s=7200.0, holds=np.array([600.0, 14400.0, 600.0]), strategies=np.array(["capitulation", "breakout", "ev"], dtype=object),
                                        allow=np.array([True, True, False]), reasons=np.array(["trade", "trade", "dump veto"], dtype=object), tables=[table] * 3)
        assert st["entered"] == 2
        blk = conn.execute("SELECT rail FROM decisions WHERE run_id = %s AND kind = 'blocked' AND mint = 'BLOCKpump'", (run,)).fetchone()
        assert blk["rail"] == "dump veto"
        holds = {p["mint"]: (p["hold_s"], p["strategy"]) for p in ledger.open_positions(conn, BOOK)}
        assert holds == {"SHORTpump": (600.0, "capitulation"), "LONGpump": (14400.0, "breakout")}
        for minutes, left in ((11, {"LONGpump"}), (241, set())):
            m1 = m0 + timedelta(minutes=minutes)
            paper_trading.trade_minute(_ctx(conn, m1, mints), book=BOOK, run_id=run, beat_no=2, broker=broker, kind="selector", mints=[], infos=[], scores=np.array([]),
                                       threshold=0.01, table=None, horizon_s=7200.0)
            assert {p["mint"] for p in ledger.open_positions(conn, BOOK)} == left
        # the scalar call of the single-strategy engine: the book's horizon applies
        paper_trading.trade_minute(_ctx(conn, m0 + timedelta(minutes=300), ["OLDpump"]), book=BOOK, run_id=run, beat_no=3, broker=broker, kind="selector",
                                   mints=["OLDpump"], infos=[_info("OLDpump")], scores=np.array([0.05]), threshold=0.01, table=table, horizon_s=7200.0)
        p = ledger.open_positions(conn, BOOK)[0]
        assert p["mint"] == "OLDpump" and p["hold_s"] is None and p["strategy"] is None
        conn.execute("DELETE FROM positions WHERE book = %s", (BOOK,)); conn.execute("DELETE FROM wealth_marks WHERE book = %s", (BOOK,))
        conn.execute("DELETE FROM decisions WHERE run_id = %s", (run,)); conn.execute("DELETE FROM fills WHERE book = %s", (BOOK,))


def test_a_minute_traded_long_after_its_end_takes_no_entries_but_its_exits_run(db_conn, monkeypatch):
    """A lagging stream hands the books minutes long gone: a buy at a minutes-old price would be filled live at a newer one
    (+1.44 % adverse at 240 s), so neither book enters on it; exits still run. A fresh minute's entry hands the live mirror
    its fill's all-in price."""
    monkeypatch.setattr(config, "CAPITAL_SOL", 5.0); monkeypatch.setattr(config, "ENTRY_MAX_LAG_S", 90.0)
    table = [{"lo": 0.0, "n": 100, "mean": 0.02, "win": 0.7, "kelly": 0.4}]
    run = str(uuid.uuid4()); broker = PaperBroker(BOOK); m0 = datetime(2026, 9, 11, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        conn.execute("DELETE FROM positions WHERE book = %s", (BOOK,)); conn.execute("DELETE FROM wealth_marks WHERE book = %s", (BOOK,))
        conn.execute("UPDATE circuit_state SET kill_switch = false, entries_paused = false WHERE id = 1")
        st = paper_trading.trade_minute(_ctx(conn, m0, ["DUEpump"]), book=BOOK, run_id=run, beat_no=1, broker=broker, kind="selector", mints=["DUEpump"],
                                        infos=[_info("DUEpump")], scores=np.array([0.05]), threshold=0.01, table=table, horizon_s=600.0)
        e = st["entries"][0]; p = ledger.open_positions(conn, BOOK)[0]
        assert st["entered"] == 1 and e["ref_price"] == pytest.approx(float(p["cost_sol"]) / (int(p["qty"]) / 1e6)) and e["ref_price"] > 1.0
        st = paper_trading.trade_minute(_ctx(conn, m0 + timedelta(minutes=11), ["DUEpump", "NEWpump"], lag_s=120.0), book=BOOK, run_id=run, beat_no=2, broker=broker,
                                        kind="selector", mints=["NEWpump"], infos=[_info("NEWpump")], scores=np.array([0.05]), threshold=0.01, table=table, horizon_s=600.0)
        rail = conn.execute("SELECT rail FROM decisions WHERE run_id = %s AND kind = 'blocked' AND mint = 'NEWpump'", (run,)).fetchone()["rail"]
        assert st["exited"] == 1 and st["entered"] == 0 and st["blocked"] == "stale minute" and rail == "stale minute" and ledger.open_positions(conn, BOOK) == []
        conn.execute("DELETE FROM positions WHERE book = %s", (BOOK,)); conn.execute("DELETE FROM wealth_marks WHERE book = %s", (BOOK,))
        conn.execute("DELETE FROM decisions WHERE run_id = %s", (run,)); conn.execute("DELETE FROM fills WHERE book = %s", (BOOK,))
