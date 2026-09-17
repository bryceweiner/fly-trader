"""Per-strategy holds in the book: a 10-minute and a 240-minute position exit at their own times; a row a filter
blocks is recorded with the filter's reason; a call without the new arguments behaves as before."""
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np

from fly_trader import config
from fly_trader.agent import paper_trading
from fly_trader.db.connection import transaction
from fly_trader.execution import ledger
from fly_trader.execution.broker_paper import PaperBroker

BOOK = "replay_hold_test"


def _ctx(conn, m1, mints, price=1.0):
    return SimpleNamespace(conn=conn, m1=m1, m1_epoch=m1.timestamp(), agg={m: {} for m in mints}, prices={m: price for m in mints}, resqs={m: 1000.0 for m in mints},
                           fees={}, last_resq=lambda m: 1000.0, mcap=lambda m, p: 1e6)


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
