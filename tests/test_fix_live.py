"""Review fixes: partial-sell realized P&L, training reset leaves the circuit armed, close guard + runner lock,
confirmation polling past lastValidBlockHeight."""
import uuid

import pytest

from fly_trader import config
from fly_trader.agent import runner
from fly_trader.db.connection import transaction
from fly_trader.execution import broker_live, ledger
from fly_trader.execution.broker_live import FillResult
from fly_trader.execution.broker_paper import PaperBroker
from fly_trader.execution.worker import ExecRequest, ExecutionWorker


def _pos(conn, pid):
    return conn.execute("SELECT * FROM positions WHERE id=%s", (pid,)).fetchone()


# ---- 1. partial sells accumulate realized P&L; paper cash stays consistent ----
def test_paper_partial_then_close_accumulates_realized(db_conn):
    book, mint = f"replay_t{uuid.uuid4().hex[:8]}", f"M{uuid.uuid4().hex[:10]}"
    pid = ledger.open_position(db_conn, book=book, mint=mint, pool=None, qty_raw=1_000_000, cost_sol=1.0, entry_price=1.0,
                               decision_id=None, fees_sol=0.0)
    assert ledger.paper_cash(db_conn, book) == pytest.approx(config.CAPITAL_SOL - 1.0)
    pb = PaperBroker(book)
    kw = dict(decision_id=None, res_quote_sol=100.0, mcap_sol=100.0, program_label=None, forced_kind=None)
    f1 = pb.sell(db_conn, position=ledger.open_positions(db_conn, book)[0], price=1.2, fraction=0.5, **kw)
    assert f1.ok and 0 < f1.sol_delta < 0.6
    p = _pos(db_conn, pid)
    assert p["status"] == "open" and int(p["qty"]) == 500_000 and p["cost_sol"] == pytest.approx(0.5)
    assert p["realized_sol"] == pytest.approx(f1.sol_delta - 0.5)
    assert ledger.paper_cash(db_conn, book) == pytest.approx(config.CAPITAL_SOL - 1.0 + f1.sol_delta)
    stale = ledger.open_positions(db_conn, book)[0]
    f2 = pb.sell(db_conn, position=stale, price=0.8, **kw)
    assert f2.ok and "realized" in f2.reason
    p = _pos(db_conn, pid)
    assert p["status"] == "closed" and p["realized_sol"] == pytest.approx(f1.sol_delta + f2.sol_delta - 1.0)
    assert ledger.paper_cash(db_conn, book) == pytest.approx(config.CAPITAL_SOL - 1.0 + f1.sol_delta + f2.sol_delta)
    # 4. the close guard: a second close of the same position books nothing and records no fill
    f3 = pb.sell(db_conn, position=stale, price=0.8, **kw)
    assert not f3.ok and f3.fill_id is None
    assert ledger.close_position(db_conn, position_id=pid, exit_price=1.0, proceeds_sol=5.0, fees_sol=0.0,
                                 decision_id=None, forced_kind=None) is None
    assert _pos(db_conn, pid)["realized_sol"] == pytest.approx(f1.sol_delta + f2.sol_delta - 1.0)
    n = db_conn.execute("SELECT count(*) AS n FROM fills WHERE book=%s AND mint=%s AND side='sell'", (book, mint)).fetchone()["n"]
    assert n == 2


class _FakeBroker:
    def __init__(self):
        self.next: FillResult | None = None

    def swap(self, conn, **kw):
        return self.next


def test_worker_partial_then_close_books_each_sale_once():
    mint = f"M{uuid.uuid4().hex[:10]}"
    with transaction() as conn:
        circ = dict(conn.execute("SELECT fail_count, tripped FROM circuit_state WHERE id=1").fetchone())
        pid = ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000_000, cost_sol=1.0, entry_price=1.0,
                                   decision_id=None, fees_sol=0.0)
    fb = _FakeBroker()
    w = ExecutionWorker(fb)
    try:
        req = ExecRequest(decision_id=None, mint=mint, pool=None, side="sell", amount_in=400_000, slippage_bps=100,
                          max_slippage_bps=100, decimals=6, position_id=pid)
        fb.next = FillResult(True, None, None, "s1", "confirmed", 0, None, -400_000, 500_000_000, 1.25, 1)
        r1 = w._execute(req)
        assert r1.ok and r1.realized_sol == pytest.approx(0.1)
        fb.next = FillResult(True, None, None, "s2", "confirmed", 0, None, -600_000, 900_000_000, 1.5, 1)
        r2 = w._execute(req)
        assert r2.ok and r2.realized_sol == pytest.approx(0.3)
        fb.next = FillResult(True, None, None, "s3", "confirmed", 0, None, -1, 1, 1.0, 1)
        r3 = w._execute(req)
        assert r3.ok and r3.realized_sol is None
        with transaction() as conn:
            p = _pos(conn, pid)
        assert p["status"] == "closed" and p["realized_sol"] == pytest.approx(0.4) and p["cost_sol"] == pytest.approx(0.6)
    finally:
        w.stop()
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE id=%s", (pid,))
            conn.execute("UPDATE circuit_state SET fail_count=%s, tripped=%s WHERE id=1", (circ["fail_count"], circ["tripped"]))


# ---- 2. a training reset never re-arms the circuit ----
def test_training_reset_leaves_circuit_alone(tmp_path, monkeypatch):
    from fly_trader.ops.reset import reset_training_state
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(config, "BRAIN_DIR", tmp_path / "brain")
    cols = "fail_count, tripped, kill_switch, kill_reason, peak_wealth"
    with transaction() as conn:
        before = dict(conn.execute(f"SELECT {cols} FROM circuit_state WHERE id=1").fetchone())
        conn.execute("UPDATE circuit_state SET fail_count=3, tripped=true, kill_switch=true, kill_reason='dd', peak_wealth=5.0 WHERE id=1")
    try:
        reset_training_state(reason="test", archive=False)
        with transaction() as conn:
            after = dict(conn.execute(f"SELECT {cols} FROM circuit_state WHERE id=1").fetchone())
        assert after == {"fail_count": 3, "tripped": True, "kill_switch": True, "kill_reason": "dd", "peak_wealth": 5.0}
    finally:
        with transaction() as conn:
            conn.execute("UPDATE circuit_state SET fail_count=%(fail_count)s, tripped=%(tripped)s, kill_switch=%(kill_switch)s, "
                         "kill_reason=%(kill_reason)s, peak_wealth=%(peak_wealth)s WHERE id=1", before)


# ---- 4. one runner per database ----
def test_runner_lock_refuses_a_second_runner(monkeypatch):
    monkeypatch.setattr(runner, "setup", lambda name: None)
    ran = []
    monkeypatch.setattr(runner, "_main", lambda stop_event=None: ran.append(1))
    held = runner.acquire_runner_lock()
    try:
        with pytest.raises(RuntimeError, match="runner lock"):
            runner.acquire_runner_lock()
        with pytest.raises(RuntimeError, match="runner lock"):
            runner.main()
        assert ran == []
    finally:
        held.close()
    runner.main()                       # lock free again; main releases it on return
    assert ran == [1]
    runner.acquire_runner_lock().close()


# ---- 5. a processed signature is polled past lastValidBlockHeight ----
class _StatusRpc:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def get_signature_statuses(self, sigs):
        return [self.statuses.pop(0)]

    def get_block_height(self):
        return 10_000


def test_await_confirmation_keeps_polling_a_processed_signature(monkeypatch):
    monkeypatch.setattr(broker_live.time, "sleep", lambda s: None)
    proc = {"err": None, "confirmationStatus": "processed"}
    rpc = _StatusRpc([proc, proc, {"err": None, "confirmationStatus": "confirmed"}])
    assert broker_live.await_confirmation(rpc, "sig", 100)[0] == "confirmed"
    rpc = _StatusRpc([proc, None])
    assert broker_live.await_confirmation(rpc, "sig", 100)[0] == "expired"
