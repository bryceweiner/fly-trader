"""Review fixes: partial-sell realized P&L, training reset leaves the circuit armed, close guard + runner lock,
confirmation polling past lastValidBlockHeight."""
import threading
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
        self.kw: dict = {}

    def swap(self, conn, **kw):
        self.kw = kw
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



# ---- 6. live keeps up with paper: retryable failures, sweeps, a buy behind a sale, rent back, a clean stop ----
class _ScriptBroker:
    """Fills from a script, in order; ``gate`` holds each swap until set (an order in flight); every call recorded."""

    def __init__(self, fills, gate=None, refund=None):
        self.fills, self.gate, self.refund = list(fills), gate, refund
        self.calls, self.closed, self.started = [], [], threading.Event()

    def swap(self, conn, **kw):
        self.started.set(); self.calls.append(kw)
        if self.gate is not None:
            self.gate.wait(10)
        return self.fills.pop(0)

    def close_token_account(self, mint):
        self.closed.append(mint)
        return self.refund


@pytest.fixture
def circuit1():
    with transaction() as conn:
        before = dict(conn.execute("SELECT fail_count, tripped, last_failure_ts FROM circuit_state WHERE id=1").fetchone())
        conn.execute("UPDATE circuit_state SET fail_count = 0, tripped = false WHERE id = 1")
    yield
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET fail_count=%(fail_count)s, tripped=%(tripped)s, last_failure_ts=%(last_failure_ts)s WHERE id=1", before)


def _req(mint, side, amount_in=1_000_000, **kw):
    return ExecRequest(decision_id=None, mint=mint, pool=None, side=side, amount_in=amount_in, slippage_bps=300, max_slippage_bps=500, decimals=6, **kw)


def _fails():
    with transaction() as conn:
        return conn.execute("SELECT fail_count FROM circuit_state WHERE id=1").fetchone()["fail_count"]


def _fill(token_delta, lamports_delta, price=1e-7):
    return FillResult(True, None, None, "sig", "confirmed", 0, None, token_delta, lamports_delta, price, 1)


def test_failed_buys_that_moved_nothing_may_go_again_and_a_price_that_ran_is_no_execution_failure(circuit1):
    mint = f"M{uuid.uuid4().hex[:10]}"
    fb = _FakeBroker(); w = ExecutionWorker(fb)
    try:
        fb.next = FillResult(False, 1, None, None, "price_ran", None, "price ran: quoted 1.1e-07 SOL per token, +10.0% over the decision's 1e-07", 0, 0, None, 1)
        r = w._execute(_req(mint, "buy", 10 ** 8, ref_price=1e-7))
        assert not r.ok and r.retryable and r.error.startswith("price ran") and (fb.kw["ref_price"], fb.kw["decimals"]) == (1e-7, 6) and _fails() == 0
        fb.next = FillResult(False, 2, None, None, "order_error", 1, "Insufficient funds", 0, 0, None, 1)
        r = w._execute(_req(mint, "buy", 10 ** 8))
        assert not r.ok and r.retryable and _fails() == 1                              # a failure, but nothing landed
        fb.next = FillResult(False, 3, 99, "sig", "confirmed", 0, None, 0, 0, None, 1)
        assert not w._execute(_req(mint, "buy", 10 ** 8)).retryable and _fails() == 2  # an unverified fill row: never sent again
        with transaction() as conn:
            ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000_000, cost_sol=0.1, entry_price=1e-7, decision_id=None, fees_sol=0.0)
        fb.kw = {}
        r = w._execute(_req(mint, "buy", 10 ** 8))                                     # the sale queued before it left the position open
        assert not r.ok and r.retryable and r.error.startswith("held") and fb.kw == {} and _fails() == 2
    finally:
        w.stop()
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint = %s", (mint,))


def test_a_dead_bag_sweep_books_its_proceeds_onto_the_written_off_position(circuit1):
    mint = f"M{uuid.uuid4().hex[:10]}"
    with transaction() as conn:
        pid = ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000_000, cost_sol=0.2, entry_price=2e-7, decision_id=None, fees_sol=0.0)
        ledger.close_position(conn, position_id=pid, exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=None, forced_kind="dead_bag")
    fb = _FakeBroker(); w = ExecutionWorker(fb)
    try:
        fb.next = _fill(-1_000_000, 30_000_000, 3e-8)
        r = w._execute(_req(mint, "sell", forced_kind="dead_bag_sweep"))
        assert r.ok and r.position_id == pid and r.closed and r.realized_sol == pytest.approx(0.03)
        with transaction() as conn:
            p = _pos(conn, pid)
        assert p["status"] == "closed" and p["realized_sol"] == pytest.approx(-0.2 + 0.03)
    finally:
        w.stop()
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint = %s", (mint,))


def test_a_sale_that_empties_its_mint_closes_the_token_account_and_books_the_rent(circuit1):
    mint = f"M{uuid.uuid4().hex[:10]}"
    with transaction() as conn:
        pid = ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000_000, cost_sol=0.2, entry_price=2e-7, decision_id=None, fees_sol=0.0)
    fb = _ScriptBroker([_fill(-1_000_000, 250_000_000, 2.5e-7)], refund=2_034_280); w = ExecutionWorker(fb)
    try:
        assert w.submit(_req(mint, "sell", position_id=pid))
        r = w.results.get(timeout=10)
        assert r.ok and r.closed and fb.closed == [mint] and r.realized_sol == pytest.approx(0.05 + 0.00203428)
        with transaction() as conn:
            p = _pos(conn, pid)
        assert p["status"] == "closed" and p["realized_sol"] == pytest.approx(0.25 - 0.2 + 0.00203428) and w.pending() == set()
    finally:
        w.stop()
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint = %s", (mint,))


def test_a_buy_queues_behind_the_sale_of_its_mint_which_keeps_its_token_account(circuit1):
    """Paper sold the token and bought it back in one minute: the live buy waits behind the sale, runs once the sale has
    closed the position, and the token account stays for it."""
    mint = f"M{uuid.uuid4().hex[:10]}"
    with transaction() as conn:
        pid = ledger.open_position(conn, book="live", mint=mint, pool=None, qty_raw=1_000_000, cost_sol=0.2, entry_price=2e-7, decision_id=None, fees_sol=0.0)
    gate = threading.Event()
    fb = _ScriptBroker([_fill(-1_000_000, 250_000_000, 2.5e-7), _fill(900_000, -200_000_000, 2.2e-7)], gate=gate, refund=2_034_280); w = ExecutionWorker(fb)
    try:
        assert w.submit(_req(mint, "sell", position_id=pid)) and fb.started.wait(10) and w.selling() == {mint}
        assert w.submit(_req(mint, "buy", 200_000_000))
        assert not w.submit(_req(mint, "buy", 1)) and not w.submit(_req(mint, "sell"))     # one buy behind one sale, nothing more
        assert w.selling() == set() and w.pending() == {mint}
        gate.set()
        r1, r2 = w.results.get(timeout=10), w.results.get(timeout=10)
        assert (r1.request.side, r1.ok, r1.closed, r2.request.side, r2.ok) == ("sell", True, True, "buy", True) and fb.closed == []
        with transaction() as conn:
            rows = conn.execute("SELECT status, qty FROM positions WHERE book = 'live' AND mint = %s ORDER BY id", (mint,)).fetchall()
        assert [(r["status"], int(r["qty"])) for r in rows] == [("closed", 1_000_000), ("open", 900_000)]
    finally:
        gate.set(); w.stop()
        with transaction() as conn:
            conn.execute("DELETE FROM positions WHERE mint = %s", (mint,))


def test_stop_waits_for_the_order_in_flight_and_drops_the_queued_ones(circuit1):
    a, b = f"M{uuid.uuid4().hex[:10]}", f"M{uuid.uuid4().hex[:10]}"
    gate = threading.Event()
    fb = _ScriptBroker([_fill(-5, 5), _fill(-5, 5)], gate=gate); w = ExecutionWorker(fb)
    assert w.submit(_req(a, "sell")) and fb.started.wait(10) and w.submit(_req(b, "sell"))
    threading.Timer(0.3, gate.set).start()
    w.stop(timeout=10)
    assert not w.thread.is_alive() and [c["mint"] for c in fb.calls] == [a] and w.pending() == set()
    assert w.results.get_nowait().request.mint == a and not w.submit(_req(b, "sell"))
