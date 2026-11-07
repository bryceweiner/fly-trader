"""A written-off RH token whose sweep keeps failing (0x493f2e7b, 2026-11-05: its contract reports a balance it never lets
move, and each hourly sweep's approve cost gas): left alone after ``SWEEP_GIVE_UP`` failures; a failed sweep's gas is
charged to the written-off position, not booked as an aborted buy; and the console counts no aborted position as a
trade."""
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fly_trader.rh import accounting as A, live as L

TOK = "0xdeadbagdeadbagdeadbagdeadbagdeadbag00001"


def _dead_bag(conn, failed: int):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM positions WHERE mint = %s", (TOK,)); cur.execute("DELETE FROM rh_intents WHERE token = %s", (TOK,))
        pid = cur.execute("""INSERT INTO positions (book, mint, pool, opened_at, closed_at, qty, cost_sol, realized_sol, fees_sol, status, chain, forced_exit_kind)
                             VALUES ('live_rh', %s, 'P', now() - interval '2 days', now() - interval '1 day', 1000, 0.004, -0.004, 0, 'closed', 'rh', 'dead_bag')
                             RETURNING id""", (TOK,)).fetchone()["id"]
        for k in range(failed):
            cur.execute("INSERT INTO rh_intents (book, kind, token, state, error, created_at) VALUES ('live_rh', 'sweep', %s, 'failed', 'simulation: reverted', now() - %s * interval '1 hour')",
                        (TOK, k + 1))
    conn.commit()
    return pid


def _clear(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM positions WHERE mint = %s", (TOK,)); cur.execute("DELETE FROM rh_intents WHERE token = %s", (TOK,))
    conn.commit()


class _Ex:
    def __init__(self):
        self.sent = []; self.wallet = SimpleNamespace(balance_of=lambda t: 1000)     # the token still reads as held

    def pending(self):
        return set()

    def submit(self, req):
        self.sent.append(req); return True


def _mirror():
    ex = _Ex(); m = L.RhLiveMirror(600.0, executor=ex)
    return m, ex


def test_a_token_whose_sweeps_keep_failing_is_left_alone(db_conn, monkeypatch):
    monkeypatch.setattr(L.A, "open_lots", lambda conn: [])
    now = datetime.now(timezone.utc)
    try:
        _dead_bag(db_conn, L.SWEEP_GIVE_UP - 1); m, ex = _mirror()
        assert m.sweep(db_conn, now) == 1 and ex.sent[0].token == TOK and ex.sent[0].kind == "sweep"       # tried again
        _dead_bag(db_conn, L.SWEEP_GIVE_UP); m, ex = _mirror()
        assert m.sweep(db_conn, now) == 0 and ex.sent == []                                                  # given up
    finally:
        _clear(db_conn)


def test_a_failed_sweeps_gas_goes_onto_its_written_off_position(db_conn, monkeypatch):
    monkeypatch.setattr(A, "intent_gas_wei", lambda conn, iid: 2 * 10 ** 12)                               # 0.000002 ETH
    try:
        pid = _dead_bag(db_conn, 1)
        iid = db_conn.execute("SELECT id FROM rh_intents WHERE token = %s", (TOK,)).fetchone()["id"]
        n_before = db_conn.execute("SELECT count(*) AS n FROM positions WHERE book = 'live_rh'").fetchone()["n"]
        assert A.charge_gas(db_conn, iid, None) == 2e-6; db_conn.commit()
        row = db_conn.execute("SELECT realized_sol, fees_sol, gas_q FROM positions WHERE id = %s", (pid,)).fetchone()
        n_after = db_conn.execute("SELECT count(*) AS n FROM positions WHERE book = 'live_rh'").fetchone()["n"]
        tied = db_conn.execute("SELECT position_id FROM rh_intents WHERE id = %s", (iid,)).fetchone()["position_id"]
    finally:
        _clear(db_conn)
    assert n_after == n_before and tied == pid                                                              # no aborted row
    assert abs(row["realized_sol"] - (-0.004 - 2e-6)) < 1e-12 and abs(row["fees_sol"] - 2e-6) < 1e-12 and abs(row["gas_q"] - 2e-6) < 1e-12


def test_a_failed_buys_gas_is_still_an_aborted_position(db_conn, monkeypatch):
    monkeypatch.setattr(A, "intent_gas_wei", lambda conn, iid: 10 ** 12)
    try:
        iid = db_conn.execute("INSERT INTO rh_intents (book, kind, token, state) VALUES ('live_rh', 'buy', %s, 'failed') RETURNING id", (TOK,)).fetchone()["id"]
        A.charge_gas(db_conn, iid, None); db_conn.commit()
        kinds = [r["forced_exit_kind"] for r in db_conn.execute("SELECT forced_exit_kind FROM positions WHERE mint = %s", (TOK,)).fetchall()]
    finally:
        _clear(db_conn)
    assert kinds == ["aborted"]
