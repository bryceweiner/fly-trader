"""RH's live-versus-paper gap pause (rh/live.RhLiveMirror.gap) leaves out a position whose tokens the reconciler found gone from
the wallet while it was held (ANT 2026-11-03, 0x493f2e7b 11-05): the coin's contract took them, the reconcile pause handles it,
and it is not an execution gap. A loss the wallet itself made still counts."""
import json
from datetime import datetime, timedelta, timezone

from fly_trader.rh import live as L

T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _pairs(conn, n, live_r, paper_r, mint_of=lambda k: f"0xgap{k:04d}"):
    with conn.cursor() as cur:
        for k in range(n):
            t = T0 + timedelta(hours=k)
            d = cur.execute("INSERT INTO decisions (ts, kind, mint) VALUES (%s, 'fly_enter', %s) RETURNING id", (t, mint_of(k))).fetchone()["id"]
            for book, r in ((L.BOOK, live_r(k)), (L.MIRROR, paper_r(k))):
                cur.execute("""INSERT INTO positions (book, mint, opened_at, closed_at, qty, cost_sol, realized_sol, status, chain, entry_decision_id)
                               VALUES (%s, %s, %s, %s, 1, 0.01, %s, 'closed', 'rh', %s)""", (book, mint_of(k), t, t + timedelta(hours=4), 0.01 * r, d))
    conn.commit()


def _clear(conn):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM positions WHERE mint LIKE '0xgap%%'"); cur.execute("DELETE FROM decisions WHERE mint LIKE '0xgap%%'")
        cur.execute("DELETE FROM circuit_events WHERE detail::text LIKE '%%0xgap%%'")
        cur.execute("UPDATE circuit_state SET entries_paused = false WHERE id = %s", (L.RH_CIRCUIT,))
    conn.commit()


def test_a_position_taken_from_the_wallet_is_not_an_execution_gap(db_conn):
    _clear(db_conn)
    stolen = {3, 17}                                                  # live -100 %, paper -8 %: alone a ~3.7-point gap over 50
    _pairs(db_conn, L.GAP_TRADES + 2, lambda k: -1.0 if k in stolen else 0.02, lambda k: -0.08 if k in stolen else 0.02)
    try:
        before = L.RhLiveMirror.gap(None, db_conn); db_conn.commit()
        assert before.get("paused")                                   # the write-offs read as execution
        db_conn.execute("UPDATE circuit_state SET entries_paused = false WHERE id = %s", (L.RH_CIRCUIT,))
        for k in stolen:
            db_conn.execute("INSERT INTO circuit_events (ts, kind, detail) VALUES (%s, 'rh_reconcile', %s)",
                            (T0 + timedelta(hours=k, minutes=10), json.dumps({"token_gaps": [{"token": f"0xgap{k:04d}", "books": 1, "chain": 0}]})))
        db_conn.commit()
        after = L.RhLiveMirror.gap(None, db_conn); db_conn.commit()
        assert not after.get("paused") and after["trades"] == L.GAP_TRADES and abs(after["live_mean"] - after["mirror_mean"]) < 1e-9
    finally:
        _clear(db_conn)


def test_a_real_execution_gap_still_pauses(db_conn):
    _clear(db_conn)
    _pairs(db_conn, L.GAP_TRADES, lambda k: -0.01, lambda k: 0.02)  # live 3 points behind on every trade
    try:
        assert L.RhLiveMirror.gap(None, db_conn).get("paused")
    finally:
        db_conn.rollback(); _clear(db_conn)
