"""The selector's would-be trades keep their features after they resolve (agent/fly_session.py, Bryce 2026-11-18: the fly
should learn what a bad coin looks like): fly_scored dropped every row's feature vector once its outcome was known, so
no live pick could ever train or judge a bad-coin veto. A row the selector would trade (teacher_allow) keeps it, with its
outcome, for PICKS_KEEP_DAYS; every other row loses it as before and goes after SCORED_KEEP_DAYS."""
from datetime import datetime, timedelta, timezone

from fly_trader.agent import fly_session as F

MINT = "FLYPICKtest"
T0 = datetime(2026, 11, 1, tzinfo=timezone.utc)


def _rows(conn, ts=T0):
    with conn.cursor() as cur:
        cur.execute("DELETE FROM fly_scored WHERE mint LIKE 'FLYPICK%%'")
        for k, (allow, state) in enumerate(((True, "pending"), (False, "pending"), (True, "awaiting_fill"), (False, "awaiting_fill"))):
            cur.execute("INSERT INTO fly_scored (ts, mint, strategy, hold_min, x, score, frozen_score, line, frozen_line, bootstrap_id, teacher_allow, chain, state) "
                        "VALUES (%s, %s, 'ev', 240, %s, 0.1, 0.1, 0.05, 0.05, 641, %s, 'sol', %s)", (ts, f"{MINT}{k}", [0.5, 1.5], allow, state))
    conn.commit()


def _x(conn, k):
    return conn.execute("SELECT x, state FROM fly_scored WHERE mint = %s", (f"{MINT}{k}",)).fetchone()


def test_resolving_keeps_the_features_of_the_selectors_would_be_trades_only(db_conn):
    _rows(db_conn)
    try:
        with db_conn.cursor() as cur:
            cur.executemany(F.RESOLVE_SQL, [(T0, f"{MINT}0", "ev"), (T0, f"{MINT}1", "ev")])
            cur.executemany(F.LABEL_SQL, [(0.2, "resolved", 0.2, "resolved", T0, f"{MINT}2", "ev"), (0.2, "resolved", 0.2, "resolved", T0, f"{MINT}3", "ev")])
        db_conn.commit()
        got = [_x(db_conn, k) for k in range(4)]
    finally:
        db_conn.execute("DELETE FROM fly_scored WHERE mint LIKE 'FLYPICK%%'"); db_conn.commit()
    assert [r["state"] for r in got] == ["resolved"] * 4
    assert [r["x"] is not None for r in got] == [True, False, True, False]


def test_a_row_waiting_for_its_fill_keeps_its_features_and_the_fill_resolves_it(db_conn):
    _rows(db_conn)
    try:
        with db_conn.cursor() as cur:
            cur.executemany(F.LABEL_SQL, [(0.2, "awaiting_fill", 0.2, "awaiting_fill", T0, f"{MINT}1", "ev")])     # not the selector's: kept only while waiting
        db_conn.commit()
        waiting = _x(db_conn, 1)
        with db_conn.cursor() as cur:
            cur.executemany(F.FILL_SQL, [(0.01, T0, f"{MINT}1", "ev"), (0.01, T0, f"{MINT}2", "ev")])
        db_conn.commit()
        after = [_x(db_conn, 1), _x(db_conn, 2)]
    finally:
        db_conn.execute("DELETE FROM fly_scored WHERE mint LIKE 'FLYPICK%%'"); db_conn.commit()
    assert waiting["x"] is not None and after[0]["x"] is None and after[1]["x"] is not None


def test_pruning_keeps_the_selectors_would_be_trades_for_a_year(db_conn):
    now = datetime.now(timezone.utc)
    try:
        for age_d, expect in ((40, {0: True, 1: False}), (400, {0: False, 1: False}), (10, {0: True, 1: True})):
            _rows(db_conn, ts=now - timedelta(days=age_d))
            db_conn.execute("UPDATE fly_scored SET state = 'resolved' WHERE mint LIKE 'FLYPICK%%'"); db_conn.commit()
            F.prune_scored(db_conn, now.timestamp()); db_conn.commit()
            left = {k: _x(db_conn, k) is not None for k in (0, 1)}
            assert left == expect, (age_d, left)
    finally:
        db_conn.execute("DELETE FROM fly_scored WHERE mint LIKE 'FLYPICK%%'"); db_conn.commit()
