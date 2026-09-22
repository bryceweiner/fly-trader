import threading, time
from fly_trader.ops import supervisor as S


def test_supervisor_thread_lifecycle(monkeypatch, db_conn):
    ticks = {"n": 0}

    def fake(stop_event=None):
        while not stop_event.is_set():
            ticks["n"] += 1
            time.sleep(0.05)

    monkeypatch.setattr(S, "_entry", lambda name: fake)
    monkeypatch.setattr(S.Supervisor, "external", lambda self: {})
    sup = S.Supervisor()
    assert sup.start("discover", started_by="test")
    time.sleep(0.3)
    assert sup.alive("discover") and ticks["n"] > 2
    assert not sup.start("discover")           # idempotent while alive
    assert sup.stop("discover", grace_s=5)
    assert not sup.alive("discover")
    st = sup.status()["discover"]
    assert st["stopped_at"] is not None and st["error"] is None


def test_supervisor_records_crash(monkeypatch, db_conn):
    def boom(stop_event=None):
        raise ValueError("kaput")
    monkeypatch.setattr(S, "_entry", lambda name: boom)
    monkeypatch.setattr(S.Supervisor, "external", lambda self: {})
    sup = S.Supervisor()
    sup.start("runner", started_by="test")
    sup.threads["runner"].join(5)
    assert "kaput" in sup.status()["runner"]["error"]


def test_a_previous_consoles_row_with_our_own_pid_is_stale(monkeypatch, db_conn):
    """In a container the console is always pid 1, so the row the last boot left carries this boot's pid. A new console
    has registered nothing yet: such a row is stale, and without closing it every restart refused to register."""
    import os
    from fly_trader.db.connection import transaction
    with transaction() as conn:
        conn.execute("UPDATE processes SET stopped_at = now() WHERE stopped_at IS NULL AND cmd[1] = 'thread'")
        rid = conn.execute("INSERT INTO processes (name, pid, cmd, log_path, started_by) VALUES ('discover', %s, %s, 'logs/discover.log', 'test') RETURNING id",
                           (os.getpid(), ["thread", "discover"])).fetchone()["id"]
    monkeypatch.setattr(S, "_entry", lambda name: (lambda stop_event=None: stop_event.wait(5)))
    monkeypatch.setattr(S.Supervisor, "external", lambda self: {})
    sup = S.Supervisor()
    with transaction() as conn:
        row = conn.execute("SELECT stopped_at, exit_code FROM processes WHERE id = %s", (rid,)).fetchone()
    assert row["stopped_at"] is not None and row["exit_code"] == -1                     # closed at construction, not at start
    assert sup.start("discover", started_by="test")                                     # and the worker registers again
    sup.stop("discover", grace_s=5)
