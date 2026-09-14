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
