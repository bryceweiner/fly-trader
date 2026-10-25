"""The console builds the corpus in a child process every round, corpus_meta's rebuild first (2026-10-23): until then
only more than three stale aggregates went to the child, so a restart that killed one after its aggregates but before its
features left the console re-featuring 20 days in its own interpreter (80 s a day, 16 GB, ~100 % CPU), and corpus_meta's
rebuild took 36-82 s of it every 30 minutes."""
import threading

from fly_trader.train import corpus_meta, mature, replay_assemble


def test_every_round_runs_in_a_child_with_corpus_metas_rebuild(monkeypatch):
    calls = []; stop = threading.Event()
    monkeypatch.setattr(replay_assemble, "assemblable_days", lambda: [])
    monkeypatch.setattr(corpus_meta, "rebuild", lambda: calls.append("console rebuild"))
    monkeypatch.setattr(mature, "loop_once", lambda: calls.append("console build"))
    monkeypatch.setattr(mature, "loop_once_apart", lambda ev=None, meta=False: (calls.append(("child", meta)), stop.set(), 0)[-1])
    monkeypatch.setattr(replay_assemble, "_last_meta", [0.0])
    replay_assemble.assemble_loop(stop, idle_s=0.0)
    assert calls == [("child", True)] and replay_assemble._last_meta[0] > 0


def test_the_child_rebuilds_corpus_meta_before_its_builds_and_stops_when_that_fails(monkeypatch):
    calls = []
    monkeypatch.setattr(mature.logging_setup, "setup", lambda *a, **k: None)
    monkeypatch.setattr(mature, "loop_once", lambda: calls.append("build") or 0)
    monkeypatch.setattr(corpus_meta, "rebuild", lambda: calls.append("meta"))
    assert mature.main(meta=True) == 0 and calls == ["meta", "build"]
    calls.clear()

    def boom():
        raise RuntimeError("archive unreadable")
    monkeypatch.setattr(corpus_meta, "rebuild", boom)
    assert mature.main(meta=True) == mature.META_FAILED and calls == []          # no day is built on a stale corpus_meta
