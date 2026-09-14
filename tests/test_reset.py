from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ops.reset import reset_training_state


def test_reset_wipes_paper_but_keeps_live_and_market_data(db_conn, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(config, "BRAIN_DIR", tmp_path / "brain")
    with transaction() as conn:
        conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, status) VALUES ('paper_free','M1',1,0.1,'open'),('live','M2',1,0.1,'open')")
        conn.execute("INSERT INTO beats (beat_no) VALUES (1)")
        conn.execute("INSERT INTO tokens (mint, watch_status) VALUES ('M1','watch') ON CONFLICT DO NOTHING")
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('replay_clock:replay', '{}'::jsonb) ON CONFLICT (key) DO NOTHING")
    out = reset_training_state(reason="test")
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM positions WHERE book='paper_free'").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM positions WHERE book='live'").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM beats").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM tokens WHERE mint='M1'").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM ui_settings WHERE key LIKE 'replay_clock:%%'").fetchone()["n"] == 0
        conn.execute("DELETE FROM positions WHERE book='live' AND mint='M2'")
    assert (tmp_path / "arch").exists() and out["rows"]["positions"] >= 1


def test_selector_status_cleared_and_training_stats_reset(db_conn, tmp_path, monkeypatch):
    from fly_trader.ops.reset import reset_training_stats
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(config, "BRAIN_DIR", tmp_path / "brain")
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', '{}'::jsonb), ('training_status', '{}'::jsonb) ON CONFLICT (key) DO NOTHING")
        conn.execute("INSERT INTO events (level, source, message) VALUES ('info','selector','walk-forward 2026-09-01'), ('info','selector','selector saved (snapshot 1)'), "
                     "('info','ppo','iteration 3'), ('info','selector','selector session started')")
    reset_training_state(reason="test")
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM ui_settings WHERE key = 'selector_status'").fetchone()["n"] == 0
    out = reset_training_stats("selector", reason="test")
    msgs = ("('walk-forward 2026-09-01','selector saved (snapshot 1)','iteration 3','selector session started')")
    with transaction() as conn:
        left = sorted(r["message"] for r in conn.execute("SELECT message FROM events WHERE source IN ('selector','ppo') AND message IN " + msgs).fetchall())
        assert left == ["iteration 3", "selector session started"]         # another regimen's stats and session events stay
        assert conn.execute("SELECT count(*) AS n FROM ui_settings WHERE key = 'training_status'").fetchone()["n"] == 0
    assert out["events"] >= 2
    reset_training_stats("ppo", reason="test")
    with transaction() as conn:
        assert [r["message"] for r in conn.execute("SELECT message FROM events WHERE source IN ('selector','ppo') AND message IN " + msgs).fetchall()] == ["selector session started"]
        conn.execute("DELETE FROM events WHERE message = 'selector session started'")


def test_selector_runner_resets_on_start(monkeypatch):
    from fly_trader.agent import runner, selector_session
    from fly_trader.ops import reset
    calls = []
    monkeypatch.setattr(config, "LIVE_ENABLED", False); monkeypatch.setattr(config, "BRAIN_MODE", "selector"); monkeypatch.setattr(config, "RESET_ON_START", True)
    monkeypatch.setattr(reset, "reset_training_state", lambda reason="": calls.append("reset"))
    monkeypatch.setattr(selector_session, "main", lambda stop_event=None, live=False: calls.append("session"))
    monkeypatch.setattr(runner, "record_event", lambda *a, **k: None)
    runner._main()
    assert calls == ["reset", "session"]
