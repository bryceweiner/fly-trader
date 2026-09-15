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
    out = reset_training_state(reason="test")
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM positions WHERE book='paper_free'").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM positions WHERE book='live'").fetchone()["n"] == 1
        assert conn.execute("SELECT count(*) AS n FROM beats").fetchone()["n"] == 0
        assert conn.execute("SELECT count(*) AS n FROM tokens WHERE mint='M1'").fetchone()["n"] == 1
        conn.execute("DELETE FROM positions WHERE book='live' AND mint='M2'")
    assert (tmp_path / "arch").exists() and out["rows"]["positions"] >= 1


def test_selector_status_cleared_and_training_stats_reset(db_conn, tmp_path, monkeypatch):
    from fly_trader.ops.reset import reset_training_stats
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(config, "BRAIN_DIR", tmp_path / "brain")
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', '{}'::jsonb), ('training_status', '{}'::jsonb) ON CONFLICT (key) DO NOTHING")
        conn.execute("INSERT INTO events (level, source, message) VALUES ('info','selector','walk-forward 2026-09-01'), ('info','selector','selector saved (snapshot 1)'), "
                     "('info','fly_selector','fly epoch 1'), ('info','selector','selector session started')")
    reset_training_state(reason="test")
    with transaction() as conn:
        assert conn.execute("SELECT count(*) AS n FROM ui_settings WHERE key = 'selector_status'").fetchone()["n"] == 0
    out = reset_training_stats("selector", reason="test")
    msgs = ("('walk-forward 2026-09-01','selector saved (snapshot 1)','fly epoch 1','selector session started')")
    with transaction() as conn:
        left = sorted(r["message"] for r in conn.execute("SELECT message FROM events WHERE source IN ('selector','fly_selector') AND message IN " + msgs).fetchall())
        assert left == ["fly epoch 1", "selector session started"]         # another regimen's stats and session events stay
        assert conn.execute("SELECT count(*) AS n FROM ui_settings WHERE key = 'training_status'").fetchone()["n"] == 0
    assert out["events"] >= 2
    reset_training_stats("fly_selector", reason="test")
    with transaction() as conn:
        assert [r["message"] for r in conn.execute("SELECT message FROM events WHERE source IN ('selector','fly_selector') AND message IN " + msgs).fetchall()] == ["selector session started"]
        conn.execute("DELETE FROM events WHERE message = 'selector session started'")


def _race_fixture(conn):
    import uuid
    runs = {k: str(uuid.uuid4()) for k in ("selector", "fly", "pretrain")}
    for kind, rid in runs.items():
        conn.execute("INSERT INTO runs (run_id, kind) VALUES (%s, %s)", (rid, kind))
    beats = {k: conn.execute("INSERT INTO beats (run_id, beat_no) VALUES (%s, 1) RETURNING id", (rid,)).fetchone()["id"] for k, rid in runs.items()}
    for k, rid in runs.items():
        conn.execute("INSERT INTO decisions (run_id, beat_id, kind, mint) VALUES (%s, %s, %s, 'RACE_T')", (rid, beats[k], f"{k}_enter"))
    conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, status) VALUES ('paper_selector','RACE_S',1,0.1,'open'),('paper_fly','RACE_F',1,0.1,'open'),('paper_free','RACE_X',1,0.1,'open')")
    for book, k in (("paper_selector", "selector"), ("paper_fly", "fly"), ("paper_free", "pretrain")):
        conn.execute("INSERT INTO wealth_marks (beat_id, book, wealth) VALUES (%s, %s, 5.0)", (beats[k], book))
    conn.execute("INSERT INTO fly_scored (ts, mint, score) VALUES (now(), 'RACE_F', 0.01)")
    conn.execute("INSERT INTO fly_rollbacks (reason) VALUES ('test')")
    conn.execute("INSERT INTO brain_snapshots (path, kind) VALUES ('p1','fly_plastic'), ('p2','fly_selector'), ('p3','checkpoint')")
    conn.execute("INSERT INTO book_state (book, halted) VALUES ('paper_fly', true) ON CONFLICT (book) DO UPDATE SET halted = true")
    return runs


def _n(conn, sql, *a):
    return conn.execute(sql, a).fetchone()["n"]


def test_runner_reset_keeps_the_race_books_and_the_fly_and_reset_fly_clears_only_the_fly(db_conn, tmp_path, monkeypatch):
    from fly_trader.ops.reset import reset_fly
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path / "arch")
    monkeypatch.setattr(config, "BRAIN_DIR", tmp_path / "brain")
    (tmp_path / "brain" / "plastic").mkdir(parents=True); (tmp_path / "brain" / "plastic" / "state.pt").write_bytes(b"x")
    with transaction() as conn:
        runs = _race_fixture(conn)
    reset_training_state(reason="test")
    with transaction() as conn:
        assert _n(conn, "SELECT count(*) AS n FROM positions WHERE mint IN ('RACE_S','RACE_F')") == 2
        assert _n(conn, "SELECT count(*) AS n FROM positions WHERE mint = 'RACE_X'") == 0
        assert _n(conn, "SELECT count(*) AS n FROM wealth_marks WHERE book IN ('paper_selector','paper_fly')") >= 2
        assert _n(conn, "SELECT count(*) AS n FROM wealth_marks WHERE book = 'paper_free'") == 0
        assert _n(conn, "SELECT count(*) AS n FROM decisions WHERE mint = 'RACE_T'") == 2                    # the race sessions' decisions
        assert _n(conn, "SELECT count(*) AS n FROM runs WHERE run_id = %s", runs["pretrain"]) == 0
        assert _n(conn, "SELECT count(*) AS n FROM fly_scored WHERE mint = 'RACE_F'") == 1
        assert _n(conn, "SELECT count(*) AS n FROM brain_snapshots WHERE path IN ('p1','p2')") == 2
        assert _n(conn, "SELECT count(*) AS n FROM brain_snapshots WHERE path = 'p3'") == 0
    assert (tmp_path / "brain" / "plastic" / "state.pt").exists()
    reset_fly(reason="test")
    with transaction() as conn:
        assert _n(conn, "SELECT count(*) AS n FROM positions WHERE mint = 'RACE_F'") == 0
        assert _n(conn, "SELECT count(*) AS n FROM positions WHERE mint = 'RACE_S'") == 1                    # the selector's book stays
        assert _n(conn, "SELECT count(*) AS n FROM decisions WHERE mint = 'RACE_T'") == 1
        assert _n(conn, "SELECT count(*) AS n FROM fly_scored") == 0 and _n(conn, "SELECT count(*) AS n FROM fly_rollbacks") == 0
        assert _n(conn, "SELECT count(*) AS n FROM brain_snapshots WHERE path = 'p1'") == 0
        assert _n(conn, "SELECT count(*) AS n FROM brain_snapshots WHERE path = 'p2'") == 1                   # the bootstrap stays
        assert _n(conn, "SELECT count(*) AS n FROM book_state WHERE book = 'paper_fly'") == 0
        conn.execute("DELETE FROM positions WHERE mint LIKE 'RACE_%%'"); conn.execute("DELETE FROM decisions WHERE mint = 'RACE_T'")
        conn.execute("DELETE FROM wealth_marks WHERE book IN ('paper_selector','paper_fly')"); conn.execute("DELETE FROM runs WHERE run_id = %s", (runs["selector"],))
        conn.execute("DELETE FROM brain_snapshots WHERE path = 'p2'")
    assert not (tmp_path / "brain" / "plastic").exists() and list((tmp_path / "arch").glob("reset_fly_*/plastic/state.pt"))


def test_selector_runner_resets_on_start(monkeypatch):
    from fly_trader.agent import minute_engine, runner
    from fly_trader.ops import reset
    calls = []
    monkeypatch.setattr(config, "LIVE_ENABLED", False); monkeypatch.setattr(config, "RESET_ON_START", True)
    monkeypatch.setattr(reset, "reset_training_state", lambda reason="": calls.append("reset"))
    monkeypatch.setattr(minute_engine, "main", lambda stop_event=None, live=False: calls.append("session"))
    monkeypatch.setattr(runner, "record_event", lambda *a, **k: None)
    runner._main()
    assert calls == ["reset", "session"]
