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
