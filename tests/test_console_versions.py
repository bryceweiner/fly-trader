"""The version strip along the top of the console (ui/api/meta.py, Bryce 2026-11-07: so two installs can be told
apart at a glance): the code it runs and each chain's selector and fly, the models by their file's
sha256 so two installs can be compared."""
import json

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ui.api import meta


def test_the_code_version_prefers_a_stamp_then_a_release_then_git(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPO_ROOT", tmp_path)
    monkeypatch.setenv("FLY_CODE_VERSION", "box 2026-11-07a")
    assert meta.code_version()["label"] == "box 2026-11-07a"
    monkeypatch.delenv("FLY_CODE_VERSION")
    (tmp_path / "release.json").write_text(json.dumps({"seq": 21, "commit": "654387b199aa", "created_at": "2026-11-07T09:20:00Z"}))
    c = meta.code_version()
    assert (c["label"], c["release"], c["commit"], c["source"]) == ("release 21 · 654387b", 21, "654387b", "release.json")
    (tmp_path / "release.json").unlink()
    c = meta.code_version()                                       # no git here either: the package's version
    assert c["source"] == "package" and c["commit"] is None


def test_this_tree_reports_its_git_commit(monkeypatch):
    monkeypatch.delenv("FLY_CODE_VERSION", raising=False)
    c = meta.code_version()
    assert c["source"] == "git" and len(c["commit"]) == 7 and c["label"].startswith(c["commit"])


def test_each_chain_reports_its_selector_and_fly_by_number_and_sha(db_conn, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    keys = ("selector_status", "fly_status", "selector_status_rh", "fly_status_rh")
    with transaction() as conn:
        saved = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM ui_settings WHERE key = ANY(%s)", (list(keys),)).fetchall()}
        sel = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind) VALUES ('/x/selector_20261101T162142Z.skops', '0dc93f9dd6b5aa', 'selector') RETURNING id").fetchone()["id"]
        fly = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind) VALUES ('/x/fly_20261026T112337Z.pt', '59115da694c3bb', 'fly_selector') RETURNING id").fetchone()["id"]
        for k, v in (("selector_status", {"snapshot": sel}), ("fly_status", {"bootstrap": fly}), ("selector_status_rh", {"snapshot": sel}), ("fly_status_rh", {})):
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (k, json.dumps(v)))
    try:
        got = meta.model_versions()
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM brain_snapshots WHERE id = ANY(%s)", ([sel, fly],))
            conn.execute("DELETE FROM ui_settings WHERE key = ANY(%s)", (list(keys),))
            for k, v in saved.items():
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s)", (k, json.dumps(v) if not isinstance(v, str) else v))
    assert got == [{"chain": "sol", "selector": {"id": sel, "sha": "0dc93f9d", "file": "selector_20261101T162142Z.skops"},
                    "fly": {"id": fly, "sha": "59115da6", "file": "fly_20261026T112337Z.pt"}},
                   {"chain": "rh", "selector": {"id": sel, "sha": "0dc93f9d", "file": "selector_20261101T162142Z.skops"}, "fly": None}]


def test_the_selector_shown_is_the_flys_gate_once_the_selector_book_stopped(db_conn, monkeypatch):
    """A paper-only install handed over on 2026-11-04: its selector book's last report names the old selector while the fly's gate
    follows the pinned one."""
    monkeypatch.setattr(config, "RH_ENABLED", False)
    with transaction() as conn:
        saved = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM ui_settings WHERE key IN ('selector_status', 'fly_status')").fetchall()}
        old = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind) VALUES ('/x/selector_20261026T112337Z.skops', 'ce5ece690558aa', 'selector') RETURNING id").fetchone()["id"]
        new = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind) VALUES ('/x/selector_20261101T162142Z.skops', '0dc93f9dd6b5aa', 'selector') RETURNING id").fetchone()["id"]
        for k, v in (("selector_status", {"snapshot": old, "stage": "retired (handed over to the fly)"}), ("fly_status", {"bootstrap": None, "gate": new})):
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (k, json.dumps(v)))
    try:
        got = meta.model_versions()
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM brain_snapshots WHERE id = ANY(%s)", ([old, new],))
            conn.execute("DELETE FROM ui_settings WHERE key IN ('selector_status', 'fly_status')")
            for k, v in saved.items():
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s)", (k, json.dumps(v) if not isinstance(v, str) else v))
    assert got == [{"chain": "sol", "selector": {"id": new, "sha": "0dc93f9d", "file": "selector_20261101T162142Z.skops"}, "fly": None}]


def test_the_meta_topic_carries_the_versions(console):
    r = console.topic("meta")
    assert r.status_code == 200 and set(r.json()["versions"]) == {"code", "models"} and r.json()["versions"]["code"]["label"]


def test_the_model_at_work_is_the_flys_gate_not_a_dead_sessions_run(db_conn):
    """2026-11-12: three selector sessions had died unclosed, and after the handover the selector book had none, so the
    Selector tile showed the newest of them (#154, 10-17) "switching to #791" while the fly bought on #791."""
    from fly_trader.ui import state
    with transaction() as conn:
        saved = {r["key"]: r["value"] for r in conn.execute("SELECT key, value FROM ui_settings WHERE key IN ('selector_status', 'fly_status')").fetchall()}
        old = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind, note) VALUES ('/x/old.skops', 'aa', 'selector', '{\"threshold\": 0.115, \"horizon_min\": 120}') RETURNING id").fetchone()["id"]
        new = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind, note) VALUES ('/x/new.skops', 'bb', 'selector', '{\"threshold\": 0.06, \"horizon_min\": 240}') RETURNING id").fetchone()["id"]
        conn.execute("INSERT INTO runs (run_id, kind, started_at, brain_snapshot_id, status, config) VALUES (gen_random_uuid(), 'selector', now(), %s, 'running', %s)",
                     (old, json.dumps({"chain": "sol", "threshold": 0.115, "horizon_min": 120})))
        conn.execute("INSERT INTO runs (run_id, kind, started_at, brain_snapshot_id, status, config) VALUES (gen_random_uuid(), 'selector', now() - interval '1 day', %s, 'finished', %s)",
                     (new, json.dumps({"chain": "sol", "threshold": 0.06, "horizon_min": 240})))
        for k, v in (("selector_status", {"snapshot": new, "stage": "retired (handed over to the fly)"}), ("fly_status", {"gate": new})):
            conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value", (k, json.dumps(v)))
    try:
        m = state.loaded_model()
        with transaction() as conn:
            conn.execute("UPDATE ui_settings SET value = %s WHERE key = 'fly_status'", (json.dumps({}),))
        by_book = state.loaded_model()                                                 # no fly report: the selector book's
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM runs WHERE brain_snapshot_id = ANY(%s)", ([old, new],))
            conn.execute("DELETE FROM brain_snapshots WHERE id = ANY(%s)", ([old, new],))
            conn.execute("DELETE FROM ui_settings WHERE key IN ('selector_status', 'fly_status')")
            for k, v in saved.items():
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s)", (k, json.dumps(v) if not isinstance(v, str) else v))
    assert m["id"] == new and m["run"]["horizon_min"] == 240 and m["run"]["threshold"] == 0.06
    assert by_book["id"] == new
