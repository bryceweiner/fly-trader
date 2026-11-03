"""The console's four Kalshi pages (fly_trader/ui/api/kalshi_*.py): a dormant install answers every topic with the
dormant marker and refuses every action; the topics read an empty or seeded test database without failing; the stack
and bootstrap captions get their saved time from brain_snapshots (the Streamlit page raised KeyError on ``sel['ts']``);
every action calls what it should, behind the same guards the Streamlit buttons had, now enforced server-side."""
import json
import re
from pathlib import Path

import pytest

from fly_trader import config
from fly_trader.db.connection import transaction

ROOT = Path(__file__).resolve().parents[1]
TOPICS = [("kalshi_overview.summary", {}), ("kalshi_overview.positions", {}), ("kalshi_trades.settled", {"book": "paper_kalshi_taker"}),
          ("kalshi_trades.orders", {"book": "paper_kalshi_maker"}), ("kalshi_trades.fills", {"book": "live_kalshi_taker"}),
          ("kalshi_trades.decisions", {"book": "paper_kalshi_taker"}), ("kalshi_model.stack", {}), ("kalshi_model.retraining", {}),
          ("kalshi_model.visual_fly", {}), ("kalshi_model.history", {}), ("kalshi_data.pipelines", {})]
ACTIONS = [("kalshi_model.retrain", {}), ("kalshi_model.train_stop", {}), ("kalshi_model.fly_command", {"cmd": "pause"}),
           ("kalshi_model.fly_rebootstrap", {}), ("kalshi_model.fly_reset", {})]


@pytest.fixture
def kalshi_on(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", "test-key-id")
    monkeypatch.setattr(config, "KALSHI_DIR", tmp_path / "kalshi")          # no feature parts: the real data dir has thousands
    from fly_trader.ui.api import kalshi_data
    for f in (kalshi_data.corpus_counts, kalshi_data.feature_parts, kalshi_data.quoted_per_minute, kalshi_data.feature_days): f.cache_clear()
    yield
    for f in (kalshi_data.corpus_counts, kalshi_data.feature_parts, kalshi_data.quoted_per_minute, kalshi_data.feature_days): f.cache_clear()


class Calls(list):
    """The recorded calls, plus ``state``: what the patched kalshi.pipeline.state() returns."""
    state: dict


@pytest.fixture
def calls(monkeypatch):
    """Record the Kalshi library calls the actions make, instead of making them."""
    from fly_trader.kalshi import fly_session, pipeline
    from fly_trader.ops import reset
    log = Calls(); log.state = state = {"value": {}}
    monkeypatch.setattr(pipeline, "request_run", lambda fly=False, reason=None: log.append(("request_run", fly, reason)))
    monkeypatch.setattr(pipeline, "state", lambda: dict(state["value"]))
    monkeypatch.setattr(fly_session, "send_command", lambda cmd: log.append(("send_command", cmd)))
    monkeypatch.setattr(reset, "reset_kalshi_fly", lambda reason="operator", archive=True: log.append(("reset_kalshi_fly", reason)) or {"archived_to": "/tmp/reset_kalshi_fly_x", "rows": {}})
    return log


# ---------------------------------------------------------------- dormant
def test_dormant_install_answers_every_topic_with_the_marker_and_refuses_actions(console, monkeypatch, calls):
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", "")
    for name, params in TOPICS:
        r = console.topic(name, **params)
        assert r.status_code == 200, (name, r.text)
        assert r.json() == {"dormant": True}, name
    for name, body in ACTIONS:
        r = console.act(name, body)
        assert r.status_code == 409 and "KALSHI_API_KEY_ID" in r.json()["error"], name
    assert calls == [] and console.sup.calls == []


# ---------------------------------------------------------------- topics on the test database
def test_every_topic_reads_the_test_database(console, kalshi_on):
    for name, params in TOPICS:
        r = console.topic(name, **params)
        assert r.status_code == 200, (name, r.text)
        assert "dormant" not in r.json(), name
    s = console.topic("kalshi_overview.summary").json()
    assert [a["book"] for a in s["arms"]] == ["paper_kalshi_taker", "paper_kalshi_maker"] and s["capital"] == config.KALSHI_CAPITAL_USD
    assert "Start the Kalshi trading engine on the Processes page." in s["narrative"]          # no runner in the fake supervisor
    assert s["live"] is None or config.KALSHI_LIVE_ENABLED
    d = console.topic("kalshi_trades.decisions", book="paper_kalshi_taker").json()
    assert d["default"] == ["Taker bought", "Maker posted", "Maker filled", "Settled"] and len(d["kinds"]) == 7
    assert console.topic("kalshi_trades.decisions", book="paper_kalshi_taker", kinds=[]).json()["rows"] == []
    p = console.topic("kalshi_data.pipelines").json()
    assert p["feed"]["badge"] == {"text": "stopped", "color": "gray"} and p["features"]["n"] == 0
    r = console.topic("kalshi_model.retraining").json()
    assert r["status"] == "Off" and r["alive"] is False and r["running"] is False


def test_topics_validate_their_parameters(console, kalshi_on):
    assert console.topic("kalshi_trades.settled", book="paper_selector").status_code == 400
    assert console.topic("kalshi_trades.decisions", book="paper_kalshi_taker", kinds=["Bought everything"]).status_code == 400
    assert console.topic("kalshi_trades.decisions", book="paper_kalshi_taker", kinds="Settled").status_code == 400


def test_book_topics_read_seeded_rows(console, kalshi_on):
    book = "paper_kalshi_maker"
    with transaction() as conn:
        conn.execute("INSERT INTO kalshi_markets (ticker, title, close_time) VALUES ('KXTEST-UI-A', 'Will the UI test pass?', now() + interval '5 hours') ON CONFLICT DO NOTHING")
        conn.execute("""INSERT INTO kalshi_positions (book, ticker, side, contracts, cost_cents, fee_cents, avg_price_cents, status, result, payout_cents, realized_cents, closed_at, strategy)
                        VALUES (%s, 'KXTEST-UI-A', 'yes', 10, 400, 10, 40, 'settled', 'yes', 1000, 590, now(), 'favorite'),
                               (%s, 'KXTEST-UI-A', 'no', 5, 300, 5, 60, 'open', NULL, NULL, NULL, NULL, 'ev')""", (book, book))
        conn.execute("""INSERT INTO kalshi_orders (book, ticker, side, price_cents, count, tif, post_only, status, strategy)
                        VALUES (%s, 'KXTEST-UI-A', 'yes', 41, 10, 'good_till_canceled', true, 'resting', 'favorite')""", (book,))
        conn.execute("""INSERT INTO decisions (kind, mint, pool, m_hat, size_sol, reason, detail) VALUES ('kalshi_maker_post', 'KXTEST-UI-A', 'yes', 0.051, 4.1, 'edge over line',
                        %s::jsonb)""", (json.dumps({"book": book}),))
    try:
        s = console.topic("kalshi_trades.settled", book=book).json()
        assert s["n"] >= 1 and any(r["ticker"] == "KXTEST-UI-A" and r["market"] == "Will the UI test pass?" and r["pnl"] == 5.9 and r["result"] == "yes" for r in s["rows"])
        row = next(r for r in s["rows"] if r["ticker"] == "KXTEST-UI-A")
        assert row["ret"] == pytest.approx(590 / 410 * 100)
        o = console.topic("kalshi_trades.orders", book=book).json()
        assert o["resting"] >= 1 and any(r["type"] == "post-only good till canceled" for r in o["rows"])
        dec = console.topic("kalshi_trades.decisions", book=book, kinds=["Maker posted"]).json()["rows"]
        assert any(r["action"] == "Maker posted" and r["market"] == "KXTEST-UI-A" and r["edge"] == pytest.approx(0.051) for r in dec)
        assert not any(r["market"] == "KXTEST-UI-A" for r in console.topic("kalshi_trades.decisions", book=book, kinds=["Settled"]).json()["rows"])
        pos = console.topic("kalshi_overview.positions").json()
        mine = [r for r in pos["rows"] if r["ticker"] == "KXTEST-UI-A"]
        assert mine and mine[0]["book"] == "Maker · paper" and mine[0]["paid"] == 60 and 4.9 < mine[0]["closes_in_h"] <= 5
        assert ["—", 1] in pos["categories"] or any(c[0] == "—" for c in pos["categories"])
        arm = next(a for a in console.topic("kalshi_overview.summary").json()["arms"] if a["book"] == book)
        assert arm["settled"] >= 1 and arm["wins"] >= 1
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM kalshi_positions WHERE ticker = 'KXTEST-UI-A'"); conn.execute("DELETE FROM kalshi_orders WHERE ticker = 'KXTEST-UI-A'")
            conn.execute("DELETE FROM decisions WHERE mint = 'KXTEST-UI-A'"); conn.execute("DELETE FROM kalshi_markets WHERE ticker = 'KXTEST-UI-A'")


# ---------------------------------------------------------------- the ts fix (§9 bug 1)
def test_stack_and_bootstrap_read_their_saved_time_from_brain_snapshots(console, kalshi_on, tmp_path):
    from fly_trader.kalshi import fly as KF, selector as kselector
    sel_file, fly_file = tmp_path / "kalshi_sel.joblib", tmp_path / "kalshi_fly.pt"
    sel_file.write_bytes(b"x"); fly_file.write_bytes(b"x")
    sel_note = {"data": kselector.DATA_VERSION, "deployable": True, "deploy_reason": "beat random", "first_day": "2025-01-01", "last_day": "2026-10-01", "rows": 123456,
                "walk_forward": {"n": 900, "mean": 0.031, "win": 0.55, "pf": 1.4, "days_positive": 40, "days": 60}, "random_baseline": {"mean": -0.02},
                "components": [{"name": "favorite", "passed": True, "reason": "ok", "evaluation": {"win": 0.6, "pf": 1.5, "n": 300}, "trials": 12}],
                "strategies": {"favorite": {"hold_min": "taker", "line": 0.02, "thr": {"p": 0.8}, "evaluation": {"n": 300, "win": 0.6, "pf": 1.5}}}}
    fly_note = {"data": KF.KALSHI_FLY_VERSION, "photoreceptors": 4096, "neurons": 50000, "diagnostics": {"mbon_share": 0.4, "mbon_saturated": 0.05, "slope": 0.9},
                "gates_ok": True, "calibration": {"trades": 500, "mean": 0.01}}
    with transaction() as conn:
        sid = conn.execute("INSERT INTO brain_snapshots (path, kind, note, ts) VALUES (%s, 'kalshi_selector', %s, now() - interval '2 hours') RETURNING id",
                           (str(sel_file), json.dumps(sel_note))).fetchone()["id"]
        fid = conn.execute("INSERT INTO brain_snapshots (path, kind, note, ts) VALUES (%s, 'kalshi_fly_selector', %s, now() - interval '1 hour') RETURNING id",
                           (str(fly_file), json.dumps(fly_note))).fetchone()["id"]
    try:
        st = console.topic("kalshi_model.stack").json()
        assert st["selector"]["id"] == sid and st["selector"]["ts"] and st["selector"]["rows"] == 123456 and st["selector"]["deployable"] is True
        assert st["wf"]["n"] == 900 and st["random_mean"] == -0.02 and st["components"][0]["win"] == pytest.approx(60) and st["strategies"][0]["arm"] == "taker"
        vf = console.topic("kalshi_model.visual_fly").json()
        assert vf["bootstrap"]["id"] == fid and vf["bootstrap"]["ts"] and vf["bootstrap"]["may_trade"] is True
        assert vf["bootstrap"]["text"].startswith("4,096 photoreceptors carry the inputs into 50,000 neurons; the mushroom body carries 40% of the score") and "May trade: " in vf["bootstrap"]["text"]
        rt = console.topic("kalshi_model.retraining").json()
        assert rt["selector_id"] == sid and rt["bootstrap_id"] == fid and rt["may_trade"] is True
        hist = {m["id"]: m for m in console.topic("kalshi_model.history").json()["models"]}
        assert hist[sid]["type"] == "stack" and hist[sid]["data"] == "current" and hist[sid]["per_trade"] == pytest.approx(3.1) and hist[sid]["put_to_work"] is True
        assert hist[fid]["type"] == "visual fly bootstrap" and hist[fid]["trades"] == 500 and hist[fid]["put_to_work"] is True
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM brain_snapshots WHERE id = ANY(%s)", ([sid, fid],))


def test_stack_survives_latest_current_without_ts_or_meta(console, kalshi_on, monkeypatch):
    """latest_current returns id, path and note (plus meta today): the topic must read ts itself and parse a bare note."""
    from fly_trader.kalshi import selector as kselector
    with transaction() as conn:
        sid = conn.execute("INSERT INTO brain_snapshots (path, kind, note) VALUES ('/nonexistent', 'kalshi_selector', '{}') RETURNING id").fetchone()["id"]
    note = json.dumps({"deployable": True, "deploy_reason": "ok", "walk_forward": {"n": 5}})
    monkeypatch.setattr(kselector, "latest_current", lambda conn: {"id": sid, "path": "/nonexistent", "note": note})
    try:
        st = console.topic("kalshi_model.stack").json()
        assert st["selector"]["id"] == sid and st["selector"]["ts"] and st["wf"]["n"] == 5
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM brain_snapshots WHERE id = %s", (sid,))


# ---------------------------------------------------------------- actions
def test_retrain_requests_a_run_and_starts_the_trainer(console, kalshi_on, calls):
    r = console.act("kalshi_model.retrain")
    assert r.status_code == 200 and "trainer started" in r.json()["message"]
    assert calls == [("request_run", False, None)] and console.sup.calls == [("start", "kalshi_train", "console")]
    calls.clear(); console.sup.calls.clear()
    r = console.act("kalshi_model.retrain")                                  # alive but idle: request only
    assert r.status_code == 200 and calls == [("request_run", False, None)] and console.sup.calls == []


def test_retrain_is_refused_while_training(console, kalshi_on, calls):
    console.sup.alive_set.add("kalshi_train"); calls.state["value"] = {"stage": "training the Kalshi selector"}
    r = console.act("kalshi_model.retrain")
    assert r.status_code == 409 and "already running" in r.json()["error"] and calls == []


def test_retrain_reports_a_trainer_that_will_not_start(console, kalshi_on, calls, monkeypatch):
    def refuse(name, started_by="console"):
        raise RuntimeError("kalshi_train is already running as an external process; stop it first")
    monkeypatch.setattr(console.sup, "start", refuse)
    r = console.act("kalshi_model.retrain")
    assert r.status_code == 409 and "external process" in r.json()["error"] and calls == [("request_run", False, None)]


def test_train_stop_only_while_training(console, kalshi_on, calls):
    assert console.act("kalshi_model.train_stop").status_code == 409 and console.sup.calls == []
    console.sup.alive_set.add("kalshi_train"); calls.state["value"] = {"stage": "bootstrapping the Kalshi fly (the selector teaches it once)"}
    r = console.act("kalshi_model.train_stop")
    assert r.status_code == 200 and console.sup.calls == [("request_stop", "kalshi_train")]


def test_fly_commands_need_a_running_visual_fly(console, kalshi_on, calls, monkeypatch):
    from fly_trader.ui.api import kalshi_model
    assert console.act("kalshi_model.fly_command", {"cmd": "explode"}).status_code == 400
    monkeypatch.setattr(kalshi_model, "setting", lambda key: ({"stage": "not trading"}, None) if key == "kalshi_fly_status" else ({}, None))
    r = console.act("kalshi_model.fly_command", {"cmd": "pause"})
    assert r.status_code == 409 and "not running" in r.json()["error"] and calls == []
    monkeypatch.setattr(kalshi_model, "setting", lambda key: ({"stage": "trading"}, None) if key == "kalshi_fly_status" else ({}, None))
    assert console.act("kalshi_model.fly_command", {"cmd": "pause"}).status_code == 409 and calls == []      # a stale status with the engine stopped
    console.sup.alive_set.add("kalshi_runner")
    for cmd in ("pause", "resume", "rollback"):
        r = console.act("kalshi_model.fly_command", {"cmd": cmd})
        assert r.status_code == 200 and r.json()["message"] == "Sent: applied at the next minute"
    assert calls == [("send_command", "pause"), ("send_command", "resume"), ("send_command", "rollback")]


def test_rebootstrap_requests_a_fly_and_starts_the_trainer(console, kalshi_on, calls):
    r = console.act("kalshi_model.fly_rebootstrap")
    assert r.status_code == 200 and r.json()["message"].startswith("Re-bootstrap requested")
    assert calls == [("request_run", True, "console")] and console.sup.calls == [("start", "kalshi_train", "console")]
    calls.clear(); console.sup.calls.clear()
    assert console.act("kalshi_model.fly_rebootstrap").status_code == 200 and calls == [("request_run", True, "console")] and console.sup.calls == []


def test_reset_is_refused_while_the_kalshi_engine_runs(console, kalshi_on, calls):
    console.sup.alive_set.add("kalshi_runner")
    r = console.act("kalshi_model.fly_reset")
    assert r.status_code == 409 and "Stop the Kalshi engine first" in r.json()["error"] and calls == []
    console.sup.alive_set.discard("kalshi_runner")
    r = console.act("kalshi_model.fly_reset")
    assert r.status_code == 200 and r.json()["message"] == "Archived to /tmp/reset_kalshi_fly_x" and calls == [("reset_kalshi_fly", "console button")]


# ---------------------------------------------------------------- navigation (the Streamlit app.py sections, now in Layout.tsx)
def test_navigation_lists_the_four_kalshi_pages_under_prediction_markets():
    src = (ROOT / "console" / "src" / "components" / "Layout.tsx").read_text()
    groups = re.findall(r"\{ label: '([^']+)', icon: '[^']+',(?: when: [^,]+,)? items: \[(.*?)\] \}", src, re.S)
    assert [g[0] for g in groups] == ["Memecoins", "Prediction markets", "System"]
    pm = dict(groups)["Prediction markets"]
    assert re.findall(r"to: '([^']+)', label: '([^']+)'", pm) == [("/kalshi", "Kalshi overview"), ("/kalshi-trades", "Kalshi trades"),
                                                                  ("/kalshi-model", "Kalshi model & training"), ("/kalshi-data", "Kalshi data")]
    assert "when: (m) => m.kalshi" in src.split("'Prediction markets'")[1].split("items")[0]
    app = (ROOT / "console" / "src" / "App.tsx").read_text()
    for path, page in (("/kalshi", "KalshiOverview"), ("/kalshi-trades", "KalshiTrades"), ("/kalshi-model", "KalshiModel"), ("/kalshi-data", "KalshiData")):
        assert f'<Route path="{path}" element={{<{page} />}} />' in app
        assert (ROOT / "console" / "src" / "pages" / f"{page}.tsx").read_text().count("export default function") == 1
