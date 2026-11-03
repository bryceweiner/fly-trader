"""The console's Trades and Model & training pages (fly_trader/ui/api/{trades,model}.py): each chain's books are offered in
order before their first trade and every view reads them, the closed-trade / decision-log / fills topics return their
rows on a seeded book, and every Model & training control calls what it should — and refuses what its disabled button
refused (a reset under a running engine, a fly command to a fly that is not running, a retrain during a run)."""
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config, markets
from fly_trader.db.connection import transaction
from fly_trader.ui.api import model, trades


# ---------------------------------------------------------------- Trades
@pytest.fixture
def seeded():
    """A Solana book of its own (replay_* books count as Solana's) with closed and open positions, decisions and fills,
    plus one real fill on the RH live book; removed afterwards."""
    book = f"replay_console_{uuid.uuid4().hex[:8]}"
    t0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
    sig_sol, sig_rh = "5" * 88, "0x" + "ab" * 32
    with transaction() as conn:
        pos = [conn.execute("INSERT INTO positions (book, mint, qty, cost_sol, realized_sol, status, opened_at, closed_at, forced_exit_kind, strategy) "
                            "VALUES (%s, %s, 1, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                            (book, mint, cost, pnl, status, t0 + timedelta(days=i), (t0 + timedelta(days=i, minutes=120)) if status == "closed" else None, kind, strat)).fetchone()["id"]
               for i, (mint, cost, pnl, status, kind, strat) in enumerate([
                   ("MintAAAAAAAA", 1.0, 0.10, "closed", None, "ev"), ("MintBBBBBBBB", 1.0, -0.05, "closed", "stop", "ev"),
                   ("MintCCCCCCCC", 0.5, 0.02, "closed", None, "capitulation"), ("MintDDDDDDDD", 1.0, None, "open", None, "ev")])]
        dec = [conn.execute("INSERT INTO decisions (ts, kind, mint, m_hat, size_sol, rail, reason, book_targets, detail) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id",
                            (t0 + timedelta(minutes=i), kind, mint, mh, size, rail, reason, targets, json.dumps(detail))).fetchone()["id"]
               for i, (kind, mint, mh, size, rail, reason, targets, detail) in enumerate([
                   ("selector_enter", "MintAAAAAAAA", 0.123, 0.5, None, "cleared the line", None, {"book": book}),
                   ("fly_exit", "MintAAAAAAAA", None, 0.5, None, "held 120 min", [book], {}),
                   ("blocked", "MintBBBBBBBB", 0.2, 0.0, "kill", "switch tripped", None, {"book": book}),
                   ("beat", None, None, None, None, None, None, {"book": book}),                       # not a kind the log shows
                   ("selector_enter", "MintZZZZZZZZ", 0.3, 0.5, None, "another book", None, {"book": book + "_other"})])]
        fil = [conn.execute("INSERT INTO fills (book, ts, side, mint, sol_delta_lamports, price_sol, fee_lamports, verified_by, signature) "
                            "VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s) RETURNING id", row).fetchone()["id"]
               for row in [(book, t0, "buy", "MintAAAAAAAA", -500_000_000, 1.5e-7, 5000, "model", None),
                           (book, t0 + timedelta(minutes=1), "sell", "MintAAAAAAAA", 550_000_000, 1.6e-7, 5000, "helius", sig_sol),
                           ("live_rh", t0, "sell", "0xabc", 22_000_000, 1.3e-7, 2000, "receipt", sig_rh)]]
    yield {"book": book, "sig_sol": sig_sol, "sig_rh": sig_rh}
    with transaction() as conn:
        conn.execute("DELETE FROM positions WHERE id = ANY(%s)", (pos,))
        conn.execute("DELETE FROM decisions WHERE id = ANY(%s)", (dec,))
        conn.execute("DELETE FROM fills WHERE id = ANY(%s)", (fil,))


def _get(console, name, **params):
    r = console.topic(name, **params)
    assert r.status_code == 200, r.text
    return r.json()


def test_trades_lists_each_chains_books_and_every_view_reads_them(console, monkeypatch):
    """Port of test_rh_ui: the chain's books come first, in mk.books order, even before their first trade; the selector
    book is the default; every view (closed, decision log, fills) answers for each of them."""
    monkeypatch.setattr(config, "RH_ENABLED", True)
    for m in (markets.SOL, markets.RH):
        b = _get(console, "trades.books", chain=m.chain)
        assert b["books"][:3] == list(m.books) and b["default"] == m.selector_book and b["unit"] == m.unit
        assert b["help"] == f"{m.selector_book} is the selector's paper book, {m.fly_book} the fly's, {m.live_book} the bot wallet; amounts in {m.unit}."
        assert all(markets.for_book(x) is m for x in b["books"])
        for book in m.books:
            for name in ("trades.summary", "trades.closed", "trades.decisions", "trades.fills"):
                d = _get(console, name, chain=m.chain, book=book)
                assert d["book"] == book and d["unit"] == m.unit and isinstance(d["metrics"] if name == "trades.summary" else d["rows"], dict if name == "trades.summary" else list)


def test_a_book_of_another_chain_falls_back_to_the_selector_book(console, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    assert _get(console, "trades.closed", chain="rh", book="paper_selector")["book"] == "paper_rh_selector"
    assert _get(console, "trades.fills", chain="sol", book="live_rh")["book"] == "paper_selector"
    assert _get(console, "trades.decisions", chain="sol", book=None)["book"] == "paper_selector"
    monkeypatch.setattr(config, "RH_ENABLED", False)                      # RH off: chain=rh is Solana
    d = _get(console, "trades.closed", chain="rh", book="paper_rh_fly")
    assert d["chain"] == "sol" and d["book"] == "paper_selector" and d["unit"] == "SOL"


def test_seen_books_follow_their_chain(console, monkeypatch, seeded):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    sol = _get(console, "trades.books", chain="sol")["books"]
    assert seeded["book"] in sol[3:] and seeded["book"] not in _get(console, "trades.books", chain="rh")["books"]


def test_closed_trades_rows(console, seeded):
    d = _get(console, "trades.closed", chain="sol", book=seeded["book"])
    assert [r["mint"] for r in d["rows"]] == ["MintCCCCCCCC", "MintBBBBBBBB", "MintAAAAAAAA"]       # newest close first; the open one is not a trade yet
    c = d["rows"][0]
    assert c["token"] == "MintCC…" and c["held_min"] == pytest.approx(120) and c["cost"] == 0.5 and c["ret"] == pytest.approx(4.0) and c["exit"] == "time"
    assert c["strategy"] == "capitulation" and d["rows"][1]["exit"] == "stop"


def test_book_summary(console, seeded):
    """The KPI widgets and charts: metrics, open positions, cumulative and daily P&L, trades by strategy and by exit."""
    d = _get(console, "trades.summary", chain="sol", book=seeded["book"])
    m = d["metrics"]
    assert m["n"] == 3 and m["wins"] == 2 and m["win_rate"] == pytest.approx(2 / 3) and m["open"] == 1 and m["open_cost"] == 1.0
    assert m["pnl"] == pytest.approx(0.07) and m["avg_ret"] == pytest.approx((0.10 - 0.05 + 0.04) / 3)
    assert [round(v, 6) for _, v in d["cum"]] == [0.1, 0.05, 0.07]                                    # running total in close order
    assert [(x["day"], x["n"], round(x["pnl"], 6)) for x in d["daily"]] == [("2026-10-01", 1, 0.1), ("2026-10-02", 1, -0.05), ("2026-10-03", 1, 0.02)]
    assert {e["kind"]: e["n"] for e in d["exits"]} == {"time": 2, "stop": 1}
    assert [(x["strategy"], x["n"], x["wins"]) for x in d["strategies"]] == [("ev", 2, 1), ("capitulation", 1, 1)]


def test_summary_of_an_empty_book(console):
    d = _get(console, "trades.summary", chain="sol", book=f"replay_empty_{uuid.uuid4().hex[:6]}")
    assert d["metrics"]["n"] == 0 and d["metrics"]["win_rate"] is None and d["metrics"]["pnl"] is None and d["metrics"]["open"] == 0
    assert d["cum"] == [] and d["daily"] == [] and d["exits"] == [] and d["strategies"] == []
    assert _get(console, "trades.closed", chain="sol", book=d["book"])["rows"] == []


def test_long_series_are_thinned():
    pts = list(range(1000))
    out = trades._thin(pts)
    assert len(out) == 180 and out[0] == 0 and out[-1] == 999 and trades._thin(pts[:50]) == pts[:50]


def test_decision_log_show_pills(console, seeded):
    book = seeded["book"]
    d = _get(console, "trades.decisions", chain="sol", book=book)                           # default: every pill
    assert d["show"] == ["Bought", "Sold", "Not taken"]
    assert [(r["action"], r["mint"]) for r in d["rows"]] == [("Not taken", "MintBBBBBBBB"), ("Sold", "MintAAAAAAAA"), ("Bought", "MintAAAAAAAA")]
    blocked, sold, bought = d["rows"]
    assert blocked["why"] == "kill: switch tripped" and sold["why"] == "held 120 min" and sold["score"] is None    # book_targets counts as this book
    assert bought["score"] == pytest.approx(0.123, rel=1e-4) and bought["size"] == 0.5
    only = _get(console, "trades.decisions", chain="sol", book=book, show=["Not taken"])
    assert [r["kind"] for r in only["rows"]] == ["blocked"]
    assert _get(console, "trades.decisions", chain="sol", book=book, show=["Bought", "Sold"])["show"] == ["Bought", "Sold"]
    assert _get(console, "trades.decisions", chain="sol", book=book, show=[])["rows"] == []
    assert console.topic("trades.decisions", chain="sol", book=book, show="Bought").status_code == 400


def test_fills_mark_simulated_and_link_the_explorer(console, monkeypatch, seeded):
    rows = _get(console, "trades.fills", chain="sol", book=seeded["book"])["rows"]
    real, sim = rows                                                                          # newest first
    assert sim["source"] == "simulated" and sim["url"] is None and sim["amount"] == pytest.approx(-0.5) and sim["fee"] == pytest.approx(5e-6)
    assert real["source"] == "helius" and real["url"] == f"https://solscan.io/tx/{seeded['sig_sol']}" and real["price"] == pytest.approx(1.6e-7)
    monkeypatch.setattr(config, "RH_ENABLED", True)
    rh = [r for r in _get(console, "trades.fills", chain="rh", book="live_rh")["rows"] if r["signature"] == seeded["sig_rh"]]
    assert len(rh) == 1 and rh[0]["url"].endswith(f"/tx/{seeded['sig_rh']}") and "solscan" not in rh[0]["url"]


# ---------------------------------------------------------------- Model & training: actions
@pytest.fixture
def requests(monkeypatch):
    """pipeline.request_run calls, recorded instead of written."""
    from fly_trader.train import pipeline
    calls = []
    monkeypatch.setattr(pipeline, "request_run", lambda fly=False, reason=None: calls.append({"fly": fly, "reason": reason}))
    return calls


def _pipeline_stage(monkeypatch, stage):
    monkeypatch.setattr(model, "_pipeline", lambda: {"stage": stage} if stage else {})


def test_retrain_requests_a_run_and_starts_the_trainer(console, monkeypatch, requests):
    _pipeline_stage(monkeypatch, None)
    r = console.act("model.retrain")
    assert r.status_code == 200 and "started" in r.json()["message"]
    assert requests == [{"fly": False, "reason": None}] and console.sup.calls == [("start", "train", "console")]


def test_retrain_with_the_trainer_running_only_requests(console, monkeypatch, requests):
    _pipeline_stage(monkeypatch, "idle")
    console.sup.alive_set.add("train")
    assert console.act("model.retrain").status_code == 200
    assert requests == [{"fly": False, "reason": None}] and console.sup.calls == []


def test_retrain_refused_during_a_run(console, monkeypatch, requests):
    _pipeline_stage(monkeypatch, "training selector")
    console.sup.alive_set.add("train")
    r = console.act("model.retrain")
    assert r.status_code == 409 and r.json()["error"] == "A training run is already in progress." and requests == []


def test_retrain_shows_a_refused_start(console, monkeypatch, requests):
    _pipeline_stage(monkeypatch, None)

    def refuse(name, started_by="console"):
        raise RuntimeError("train already runs in pid 4242")
    monkeypatch.setattr(console.sup, "start", refuse)
    r = console.act("model.retrain")
    assert r.status_code == 409 and r.json()["error"] == "train already runs in pid 4242"


def test_train_stop(console, monkeypatch):
    _pipeline_stage(monkeypatch, "training selector")
    assert console.act("model.train_stop").status_code == 409                             # the trainer is not alive
    console.sup.alive_set.add("train")
    assert console.act("model.train_stop").status_code == 200 and console.sup.calls == [("request_stop", "train")]
    _pipeline_stage(monkeypatch, "idle")
    assert console.act("model.train_stop").status_code == 409                             # alive but no run: the button is not shown


@pytest.fixture
def sent(monkeypatch):
    from fly_trader.agent import fly_session
    out = []
    monkeypatch.setattr(fly_session, "send_command", out.append)
    return out


def test_fly_commands_go_to_a_running_fly(console, monkeypatch, sent):
    monkeypatch.setattr(model, "_fly_status", lambda: {"stage": "trading"})
    monkeypatch.setattr(model, "runs_elsewhere", lambda name: None)
    r = console.act("model.fly_command", {"cmd": "pause"})                                # a status left by a stopped engine is not a running fly
    assert r.status_code == 409 and "not running" in r.json()["error"] and sent == []
    console.sup.alive_set.add("runner")
    for cmd in ("pause", "resume", "rollback"):
        r = console.act("model.fly_command", {"cmd": cmd})
        assert r.status_code == 200 and r.json()["message"] == "Sent: applied at the next minute"
    assert sent == ["pause", "resume", "rollback"]
    assert console.act("model.fly_command", {"cmd": "explode"}).status_code == 400 and len(sent) == 3


@pytest.mark.parametrize("status", [{}, {"stage": "not trading", "detail": "engine stopped"}])
def test_fly_command_refused_when_the_fly_is_not_running(console, monkeypatch, sent, status):
    monkeypatch.setattr(model, "_fly_status", lambda: status)
    r = console.act("model.fly_command", {"cmd": "pause"})
    assert r.status_code == 409 and "not running" in r.json()["error"] and sent == []


def test_rebootstrap_requests_a_fly_and_starts_the_trainer(console, requests):
    r = console.act("model.fly_rebootstrap")
    assert r.status_code == 200 and r.json()["message"].startswith("Re-bootstrap requested")
    assert requests == [{"fly": True, "reason": "console"}] and console.sup.calls == [("start", "train", "console")]
    console.act("model.fly_rebootstrap")                                                  # now alive: only requests
    assert len(requests) == 2 and console.sup.calls == [("start", "train", "console")]


@pytest.fixture
def resets(monkeypatch):
    from fly_trader.ops import reset
    calls = []
    monkeypatch.setattr(reset, "reset_fly", lambda reason="operator", archive=True: calls.append(reason) or {"archived_to": "/tmp/arch/reset_fly_x", "rows": {"fills": 3}})
    return calls


def test_fly_reset_refused_while_the_engine_runs(console, resets):
    console.sup.alive_set.add("runner")
    r = console.act("model.fly_reset")
    assert r.status_code == 409 and r.json()["error"] == "Stop the trading engine first." and resets == []


def test_fly_reset_archives(console, resets):
    r = console.act("model.fly_reset")
    assert r.status_code == 200 and r.json()["message"] == "Archived to /tmp/arch/reset_fly_x" and r.json()["archived_to"] == "/tmp/arch/reset_fly_x"
    assert resets == ["console button"]


# ---------------------------------------------------------------- Model & training: topics
@pytest.fixture
def selector_at_work():
    """A current, deployable selector snapshot loaded by a running selector session (the newest of both), removed afterwards."""
    from fly_trader.train.selector import DATA_VERSION
    meta = {"data": DATA_VERSION, "first_day": "2026-04-19", "trained_through": "2026-10-02", "threshold": 0.05, "horizon_min": 120, "model": {"min_age_h": 6.0},
            "walk_forward": {"n": 400, "days": 20, "mean": 0.02, "win": 0.55, "pf": 1.4, "days_positive": 14}, "random_baseline": {"mean": -0.03},
            "selection": {"n": 300, "mean": 0.015, "win": 0.5, "pf": 1.2, "weeks": 9, "weeks_pos": 7},
            "holdout": {"n": 90, "mean": 0.01, "random_mean": -0.04, "days": ["2026-09-12", "2026-10-02"], "weeks": 3, "weeks_pos": 2},
            "chains": {"sol": {"n": 50, "mean": 0.012, "random_mean": -0.05}, "zz": {"n": 1}},
            "deployable": True, "deploy_reason": "its backtest made money", "fallback": True,
            "sizing": [{"lo": 0.0, "n": 200, "mean": 0.01, "win": 0.5, "kelly": 0.2}, {"lo": 0.02, "n": 200, "mean": 0.03, "win": 0.6, "kelly": 0.8}],
            "bankroll": {"sized": {"start_sol": 5, "final_sol": 7.5, "max_drawdown": 0.2}, "fixed": {"final_sol": 6.0, "max_drawdown": 0.1}},
            "components": [{"name": "inputs: rug", "passed": True, "reason": "improves ev", "selection": {"win": 0.5, "n": 10}, "evaluation": {"win": 0.6, "pf": None, "total": 2.5, "n": 12},
                            "params": {"groups": ["rug"]}, "trials": 30, "best_seen": {"win": 0.6, "pf": 1.5, "n": 10, "total": 3.0}}],
            "strategies": {"ev": {"line": 0.0115, "hold_min": 120, "thr": {}, "evaluation": {"n": 50, "win": 0.5, "pf": 1.2}}}}
    run_id = str(uuid.uuid4())
    with transaction() as conn:
        sid = conn.execute("INSERT INTO brain_snapshots (path, kind, note) VALUES ('/tmp/sel.pkl', 'selector', %s) RETURNING id", (json.dumps(meta),)).fetchone()["id"]
        conn.execute("INSERT INTO runs (run_id, kind, status, started_at, brain_snapshot_id, config) VALUES (%s, 'selector', 'running', now() + interval '1 day', %s, %s)",
                     (run_id, sid, json.dumps({"threshold": 0.06})))
    yield sid
    with transaction() as conn:
        conn.execute("DELETE FROM runs WHERE run_id = %s", (run_id,))
        conn.execute("DELETE FROM brain_snapshots WHERE id = %s", (sid,))


def test_model_at_work(console, selector_at_work):
    console.sup.alive_set.add("runner")
    d = _get(console, "model.loaded")
    m = d["model"]
    assert m["id"] == selector_at_work and m["current"] and m["threshold"] == 0.06 and m["horizon_min"] == 120 and m["min_age_h"] == 6.0  # the session's line wins
    assert m["wf"]["n"] == 400 and m["random_mean"] == -0.03 and m["backtest_line"].startswith("Backtest on 20 held-out days")
    assert [b["band"] for b in m["sizing"]["bands"]] == ["+0.000 and up", "+0.020 and up"]
    assert m["sizing"]["bands"][1]["bet"] == pytest.approx(min(config.KELLY_FRACTION * 0.8, config.MAX_POSITION_FRACTION) * 100)
    assert "7.50 SOL (worst drawdown 20%)" in m["sizing"]["bankroll"] and m["sizing"]["caption"].startswith("Each buy's size")
    st = m["stack"]
    assert st["fallback"] and st["components"][0]["win_eval"] == pytest.approx(60) and st["components"][0]["pf_eval"] is None
    assert st["components"][0]["best"] == "60% winners · PF 1.50 · 10 trades · total +3.0" and st["strategies"][0]["line"] == pytest.approx(1.15)
    assert [(h["name"], h["random_mean"]) for h in m["halves"]] == [("Selection", None), ("Evaluation", -0.03), ("Holdout", -0.04), ("Holdout · Solana", -0.05)]
    assert m["halves"][2]["days"] == ["2026-09-12", "2026-10-02"] and m["halves"][2]["weeks_pos"] == 2       # a part without a mean (zz) is left out
    assert d["latest"]["id"] >= selector_at_work and "deployable" in d["latest"]
    console.sup.alive_set.discard("runner")
    assert _get(console, "model.loaded")["model"] is None                                 # a stopped engine has no model at work


def test_saved_models_mark_the_one_in_use(console, selector_at_work):
    console.sup.alive_set.add("runner")
    row = next(r for r in _get(console, "model.history")["rows"] if r["id"] == selector_at_work)
    assert row["type"] == "selector" and row["data"] == "current" and row["trades"] == 400 and row["mean"] == pytest.approx(2.0)
    assert row["random"] == pytest.approx(-3.0) and row["pf"] == 1.4 and row["deployed"] is True and row["in_use"] is True


def test_retraining_status(console, monkeypatch):
    from fly_trader.train import pipeline
    vals = {pipeline.KEY: {"stage": "training selector", "last_run_at": "2026-10-01T00:00:00+00:00"},
            "training_status": {"stage": "training selector", "step": 5, "total": 10, "eta_s": 120}, markets.SOL.handover_key: {}}
    monkeypatch.setattr(model, "setting", lambda k: (vals.get(k, {}), None))
    console.sup.alive_set.add("train")
    d = _get(console, "model.retraining")
    assert d["running"] and d["status"] == "Training selector" and d["interval_days"] == pipeline.INTERVAL_DAYS and not d["handover"]
    assert d["progress"] == {"value": 0.5, "text": "training selector · about 2 min left"} and d["next_run"] == "as soon as the data is ready"
    vals[pipeline.KEY] = {"stage": "failed", "last_error": "disk full", "retry_after": None}; vals[markets.SOL.handover_key] = {"at": "x"}
    console.sup.alive_set.clear()
    d = _get(console, "model.retraining")
    assert not d["running"] and d["status"] == "Off" and d["next_run"] == "—" and d["progress"] is None and d["handover"]
    assert d["failed"] == "The last run failed: disk full. Retrying —."
    vals[pipeline.KEY] = {"stage": "waiting for data", "waiting": "3 days not built"}
    assert _get(console, "model.retraining")["waiting"] == "Waiting for the training data: 3 days not built. Checking again —."


def test_backtest_days_and_their_outdated_guard(console, monkeypatch):
    day = f"2099-{uuid.uuid4().int % 12 + 1:02d}-01"
    with transaction() as conn:
        eid = conn.execute("INSERT INTO events (level, source, message, detail) VALUES ('info', 'selector', %s, %s) RETURNING id",
                           (f"walk-forward {day}", json.dumps({"day": day, "half": "evaluation", "ic": 0.05, "n": 40, "mean": 0.012, "random_mean": None, "win": 0.6, "pf": 1.3}))).fetchone()["id"]
    try:
        monkeypatch.setattr(model, "latest_snapshot", lambda kind="selector", deployable=False: {"id": 1, "meta": {}})
        row = next(r for r in _get(console, "model.results")["rows"] if r["day"] == day)
        assert row["model"] == pytest.approx(1.2) and row["random"] is None and row["winners"] == pytest.approx(60) and row["half"] == "evaluation"
        monkeypatch.setattr(model, "latest_snapshot", lambda kind="selector", deployable=False: None)
        assert _get(console, "model.results") == {"hidden": True, "rows": []}
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM events WHERE id = %s", (eid,))


def test_plastic_fly_panel(console, monkeypatch):
    monkeypatch.setattr(model, "_fly_status", lambda: {"stage": "not trading", "detail": "no deployable bootstrap"})
    d = _get(console, "model.fly")
    assert not d["running"] and d["live"] is None and d["detail"] == "Not trading: no deployable bootstrap." and d["updates"] == []
    monkeypatch.setattr(model, "_fly_status", lambda: {})
    assert _get(console, "model.fly")["detail"] == "Not trading: the trading engine is stopped."
    status = {"stage": "trading", "line": 0.16, "frozen_line": 0.15, "drift": 0.01, "pending": 25, "device": "cpu", "learning_frozen": True,
              "lines": {"ev": 0.16}, "frozen_lines": {"ev": 0.15},
              "checks": {"ev": {"ic": -0.08, "drift": 0.0, "shadow": {"n_frozen": 0, "n_plastic": 0, "mean_frozen": None, "mean_plastic": None}, "triggers": ["ic"]}}}
    monkeypatch.setattr(model, "_fly_status", lambda: status)
    console.sup.alive_set.add("runner")
    d = _get(console, "model.fly")
    assert d["running"] and d["frozen"] and d["runner_alive"] and d["detail"] is None
    assert d["live"]["line"] == 0.16 and d["live"]["ic"] is None and d["live"]["per_strategy"][0] == {
        "strategy": "ev", "ic": -0.08, "drift": 0.0, "mean_plastic": None, "mean_frozen": None, "n_plastic": 0, "n_frozen": 0, "triggers": "ic", "line": 0.16, "frozen_line": 0.15}
    assert d["live"]["drift_max"] == 0.5
    status["checks"] = {"ic": 0.04, "shadow": {"mean_plastic": 0.02, "mean_frozen": 0.01}}            # one strategy: the checks at the top level
    live = _get(console, "model.fly")["live"]
    assert live["ic"] == 0.04 and live["shadow_plastic"] == 0.02 and live["per_strategy"] == []


def test_replay_verdict(console, monkeypatch):
    rp = {"passed": False, "reason": "lost money", "evaluation": {"n": 300, "mean": -0.01, "days_positive": 4, "days": 16}, "random": {"mean": -0.04},
          "frozen": {"mean": -0.02}, "alpha": 0.0001, "half_life_days": 1.0, "S": "2026-09-09", "selection_days": ["2026-09-09", "2026-09-23"], "evaluation_days": []}
    real = model.setting
    monkeypatch.setattr(model, "setting", lambda k: (rp, None) if k == "fly_replay" else real(k))
    monkeypatch.setattr(model, "_fly_status", lambda: {})
    r = _get(console, "model.fly")["replay"]
    assert not r["passed"] and r["reason"] == "lost money" and r["n"] == 300 and r["random_mean"] == -0.04 and r["frozen_mean"] == -0.02
    assert r["alpha"] == "0.0001" and r["half_life"] == "1 days"
    assert r["caption"].endswith("Configuration chosen on 2026-09-09 → 2026-09-23, judged on —.")


def test_trainer_log(console, monkeypatch):
    assert _get(console, "model.trainer_log") == {"text": "(no log yet)"}
    monkeypatch.setattr(console.sup, "log_tail", lambda name, n=60: f"{name} {n} lines")
    assert _get(console, "model.trainer_log") == {"text": "train 40 lines"}
