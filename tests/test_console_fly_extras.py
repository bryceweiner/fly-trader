"""The console's fly extras (fly_trader/ui/api/{inspector,ticker,notifications,search,logs}.py): the decision inspector, the
live ticker behind the live-money toasts, the notifications bell, global search and the Logs page — over the API, against
rows seeded in the test database (each test removes what it added)."""
import json
import uuid
from datetime import datetime, timedelta, timezone

import pytest

from fly_trader import config
from fly_trader.db.connection import transaction


def _mint(tag: str) -> str:
    return (f"Zq{tag}" + uuid.uuid4().hex)[:44]


@pytest.fixture
def seeded():
    """Rows to delete afterwards: {table: [(column, value), ...]}."""
    added: list[tuple[str, str, object]] = []
    yield added
    with transaction() as conn:
        for table, col, val in reversed(added):
            conn.execute(f"DELETE FROM {table} WHERE {col} = %s", (val,))


def _ins(conn, added, table: str, cols: dict, key: str = "id"):
    names = ", ".join(cols); ph = ", ".join(["%s"] * len(cols))
    r = conn.execute(f"INSERT INTO {table} ({names}) VALUES ({ph}) RETURNING {key}",
                     [json.dumps(v) if isinstance(v, dict) else v for v in cols.values()]).fetchone()
    added.append((table, key, r[key]))
    return r[key]


# ---------------------------------------------------------------- the inspector
def test_inspector_token_on_seeded_rows(console, seeded, monkeypatch):
    monkeypatch.setattr(config, "FLY_SCORE", "teacher"); monkeypatch.setattr(config, "FLY_TEACHER_GATE", True); monkeypatch.setattr(config, "FLY_LEARNED_VETO", 0.02)
    mint = _mint("ins"); t0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        _ins(conn, seeded, "tokens", {"mint": mint, "symbol": "INSPX"}, key="mint")
        # five minutes on 'ev': the fly clears its line (0.10) at minutes 2-4, its teacher would buy minutes 3-4; three resolved
        for i, (sc, fr, lab, st, ta) in enumerate([(0.05, 0.05, -0.02, "resolved", False), (0.08, 0.079, 0.01, "resolved", False),
                                                    (0.12, 0.121, 0.06, "resolved", False), (0.15, 0.149, None, "pending", True), (0.11, 0.14, None, "pending", True)]):
            conn.execute("INSERT INTO fly_scored (ts, mint, strategy, hold_min, score, frozen_score, line, frozen_line, label, state, teacher_allow, chain, bootstrap_id) "
                         "VALUES (%s,%s,'ev',120,%s,%s,0.10,0.10,%s,%s,%s,'sol',157)", (t0 + timedelta(minutes=i), mint, sc, fr, lab, st, ta))
        seeded.append(("fly_scored", "mint", mint))
        _ins(conn, seeded, "decisions", {"ts": t0, "mint": mint, "kind": "fly_enter", "m_hat": 0.149, "size_sol": 0.2, "reason": "ev: score 0.149 ≥ 0.100",
                                          "book_targets": ["paper_fly"], "detail": {"book": "paper_fly", "strategy": "ev", "threshold": 0.1}})
        _ins(conn, seeded, "decisions", {"ts": t0, "mint": mint, "kind": "blocked", "m_hat": 0.12, "size_sol": 0, "rail": "selector: below line",
                                          "reason": "fly pick not taken", "detail": {"book": "paper_fly"}})
        _ins(conn, seeded, "decisions", {"ts": t0, "mint": mint, "kind": "selector_enter", "m_hat": 0.149, "size_sol": 0.2, "book_targets": ["paper_selector"],
                                          "detail": {"book": "paper_selector"}})
        _ins(conn, seeded, "decisions", {"ts": t0, "mint": mint, "kind": "fly_rh_enter", "size_sol": 0.01, "detail": {"book": "paper_rh_fly"}})   # another chain's book
        _ins(conn, seeded, "positions", {"book": "paper_fly", "mint": mint, "opened_at": t0, "closed_at": t0 + timedelta(hours=2), "cost_sol": 0.2, "entry_price": 1e-6,
                                          "exit_price": 1.1e-6, "realized_sol": 0.02, "status": "closed"})
        _ins(conn, seeded, "fills", {"book": "paper_fly", "ts": t0, "mint": mint, "side": "buy", "sol_delta_lamports": -200_000_000, "price_sol": 1e-6,
                                      "fee_lamports": 5000, "verified_by": "model", "signature": None})
    r = console.topic("inspector.token", chain="sol", mint=mint)
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["label"].startswith("INSPX") and d["unit"] == "SOL" and d["explorer"].endswith(mint)
    t = d["total"]
    assert t["n"] == 5 and t["pending"] == 2 and t["resolved"] == 3 and t["cleared"] == 3 and t["teacher_allowed"] == 2 and t["teacher_known"] == 5
    assert t["label_cleared"] == pytest.approx(0.06, abs=1e-6) and t["label_not"] == pytest.approx(-0.005, abs=1e-6)
    assert [s["strategy"] for s in d["stats"]] == [None, "ev"]
    assert [x["ts"] for x in d["series"]] == sorted(x["ts"] for x in d["series"]) and len(d["series"]) == 5
    assert set(d["series"][0]) == {"ts", "strategy", "score", "frozen_score", "line", "label", "state", "teacher_allow"}
    assert len(d["latest"]) == 1 and d["latest"][0]["teacher_allow"] is True
    # the newest minute: teacher would buy, but learning moved the score 0.03 below the teacher's (> the 0.02 veto)
    assert "Score now vs line" in d["explain"] and "teacher's (the selector's) score" in d["explain"] and "veto" in d["explain"]
    kinds = {x["kind"]: x for x in d["decisions"]}
    assert set(kinds) == {"fly_enter", "blocked", "selector_enter"}                 # the RH book's decision belongs to the other chain
    assert kinds["fly_enter"]["action"] == "bought" and kinds["fly_enter"]["actor"] == "fly" and kinds["selector_enter"]["actor"] == "selector"
    assert kinds["blocked"]["action"] == "not taken"
    (p,) = d["positions"]
    assert p["return"] == pytest.approx(0.1) and p["held_min"] == pytest.approx(120)
    assert d["pnl"] == {"closed": 1, "realized": pytest.approx(0.02), "open": 0, "wins": 1}
    (f,) = d["fills"]
    assert f["amount"] == pytest.approx(-0.2) and f["source"] == "simulated" and f["tx_url"] is None
    assert isinstance(d["pathways"]["items"], list) and "note" in d["pathways"]


def test_inspector_token_shows_the_scored_minutes_price_and_volume_against_the_five_before(console, seeded):
    mint = _mint("bar"); t0 = datetime(2026, 10, 2, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        conn.execute("INSERT INTO fly_scored (ts, mint, strategy, hold_min, score, frozen_score, line, frozen_line, state, chain, bootstrap_id) "
                     "VALUES (%s, %s, 'ev', 120, 0.1, 0.1, 0.1, 0.1, 'pending', 'sol', 157)", (t0, mint))
        seeded.append(("fly_scored", "mint", mint))
        # the scored minute (t0) and the six before it, t0-3 without trades; the newer bar (t0+1) is not the scored minute
        for k, (close, buy, sell) in {-6: (0.5e-6, 9.0, 9.0), -5: (1.0e-6, 1.0, 1.0), -4: (1.1e-6, 2.0, 0.0), -2: (1.2e-6, 1.0, 1.0), -1: (1.25e-6, 0.5, 0.5),
                                      0: (1.3e-6, 4.0, 2.0), 1: (2.0e-6, 50.0, 50.0)}.items():
            conn.execute("INSERT INTO pump_minutes (mint, ts, close, buy_sol, sell_sol) VALUES (%s, %s, %s, %s, %s)", (mint, t0 + timedelta(minutes=k), close, buy, sell))
        seeded.append(("pump_minutes", "mint", mint))
    m = console.topic("inspector.token", chain="sol", mint=mint).json()["minute"]
    assert datetime.fromisoformat(m["ts"]) == t0 and m["price"] == pytest.approx(1.3e-6) and m["volume"] == pytest.approx(6.0)
    assert m["price_5m"] == pytest.approx(1.0e-6)                          # the close five minutes before the scored minute
    assert m["volume_5m"] == pytest.approx((2.0 + 2.0 + 2.0 + 1.0) / 5)    # t0-5 .. t0-1, the quiet minute t0-3 counted as none


def test_inspector_token_refuses_bad_params(console):
    assert console.topic("inspector.token", chain="eth", mint="A" * 40).status_code == 400
    assert console.topic("inspector.token", chain="sol", mint="../../etc/passwd").status_code == 400
    assert console.topic("inspector.token", chain="sol").status_code == 400
    assert console.topic("inspector.token", chain="sol", mint="A" * 40, mode="week").status_code == 400


def test_inspector_token_unscored_explains_why(console):
    d = console.topic("inspector.token", chain="sol", mint=_mint("none")).json()
    assert d["series"] == [] and d["total"]["n"] == 0 and d["decisions"] == [] and "has not scored this token" in d["explain"] and d["minute"] is None


def test_inspector_market_shape(console, seeded):
    d = console.topic("inspector.market", ticker="KXNOPE-26OCT01-T1").json()
    assert d["market"] is None and d["series"] == [] and d["decisions"] == d["orders"] == d["positions"] == d["fills"] == []
    assert d["total"]["n"] == 0 and "has not scored this market" in d["explain"] and set(d["books"]) >= {"paper_kalshi_taker", "paper_kalshi_maker"}
    assert console.topic("inspector.market", ticker="a b; drop").status_code == 400
    tk = f"KXINSP-{uuid.uuid4().hex[:8].upper()}"; t0 = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
    with transaction() as conn:
        _ins(conn, seeded, "kalshi_markets", {"ticker": tk, "title": "Will the fly test pass?", "status": "active", "close_time": t0 + timedelta(days=2)}, key="ticker")
        conn.execute("INSERT INTO kalshi_fly_scored (ts, ticker, side, strategy, score, frozen_score, line, frozen_line, state) VALUES (%s,%s,'yes','fav',0.05,0.04,0.03,0.03,'pending')",
                     (t0, tk))
        seeded.append(("kalshi_fly_scored", "ticker", tk))
        _ins(conn, seeded, "decisions", {"ts": t0, "mint": tk, "pool": "yes", "kind": "kalshi_taker_enter", "m_hat": 0.05, "size_sol": 3.0,
                                          "book_targets": ["paper_kalshi_taker"], "detail": {"book": "paper_kalshi_taker"}})
        _ins(conn, seeded, "kalshi_positions", {"book": "paper_kalshi_taker", "ticker": tk, "side": "yes", "contracts": 10, "cost_cents": 300, "fee_cents": 7, "opened_at": t0})
    d = console.topic("inspector.market", ticker=tk).json()
    assert d["market"]["title"] == "Will the fly test pass?" and d["label"] == "Will the fly test pass?"
    assert d["total"]["n"] == 1 and d["total"]["cleared"] == 1 and d["latest"][0]["side"] == "yes"
    assert d["decisions"][0]["action"] == "bought" and d["decisions"][0]["actor"] == "visual fly" and len(d["positions"]) == 1
    assert "edge" in d["explain"] and "above the line" in d["explain"]


# ---------------------------------------------------------------- the ticker
def test_ticker_ids_unique_stable_and_live_flags(console, seeded, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", "test-key")
    mint = _mint("tick"); now = datetime.now(timezone.utc) + timedelta(days=3650)          # newer than anything else in the test database
    with transaction() as conn:
        d_enter = _ins(conn, seeded, "decisions", {"ts": now, "mint": mint, "kind": "fly_enter", "m_hat": 0.2, "size_sol": 0.1, "reason": "ev: score 0.2 ≥ 0.1",
                                                    "book_targets": ["paper_fly"], "detail": {"book": "paper_fly", "strategy": "ev"}})
        d_live = _ins(conn, seeded, "decisions", {"ts": now + timedelta(seconds=1), "mint": mint, "kind": "fly_exit", "size_sol": 0.1, "reason": "held 120 min",
                                                   "detail": {"book": "live"}})
        d_sel = _ins(conn, seeded, "decisions", {"ts": now, "mint": mint, "kind": "selector_enter", "size_sol": 0.1, "detail": {"book": "paper_selector"}})
        f_paper = _ins(conn, seeded, "fills", {"book": "paper_fly", "ts": now, "mint": mint, "side": "buy", "sol_delta_lamports": -100_000_000, "verified_by": "model"})
        f_live = _ins(conn, seeded, "fills", {"book": "live", "ts": now + timedelta(seconds=2), "mint": mint, "side": "sell", "sol_delta_lamports": 110_000_000,
                                               "verified_by": "balance_delta", "signature": "LiveSig" + uuid.uuid4().hex})
        k_dec = _ins(conn, seeded, "decisions", {"ts": now, "mint": "KXTICK-1", "pool": "no", "kind": "kalshi_taker_enter", "m_hat": 0.04, "size_sol": 2.5,
                                                  "detail": {"book": "paper_kalshi_taker"}})
        trade = "trade-" + uuid.uuid4().hex
        _ins(conn, seeded, "kalshi_fills", {"trade_id": trade, "ticker": "KXTICK-1", "side": "yes", "price_cents": 40, "count": 5, "ts": now, "book": None}, key="trade_id")
    a = console.topic("ticker.recent").json(); b = console.topic("ticker.recent").json()
    ids = [x["id"] for x in a["items"]]
    assert len(ids) == len(set(ids)) and ids == [x["id"] for x in b["items"]]           # unique, and the same on every read
    by = {x["id"]: x for x in a["items"]}
    assert f"d:{d_sel}" not in by                                                         # the selector's book is not the fly's
    assert by[f"d:{d_enter}"]["live"] is False and by[f"d:{d_enter}"]["action"] == "bought" and by[f"d:{d_enter}"]["token"].startswith(mint[:6])
    assert by[f"d:{d_live}"]["live"] is True and by[f"d:{d_live}"]["side"] == "sell"
    assert by[f"f:{f_paper}"]["live"] is False and by[f"f:{f_live}"]["live"] is True and by[f"f:{f_live}"]["size"] == pytest.approx(0.11)
    assert by[f"d:{k_dec}"]["chain"] == "kalshi" and by[f"d:{k_dec}"]["live"] is False and by[f"d:{k_dec}"]["ticker"] == "KXTICK-1"
    assert by[f"k:{trade}"]["live"] is True
    assert ids[0] == f"f:{f_live}"                                                         # newest first
    assert "kalshi" in a["chains"]


# ---------------------------------------------------------------- notifications
def test_notifications_warnings_errors_circuit_and_live_fills(console, seeded):
    mint = _mint("note"); src = "test-" + uuid.uuid4().hex[:6]
    with transaction() as conn:
        e_info = _ins(conn, seeded, "events", {"level": "info", "source": src, "message": "all fine"})
        e_warn = _ins(conn, seeded, "events", {"level": "warning", "source": src, "message": "feed lagging"})
        e_err = _ins(conn, seeded, "events", {"level": "error", "source": src, "message": "stats refresh failed"})
        e_rails = _ins(conn, seeded, "events", {"level": "error", "source": "rails", "message": "kill switch tripped: entries blocked"})
        c_kill = _ins(conn, seeded, "circuit_events", {"kind": "kill_switch", "detail": {"wealth": 4.0, "peak": 5.0, "circuit": 1}})
        c_reset = _ins(conn, seeded, "circuit_events", {"kind": "reset", "detail": {"kill": True, "circuit": 3}})
        f_live = _ins(conn, seeded, "fills", {"book": "live", "mint": mint, "side": "buy", "sol_delta_lamports": -50_000_000, "verified_by": "balance_delta"})
        f_paper = _ins(conn, seeded, "fills", {"book": "paper_fly", "mint": mint, "side": "buy", "sol_delta_lamports": -50_000_000, "verified_by": "model"})
    d = console.topic("notifications.recent").json()
    by = {x["id"]: x for x in d["items"]}
    assert by[f"e:{e_warn}"]["level"] == "warning" and by[f"e:{e_err}"]["level"] == "error" and by[f"e:{e_err}"]["source"] == src
    assert f"e:{e_info}" not in by and f"e:{e_rails}" not in by                            # info never; the rails' own come as circuit events
    assert by[f"c:{c_kill}"]["level"] == "error" and "Solana kill switch tripped" in by[f"c:{c_kill}"]["text"]
    assert by[f"c:{c_reset}"]["level"] == "info" and "Robinhood Chain circuit reset" in by[f"c:{c_reset}"]["text"]
    assert by[f"f:{f_live}"]["level"] == "live" and by[f"f:{f_live}"]["mint"] == mint and f"f:{f_paper}" not in by
    ts = [datetime.fromisoformat(x["ts"]) for x in d["items"]]
    assert ts == sorted(ts, reverse=True)


# ---------------------------------------------------------------- search
def test_search_finds_symbol_and_signature(console, seeded, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", "")
    mint = _mint("srch"); sym = "ZQ" + uuid.uuid4().hex[:6].upper(); sig = "SigZq" + uuid.uuid4().hex + "x" * 20
    with transaction() as conn:
        _ins(conn, seeded, "tokens", {"mint": mint, "symbol": sym}, key="mint")
        _ins(conn, seeded, "fills", {"book": "paper_rh_fly", "mint": mint, "side": "sell", "sol_delta_lamports": 1, "verified_by": "model", "signature": sig})
    d = console.topic("search.query", q=sym.lower()).json()
    assert {"kind": "token", "chain": "sol", "mint": mint} in [r["target"] for r in d["results"]]
    assert any(r["label"] == sym for r in d["results"])
    d = console.topic("search.query", q=sig[:16]).json()
    tx = [r for r in d["results"] if r["kind"] == "tx"]
    assert tx and tx[0]["target"] == {"kind": "token", "chain": "rh", "mint": mint} and "paper_rh_fly" in tx[0]["sub"]
    d = console.topic("search.query", q=mint).json()                                       # a full mint
    assert d["results"][0]["target"]["mint"] == mint


def test_search_short_and_long_queries_are_graceful(console):
    for q in ("", "a", " b "):
        r = console.topic("search.query", q=q)
        assert r.status_code == 200 and r.json()["results"] == [] and "at least 2" in r.json()["note"]
    r = console.topic("search.query", q="x" * 500)
    assert r.status_code == 200 and r.json()["results"] == []
    r = console.topic("search.query", q="%_\\'")                                           # LIKE wildcards and quotes are data
    assert r.status_code == 200


# ---------------------------------------------------------------- logs
def test_logs_refuse_unknown_names_and_traversal(console, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    (tmp_path / "secret.log").write_text("do not read\n")
    for name in ("../../etc/passwd", "secret", "runner.log", "", "/etc/passwd", "runner/../secret"):
        r = console.topic("logs.tail", name=name)
        assert r.status_code == 400 and "no log named" in r.json()["error"], name
    assert console.topic("logs.tail", name=["runner"]).status_code == 400
    assert console.topic("logs.tail", name="runner", n="lots").status_code == 400


def test_logs_tail_reads_the_workers_file(console, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "LOG_DIR", tmp_path)
    lines = [json.dumps({"ts": f"2026-10-16T10:00:0{i}", "level": lvl, "logger": "fly_trader.agent.runner", "thread": "runner", "msg": f"line {i}"})
             for i, lvl in enumerate(["INFO", "WARNING", "ERROR"])]
    (tmp_path / "runner.log").write_text("\n".join(lines) + "\n")
    d = console.topic("logs.tail", name="runner", n=100).json()
    assert d["exists"] is True and d["lines"] == lines and d["n"] == 100 and d["size"] > 0
    assert console.topic("logs.tail", name="runner", n=1).json()["n"] == 10                 # clamped
    d = console.topic("logs.tail", name="vault").json()
    assert d["exists"] is False and d["lines"] == []
    (tmp_path / "console.log").write_text(lines[0] + "\n")
    assert console.topic("logs.tail", name="console").json()["lines"] == [lines[0]]
    files = {f["name"]: f for f in console.topic("logs.files").json()["files"]}
    from fly_trader.ops.supervisor import WORKERS
    assert set(WORKERS) | {"console"} <= set(files) and files["runner"]["exists"] and not files["vault"]["exists"]
