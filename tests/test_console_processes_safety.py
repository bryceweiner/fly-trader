"""The console's Processes and Safety & wallet pages (fly_trader/ui/api/processes.py, safety.py): worker start/stop and their
refusals, the autostart map, the rails controls on a real circuit_state row, resets refused while an engine runs, the
Kalshi key never shown and dormant Kalshi hidden, and real-money withdrawals: one send per dialog opening, one in flight per
chain, every guard re-checked on the server (the Streamlit AppTest of tests/test_withdraw.py, ported)."""
import json
import threading

import pytest
from solders.keypair import Keypair

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ui import runtime

LAM = config.LAMPORTS_PER_SOL
KEY = "KEYID-SECRET-7f3a9c21-test"


@pytest.fixture(autouse=True)
def _fresh(monkeypatch):
    """Each test starts with Kalshi dormant, RH off, empty caches and no withdrawal remembered."""
    from fly_trader.ui.api import safety
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", ""); monkeypatch.setattr(config, "RH_ENABLED", False)
    safety.wallet_summary.cache_clear(); safety.withdraw_limits.cache_clear()
    safety._WD_USED.clear(); safety._WD_INFLIGHT.clear()
    yield
    safety.wallet_summary.cache_clear(); safety.withdraw_limits.cache_clear()
    safety._WD_USED.clear(); safety._WD_INFLIGHT.clear()


def _max_id(table: str) -> int:
    with transaction() as conn:
        return int(conn.execute(f"SELECT COALESCE(max(id), 0) AS m FROM {table}").fetchone()["m"])


# ---------------------------------------------------------------- Processes
def test_worker_cards_are_grouped_as_the_menu_with_the_vault_and_without_dormant_kalshi(console, monkeypatch):
    d = console.topic("processes.workers").json()
    assert [g["title"] for g in d["groups"]] == ["Memecoins", "System"]
    assert [w["name"] for w in d["groups"][1]["workers"]] == ["vault"]
    vault = d["groups"][1]["workers"][0]
    assert vault["role_label"] == "runs the vault" and vault["state"] == "stopped" and vault["autostart"] in (True, False)
    assert not any(w["name"].startswith("kalshi") for g in d["groups"] for w in g["workers"])
    assert d["pid"] == console.sup.pid and d["observer"] is False and d["external"] == []
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", KEY)
    console.sup.alive_set.add("runner")
    d = console.topic("processes.workers").json()
    assert [g["title"] for g in d["groups"]] == ["Memecoins", "Prediction markets", "System"]
    runner = d["groups"][0]["workers"][0]
    assert runner["name"] == "runner" and runner["state"] == "running" and runner["alive"]
    assert KEY not in json.dumps(d) and KEY[:8] not in json.dumps(d)


def test_start_and_stop_go_through_the_supervisor_and_refuse_what_the_buttons_disable(console, monkeypatch):
    sup = console.sup
    r = console.act("processes.start", {"name": "replay"})
    assert r.status_code == 200 and sup.calls == [("start", "replay", "console")]
    r = console.act("processes.start", {"name": "replay"})                          # already running
    assert r.status_code == 409 and "already running" in r.json()["error"] and len(sup.calls) == 1
    r = console.act("processes.stop", {"name": "replay"})
    assert r.status_code == 200 and sup.calls[-1] == ("request_stop", "replay")
    r = console.act("processes.stop", {"name": "train"})                            # not running
    assert r.status_code == 409 and "not running" in r.json()["error"] and sup.calls[-1] == ("request_stop", "replay")
    assert console.act("processes.start", {"name": "nope"}).status_code == 400
    assert console.act("processes.start", {"name": "kalshi_runner"}).status_code == 400   # dormant Kalshi: no card, no control
    monkeypatch.setattr(sup, "external", lambda: {"discover": {"pid": 4242, "started_at": None}})
    r = console.act("processes.start", {"name": "discover"})
    assert r.status_code == 409 and "pid 4242" in r.json()["error"] and ("start", "discover", "console") not in sup.calls
    monkeypatch.setattr(sup, "external", dict)

    def boom(name, started_by="console"):
        raise RuntimeError("cannot register discover: UniqueViolation")
    monkeypatch.setattr(sup, "start", boom)
    r = console.act("processes.start", {"name": "discover"})
    assert r.status_code == 409 and "cannot register discover" in r.json()["error"]
    sup.alive_set.add("pumpstream")
    monkeypatch.setattr(sup, "status", lambda: {n: {"alive": n == "pumpstream", "started_at": None, "stopped_at": None, "error": None,
                                                    "stopping": n == "pumpstream"} for n in sup.workers})
    r = console.act("processes.stop", {"name": "pumpstream"})                       # already stopping
    assert r.status_code == 409 and "already stopping" in r.json()["error"]


def test_stop_external_refuses_without_one_and_calls_procs_stop(console, monkeypatch):
    from fly_trader.ops import procs
    stopped = []
    monkeypatch.setattr(procs, "stop", lambda name, grace_s=30.0: stopped.append(name) or True)
    r = console.act("processes.stop_external", {"name": "runner"})
    assert r.status_code == 409 and stopped == []
    monkeypatch.setattr(console.sup, "external", lambda: {"runner": {"pid": 777, "started_at": None}})
    d = console.topic("processes.workers").json()
    assert d["external"] == [{"name": "runner", "title": "Trading engine", "pid": 777, "started_at": None}]
    runner = d["groups"][0]["workers"][0]
    assert runner["external"] is True and runner["state"] == "external" and runner["ext_pid"] == 777
    assert d["counts"]["external"] == 1 and d["counts"]["running"] == 0 and d["counts"]["total"] == len([w for g in d["groups"] for w in g["workers"]])
    r = console.act("processes.stop_external", {"name": "runner"})
    assert r.status_code == 200 and stopped == ["runner"] and "777" in r.json()["message"]


def test_autostart_toggle_upserts_the_whole_map_and_records_it(console):
    with transaction() as conn:
        before = conn.execute("SELECT value FROM ui_settings WHERE key = 'autostart'").fetchone()
    ev0 = _max_id("events")
    try:
        with transaction() as conn:
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('autostart', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                         (json.dumps({"runner": True, "pumpstream": True}),))
        assert console.act("processes.autostart", {"name": "vault", "on": True}).status_code == 200
        assert console.act("processes.autostart", {"name": "runner", "on": False}).status_code == 200
        with transaction() as conn:
            val = conn.execute("SELECT value FROM ui_settings WHERE key = 'autostart'").fetchone()["value"]
            ev = conn.execute("SELECT detail FROM events WHERE id > %s AND source = 'console' AND message = 'autostart changed' ORDER BY id", (ev0,)).fetchall()
        assert val == {"runner": False, "pumpstream": True, "vault": True}
        assert [e["detail"] for e in ev] == [{"runner": True, "pumpstream": True, "vault": True}, val]
        d = console.topic("processes.workers").json()
        cards = {w["name"]: w for g in d["groups"] for w in g["workers"]}
        assert cards["vault"]["autostart"] and not cards["runner"]["autostart"] and cards["pumpstream"]["autostart"]
        assert console.act("processes.autostart", {"name": "vault", "on": "yes"}).status_code == 400
        assert console.act("processes.autostart", {"name": "kalshi_stream", "on": True}).status_code == 400   # dormant
    finally:
        with transaction() as conn:
            if before:
                conn.execute("UPDATE ui_settings SET value = %s WHERE key = 'autostart'", (json.dumps(before["value"]),))
            else:
                conn.execute("DELETE FROM ui_settings WHERE key = 'autostart'")
            conn.execute("DELETE FROM events WHERE id > %s AND source = 'console' AND message = 'autostart changed'", (ev0,))


def test_activity_shows_the_last_12_log_lines_newest_first(console, monkeypatch):
    lines = [json.dumps({"ts": f"2026-10-16T12:00:{i:02d}.000Z", "level": "INFO", "msg": f"line {i}"}) for i in range(20)] + ["", "not json at all"]
    monkeypatch.setattr(console.sup, "log_tail", lambda name, n=60: "\n".join(lines))
    d = console.topic("processes.activity", worker="runner").json()
    assert len(d["rows"]) == 12 and d["rows"][0] == {"time": "", "level": "", "what": "not json at all"}
    assert d["rows"][1] == {"time": "12:00:19", "level": "INFO", "what": "line 19"} and d["rows"][-1]["what"] == "line 9"
    assert console.topic("processes.activity", worker="evil").status_code == 400


def test_event_log_filters_by_level_and_falls_back_to_errors(console):
    ev0 = _max_id("events")
    with transaction() as conn:
        conn.execute("INSERT INTO events (level, source, message) VALUES ('warning', 'test_console_ps', 'a warning'), ('error', 'test_console_ps', 'an error')")
    try:
        def msgs(**p):
            return [r["message"] for r in console.topic("processes.events", **p).json()["rows"] if r["source"] == "test_console_ps"]
        assert msgs(levels=["warning", "error"]) == ["an error", "a warning"]
        assert msgs(levels=["warning"]) == ["a warning"]
        d = console.topic("processes.events", levels=[]).json()
        assert d["levels"] == ["error"] and "a warning" not in [r["message"] for r in d["rows"]]
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM events WHERE id > %s AND source = 'test_console_ps'", (ev0,))


def test_worker_counts_and_the_last_24_hours_of_events(console):
    console.sup.alive_set.update({"runner", "pumpstream"})
    c = console.topic("processes.workers").json()["counts"]
    assert c["running"] == 2 and c["trade_up"] == 2 and c["stopped"] == c["total"] - 2 and c["crashed"] == 0
    before = console.topic("processes.event_stats").json()
    ev0 = _max_id("events")
    with transaction() as conn:
        conn.execute("INSERT INTO events (level, source, message) VALUES ('warning', 'test_console_ps', 'w'), ('error', 'test_console_ps', 'e'), ('error', 'test_console_ps', 'e')")
    try:
        after = console.topic("processes.event_stats").json()
        assert len(after["hours"]) == 24 == len(after["error"]) == len(after["warning"])
        assert after["levels"]["error"] - before["levels"]["error"] >= 2 and after["levels"]["warning"] - before["levels"]["warning"] >= 1
        assert after["error"][-1] - before["error"][-1] >= 2 or after["error"][-2] >= 2                    # in the current hour's bucket
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM events WHERE id > %s AND source = 'test_console_ps'", (ev0,))


def test_live_drawdown_is_read_against_each_circuits_kill_limit(console, monkeypatch):
    monkeypatch.setattr(config, "KILL_SWITCH_DRAWDOWN", 0.3)
    d = {b["id"]: b for b in console.topic("safety.live_books").json()["books"]}
    assert list(d) == [1] and d[1]["book"] == "live" and d[1]["limit"] == 0.3
    with transaction() as conn:
        conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, wealth, peak, drawdown, n_open) VALUES "
                     "(-515151, 'live', '2100-01-02 00:00+00', 5.0, 5.0, 0.0, 1), (-515152, 'live', '2100-01-02 00:01+00', 4.4, 5.0, 0.12, 2)")
    try:
        b = console.topic("safety.live_books").json()["books"][0]
        assert b["wealth"] == 4.4 and b["peak"] == 5.0 and abs(b["drawdown"] - 0.12) < 1e-6 and b["n_open"] == 2
        assert [p[1] for p in b["series"]][-2:] == [5.0, 4.4] and b["note"] is None
        monkeypatch.setattr(config, "RH_ENABLED", True); monkeypatch.setattr(config, "KALSHI_API_KEY_ID", KEY)
        books = {x["id"]: x for x in console.topic("safety.live_books").json()["books"]}
        assert list(books) == [1, 3, 2] and books[3]["book"] == "live_rh" and books[2]["unit"] == "USD"
        assert books[2]["limit"] == config.KALSHI_KILL_SWITCH_DRAWDOWN
        monkeypatch.setattr(config, "VAULT_ENABLED", True)
        assert "NAV index" in console.topic("safety.live_books").json()["books"][0]["note"]
    finally:
        with transaction() as conn:
            conn.execute("DELETE FROM wealth_marks WHERE beat_id IN (-515151, -515152)")


def test_an_observer_refuses_every_worker_control_withdrawal_and_reset(console, monkeypatch):
    obs = runtime.Observer()
    monkeypatch.setattr(runtime, "sup", lambda: obs)
    for name, body in (("processes.start", {"name": "replay"}), ("processes.stop", {"name": "replay"}), ("processes.stop_external", {"name": "replay"}),
                       ("processes.autostart", {"name": "replay", "on": True}), ("safety.reset_training", {}), ("safety.entries", {"circuit": 1, "paused": True}),
                       ("safety.kill_clear", {"circuit": 1}), ("safety.circuit_reset", {"circuit": 1}), ("safety.clear_halt", {"circuit": 1, "book": "paper_selector"})):
        r = console.act(name, body)
        assert r.status_code == 409, name
    d = console.topic("processes.workers").json()
    assert d["observer"] is True
    assert console.topic("safety.rails").json()["observer"] is True and console.topic("safety.maintenance").json()["observer"] is True
    _withdraw_setup(monkeypatch, blockers=[])
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "observer-0001"})
    assert r.status_code == 409 and "only observes" in r.json()["error"]


# ---------------------------------------------------------------- rails
@pytest.fixture
def circuits():
    """circuit_state rows 1..3 and the halts as they were, restored after; the circuit events the test wrote, removed."""
    with transaction() as conn:
        rows = {r["id"]: dict(r) for r in conn.execute("SELECT * FROM circuit_state WHERE id IN (1, 2, 3)").fetchall()}
        for cid in (1, 2, 3):
            conn.execute("INSERT INTO circuit_state (id) VALUES (%s) ON CONFLICT DO NOTHING", (cid,))
            conn.execute("UPDATE circuit_state SET fail_count = 0, tripped = false, kill_switch = false, kill_reason = NULL, entries_paused = false WHERE id = %s", (cid,))
    ce0, ev0 = _max_id("circuit_events"), _max_id("events")

    def set_(cid, **kw):
        with transaction() as conn:
            conn.execute("UPDATE circuit_state SET " + ", ".join(f"{k} = %s" for k in kw) + " WHERE id = %s", (*kw.values(), cid))

    def get(cid):
        with transaction() as conn:
            return dict(conn.execute("SELECT * FROM circuit_state WHERE id = %s", (cid,)).fetchone())
    yield type("C", (), {"set": staticmethod(set_), "get": staticmethod(get)})
    with transaction() as conn:
        for cid, r in rows.items():
            conn.execute("UPDATE circuit_state SET fail_count = %s, tripped = %s, kill_switch = %s, kill_reason = %s, peak_wealth = %s, entries_paused = %s WHERE id = %s",
                         (r["fail_count"], r["tripped"], r["kill_switch"], r["kill_reason"], r["peak_wealth"], r["entries_paused"], cid))
        conn.execute("DELETE FROM book_state WHERE book IN ('paper_selector', 'paper_rh_fly', 'paper_kalshi_taker')")
        conn.execute("DELETE FROM circuit_events WHERE id > %s AND kind IN ('reset', 'paused', 'resumed', 'book_resumed')", (ce0,))
        conn.execute("DELETE FROM events WHERE id > %s AND source = 'rails'", (ev0,))


def test_kill_switch_clear_is_refused_until_it_trips_then_rebases(console, circuits):
    r = console.act("safety.kill_clear", {"circuit": 1})
    assert r.status_code == 409 and "not tripped" in r.json()["error"]
    circuits.set(1, kill_switch=True, kill_reason="wealth 3.0 SOL <= 0.70 x peak 5.0 SOL", peak_wealth=5.0, fail_count=2)
    d = console.topic("safety.rails").json()["circuits"][0]
    assert d["id"] == 1 and d["kill_switch"] and d["kill_reason"].startswith("wealth 3.0")
    r = console.act("safety.kill_clear", {"circuit": 1})
    assert r.status_code == 200 and r.json()["message"] == "Kill switch cleared"
    c = circuits.get(1)
    assert not c["kill_switch"] and c["kill_reason"] is None and c["peak_wealth"] is None and c["fail_count"] == 0
    assert console.act("safety.kill_clear", {"circuit": "1"}).status_code == 400


def test_circuit_reset_and_entries_pause_toggle(console, circuits):
    r = console.act("safety.circuit_reset", {"circuit": 1})
    assert r.status_code == 409 and "nothing to reset" in r.json()["error"]
    circuits.set(1, fail_count=2)
    assert console.act("safety.circuit_reset", {"circuit": 1}).status_code == 200 and circuits.get(1)["fail_count"] == 0
    assert console.act("safety.entries", {"circuit": 1, "paused": True}).status_code == 200 and circuits.get(1)["entries_paused"]
    assert console.topic("safety.rails").json()["circuits"][0]["entries_paused"] is True
    r = console.act("safety.entries", {"circuit": 1, "paused": True})                  # a stale page: nothing changes
    assert r.status_code == 409 and "already paused" in r.json()["error"]
    assert console.act("safety.entries", {"circuit": 1, "paused": False}).status_code == 200 and not circuits.get(1)["entries_paused"]
    assert console.act("safety.entries", {"circuit": 1}).status_code == 400


def test_circuits_and_halts_follow_what_the_install_trades(console, circuits, monkeypatch):
    with transaction() as conn:
        for b in ("paper_selector", "paper_rh_fly", "paper_kalshi_taker"):
            conn.execute("INSERT INTO book_state (book, halted, reason) VALUES (%s, true, 'wealth below peak') ON CONFLICT (book) DO UPDATE SET halted = true", (b,))
    d = console.topic("safety.rails").json()["circuits"]
    assert [c["id"] for c in d] == [1] and [b["book"] for b in d[0]["halted"]] == ["paper_selector"]
    for cid in (2, 3):                                                                # dormant Kalshi, RH off: not on the page, not controllable
        assert console.act("safety.entries", {"circuit": cid, "paused": True}).status_code == 400
    r = console.act("safety.clear_halt", {"circuit": 1, "book": "paper_kalshi_taker"})  # not a book of circuit 1
    assert r.status_code == 400
    monkeypatch.setattr(config, "RH_ENABLED", True); monkeypatch.setattr(config, "KALSHI_API_KEY_ID", KEY)
    d = {c["id"]: c for c in console.topic("safety.rails").json()["circuits"]}
    assert list(d) == [1, 3, 2] and d[3]["section"] == "Robinhood Chain memecoins" and d[2]["title"] == "Kalshi trading rails"
    assert [b["book"] for b in d[3]["halted"]] == ["paper_rh_fly"] and [b["book"] for b in d[2]["halted"]] == ["paper_kalshi_taker"]
    assert console.act("safety.clear_halt", {"circuit": 3, "book": "paper_selector"}).status_code == 400
    assert console.act("safety.clear_halt", {"circuit": 1, "book": "paper_selector"}).status_code == 200
    r = console.act("safety.clear_halt", {"circuit": 1, "book": "paper_selector"})
    assert r.status_code == 409 and "not halted" in r.json()["error"]
    assert console.act("safety.clear_halt", {"circuit": 2, "book": "paper_kalshi_taker"}).status_code == 200


# ---------------------------------------------------------------- maintenance
def test_resets_are_refused_while_an_engine_runs(console, monkeypatch):
    from fly_trader.ops import reset
    from fly_trader.ui.api import safety
    calls = []
    monkeypatch.setattr(reset, "reset_training_state", lambda reason="", archive=True: calls.append(("paper", reason)) or {"archived_to": "/tmp/a", "rows": {}})
    monkeypatch.setattr(reset, "reset_kalshi_fly", lambda reason="", archive=True: calls.append(("kalshi", reason)) or {"archived_to": "/tmp/k", "rows": {}})
    console.sup.alive_set.add("runner")
    assert console.topic("safety.maintenance").json()["runner_alive"] is True
    r = console.act("safety.reset_training")
    assert r.status_code == 409 and "Stop the trading engine first." == r.json()["error"] and calls == []
    console.sup.alive_set.discard("runner")
    monkeypatch.setattr(safety, "runs_elsewhere", lambda name: "it runs outside the console (pid 99)" if name == "runner" else None)
    r = console.act("safety.reset_training")                                              # a CLI runner, or another console's, counts too
    assert r.status_code == 409 and "pid 99" in r.json()["error"] and calls == []
    monkeypatch.setattr(safety, "runs_elsewhere", lambda name: None)
    r = console.act("safety.reset_training")
    assert r.status_code == 200 and r.json()["message"] == "Archived to /tmp/a" and calls == [("paper", "console button")]
    assert console.act("safety.reset_kalshi_fly").status_code == 409 and len(calls) == 1        # dormant Kalshi
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", KEY)
    console.sup.alive_set.add("kalshi_runner")
    r = console.act("safety.reset_kalshi_fly")
    assert r.status_code == 409 and "Kalshi engine" in r.json()["error"] and len(calls) == 1
    console.sup.alive_set.discard("kalshi_runner")
    assert console.act("safety.reset_kalshi_fly").status_code == 200 and calls[-1] == ("kalshi", "console button")
    with safety.no_reset():                                                            # an engine never starts under a reset
        r = console.act("processes.start", {"name": "runner"})
        assert r.status_code == 409 and "reset" in r.json()["error"]
        assert console.act("safety.reset_training").status_code == 409
    assert ("start", "runner", "console") not in console.sup.calls and len(calls) == 2


# ---------------------------------------------------------------- Kalshi: the key never shown, dormant Kalshi hidden
def _summary(chain, address, native, px):
    return {"chain": chain, "unit": "SOL" if chain == "sol" else "ETH", "address": address, "native": native, "error": None, "n_open": 0,
            "positions": 0.0, "marked_at": None, "price_usd": px, "price_at": None, "native_usd": native * px, "positions_usd": 0.0, "total_usd": native * px}


def test_the_kalshi_key_id_never_appears_in_any_topic(console, monkeypatch):
    from fly_trader.ops import wallets
    monkeypatch.setattr(config, "KALSHI_API_KEY_ID", KEY); monkeypatch.setenv("KALSHI_API_KEY_ID", KEY)
    monkeypatch.setattr(wallets, "summary", lambda chain: _summary(chain, None, 0.0, 100.0))
    monkeypatch.setattr(wallets, "address", lambda chain: None)
    acct = console.topic("safety.kalshi_account").json()
    assert acct["enabled"] is True and acct["key"] == "configured"
    assert "kalshi" in console.topic("safety.settings").json()["tables"]
    assert console.topic("safety.maintenance").json()["kalshi"] is True
    for name in ("safety.wallets", "safety.rails", "safety.settings", "safety.kalshi_account", "safety.maintenance", "safety.api_errors",
                 "processes.workers", "processes.events"):
        text = console.topic(name).text
        assert KEY not in text and KEY[:8] not in text and KEY[:6] not in text, name


def test_dormant_kalshi_leaves_no_trace_on_the_safety_page(console):
    assert console.topic("safety.kalshi_account").json() == {"enabled": False}
    assert list(console.topic("safety.settings").json()["tables"]) == ["sol"]
    assert [c["id"] for c in console.topic("safety.rails").json()["circuits"]] == [1]
    m = console.topic("safety.maintenance").json()
    assert m["kalshi"] is False and m["kalshi_runner_alive"] is False


# ---------------------------------------------------------------- withdrawals (tests/test_withdraw.py's page test, ported)
DEST = str(Keypair().pubkey())
LIM = {"balance": 2 * LAM, "fee": 5_000, "keep": 0, "n_open": 0, "max": 2 * LAM - 5_000, "to_balance": LAM}


def _withdraw_setup(monkeypatch, blockers=None, sent=None, result=None):
    from fly_trader.ops import wallets
    monkeypatch.setattr(config, "FUNDING_ADDRESSES", [DEST])
    if blockers is not None:
        monkeypatch.setattr(wallets, "withdraw_blockers", lambda chain: list(blockers))
    monkeypatch.setattr(wallets, "summary", lambda chain: _summary(chain, "So1" + "1" * 40, 2.0, 100.0))
    monkeypatch.setattr(wallets, "limits", lambda chain, dest: dict(LIM))
    sent = sent if sent is not None else []
    monkeypatch.setattr(wallets, "withdraw", lambda chain, amt, dest: sent.append((chain, amt, dest)) or (result or {"status": "confirmed", "tx": "sig1", "amount": amt, "to": dest}))
    return sent


def test_safety_shows_both_wallets_in_usd_and_withdraw_is_blocked_without_a_funding_address(console, monkeypatch):
    from fly_trader.ops import wallets
    from fly_trader.ui.state import usd
    sol_addr = str(Keypair().pubkey())
    fake = {"sol": _summary("sol", sol_addr, 2.0, 100.0), "rh": _summary("rh", "0x" + "ab" * 20, 0.5, 2600.0)}
    monkeypatch.setattr(config, "RH_ENABLED", True); monkeypatch.setattr(config, "FUNDING_ADDRESSES", [])
    monkeypatch.setattr(wallets, "summary", lambda chain: fake[chain])
    monkeypatch.setattr(wallets, "address", lambda chain: fake[chain]["address"])
    d = {c["chain"]: c for c in console.topic("safety.wallets").json()["chains"]}
    assert list(d) == ["sol", "rh"] and d["sol"]["name"] == "Solana" and d["rh"]["unit"] == "ETH"
    assert {usd(d["sol"]["summary"]["native_usd"]), usd(d["rh"]["summary"]["native_usd"])} == {"$200.00", "$1,300.00"}
    assert d["sol"]["explorer"] == f"https://solscan.io/account/{sol_addr}"
    assert any("FUNDING_ADDRESSES" in b for b in d["sol"]["blockers"])                 # nowhere to send: Withdraw disabled
    p = console.topic("safety.withdraw_plan", chain="sol").json()
    assert p["destinations"] == [] and p["to"] is None
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "dialog-000001"})
    assert r.status_code == 409 and "FUNDING_ADDRESSES" in r.json()["error"]


def test_the_dialog_plans_then_sends_exactly_once_per_opening(console, monkeypatch):
    sent = _withdraw_setup(monkeypatch, blockers=[])
    p = console.topic("safety.withdraw_plan", chain="sol").json()
    assert p["destinations"] == [DEST] and p["to"] == DEST and p["why"] == "enter an amount" and not p["show_why"] and p["button"] == "Withdraw"
    assert p["everything_label"] == "Everything withdrawable: 1.999995 SOL" and p["placeholder"] == "at most 1.999995"
    p = console.topic("safety.withdraw_plan", chain="sol", to=DEST, amount=0.5).json()
    assert p["why"] is None and p["button"] == "Withdraw 0.500000 SOL"
    assert p["summary"] == f"Sends **0.500000 SOL** ($50.00) to `{DEST[:6]}…{DEST[-4:]}`; fee ≈ 0.000005 SOL."
    p = console.topic("safety.withdraw_plan", chain="sol", to=DEST, amount=5.0).json()
    assert p["show_why"] and "at most 1.999995 SOL" in p["why"] and p["button"] == "Withdraw"
    assert console.topic("safety.withdraw_plan", chain="sol", to="NotMine111").status_code == 400
    assert sent == []                                                                  # planning never sends
    body = {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "dialog-000002"}
    r = console.act("safety.withdraw", body)
    assert r.status_code == 200 and sent == [("sol", 0.5, DEST)]
    res = r.json()["result"]
    assert res["status"] == "confirmed" and res["tx"] == "sig1" and res["tx_url"] == "https://solscan.io/tx/sig1" and res["amount"] == 0.5
    r = console.act("safety.withdraw", body)                                          # the same opening clicked again: no second send
    assert r.status_code == 409 and "already sent" in r.json()["error"] and sent == [("sol", 0.5, DEST)]
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "all": True, "opening": "dialog-000003"})
    assert r.status_code == 200 and sent[-1] == ("sol", None, DEST)                   # everything: wallets.withdraw(chain, None, to)


def test_every_guard_is_checked_again_on_the_server(console, monkeypatch):
    from fly_trader.ops import wallets
    sent = _withdraw_setup(monkeypatch, blockers=["no bot key in this process (BOT_PRIVATE_KEY)"])
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "guard-000001"})
    assert r.status_code == 409 and "no bot key" in r.json()["error"]
    monkeypatch.setattr(wallets, "withdraw_blockers", lambda chain: [])
    bad = [({"chain": "sol", "to": "NotMine111", "amount": 0.5}, 400), ({"chain": "sol", "to": DEST}, 400),            # no amount never means everything
           ({"chain": "sol", "to": DEST, "amount": 0.5, "all": True}, 400), ({"chain": "sol", "to": DEST, "amount": -1}, 400),
           ({"chain": "sol", "to": DEST, "amount": "0.5"}, 400), ({"chain": "rh", "to": DEST, "amount": 0.5}, 400),   # RH off
           ({"chain": "sol", "to": DEST, "amount": 0.5, "opening": "x"}, 400)]
    for i, (b, code) in enumerate(bad):
        r = console.act("safety.withdraw", {"opening": f"guard-bad-{i:04d}", **b})
        assert r.status_code == code, (b, r.json())
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 5.0, "opening": "guard-000002"})       # over the max: the fresh plan refuses
    assert r.status_code == 409 and "at most" in r.json()["error"]
    monkeypatch.setattr(wallets, "limits", lambda chain, dest: (_ for _ in ()).throw(TimeoutError("rpc down")))
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "guard-000003"})
    assert r.status_code == 409 and "Cannot read the wallet" in r.json()["error"]
    assert sent == []


def test_refusals_and_unknown_outcomes_become_results(console, monkeypatch):
    from fly_trader.chain import withdraw as sw
    from fly_trader.ops import wallets
    _withdraw_setup(monkeypatch, blockers=[])
    monkeypatch.setattr(wallets, "withdraw", lambda chain, amt, dest: (_ for _ in ()).throw(sw.WithdrawRefused("a swap is still in flight: try again in a minute")))
    res = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "result-0001"}).json()["result"]
    assert res["status"] == "refused" and "swap is still in flight" in res["error"]
    monkeypatch.setattr(wallets, "withdraw", lambda chain, amt, dest: (_ for _ in ()).throw(RuntimeError("socket closed")))
    res = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "result-0002"}).json()["result"]
    assert res["status"] == "error" and res["error"] == "RuntimeError: socket closed"
    monkeypatch.setattr(wallets, "withdraw", lambda chain, amt, dest: {"status": "pending", "tx": "sig9", "amount": amt, "to": dest})
    res = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "result-0003"}).json()["result"]
    assert res["status"] == "pending" and res["tx_url"].endswith("/sig9")


def test_one_withdrawal_per_chain_is_in_flight_at_a_time(console, monkeypatch):
    from fly_trader.ops import wallets
    from fly_trader.ui.api import safety
    _withdraw_setup(monkeypatch, blockers=[])
    inside, release, sent = threading.Event(), threading.Event(), []

    def slow(chain, amt, dest):
        sent.append((chain, amt, dest)); inside.set(); release.wait(10)
        return {"status": "confirmed", "tx": "sigA", "amount": amt, "to": dest}
    monkeypatch.setattr(wallets, "withdraw", slow)
    out = {}
    t = threading.Thread(target=lambda: out.update(first=safety.withdraw({"chain": "sol", "to": DEST, "amount": 0.5, "opening": "flight-0001"})))
    t.start()
    assert inside.wait(10)
    r = console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.25, "opening": "flight-0002"})       # a second dialog, meanwhile
    assert r.status_code == 409 and "already in flight" in r.json()["error"]
    release.set(); t.join(10)
    assert out["first"]["result"]["status"] == "confirmed" and sent == [("sol", 0.5, DEST)]
    assert console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.25, "opening": "flight-0003"}).status_code == 200
    assert len(sent) == 2


def test_a_withdrawal_drops_the_cached_balance(console, monkeypatch):
    from fly_trader.ops import wallets
    _withdraw_setup(monkeypatch, blockers=[])
    reads = []
    monkeypatch.setattr(wallets, "summary", lambda chain: reads.append(chain) or _summary(chain, "So1" + "1" * 40, 2.0, 100.0))
    monkeypatch.setattr(wallets, "address", lambda chain: "So1" + "1" * 40)
    console.topic("safety.wallets"); console.topic("safety.wallets")
    assert reads == ["sol"]                                                            # cached 15 s
    assert console.act("safety.withdraw", {"chain": "sol", "to": DEST, "amount": 0.5, "opening": "cache-00001"}).status_code == 200
    console.topic("safety.wallets")
    assert reads == ["sol", "sol"]
