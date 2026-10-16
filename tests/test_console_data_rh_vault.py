"""The console's Data pipelines, Robinhood Chain and $FLY vault pages (fly_trader/ui/api/{data,rh,vault}.py): every panel
answers on an empty database and on seeded rows, RH and the vault answer their off message when disabled, and the
vault's two writes (resume, reclassify) validate their input and refuse what the Streamlit page could not reach."""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from psycopg.types.json import Jsonb

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ui.api import data as data_api, rh as rh_api, vault as vault_api

LAYOUT = Path(__file__).resolve().parents[1] / "console" / "src" / "components" / "Layout.tsx"
SIG = "5" * 87                                  # base58, a Solana signature's length
VAULT_TABLES = ("vault_allocations", "vault_settlements", "vault_claims", "vault_flows", "vault_kv", "vault_nav", "vault_scan")


@contextlib.contextmanager
def setting(key: str, value):
    """ui_settings[key] = value (None: absent) for the block, then the row as it was."""
    with transaction() as c:
        old = c.execute("SELECT value, updated_at FROM ui_settings WHERE key = %s", (key,)).fetchone()
        c.execute("DELETE FROM ui_settings WHERE key = %s", (key,))
        if value is not None:
            c.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s)", (key, Jsonb(value)))
    try:
        yield
    finally:
        with transaction() as c:
            c.execute("DELETE FROM ui_settings WHERE key = %s", (key,))
            if old:
                c.execute("INSERT INTO ui_settings (key, value, updated_at) VALUES (%s, %s, %s)", (key, Jsonb(old["value"]), old["updated_at"]))


# ---------------------------------------------------------------- data pipelines
def _part(root: Path, day: str, version: int) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq
    d = root / "features_mature" / day
    d.mkdir(parents=True)
    pq.write_table(pa.table({"x": [1]}).replace_schema_metadata({"fly_version": str(version)}), d / "part.parquet")


def test_data_pipelines_on_an_empty_install(console, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "CORPUS_DIR", tmp_path)
    data_api.feature_parts.cache_clear(); data_api.token_stats_coverage.cache_clear()
    with setting("pumpstream_status", None), setting("replay_status", None):
        r = console.topic("data.pipelines")
    assert r.status_code == 200, r.text
    d = r.json()
    assert d["feed"]["badge"] == {"text": "stopped", "color": "red"} and d["feed"]["age"] is None and d["feed"]["updated_at"] is None
    assert isinstance(d["feed"]["per_minute"], list) and isinstance(d["feed"]["eps_spark"], list)
    assert d["archive"]["badge"]["text"] == "stopped" and d["archive"]["progress"] is None and d["archive"]["last_hour"] == "—" and d["archive"]["eta_h"] is None
    assert isinstance(d["archive"]["per_day"], list)
    assert d["features"] == {"n": 0, "first": None, "last": None, "current": 0}
    assert d["stats"]["badge"] == {"text": "stopped", "color": "gray"}                      # optional worker: idle, not alarming
    assert set(d["meta"]) == {"graduations", "creators", "launches"}
    cov = console.topic("data.token_stats").json()
    assert set(cov) == {"universe", "covered", "first"} and cov["covered"] <= cov["universe"]


def test_data_pipelines_states_progress_and_feature_versions(console, monkeypatch, tmp_path):
    from fly_trader.market.features import FEATURE_VERSION
    monkeypatch.setattr(config, "CORPUS_DIR", tmp_path)
    _part(tmp_path, "2026-05-01", FEATURE_VERSION); _part(tmp_path, "2026-05-02", FEATURE_VERSION - 1)
    data_api.feature_parts.cache_clear()
    now = datetime.now(timezone.utc)
    console.sup.alive_set |= {"pumpstream", "replay", "discover"}
    fresh = {"flushed_through": (now - timedelta(seconds=90)).isoformat(), "events_per_s": 812.4, "migrates": 7, "reconnects": 2, "dropped_band": 1234, "dropped_late": 5}
    replay = {"hours_done": 30, "hours_total": 120, "errors": 1, "last_hour": "2026-10-16T12:00:00+00:00", "eta_h": 4.25}
    with setting("pumpstream_status", fresh), setting("replay_status", replay):
        d = console.topic("data.pipelines").json()
    assert d["feed"]["badge"] == {"text": "running", "color": "green"} and 29 < d["feed"]["age"] < 40
    assert (d["feed"]["events_per_s"], d["feed"]["migrates"], d["feed"]["dropped_band"]) == (812.4, 7, 1234) and d["feed"]["updated_at"]
    assert d["archive"]["badge"]["text"] == "running" and d["archive"]["progress"] == 0.25 and d["archive"]["last_hour"] == "2026-10-16T12"
    assert d["archive"]["eta_h"] == 4.25 and d["archive"]["replay_start"] == config.REPLAY_START
    assert d["features"] == {"n": 2, "first": "2026-05-01", "last": "2026-05-02", "current": 1}
    assert d["stats"]["badge"]["text"] == "running"
    with transaction() as c:
        c.execute("INSERT INTO replay_days (day, graduations, tokens_written) VALUES ('1999-01-01', 7, 6) ON CONFLICT (day) DO NOTHING")
    try:
        days = console.topic("data.pipelines").json()["archive"]["per_day"]
    finally:
        with transaction() as c:
            c.execute("DELETE FROM replay_days WHERE day = '1999-01-01'")
    assert days == sorted(days) and len(days) <= data_api.ARCHIVE_DAYS
    stale = {**fresh, "flushed_through": (now - timedelta(minutes=10)).isoformat()}
    with setting("pumpstream_status", stale):
        assert console.topic("data.pipelines").json()["feed"]["badge"] == {"text": "stale", "color": "orange"}   # alive, but the newest minute is old


# ---------------------------------------------------------------- Robinhood Chain
RH_TOPICS = ("rh.indexer", "rh.index_activity", "rh.assets", "rh.verdict", "rh.wallet", "rh.activity")


def test_rh_page_is_a_memecoin_page():
    nav = LAYOUT.read_text()
    memecoins = nav[nav.index("label: 'Memecoins'"):nav.index("label: 'Prediction markets'")]
    assert "to: '/rh'" in memecoins


def test_rh_topics_answer_off_when_rh_is_disabled(console, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", False)
    for t in RH_TOPICS:
        assert console.topic(t).json() == {"enabled": False, "message": "Robinhood Chain is off (RH_ENABLED=0)."}, t
    assert console.get("/api/meta").json()["rh"] is False


def test_rh_topics_render_on_an_empty_ledger(console, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    with setting("rh_stream_status", None), setting("fly_replay", None), setting("handover_rh", None):
        for t in RH_TOPICS:
            r = console.topic(t)
            assert r.status_code == 200 and r.json()["enabled"] is True, (t, r.text)
        assert console.topic("rh.indexer").json() == {"enabled": True, "status": None}
        ia = console.topic("rh.index_activity").json()
        assert set(ia["counts"]) == {"launches", "graduated", "swaps", "minutes"} and len(ia["hours"]) == 24
        assert set(ia["hours"][-1]) == {"hour", "swaps", "minutes", "launches"}
        v = console.topic("rh.verdict").json()
        assert v["chains"] == [] and v["handover"] is None and v["live_money"] is False
    for view in rh_api.VIEWS:
        a = console.topic("rh.activity", view=view).json()
        assert a["view"] == view and isinstance(a["rows"], list) and a["tx_url"].endswith("/tx/")
    assert console.topic("rh.activity", view="Everything").status_code == 400


def test_rh_indexer_reads_the_stream_status(console, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    mt = 1_760_000_000
    with setting("rh_stream_status", {"mode": "backfill", "behind_blocks": 12345, "head": 900, "index_through": 800, "minutes_through": mt, "error": "rpc 429"}):
        d = console.topic("rh.indexer").json()
    s = d["status"]
    assert (s["mode"], s["behind_blocks"], s["head"], s["index_through"], s["error"]) == ("backfill", 12345, 900, 800, "rpc 429")
    assert s["minutes_through"] == datetime.fromtimestamp(mt, timezone.utc).strftime("%b %d %H:%M") and d["updated_at"]


def test_rh_wallet_panel(console, monkeypatch):
    """The ETH wallet: the pinned address with its explorer page, native ETH at the last reconciliation, live wealth, open
    positions, whether the books match the chain, and the open base lots (uint256 quantities kept exact)."""
    monkeypatch.setattr(config, "RH_ENABLED", True)
    monkeypatch.setattr(config, "RH_TESTNET", False)
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", "0x" + "ab" * 20)
    qty = "123456789012345678901234567890"
    with transaction() as c:
        c.execute("INSERT INTO rh_wallet_marks (ts, native_wei, consistent) VALUES ('2100-01-01', 1500000000000000000, false)")
        c.execute("INSERT INTO wealth_marks (beat_id, book, ts, wealth, n_open) VALUES (-424242, 'live_rh', '2100-01-01', 2.25, 3)")
        lot = c.execute("INSERT INTO rh_base_lots (asset, qty_raw, basis_eth, source) VALUES ('0xfeed', %s, 0.5, 'entry_leg') RETURNING id", (qty,)).fetchone()["id"]
    try:
        d = console.topic("rh.wallet").json()
    finally:
        with transaction() as c:
            c.execute("DELETE FROM rh_wallet_marks WHERE ts = '2100-01-01'")
            c.execute("DELETE FROM wealth_marks WHERE beat_id = -424242")
            c.execute("DELETE FROM rh_base_lots WHERE id = %s", (lot,))
    assert d["address"] == "0x" + "ab" * 20 and d["address_url"] == f"https://robinhoodchain.blockscout.com/address/0x{'ab' * 20}"
    assert d["reconciled"] and d["native_eth"] == 1.5 and d["consistent"] is False and d["marked_at"].startswith("2100-01-01")
    assert d["wealth"] == 2.25 and d["n_open"] == 3 and d["book"] == "live_rh" and d["spark"][-1] == 2.25
    mine = [x for x in d["lots"] if x["asset"] == "0xfeed"]
    assert mine and mine[0]["qty_raw"] == qty and mine[0]["source"] == "entry_leg"
    monkeypatch.setattr(config, "RH_BOT_ADDRESS", None)
    d = console.topic("rh.wallet").json()
    assert d["address"] is None and d["address_url"] is None


def test_rh_activity_lists_intents_transactions_and_legs(console, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    token = "0x" + "cd" * 20; tx_hash = "0x" + "ef" * 32
    with transaction() as c:
        i = c.execute("INSERT INTO rh_intents (book, kind, token, route, state, attempts) VALUES ('live_rh', 'buy', %s, 'atomic_kyber', 'done', 1) RETURNING id",
                      (token,)).fetchone()["id"]
        t = c.execute("INSERT INTO rh_txs (intent_id, kind, from_addr, nonce, hash, raw, status, fee_wei) VALUES (%s, 'swap', '0xme', 7, %s, 'raw', 'mined_ok', 21000000000000) "
                      "RETURNING id", (i, tx_hash)).fetchone()["id"]
        g = c.execute("INSERT INTO rh_legs (intent_id, tx_id, leg_no, asset_in, asset_out, amount_in_raw, amount_out_raw, eth_in, verified_by) "
                      "VALUES (%s, %s, 1, '0x00', %s, 1000000000000000000000000, 5, 1.0, 'receipt_logs') RETURNING id", (i, t, token)).fetchone()["id"]
    try:
        intents = console.topic("rh.activity", view="Intents").json()["rows"]
        txs = console.topic("rh.activity", view="Transactions").json()
        legs = console.topic("rh.activity", view="Legs").json()["rows"]
    finally:
        with transaction() as c:
            c.execute("DELETE FROM rh_legs WHERE id = %s", (g,)); c.execute("DELETE FROM rh_txs WHERE id = %s", (t,)); c.execute("DELETE FROM rh_intents WHERE id = %s", (i,))
    assert intents[0]["id"] == i and intents[0]["mint"] == token and intents[0]["token"] == token[:6] + "…" and intents[0]["route"] == "atomic_kyber"
    assert txs["rows"][0]["hash"] == tx_hash and txs["rows"][0]["fee_eth"] == pytest.approx(2.1e-5) and txs["tx_url"] == rh_api.explorer("tx")
    assert legs[0]["id"] == g and legs[0]["amount_in"] == "1000000000000000000000000" and legs[0]["verified_by"] == "receipt_logs"


def test_rh_verdict_and_seat(console, monkeypatch):
    monkeypatch.setattr(config, "RH_ENABLED", True)
    replay = {"chains": {"rh": {"passed": False, "reason": "too few trades", "evaluation": {"n": 12, "mean": -0.01}}, "sol": {"passed": True}}}
    at = datetime.now(timezone.utc).isoformat()
    with setting("fly_replay", replay), setting("handover_rh", {"at": at}):
        d = console.topic("rh.verdict").json()
    by = {c["chain"]: c for c in d["chains"]}
    assert by["rh"] == {"chain": "rh", "name": "Robinhood Chain", "passed": False, "evaluated": True, "n": 12, "mean": -0.01, "reason": "too few trades"}
    assert by["sol"]["name"] == "Solana" and by["sol"]["passed"] and not by["sol"]["evaluated"]
    assert d["handover"] == {"at": at} and d["live_money"] is False and d["replay_at"]          # the engine is not running: no live money


# ---------------------------------------------------------------- $FLY vault
@pytest.fixture
def vault_on(monkeypatch):
    monkeypatch.setattr(config, "VAULT_ENABLED", True)
    with transaction() as c:
        for t in VAULT_TABLES:
            c.execute(f"DELETE FROM {t}")
    yield
    with transaction() as c:
        for t in VAULT_TABLES:
            c.execute(f"DELETE FROM {t}")


def test_vault_answers_off_and_refuses_writes_when_disabled(console, monkeypatch):
    monkeypatch.setattr(config, "VAULT_ENABLED", False)
    for t in ("vault.summary", "vault.tables"):
        assert console.topic(t).json() == {"enabled": False, "message": vault_api.OFF}
    for a, body in (("vault.resume", {}), ("vault.reclassify", {"signature": SIG, "kind": "deposit"})):
        r = console.act(a, body)
        assert r.status_code == 409 and r.json()["error"] == vault_api.OFF


def test_vault_topics_on_an_empty_ledger(console, vault_on):
    s = console.topic("vault.summary").json()
    assert s["enabled"] and s["halted"] is None
    assert [s[k] for k in ("nav", "deposits", "withdrawn", "allocated", "paid", "owed")] == [0] * 6
    assert s["scan_slot"] is None and s["rh_block"] is None and s["impl"] is None and s["nav_spark"] == [] and s["profit"] == 0
    t = console.topic("vault.tables").json()
    assert t["settlements"] == t["claims"] == t["flows"] == t["holders"] == [] and t["tx_url"] == "https://solscan.io/tx/"


def test_vault_summary_and_tables_count_the_ledger(console, vault_on):
    L = 1_000_000_000
    with transaction() as c:
        c.execute("INSERT INTO vault_flows (signature, slot, direction, kind, counterparty, lamports, fee_lamports) VALUES "
                  "('d1', 1, 'in', 'deposit', 'F', %s, 0), ('w1', 2, 'out', 'withdrawal', 'F', %s, 5000), ('c1', 3, 'out', 'claim', 'H', %s, 0), "
                  "('p1', 4, 'in', 'profit', 'X', %s, 0)", (5 * L, L, L // 2, L // 4))
        c.execute("INSERT INTO vault_settlements (period_start, period_end, status, allocated, realized, pot) VALUES "
                  "('2026-10-05', '2026-10-12', 'allocated', %s, %s, %s), ('2026-10-12', '2026-10-19', 'snapshotted', %s, 0, 0)", (2 * L, 3 * L, 2 * L, 9 * L))
        c.execute("INSERT INTO vault_nav (ts, native, token_acct, positions_value, exit_cost, wealth) VALUES (now(), 1, 0, 0, 0, %s)", (6 * L,))
        c.execute("INSERT INTO vault_scan (name, cursor, slot) VALUES ('solana_wallet', 'sig', 4242)")
        c.execute("INSERT INTO vault_scan (name, slot, detail) VALUES ('rh_vault', 777, %s)", (Jsonb({"impl": "0xImpl"}),))
    s = console.topic("vault.summary").json()
    assert (s["nav"], s["deposits"], s["withdrawn"], s["paid"], s["profit"]) == (6 * L, 5 * L, L + 5000, L // 2, L // 4) and s["nav_spark"] == [6 * L]
    assert s["allocated"] == 2 * L and s["owed"] == 2 * L - L // 2                       # only allocated weeks count
    assert (s["scan_slot"], s["rh_block"], s["impl"]) == (4242, 777, "0xImpl") and s["rh_updated_at"]
    t = console.topic("vault.tables").json()
    assert [f["signature"] for f in t["flows"]] == ["p1", "c1", "w1", "d1"] and t["flows"][-1]["sol_amount"] == 5.0
    assert [x["status"] for x in t["settlements"]] == ["snapshotted", "allocated"] and t["settlements"][1]["allocated_sol"] == 2.0


def test_vault_resume_refuses_when_not_halted(console, vault_on):
    r = console.act("vault.resume", {})
    assert r.status_code == 409 and "not halted" in r.json()["error"]


def test_vault_resume_clears_a_halt_the_operator_saw(console, vault_on):
    from fly_trader.vault import state
    state.put("halt", {"reasons": ["unknown transaction signed by our key"], "detail": None})
    state.put("panic", {"at": 1, "by": "test"})
    s = console.topic("vault.summary").json()
    assert s["halted"] == {"reasons": ["unknown transaction signed by our key"]}
    r = console.act("vault.resume", {"reasons": ["something else"]})                       # the page showed other reasons
    assert r.status_code == 409 and "changed" in r.json()["error"] and state.halted()
    r = console.act("vault.resume", {"reasons": s["halted"]["reasons"]})
    assert r.status_code == 200 and "halt cleared" in r.json()["message"]
    assert state.halted() is None and state.get("panic") is None
    assert console.topic("vault.summary").json()["halted"] is None


def test_vault_reclassify_validates_and_calls_flows(console, vault_on, monkeypatch):
    calls = []

    def fake(sig, kind):
        calls.append((sig, kind))
        if sig == "4" * 87:
            raise LookupError("no flow with that signature")
        return 3
    monkeypatch.setattr(vault_api.flows, "reclassify", fake)
    for body in ({}, {"signature": "  ", "kind": "deposit"}, {"signature": SIG, "kind": "withdrawal"}, {"signature": SIG}, {"signature": "not a sig!", "kind": "profit"},
                 {"signature": "0OIl" * 22, "kind": "profit"}, {"signature": 123, "kind": "profit"}):
        r = console.act("vault.reclassify", body)
        assert r.status_code == 400, body
    assert calls == []
    r = console.act("vault.reclassify", {"signature": f"  {SIG}\n", "kind": "internal"})
    assert r.status_code == 200 and r.json()["message"] == "3 row(s) changed" and r.json()["changed"] == 3
    assert calls == [(SIG, "internal")]
    r = console.act("vault.reclassify", {"signature": "4" * 87, "kind": "deposit"})
    assert r.status_code == 409 and r.json()["error"] == "no flow with that signature"


def test_vault_reclassify_relabels_a_flow(console, vault_on):
    with transaction() as c:
        c.execute("INSERT INTO vault_flows (signature, slot, direction, kind, counterparty, lamports) VALUES (%s, 1, 'in', 'profit', 'NEW', 1000)", (SIG,))
    r = console.act("vault.reclassify", {"signature": SIG, "kind": "deposit"})
    assert r.status_code == 200 and r.json()["message"] == "1 row(s) changed"
    with transaction() as c:
        f = c.execute("SELECT kind, classified_by FROM vault_flows WHERE signature = %s", (SIG,)).fetchone()
    assert (f["kind"], f["classified_by"]) == ("deposit", "operator")
