"""The console's Overview page and the 3D brain (fly_trader/ui/api/{overview,brain}.py): the narrative says what the
Streamlit page said, the book's metrics, this minute's scoring, the race (hidden, racing, handed over) and the open
positions answer on either chain; wealth lines are thinned but keep their last mark; the brain topics keep the payload
contract (version 1 and the two-fly version 2 with /static/brain index URLs), and the playback endpoints serve a
minute's raw int8 activity and one neuron's day."""
from __future__ import annotations

import base64
import contextlib
import json
import re
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from psycopg.types.json import Jsonb

from fly_trader import config, markets
from fly_trader.brain import activity, geometry
from fly_trader.db.connection import transaction
from fly_trader.ui.api import brain, overview
from tests.test_fly_brain_view import _npz, _tsv

MINT_A = "FLYmintAAAA1111111111111111111111111111pump"
MINT_B = "FLYmintBBBB2222222222222222222222222222pump"


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


@contextlib.contextmanager
def quiet(*extra: str):
    """No fly, no selector blob, no training, no handover on either chain: the state a stopped install shows."""
    keys = ("fly_status", "fly_status_rh", "selector_status", "selector_status_rh", "training_status", "handover", "handover_rh") + extra
    with contextlib.ExitStack() as st:
        for k in keys:
            st.enter_context(setting(k, None))
        yield


@contextlib.contextmanager
def seeded(marks=(), positions=()):
    """Insert wealth marks ``(beat_id, book, ts, wealth, exposure, n_open, drawdown)`` and positions (dicts), remove them after."""
    ids = []
    with transaction() as c:
        for beat, book, ts, w, exp, n_open, dd in marks:
            c.execute("INSERT INTO wealth_marks (beat_id, book, ts, wealth, exposure, n_open, drawdown) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                      (beat, book, ts, w, exp, n_open, dd))
        for p in positions:
            cols = list(p)
            ids.append(c.execute(f"INSERT INTO positions ({', '.join(cols)}) VALUES ({', '.join(['%s'] * len(cols))}) RETURNING id",
                                 [p[k] for k in cols]).fetchone()["id"])
    try:
        yield ids
    finally:
        with transaction() as c:
            c.execute("DELETE FROM wealth_marks WHERE beat_id = ANY(%s)", ([m[0] for m in marks],))
            c.execute("DELETE FROM positions WHERE id = ANY(%s)", (ids,))


def _ok(r):
    assert r.status_code == 200, r.text
    return r.json()


# ---------------------------------------------------------------- the narrative
def _state(**kw) -> dict:
    s = {"model": {"id": 154, "run": {"threshold": 0.115, "horizon_min": 120},
                   "meta": {"trained_through": "2026-10-02", "horizon_min": 120, "model": {"min_age_h": 6.0, "min_resq_sol": 10.0, "min_vol_15m_sol": 5.0}}},
         "market": markets.SOL, "live_money": False, "trading_why": "Trading every minute.", "fly": {"stage": "trading", "line": 0.161}, "handover": None,
         "training": False, "train_status": {}}
    return {**s, **kw}


def test_narrative_says_what_the_streamlit_page_said(monkeypatch):
    monkeypatch.setattr(overview, "pinned_snapshot", lambda: None)
    paras = overview.narrative(_state()).split("\n\n")
    assert paras[0] == ("**Trading every minute.** The selector is paper trading with model **#154** (trained through 2026-10-02). Every minute it predicts the net "
                        "120-minute return of each PumpSwap token that traded, at least 6.0 hours past graduation, with a pool of at least 10.0 SOL and 5.0 SOL traded "
                        "in the last 15 minutes, and buys those predicted to make at least **+11.5%** — sized by how certain the model is (up to "
                        f"{config.MAX_POSITION_FRACTION:.0%} of the bankroll, never the {markets.SOL.gas_reserve():g} SOL gas reserve) — and sells it **120 minutes** "
                        "later. No real money is at risk.")
    assert "you pinned it" not in paras[1] and len(paras) == 3
    assert paras[2] == ("**The plastic fly** races the selector on its own paper book: it buys at a predicted +16.1% and learns every minute from each scored "
                        "token's realized return, two hours later.")
    monkeypatch.setattr(overview, "pinned_snapshot", lambda: 154)
    paras = overview.narrative(_state(live_money=True, handover={"at": "x"}, training=True, train_status={"stage": "training", "step": 3, "total": 10})).split("\n\n")
    assert paras[0].endswith("Trades spend real SOL from the bot wallet.")
    assert paras[1].endswith(" **It trades because you pinned it** (Model page); unpin it to let a current model take over.")
    assert paras[2].startswith("**The plastic fly** holds the seat: ") and paras[3] == "Training is running: training (3/10)."
    rh = overview.narrative(_state(market=markets.RH, fly={}, trading_why="The trading engine is stopped.")).split("\n\n")
    assert "graduated Pons pool on Robinhood Chain (prices in ETH" in rh[0] and f"a pool of at least {markets.RH.min_resq():.3g} ETH" in rh[0]
    assert rh[-1] == "**The trading engine is stopped.** Start the trading engine on the Processes page to trade."
    assert overview.narrative(_state(model=None, fly={"stage": "not trading"}, trading_why="X.")) == "**X.** Start the trading engine on the Processes page to trade."


# ---------------------------------------------------------------- summary, this minute, race, book
def test_summary_metrics_follow_the_selector_book(console):
    with quiet():
        base = _ok(console.topic("overview.summary", chain="sol"))
        marks = [(-81001, "paper_selector", "2100-01-01 00:00+00", 5.1, 0.1, 1, 0.0), (-81002, "paper_selector", "2100-01-01 00:01+00", 5.2, 0.2, 1, 0.0),
                 (-81003, "paper_selector", "2100-01-01 00:02+00", 5.3, 0.25, 2, 0.0123)]
        now = datetime.now(timezone.utc)
        pos = [{"book": "paper_selector", "mint": MINT_A, "status": "closed", "realized_sol": 0.2, "cost_sol": 0.5, "closed_at": now},
               {"book": "paper_selector", "mint": MINT_B, "status": "closed", "realized_sol": -0.05, "cost_sol": 0.5, "closed_at": now}]
        with seeded(marks, pos):
            d = _ok(console.topic("overview.summary", chain="sol"))
    cap = markets.SOL.capital()
    assert d["market"]["chain"] == "sol" and d["market"]["unit"] == "SOL" and d["capital"] == cap
    assert d["narrative"] == "**The trading engine is stopped.** Start the trading engine on the Processes page to trade."
    assert d["marked"] and d["wealth"] == 5.3 and d["exposure"] == 0.25 and d["n_open"] == 2 and d["drawdown"] == pytest.approx(0.0123)
    assert d["closed"] == base["closed"] + 2 and d["wins"] == base["wins"] + 1 and d["pnl"] == pytest.approx((base["pnl"] or 0) + 0.15)
    assert d["spark"][-1] == 5.3 and len(d["spark"]) <= 180 and len(d["dd_spark"]) == len(d["spark"])
    assert d["pnl_spark"][-2:] == pytest.approx([d["pnl"] + 0.05, d["pnl"]])            # cumulative, ending at the total (the −0.05 trade closed last)
    assert d["wealth_help"] == f"Started at {cap:g} SOL; open positions valued net of exit costs."


def test_this_minute_from_the_selector_blob(console):
    blob = {"minute": "2026-10-16T14:53:00+00:00", "mints_traded": 411, "eligible": 150, "score_max": 0.21, "threshold": 0.115, "picks": 1, "entered": 1, "exited": 2,
            "latency_s": 9.2, "top_scores": [[MINT_A, 0.21], [MINT_B, 0.05]]}
    with quiet():
        d = _ok(console.topic("overview.minute", chain="sol"))
        assert d["minute"] is None and d["runner"] is False and d["top"] == []
        console.sup.alive_set.add("runner")
        with transaction() as c:
            c.execute("INSERT INTO corpus_meta (mint, symbol) VALUES (%s, 'FLY') ON CONFLICT (mint) DO UPDATE SET symbol = 'FLY'", (MINT_A,))
        try:
            with setting("selector_status", blob):
                d = _ok(console.topic("overview.minute", chain="sol"))
        finally:
            with transaction() as c:
                c.execute("DELETE FROM corpus_meta WHERE mint = %s", (MINT_A,))
    assert d["runner"] is True and d["minute_hm"] == "14:53" and d["scored_at"]
    assert (d["mints_traded"], d["eligible"], d["score_max"], d["threshold"], d["picks"], d["entered"], d["exited"], d["latency_s"]) == (411, 150, 0.21, 0.115, 1, 1, 2, 9.2)
    assert d["top"] == [{"mint": MINT_A, "token": f"FLY · {MINT_A[:4]}…", "score": 0.21}, {"mint": MINT_B, "token": f"{MINT_B[:6]}…", "score": 0.05}]
    assert d["trading"] == "holding" and d["trading_why"].startswith("The market feed is stale")          # the feed thread is not running
    mk = markets.SOL
    assert d["eligible_help"] == f"Pool ≥ {mk.min_resq():.3g} SOL and ≥ {mk.min_vol_15m():.3g} SOL traded in the last 15 minutes."


def test_race_hidden_racing_and_handed_over(console, monkeypatch):
    with quiet():
        real = overview.marks
        monkeypatch.setattr(overview, "marks", lambda *a, **k: [])
        assert _ok(console.topic("overview.race", chain="sol")) == {"hidden": True, "unit": "SOL"}
        monkeypatch.setattr(overview, "marks", real)
        base = {m["book"]: m for m in _ok(console.topic("overview.race", chain="sol")).get("metrics") or []}
        now = datetime.now(timezone.utc)
        marks = [(-82001, "paper_selector", now - timedelta(days=3), 5.0, 0, 0, 0), (-82002, "paper_fly", now - timedelta(days=3), 5.0, 0, 0, 0),
                 (-82003, "paper_selector", now - timedelta(days=1), 5.4, 0, 0, 0), (-82004, "paper_fly", now - timedelta(days=1), 5.2, 0, 0, 0)]
        pos = [{"book": b, "mint": m, "status": "closed", "realized_sol": r, "cost_sol": 0.5, "closed_at": now - timedelta(days=2)}
               for b, m, r in (("paper_selector", MINT_A, 0.4), ("paper_fly", MINT_A, 0.1), ("paper_fly", MINT_B, 0.1), ("paper_fly", "OLDmint", 9.0))]
        pos[-1]["closed_at"] = now - timedelta(days=20)                                    # outside the 14 days
        with seeded(marks, pos):
            d = _ok(console.topic("overview.race", chain="sol"))
            with setting("handover", {"at": (now - timedelta(hours=3)).isoformat(), "fly_pnl_sol": 0.5, "selector_pnl_sol": 0.25, "days": 14}):
                h = _ok(console.topic("overview.race", chain="sol"))
    assert d["hidden"] is False and d["handover"] is False and d["success"] is None
    assert re.fullmatch(r"The fly takes the seat when its realized P&L over the last 14 days is at least the selector's, with at least 30 trades · racing for \d+\.\d days\.", d["caption"])
    m = {x["book"]: x for x in d["metrics"]}
    assert [x["title"] for x in d["metrics"]] == ["Selector · 14 days", "Fly · 14 days"]
    assert m["paper_selector"]["n"] == base.get("paper_selector", {}).get("n", 0) + 1 and m["paper_fly"]["n"] == base.get("paper_fly", {}).get("n", 0) + 2
    assert m["paper_fly"]["pnl"] == pytest.approx((base.get("paper_fly", {}).get("pnl") or 0) + 0.2)
    assert len(d["days"]) == 15 and all(len(x["daily"]) == 15 for x in d["metrics"])
    assert sum(m["paper_fly"]["daily"]) == pytest.approx(m["paper_fly"]["pnl"]) and sum(m["paper_selector"]["daily"]) == pytest.approx(m["paper_selector"]["pnl"])
    names = [s["name"] for s in d["series"]]
    assert names == [b for b in ("paper_selector", "paper_fly", "live") if b in names] and {"paper_selector", "paper_fly"} <= set(names)
    assert h["handover"] is True and h["caption"] is None and [x["title"] for x in h["metrics"]][-1] == "Live · 14 days"
    assert h["success"] == ("The fly took the selector's seat 3.0 h ago: +0.5 SOL vs +0.25 SOL over 14 days. "
                            "Live trading waits for LIVE_ENABLED=1, the live prerequisites and a funded wallet.")


def test_book_lists_open_positions_and_the_wealth_line(console):
    now = datetime.now(timezone.utc)
    pos = [{"book": "paper_selector", "mint": MINT_A, "status": "open", "cost_sol": 0.5, "entry_price": 1.0, "last_mark_price": 1.1, "hold_s": 1800,
            "opened_at": now - timedelta(minutes=10)},
           {"book": "paper_selector", "mint": MINT_B, "status": "open", "cost_sol": 0.25, "entry_price": None, "last_mark_price": None, "hold_s": None,
            "opened_at": now - timedelta(minutes=12), "strategy": "capitulation"}]
    marks = [(-83001, "paper_selector", "2100-01-02 00:00+00", 6.0, 0, 0, 0), (-83002, "paper_selector", "2100-01-02 00:01+00", 6.5, 0, 0, 0)]
    with quiet(), seeded(marks, pos):
        d = _ok(console.topic("overview.book", chain="sol"))
    assert d["unit"] == "SOL" and d["history"][-2:] == [["2100-01-02T00:00:00+00:00", 6.0], ["2100-01-02T00:01:00+00:00", 6.5]]
    rows = {r["mint"]: r for r in d["positions"]}
    a, b = rows[MINT_A], rows[MINT_B]
    assert a["token"] == f"{MINT_A[:6]}…" and a["strategy"] == "—" and a["held_min"] == pytest.approx(10, abs=0.2) and a["exits_in_min"] == pytest.approx(20, abs=0.2)
    assert a["cost"] == 0.5 and a["ret"] == pytest.approx(10.0)
    assert b["strategy"] == "capitulation" and b["ret"] is None and b["exits_in_min"] == pytest.approx(30 - 12, abs=0.2)     # no hold_s: the model's horizon, else 30


def test_wealth_lines_are_thinned_but_end_at_the_last_mark():
    t0 = datetime(2100, 2, 1, tzinfo=timezone.utc)
    marks = [(-84000 - i, "test_overview_thin", t0 + timedelta(minutes=i), 1.0 + i / 100, 0, 0, 0) for i in range(60)]
    with seeded(marks):
        raw = overview.marks(["test_overview_thin"], points=100)
        thin = overview.marks(["test_overview_thin"], points=10)
    assert len(raw) == 60 and [r["wealth"] for r in raw] == [m[3] for m in marks]
    assert 8 <= len(thin) <= 12 and thin[-1]["ts"] == marks[-1][2] and thin[-1]["wealth"] == pytest.approx(marks[-1][3])
    assert [r["ts"] for r in thin] == sorted(r["ts"] for r in thin)


def test_overview_answers_on_robinhood_chain(console, monkeypatch):
    """The page follows the chain picker: Robinhood Chain's books and unit (test_rh_ui rendered overview.py on both chains)."""
    monkeypatch.setattr(config, "RH_ENABLED", True)
    with quiet():
        s = _ok(console.topic("overview.summary", chain="rh"))
        m = _ok(console.topic("overview.minute", chain="rh"))
        r = _ok(console.topic("overview.race", chain="rh"))
        b = _ok(console.topic("overview.book", chain="rh"))
    assert s["market"]["chain"] == "rh" and s["market"]["selector_book"] == "paper_rh_selector" and s["wealth_help"].endswith("ETH; open positions valued net of exit costs.")
    assert m["unit"] == "ETH" and "ETH traded in the last 15 minutes" in m["eligible_help"]
    assert r["unit"] == "ETH" and b["unit"] == "ETH"
    monkeypatch.setattr(config, "RH_ENABLED", False)                                          # RH off: the picker's choice falls back to Solana
    assert _ok(console.topic("overview.summary", chain="rh"))["market"]["chain"] == "sol"


# ---------------------------------------------------------------- the brain
T0 = datetime(2026, 10, 16, 14, 0, tzinfo=timezone.utc).timestamp()
META = {"connectome_sha256": "abc", "connectome_file": "c.npz", "n": 6, "pop_ranges": {"KC": [0, 2], "MBON_APP": [2, 3], "MBON_AV": [3, 4]}, "coverage": {"centroid": 2}}
META_FULL = {"connectome_sha256": "abc", "connectome_file": "c.npz", "n": 8, "pop_ranges": {"KC": [0, 2], "MBON_APP": [2, 3], "MBON_AV": [3, 4]},
             "flies": {"memecoin": {"n": 6, "index": "index_memecoin.bin"}, "kalshi": {"n": 6, "index": "index_kalshi.bin"}}}


@pytest.fixture
def brain_dirs(tmp_path, monkeypatch):
    """Each fly's activity and state directories under tmp_path, the brain caches cleared."""
    d = {"memecoin": (tmp_path / "act", tmp_path / "plastic"), "kalshi": (tmp_path / "act_k", tmp_path / "plastic_k")}
    monkeypatch.setattr(brain, "_dirs", lambda fly: d[fly])
    monkeypatch.setattr(brain, "activity_dir", lambda: d["memecoin"][0])
    monkeypatch.setattr(brain, "geometry", lambda: META)
    monkeypatch.setattr(brain, "geometry_full", lambda: META_FULL)
    for fn in (brain.minutes, brain.activity_at, brain.neuron_history):
        fn.cache_clear()
    return d


def _write(d, minute: int, values, *, n_rows=40, n_cand=7, connectome="c.npz"):
    activity.write(d, T0 + 60 * minute, np.asarray(values, dtype=np.int8), n_rows=n_rows, n_cand=n_cand, connectome=connectome)


def test_activity_endpoint_serves_a_minute_as_raw_int8(console, brain_dirs):
    d = brain_dirs["memecoin"][0]
    _write(d, 0, [1, 2, 3, 4, 5, 6]); _write(d, 1, [-127, 0, 127, 9, 8, 7], n_cand=0)
    r = console.get("/api/brain/activity/memecoin/20261016T1401")
    assert r.status_code == 200 and r.headers["content-type"] == "application/octet-stream"
    assert np.frombuffer(r.content, dtype=np.int8).tolist() == [-127, 0, 127, 9, 8, 7]
    assert (r.headers["x-minute"], r.headers["x-rows"], r.headers["x-candidates"], r.headers["x-connectome"]) == ("2026-10-16T14:01:00+00:00", "40", "0", "c.npz")
    assert "immutable" in r.headers["cache-control"]
    for bad in ("/api/brain/activity/memecoin/20261016T1402", "/api/brain/activity/memecoin/..%2F..%2Fx", "/api/brain/activity/memecoin/2026101", "/api/brain/activity/dog/20261016T1401"):
        assert console.get(bad).status_code == 404, bad
    assert console.get("/api/brain/activity/kalshi/20261016T1401").status_code == 404         # the Kalshi fly keeps its own minutes


def test_neuron_endpoint_returns_a_neurons_day(console, brain_dirs):
    d = brain_dirs["kalshi"][0]
    for k in range(3):
        _write(d, k, [k, 10 * k, -k, 0, 0, 0], n_cand=k)
    r = _ok(console.get("/api/brain/neuron/kalshi/1"))
    assert r["fly"] == "kalshi" and r["index"] == 1
    assert r["series"] == [{"t": f"2026-10-16T14:0{k}:00+00:00", "v": 10 * k, "candidates": k} for k in range(3)]
    assert _ok(console.get("/api/brain/neuron/kalshi/99"))["series"] == []                     # beyond the sub-graph: no points
    assert _ok(console.get("/api/brain/neuron/memecoin/0"))["series"] == []                    # no minutes on disk
    assert console.get("/api/brain/neuron/kalshi/-1").status_code == 404 and console.get("/api/brain/neuron/dog/1").status_code == 404


def test_brain_memecoin_topic_keeps_the_version_1_contract(console, brain_dirs):
    d = brain_dirs["memecoin"][0]
    with quiet():
        p = _ok(console.topic("brain.memecoin", mode="bootstrap"))
        assert p["message"] == "The fly is not running: this is the resting brain." and p["activity_b64"] is None and p["minutes"] == []
        _write(d, 0, [0] * 6); _write(d, 1, [1, -2, 3, -4, 5, -6])
        brain.minutes.cache_clear()
        with setting("fly_status", {"stage": "trading", "bootstrap": 987654321, "channels": {"ev": [0.0, None], "capitulation": [0.0001, 1.0]},
                                    "drift_by_strategy": {"ev": 0.0, "capitulation": 0.0002}}):
            p = _ok(console.topic("brain.memecoin", mode="24h"))
    assert p["version"] == 1 and p["fly"] == "memecoin" and p["running"] is True and p["message"] is None
    assert p["geometry"] == {"url": "/static/brain/geometry.bin?v=abc", "meta_url": "/static/brain/geometry.json?v=abc", "sha": "abc", "n": 6}
    assert p["minutes"] == ["20261016T1400", "20261016T1401"] and p["stamp"] == "20261016T1401" and p["minute"] == "2026-10-16T14:01:00+00:00"
    assert np.frombuffer(base64.b64decode(p["activity_b64"]), dtype=np.int8).tolist() == [1, -2, 3, -4, 5, -6] and (p["rows"], p["candidates"]) == (40, 7)
    assert p["strategies"] == ["ev", "capitulation"] and p["emphasis"] == ["KC", "MBON_APP", "MBON_AV", "MBON_OTHER"]
    assert p["pathways"]["mode"] == "24h" and p["pathways"]["items"] == [] and p["pathways"]["note"] == "the fly's bootstrap file is missing"
    rest = ("drift ev 0.00%, capitulation 0.02% · no KC→MBON synapse has changed yet · the fly's bootstrap file is missing · "
            "ev: channel does not learn (α = 0), so its pathways never change · 2 neurons without a recorded position sit at their population's centroid")
    assert p["caption_rest"] == rest and p["caption"] == "16 Oct 14:01 UTC: mean activity over 7 candidate rows of 40 eligible · " + rest
    assert p["fly_detail"]["bootstrap"] == 987654321
    assert console.topic("brain.memecoin", mode="bogus").status_code == 400


def test_brain_memecoin_topic_flags_a_foreign_connectome_and_a_missing_geometry(console, brain_dirs, monkeypatch):
    _write(brain_dirs["memecoin"][0], 0, [1] * 6, connectome="other.npz")
    with quiet():
        p = _ok(console.topic("brain.memecoin"))
        assert p["activity_b64"] is None and p["minute"] is None and p["message"] == "this minute's activity was recorded on another connectome"
        assert p["minutes"] == ["20261016T1400"] and p["pathways"]["key"] == "bootstrap:none"

        def gone():
            raise FileNotFoundError("connectome missing")
        monkeypatch.setattr(brain, "geometry", gone)
        assert _ok(console.topic("brain.memecoin")) == {"version": 1, "unavailable": "No brain to draw: connectome missing"}


def test_brain_flies_topic_places_each_fly_and_flags_a_foreign_connectome(console, brain_dirs):
    """Port of test_kalshi_ui's payload_flies contract: each fly's index map URL (now /static/brain/...), its own minute,
    and a fly whose minute was recorded on another connectome reported in the message instead of drawn."""
    _write(brain_dirs["memecoin"][0], 5, [0, 1, 2, 3, 4, 5])
    _write(brain_dirs["kalshi"][0], 3, [9] * 6, connectome="other.npz")
    with quiet():
        p = _ok(console.topic("brain.flies", mode="bootstrap"))
    assert p["version"] == 2 and p["geometry"] == {"url": "/static/brain/geometry_full.bin?v=abc", "meta_url": "/static/brain/geometry_full.json?v=abc", "sha": "abc:full", "n": 8}
    m, k = p["flies"]
    assert [m["name"], k["name"]] == ["memecoin", "kalshi"] and "central_owner" not in p                    # the browser picks the central owner
    assert m["index"] == {"url": "/static/brain/index_memecoin.bin?v=abc", "n": 6} and k["index"] == {"url": "/static/brain/index_kalshi.bin?v=abc", "n": 6}
    assert np.frombuffer(base64.b64decode(m["activity_b64"]), dtype=np.int8).tolist() == list(range(6)) and m["candidates"] == 7 and m["minute"].startswith("2026-10-16T14:05")
    assert m["minutes"] == ["20261016T1405"] and m["stamp"] == "20261016T1405" and m["n"] == 6 and m["note"] is None
    assert k["activity_b64"] is None and k["minutes"] == ["20261016T1403"] and k["note"] == "kalshi: this minute's activity was recorded on another connectome"
    assert p["message"] == "kalshi: this minute's activity was recorded on another connectome"
    assert p["caption"] == "memecoin: 14:05 UTC, 7 candidates of 40 rows · kalshi: no minute"
    assert all(f["pathways"]["key"] == "bootstrap:none" and f["running"] is False for f in p["flies"])
    json.dumps(p)


def test_brain_flies_topic_rests_when_neither_fly_has_scored(console, brain_dirs):
    with quiet():
        p = _ok(console.topic("brain.flies"))
    assert p["message"] == "Neither fly has scored a minute yet: this is the resting brain." and all(f["activity_b64"] is None for f in p["flies"])


def test_full_geometry_maps_each_fly_into_the_whole_brain(tmp_path, monkeypatch):
    """The whole-brain geometry the flies view draws carries a sub→full index map per fly (ported from test_kalshi_ui)."""
    npz, tsv, out = tmp_path / "c.npz", tmp_path / "ann.tsv", tmp_path / "static"
    _npz(npz, "sha-f"); _tsv(tsv)
    monkeypatch.setattr(geometry, "FLY_EXCLUDES", {"memecoin": ("VISUAL",), "kalshi": ("OTHER",)})
    m = geometry.build_full_geometry(out, npz, tsv)
    xyz = np.frombuffer((out / "geometry_full.bin").read_bytes(), dtype=np.float32).reshape(-1, 3)
    assert m["n"] == 8 and xyz.shape == (8, 3) and m["excluded"] == [] and m["pop_ranges"]["VISUAL"] == [4, 6]
    mi = np.frombuffer((out / m["flies"]["memecoin"]["index"]).read_bytes(), dtype=np.int32); ki = np.frombuffer((out / m["flies"]["kalshi"]["index"]).read_bytes(), dtype=np.int32)
    assert mi.tolist() == [0, 1, 2, 3, 6, 7] and m["flies"]["memecoin"]["n"] == 6 and ki.tolist() == [0, 1, 2, 3, 4, 5] and m["flies"]["kalshi"]["n"] == 6
    mt = (out / "geometry_full.bin").stat().st_mtime_ns
    assert geometry.build_full_geometry(out, npz, tsv) == m and (out / "geometry_full.bin").stat().st_mtime_ns == mt
    assert geometry.build_geometry(out, npz, tsv)["n"] == 6 and (out / "geometry.bin").exists()
