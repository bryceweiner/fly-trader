"""The Kalshi engine and the fly's session: minute rows become the same feature vectors the corpus builder makes from
the same bars (parity), both sides of every market in the entry window pass to the books, and the plastic Kalshi fly
scores, tags, trades both paper arms, and learns from a settlement when it comes."""
import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
import torch

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.kalshi import engine as KE, fly as KF, fly_session as KFS, paper as P
from fly_trader.kalshi.features import K_COLS, KIDX, Bar, EventState, MarketMeta, MarketState
from fly_trader.train import fly_selector
from tests.test_kalshi_fly import _boot, _ds, _graph

T0 = datetime(2026, 9, 22, 12, 0, tzinfo=timezone.utc)
TICKERS = ("TSTE-A-YES1", "TSTE-A-YES2")


def _seed_markets(close: datetime):
    with transaction() as c:
        c.execute("DELETE FROM kalshi_minutes WHERE ticker LIKE 'TSTE-%'"); c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TSTE-%'")
        c.execute("DELETE FROM kalshi_events WHERE event_ticker = 'TSTE-A'"); c.execute("DELETE FROM kalshi_series WHERE ticker = 'TSTE'")
        c.execute("INSERT INTO kalshi_series (ticker, title, category, frequency, fee_type, fee_multiplier) VALUES ('TSTE', 't', 'Politics', 'one_off', 'quadratic', 1.0)")
        c.execute("INSERT INTO kalshi_events (event_ticker, series_ticker, title, category, mutually_exclusive) VALUES ('TSTE-A', 'TSTE', 'e', 'Politics', true)")
        for tk, strike in zip(TICKERS, (1.0, 2.0)):
            c.execute("INSERT INTO kalshi_markets (ticker, event_ticker, status, open_time, close_time, floor_strike, source) VALUES (%s, 'TSTE-A', 'active', %s, %s, %s, 'test')",
                      (tk, T0 - timedelta(days=3), close, strike))


def _minute_rows(minutes: int):
    """Both markets quoted every minute, one trade a minute on the first."""
    rows = []
    for k in range(minutes):
        ts = T0 - timedelta(minutes=minutes - k)
        for i, tk in enumerate(TICKERS):
            yb, ya = 60.0 + i * 5 + (k % 3), 63.0 + i * 5 + (k % 3)
            rows.append((tk, ts, yb, ya, ya - 1, 20.0, 30.0, 500.0 + k, 400.0, 3.0 if i == 0 else 0.0, 1.0 if i == 0 else 0.0, 2 if i == 0 else 0, 3.0 if i == 0 else 0.0, 0.0, ya - 0.5, yb + 0.5))
    with transaction() as c:
        c.cursor().executemany("INSERT INTO kalshi_minutes (ticker, ts, yes_bid, yes_ask, last, bid_size, ask_size, volume_fp, open_interest_fp, taker_buy_yes, taker_buy_no, n_trades, max_trade, "
                               "block_contracts, yes_ask_low, yes_bid_high) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)", rows)
        c.execute("INSERT INTO ui_settings (key, value) VALUES ('kalshi_stream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                  (json.dumps({"flushed_through": (T0 - timedelta(minutes=1)).isoformat(), "connected": True}),))
    return rows


class Recorder:
    name = "recorder"; done = False

    def __init__(self):
        self.ctxs = []

    def on_minute(self, ctx):
        self.ctxs.append((ctx.X.copy(), list(ctx.keys), ctx.hold_s.copy(), dict(ctx.extremes), ctx.n_rows)); return {"ok": True}

    def tickers_watched(self):
        return set()

    def maybe_reload(self):
        return False

    def finish(self):
        pass


def test_engine_rows_match_the_feature_engine_fed_the_same_bars(monkeypatch):
    monkeypatch.setattr(config, "KALSHI_MIN_OPEN_INTEREST", 100); monkeypatch.setattr(config, "KALSHI_MIN_VOLUME_24H", 50)
    close = T0 + timedelta(days=2); _seed_markets(close); rows = _minute_rows(5)
    eng = KE.KalshiMinuteEngine([rec := Recorder()])
    m1 = T0.timestamp()
    assert eng.warm_up(m1, minutes=4) == 8 and len(eng.states) == 2 and set(eng.metas) == set(TICKERS)
    out = eng.run_minute(m1, trade=True)
    assert out["markets_active"] == 2 and out["in_window"] == 2 and out["eligible"] == 4
    X, keys, hold, extremes, n_rows = rec.ctxs[-1]
    assert keys == [(TICKERS[0], "yes"), (TICKERS[0], "no"), (TICKERS[1], "yes"), (TICKERS[1], "no")] and n_rows == 4
    assert np.allclose(hold, close.timestamp() - (m1 - 60.0)) and extremes[TICKERS[0]] == (pytest.approx(63.0 + (4 % 3) - 0.5), pytest.approx(60.0 + (4 % 3) + 0.5))
    # parity: the same bars through a fresh MarketState / EventState give the same vectors
    meta = {tk: MarketMeta(tk, event_ticker="TSTE-A", series_ticker="TSTE", category="politics", open_ts=(T0 - timedelta(days=3)).timestamp(), close_ts=close.timestamp(),
                           mutually_exclusive=1.0, fee_multiplier=1.0, maker_fee=0.0, strike=float(i + 1)) for i, tk in enumerate(TICKERS)}
    states = {tk: MarketState(tk) for tk in TICKERS}; ev = EventState()
    for r in rows:
        tk, ts = r[0], r[1]; t_end = ts.timestamp() + 60.0
        bar = Bar(t_end, r[2], r[3], r[4], r[5], r[6], r[7], r[8], r[9], r[10], float(r[11]), r[12], r[13])
        states[tk].append(bar); ev.update(tk, bar, meta[tk].strike)
    want = np.asarray([states[tk].features(m1, meta[tk], side, ev) for tk, side in keys], dtype=np.float32)
    assert np.allclose(X, want, atol=1e-5), np.abs(X - want).max(0)
    assert X[0, KIDX["mutually_exclusive"]] == 1.0 and X[0, KIDX["log_n_siblings"]] > 0 and X[0, KIDX["cat_politics"]] == 1.0 and X[0, KIDX["ladder_resid"]] >= 0
    # a market outside the window (closing in 2 minutes) is fed but not scored
    with transaction() as c:
        c.execute("UPDATE kalshi_markets SET close_time = %s WHERE ticker = %s", (T0 + timedelta(minutes=2), TICKERS[1]))
    eng.metas.clear(); eng.meta_at.clear()
    with transaction() as c:
        c.cursor().executemany("INSERT INTO kalshi_minutes (ticker, ts, yes_bid, yes_ask, volume_fp, open_interest_fp, taker_buy_yes, taker_buy_no, n_trades, max_trade, block_contracts) "
                               "VALUES (%s,%s,60,63,600,400,0,0,0,0,0)", [(tk, T0) for tk in TICKERS])
    out = eng.run_minute(m1 + 60, trade=True)
    assert out["in_window"] == 1 and out["eligible"] == 2
    with transaction() as c:
        c.execute("DELETE FROM kalshi_minutes WHERE ticker LIKE 'TSTE-%'"); c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TSTE-%'")


def _ctx(conn, m1: datetime, X, keys, hold, metas, quotes, extremes):
    return KE.KMinute(conn=conn, m0=m1 - timedelta(minutes=1), m1=m1, m1_epoch=m1.timestamp(), X=X, keys=keys, hold_s=hold, metas=metas, quotes=quotes, extremes=extremes,
                      n_rows=len(keys), fresh=True, trade=True, engine=None)


def test_fly_session_scores_trades_both_arms_and_learns_from_a_settlement(monkeypatch, tmp_path, db_conn):
    monkeypatch.setattr(fly_selector.prog, "update", lambda *a, **k: None); monkeypatch.setattr(KF.prog, "update", lambda *a, **k: None)
    monkeypatch.setattr(fly_selector, "EPOCHS", 2); monkeypatch.setattr(config, "BRAIN_DIR", tmp_path); monkeypatch.setattr(config, "KALSHI_LIVE_ENABLED", False)
    monkeypatch.setattr(config, "KALSHI_CAPITAL_USD", 100.0)
    monkeypatch.setattr(KF, "graph", lambda: _graph()); monkeypatch.setattr(KF, "_GRAPH", None)
    from fly_trader.ops import reset as R
    monkeypatch.setattr(R, "kalshi_fly_state_dir", lambda: tmp_path / "plastic_kalshi"); monkeypatch.setattr(R, "kalshi_activity_dir", lambda: tmp_path / "activity_kalshi")
    monkeypatch.setattr(KFS, "kalshi_fly_state_dir", lambda: tmp_path / "plastic_kalshi"); monkeypatch.setattr(KFS, "kalshi_activity_dir", lambda: tmp_path / "activity_kalshi")
    ds = _ds(days=12, per_day=300); fly, info = _boot(ds, 10)
    for k in fly.lines:                                       # every candidate trades: the test is about the plumbing, not the line
        fly.lines[k] = -1.0; fly.sizings[k] = [{"lo": 0.0, "n": 50, "mean": 0.05, "win": 0.7, "kelly": 0.10}]
    info["lines"] = dict(fly.lines); info["gates_ok"] = True; info["calibration"] = {"trades": 150, "mean": 0.02}
    path, sid = KF.save(fly, info)
    with transaction() as c:
        c.execute("INSERT INTO ui_settings (key, value) VALUES ('kalshi_fly_replay', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                  (json.dumps({"passed": True, "data": KF.KALSHI_FLY_VERSION, "per_strategy": {k: {"alpha": 1e-3, "half_life_days": 3.0} for k in fly.strategies}}),))
        c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TSTS-%'"); c.execute("DELETE FROM kalshi_fly_scored WHERE ticker LIKE 'TSTS-%'")
        c.execute("DELETE FROM kalshi_positions WHERE ticker LIKE 'TSTS-%'"); c.execute("DELETE FROM kalshi_orders WHERE ticker LIKE 'TSTS-%'")
        c.execute("DELETE FROM ui_settings WHERE key = 'kalshi_fly_command'")
    book, why = KFS.try_start()
    assert book is not None, why
    m1 = T0; close = T0 + timedelta(hours=3)
    metas = {"TSTS-1": MarketMeta("TSTS-1", event_ticker="E", category="politics", close_ts=close.timestamp(), maker_fee=0.0),
             "TSTS-2": MarketMeta("TSTS-2", event_ticker="E", category="politics", close_ts=close.timestamp(), maker_fee=0.0)}
    X = ds.X[:4].copy(); keys = [("TSTS-1", "yes"), ("TSTS-1", "no"), ("TSTS-2", "yes"), ("TSTS-2", "no")]
    for i, (tk, side) in enumerate(keys):
        ask = 75.0 if side == "yes" else 30.0
        X[i, KIDX["side_ask"]] = ask; X[i, KIDX["side_bid"]] = ask - 2; X[i, KIDX["eff_price"]] = ask + 1.0
    quotes = {"TSTS-1": (73.0, 75.0, 50.0, 50.0), "TSTS-2": (73.0, 75.0, 50.0, 50.0)}
    hold = np.full(4, close.timestamp() - (m1.timestamp() - 60.0))
    out = book.on_minute(_ctx(db_conn, m1, X, keys, hold, metas, quotes, {}))
    assert out["stage"] == "trading" and out["picks"] > 0 and out["pending"] > 0
    n_scored = db_conn.execute("SELECT count(*) AS n FROM kalshi_fly_scored WHERE ticker LIKE 'TSTS-%' AND state = 'pending'").fetchone()["n"]
    assert n_scored > 0 and (out["books"]["taker"]["entered"] + out["books"]["maker"]["posted"]) > 0
    assert db_conn.execute("SELECT count(*) AS n FROM wealth_marks WHERE book IN ('paper_kalshi_taker', 'paper_kalshi_maker')").fetchone()["n"] >= 2
    st = json.loads(db_conn.execute("SELECT value FROM ui_settings WHERE key = 'kalshi_fly_status'").fetchone()["value"]) if isinstance(db_conn.execute(
        "SELECT value FROM ui_settings WHERE key = 'kalshi_fly_status'").fetchone()["value"], str) else db_conn.execute("SELECT value FROM ui_settings WHERE key = 'kalshi_fly_status'").fetchone()["value"]
    assert st["stage"] == "trading" and st["bootstrap"] == sid
    # the market settles: tags due after close learn the outcome, positions settle, labels are written
    db_conn.execute("INSERT INTO kalshi_markets (ticker, status, result, source) VALUES ('TSTS-1', 'settled', 'yes', 'test'), ('TSTS-2', 'settled', 'no', 'test')")
    d0 = float(book.bank.drift()[0])
    m2 = close + timedelta(minutes=2)
    out2 = book.on_minute(_ctx(db_conn, m2, np.zeros((0, len(K_COLS)), np.float32), [], np.zeros(0), metas, quotes, {}))
    assert out2["learned"]["n"] > 0 and float(book.bank.drift()[0]) > d0 and out2["pending"] == 0
    resolved = db_conn.execute("SELECT state, label, strategy FROM kalshi_fly_scored WHERE ticker LIKE 'TSTS-%' AND state <> 'pending'").fetchall()
    assert resolved and all(r["state"] in ("resolved", "unfilled") for r in resolved)
    taker_labels = [r["label"] for r in resolved if book.arms[r["strategy"]] == "taker"]
    assert taker_labels and all(l is not None for l in taker_labels)
    assert all(p["status"] == "settled" for p in db_conn.execute("SELECT status FROM kalshi_positions WHERE ticker LIKE 'TSTS-%'").fetchall())
    book.finish()
    with transaction() as c:
        c.execute("DELETE FROM kalshi_markets WHERE ticker LIKE 'TSTS-%'"); c.execute("DELETE FROM kalshi_fly_scored WHERE ticker LIKE 'TSTS-%'")
