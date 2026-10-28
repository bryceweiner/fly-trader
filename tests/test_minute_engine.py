"""The minute engine: readiness on the stream, minute aggregation, the full feature vector built once per mint whatever
the number of books, the training eligibility rules, and each book trading in its own savepoint."""
import json
import math
import time
from datetime import datetime, timedelta, timezone

import numpy as np

from fly_trader.agent import minute_engine as me
from fly_trader.db.connection import transaction
from fly_trader.train.decisions import X_COLS


def _engine(books=()):
    return me.MinuteEngine(list(books))


def test_ready_through_waits_for_the_stream(db_conn):
    s = _engine()
    m0 = math.floor(time.time() / 60) * 60 - 60
    with db_conn.cursor() as cur:
        cur.execute("INSERT INTO ui_settings (key, value) VALUES ('pumpstream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (json.dumps({"flushed_through": datetime.fromtimestamp(m0 - 60, timezone.utc).isoformat()}),))
    db_conn.commit()
    assert s.ready_through(m0 + 60) == m0                        # minute m0 is not written yet: it waits, however late
    with db_conn.cursor() as cur:
        cur.execute("UPDATE ui_settings SET value = %s WHERE key = 'pumpstream_status'", (json.dumps({"flushed_through": datetime.fromtimestamp(m0, timezone.utc).isoformat()}),))
    db_conn.commit()
    assert s.ready_through(m0 + 60) == m0 + 60


def test_scale_break_blocks_the_mint_like_training(db_conn):
    s = _engine(); t = 1_800_000_000.0
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 1.0, "sell": 0.0, "nb": 1, "ns": 0, "n_traders": 1, "resq": 50.0, "pool": "P", "program_label": "Pump.fun Amm"}
    assert s._features(db_conn, "Bkpump", a, t)[1]["broken"] is False
    assert s._features(db_conn, "Bkpump", {**a, "close": 80.0}, t + 60)[1]["broken"] is True
    assert s._features(db_conn, "Bkpump", a, t + 120)[1]["broken"] is True          # from that minute on, as in decisions.build


def test_aggregate_and_full_feature_vector(db_conn):
    s = _engine()
    m0 = datetime(2026, 9, 1, 12, 0, tzinfo=timezone.utc)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM pump_minutes WHERE mint = 'Zpump'")
        cur.execute("INSERT INTO pump_minutes (mint, ts, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol) VALUES "
                    "('Zpump', %s, 'POOL', 1.0, 1.2, 0.9, 1.1, 6.0, 2.0, 3, 1, 4, 55.0)", (m0,))
    db_conn.commit()
    agg = s._aggregate(db_conn, m0, m0.replace(minute=1))
    assert list(agg) == ["Zpump"] and agg["Zpump"]["resq"] == 55.0 and agg["Zpump"]["n_traders"] == 4
    x, info = s._features(db_conn, "Zpump", agg["Zpump"], m0.timestamp() + 60)
    assert x.shape == (len(X_COLS),) and np.isfinite(x).all()
    assert info["resq"] == 55.0 and info["price"] == 1.1 and info["open"] == 1.0 and info["ec"] > 0
    assert x[X_COLS.index("n_trades_1m")] == 4.0 and x[X_COLS.index("traders_15m")] == 4.0
    assert x[X_COLS.index("meta_known")] == 0.0 and x[X_COLS.index("age_known")] == 0.0


def test_graduation_time_comes_from_corpus_meta(db_conn):
    s = _engine()
    g = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint = 'Ypump'")
        cur.execute("INSERT INTO corpus_meta (mint, graduated_at, ttg_min, dev_sol) VALUES ('Ypump', %s, 45.0, 1.5)", (g,))
    db_conn.commit()
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 6.0, "sell": 0.0, "nb": 2, "ns": 0, "n_traders": 2, "resq": 60.0, "pool": "P", "program_label": "Pump.fun Amm"}
    x, info = s._features(db_conn, "Ypump", a, g.timestamp() + 2 * 3600)
    assert info["age_h"] == 2.0
    assert x[X_COLS.index("age_known")] == 1.0 and x[X_COLS.index("meta_known")] == 1.0
    assert x[X_COLS.index("ttg_min")] == 45.0 and x[X_COLS.index("dev_sol")] == 1.5
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint = 'Ypump'")
    db_conn.commit()


def test_a_creator_heavy_coin_is_marked_and_leaves_the_universe_at_8_h(db_conn):
    from fly_trader.train.selector import in_universe
    s = _engine()
    g = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint IN ('Hvpump', 'Ltpump')")
        cur.execute("INSERT INTO corpus_meta (mint, graduated_at, ttg_min, dev_sol, dev_share) VALUES ('Hvpump', %s, 0.0, 85.0, 0.7931), ('Ltpump', %s, 30.0, 1.0, 0.05)", (g, g))
    db_conn.commit()
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 6.0, "sell": 0.0, "nb": 2, "ns": 0, "n_traders": 2, "resq": 60.0, "pool": "P", "program_label": "Pump.fun Amm"}
    seen = {}
    for mint in ("Hvpump", "Ltpump"):
        for h in (7.0, 9.0):
            x, info = s._features(db_conn, mint, a, g.timestamp() + h * 3600)
            seen[(mint, h)] = (info["creator_heavy"], bool(in_universe(x[None], X_COLS)[0]))
    assert seen == {("Hvpump", 7.0): (True, True), ("Hvpump", 9.0): (True, False), ("Ltpump", 7.0): (False, True), ("Ltpump", 9.0): (False, True)}
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint IN ('Hvpump', 'Ltpump')")
    db_conn.commit()


class _Book:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.done, self.seen = name, fail, False, None

    def on_bars(self, t, bars):
        self.bars = bars

    def on_minute(self, ctx):
        ctx.conn.execute("INSERT INTO events (level, source, message) VALUES ('info', 'engine_test', %s)", (self.name,))
        self.seen = (ctx.X.shape, list(ctx.mints)); self.lag = ctx.lag_s
        if self.fail:
            raise RuntimeError("boom")
        return {"ok": True}

    def open_mints(self, conn):
        return set()

    def finish(self):
        pass


class _LiveBook(_Book):
    """A book that trades live money: its order goes out, committed, in the live phase."""
    LIVE = True

    def __init__(self, name, calls):
        super().__init__(name); self.calls = calls

    def on_minute(self, ctx):
        self.calls.append(("live", self.name))
        ctx.conn.execute("INSERT INTO events (level, source, message) VALUES ('info', 'engine_test', 'live order')")
        return {"live": True}

    def after_minute(self, ctx):
        self.calls.append(("after", self.name))
        return {"ok": True}


class _PaperBook(_Book):
    def __init__(self, name, calls):
        super().__init__(name); self.calls = calls

    def on_minute(self, ctx):
        self.calls.append(("paper", self.name))
        with transaction() as other:                     # another connection: the live phase is committed already
            self.saw = other.execute("SELECT count(*) AS n FROM events WHERE source = 'engine_test' AND message = 'live order'").fetchone()["n"]
        return {"ok": True}


def test_live_books_trade_and_commit_before_any_paper_book(db_conn, monkeypatch):
    """Live money first: the live book's minute runs and commits before any other book's, whatever the books' order;
    its paper trades, learning and bookkeeping (after_minute) come after the paper books."""
    m0 = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc); calls = []
    with transaction() as conn:
        conn.execute("DELETE FROM events WHERE source = 'engine_test'")
    paper, live = _PaperBook("selector", calls), _LiveBook("fly", calls)
    e = _engine([paper, live]); monkeypatch.setattr(e, "stream_fresh", lambda m0_epoch: True)
    out = e.run_minute(m0.timestamp() + 60)
    with transaction() as conn:
        conn.execute("DELETE FROM events WHERE source = 'engine_test'")
    assert calls == [("live", "fly"), ("paper", "selector"), ("after", "fly")] and paper.saw == 1
    assert out["fly"] == {"ok": True} and out["selector"] == {"ok": True}


def test_every_chains_live_phase_runs_before_any_chains_paper_phase(monkeypatch):
    """Live money first across chains: older minutes are caught up whole, then every chain's newest minute's live phase
    (the chains at once), and only then any chain's paper phase."""
    calls, engines = [], []
    for name in ("sol", "rh"):
        e = _engine([_Book(f"book_{name}")]); e.last_minute = 600.0
        monkeypatch.setattr(e, "ready_through", lambda m1: 720.0)
        monkeypatch.setattr(e, "run_minute", lambda m, trade=True, n=name: calls.append((n, "whole", m)) or {})
        monkeypatch.setattr(e, "live_phase", lambda m, trade=True, n=name: calls.append((n, "live", m)) or (None, {"minute": m}))
        monkeypatch.setattr(e, "paper_phase", lambda ctx, out, n=name: calls.append((n, "paper")) or out)
        engines.append(e)
    ran = me.run_ready(engines, 720.0)
    assert calls[:2] == [("sol", "whole", 660.0), ("rh", "whole", 660.0)] and calls[4:] == [("sol", "paper"), ("rh", "paper")]
    assert sorted(calls[2:4]) == [("rh", "live", 720.0), ("sol", "live", 720.0)]       # side by side, either first
    assert [(m, st) for _, m, st in ran] == [(720, {"minute": 720.0})] * 2 and all(e.last_minute == 720 for e in engines)
    assert me.run_ready(engines, 720.0) == []                                       # nothing new: the loop goes idle


def test_books_share_one_feature_pass_and_trade_in_their_own_savepoints(db_conn, monkeypatch):
    m0 = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM pump_minutes WHERE mint = 'Wpump'"); cur.execute("DELETE FROM events WHERE source = 'engine_test'")
        cur.execute("INSERT INTO pump_minutes (mint, ts, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol) VALUES "
                    "('Wpump', %s, 'POOL', 1.0, 1.2, 0.9, 1.1, 6.0, 2.0, 3, 1, 4, 55.0)", (m0,))
    db_conn.commit()
    bad, good = _Book("bad", fail=True), _Book("good")
    e = _engine([bad, good]); calls = []
    real = e._features
    monkeypatch.setattr(e, "_features", lambda *a: calls.append(a[1]) or real(*a))
    monkeypatch.setattr(e, "stream_fresh", lambda m0_epoch: True)
    out = e.run_minute(m0.timestamp() + 60)
    assert calls == ["Wpump"]                                                      # one feature pass per mint, two books
    assert good.lag > 86400                                                         # how long after its end the minute was traded (a replayed past one)
    assert good.seen == ((1, len(X_COLS)), ["Wpump"]) and good.bars["Wpump"][0] == m0.timestamp()
    assert "good" in out and "bad" not in out
    with transaction() as conn:
        left = [r["message"] for r in conn.execute("SELECT message FROM events WHERE source = 'engine_test'").fetchall()]
        conn.execute("DELETE FROM events WHERE source = 'engine_test'"); conn.execute("DELETE FROM pump_minutes WHERE mint = 'Wpump'")
    assert left == ["good"]                                                        # the failing book's writes were rolled back alone


def test_a_quiet_minute_is_charged_the_mints_last_known_fee(db_conn):
    """A held coin that did not trade this minute reports no fee: its sale and its mark pay the last one known (the
    engine's own minutes, else the stream's newest row with one; RH: else the pool's measured hook fee), not the cost
    model's generic fallback (on RH ~1 % against hook fees of 2.8-5 %)."""
    from fly_trader import markets
    m0 = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc); s = _engine(); rh = me.MinuteEngine(market=markets.RH)
    tok, pool = "0x" + "f1" * 20, "0x" + "f2" * 32

    def ctx(engine, fees):
        return me.Minute(conn=db_conn, m0=m0, m1=m0, m1_epoch=m0.timestamp(), agg={}, X=np.zeros((0, len(X_COLS)), np.float32), infos=[], mints=[], bars={},
                         prices={}, resqs={}, fees=fees, fresh=True, trade=True, engine=engine)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM pump_minutes WHERE mint IN ('Fpump', 'Npump')"); cur.execute("DELETE FROM rh_pools WHERE pool_id = %s", (pool,))
        cur.execute("INSERT INTO pump_minutes (mint, ts, pool_id, close, resq_sol, fee_rate) VALUES ('Fpump', %s, 'P', 1.0, 50.0, 0.0115), ('Fpump', %s, 'P', 1.0, 50.0, NULL)",
                    (m0, m0 + timedelta(minutes=1)))
        cur.execute("INSERT INTO rh_pools (pool_id, token, currency0, currency1, is_pons, hook_fee) VALUES (%s, %s, %s, %s, true, 0.029)", (pool, tok, "0x" + "00" * 20, tok))
    db_conn.commit()
    try:
        c = ctx(s, {"Gpump": 0.01, "Fpump": None})
        assert c.fee("Gpump") == 0.01 and c.fee("Fpump") == 0.0115 and c.fee("Npump") is None
        a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 1.0, "sell": 0.0, "nb": 1, "ns": 0, "n_traders": 1, "resq": 50.0, "pool": "P",
             "program_label": "Pump.fun Amm", "fee_rate": 0.0095}
        s._features(db_conn, "Fpump", a, m0.timestamp() + 180)
        assert c.fee("Fpump") == 0.0095                                                # the engine's own newer minute
        assert ctx(rh, {}).fee(tok) == 0.029 and c.cost_label("Fpump", "Pump.fun Amm") == "Pump.fun Amm"
    finally:
        with db_conn.cursor() as cur:
            cur.execute("DELETE FROM pump_minutes WHERE mint IN ('Fpump', 'Npump')"); cur.execute("DELETE FROM rh_pools WHERE pool_id = %s", (pool,))
        db_conn.commit()
