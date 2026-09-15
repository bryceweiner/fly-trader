"""The minute engine: readiness on the stream, minute aggregation, the full feature vector built once per mint whatever
the number of books, the training eligibility rules, and each book trading in its own savepoint."""
import json
import math
import time
from datetime import datetime, timezone

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


class _Book:
    def __init__(self, name, fail=False):
        self.name, self.fail, self.done, self.seen = name, fail, False, None

    def on_bars(self, t, bars):
        self.bars = bars

    def on_minute(self, ctx):
        ctx.conn.execute("INSERT INTO events (level, source, message) VALUES ('info', 'engine_test', %s)", (self.name,))
        self.seen = (ctx.X.shape, list(ctx.mints))
        if self.fail:
            raise RuntimeError("boom")
        return {"ok": True}

    def open_mints(self, conn):
        return set()

    def finish(self):
        pass


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
    assert good.seen == ((1, len(X_COLS)), ["Wpump"]) and good.bars["Wpump"][0] == m0.timestamp()
    assert "good" in out and "bad" not in out
    with transaction() as conn:
        left = [r["message"] for r in conn.execute("SELECT message FROM events WHERE source = 'engine_test'").fetchall()]
        conn.execute("DELETE FROM events WHERE source = 'engine_test'"); conn.execute("DELETE FROM pump_minutes WHERE mint = 'Wpump'")
    assert left == ["good"]                                                        # the failing book's writes were rolled back alone
