"""Live selector session: readiness on the stream, minute aggregation, feature assembly by model column names."""
import json
import math
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from fly_trader import config
from fly_trader.agent import selector_session as ss
from fly_trader.train.decisions import X_COLS


class _Model:
    cols = list(X_COLS); threshold = 0.9; horizon_min = 30
    def score(self, X):
        return np.full(len(X), 0.5)


def _session(monkeypatch, db_conn):
    monkeypatch.setattr(config, "SELECTOR_SOURCE", "stream")
    s = ss.SelectorSession.__new__(ss.SelectorSession)
    s.model = _Model(); s.horizon_s = 1800; s.states = {}; s.meta_cache = {}; s.last_sweep = 0.0; s.live = False
    return s


def test_minute_ready_waits_for_the_stream(monkeypatch, db_conn):
    s = _session(monkeypatch, db_conn)
    m0 = math.floor(time.time() / 60) * 60 - 60
    with db_conn.cursor() as cur:
        cur.execute("INSERT INTO ui_settings (key, value) VALUES ('pumpstream_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (json.dumps({"flushed_through": datetime.fromtimestamp(m0 - 60, timezone.utc).isoformat()}),))
    db_conn.commit()
    assert s.minute_ready(m0, max_wait_s=3600) is False          # the stream has not written m0 yet
    with db_conn.cursor() as cur:
        cur.execute("UPDATE ui_settings SET value = %s WHERE key = 'pumpstream_status'", (json.dumps({"flushed_through": datetime.fromtimestamp(m0, timezone.utc).isoformat()}),))
    db_conn.commit()
    assert s.minute_ready(m0, max_wait_s=3600) is True


def test_aggregate_and_feature_vector(monkeypatch, db_conn):
    s = _session(monkeypatch, db_conn)
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
    assert info["resq"] == 55.0 and info["price"] == 1.1
    assert x[X_COLS.index("n_trades_1m")] == 4.0 and x[X_COLS.index("traders_15m")] == 4.0
    assert x[X_COLS.index("meta_known")] == 0.0 and x[X_COLS.index("age_known")] == 0.0


def test_graduation_time_comes_from_corpus_meta(monkeypatch, db_conn):
    s = _session(monkeypatch, db_conn)
    g = datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint = 'Ypump'")
        cur.execute("INSERT INTO corpus_meta (mint, graduated_at, ttg_min, dev_sol) VALUES ('Ypump', %s, 45.0, 1.5)", (g,))
    db_conn.commit()
    a = {"open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "buy": 6.0, "sell": 0.0, "nb": 2, "ns": 0, "n_traders": 2, "resq": 60.0, "pool": "P", "program_label": "Pump.fun Amm"}
    t_end = g.timestamp() + 2 * 3600
    x, info = s._features(db_conn, "Ypump", a, t_end)
    assert info["age_h"] == 2.0
    assert x[X_COLS.index("age_known")] == 1.0 and x[X_COLS.index("meta_known")] == 1.0
    assert x[X_COLS.index("ttg_min")] == 45.0 and x[X_COLS.index("dev_sol")] == 1.5
    with db_conn.cursor() as cur:
        cur.execute("DELETE FROM corpus_meta WHERE mint = 'Ypump'")
    db_conn.commit()
