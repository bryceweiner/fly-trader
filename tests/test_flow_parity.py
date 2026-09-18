"""Wallet flow parity: the same trades through the archive (mature.aggregate_day, DuckDB) and the live stream
(pumpstream.Aggregator → pump_minutes) give identical per-minute wallet columns — buyers, wash round trips, the top
seller, insider selling and skill-bucket buying — and FlowWindow derives the model inputs from them."""
from datetime import date, datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ingest import pumpstream
from fly_trader.train import flow, mature

WSOL = pumpstream.WSOL
MINT = "FlowTestpump"
T0 = datetime(2026, 9, 3, 12, 0, tzinfo=timezone.utc)
LEGS = [  # (seconds after T0, trader, side, sol)
    (5, "W1", 1, 1.0), (7, "W1", -1, 0.4), (9, "W2", 1, 2.0), (12, "I1", -1, 3.0), (20, "W3", -1, 0.5), (25, "W4", 1, 0.7),
    (65, "W2", -1, 1.0), (70, "W1", 1, 0.3)]


@pytest.fixture()
def world(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay"); monkeypatch.setattr(mature, "MATURE_DIR", tmp_path / "mature")
    monkeypatch.setattr(mature, "SKILL_DIR", tmp_path / "skill")
    (tmp_path / "skill").mkdir()
    pq.write_table(pa.table({"wallet": ["W1", "W2"], "bucket": pa.array([9, 3], pa.int8())}), tmp_path / "skill" / "2026-09-03.parquet")
    n = len(LEGS)
    trades = pa.table({"ts": pa.array([datetime.fromtimestamp(T0.timestamp() + s, timezone.utc) for s, *_ in LEGS], pa.timestamp("ms", tz="UTC")),
                       "slot": list(range(n)), "pool": ["pump-amm"] * n, "mint": [MINT] * n, "trader": [t for _, t, _, _ in LEGS],
                       "side": pa.array([sd for *_, sd, _ in LEGS], pa.int8()), "sol": [v for *_, v in LEGS], "tokens": [10.0] * n, "price": [0.001] * n,
                       "quote_in_pool": [50.0] * n, "quote_mint": [WSOL] * n, "pool_id": ["P1"] * n})
    (tmp_path / "replay" / "2026-09-03").mkdir(parents=True)
    pq.write_table(trades, tmp_path / "replay" / "2026-09-03" / "12_trades.parquet")
    with transaction() as conn:
        conn.execute("INSERT INTO token_insiders (mint, wallet, kind) VALUES (%s, 'I1', 'bundle') ON CONFLICT DO NOTHING", (MINT,))
        conn.execute("DELETE FROM pump_minutes WHERE mint = %s", (MINT,))
    yield tmp_path
    with transaction() as conn:
        conn.execute("DELETE FROM token_insiders WHERE mint = %s", (MINT,)); conn.execute("DELETE FROM pump_minutes WHERE mint = %s", (MINT,))
        conn.execute("DELETE FROM mature_days WHERE day = %s", (date(2026, 9, 3),))


def _live(tmp_path):
    agg = pumpstream.Aggregator()
    for s, trader, side, sol in LEGS:
        agg.ingest({"action": "buy" if side == 1 else "sell", "pool": "pump-amm", "mint": MINT, "timestamp": int((T0.timestamp() + s) * 1000), "quoteMint": WSOL,
                    "poolId": "P1", "price": 0.001, "quoteInPool": 50.0, "txSigner": trader, "quoteAmount": sol, "tokenAmount": 10.0})
    pumpstream._load_insiders(agg); pumpstream.load_skill(agg, T0.timestamp())
    agg.flush(int(T0.timestamp()) + 3600)
    with transaction() as conn:
        return {r["ts"]: r for r in conn.execute("SELECT * FROM pump_minutes WHERE mint = %s ORDER BY ts", (MINT,)).fetchall()}


def test_archive_and_live_wallet_columns_match(world):
    assert mature.aggregate_day(date(2026, 9, 3)) > 0
    arch = {r["ts"]: r for r in pq.read_table(world / "mature" / "2026-09-03.parquet").to_pylist() if r["mint"] == MINT}
    live = _live(world)
    assert set(arch) == set(live) and len(arch) == 2
    for ts in arch:
        a, l = arch[ts], live[ts]
        for c in ("n_buyers", "wash_sol", "wash_buy_sol", "top_sell_sol", "insider_sell_sol", "n_traders", "buy_sol", "sell_sol"):
            assert float(a[c]) == pytest.approx(float(l[c])), (ts, c)
        assert [pytest.approx(v) for v in a["skill_buy"]] == [float(v) for v in l["skill_buy"]], ts
    m0 = arch[T0]
    assert m0["n_buyers"] == 3 and m0["wash_sol"] == pytest.approx(1.4) and m0["wash_buy_sol"] == pytest.approx(1.0)
    assert m0["top_sell_sol"] == pytest.approx(3.0) and m0["insider_sell_sol"] == pytest.approx(3.0)
    assert m0["skill_buy"][9] == pytest.approx(1.0) and m0["skill_buy"][3] == pytest.approx(2.0) and m0["skill_buy"][10] == pytest.approx(0.7)


def test_flow_window_derivations():
    fw = flow.FlowWindow(); t = 1_800_000_000.0
    fw.append(t, 3.7, 3.9, 3, 1.4, 1.0, 3.0, 3.0, [0, 0, 0, 2.0, 0, 0, 0, 0, 0, 1.0, 0.7])
    f = fw.features(t)
    assert f["top_sell_share_1m"] == pytest.approx(3.0 / 3.9) and f["insider_sell_share_1m"] == pytest.approx(3.0 / 3.9)
    assert f["wash_share_15m"] == pytest.approx(1.4 / 7.6) and f["org_imb_5m"] == pytest.approx(((3.7 - 1.0) - (3.9 - 0.4)) / ((3.7 - 1.0) + (3.9 - 0.4)))
    assert f["skill_top9_15m"] == pytest.approx(1.0 / 3.7) and f["skill_top3_15m"] == pytest.approx(3.0 / 3.7) and f["skill_known_15m"] == pytest.approx(3.0 / 3.7)
    fw.append(t + 60, 1.0, 0.0, 1, 0, 0, 0, 0, None)                      # a day without a skill table counts as unknown buyers
    assert fw.features(t + 60)["buyers_ret"] == pytest.approx(0.6931471805599453 - __import__("math").log1p(3 / 5))
    mw = flow.MarketWindow(); mw.add(t, 5.0); mw.add(t + 3600, 1.0)
    assert mw.value(t + 3600) == pytest.approx(__import__("math").log1p(1.0))          # the first minute left the hour


def test_skill_window_matches_flow_window():
    rng = np.random.default_rng(0); rows = []
    for m in ("A", "B", "C"):
        t = 1_800_000_000
        for _ in range(80):
            t += 60 * int(rng.integers(1, 6))                                       # traded minutes with gaps
            rows.append((m, t, None if rng.random() < 0.2 else list(rng.exponential(1.0, flow.N_SKILL) * (rng.random(flow.N_SKILL) < 0.5))))
    rng.shuffle(rows)
    got = flow.skill_features(np.array([r[0] for r in rows]), np.array([r[1] for r in rows], float),
                              np.array([r[2] if r[2] is not None else [0.0] * flow.N_SKILL for r in rows]))
    for m in ("A", "B", "C"):
        fw = flow.FlowWindow()
        for i in sorted((i for i, r in enumerate(rows) if r[0] == m), key=lambda i: rows[i][1]):
            fw.append(rows[i][1] + 60.0, 1.0, 0.0, 1, 0, 0, 0, 0, rows[i][2])
            f = fw.features(rows[i][1] + 60.0)
            assert got[i] == pytest.approx([f[c] for c in flow.SKILL_COLS], abs=1e-6)


def test_candidate_skill_inputs_equal_the_archive_build(world, monkeypatch):
    from fly_trader.train import decisions, progress, wallet_skill
    monkeypatch.setattr(progress, "update", lambda *a, **k: None)
    d = date(2026, 9, 3); table = mature.skill_table_for(d)
    assert mature.aggregate_day(d) > 0
    agg = pq.read_table(world / "mature" / f"{d}.parquet")
    same = mature.aggregate_table(mature._hour_files(d), table)
    assert same.to_pylist() == agg.to_pylist()                                     # the extracted SQL is aggregate_day's
    monkeypatch.setattr(wallet_skill, "skill_table", lambda dd, h, L, day_dir=None: table if dd == d else None)
    a = agg.to_pandas(); ts = mature._epoch_s(a["ts"]); n = len(a)
    ds = decisions.DecisionSet(X=np.zeros((n, len(decisions.X_COLS)), np.float32), y=np.zeros(n, np.int8), fwd=np.zeros(n, np.float32),
                               fwd_pess=np.zeros(n, np.float32), day=np.array([d] * n, dtype=object), ts=ts, mint=a["mint"].to_numpy(),
                               cols=list(decisions.X_COLS), horizon_s=7200.0)
    got = wallet_skill.candidate_skill_cols(ds, 30, 1)
    fws = {}
    for i, r in a.iterrows():
        fw = fws.setdefault(r["mint"], flow.FlowWindow())
        fw.append(ts[i] + 60.0, r["buy_sol"], r["sell_sol"], r["n_buyers"], r["wash_sol"], r["wash_buy_sol"], r["top_sell_sol"], r["insider_sell_sol"],
                  None if r["skill_buy"] is None else list(r["skill_buy"]))
        f = fw.features(ts[i] + 60.0)
        assert got[i] == pytest.approx([f[c] for c in flow.SKILL_COLS], abs=1e-6)
    assert got[:, flow.SKILL_COLS.index("skill_known_15m")].max() > 0              # the candidate table was used


def test_passed_in_lookups_match_the_database(world):
    """The skill fit reads the blocked pools and insiders once and hands them to every day instead of reconnecting for
    each; the rows must be identical to the ones the database path builds."""
    from fly_trader.db.connection import transaction
    from fly_trader.train.corpus_meta import blocked_pool_ids
    d = date(2026, 9, 3); files = mature._hour_files(d)
    from_db = mature.aggregate_table(files, mature.skill_table_for(d))
    with transaction() as conn:
        blocked = blocked_pool_ids(conn)
        ins: dict = {}
        for r in conn.execute("SELECT mint, wallet FROM token_insiders").fetchall():
            ins.setdefault(r["mint"], []).append(r["wallet"])
    passed_in = mature.aggregate_table(files, mature.skill_table_for(d), blocked=blocked, insiders=ins)
    assert from_db.to_pylist() == passed_in.to_pylist()
    assert any(r["insider_sell_sol"] > 0 for r in passed_in.to_pylist())      # the insider join really fired
