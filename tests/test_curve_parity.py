"""Graduation-time rug facts: the archive scan (corpus_meta.update_curve_facts) and the live stream (pumpstream curve
tracking + _graduation) compute the same bundle share, dev holding and selling, holding concentration and insider
wallets from the same curve trading; the creator's first buy rides on the create event."""
from datetime import datetime, timedelta, timezone

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from fly_trader import config
from fly_trader.db.connection import transaction
from fly_trader.ingest import pumpstream
from fly_trader.train import corpus_meta

T0 = datetime(2026, 9, 4, 10, 0, tzinfo=timezone.utc)
SUPPLY, IB = 1e9, 1e6
LEGS = [  # (seconds, slot, trader, side, tokens)
    (1, 100, "B1", 1, 2e6), (2, 100, "B9", 1, 1e6), (10, 105, "B2", 1, 3e6), (20, 110, "C", -1, 5e5), (30, 120, "B2", -1, 1e6), (35, 121, "B9", -1, 1e6)]


def _facts(conn, mint):
    r = conn.execute("SELECT " + ", ".join(corpus_meta.CURVE_COLS) + ", curve_known FROM corpus_meta WHERE mint = %s", (mint,)).fetchone()
    ins = sorted((x["wallet"], x["kind"]) for x in conn.execute("SELECT wallet, kind FROM token_insiders WHERE mint = %s", (mint,)).fetchall())
    return dict(r), ins


def test_expected_values():
    facts, ins = corpus_meta.curve_facts(100, "C", SUPPLY, IB, {"B1": [100, 2e6, 0], "B9": [100, 1e6, 1e6], "B2": [105, 3e6, 1e6], "C": [None, 0, 5e5]})
    assert facts["bundle_share"] == pytest.approx(2e6 / SUPPLY) and facts["dev_hold_share"] == pytest.approx(5e5 / SUPPLY)
    assert facts["dev_sold_frac"] == pytest.approx(0.5) and facts["grad_hhi"] == pytest.approx((2 / 4.5) ** 2 + (2 / 4.5) ** 2 + (0.5 / 4.5) ** 2)
    assert ins == [("C", "creator"), ("B1", "bundle"), ("B9", "bundle")]


def test_archive_and_live_curve_facts_match(tmp_path, monkeypatch, db_conn):
    monkeypatch.setattr(config, "REPLAY_DIR", tmp_path / "replay")
    day = tmp_path / "replay" / T0.date().isoformat(); day.mkdir(parents=True)
    arch, live = "CurveArchpump", "CurveLivepump"
    g = T0 + timedelta(seconds=40)
    pq.write_table(pa.table({"ts": pa.array([T0], pa.timestamp("ms", tz="UTC")), "slot": [100], "action": ["create"], "pool": ["pump"], "mint": [arch]}),
                   day / "10_events.parquet")
    pq.write_table(pa.table({"ts": pa.array([T0 + timedelta(seconds=s) for s, *_ in LEGS], pa.timestamp("ms", tz="UTC")), "slot": [sl for _, sl, *_ in LEGS],
                             "pool": ["pump"] * len(LEGS), "mint": [arch] * len(LEGS), "trader": [t for *_, t, _, _ in LEGS],
                             "side": pa.array([sd for *_, sd, _ in LEGS], pa.int8()), "tokens": [v for *_, v in LEGS]}), day / "10_trades.parquet")
    with transaction() as conn:
        conn.execute("INSERT INTO replay_hours (hour, status) VALUES (%s, 'done') ON CONFLICT (hour) DO UPDATE SET status = 'done'", (T0,))
        conn.execute("DELETE FROM corpus_meta WHERE mint IN (%s, %s)", (arch, live)); conn.execute("DELETE FROM token_insiders WHERE mint IN (%s, %s)", (arch, live))
        conn.execute("INSERT INTO corpus_meta (mint, create_ts, graduated_at, creator, supply, dev_tokens) VALUES (%s,%s,%s,'C',%s,%s)", (arch, T0, g, SUPPLY, IB))
    assert corpus_meta.update_curve_facts() >= 1
    agg = pumpstream.Aggregator(); ms = lambda s: int((T0.timestamp() + s) * 1000)
    agg.ingest({"action": "create", "pool": "pump", "mint": live, "timestamp": ms(0), "block": 100, "txSigner": "C", "supply": SUPPLY, "initialBuy": IB,
                "quoteAmount": 0.03, "signature": "sigCurveLive"})
    for s, sl, t, sd, v in LEGS:
        agg.ingest({"action": "buy" if sd == 1 else "sell", "pool": "pump", "mint": live, "timestamp": ms(s), "block": sl, "txSigner": t, "tokenAmount": v})
    agg.ingest({"action": "migrate", "pool": "pump-amm", "mint": live, "timestamp": ms(40), "block": 130, "poolId": "PL", "txSigner": "M", "quoteInPool": 80.0,
                "signature": "sigCurveMig"})
    with transaction() as conn:
        fa, ia = _facts(conn, arch); fl, il = _facts(conn, live)
        conn.execute("DELETE FROM corpus_meta WHERE mint IN (%s, %s)", (arch, live)); conn.execute("DELETE FROM token_insiders WHERE mint IN (%s, %s)", (arch, live))
        conn.execute("DELETE FROM pump_events WHERE sig IN ('sigCurveLive', 'sigCurveMig')"); conn.execute("DELETE FROM corpus_tokens WHERE mint = %s", (live,))
        conn.execute("DELETE FROM replay_hours WHERE hour = %s", (T0,))
    assert fa["curve_known"] is True and fl["curve_known"] is True
    for c in corpus_meta.CURVE_COLS:
        assert fa[c] == pytest.approx(fl[c]), c
    assert ia == il == [("B1", "bundle"), ("B9", "bundle"), ("C", "creator")]
    assert agg.insiders[live] == frozenset({"C", "B1", "B9"})
