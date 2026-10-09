"""The maker arm's order and its training label are one rule: a bid one tick above the side's bid (never at or through
the ask) resting for the order's life, filled when the ask comes down to it or a taker sells into it — below it, or at
it when the bid was first at its price. And the Kalshi stack's bars: a positive mean with the weekly sign test on both
halves and the holdout, not the memecoin PF and win rate."""
from datetime import datetime, timezone

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from fly_trader import config
from fly_trader.kalshi import decisions as KD, mature
from fly_trader.kalshi.features import K_COLS, KALSHI_FEATURE_VERSION
from fly_trader.kalshi.selector import holdout_passes
from fly_trader.train.strategies import Score

T0 = 1790089800                  # a minute end


def test_the_rest_improves_the_bid_by_one_tick_and_never_reaches_the_ask():
    assert list(KD.maker_rest([69, 70, 70, 98, 1], [72, 72, 71, 99, 3])) == [70, 71, 70, 98, 2]      # a one-tick spread joins the bid's queue


def test_a_bid_fills_on_the_ask_reaching_it_or_a_taker_sale_into_it():
    nan = np.nan
    # first at its price (rest 70 over a bid of 69): a sale at 70 reaches it
    assert list(KD.maker_filled(70, 69, [nan, 71, 70, nan, nan, nan], [nan, nan, nan, 71, 70, 69])) == [False, False, True, False, True, True]
    # joined the queue at the bid (rest 70, bid 70): a sale at 70 may go to the orders ahead; one below 70 cannot
    assert list(KD.maker_filled(70, 70, [nan, 70, nan, nan], [nan, nan, 70, 69])) == [False, True, False, True]


def test_window_lows_are_the_minimum_over_the_window_after_each_decision():
    rng = np.random.default_rng(7)
    ends = np.sort(rng.choice(np.arange(0, 6000, 60), 70, replace=False)).astype(np.int64)
    vals = np.where(rng.random(70) < 0.2, np.nan, rng.integers(1, 99, 70).astype(float))
    after = rng.integers(-300, 6300, 400).astype(float); until = after + rng.integers(-120, 2400, 400)
    got = mature._window_lows(ends, vals, after, until)
    for a, u, g in zip(after, until, got):
        seg = vals[(ends > a) & (ends <= u)]; seg = seg[np.isfinite(seg)]
        assert (np.isnan(g) and not len(seg)) or g == seg.min()


def _candles(path, rows):
    pq.write_table(pa.table({"end_ts": np.array([r[0] for r in rows], np.int64), "yes_ask_low": [float(r[1]) for r in rows], "yes_bid_high": [float(r[2]) for r in rows]}), path)


def _trades(path, rows):
    pq.write_table(pa.table({"ts": pa.array([datetime.fromtimestamp(r[0], timezone.utc) for r in rows], pa.timestamp("ms", tz="UTC")),
                             "yes_price": pa.array([r[1] for r in rows], pa.int16()), "taker_side": [r[2] for r in rows]}), path)


def test_the_fill_window_is_the_orders_life(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "KALSHI_MAKER_TTL_H", 1.0); monkeypatch.setattr(config, "KALSHI_MAKER_QUIET_MIN", 30.0)
    c, t = tmp_path / "c.parquet", tmp_path / "t.parquet"
    _candles(c, [(T0, 60, 40), (T0 + 60, 58, 42), (T0 + 3600, 55, 44), (T0 + 3660, 50, 45)])
    _trades(t, [(T0 - 30, 30, "no"),                    # in the decision minute: before the order existed
                (T0 + 30, 57, "no"), (T0 + 40, 62, "yes"),
                (T0 + 3650, 20, "no"), (T0 + 3655, 80, "yes")])   # after the order's hour
    ends, fl = mature.fill_lows(str(c), str(t), anchor_close=T0 + 2 * 3600 + 1800)
    yes_ask, yes_sold = fl["yes"]; no_ask, no_sold = fl["no"]
    assert ends[0] == T0 and yes_ask[0] == 55 and yes_sold[0] == 57                      # minutes T0+60 .. T0+3600 only
    assert no_ask[0] == 100 - 44 and no_sold[0] == 100 - 62                               # NO ask = 100 − YES bid; a YES buyer sold NO
    ends, fl = mature.fill_lows(str(c), str(t), anchor_close=T0 + 1800 + 60)              # the order ends 30 min before close: T0+60
    assert fl["yes"][0][0] == 58 and fl["yes"][1][0] == 57


def test_the_label_is_the_arms_order(tmp_path):
    day = tmp_path / "2026-09-24"; day.mkdir()
    n = 4; base = {c: np.zeros(n, np.float32) for c in K_COLS}
    base["spread"][:] = 3; base["side_bid"][:] = 69; base["side_ask"][:] = 72; base["log_oi"][:] = np.log1p(1e4); base["log_vol_24h"][:] = np.log1p(1e4)
    base["eff_price"][:] = 73; base["side_mid"][:] = 70.5; base["fee_mult"][:] = 1.0
    ts = [datetime.fromtimestamp(T0 - 60 + 300 * i, timezone.utc) for i in range(n)]
    meta = {"ticker": ["M"] * n, "side": ["yes"] * n, "ts": ts, "event_ticker": ["E"] * n, "category": ["economics"] * n,
            "close_ts": [T0 + 86400.0] * n, "end_ts": [T0 + 86400.0] * n, "settled_ts": [T0 + 86400.0] * n, "result": ["yes"] * n,
            "fill_ask_low": [np.nan, 70.0, np.nan, np.nan], "fill_sold_low": [71.0, np.nan, 70.0, np.nan]}
    tab = pa.Table.from_pydict({**meta, **base}, schema=mature.SCHEMA).replace_schema_metadata(
        {b"fly_version": str(KALSHI_FEATURE_VERSION).encode(), b"stride_min": b"5", b"maker_window": mature.MAKER_WINDOW.encode()})
    pq.write_table(tab, day / "part-1.parquet")
    ds = KD.build(feature_dir=tmp_path)
    m = ds.fwd_h["maker"]
    assert np.isnan(m[0]) and np.isnan(m[3])                                              # no sale at or below 70, no ask down to it
    assert np.allclose(m[1:3], (100 - 70) / 70)                                            # filled at 70 (no maker fee on this series): won
    assert ds.gate == "significance"


def test_a_part_with_another_order_window_is_stale(tmp_path):
    p = tmp_path / "part-1.parquet"
    pq.write_table(pa.table({"x": [1]}).replace_schema_metadata({b"fly_version": str(KALSHI_FEATURE_VERSION).encode(), b"maker_window": b"24h-quiet30m"}), p)
    assert not mature.part_current(p)


def test_the_kalshi_bars_are_a_positive_mean_and_the_sign_test():
    sig = dict(n=400, win=0.55, pf=1.05, total=4.0, mean=0.01, weeks=39, weeks_pos=27, sign_ok=True)
    assert Score(**sig, gate="significance").admissible                                   # PF 1.05 and 55 % winners do not matter
    assert not Score(**{**sig, "sign_ok": False}, gate="significance").base_ok
    assert not Score(**{**sig, "mean": -0.001}, gate="significance").base_ok
    assert not Score(**sig).base_ok                                                       # the memecoin bars: PF ≥ 1.3
    h = {**Score(**sig, gate="significance").dict(), "random_mean": -0.05}
    assert holdout_passes(h)[0] and not holdout_passes({**h, "random_mean": 0.02})[0]
    assert not holdout_passes({**h, "sign_ok": False, "base_ok": False})[0] and not holdout_passes({})[0]
