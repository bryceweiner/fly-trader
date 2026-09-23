"""The fly's daily calibration: the vectorised one-position-per-token rule equals the original loop, the line with the
most total profit over enough trades wins (quantile candidates included), and too few trades keep the previous line."""
import numpy as np
import pytest

from fly_trader.train import fly_calibrate
from fly_trader.train.decisions import taken_idx


def _loop(ts, mint, horizon_s, idx):
    idx = np.asarray(idx, dtype=int)
    idx = idx[np.lexsort((ts[idx], mint[idx]))]; out = []; last_mint = None; last_t = -1e18
    for i in idx:
        if mint[i] != last_mint:
            last_mint = mint[i]; last_t = -1e18
        if ts[i] >= last_t + horizon_s:
            out.append(i); last_t = ts[i]
    return np.asarray(out, dtype=int)


@pytest.mark.parametrize("seed", range(5))
def test_taken_idx_equals_the_loop(seed):
    rng = np.random.default_rng(seed); n = 3000
    ts = rng.integers(0, 20000, n).astype(float) * 60; mint = rng.choice([f"m{i}" for i in range(40)], n)
    idx = rng.choice(n, 1200, replace=False)
    assert np.array_equal(taken_idx(ts, mint, 1800.0, idx), _loop(ts, mint, 1800.0, idx))
    assert len(taken_idx(ts, mint, 1800.0, np.array([], dtype=int))) == 0


def test_calibrate_picks_the_most_profitable_line_and_keeps_it_when_too_few_trades():
    rng = np.random.default_rng(0); n = 20000
    ts = np.arange(n, dtype=float) * 60; mint = np.array([f"m{i % 500}" for i in range(n)])
    scores = rng.normal(0, 0.001, n)                                               # a compressed scale: fixed candidates alone never trade
    rets = np.where(scores > np.quantile(scores, 0.98), 0.02, -0.01) + rng.normal(0, 0.001, n)
    cal = fly_calibrate.calibrate(ts, mint, 1800.0, scores, rets)
    assert cal.changed and cal.trades >= 100 and cal.mean > 0 and cal.line >= np.quantile(scores, 0.95) - 1e-12
    assert cal.line in fly_calibrate.candidates(scores) and cal.sizing
    few = fly_calibrate.calibrate(ts[:50], mint[:50], 1800.0, scores[:50], rets[:50], prev_line=0.123, prev_sizing=[{"lo": 0.0, "kelly": 0.1}])
    assert few.line == 0.123 and not few.changed and few.sizing == [{"lo": 0.0, "kelly": 0.1}]


def test_a_losing_line_never_wins_even_when_it_is_the_only_one_with_enough_trades():
    """With a day of resolved rows only the lowest line clears 100 trades; on 2026-09-23 it took the live fly from a 4.1 %
    line to 0.25 % on 119 trades that lost 1.1 % each. Most total profit means profit: a losing line keeps the previous one."""
    rng = np.random.default_rng(1); n = 400
    ts = np.arange(n, dtype=float) * 60; mint = np.array([f"m{i % 200}" for i in range(n)])
    scores = rng.uniform(0.0, 0.03, n); rets = np.full(n, -0.011) + rng.normal(0, 0.001, n)            # everything above 0.0025 trades and loses
    cal = fly_calibrate.calibrate(ts, mint, 1800.0, scores, rets, prev_line=0.0406, prev_sizing=[{"lo": 0.0, "kelly": 0.2}])
    assert cal.line == 0.0406 and not cal.changed                                                       # the bootstrap's line survives the bad day
    assert any(t["trades"] >= 100 and t["total"] < 0 for t in cal.lines)                                # the losing candidate was there and was refused
    assert fly_calibrate.calibrate(ts, mint, 1800.0, scores, rets).line == fly_calibrate.MIN_EV           # no previous line: the floor, never the loser
