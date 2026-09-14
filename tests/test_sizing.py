"""Position sizing: growth-optimal fractions, score bands, caps and the bankroll replay."""
import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import sizing


def test_kelly_counts_the_tail_and_needs_an_edge():
    assert sizing.kelly_fraction(np.array([-0.01, 0.005, -0.02])) == 0.0                    # no edge
    wins = np.full(90, 0.10); rugs = np.full(10, -1.0)
    assert sizing.kelly_fraction(np.r_[wins, rugs]) == 0.0                                  # +10 % x 90 cannot pay for 10 total losses (mean -1 %)
    f = sizing.kelly_fraction(np.r_[np.full(95, 0.10), np.full(5, -1.0)])                  # mean +4.5 %: a small bet
    assert 0.0 < f < 0.5
    f_even = sizing.kelly_fraction(np.r_[np.full(50, 0.2), np.full(50, -0.1)])             # classic: p/L - q/W = 0.5/0.1 - 0.5/0.2 = 2.5 -> capped by the grid
    assert f_even == pytest.approx(0.99)


def test_bands_rank_by_margin_and_higher_certainty_bets_more(monkeypatch):
    rng = np.random.default_rng(0); m = rng.uniform(0, 0.1, 4000)
    r = np.where(rng.random(4000) < 0.86 + 0.8 * m, 0.08, -0.50)                            # +8 % wins vs -50 % crashes; the win rate rises with the margin
    table = sizing.build_table(m, r)
    assert len(table) == sizing.N_BANDS and table[0]["lo"] == 0.0 and [b["lo"] for b in table] == sorted(b["lo"] for b in table)
    assert table[-1]["kelly"] > table[0]["kelly"]
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3); monkeypatch.setattr(config, "KELLY_FRACTION", 0.25); monkeypatch.setattr(config, "MAX_POSITION_FRACTION", 1.0)
    monkeypatch.setattr(config, "MAX_POOL_SHARE", 1.0); monkeypatch.setattr(config, "MIN_POSITION_SOL", 0.0)
    lo, _ = sizing.size_position(0.5 + 0.001, 0.5, table, 10.3, 10.3, 1000.0)
    hi, _ = sizing.size_position(0.5 + 0.099, 0.5, table, 10.3, 10.3, 1000.0)
    assert hi > lo >= 0 and hi > 0                                                           # the weakest band may have no edge at all


def test_caps_minimum_and_fallback(monkeypatch):
    table = [{"lo": 0.0, "n": 100, "mean": 0.05, "win": 0.7, "kelly": 0.8}]
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3); monkeypatch.setattr(config, "KELLY_FRACTION", 0.25); monkeypatch.setattr(config, "MAX_POSITION_FRACTION", 0.10)
    monkeypatch.setattr(config, "MAX_POOL_SHARE", 0.02); monkeypatch.setattr(config, "MIN_POSITION_SOL", 0.02); monkeypatch.setattr(config, "MAX_POSITION_SOL", 0.1)
    s, why = sizing.size_position(0.9, 0.8, table, 5.3, 5.3, 1000.0)
    assert s == pytest.approx(0.5) and "bankroll cap" in why                                 # 0.25*0.8*5 = 1.0 -> 10 % of 5 = 0.5
    s, why = sizing.size_position(0.9, 0.8, table, 5.3, 5.3, 10.0)
    assert s == pytest.approx(0.2) and "pool depth" in why                                   # 2 % of a 10 SOL pool
    s, why = sizing.size_position(0.9, 0.8, table, 5.3, 0.31, 1000.0)
    assert s == 0.0 and "minimum" in why                                                     # only 0.01 SOL free above the reserve
    assert sizing.size_position(0.7, 0.8, table, 5.3, 5.3, 1000.0)[0] == 0.0                 # below the line: no band
    assert sizing.size_position(0.9, 0.8, [], 5.3, 5.3, 1000.0)[0] == pytest.approx(0.1)     # no table: the fixed size


def test_bankroll_replay_enforces_cash_and_compounds(monkeypatch):
    monkeypatch.setattr(config, "GAS_RESERVE_SOL", 0.3); monkeypatch.setattr(config, "KELLY_FRACTION", 0.25); monkeypatch.setattr(config, "MAX_POSITION_FRACTION", 0.10)
    monkeypatch.setattr(config, "MIN_POSITION_SOL", 0.02); monkeypatch.setattr(config, "MAX_POSITION_SOL", 0.1)
    ts = np.arange(200) * 3600.0; m = np.full(200, 0.05); r = np.full(200, 0.05)                  # every trade +5 %, one at a time
    table = [{"lo": 0.0, "n": 200, "mean": 0.05, "win": 1.0, "kelly": 0.99}]
    sized = sizing.simulate_bankroll(ts, 1800.0, m, r, table, start_sol=5.0); fixed = sizing.simulate_bankroll(ts, 1800.0, m, r, None, start_sol=5.0)
    assert sized["final_sol"] > fixed["final_sol"] > 5.0 and sized["max_drawdown"] == 0.0 and sized["trades"] == 200
