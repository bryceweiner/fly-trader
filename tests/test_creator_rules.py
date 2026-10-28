"""The creator rules of the memecoin universe (train/selector.py, 2026-10-28): a serial rugger's coin is never traded; a
coin whose creator bought most of its supply at launch only while its position is sold before the coin is 12 h old, and
at most CREATOR_HEAVY_CAP_SOL a buy (agent/sizing.py)."""
import math

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import sizing
from fly_trader.markets import RH
from fly_trader.train import selector, strategies
from fly_trader.train.decisions import X_COLS


def _row(age_h=None, dev_share=0.0, prior_known=0.0, prior_rug_share=0.0):
    x = np.zeros(len(X_COLS), np.float32)
    if age_h is not None:
        x[X_COLS.index("age_known")] = 1.0; x[X_COLS.index("log_age_h")] = math.log1p(age_h)
    x[X_COLS.index("dev_share")] = dev_share; x[X_COLS.index("prior_known")] = prior_known; x[X_COLS.index("prior_rug_share")] = prior_rug_share
    return x


def _uni(**kw):
    return bool(selector.in_universe(_row(**kw)[None], X_COLS)[0])


def test_a_creator_heavy_coin_trades_only_while_its_hold_ends_before_12_h():
    assert _uni(age_h=9.0, dev_share=0.3)                                   # an ordinary coin: any age past 6 h
    assert not _uni(age_h=5.0, dev_share=0.79)                              # the 6 h minimum still holds
    assert _uni(age_h=7.0, dev_share=0.79) and _uni(age_h=8.0, dev_share=0.79)
    assert not _uni(age_h=8.5, dev_share=0.79) and not _uni(age_h=20.0, dev_share=0.79)
    assert not _uni(age_h=None, dev_share=0.79)                             # an unknown age can't be shown to be young
    assert _uni(age_h=9.0, dev_share=0.7)                                   # more than 70 %, not 70 % itself
    assert selector.LONGEST_HOLD_H * 60 >= max(strategies.HOLDS_MIN)        # every hold a strategy may fit ends in time
    assert selector.CREATOR_HEAVY_SELL_BY_H - selector.LONGEST_HOLD_H >= selector.MIN_AGE_H


def test_a_serial_ruggers_coin_is_never_traded():
    assert not _uni(age_h=9.0, prior_known=2, prior_rug_share=0.5)
    assert _uni(age_h=9.0, prior_known=2, prior_rug_share=0.4)
    assert _uni(age_h=9.0, prior_known=0, prior_rug_share=0.0)              # no history: not a serial rugger
    assert _uni(age_h=None)                                                 # graduated before the archive, no meta: as before


def test_a_universe_without_the_creator_inputs_keeps_the_age_rule_alone():
    cols = ["age_known", "log_age_h"]
    assert list(selector.in_universe(np.array([[1, math.log1p(30.0)], [1, math.log1p(5.0)], [0, 0]]), cols)) == [True, False, True]


def test_creator_heavy():
    assert selector.creator_heavy(0.7931) and not selector.creator_heavy(0.7)
    assert not selector.creator_heavy(None) and not selector.creator_heavy(float("nan"))


def test_a_creator_heavy_buy_stakes_at_most_the_cap():
    table = [{"lo": 0.0, "n": 100, "mean": 0.05, "win": 0.5, "kelly": 0.27}]
    kw = dict(score=0.1, threshold=0.06, table=table, bankroll_sol=5.66, cash_sol=5.66, res_quote_sol=1000.0)
    full, _ = sizing.size_position(**kw)
    capped, why = sizing.size_position(**kw, creator_heavy=True)
    assert full > config.CREATOR_HEAVY_CAP_SOL and capped == pytest.approx(config.CREATOR_HEAVY_CAP_SOL) and "creator-heavy cap" in why
    poor = {**kw, "bankroll_sol": 0.8, "cash_sol": 0.8}
    small, _ = sizing.size_position(**poor, creator_heavy=True); kelly_small, _ = sizing.size_position(**poor)
    assert 0 < kelly_small < config.CREATOR_HEAVY_CAP_SOL and small == pytest.approx(kelly_small)           # never more than Kelly
    rh = dict(kw, bankroll_sol=0.2, cash_sol=0.2, market=RH)
    full_rh, _ = sizing.size_position(**rh)
    capped_rh, _ = sizing.size_position(**rh, creator_heavy=True)
    assert capped_rh == pytest.approx(min(full_rh, config.CREATOR_HEAVY_CAP_SOL * RH.k())) and capped_rh < full_rh      # 0.1 SOL's worth of ETH
