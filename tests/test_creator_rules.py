"""The creator rules of the memecoin universe (train/selector.py, 2026-10-28/29): a serial rugger's coin is never traded,
nor a coin whose creator owned most of its supply (its launch buy, or its holding at graduation with the launch block's
wallets), nor one whose launch or curve is unknown; a creator-heavy buy that reached sizing would stake at most
CREATOR_HEAVY_CAP_FRACTION of the bankroll (agent/sizing.py)."""
import math

import numpy as np
import pytest

from fly_trader import config
from fly_trader.agent import sizing
from fly_trader.markets import RH
from fly_trader.train import selector
from fly_trader.train.decisions import X_COLS


def _row(age_h=None, dev_share=0.0, prior_known=0.0, prior_rug_share=0.0, dev_hold_share=0.0, bundle_share=0.0, meta_known=1.0, curve_known=1.0):
    x = np.zeros(len(X_COLS), np.float32)
    if age_h is not None:
        x[X_COLS.index("age_known")] = 1.0; x[X_COLS.index("log_age_h")] = math.log1p(age_h)
    x[X_COLS.index("dev_share")] = dev_share; x[X_COLS.index("prior_known")] = prior_known; x[X_COLS.index("prior_rug_share")] = prior_rug_share
    x[X_COLS.index("dev_hold_share")] = dev_hold_share; x[X_COLS.index("bundle_share")] = bundle_share
    x[X_COLS.index("meta_known")] = meta_known; x[X_COLS.index("curve_known")] = curve_known
    return x


def _uni(**kw):
    return bool(selector.in_universe(_row(**kw)[None], X_COLS)[0])


def test_a_coin_whose_creator_owned_most_of_the_supply_is_never_traded():
    assert _uni(age_h=9.0, dev_share=0.3)                                   # an ordinary coin: any age past 6 h
    assert not _uni(age_h=5.0, dev_share=0.3)                               # the 6 h minimum still holds
    for h in (6.0, 7.5, 8.0, 20.0, None):                                   # XTO: 79 % at launch, bought at 6 and 7.5 h
        assert not _uni(age_h=h, dev_share=0.7931)
    assert not _uni(age_h=9.0, dev_share=0.3, dev_hold_share=0.75)          # bought the rest on the curve
    assert not _uni(age_h=9.0, dev_share=0.79, dev_hold_share=0.0)          # sold on the curve: its launch buy still counts
    assert _uni(age_h=9.0, dev_share=0.7) and _uni(age_h=9.0, dev_hold_share=0.7)   # more than 70 %, not 70 % itself


def test_the_launch_blocks_wallets_count_as_the_creators():
    assert not _uni(age_h=9.0, dev_share=0.3, dev_hold_share=0.3, bundle_share=0.45)        # 75 % between them
    assert not _uni(age_h=9.0, dev_share=0.0, dev_hold_share=0.0, bundle_share=0.75)        # the creator kept none itself
    assert _uni(age_h=9.0, dev_share=0.3, dev_hold_share=0.3, bundle_share=0.35)            # 65 % between them


def test_a_coin_whose_launch_or_curve_is_unknown_is_never_traded():
    assert not _uni(age_h=9.0, meta_known=0.0)                              # no launch seen: its creator's buy is unknown
    assert not _uni(age_h=9.0, curve_known=0.0)                             # no curve seen: who held at graduation is unknown
    assert not _uni(age_h=9.0, curve_known=np.nan)                          # a training row's NULL fails as well
    assert not _uni(age_h=None, meta_known=0.0, curve_known=0.0)            # graduated before the archive: no creator facts
    x = _row(age_h=9.0, dev_share=0.3); x[X_COLS.index("dev_hold_share")] = np.nan; x[X_COLS.index("bundle_share")] = np.nan
    assert bool(selector.in_universe(x[None], X_COLS)[0])                   # a NaN share with the facts known is no share


def test_a_serial_ruggers_coin_is_never_traded():
    assert not _uni(age_h=9.0, prior_known=2, prior_rug_share=0.5)
    assert _uni(age_h=9.0, prior_known=2, prior_rug_share=0.4)
    assert _uni(age_h=9.0, prior_known=0, prior_rug_share=0.0)              # no history: not a serial rugger
    assert _uni(age_h=None)                                                 # age unknown but its creator known: as before


def test_a_universe_without_the_creator_inputs_keeps_the_age_rule_alone():
    cols = ["age_known", "log_age_h"]
    assert list(selector.in_universe(np.array([[1, math.log1p(30.0)], [1, math.log1p(5.0)], [0, 0]]), cols)) == [True, False, True]


def test_creator_heavy():
    assert selector.creator_heavy(0.7931) and not selector.creator_heavy(0.7)
    assert not selector.creator_heavy(None) and not selector.creator_heavy(float("nan"))


def test_a_creator_heavy_buy_stakes_at_most_its_share_of_the_bankroll():
    table = [{"lo": 0.0, "n": 100, "mean": 0.05, "win": 0.5, "kelly": 0.27}]
    size = lambda bankroll, **kw: sizing.size_position(0.1, 0.06, table, bankroll, bankroll, 1000.0, **kw)
    share = config.CREATOR_HEAVY_CAP_FRACTION
    for bankroll in (1.5, 5.0, 10.0):                                       # 0.1 SOL of a 5 SOL bankroll, 0.03 of 1.5, 0.2 of 10
        (full, _), (capped, why) = size(bankroll), size(bankroll, creator_heavy=True)
        assert full > share * bankroll and capped == pytest.approx(share * bankroll) and "creator-heavy cap" in why
    assert size(0.8)[0] > 0 and size(0.8, creator_heavy=True)[0] == 0.0     # its share of 0.8 SOL is below the minimum buy: none
    full_rh, capped_rh = size(0.2, market=RH)[0], size(0.2, market=RH, creator_heavy=True)[0]
    assert capped_rh == pytest.approx(share * 0.2) and capped_rh < full_rh  # RH: its share of the ETH bankroll
