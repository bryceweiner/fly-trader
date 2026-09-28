"""The two memecoin markets: book → market, Solana's money settings are exactly today's config, RH's are ETH and read
config at call time; K scales the fixed-unit thresholds."""
import pytest

from fly_trader import config, markets
from fly_trader.train.decisions import MIN_RESQ_SOL, MIN_VOL_15M_SOL


def test_books_and_chains():
    assert markets.for_book("paper_fly") is markets.SOL and markets.for_book("live") is markets.SOL
    assert markets.for_book("paper_rh_fly") is markets.RH and markets.for_book("live_rh") is markets.RH
    assert markets.for_book("replay_x") is markets.SOL and markets.for_chain(None) is markets.SOL
    assert markets.RH.circuit_id == 3 and markets.SOL.circuit_id == 1 and markets.RH.unit == "ETH"
    assert not set(markets.SOL.books) & set(markets.RH.books)


def test_solana_reads_todays_settings_and_rh_scales_by_k(monkeypatch):
    s = markets.SOL
    assert (s.capital(), s.label_size(), s.min_position(), s.gas_reserve(), s.k()) == \
        (config.CAPITAL_SOL, config.LABEL_SIZE_SOL, config.MIN_POSITION_SOL, config.GAS_RESERVE_SOL, 1.0)
    assert (s.min_resq(), s.min_vol_15m()) == (MIN_RESQ_SOL, MIN_VOL_15M_SOL)
    monkeypatch.setattr(config, "RH_ETH_PER_SOL", 0.05); monkeypatch.setattr(config, "RH_CAPITAL_ETH", 0.25)
    assert markets.RH.k() == 0.05 and markets.RH.capital() == 0.25
    assert markets.RH.min_resq() == pytest.approx(MIN_RESQ_SOL * 0.05) and markets.RH.min_vol_15m() == pytest.approx(MIN_VOL_15M_SOL * 0.05)


def test_pinned_rh_constants_are_the_solana_sizes_in_eth():
    k = config.RH_ETH_PER_SOL
    assert k > 0
    assert config.RH_CAPITAL_ETH == pytest.approx(5.0 * k, rel=1e-5)
    assert config.RH_LABEL_SIZE_ETH == pytest.approx(config.LABEL_SIZE_SOL * k, rel=1e-5)
    assert config.RH_MIN_POSITION_ETH == pytest.approx(config.MIN_POSITION_SOL * k, rel=1e-5)
