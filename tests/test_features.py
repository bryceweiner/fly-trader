import math, random, time
from fly_trader.market.features import TokenMeta, TokenState, FIDX, D
import numpy as np
import pytest
from fly_trader.market.exit_cost import impact_fraction, exit_cost_fraction, fee_fraction, pool_fee_rate


def _fill(mint, now, n=600, up=True):
    st = TokenState(mint)
    for i in range(n):
        p = 1e-6 * math.exp((0.001 if up else -0.001) * i)
        st.append(now - 3600 + i * 6, p, 0.2, random.random() < 0.7, f"s{i % 20}", 50.0)
    return st


def test_features_shape_and_signs():
    random.seed(0)
    now = time.time()
    up, dn = _fill("UP", now, up=True), _fill("DN", now, up=False)
    fu, mu = up.features(now, TokenMeta("UP", graduated_at=now - 7200))
    fd, md = dn.features(now, TokenMeta("DN", graduated_at=now - 7200))
    assert len(fu) == D and fu[FIDX["ret_1h"]] > 0 > fd[FIDX["ret_1h"]]
    assert fu[FIDX["imb_1h"]] > 0 and mu & (1 << FIDX["ret_1h"])
    assert abs(fu[FIDX["log_liquidity_sol"]] - math.log1p(100.0)) < 1e-6
    assert up.volume_since(now - 10800) > 0 and up.last_price > 0


def test_exit_cost():
    assert abs(impact_fraction(0.1, 100.0) - 0.1 / 100.1) < 1e-9
    assert impact_fraction(0.1, None) == 1.0
    assert 0 < exit_cost_fraction(0.1, 100.0, 1.0) < 0.02
    # measured pump.fun schedule: 1.25 % below 420 SOL market cap, stepping down to 0.30 % above 98,178 SOL
    assert pool_fee_rate(100.0) == 0.0125 and pool_fee_rate(5000.0) == 0.01 and pool_fee_rate(2e5) == 0.003
    assert pool_fee_rate(None) == 0.0125 and list(pool_fee_rate(np.array([100.0, np.nan, 2e5]))) == [0.0125, 0.0125, 0.003]
    # fees = pool fee + Jupiter 10 bps + 5,000-lamport network fee; a stream-reported pool fee overrides the schedule
    assert fee_fraction(0.1, 2e5) == pytest.approx(0.003 + 0.001 + 5e-6 / 0.1)
    assert fee_fraction(0.1, 2e5, pool_fee=0.0095) == pytest.approx(0.0095 + 0.001 + 5e-6 / 0.1)


def test_paper_costs_beyond_the_label_model():
    """A stream fee below the lowest tier (PumpSwap minutes reporting 0.0005) is no rate a swap of ours pays: the tier is
    charged. A Pons coin quoted in another asset than ETH pays its route through the base (one more pool's fee, a
    multi-hop route's gas) under its own label; features and labels never use that label, and its impact is the pool's."""
    from fly_trader import config
    from fly_trader.market.exit_cost import PONS_LABEL, PONS_MULTIHOP_LABEL, cost_label
    assert fee_fraction(0.1, 1300.0, pool_fee=0.0005) == pytest.approx(fee_fraction(0.1, 1300.0)) == pytest.approx(0.012 + 0.001 + 5e-6 / 0.1)
    assert fee_fraction(0.1, 2e5, pool_fee=0.003) == pytest.approx(0.003 + 0.001 + 5e-6 / 0.1)          # the floor tier itself stands
    assert cost_label(PONS_LABEL, "stock") == cost_label(PONS_LABEL, "stable") == PONS_MULTIHOP_LABEL
    assert cost_label(PONS_LABEL, "eth") == cost_label(PONS_LABEL, None) == PONS_LABEL and cost_label("Pump.fun Amm", "stock") == "Pump.fun Amm"
    v = 0.01
    assert fee_fraction(v, None, 0.03, PONS_MULTIHOP_LABEL) == pytest.approx(0.03 + config.RH_HOP_FEE + config.RH_TX_FEE_MULTIHOP_ETH / v)
    assert fee_fraction(v, None, 0.03, PONS_LABEL) == pytest.approx(0.03 + config.RH_TX_FEE_ETH / v)
    assert fee_fraction(v, None, None, PONS_MULTIHOP_LABEL) == pytest.approx(config.RH_HOOK_FEE + config.RH_HOP_FEE + config.RH_TX_FEE_MULTIHOP_ETH / v)
    assert impact_fraction(v, 5.0, PONS_MULTIHOP_LABEL) == impact_fraction(v, 5.0, PONS_LABEL)
