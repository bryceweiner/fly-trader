import math, random, time
from fly_trader.market.features import FeatureBank, TokenMeta, FIDX, D
from fly_trader.market.danger import danger_score
from fly_trader.market.exit_cost import impact_fraction, exit_cost_fraction


def _fill(bank, mint, now, n=600, up=True):
    for i in range(n):
        p = 1e-6 * math.exp((0.001 if up else -0.001) * i)
        bank.ingest_tape_row({"ts": now - 3600 + i * 6, "side": 1 if random.random() < 0.7 else -1, "amount_quote": int(2e8),
                              "price_sol": p, "signer": f"s{i % 20}", "res_quote": int(50e9), "mint": mint})


def test_features_shape_and_signs():
    random.seed(0)
    b = FeatureBank(); now = time.time()
    _fill(b, "UP", now, up=True); _fill(b, "DN", now, up=False)
    fu, mu = b.states["UP"].features(now, TokenMeta("UP", graduated_at=now - 7200))
    fd, md = b.states["DN"].features(now, TokenMeta("DN", graduated_at=now - 7200))
    assert len(fu) == D and fu[FIDX["ret_1h"]] > 0 > fd[FIDX["ret_1h"]]
    assert fu[FIDX["imb_1h"]] > 0 and mu & (1 << FIDX["ret_1h"])
    assert abs(fu[FIDX["log_liquidity_sol"]] - math.log1p(100.0)) < 1e-6
    assert b.activity_sol("UP", now) > 0 and b.last_price("UP") > 0


def test_exit_cost_and_danger():
    assert abs(impact_fraction(0.1, 100.0) - 0.1 / 100.1) < 1e-9
    assert impact_fraction(0.1, None) == 1.0
    assert 0 < exit_cost_fraction(0.1, 100.0, 1.0) < 0.02
    f = [0.0] * D; f[FIDX["exit_cost_0p1"]] = 1.0; f[FIDX["is_sus"]] = 1.0
    assert danger_score(f, 0, 0.5) > danger_score([0.0] * D, 0, 100.0)
