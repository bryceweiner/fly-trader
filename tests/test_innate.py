import numpy as np
from fly_trader import config
from fly_trader.agent.innate import EdgeTracker, innate_score
from fly_trader.market.features import D, FIDX


def test_innate_score_orders_by_signal():
    feats = np.zeros((5, D), dtype=np.float32)
    feats[:, FIDX["imb_5m"]] = [0.9, 0.1, -0.5, 0.3, 0.0]
    feats[:, FIDX["ret_5m"]] = [0.05, -0.02, -0.04, 0.01, 0.0]      # rising smells better than falling
    feats[:, FIDX["organic_score"]] = [0.1, 0.9, 0.5, 0.2, 0.3]
    active = np.array([True, True, True, True, False])
    s = innate_score(feats, active)
    assert s[4] == 0.0 and s[0] > s[3] > s[1] and abs(s[active].mean()) < 1e-6


def test_edge_tracker_measures_correlation(monkeypatch):
    monkeypatch.setattr(config, "EDGE_MIN_SAMPLES", 10)
    monkeypatch.setattr(config, "EDGE_HORIZON_S", 10.0)
    monkeypatch.setattr(config, "EDGE_WINDOW_S", 1000.0)
    prices = {}
    def price_of(m): return 1.0
    def price_at(m, t): return prices[m]
    et = EdgeTracker()
    mints = [f"m{i}" for i in range(20)]
    val = np.linspace(-1, 1, 20)
    for m, v in zip(mints, val):
        prices[m] = 1.0 + 0.05 * v          # forward return proportional to valence
    et.record(0.0, mints, val, price_of)
    assert et.edge()[0] is None
    et.update(20.0, price_at)
    e, n = et.edge()
    assert n == 20 and e > 0.99


def test_eligibility_rules():
    from fly_trader.agent.innate import eligible
    feats = np.zeros((3, D), dtype=np.float32)
    masks = np.full(3, (1 << 62) | ((1 << 50) - 1), dtype=np.int64)
    active = np.array([True, True, True])
    for k in range(3):
        feats[k, FIDX["ret_1m"]] = 0.01; feats[k, FIDX["ret_5m"]] = 0.02
        feats[k, FIDX["imb_5m"]] = 0.5; feats[k, FIDX["imb_15m"]] = 0.4; feats[k, FIDX["dd_1h"]] = -0.01
    feats[1, FIDX["ret_5m"]] = -0.01          # falling
    e = eligible(feats, masks, active, [10.0, 10.0, 1.0])   # third is too young
    assert e.tolist() == [True, False, False]
