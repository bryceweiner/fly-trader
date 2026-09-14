import numpy as np
from fly_trader import config
from fly_trader.agent.decide import DecisionState, decide, size_for, zscores


def _state(exposure=0.0):
    st = DecisionState(); st.exposure_frac = exposure
    return st


def test_z_thresholds_and_confirmation():
    st = _state(exposure=1.0)            # not hungry: threshold = Z_BUY
    rng = np.random.default_rng(0)
    slots = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", None]
    m = np.array([0.40, 0.00, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0, 0.0, -0.06, 0.0])   # A is > 2 spreads above the mean, J far below
    z, mu, sigma = zscores(m, np.array([True] * 10 + [False]))
    assert z[0] > config.Z_BUY and z[9] < config.Z_SELL
    assert decide(st, slots, m, held={"J"}, free_capital_sol=5.0, rng=rng) == []   # first beat: unconfirmed
    d2 = decide(st, slots, m, held={"J"}, free_capital_sol=5.0, rng=rng)
    kinds = {d.mint: d.kind for d in d2}
    assert kinds == {"A": "enter"}          # exits are reflexes, never valence


def test_hunger_lowers_threshold_and_sizing_grows_with_z():
    full, empty = _state(1.0), _state(0.0)
    assert empty.z_buy_eff() < full.z_buy_eff()
    assert size_for(config.Z_BUY, config.Z_BUY) == config.SIZE_MIN_SOL
    assert size_for(config.Z_BUY + config.Z_SIZE_RANGE, config.Z_BUY) == config.MAX_POSITION_SOL


def test_capital_constraint_softmax():
    st = _state(0.0)
    rng = np.random.default_rng(1)
    slots = ["A", "B", "C", "D", "E", "F"]
    m = np.array([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    decide(st, slots, m, held=set(), free_capital_sol=0.03, rng=rng)
    out = decide(st, slots, m, held=set(), free_capital_sol=0.03, rng=rng)
    assert out and sum(d.size_sol for d in out) <= 0.03 + 1e-9
    assert all(d.softmax_p is not None for d in out)


def test_sniff_before_feeding():
    st = _state(exposure=0.0)
    rng = np.random.default_rng(2)
    slots = ["A", "B", "C", "D", "E"]
    m = np.array([3.0, 0.0, 0.0, 0.0, 0.0])
    young = [0] * 5
    for _ in range(3):
        assert decide(st, slots, m, held=set(), free_capital_sol=5.0, rng=rng, dwell=young) == []       # still sniffing
    for _ in range(3):
        assert decide(st, slots, m, held=set(), free_capital_sol=5.0, rng=rng, dwell=[10 ** 6] * 5, allow_entries=False) == []  # warm-up
    decide(st, slots, m, held=set(), free_capital_sol=5.0, rng=rng, dwell=[10 ** 6] * 5)
    out = decide(st, slots, m, held=set(), free_capital_sol=5.0, rng=rng, dwell=[10 ** 6] * 5)
    assert [d.mint for d in out] == ["A"]
