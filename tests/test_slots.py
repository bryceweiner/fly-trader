from fly_trader import config
from fly_trader.agent.slots import SlotManager


def test_pins_floor_and_rotation():
    sm = SlotManager(4)
    cands = {"A": 5.0, "B": 0.1, "C": 2.0, "D": 3.0, "E": 9.0, "F": 1.5}
    s1 = sm.assign(1000.0, held={"B"}, candidates=cands, m_hat_by_mint={})
    assert "B" in s1                      # held is pinned even below the floor
    assert None not in s1 and len(set(s1)) == 4
    # after dwell expiry with low valence, unvisited tokens rotate in
    for i in range(config.DWELL_BEATS + 1):
        s2 = sm.assign(1000.0 + i, held={"B"}, candidates=cands, m_hat_by_mint={m: -1.0 for m in cands})
    visited = set(sm.visits)
    assert visited >= {m for m, a in cands.items() if a >= config.ACTIVITY_FLOOR_SOL_3H}
    assert "B" in s2


def test_building_signal_not_evicted():
    sm = SlotManager(2)
    cands = {"A": 5.0, "B": 5.0, "C": 5.0}
    s = sm.assign(0.0, set(), cands, {})
    keep = s[0]
    for i in range(1, 2 * config.DWELL_BEATS):
        s = sm.assign(float(i), set(), cands, {keep: 0.5})
        assert keep in s
