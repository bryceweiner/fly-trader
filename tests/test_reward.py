import math
from fly_trader import config
from fly_trader.agent.reward import RewardTracker, mark_book


def test_mark_book_and_attribution():
    positions = [{"mint": "A", "qty": 1_000_000, "decimals": 6, "cost_sol": 0.1, "entry_price": 0.0001},
                 {"mint": "B", "qty": 2_000_000, "decimals": 6, "cost_sol": 0.1, "entry_price": 0.00005}]
    wm, per = mark_book(4.8, positions, {"A": 0.0002, "B": 0.00005}, {"A": 100.0, "B": 100.0}, {"A": 30.0, "B": 30.0}, {"A": None, "B": None})
    assert wm.n_open == 2 and wm.wealth > 4.8 and abs(wm.wealth - (4.8 + wm.positions_value)) < 1e-9
    assert per["A"] > per["B"]
    rt = RewardTracker()
    d, m_prev = rt.rpe("A", 1.5, 0.2)
    assert m_prev == 0.0 and abs(d - ((1 - config.GAMMA) * 1.5 + config.GAMMA * 0.2)) < 1e-9
    rt.remember_m_hat("A", 0.2)
    d2, m_prev2 = rt.rpe("A", 0.0, 0.2)
    assert m_prev2 == 0.2 and abs(d2 - (config.GAMMA * 0.2 - 0.2)) < 1e-9
