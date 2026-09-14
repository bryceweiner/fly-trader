import numpy as np
from types import SimpleNamespace
from fly_trader.train.env import TradingEnv, MIN_TRADE_FRAC
from fly_trader.market.features import D


def _ds(T=20, M=2, drift=0.01):
    obs = np.zeros((T, M, D), np.float16); mask = np.ones((T, M), bool)
    price = np.array([[1e-6 * (1 + drift) ** t, 1e-6 * (1 - drift) ** t] for t in range(T)], np.float32)
    ds = SimpleNamespace(obs=obs, mask=mask, price=price, resq=np.full((T, M), 50.0, np.float32), age=np.full((T, M), 48.0, np.float32),
                         T=T, M=M, D=D, mean=np.zeros(D, np.float32), std=np.ones(D, np.float32))
    ds.standardize = lambda x: x.astype(np.float32)
    return ds


def test_reward_is_net_pnl_of_the_money():
    env = TradingEnv(_ds(), 0, 20, max_pos_sol=0.1)
    obs = env.reset()
    assert obs.shape == (2, D + 3)
    a = np.array([1.0, 1.0])                       # buy both at full size
    obs, r, done, info = env.step(a)
    assert r[0] < 0 and r[1] < 0                    # entry costs fees + impact for both
    total = r.copy()
    while not done:
        obs, r, done, info = env.step(a); total += r
    assert total[0] > 0 > total[1]                  # riser made money net of fees, faller lost
    assert info["trades"] == 2 and info["fees"] > 0
    # 1 % per beat over 19 beats ≈ +20 % gross on 0.1 SOL, minus ~1 % fees → reward in % of max position
    assert 10 < total[0] < 25


def test_dust_control_and_flat_when_no_data():
    ds = _ds(); ds.mask[:, 1] = False
    env = TradingEnv(ds, 0, 5, max_pos_sol=0.1)
    env.reset()
    env.step(np.array([0.5, 1.0]))
    assert env.pos[0] == 0.5 and env.pos[1] == 0.0    # token 1 has no data: stays flat
    env.step(np.array([0.5 + MIN_TRADE_FRAC / 2, 0.0]))
    assert env.pos[0] == 0.5 and env.trades == 1       # below the dust threshold: no trade
    env.step(np.array([0.0, 0.0]))
    assert env.pos[0] == 0.0 and env.qty[0] == 0.0     # full exit always allowed
