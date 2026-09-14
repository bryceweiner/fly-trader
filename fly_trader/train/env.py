"""Vectorized trading environment over the cached dataset (offline) with the live fee model.

One "body" per token; the policy is shared. Action a ∈ [0, 1] = target exposure as a fraction of
MAX_POSITION_SOL. Trades execute only when |a − pos| ≥ MIN_TRADE_FRAC (dust control) at the last trade price
with the round-trip cost model of market/exit_cost.py (Jupiter fee by token age + pool fee + constant-product
impact against the quote reserve). Reward per beat = change in the position's marked value plus cash flows,
in percent of MAX_POSITION_SOL — i.e. the money, net of fees. Tokens without data are flat and rewardless.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .. import config
from ..market.exit_cost import POOL_FEE_BPS, jupiter_fee_bps
from .dataset import Dataset

PORTFOLIO_DIM = 3          # pos, unrealized return, log1p(beats held)
MIN_TRADE_FRAC = 0.25
DEFAULT_RESQ_SOL = 20.0


def fee_frac(age_h: float | None) -> float:
    return (jupiter_fee_bps(age_h) + POOL_FEE_BPS) / 1e4


class TradingEnv:
    def __init__(self, ds: Dataset, t_start: int, t_end: int, max_pos_sol: float | None = None):
        self.ds = ds
        self.t_start, self.t_end = t_start, t_end
        self.M = ds.M
        self.max_pos = max_pos_sol or config.MAX_POSITION_SOL
        self.obs_dim = ds.D + PORTFOLIO_DIM
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = self.t_start
        self.pos = np.zeros(self.M, np.float32)      # exposure fraction
        self.qty = np.zeros(self.M, np.float64)      # tokens held
        self.cost = np.zeros(self.M, np.float64)     # SOL paid incl. fees
        self.held = np.zeros(self.M, np.int32)
        self.trades = 0
        self.turnover = 0.0
        self.fees = 0.0
        return self.observe()

    # ---- helpers ----
    def _price(self, t: int) -> np.ndarray:
        return self.ds.price[t]

    def _resq(self, t: int) -> np.ndarray:
        r = self.ds.resq[t].astype(np.float64)
        return np.where(np.isfinite(r) & (r > 0), r, DEFAULT_RESQ_SOL)

    def _fee(self, t: int) -> np.ndarray:
        age = self.ds.age[t]
        return np.array([fee_frac(a if np.isfinite(a) else None) for a in age], dtype=np.float64)

    def value(self, t: int) -> np.ndarray:
        """Marked value of each position after exit cost, SOL."""
        p = self._price(t).astype(np.float64)
        gross = np.where(np.isfinite(p), self.qty * p, self.cost)   # no price: hold at cost
        impact = gross / (gross + self._resq(t))
        return gross * (1.0 - self._fee(t) - impact)

    def observe(self) -> np.ndarray:
        t = min(self.t, self.ds.T - 1)
        f = self.ds.standardize(self.ds.obs[t])
        f[~self.ds.mask[t]] = 0.0
        p = self._price(t).astype(np.float64)
        entry = np.where(self.qty > 0, self.cost / np.maximum(self.qty, 1e-12), np.nan)
        unreal = np.where(self.qty > 0, p / entry - 1.0, 0.0)
        unreal = np.nan_to_num(unreal, nan=0.0, posinf=0.0, neginf=0.0).clip(-1, 5)
        port = np.stack([self.pos, unreal, np.log1p(self.held)], axis=1).astype(np.float32)
        return np.concatenate([f, port], axis=1)

    def tradable(self) -> np.ndarray:
        t = min(self.t, self.ds.T - 1)
        return self.ds.mask[t] & np.isfinite(self.ds.price[t])

    def step(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool, dict]:
        """action [M] in [0,1]. Returns obs_next, reward [M] (% of max position), done, info."""
        t = self.t
        p = self._price(t).astype(np.float64)
        ok = self.tradable()
        target = np.clip(action, 0.0, 1.0)
        target = np.where(ok, target, 0.0)                     # no data → go flat
        v_before = self.value(t)
        delta = target - self.pos
        do = np.abs(delta) >= MIN_TRADE_FRAC
        do |= (target == 0.0) & (self.pos > 0)                # full exits always allowed
        cash = np.zeros(self.M, np.float64)
        fee = self._fee(t)
        resq = self._resq(t)
        buy = do & (delta > 0) & ok
        if buy.any():
            notional = delta[buy] * self.max_pos
            impact = notional / (notional + resq[buy])
            eff_price = p[buy] * (1.0 + impact)
            got = notional * (1.0 - fee[buy]) / eff_price
            self.qty[buy] += got
            self.cost[buy] += notional
            cash[buy] -= notional
            self.fees += float((notional * (fee[buy] + impact)).sum())
            self.turnover += float(notional.sum())
            self.pos[buy] = target[buy]
        sell = do & (delta < 0)
        if sell.any():
            frac = np.clip((-delta[sell]) / np.maximum(self.pos[sell], 1e-9), 0.0, 1.0)
            q = self.qty[sell] * frac
            gross = np.where(np.isfinite(p[sell]), q * p[sell], self.cost[sell] * frac)
            impact = gross / (gross + resq[sell])
            proceeds = gross * (1.0 - fee[sell] - impact)
            cash[sell] += proceeds
            self.fees += float((gross * (fee[sell] + impact)).sum())
            self.turnover += float(gross.sum())
            self.qty[sell] -= q
            self.cost[sell] *= (1.0 - frac)
            self.pos[sell] = np.where(frac >= 0.999, 0.0, target[sell])
            zero = self.qty <= 1e-12
            self.qty[zero] = 0.0; self.cost[zero] = 0.0; self.pos[zero] = 0.0
        self.trades += int(do.sum())
        self.held = np.where(self.qty > 0, self.held + 1, 0)
        self.t = t + 1
        done = self.t >= self.t_end
        t_next = min(self.t, self.ds.T - 1)
        v_after = self.value(t_next)
        reward = 100.0 * (v_after - v_before + cash) / self.max_pos
        reward = np.nan_to_num(reward, nan=0.0)
        if done:   # force liquidation at the end so nothing is left unrealised
            gross = self.qty * np.nan_to_num(self._price(t_next).astype(np.float64), nan=0.0)
            impact = gross / (gross + resq)
            liq = gross * (1.0 - fee - impact)
            reward += 100.0 * (liq - v_after) / self.max_pos
            self.fees += float((gross * (fee + impact)).sum())
            self.qty[:] = 0; self.cost[:] = 0; self.pos[:] = 0
        info = {"trades": self.trades, "turnover": self.turnover, "fees": self.fees}
        return self.observe(), reward.astype(np.float32), done, info
