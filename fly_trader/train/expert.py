"""Rule-based expert for imitation warm-start (FlyGM's first stage), built from the 2026-09-13 backtest that
was positive net of fees: enter tokens whose within-beat rank score (buy imbalance 1m/5m/15m, nearness to the
1-hour high, trend, low organic score) is ≥ Z_BUY spreads above the mean for CONFIRM beats (threshold lowered
by hunger), size by z, exit on -2 % (bitter), a winner turning negative, satiety Σ max(u,0)·dt ≥ target, or
the -50 % stop. Runs on the cached dataset through TradingEnv and returns (obs, expert target) sequences."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .. import config
from ..market.features import FIDX
from .env import TradingEnv

SCORE_FEATURES = [("ret_5m", 2.0), ("ret_1m", 1.0), ("ret_15m", 1.0), ("imb_15m", 1.0), ("imb_5m", 1.0), ("imb_1m", 1.0), ("dd_1h", 1.0), ("organic_score", -1.0)]


def rank_score(feats: np.ndarray, active: np.ndarray) -> np.ndarray:
    n = feats.shape[0]
    out = np.zeros(n, np.float32)
    idx = np.where(active)[0]
    if len(idx) < 3:
        return out
    score = np.zeros(len(idx))
    for name, w in SCORE_FEATURES:
        score += w * pd.Series(feats[idx, FIDX[name]].astype(np.float64)).rank(method="average", pct=True).to_numpy()
    out[idx] = (score - score.mean()) / max(score.std(), 1e-6)
    return out


class ExpertPolicy:
    def __init__(self, env: TradingEnv, z_buy: float = 2.0, z_hunger: float = 0.5, confirm: int = 2, satiety_target: float = 10.0,
                 exit_loss: float = 0.02, turn_margin: float = 0.01, hard_stop: float = 0.5):
        self.env = env
        self.z_buy, self.z_hunger, self.confirm = z_buy, z_hunger, confirm
        self.satiety_target, self.exit_loss, self.turn_margin, self.hard_stop = satiety_target, exit_loss, turn_margin, hard_stop
        M = env.M
        self.above = np.zeros(M, np.int32)
        self.peak = np.zeros(M, np.float64)
        self.sat = np.zeros(M, np.float64)
        self.entry = np.zeros(M, np.float64)

    def sync(self) -> None:
        """Re-derive per-position state from the env (used when the env is driven by another policy)."""
        env = self.env
        held = env.qty > 0
        entry = np.where(held, env.cost / np.maximum(env.qty, 1e-18), 0.0)
        newly = held & (self.entry == 0)
        self.entry = np.where(held, np.where(newly, entry, self.entry), 0.0)
        self.peak = np.where(held, np.where(newly, entry, self.peak), 0.0)
        self.sat = np.where(held, self.sat, 0.0)

    def act(self) -> np.ndarray:
        env = self.env
        self.sync()
        t = min(env.t, env.ds.T - 1)
        feats = env.ds.obs[t].astype(np.float32)
        active = env.tradable()
        p = env.ds.price[t].astype(np.float64)
        held = env.qty > 0
        target = env.pos.copy()
        # exits
        u = np.where(held & np.isfinite(p) & (self.entry > 0), p / np.maximum(self.entry, 1e-18) - 1.0, 0.0)
        self.peak = np.where(held, np.maximum(self.peak, np.nan_to_num(p, nan=0.0)), 0.0)
        self.sat = np.where(held, self.sat + np.maximum(u, 0.0) * env.ds.beat_s, 0.0)
        exit_ = held & ((u <= -self.hard_stop) | (u <= -self.exit_loss) |
                        ((self.peak >= self.entry * (1 + self.turn_margin)) & (u <= 0)) |
                        ((self.sat >= self.satiety_target) & (u > config.FEE_ROUND_TRIP)))
        target[exit_] = 0.0
        # entries
        deployed = float(env.cost.sum()); deployable = max(config.CAPITAL_SOL - config.GAS_RESERVE_SOL, 1e-6)
        hunger = max(0.0, 1.0 - deployed / deployable)
        z_eff = self.z_buy - self.z_hunger * hunger
        z = rank_score(feats, active)
        cand = active & ~held & (z >= z_eff)
        self.above = np.where(cand, self.above + 1, 0)
        enter = cand & (self.above >= self.confirm)
        size = np.clip(0.1 + 0.9 * (z - z_eff) / 2.0, 0.1, 1.0)
        target[enter] = size[enter]
        self.above[enter] = 0
        return target

    def after_step(self) -> None:
        env = self.env
        t = min(env.t, env.ds.T - 1)
        p = env.ds.price[max(t - 1, 0)].astype(np.float64)
        newly = (env.qty > 0) & (self.entry == 0)
        self.entry = np.where(env.qty > 0, np.where(newly, np.nan_to_num(p, nan=0.0), self.entry), 0.0)
        self.peak = np.where(newly, self.entry, self.peak)


def run_expert(ds, t_start: int, t_end: int, collect: bool = True) -> dict:
    env = TradingEnv(ds, t_start, t_end)
    obs = env.reset()
    ex = ExpertPolicy(env)
    O, A, total = [], [], np.zeros(env.M)
    while True:
        a = ex.act()
        if collect:
            O.append(obs.astype(np.float32)); A.append(a.astype(np.float32))
        obs, r, done, info = env.step(a)
        ex.after_step()
        total += r
        if done:
            break
    out = {"net_sol": float(total.sum()) / 100.0 * config.MAX_POSITION_SOL, "trades": info["trades"], "fees": info["fees"], "turnover": info["turnover"],
           "beats": t_end - t_start}
    if collect:
        out["obs"] = np.stack(O); out["act"] = np.stack(A)
    return out
