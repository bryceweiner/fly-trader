"""Wealth marks per book, per-slot reward attribution, and the reward-prediction error.

wealth_t = SOL_free + Σ q_i·p_i·(1 − c_i) with p_i the last tape price and c_i the exit-cost fraction.
r_t = log W_t − log W_{t−1}. Per slot with a position: r^i = log(1 + ΔV_i / W_{t−1}) + 0.1·r_t where
ΔV_i includes fills and fees in the beat. r̃ = r / R0 (EWMA std). RPE δ = (1−γ)·r̃ + γ·m̂_t − m̂_{t−1}
(TD(0) form of Bennett et al. 2021's d̂ = r̂ − m̂; γ = 0 reproduces it). δ clipped to ±DELTA_CLIP.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from .. import config
from ..market.exit_cost import exit_cost_fraction


@dataclass
class WealthMark:
    sol_free: float
    positions_value: float
    exit_cost: float
    wealth: float
    exposure: float
    n_open: int


def mark_book(sol_free: float, positions: list[dict], last_price: dict[str, float | None],
              res_quote: dict[str, float | None], age_hours: dict[str, float | None],
              program_label: dict[str, str | None]) -> tuple[WealthMark, dict[str, float]]:
    """Returns the mark and per-mint marked values (after exit cost)."""
    value = 0.0
    cost_total = 0.0
    exposure = 0.0
    per_mint: dict[str, float] = {}
    for p in positions:
        mint = p["mint"]
        qty = float(p["qty"]) / (10 ** int(p.get("decimals") or 6))
        price = last_price.get(mint) or p.get("last_mark_price") or p.get("entry_price") or 0.0
        gross = qty * float(price)
        c = exit_cost_fraction(gross, res_quote.get(mint), age_hours.get(mint), program_label.get(mint))
        net = gross * (1.0 - c)
        value += net
        cost_total += gross * c
        exposure += float(p.get("cost_sol") or 0.0)
        per_mint[mint] = net
    wealth = sol_free + value
    return WealthMark(sol_free, value, cost_total, wealth, exposure, len(positions)), per_mint


@dataclass
class RewardTracker:
    prev_wealth: dict[str, float] = field(default_factory=dict)          # book -> W_{t-1}
    prev_values: dict[str, dict[str, float]] = field(default_factory=dict)  # book -> mint -> V_{t-1}
    prev_m_hat: dict[str, float] = field(default_factory=dict)            # mint -> m̂_{t-1}
    r0: float = 0.01                                                      # EWMA std of per-position returns (floor 1e-4)
    r0_alpha: float = 0.01

    def global_reward(self, book: str, wealth: float) -> float:
        w0 = self.prev_wealth.get(book)
        r = math.log(wealth / w0) if (w0 and w0 > 0 and wealth > 0) else 0.0
        return r

    def slot_rewards(self, book: str, wealth_now: float, values_now: dict[str, float],
                     cash_flows: dict[str, float], r_global: float) -> dict[str, float]:
        """Per-position log return over the beat (the fly feels the token's move, not its share of the book):
        r_i = log((V_t + proceeds) / (V_{t-1} + cost)) with cash_flows: mint -> SOL received (+) or spent (−).
        A fresh buy therefore costs exactly its fees + impact, a sell realises the marked value, holding
        earns the price change net of exit-cost changes. Plus a 0.1 share of the book's log return."""
        prev = self.prev_values.get(book, {})
        out: dict[str, float] = {}
        mints = set(values_now) | set(prev) | set(cash_flows)
        for m in mints:
            cf = cash_flows.get(m, 0.0)
            num = values_now.get(m, 0.0) + max(cf, 0.0)
            den = prev.get(m, 0.0) + max(-cf, 0.0)
            if den > 1e-12 and num > 1e-12:
                out[m] = math.log(num / den) + 0.1 * r_global
            elif den > 1e-12:
                out[m] = math.log(1e-12 / den) + 0.1 * r_global
        return out

    def commit(self, book: str, wealth: float, values: dict[str, float]) -> None:
        self.prev_wealth[book] = wealth
        self.prev_values[book] = dict(values)

    def normalize(self, r: float, update: bool = True) -> float:
        if update:
            self.r0 = math.sqrt((1 - self.r0_alpha) * self.r0 ** 2 + self.r0_alpha * r * r)
        return r / max(self.r0, 1e-4)

    def rpe(self, mint: str, r_tilde: float, m_now: float) -> tuple[float, float]:
        m_prev = self.prev_m_hat.get(mint, 0.0)
        # (1-γ) scaling keeps the discounted return of a unit-variance reward stream inside m̂'s range
        # (-1, 1), so δ can reach zero at convergence as in Bennett's d̂ = r̂ − m̂ (γ = 0 reproduces it)
        delta = (1.0 - config.GAMMA) * r_tilde + config.GAMMA * m_now - m_prev
        delta = max(-config.DELTA_CLIP, min(config.DELTA_CLIP, delta))
        return delta, m_prev

    def remember_m_hat(self, mint: str, m_now: float) -> None:
        self.prev_m_hat[mint] = m_now
