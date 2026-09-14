"""Valence → ENTRY decisions (exits are reflexes in rails.py: rejection, turned, satiety).

Thresholds are expressed in units of the valence spread across the active slots this beat
(z = (m̂ − μ)/σ), so they track the readout's actual dynamic range instead of assuming one.
Hunger is the fraction of capital NOT deployed (satiety = fullness), lowering the entry threshold
when the book is empty. Sizing grows with z above the threshold up to MAX_POSITION_SOL. Softmax
selection under the capital constraint follows Bennett et al. 2021 (p_i ∝ exp(β·m̂_i)). The z
constants are unsourced defaults in config.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .. import config


@dataclass
class Decision:
    slot: int
    mint: str
    kind: str            # enter | exit
    m_hat: float
    size_sol: float = 0.0
    softmax_p: float | None = None
    reason: str = ""
    z: float = 0.0


@dataclass
class DecisionState:
    above: dict[str, int] = field(default_factory=dict)   # consecutive beats z ≥ z_buy_eff
    below: dict[str, int] = field(default_factory=dict)   # consecutive beats z ≤ z_sell
    beats_since_entry: int = 10 ** 9
    exposure_frac: float = 0.0                            # deployed / deployable capital
    last_mu: float = 0.0
    last_sigma: float = 1.0
    last_z: dict[str, float] = field(default_factory=dict)

    def hunger(self) -> float:
        return float(min(1.0, max(0.0, 1.0 - self.exposure_frac)))

    def z_buy_eff(self) -> float:
        return config.Z_BUY - config.Z_HUNGER * self.hunger()

    def theta_buy_eff(self) -> float:   # absolute equivalent, for logging
        return self.last_mu + self.z_buy_eff() * self.last_sigma


def size_for(z: float, z_eff: float) -> float:
    frac = min(1.0, max(0.0, (z - z_eff) / max(config.Z_SIZE_RANGE, 1e-6)))
    return float(config.SIZE_MIN_SOL + (config.MAX_POSITION_SOL - config.SIZE_MIN_SOL) * frac)


def zscores(m_hat: np.ndarray, active: np.ndarray) -> tuple[np.ndarray, float, float]:
    vals = m_hat[active]
    if vals.size < 3:
        return np.zeros_like(m_hat), 0.0, 1.0
    mu = float(vals.mean())
    sigma = float(max(vals.std(), config.Z_SIGMA_FLOOR))
    return (m_hat - mu) / sigma, mu, sigma


def decide(state: DecisionState, slots: list[str | None], m_hat: np.ndarray, held: set[str],
           free_capital_sol: float, rng: np.random.Generator, exposure_frac: float | None = None,
           dwell: list[int] | None = None, allow_entries: bool = True, eligible: np.ndarray | None = None) -> list[Decision]:
    if exposure_frac is not None:
        state.exposure_frac = exposure_frac
    active = np.array([m is not None for m in slots])
    z, mu, sigma = zscores(np.asarray(m_hat, dtype=np.float64), active)
    state.last_mu, state.last_sigma = mu, sigma
    z_eff = state.z_buy_eff()
    entries: list[Decision] = []
    exits: list[Decision] = []
    seen = set()
    for i, mint in enumerate(slots):
        if not mint:
            continue
        seen.add(mint)
        m, zi = float(m_hat[i]), float(z[i])
        state.last_z[mint] = zi
        if mint in held:
            state.above.pop(mint, None)   # exits are reflexes (rails.forced_exit_kind), never valence
        else:
            state.below.pop(mint, None)
            if not allow_entries or (dwell is not None and dwell[i] < config.SNIFF_BEATS) or (eligible is not None and not eligible[i]):
                state.above[mint] = 0      # still sniffing, or innately aversive: not food
                continue
            if zi >= z_eff:
                state.above[mint] = state.above.get(mint, 0) + 1
            else:
                state.above[mint] = 0
            if state.above[mint] >= config.CONFIRM_BEATS:
                entries.append(Decision(i, mint, "enter", m, size_for(zi, z_eff),
                                        reason=f"z {zi:.2f} >= z_buy {z_eff:.2f} for {state.above[mint]} beats (hunger {state.hunger():.2f})", z=zi))
                state.above[mint] = 0   # re-confirm before the next attempt (a refused fill must not retry every beat)
    for mint in list(state.last_z):
        if mint not in seen:
            del state.last_z[mint]
    for mint in list(state.above):
        if mint not in seen:
            del state.above[mint]
    for mint in list(state.below):
        if mint not in seen:
            del state.below[mint]
    chosen: list[Decision] = []
    budget = max(0.0, free_capital_sol)
    if entries:
        ms = np.array([d.m_hat for d in entries])
        beta = config.SOFTMAX_BETA / max(sigma, 1e-6)   # temperature in valence-spread units
        p = np.exp(beta * (ms - ms.max()))
        p = p / p.sum()
        order = rng.choice(len(entries), size=len(entries), replace=False, p=p)
        for j in order:
            d = entries[j]
            d.softmax_p = float(p[j])
            if d.size_sol <= budget:
                chosen.append(d)
                budget -= d.size_sol
            elif budget >= config.SIZE_MIN_SOL:
                d.size_sol = float(budget)
                chosen.append(d)
                budget = 0.0
    return exits + chosen
