"""Innate valence: how a market smells before any learning.

Flies have innate odor preferences carried by the lateral-horn pathway alongside the learned mushroom-body
valence (Aso et al. 2014 eLife 3:e04580; Jefferis et al. 2007 Cell). Here the innate term is a fixed
rank score over the features that carried forward-return edge in the recorded data (2026-09-13 backtest:
top-vs-bottom decile spread +0.75 % over 5 minutes): buy/sell imbalance over 1, 5 and 15 minutes,
nearness to the 1-hour high (dd_1h), and a LOW organic score (bot-inflated volume is bad). Tokens with no
positive signal rank low and therefore smell bad. The learned KC→MBON valence is added on top; the RPE
uses the learned valence only (Bennett 2021: the MB predicts, the LH does not).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandas as pd

from .. import config
from ..market.features import FIDX

INNATE_FEATURES: list[tuple[str, float]] = [
    ("ret_5m", 2.0), ("ret_1m", 1.0), ("ret_15m", 1.0),          # trend: a rising token smells much better than a falling one
    ("imb_15m", 1.0), ("imb_5m", 1.0), ("imb_1m", 1.0),         # buy pressure
    ("dd_1h", 1.0),                                              # near its 1-hour high
    ("organic_score", -1.0),                                     # bot-inflated volume smells bad
]


def innate_score(feats: np.ndarray, active: np.ndarray) -> np.ndarray:
    """Within-beat rank score of the active slots, standardized to mean 0 / std 1 (inactive → 0)."""
    n = feats.shape[0]
    out = np.zeros(n, dtype=np.float32)
    idx = np.where(active)[0]
    if len(idx) < 3:
        return out
    score = np.zeros(len(idx), dtype=np.float64)
    for name, w in INNATE_FEATURES:
        x = feats[idx, FIDX[name]].astype(np.float64)
        ranks = pd.Series(x).rank(method="average", pct=True).to_numpy()   # ties share a rank (no slot-index bias)
        score += w * ranks
    sd = float(score.std())
    out[idx] = (score - score.mean()) / max(sd, 1e-6)
    return out


def eligible(feats: np.ndarray, masks: np.ndarray, active: np.ndarray, age_h: list) -> np.ndarray:
    """Hard innate aversions: a token may only be entered if it is old enough, rising, under buy pressure and
    near its 1-hour high. Everything else smells bad regardless of the learned valence."""
    n = feats.shape[0]
    ok = active.copy()
    for k in range(n):
        if not ok[k]:
            continue
        f = feats[k]
        a = age_h[k]
        if a is None or a < config.ELIG_MIN_AGE_H:
            ok[k] = False; continue
        if config.ELIG_REQUIRE_RISING and not (f[FIDX["ret_1m"]] > 0 and f[FIDX["ret_5m"]] > 0):
            ok[k] = False; continue
        if not (f[FIDX["imb_5m"]] > config.ELIG_MIN_IMB and f[FIDX["imb_15m"]] > config.ELIG_MIN_IMB):
            ok[k] = False; continue
        if not (f[FIDX["dd_1h"]] > config.ELIG_MAX_DD_1H):
            ok[k] = False; continue
        # the 15-minute windows must actually have data (mask bits set)
        if not (int(masks[k]) & (1 << FIDX["ret_5m"])) or not (int(masks[k]) & (1 << FIDX["imb_15m"])):
            ok[k] = False
    return ok


class EdgeTracker:
    """Realized edge of the decision valence: Pearson correlation between the valence a slot had at time t
    and the token's return over the next EDGE_HORIZON_S seconds, over a rolling EDGE_WINDOW_S window across
    all active slots. ≤ EDGE_MIN means the smell is not working: live/mirror entries are gated."""

    def __init__(self):
        self.pending: list[tuple[float, str, float, float]] = []   # (t, mint, valence, price_then)
        self.samples: list[tuple[float, float, float]] = []       # (t, valence, forward return)

    def record(self, t: float, mints: list, valence: np.ndarray, price_of) -> None:
        for m, v in zip(mints, valence):
            if not m:
                continue
            p = price_of(m)
            if p:
                self.pending.append((t, m, float(v), float(p)))

    def update(self, now: float, price_at) -> None:
        keep = []
        for t, m, v, p0 in self.pending:
            if now - t >= config.EDGE_HORIZON_S:
                p1 = price_at(m, t + config.EDGE_HORIZON_S)
                if p1 and p0:
                    self.samples.append((t, v, p1 / p0 - 1.0))
            else:
                keep.append((t, m, v, p0))
        self.pending = keep
        cutoff = now - config.EDGE_WINDOW_S
        self.samples = [s for s in self.samples if s[0] >= cutoff]

    def edge(self) -> tuple[float | None, int]:
        n = len(self.samples)
        if n < config.EDGE_MIN_SAMPLES:
            return None, n
        v = np.array([s[1] for s in self.samples]); r = np.array([s[2] for s in self.samples])
        if v.std() < 1e-9 or r.std() < 1e-12:
            return 0.0, n
        return float(np.corrcoef(v, r)[0, 1]), n
