"""Robust feature scaling, shared verbatim by the selector and the fly, so both models receive the same signals.

Minute features are fat-tailed (a pool's reserve at graduation or a creator's first buy sits thousands of standard
deviations out); mean/std standardisation leaves such values at hundreds or thousands, which swamps a neural network's
units while leaving a tree model unaffected. Each feature is centred on its median, divided by a robust spread (the
inter-quartile range / 1.349, the standard deviation of a normal; the plain standard deviation when the IQR is zero,
1 for a constant feature) and compressed with asinh: linear near the centre, logarithmic in the tails. The transform
is monotone per feature, so tree splits are unchanged; it is fit on the training rows only.
"""
from __future__ import annotations

import numpy as np

FIT_SAMPLE = 2_000_000


class RobustScaler:
    def __init__(self, center: np.ndarray, scale: np.ndarray):
        self.center = np.asarray(center, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)

    @classmethod
    def fit(cls, X: np.ndarray, sample: int = FIT_SAMPLE, seed: int = 0) -> "RobustScaler":
        idx = np.arange(len(X)) if len(X) <= sample else np.random.default_rng(seed).choice(len(X), sample, replace=False)
        Xs = np.asarray(X[idx], dtype=np.float64)
        med = np.median(Xs, axis=0); q75, q25 = np.percentile(Xs, [75, 25], axis=0); iqr = (q75 - q25) / 1.349; sd = Xs.std(axis=0)
        return cls(med, np.where(iqr > 1e-12, iqr, np.where(sd > 1e-12, sd, 1.0)))

    def transform(self, X: np.ndarray) -> np.ndarray:
        """A scaled copy of ``X``, never touching it. One allocation: the arithmetic then runs in place, so a walk-forward
        block of tens of millions of rows costs one array instead of four (subtract, divide, asinh and cast each used to
        allocate their own)."""
        return self.transform_into(np.array(X, dtype=np.float32, order="C"))

    def transform_into(self, Z: np.ndarray) -> np.ndarray:
        """Scale ``Z`` in place and return it. Only for an array the caller owns — a block copy from fancy indexing, never
        a view of the corpus, which this would corrupt."""
        Z -= self.center; Z /= self.scale; np.arcsinh(Z, out=Z)
        return Z

    def state(self) -> dict:
        return {"center": self.center.tolist(), "scale": self.scale.tolist()}

    @classmethod
    def from_state(cls, s: dict) -> "RobustScaler":
        return cls(np.asarray(s["center"]), np.asarray(s["scale"]))
