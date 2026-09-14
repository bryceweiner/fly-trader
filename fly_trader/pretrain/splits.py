"""Walk-forward split with purge/embargo (VOC dexlp/ingest/splits.py semantics): yields timestamp
ranges only, so leakage is impossible by construction."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Fold:
    train: tuple[float, float]
    val: tuple[float, float]
    test: tuple[float, float]

    def as_dict(self) -> dict:
        return {"train": list(self.train), "val": list(self.val), "test": list(self.test)}


def walk_forward(t0: float, t1: float, train_d: float, val_d: float, test_d: float, embargo_h: float = 6.0,
                 step_d: float | None = None) -> list[Fold]:
    day = 86400.0
    emb = embargo_h * 3600.0
    step = (step_d or test_d) * day
    folds = []
    start = t0
    while True:
        tr = (start, start + train_d * day)
        va = (tr[1] + emb, tr[1] + emb + val_d * day)
        te = (va[1] + emb, va[1] + emb + test_d * day)
        if te[1] > t1 + 1e-6:
            break
        folds.append(Fold(tr, va, te))
        start += step
    return folds
