"""Strategy study, step 1: the selector's decision set (train/decisions.build over the whole corpus, labels for every hold
from 10 to 480 minutes) saved to logs/strategy_study -- X as a raw .npy that step 2 reads read-only from disk, the rest
in one .npz -- then this process exits, returning everything the build allocated (on 2026-10-19 the build peaked at
~100 GB; kept inside the searching process, its garbage squeezed the live console).

    nice -n 20 .venv/bin/python tools/strategy_study/build_ds.py
    nice -n 20 env OMP_NUM_THREADS=6 .venv/bin/python tools/strategy_study/study.py [--holds 240] [--compare <selector snapshot>]
    .venv/bin/python tools/strategy_study/variants.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent; REPO = HERE.parents[1]
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(HERE))
OUT = REPO / "logs" / "strategy_study"; OUT.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", handlers=[logging.FileHandler(OUT / "study.log"), logging.StreamHandler(sys.stdout)])

import guard  # noqa: E402
guard.MAX_FOOTPRINT_GB = 115.0                     # the build's own transient peak; the machine's free memory and the console still stop it
guard.start(str(OUT / "guard_build.log"))

from fly_trader.train import decisions, strategies  # noqa: E402

EXTRA_HOLDS = (360, 480)
log = logging.getLogger("study")
t0 = time.time()
ds = decisions.build(days=None, holds=tuple(strategies.HOLDS_MIN) + EXTRA_HOLDS)
log.info("built %d rows, %d days (%s..%s) in %.0f s", len(ds.y), len(ds.days), ds.days[0], ds.days[-1], time.time() - t0)
np.save(OUT / "ds_X.npy", ds.X)
mcodes, mu = pd.factorize(ds.mint)
extra = {"y": ds.y, "fwd": ds.fwd, "fwd_pess": ds.fwd_pess, "ts": ds.ts, "day_ord": decisions.day_ords(ds), "mint_u": np.asarray(mu, dtype=str),
         "mint_c": mcodes.astype(np.int32), "cols": np.array(ds.cols), "horizon_s": np.array(ds.horizon_s), "holds": np.array(sorted(ds.fwd_h))}
for h, v in ds.fwd_h.items():
    extra[f"fwd_h_{h}"] = v
if ds.chain is not None:
    cc, cu = pd.factorize(ds.chain)
    extra["chain_u"], extra["chain_c"] = np.asarray(cu, dtype=str), cc.astype(np.int8)
np.savez(OUT / "ds_rest.npz", **extra)
log.info("decision set saved to %s in %.0f s", OUT, time.time() - t0)
