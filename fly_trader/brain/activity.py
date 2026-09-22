"""The fly's network activity per trade minute, kept as files for the console's 3D view (``ui/brain3d``).

``agent/fly_session.py`` writes one file per scored minute: the mean activity over the rows it scored, one int8 per
neuron (value / 127 = the rate after the last propagation step, in −1..1), named by the minute's UTC stamp so a sorted
listing is chronological. Written to a temporary name and renamed, so a reader never sees a partial file; pruned to
``KEEP_S`` by the session's hourly housekeeping. The console only lists and reads.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

KEEP_S = 86400.0             # a day of minutes (the console's slider)
STAMP = "%Y%m%dT%H%M"
SUFFIX = ".npz"
TMP_MAX_AGE_S = 3600.0


def quantise(h) -> np.ndarray:
    """int8 per neuron from rates in −1..1 (torch or numpy)."""
    if hasattr(h, "detach"):
        h = h.detach().float().cpu().numpy()
    return np.clip(np.rint(np.asarray(h, dtype=np.float64) * 127.0), -127, 127).astype(np.int8)


def stamp(t: float) -> str:
    return datetime.fromtimestamp(t, timezone.utc).strftime(STAMP)


def parse(s: str) -> float:
    return datetime.strptime(s, STAMP).replace(tzinfo=timezone.utc).timestamp()


def path_of(d: Path, s: str) -> Path:
    return Path(d) / f"{s}{SUFFIX}"


def write(d: Path, t: float, h8: np.ndarray, *, n_rows: int, n_cand: int, connectome: str = "") -> Path:
    d = Path(d); d.mkdir(parents=True, exist_ok=True); s = stamp(t)
    tmp, final = d / f".{s}.tmp{SUFFIX}", path_of(d, s)
    with open(tmp, "wb") as f:
        np.savez(f, h=np.asarray(h8, dtype=np.int8), t=np.float64(t), n_rows=np.int64(n_rows), n_cand=np.int64(n_cand), connectome=np.array(str(connectome)))
    os.replace(tmp, final)
    return final


def listing(d: Path) -> list[str]:
    """Stamps of the minutes on disk, oldest first."""
    d = Path(d)
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob(f"*{SUFFIX}") if not p.name.startswith("."))


def newest(d: Path) -> str | None:
    lst = listing(d)
    return lst[-1] if lst else None


def load(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as z:
        return {"h": z["h"].astype(np.int8), "t": float(z["t"]), "n_rows": int(z["n_rows"]), "n_cand": int(z["n_cand"]), "connectome": str(z["connectome"])}


def prune(d: Path, before: float) -> int:
    """Remove minutes older than ``before`` (epoch s) and temporary files left by an interrupted write."""
    d = Path(d); n = 0
    if not d.exists():
        return 0
    now = time.time()
    for p in d.glob(f"*{SUFFIX}"):
        if p.name.startswith("."):
            if now - p.stat().st_mtime > TMP_MAX_AGE_S:
                p.unlink(missing_ok=True)
            continue
        try:
            t = parse(p.stem)
        except ValueError:
            continue
        if t < before:
            p.unlink(missing_ok=True); n += 1
    return n
