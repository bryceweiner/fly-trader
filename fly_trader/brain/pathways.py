"""The KC→MBON synapses the fly has changed most, for the console's 3D view (``ui/brain3d``): pure functions over the
files the fly session writes, so the console never builds a network.

``PlasticBank.state()["D"]`` holds the learned change of each of the connectome's KC→MBON pairs, in the row-major
order of ``np.nonzero(M_KM)`` (``FlyNet.km_kc``/``km_mbon``); positive = strengthened. The bootstrap checkpoint's
``theta_km`` gives the base weight of the same pairs (``softplus``), so ``ratio = delta / base`` ranks changes by how
much of the synapse they are. Each MBON column belongs to one strategy's dopamine channel (``learn``) or to none.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

TOP_K = 200


def km_pairs(connectome_path: str | Path) -> tuple[np.ndarray, np.ndarray, int, int]:
    """(kc row, mbon column) of every pair, and the approach / avoid column counts."""
    with np.load(connectome_path, allow_pickle=False) as z:
        kc, mbon = np.nonzero(np.asarray(z["M_KM"])); pr = json.loads(str(z["pop_ranges"]))
    return kc.astype(np.int64), mbon.astype(np.int64), int(pr["MBON_APP"][1] - pr["MBON_APP"][0]), int(pr["MBON_AV"][1] - pr["MBON_AV"][0])


def bootstrap_weights(boot_path: str | Path) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """(base weight per pair, learn [S, J], strategy names) from a fly bootstrap checkpoint (``fly_selector.save``)."""
    d = torch.load(boot_path, map_location="cpu", weights_only=False); sd = d["state_dict"]
    w0 = np.logaddexp(0.0, sd["theta_km"].detach().float().numpy()).astype(np.float32)      # softplus
    n_mbon = int(sd["c_sign"].numel()) if "c_sign" in sd else int(sd["mask"].shape[1])
    learn = sd["learn"].numpy().astype(bool) if "learn" in sd else np.ones((1, n_mbon), dtype=bool)
    strategies = list(d.get("rules") or {}) or ["ev"]
    return w0, learn, strategies


def bank_D(path: str | Path) -> tuple[np.ndarray, int | None]:
    """The learned change per pair from ``state.pt`` or a ``snap_*.pt`` (both carry ``bank`` and ``bootstrap_id``)."""
    s = torch.load(path, map_location="cpu", weights_only=False)
    D = s["bank"]["D"]; D = D[0] if D.ndim == 2 else D
    bid = s.get("bootstrap_id")
    return D.detach().float().numpy(), (int(bid) if bid is not None else None)


def _round(x: float) -> float:
    return float(f"{x:.4g}")


def top_pathways(D: np.ndarray, w0: np.ndarray, kc: np.ndarray, mbon: np.ndarray, learn: np.ndarray, strategies: list[str],
                 n_app: int, n_av: int, top_k: int = TOP_K, D_ref: np.ndarray | None = None, mbon_offset: int = 0) -> list[dict]:
    """The ``top_k`` pairs by |change / base weight|; ``D_ref`` (an older snapshot) makes it the change since then.
    ``mbon`` is reported as ``mbon_offset + column`` so a renderer can address the MBON as a graph index."""
    delta = np.asarray(D, dtype=np.float64) - (np.asarray(D_ref, dtype=np.float64) if D_ref is not None else 0.0)
    ratio = delta / np.maximum(np.asarray(w0, dtype=np.float64), 1e-6)
    nz = np.flatnonzero(delta != 0)
    if not len(nz):
        return []
    order = nz[np.argsort(-np.abs(ratio[nz]), kind="stable")[:top_k]]
    learn = np.asarray(learn, dtype=bool); owner = np.full(learn.shape[1], "shared", dtype=object); single = learn.sum(0) == 1
    for s, name in enumerate(strategies[:learn.shape[0]]):
        owner[learn[s] & single] = name
    def valence(j: int) -> str:
        return "approach" if j < n_app else "avoid" if j < n_app + n_av else "other"
    return [{"kc": int(kc[p]), "mbon": int(mbon_offset + mbon[p]), "delta": _round(delta[p]), "ratio": _round(ratio[p]),
             "strategy": str(owner[mbon[p]]), "valence": valence(int(mbon[p]))} for p in order]


def _ts(v) -> float:
    return v.timestamp() if isinstance(v, datetime) else float(v)


def _bootstrap_of(row: dict) -> int | None:
    note = row.get("note")
    try:
        d = note if isinstance(note, dict) else json.loads(note or "{}")
        b = d.get("bootstrap_id")
        return int(b) if b is not None else None
    except (ValueError, TypeError):
        return None


def reference_snapshot(rows: list[dict], now: float, bootstrap_id: int | None, keep_s: float = 86400.0) -> str | None:
    """The snapshot to diff against for "the last day": the newest at least ``keep_s`` old, of this bootstrap, whose file
    exists; else the oldest such snapshot; else None. ``rows`` (``ts``, ``path``, ``note``) newest first."""
    cand = [r for r in rows if _bootstrap_of(r) == bootstrap_id and Path(r["path"]).exists()]
    old = [r for r in cand if _ts(r["ts"]) <= now - keep_s]
    if old:
        return str(old[0]["path"])
    return str(cand[-1]["path"]) if cand else None
