"""Where each of the fly's neurons sits in the brain, for the console's 3D view (``ui/brain3d``).

The connectome artifact carries no positions. FlyWire's annotations TSV (``populations.ANNOTATIONS_TSV``) has a soma
centroid for most neurons and a point on the neuron for nearly all, in FAFB voxels (4 nm × 4 nm × 40 nm). This joins
them to the graph index by ``root_ids`` and restricts to the sub-graph the fly runs on (``SubConnectome``, everything
but VISUAL) with the same re-indexing, so index i here is neuron i of the fly's activity vector. Output: ``geometry.bin``
(float32 xyz per neuron, micrometres, centred) and ``geometry.json`` (populations, cell types, coverage), written once
per connectome and served as static files.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .connectome import current_connectome_path
from .populations import ANNOTATIONS_TSV

VERSION = 1
VOXEL_UM = np.array([4.0, 4.0, 40.0]) / 1000.0     # FAFB voxel → micrometre
EXCLUDE = ("VISUAL",)
POS_COLS = ["pos_x", "pos_y", "pos_z"]
SOMA_COLS = ["soma_x", "soma_y", "soma_z"]
BIN, META = "geometry.bin", "geometry.json"


def keep_mask(pop_ranges: dict, n: int, exclude: tuple[str, ...] = EXCLUDE) -> np.ndarray:
    """Which full-graph neurons the sub-graph keeps (``SubConnectome.__init__``)."""
    keep = np.ones(n, dtype=bool)
    for name in exclude:
        if name in pop_ranges:
            lo, hi = pop_ranges[name]; keep[lo:hi] = False
    return keep


def sub_ranges(pop_ranges: dict, keep: np.ndarray, exclude: tuple[str, ...] = EXCLUDE) -> dict[str, tuple[int, int]]:
    """The kept populations' ranges in sub-graph indices (``cumsum(keep) - 1``, as ``SubConnectome`` re-indexes)."""
    new_index = np.cumsum(keep) - 1
    out = {}
    for name, (lo, hi) in pop_ranges.items():
        if name in exclude or hi <= lo:
            continue
        out[name] = (int(new_index[lo]), int(new_index[hi - 1]) + 1)
    return out


def _atomic(path: Path, data: bytes) -> None:
    tmp = path.with_name("." + path.name + ".tmp")
    tmp.write_bytes(data); os.replace(tmp, path)


def read_meta(out_dir: Path) -> dict | None:
    try:
        return json.loads((Path(out_dir) / META).read_text())
    except (FileNotFoundError, ValueError):
        return None


def build_geometry(out_dir: Path, connectome_path: str | Path | None = None, annotations: str | Path | None = None,
                   exclude: tuple[str, ...] = EXCLUDE) -> dict:
    """Write (once per connectome) and return the geometry metadata. Soma centroid where known, else the neuron's
    annotation point, else its population's centroid (a handful of neurons); scaled to micrometres and centred."""
    out_dir = Path(out_dir); jpath, bpath = out_dir / META, out_dir / BIN
    path = Path(connectome_path) if connectome_path else current_connectome_path()
    with np.load(path, allow_pickle=False) as z:
        sha = str(z["content_sha256"]); n = int(z["N"]); pop_ranges = {k: tuple(v) for k, v in json.loads(str(z["pop_ranges"])).items()}
        root_ids = np.asarray(z["root_ids"]); cell_type = np.asarray(z["cell_type"]); pop_order = [str(p) for p in z["pop_order"]]
    meta = read_meta(out_dir)
    if meta and meta.get("connectome_sha256") == sha and meta.get("version") == VERSION and bpath.exists():
        return meta
    keep = keep_mask(pop_ranges, n, exclude); sub = sub_ranges(pop_ranges, keep, exclude); order = [p for p in pop_order if p in sub]
    rid = root_ids[keep].astype(np.int64); ct = cell_type[keep].astype(str)
    ann_path = Path(annotations) if annotations else ANNOTATIONS_TSV
    if not ann_path.exists():
        raise FileNotFoundError(f"{ann_path} missing: the FlyWire annotations give the neurons their positions")
    ann = pd.read_csv(ann_path, sep="\t", usecols=["root_id", *POS_COLS, *SOMA_COLS], dtype={"root_id": str}, low_memory=False)
    ann["root_id"] = ann["root_id"].astype(np.int64)
    ann = ann.drop_duplicates("root_id").set_index("root_id").reindex(rid)
    soma = ann[SOMA_COLS].to_numpy(dtype=np.float64); pos = ann[POS_COLS].to_numpy(dtype=np.float64)
    has_soma = np.isfinite(soma).all(1); has_pos = np.isfinite(pos).all(1); located = has_soma | has_pos
    xyz = np.where(has_soma[:, None], soma, pos)
    if not located.all():                                   # place the few unlocated neurons at their population's centroid
        glob = xyz[located].mean(0) if located.any() else np.zeros(3)
        for lo, hi in sub.values():
            m = located[lo:hi]
            xyz[lo:hi][~m] = xyz[lo:hi][m].mean(0) if m.any() else glob
    xyz = xyz * VOXEL_UM; center = xyz.mean(0); xyz = (xyz - center).astype(np.float32)
    names, index = np.unique(ct, return_inverse=True)
    meta = {"version": VERSION, "n": int(len(rid)), "units": "um", "connectome_sha256": sha, "connectome_file": path.name, "excluded": list(exclude),
            "pop_order": order, "pop_ranges": {k: [lo, hi] for k, (lo, hi) in sub.items()},
            "cell_type": {"names": names.tolist(), "index": index.astype(int).tolist()},
            "coverage": {"soma": int(has_soma.sum()), "pos": int((has_pos & ~has_soma).sum()), "centroid": int((~located).sum())},
            "center_um": [float(v) for v in center], "extent_um": [float(v) for v in (xyz.max(0) - xyz.min(0))]}
    out_dir.mkdir(parents=True, exist_ok=True)
    _atomic(bpath, np.ascontiguousarray(xyz).tobytes()); _atomic(jpath, json.dumps(meta).encode())
    return meta
