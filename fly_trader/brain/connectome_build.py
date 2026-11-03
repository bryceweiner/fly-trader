"""Build the FAFB v783 connectome artifact (``fly-trader build-connectome``).

Inputs (data/brain/raw/): flybrain ``connections.csv.gz`` (pre_root_id, post_root_id, neuropil,
syn_count, nt_type; one row per (pre, post, neuropil)), ``neurons.csv.gz``, ``classification.csv.gz``
and the Schlegel et al. 2024 annotation TSV. Output: ``data/brain/connectome/fafb783_<sha12>.npz``
(COO ``indices`` int64[2, nnz] with row = post, col = pre, sorted; ``values`` float32 pre-calibration
weights; ``values_raw`` = signed synapse counts; the dense plastic KC->MBON block ``W_KM0``) plus
``current.txt`` naming the artifact, the ``populations`` table and a ``runs`` row.

Sign convention (config.NT_SIGN_MODE): ``shiu`` = GABA and glutamate inhibitory, everything else
excitatory (Shiu et al. 2024 Nature 634:210); ``flybrain`` = only GABA inhibitory (flybrain's
NT_SIGN). Weights are summed over neuropils per (pre, post) pair with the edge's own nt_type
(nt_type varies within a pre neuron in FlyWire's connection table); zero sums are dropped, as
flybrain does, which reproduces its 2,698,236 edges in ``flybrain`` mode.

Scale: s0 = 0.99 / rho(|W_raw|) (spectral radius by shifted power iteration) is only the INITIAL
guess, motivated by Costi et al. 2025 (PMC12109256), who ran the FlyWire connectome as an echo-state
reservoir rescaled to rho = 0.99 (tanh units, glutamate treated as excitatory there). The fly
(``train/fly_selector.py``) initialises its synaptic magnitudes from s (``scale.json`` when present, else s0).
KC->MBON magnitudes are forced positive because KC output synapses are cholinergic (Barnstedt et
al. 2016 Neuron), regardless of FlyWire's per-cell NT prediction.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import warnings

import numpy as np
import pyarrow as pa
import pyarrow.csv as pc
import torch

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

from .. import config
from . import populations as P

CONNECTIONS_CSV = P.RAW_DIR / "connections.csv.gz"
CONNECTOME_DIR = config.BRAIN_DIR / "connectome"
CURRENT_FILE = CONNECTOME_DIR / "current.txt"
EXPECTED_EDGES = 2_698_236
W_CLIP = 1.0                 # = theta (plan section 1.7)
TARGET_RADIUS = 0.99         # reservoir operating point (Vienna 2025)
POWER_ITER_MAX = 5000
POWER_ITER_TOL = 1e-9
SOURCE_FILES = ["connections.csv.gz", "neurons.csv.gz", "classification.csv.gz",
                "Supplemental_file1_neuron_annotations.tsv", "consolidated_cell_types.csv.gz", "neuron_meta.json"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def nt_sign(nt: np.ndarray, mode: str) -> np.ndarray:
    """+1/-1 per edge from the edge's nt_type (flybrain vocabulary ACH/GABA/GLUT/DA/SER/OCT)."""
    nt = np.char.upper(nt.astype(str))
    if mode == "shiu":
        neg = (nt == "GABA") | (nt == "GLUT")
    elif mode == "flybrain":
        neg = nt == "GABA"
    else:
        raise ValueError(f"unknown NT_SIGN_MODE {mode!r} (shiu | flybrain)")
    return np.where(neg, -1.0, 1.0)


def read_connections(path: Path = CONNECTIONS_CSV) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    tbl = pc.read_csv(str(path), convert_options=pc.ConvertOptions(column_types={
        "pre_root_id": pa.int64(), "post_root_id": pa.int64(), "neuropil": pa.string(),
        "syn_count": pa.int32(), "nt_type": pa.string()}))
    pre = tbl["pre_root_id"].to_numpy()
    post = tbl["post_root_id"].to_numpy()
    syn = tbl["syn_count"].to_numpy().astype(np.int64)
    nt = tbl["nt_type"].to_numpy(zero_copy_only=False).astype(str)
    return pre, post, syn, nt


def root_id_lookup(canonical_root_ids: np.ndarray):
    order = np.argsort(canonical_root_ids)
    srt = canonical_root_ids[order]

    def lookup(x: np.ndarray) -> np.ndarray:
        pos = np.searchsorted(srt, x)
        pos[pos >= len(srt)] = 0
        return np.where(srt[pos] == x, order[pos], -1)
    return lookup


def aggregate_edges(pre: np.ndarray, post: np.ndarray, syn: np.ndarray, nt: np.ndarray, N: int, mode: str) -> dict:
    """Sum syn_count * sign over neuropils per (pre, post). Returns arrays over distinct pairs."""
    key = pre.astype(np.int64) * N + post.astype(np.int64)
    uniq, inv = np.unique(key, return_inverse=True)
    n = len(uniq)
    sgn = nt_sign(nt, mode)
    signed = np.bincount(inv, weights=syn * sgn, minlength=n)
    signed_fb = np.bincount(inv, weights=syn * nt_sign(nt, "flybrain"), minlength=n)
    unsigned = np.bincount(inv, weights=syn, minlength=n)
    return {
        "pre": (uniq // N).astype(np.int64), "post": (uniq % N).astype(np.int64),
        "signed": signed, "unsigned": unsigned,
        "n_pairs": int(n), "nnz_mode": int((signed != 0).sum()), "nnz_flybrain": int((signed_fb != 0).sum()),
    }


def spectral_radius(row: np.ndarray, col: np.ndarray, values_abs: np.ndarray, N: int,
                    max_iter: int = POWER_ITER_MAX, tol: float = POWER_ITER_TOL) -> dict:
    """rho(|W|) by power iteration on the shifted matrix |W| + I (rho(|W| + I) = rho(|W|) + 1 for a
    non-negative matrix; the shift makes the iteration converge for periodic components). Also
    returns the Collatz-Wielandt bracket min_i (|W|x)_i / x_i <= rho <= max_i (|W|x)_i / x_i."""
    idx = torch.from_numpy(np.stack([row, col]).astype(np.int64))
    W = torch.sparse_coo_tensor(idx, torch.from_numpy(values_abs.astype(np.float64)), (N, N)).coalesce().to_sparse_csr()
    x = torch.full((N, 1), 1.0 / math.sqrt(N), dtype=torch.float64)
    lam_prev, it, converged = 0.0, 0, False
    for it in range(1, max_iter + 1):
        y = W @ x + x
        lam = float(torch.linalg.vector_norm(y))
        x = y / lam
        if abs(lam - lam_prev) <= tol * lam:
            converged = True
            break
        lam_prev = lam
    wx = (W @ x).squeeze(1)
    xs = x.squeeze(1)
    ratio = wx / xs
    return {"rho": lam - 1.0, "iterations": it, "converged": converged,
            "cw_lower": float(ratio.min()), "cw_upper": float(ratio.max())}


def content_hash(indices: np.ndarray, values_raw: np.ndarray, W_KM0: np.ndarray, root_ids: np.ndarray,
                 pop_ranges_json: str, nt_sign_mode: str) -> str:
    h = hashlib.sha256()
    for part in (indices.tobytes(), values_raw.tobytes(), W_KM0.tobytes(), root_ids.tobytes(),
                 pop_ranges_json.encode(), nt_sign_mode.encode()):
        h.update(part)
    return h.hexdigest()


def _git_sha() -> str | None:
    try:
        import subprocess
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=config.REPO_ROOT, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def persist(pops: P.Populations, metrics: dict, build_cfg: dict, file_sha: str, started_at: datetime,
            annotations_label: str) -> str:
    """populations rows (id = order index) + runs row; returns the run_id."""
    from psycopg.types.json import Jsonb
    from ..db.connection import transaction
    run_id = str(uuid.uuid4())
    with transaction() as conn:
        with conn.cursor() as cur:
            for i, name in enumerate(pops.order):
                cur.execute("DELETE FROM populations WHERE name = %s AND id <> %s", (name, i))
                cur.execute(
                    "INSERT INTO populations (id, name, n, kind, source) VALUES (%s, %s, %s, %s, %s) "
                    "ON CONFLICT (id) DO UPDATE SET name = EXCLUDED.name, n = EXCLUDED.n, kind = EXCLUDED.kind, "
                    "source = EXCLUDED.source",
                    (i, name, pops.n(name), pops.kind.get(name, "other"), annotations_label))
            cur.execute(
                "INSERT INTO runs (run_id, kind, started_at, ended_at, git_sha, config, connectome_sha256, status, metrics) "
                "VALUES (%s, 'connectome_build', %s, now(), %s, %s, %s, 'done', %s)",
                (run_id, started_at, _git_sha(), Jsonb(build_cfg), file_sha, Jsonb(metrics)))
            if pops.source != "annotations":
                cur.execute(
                    "INSERT INTO events (level, source, message, detail) VALUES ('warning', 'connectome_build', %s, %s)",
                    ("cell types from flybrain classification.csv fallback (annotation TSV unavailable); "
                     "MBON valence by neurotransmitter, DAN valence by synapse counts",
                     Jsonb({"run_id": run_id})))
    return run_id


def main(annotations: str | None = None, nt_sign_mode: str | None = None, source: str | None = None,
         out_dir: str | Path | None = None, persist_db: bool = True) -> Path:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    mode = nt_sign_mode or config.NT_SIGN_MODE
    out_dir = Path(out_dir) if out_dir else CONNECTOME_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- source hashes ----
    source_sha = {}
    for name in SOURCE_FILES:
        p = P.RAW_DIR / name
        if p.exists():
            source_sha[name] = sha256_file(p)
    if annotations:
        source_sha[Path(annotations).name] = sha256_file(Path(annotations))

    # ---- connections (canonical flybrain indices) ----
    pre_rid, post_rid, syn, nt = read_connections()
    n_rows = len(pre_rid)
    neu_ids = np.asarray(__import__("pandas").read_csv(P.NEURONS_CSV, usecols=["root_id"], dtype={"root_id": np.int64})["root_id"])
    N = len(neu_ids)
    lookup = root_id_lookup(neu_ids)
    pre_c, post_c = lookup(pre_rid), lookup(post_rid)
    ok = (pre_c >= 0) & (post_c >= 0)
    n_skipped = int((~ok).sum())
    unknown_nt = int((~np.isin(np.char.upper(nt), ["ACH", "GABA", "GLUT", "DA", "SER", "OCT", "OA"])).sum())
    pre_c, post_c, syn, nt = pre_c[ok], post_c[ok], syn[ok], nt[ok]
    print(f"[build] connections rows={n_rows:,} skipped_unknown_root_id={n_skipped} unknown_nt_rows={unknown_nt} "
          f"({time.perf_counter() - t0:.1f}s)")

    # ---- populations ----
    pops = P.build_populations(annotations=annotations, source=source, edges=(pre_c, post_c, syn.astype(np.float64)))
    assert pops.N == N
    inv = pops.inverse_index()          # canonical -> sorted position
    print(f"[build] populations source={pops.source} N={N:,} ({time.perf_counter() - t0:.1f}s)")

    # ---- aggregate over neuropils ----
    agg = aggregate_edges(pre_c, post_c, syn, nt, N, mode)
    nnz_fb, nnz_mode, n_pairs = agg["nnz_flybrain"], agg["nnz_mode"], agg["n_pairs"]
    edges_match = nnz_fb == EXPECTED_EDGES
    print(f"[build] distinct (pre,post) pairs={n_pairs:,}; nonzero edges: flybrain-sign={nnz_fb:,} "
          f"(expected {EXPECTED_EDGES:,} -> {'MATCH' if edges_match else 'MISMATCH'}), {mode}-sign={nnz_mode:,}")
    if not edges_match:
        print(f"[build] WARNING flybrain-mode edge count {nnz_fb:,} != {EXPECTED_EDGES:,}; continuing", file=sys.stderr)
    keep = agg["signed"] != 0
    pre_s = inv[agg["pre"][keep]]        # sorted (population-order) indices
    post_s = inv[agg["post"][keep]]
    signed = agg["signed"][keep]
    unsigned = agg["unsigned"][keep]

    # ---- KC -> MBON block (dense, positive magnitudes) ----
    kc0, kc1 = pops.ranges["KC"]
    m0 = pops.ranges["MBON_APP"][0]
    m1 = pops.ranges["MBON_OTHER"][1]
    assert pops.ranges["MBON_APP"][1] == pops.ranges["MBON_AV"][0] and pops.ranges["MBON_AV"][1] == pops.ranges["MBON_OTHER"][0]
    n_kc, n_mbon = kc1 - kc0, m1 - m0
    is_km = (pre_s >= kc0) & (pre_s < kc1) & (post_s >= m0) & (post_s < m1)
    W_KM0 = np.zeros((n_kc, n_mbon), dtype=np.float32)
    W_KM0[pre_s[is_km] - kc0, post_s[is_km] - m0] = np.abs(unsigned[is_km]).astype(np.float32)
    M_KM = W_KM0 != 0
    n_km_neg = int((signed[is_km] < 0).sum())
    rest = ~is_km
    row = post_s[rest]
    col = pre_s[rest]
    values_raw = signed[rest].astype(np.float32)
    order = np.lexsort((col, row))       # coalesced: sorted by (row, col)
    row, col, values_raw = row[order], col[order], values_raw[order]
    indices = np.stack([row, col]).astype(np.int64)
    nnz = int(len(values_raw))
    print(f"[build] KC->MBON pairs={int(M_KM.sum()):,} (non-cholinergic sign on {n_km_neg} pairs ignored, magnitudes kept); "
          f"sparse nnz={nnz:,} ({time.perf_counter() - t0:.1f}s)")

    # ---- spectral radius and initial scale ----
    sr = spectral_radius(row, col, np.abs(values_raw), N)
    rho = sr["rho"]
    s0 = TARGET_RADIUS / rho
    values = np.clip(s0 * values_raw, -W_CLIP, W_CLIP).astype(np.float32)
    print(f"[build] rho(|W_raw|)={rho:.3f} (Collatz-Wielandt upper bound {sr['cw_upper']:.3f}, "
          f"{sr['iterations']} iters, converged={sr['converged']}) s0={s0:.6g} ({time.perf_counter() - t0:.1f}s)")

    # ---- artifact ----
    pop_ranges_json = json.dumps(pops.ranges)
    chash = content_hash(indices, values_raw, W_KM0, pops.root_ids, pop_ranges_json, mode)
    fname = f"fafb783_{chash[:12]}.npz"
    path = out_dir / fname
    build_cfg = {
        "nt_sign_mode": mode, "cell_types_source": pops.source,
        "annotations": str(Path(annotations) if annotations else P.ANNOTATIONS_TSV),
        "w_clip": W_CLIP, "target_radius": TARGET_RADIUS, "power_iter_max": POWER_ITER_MAX, "power_iter_tol": POWER_ITER_TOL,
        "expected_edges": EXPECTED_EDGES, "torch": torch.__version__, "numpy": np.__version__,
        "built_at": started_at.isoformat(), "content_sha256": chash, "file": fname,
        "population_order": pops.order,
    }
    counts = pops.counts()
    metrics = {
        "N": N, "rows": n_rows, "distinct_pairs": n_pairs, "nnz_flybrain_mode": nnz_fb, "nnz_mode": nnz_mode,
        "edges_expected": EXPECTED_EDGES, "edges_match": edges_match, "nnz_sparse": nnz,
        "skipped_rows": n_skipped, "unknown_nt_rows": unknown_nt,
        "kc_mbon_pairs": int(M_KM.sum()), "kc_mbon_max_syn": float(W_KM0.max()), "kc_mbon_mean_syn": float(W_KM0[M_KM].mean()),
        "spectral_radius_raw": rho, "spectral_radius_bracket": [sr["cw_lower"], sr["cw_upper"]],
        "power_iterations": sr["iterations"], "power_converged": sr["converged"], "s0": s0,
        "n_glomeruli": len(pops.glomerulus_names), "orn_food_untyped": int((pops.glomerulus_of_orn < 0).sum()),
        "populations": counts, "cell_types_source": pops.source,
        "build_seconds": None,
    }
    if path.exists():
        print(f"[build] {path} already exists with identical content hash; keeping it")
    else:
        tmp = out_dir / f"{fname}.tmp.npz"   # numpy appends .npz to names that lack it
        np.savez_compressed(
            tmp, indices=indices, values=values, values_raw=values_raw, N=np.int64(N),
            pop_ranges=np.array(pop_ranges_json), pop_order=np.array(pops.order),
            root_ids=pops.root_ids, nt=pops.nt.astype(str), flybrain_group=pops.flybrain_group,
            flybrain_group_names=np.array(pops.flybrain_group_names), cell_type=pops.cell_type.astype(str),
            W_KM0=W_KM0, M_KM=M_KM,
            kc_idx=pops.index("KC"), mbon_app_idx=pops.index("MBON_APP"), mbon_av_idx=pops.index("MBON_AV"),
            mbon_other_idx=pops.index("MBON_OTHER"), dan_pam_idx=pops.index("DAN_PAM"), dan_ppl1_idx=pops.index("DAN_PPL1"),
            glomerulus_of_orn=pops.glomerulus_of_orn, glomerulus_names=np.array(pops.glomerulus_names, dtype=str),
            spectral_radius_raw=np.float64(rho), s0=np.float64(s0), nt_sign_mode=np.array(mode),
            w_clip=np.float32(W_CLIP), source_sha256=np.array(json.dumps(source_sha)),
            build_config=np.array(json.dumps(build_cfg)), content_sha256=np.array(chash),
            canonical_index=pops.canonical_index,
        )
        os.replace(tmp, path)
    file_sha = sha256_file(path)
    CURRENT_FILE.parent.mkdir(parents=True, exist_ok=True)
    if out_dir == CONNECTOME_DIR:
        CURRENT_FILE.write_text(fname + "\n")
    metrics["build_seconds"] = round(time.perf_counter() - t0, 1)
    metrics["file_sha256"] = file_sha

    # ---- report ----
    print(f"[build] N={N:,} nnz={nnz:,} KC={counts['KC']} MBON_APP={counts['MBON_APP']} MBON_AV={counts['MBON_AV']} "
          f"MBON_OTHER={counts['MBON_OTHER']} DAN_PAM={counts['DAN_PAM']} DAN_PPL1={counts['DAN_PPL1']} "
          f"glomeruli={len(pops.glomerulus_names)}")
    for name in pops.order:
        print(f"    {name:<13} {counts[name]:>7}  kind={pops.kind.get(name, 'other')}")
    print(f"[build] wrote {path} ({path.stat().st_size / 1e6:.1f} MB, sha256 {file_sha[:12]}); current.txt -> {fname}")
    if counts["KC"] != P.EXPECTED_KC or (counts["MBON_APP"] + counts["MBON_AV"] + counts["MBON_OTHER"]) != P.EXPECTED_MBON:
        raise RuntimeError("KC/MBON counts do not match FAFB v783 expectations")  # build_populations already checks
    if persist_db:
        label = f"{pops.source}:{Path(build_cfg['annotations']).name if pops.source == 'annotations' else 'classification.csv.gz'}"
        run_id = persist(pops, metrics, {"build": build_cfg, "env": config.summary()}, file_sha, started_at, label)
        print(f"[build] runs row {run_id} (kind=connectome_build) and {len(pops.order)} populations rows written")
    return path


if __name__ == "__main__":
    main()
