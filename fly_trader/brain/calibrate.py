"""Brain calibration (``fly-trader calibrate-brain``) and self-test (``fly-trader brain-selftest``).

Calibration (plan, "Calibration" paragraph), all with the standard odor = 12 seeded random glomeruli
driven at ORN current 0.10:
 1. rho(|W_raw|) by power iteration (recomputed here, compared with the build's value).
 2. Weight scale s: sweep s in s0 * {0.25, 0.5, 1, 2, 4}, extended geometrically upward while the
    network stays quiet (the reservoir-derived s0 = 0.99/rho is dominated by the optic lobe and is
    far below what the olfactory pathway needs), then bisect (<= 6 extra points). Drive 300 ticks,
    silence 300 ticks; f_on = mean fraction of neurons firing per tick during the drive, f_tail = the
    same over the last 50 silent ticks, sigma = branching ratio over the silent phase
    (sum n_t n_{t+1} / sum n_t^2). Choose the largest s with f_on in [0.5%, 5%], f_tail <= 0.5%,
    sigma <= 1; if none qualifies pick the closest by violation score and flag it.
 3. KC->MBON scale s_km: the mean approach + avoidance MBON rate under the odor (rho_app + rho_av,
    the quantity the beat loop bounds; spikes per readout tick) in [0.15, 0.40], target the midpoint.
 4. DAN gain: currents {0.05, 0.1, 0.2, 0.3, 0.4} into DAN_PAM and DAN_PPL1 rows -> rates;
    dan_rate_max = rate at the largest current, dan_gain = the current giving half of it.
 5. Natural KC active fraction with k-WTA off (keep k-WTA unless it lies in [2%, 8%]).
 6. Odor separability: 64 random glomerular patterns -> KC codes; mean Jaccard across tokens (< 0.2
    wanted) and for the same pattern under 1% multiplicative noise (> 0.6 wanted).

Persists data/brain/calibration.json, data/brain/connectome/scale.json (read by Connectome.load) and
a ``runs`` row (kind = calibration).
"""
from __future__ import annotations

import json
import math
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .. import config
from . import lif as L

CALIBRATION_FILE = config.BRAIN_DIR / "calibration.json"
SCALE_FILE = L.SCALE_FILE

# ---- protocol constants (plan calibration paragraph; unsourced choices marked) ----
SEED = 0
ODOR_K = 12                 # = config.K_ODOR
ODOR_I = 0.10
DRIVE_TICKS = 300
SILENT_TICKS = 300
TAIL_TICKS = 50
F_ON_RANGE = (0.005, 0.05)
F_TAIL_MAX = 0.005
SIGMA_MAX = 1.0
SCALE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
SCALE_EXTEND_MAX = 4096.0   # relative to s0; unsourced safety cap for the geometric extension
BISECT_STEPS = 6
MBON_RANGE = (0.15, 0.40)
KM_GRID_MAX = 4096.0        # relative to s; unsourced cap
DAN_CURRENTS = (0.05, 0.1, 0.2, 0.3, 0.4)
KC_NATURAL_RANGE = (0.02, 0.08)
N_ODORS = 64
NOISE_FRAC = 0.01
SEP_ACROSS_MAX = 0.2
SEP_SAME_MIN = 0.6
SETTLE_BEATS = 2            # beats of the standard odor before a steady-state readout (unsourced)
PARITY_V_TOL = 1e-4
PARITY_SPIKE_TOL = 1e-3


def _jsonable(x):
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().tolist()
    if isinstance(x, float) and not math.isfinite(x):
        return None
    return x


# ---------------------------------------------------------------------------------------------
# stimuli and measurements
# ---------------------------------------------------------------------------------------------
def random_odors(n_glomeruli: int, n: int, seed: int = SEED, k: int = ODOR_K, I: float = ODOR_I) -> np.ndarray:
    """[G, n] glomerular currents: column j = k random glomeruli of rng(seed + j) at current I.
    Column 0 with seed=SEED is the standard odor."""
    A = np.zeros((n_glomeruli, n), dtype=np.float32)
    for j in range(n):
        rng = np.random.default_rng(seed + j)
        A[rng.choice(n_glomeruli, size=min(k, n_glomeruli), replace=False), j] = I
    return A


def branching_ratio(n: torch.Tensor) -> float:
    """sigma = sum_t n_t n_{t+1} / sum_t n_t^2 over a spike-count trace (0 if the trace is silent)."""
    x = n.double()
    a, b = x[:-1], x[1:]
    d = float((a * a).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def measure_scale(conn: L.Connectome, s: float, odor: np.ndarray) -> dict:
    """Drive the odor for DRIVE_TICKS then silence for SILENT_TICKS at scale s (s_km = s)."""
    conn.scale(s)
    conn.scale_km(s)
    lif = L.LIF(conn, odor.shape[1])
    lif.set_glomerulus_input(odor)
    r1 = lif.run(DRIVE_TICKS, readout_window=TAIL_TICKS, trace=True)
    lif.clear_input()
    r2 = lif.run(SILENT_TICKS, readout_window=TAIL_TICKS, trace=True)
    N = conn.N
    n_on = r1.spikes_per_tick[:, 0]
    n_off = r2.spikes_per_tick[:, 0]
    tail_pop = r2.pop_rates[:, 0]
    top_tail = sorted(((float(tail_pop[i]), conn.pop_order[i]) for i in range(conn.P)), reverse=True)[:8]
    return {
        "s": float(s), "mult": float(s / conn.s0),
        "f_on": float(n_on.mean()) / N, "f_on_last50": float(n_on[-TAIL_TICKS:].mean()) / N,
        "f_tail": float(n_off[-TAIL_TICKS:].mean()) / N, "f_off_first50": float(n_off[:TAIL_TICKS].mean()) / N,
        "sigma": branching_ratio(n_off), "nan": bool(r1.nan_flag or r2.nan_flag),
        "v_max_on": r1.v_max,
        "kc_rate_on": float(r1.pop_rates[conn.pop_id["KC"], 0]), "kc_active_frac_on": float(r1.kc_active_frac[0]),
        "mbon_app_on": float(r1.mbon_app_rate[0]), "mbon_av_on": float(r1.mbon_av_rate[0]),
        "alpn_rate_on": float(r1.pop_rates[conn.pop_id["ALPN"], 0]),
        "tail_top_populations": [(name, round(rate, 4)) for rate, name in top_tail],
    }


def qualifies(m: dict) -> bool:
    return (not m["nan"]) and F_ON_RANGE[0] <= m["f_on"] <= F_ON_RANGE[1] and m["f_tail"] <= F_TAIL_MAX and m["sigma"] <= SIGMA_MAX


def runaway(m: dict) -> bool:
    return m["nan"] or m["f_tail"] > F_TAIL_MAX or m["sigma"] > SIGMA_MAX or m["f_on"] > F_ON_RANGE[1]


def violation(m: dict) -> float:
    """Relative distance to the acceptance region (0 inside)."""
    if m["nan"]:
        return float("inf")
    v = max(0.0, F_ON_RANGE[0] - m["f_on"]) / F_ON_RANGE[0]
    v += max(0.0, m["f_on"] - F_ON_RANGE[1]) / F_ON_RANGE[1]
    v += max(0.0, m["f_tail"] - F_TAIL_MAX) / F_TAIL_MAX
    v += max(0.0, m["sigma"] - SIGMA_MAX)
    return v


def sweep_scale(conn: L.Connectome, odor: np.ndarray, verbose: bool = True) -> dict:
    s0 = conn.s0
    points: dict[float, dict] = {}

    def ev(mult: float) -> dict:
        mult = float(mult)
        if mult not in points:
            t0 = time.perf_counter()
            points[mult] = measure_scale(conn, s0 * mult, odor)
            m = points[mult]
            if verbose:
                print(f"[calibrate] s={m['s']:.6f} ({mult:.3g} x s0): f_on={m['f_on']*100:.3f}% f_tail={m['f_tail']*100:.3f}% "
                      f"sigma={m['sigma']:.3f} KC={m['kc_rate_on']:.4f} app={m['mbon_app_on']:.3f} av={m['mbon_av_on']:.3f} "
                      f"{'OK' if qualifies(m) else 'no'} ({time.perf_counter() - t0:.1f}s)")
        return points[mult]

    for mult in SCALE_GRID:
        ev(mult)
    # extend upward while nothing qualifies and the largest point has not run away
    while (not any(qualifies(m) for m in points.values()) and not runaway(points[max(points)])
           and max(points) < SCALE_EXTEND_MAX):
        ev(max(points) * 2.0)
    # extend while the largest point still qualifies (find the upper boundary)
    while qualifies(points[max(points)]) and max(points) < SCALE_EXTEND_MAX:
        ev(max(points) * 2.0)
    n_grid = len(points)
    good = sorted(k for k, m in points.items() if qualifies(m))
    flagged = False
    if good:
        lo = good[-1]
        above = sorted(k for k in points if k > lo)
        if above:
            hi = above[0]
            for _ in range(BISECT_STEPS):
                mid = math.sqrt(lo * hi)
                if qualifies(ev(mid)):
                    lo = mid
                else:
                    hi = mid
        chosen = lo
        reason = "largest s satisfying f_on in [0.5%,5%], f_tail <= 0.5%, sigma <= 1"
    else:
        flagged = True
        # refine around the least-violating point: 3 rounds x 2 points = 6 extra points
        for _ in range(3):
            best = min(points, key=lambda k: (violation(points[k]), -k))
            ks = sorted(points)
            i = ks.index(best)
            lo = ks[i - 1] if i > 0 else best / 2.0
            hi = ks[i + 1] if i + 1 < len(ks) else best * 2.0
            ev(math.sqrt(lo * best))
            ev(math.sqrt(best * hi))
        chosen = min(points, key=lambda k: (violation(points[k]), -k))
        reason = "no sweep point satisfied all criteria; closest by violation score"
    m = points[chosen]
    return {
        "s0": s0, "s": float(s0 * chosen), "mult": float(chosen), "flagged": flagged, "reason": reason,
        "chosen": m, "n_points": len(points), "n_grid_points": n_grid,
        "points": [points[k] for k in sorted(points)],
        "criteria": {"f_on_range": list(F_ON_RANGE), "f_tail_max": F_TAIL_MAX, "sigma_max": SIGMA_MAX,
                     "drive_ticks": DRIVE_TICKS, "silent_ticks": SILENT_TICKS, "tail_ticks": TAIL_TICKS},
    }


def steady_readout(conn: L.Connectome, odors: np.ndarray, extra=None, beats: int = SETTLE_BEATS, **lif_kw) -> L.Readout:
    """Reset, present ``odors`` [G, B] for ``beats`` beats of BEAT_TICKS and return the last beat's readout."""
    lif = L.LIF(conn, odors.shape[1], **lif_kw)
    lif.set_glomerulus_input(odors)
    if extra is not None:
        extra(lif)
    r = None
    for _ in range(beats):
        r = lif.run(trace=True)
    return r


def mbon_rate(conn: L.Connectome, odors: np.ndarray, s_km: float) -> dict:
    conn.scale_km(s_km)
    r = steady_readout(conn, odors)
    app, av = r.mbon_app_rate, r.mbon_av_rate
    return {"s_km": float(s_km), "mult": float(s_km / conn.s), "app": float(app.mean()), "av": float(av.mean()),
            "app_plus_av": float((app + av).mean()), "std_over_odors": float((app + av).std()),
            "m_hat_mean": float(r.m_hat.mean()), "m_hat_std": float(r.m_hat.std()), "nan": bool(r.nan_flag)}


def sweep_km(conn: L.Connectome, odors: np.ndarray, verbose: bool = True) -> dict:
    """Bisection on s_km toward the midpoint of MBON_RANGE for rho_app + rho_av."""
    target = 0.5 * (MBON_RANGE[0] + MBON_RANGE[1])
    base = conn.s
    points: dict[float, dict] = {}

    def ev(mult: float) -> dict:
        mult = float(mult)
        if mult not in points:
            points[mult] = mbon_rate(conn, odors, base * mult)
            m = points[mult]
            if verbose:
                print(f"[calibrate] s_km={m['s_km']:.6f} ({mult:.3g} x s): app={m['app']:.3f} av={m['av']:.3f} "
                      f"app+av={m['app_plus_av']:.3f} m_hat={m['m_hat_mean']:+.3f}+-{m['m_hat_std']:.3f}")
        return points[mult]

    for mult in (0.25, 0.5, 1.0, 2.0, 4.0):
        ev(mult)
    while points[max(points)]["app_plus_av"] < target and max(points) < KM_GRID_MAX:
        ev(max(points) * 2.0)
    while points[min(points)]["app_plus_av"] > target and min(points) > 1.0 / KM_GRID_MAX:
        ev(min(points) / 2.0)
    below = [k for k, m in points.items() if m["app_plus_av"] <= target]
    above = [k for k, m in points.items() if m["app_plus_av"] > target]
    if below and above:
        lo, hi = max(below), min(above)
        for _ in range(BISECT_STEPS):
            mid = math.sqrt(lo * hi)
            if ev(mid)["app_plus_av"] <= target:
                lo = mid
            else:
                hi = mid
    chosen = min(points, key=lambda k: (abs(points[k]["app_plus_av"] - target), k))
    m = points[chosen]
    ok = MBON_RANGE[0] <= m["app_plus_av"] <= MBON_RANGE[1]
    return {"s_km": float(base * chosen), "mult": float(chosen), "flagged": not ok, "chosen": m, "target": target,
            "range": list(MBON_RANGE), "quantity": "mean over odors of (rho_app + rho_av), spikes per readout tick",
            "points": [points[k] for k in sorted(points)]}


def dan_gain_curve(conn: L.Connectome, odor: np.ndarray, verbose: bool = True) -> dict:
    """Rates of DAN_PAM / DAN_PPL1 for injected currents (column 0 = no injection) under the odor."""
    currents = [0.0] + list(DAN_CURRENTS)
    odors = np.repeat(odor[:, :1], len(currents), axis=1)
    cur = torch.tensor(currents, dtype=torch.float32)

    def inject(lif: L.LIF):
        lif.set_population_input("DAN_PAM", cur)
        lif.set_population_input("DAN_PPL1", cur)

    r = steady_readout(conn, odors, extra=inject)
    pam = [float(x) for x in r.dan_pam_rate]
    ppl1 = [float(x) for x in r.dan_ppl1_rate]
    mean = [(a + b) / 2 for a, b in zip(pam, ppl1)]
    rate_max = mean[-1]
    half = 0.5 * rate_max
    gain = None
    for i in range(1, len(currents)):
        if mean[i] >= half:
            lo_c, hi_c, lo_r, hi_r = currents[i - 1], currents[i], mean[i - 1], mean[i]
            gain = lo_c + (hi_c - lo_c) * (half - lo_r) / (hi_r - lo_r) if hi_r > lo_r else hi_c
            break
    if gain is None:
        gain = currents[-1]
    if verbose:
        for c, a, b in zip(currents, pam, ppl1):
            print(f"[calibrate] DAN I={c:.2f}: PAM rate={a:.4f} PPL1 rate={b:.4f}")
        print(f"[calibrate] dan_rate_max={rate_max:.4f} dan_gain (half-max current)={gain:.4f}")
    return {"currents": currents, "pam_rate": pam, "ppl1_rate": ppl1, "mean_rate": mean,
            "dan_rate_max": float(rate_max), "pam_rate_max": pam[-1], "ppl1_rate_max": ppl1[-1],
            "dan_gain": float(gain), "baseline_pam": pam[0], "baseline_ppl1": ppl1[0],
            "protocol": f"{SETTLE_BEATS} beats of {config.BEAT_TICKS} ticks under the standard odor, readout last {config.READOUT_WINDOW}"}


def kc_sparsity(conn: L.Connectome, odors: np.ndarray, verbose: bool = True) -> dict:
    out = {}
    for name, kwta in (("kwta_off", False), ("kwta_on", True)):
        r = steady_readout(conn, odors, kwta=kwta)
        per_tick = r.kc_spikes_per_tick[-r.readout_window:] / conn.n_KC          # [ticks, B]
        out[name] = {"frac_per_tick": float(per_tick.mean()), "active_frac_window": float(r.kc_active_frac.mean()),
                     "kc_rate": float(r.kc_rates.mean())}
    nat = out["kwta_off"]["frac_per_tick"]
    keep = not (KC_NATURAL_RANGE[0] <= nat <= KC_NATURAL_RANGE[1])
    if verbose:
        print(f"[calibrate] KC natural active fraction per tick (k-WTA off)={nat*100:.2f}% "
              f"(window {out['kwta_off']['active_frac_window']*100:.1f}%); with k-WTA {out['kwta_on']['frac_per_tick']*100:.2f}% "
              f"-> keep k-WTA: {keep}")
    return {**out, "natural_frac": nat, "natural_range": list(KC_NATURAL_RANGE), "kwta_recommended": keep,
            "k": max(1, int(config.KC_KWTA_FRAC * conn.n_KC))}


def _jaccard_matrix(codes: torch.Tensor) -> torch.Tensor:
    """codes [n_KC, B] in {0,1} -> [B, B] Jaccard."""
    c = codes.float()
    inter = c.T @ c
    sz = c.sum(0)
    union = sz[:, None] + sz[None, :] - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-9), torch.ones_like(inter))


def _pairwise_same(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    inter = (a.float() * b.float()).sum(0)
    union = a.float().sum(0) + b.float().sum(0) - inter
    return torch.where(union > 0, inter / union.clamp(min=1e-9), torch.ones_like(inter))


def _code_metrics(K: torch.Tensor, K2: torch.Tensor, frac: float = config.KC_KWTA_FRAC) -> dict:
    """K, K2: KC rate matrices [n_KC, B] for the patterns and their noisy copies. Three code definitions:
    binary window code (spiked at least once in the readout window; the plan's metric), the top-frac KCs
    by rate (a sparse code of the same size as one k-WTA tick), and cosine similarity of the rate vectors."""
    K, K2 = K.cpu().float(), K2.cpu().float()
    B = K.shape[1]
    off = ~torch.eye(B, dtype=torch.bool)
    codes, codes2 = K > 0, K2 > 0
    J = _jaccard_matrix(codes)[off]
    k = max(1, int(frac * K.shape[0]))
    top = torch.zeros_like(K).scatter_(0, K.topk(k, dim=0).indices, 1.0)
    top2 = torch.zeros_like(K2).scatter_(0, K2.topk(k, dim=0).indices, 1.0)
    Jt = _jaccard_matrix(top)[off]
    Kn = K / K.norm(dim=0, keepdim=True).clamp(min=1e-9)
    K2n = K2 / K2.norm(dim=0, keepdim=True).clamp(min=1e-9)
    cos = (Kn.T @ Kn)[off]
    return {
        "jaccard_across_mean": float(J.mean()), "jaccard_across_max": float(J.max()),
        "jaccard_same_noisy_mean": float(_pairwise_same(codes, codes2).mean()),
        "jaccard_same_noisy_min": float(_pairwise_same(codes, codes2).min()),
        "code_size_mean": float(codes.float().sum(0).mean()), "code_frac_mean": float(codes.float().mean()),
        "topk_jaccard_across_mean": float(Jt.mean()), "topk_jaccard_same_noisy_mean": float(_pairwise_same(top, top2).mean()),
        "topk": k,
        "cosine_across_mean": float(cos.mean()), "cosine_same_noisy_mean": float((Kn * K2n).sum(0).mean()),
    }


def odor_separability(conn: L.Connectome, seed: int = SEED, verbose: bool = True) -> dict:
    """64 random 12-glomerulus patterns (equal currents, the plan's protocol) plus a graded variant
    (currents ODOR_I * U(0.25, 1.5) per active glomerulus, closer to the encoder's relu-normalised
    drive); each also under 1% multiplicative noise."""
    odors = random_odors(conn.n_glomeruli, N_ODORS, seed=seed + 1000)
    rng = np.random.default_rng(seed + 2000)
    noisy = (odors * (1.0 + NOISE_FRAC * rng.standard_normal(odors.shape))).astype(np.float32)
    graded = (odors * rng.uniform(0.25, 1.5, size=odors.shape)).astype(np.float32)
    graded_noisy = (graded * (1.0 + NOISE_FRAC * rng.standard_normal(odors.shape))).astype(np.float32)

    def codes_for(A: np.ndarray) -> torch.Tensor:
        lif = L.LIF(conn, N_ODORS)
        lif.set_glomerulus_input(A)
        return lif.run().kc_rates

    equal = _code_metrics(codes_for(odors), codes_for(noisy))
    grad = _code_metrics(codes_for(graded), codes_for(graded_noisy))
    out = {"n_odors": N_ODORS, "noise_frac": NOISE_FRAC, **equal, "graded": grad,
           "across_ok": equal["jaccard_across_mean"] < SEP_ACROSS_MAX, "same_ok": equal["jaccard_same_noisy_mean"] > SEP_SAME_MIN,
           "targets": {"across_max": SEP_ACROSS_MAX, "same_min": SEP_SAME_MIN},
           "code": "binary over the readout window (spiked at least once); topk_* = top 5% KCs by rate; cosine over rate vectors"}
    if verbose:
        print(f"[calibrate] separability (equal currents): Jaccard across tokens={out['jaccard_across_mean']:.3f} (< {SEP_ACROSS_MAX} "
              f"{'OK' if out['across_ok'] else 'FAIL'}), same token 1% noise={out['jaccard_same_noisy_mean']:.3f} "
              f"(> {SEP_SAME_MIN} {'OK' if out['same_ok'] else 'FAIL'}), KC window code size={out['code_size_mean']:.0f}; "
              f"top-5% code Jaccard across={out['topk_jaccard_across_mean']:.3f} same={out['topk_jaccard_same_noisy_mean']:.3f}; "
              f"cosine across={out['cosine_across_mean']:.3f} same={out['cosine_same_noisy_mean']:.3f}")
        print(f"[calibrate] separability (graded currents): Jaccard across={grad['jaccard_across_mean']:.3f} same={grad['jaccard_same_noisy_mean']:.3f}; "
              f"top-5% across={grad['topk_jaccard_across_mean']:.3f} same={grad['topk_jaccard_same_noisy_mean']:.3f}; "
              f"cosine across={grad['cosine_across_mean']:.3f} same={grad['cosine_same_noisy_mean']:.3f}")
    return out


def spectral_radius_check(conn: L.Connectome) -> dict:
    from .connectome_build import spectral_radius
    idx = conn.indices.numpy()
    sr = spectral_radius(idx[0], idx[1], np.abs(conn.values_raw.numpy()), conn.N)
    sr["rho_build"] = conn.spectral_radius_raw
    return sr


def persist(conn: L.Connectome, cal: dict, started_at: datetime) -> str:
    from psycopg.types.json import Jsonb
    from ..db.connection import transaction
    from .connectome_build import _git_sha
    run_id = str(uuid.uuid4())
    with transaction() as db:
        db.execute(
            "INSERT INTO runs (run_id, kind, started_at, ended_at, git_sha, config, connectome_sha256, status, metrics) "
            "VALUES (%s, 'calibration', %s, now(), %s, %s, %s, 'done', %s)",
            (run_id, started_at, _git_sha(), Jsonb({"env": config.summary(), "protocol": cal["protocol"]}),
             conn.sha256, Jsonb(cal)))
    return run_id


# ---------------------------------------------------------------------------------------------
def main(path: str | None = None, device: str | None = None, seed: int = SEED, persist_db: bool = True) -> dict:
    started_at = datetime.now(timezone.utc)
    t0 = time.perf_counter()
    conn = L.Connectome.load(path, device=device, use_scale_file=False)
    print(f"[calibrate] {conn.describe()}")
    odor = random_odors(conn.n_glomeruli, 4, seed=seed)     # column 0 = standard odor
    odors8 = random_odors(conn.n_glomeruli, 8, seed=seed)

    print("[calibrate] 1/6 spectral radius")
    sr = spectral_radius_check(conn)
    print(f"[calibrate] rho(|W_raw|)={sr['rho']:.3f} (build {sr['rho_build']:.3f}) s0={conn.s0:.6g}")

    print("[calibrate] 2/6 weight scale sweep")
    sc = sweep_scale(conn, odor)
    s = sc["s"]
    print(f"[calibrate] chosen s={s:.6f} ({sc['mult']:.3g} x s0) flagged={sc['flagged']}: f_on={sc['chosen']['f_on']*100:.3f}% "
          f"f_tail={sc['chosen']['f_tail']*100:.3f}% sigma={sc['chosen']['sigma']:.3f} -- {sc['reason']}")
    conn.scale(s)

    print("[calibrate] 3/6 KC->MBON scale")
    km = sweep_km(conn, odors8)
    s_km = km["s_km"]
    print(f"[calibrate] chosen s_km={s_km:.6f} ({km['mult']:.3g} x s) app+av={km['chosen']['app_plus_av']:.3f} flagged={km['flagged']}")
    conn.scale_km(s_km)

    print("[calibrate] 4/6 DAN gain")
    dan = dan_gain_curve(conn, odor)

    print("[calibrate] 5/6 KC sparsity")
    kc = kc_sparsity(conn, odors8)

    print("[calibrate] 6/6 odor separability")
    sep = odor_separability(conn, seed)

    cal = {
        "s": s, "s_km": s_km, "s0": conn.s0, "s_mult": sc["mult"], "s_km_mult": km["mult"],
        "dan_gain": dan["dan_gain"], "dan_rate_max": dan["dan_rate_max"],
        "dan_baseline_pam": dan["baseline_pam"], "dan_baseline_ppl1": dan["baseline_ppl1"],
        "dan_rate_max_above_baseline": dan["dan_rate_max"] - 0.5 * (dan["baseline_pam"] + dan["baseline_ppl1"]),
        "f_on": sc["chosen"]["f_on"], "f_tail": sc["chosen"]["f_tail"], "sigma": sc["chosen"]["sigma"],
        "scale_flagged": sc["flagged"], "s_km_flagged": km["flagged"],
        "mbon_app_rate": km["chosen"]["app"], "mbon_av_rate": km["chosen"]["av"], "mbon_app_plus_av": km["chosen"]["app_plus_av"],
        "kc_natural_frac": kc["natural_frac"], "kwta_recommended": kc["kwta_recommended"],
        "jaccard_across": sep["jaccard_across_mean"], "jaccard_same_noisy": sep["jaccard_same_noisy_mean"],
        "separability_ok": bool(sep["across_ok"] and sep["same_ok"]),
        "cosine_across": sep["cosine_across_mean"], "topk_jaccard_across": sep["topk_jaccard_across_mean"],
        "spectral_radius": sr, "scale_sweep": sc, "km_sweep": km, "dan": dan, "kc_sparsity": kc, "separability": sep,
        "connectome": conn.path.name, "connectome_sha256": conn.sha256, "content_sha256": conn.content_sha256,
        "device": str(conn.device), "seed": seed, "standard_odor_glomeruli": [conn.glomerulus_names[i] for i in np.nonzero(odor[:, 0])[0]],
        "protocol": {"odor_k": ODOR_K, "odor_I": ODOR_I, "drive_ticks": DRIVE_TICKS, "silent_ticks": SILENT_TICKS,
                     "tail_ticks": TAIL_TICKS, "beat_ticks": config.BEAT_TICKS, "readout_window": config.READOUT_WINDOW,
                     "leak": config.LEAK, "theta": config.THETA, "refractory": config.REFRACTORY,
                     "kwta": config.KWTA_ENABLED, "kwta_frac": config.KC_KWTA_FRAC, "nt_sign_mode": conn.nt_sign_mode,
                     "dan_currents": list(DAN_CURRENTS), "n_odors": N_ODORS, "noise_frac": NOISE_FRAC},
        "seconds": round(time.perf_counter() - t0, 1), "ts": started_at.isoformat(),
    }
    cal = _jsonable(cal)
    CALIBRATION_FILE.parent.mkdir(parents=True, exist_ok=True)
    CALIBRATION_FILE.write_text(json.dumps(cal, indent=2))
    SCALE_FILE.write_text(json.dumps({"connectome": conn.path.name, "content_sha256": conn.content_sha256,
                                      "s": s, "s_km": s_km, "ts": started_at.isoformat()}, indent=2))
    print(f"[calibrate] wrote {CALIBRATION_FILE} and {SCALE_FILE} ({cal['seconds']}s)")
    if persist_db:
        run_id = persist(conn, cal, started_at)
        print(f"[calibrate] runs row {run_id} (kind=calibration)")
        cal["run_id"] = run_id
    return cal


def selftest(path: str | None = None, ticks: int = 20, batch: int = 4, timing_ticks: int = config.BEAT_TICKS,
             timing_batch: int = config.SLOTS, timing_repeats: int = 2) -> dict:
    """Timing (50 ticks x batch 128, best of ``timing_repeats``) then MPS-vs-CPU parity (20 ticks, batch 4).
    Timing runs first so it is measured before the CPU copy of the connectome is resident."""
    dev = L.resolve_device()
    print(f"[selftest] device={dev} torch={torch.__version__} mps_available={torch.backends.mps.is_available()}")
    c_dev = L.Connectome.load(path, device=dev)
    print(f"[selftest] {c_dev.describe()}")
    runs = [L.timing(timing_ticks, timing_batch, connectome=c_dev) for _ in range(max(1, timing_repeats))]
    t = min(runs, key=lambda r: r["ms_per_tick"])
    runs_str = ", ".join(f"{r['ms_per_tick']:.2f}" for r in runs)
    print(f"[selftest] timing {timing_ticks} ticks x batch {timing_batch} on {t['device']}: best {t['ms_per_tick']:.2f} ms/tick, "
          f"{t['s_per_beat']:.3f} s/beat (target < 1.1 s; runs: {runs_str} ms/tick); "
          f"spikes={t['total_spikes']:.0f} nan={t['nan_flag']}")
    out = {"device": str(dev), "connectome": c_dev.path.name, "s": c_dev.s, "s_km": c_dev.s_km,
           "ms_per_tick": t["ms_per_tick"], "s_per_beat": t["s_per_beat"], "ms_per_tick_runs": [r["ms_per_tick"] for r in runs],
           "timing_ticks": timing_ticks, "timing_batch": timing_batch, "nan_flag": t["nan_flag"]}
    odor = random_odors(c_dev.n_glomeruli, batch)
    a = L.LIF(c_dev, batch, ticks=ticks)
    a.set_glomerulus_input(odor)
    ra = a.run(ticks)
    if dev.type != "cpu":
        c_cpu = L.Connectome.load(path, device="cpu", s=c_dev.s, s_km=c_dev.s_km)
        b = L.LIF(c_cpu, batch, ticks=ticks)
        b.set_glomerulus_input(odor)
        rb = b.run(ticks)
        v_diff = float((a.V.cpu() - b.V).abs().max())
        spikes_dev, spikes_cpu = ra.total_spikes, rb.total_spikes
        rel = abs(spikes_dev - spikes_cpu) / max(spikes_cpu, 1.0)
        sum_diff = float((ra.spike_sum.cpu() - rb.spike_sum).abs().sum())
        ok = v_diff < PARITY_V_TOL and rel <= PARITY_SPIKE_TOL
        print(f"[selftest] parity {dev} vs cpu over {ticks} ticks x {batch}: max|dV|={v_diff:.2e} (< {PARITY_V_TOL}) "
              f"spikes {spikes_dev:.0f} vs {spikes_cpu:.0f} (rel diff {rel:.2e} <= {PARITY_SPIKE_TOL}) "
              f"spike-sum L1 diff={sum_diff:.0f} -> {'PASS' if ok else 'FAIL'}")
        out.update({"parity_v_maxdiff": v_diff, "parity_spikes_device": spikes_dev, "parity_spikes_cpu": spikes_cpu,
                    "parity_rel_diff": rel, "parity_ok": ok})
        del b, c_cpu
    else:
        print("[selftest] cpu only: parity check skipped")
        out["parity_ok"] = None
    if out.get("parity_ok") is False or t["nan_flag"]:
        raise RuntimeError(f"brain self-test failed: {out}")
    return out


if __name__ == "__main__":
    main()
