"""Balanced KC→MBON prior and KC-code separability check (run after calibrate-brain).

balance_km(): finds column multipliers k_app, k_av for the approach / avoidance MBON blocks of W_KM0
so that, under a panel of random odors, mean approach and avoidance MBON rates are equal — i.e. the
untrained valence m̂ is centred at zero, matching the symmetric initialisation of Bennett et al. 2021.
separability(): Jaccard overlap of the KC codes (KCs that spiked in the readout window) across
random tokens, and for the same token with 1 % feature noise, through the real Encoder path.
"""
from __future__ import annotations

import json
import logging

import numpy as np
import torch

from .. import config
from ..market.features import D
from .encoders import Encoder
from .lif import LIF, Connectome

log = logging.getLogger(__name__)


def _panel(enc: Encoder, B: int, rng: np.random.Generator, noise: float = 0.0, base=None):
    feats = rng.standard_normal((B, D)).astype(np.float32) if base is None else base + noise * rng.standard_normal((B, D)).astype(np.float32)
    masks = np.full(B, (1 << 30) | ((1 << 30) - 1), dtype=np.int64)
    mints = [f"tok{i}" for i in range(B)]
    active = np.ones(B, dtype=bool)
    z = np.zeros(B, dtype=np.float32)
    enc.std.frozen = True
    enc.std.n[:] = 100; enc.std.mean[:] = 0.0; enc.std.m2[:] = 99.0
    I, drive, stim = enc.build(feats, masks, mints, active, z, z, z, z, z, z, update_stats=False)
    enc.last_I_keep = enc.last_keep
    return feats, I


def balance_km(iters: int = 5, B: int = 64, settle_beats: int = 2) -> dict:
    c = Connectome.load()
    lif = LIF(c, batch=B)
    enc = Encoder(c, B)
    rng = np.random.default_rng(7)
    feats, I = _panel(enc, B, rng)
    lif.set_input(I); lif.set_kc_winners(enc.last_keep)
    k_app, k_av = 1.0, 1.0
    W0 = c.W_KM0.clone()
    hist = []
    for it in range(iters):
        W = W0.clone()
        W[:, c.mbon_app_cols] *= k_app
        W[:, c.mbon_av_cols] *= k_av
        lif.set_W_KM(W)
        lif.reset()
        for _ in range(settle_beats):
            ro = lif.run()
        app, av = float(ro.mbon_app_rate.mean()), float(ro.mbon_av_rate.mean())
        m = float(ro.m_hat.mean())
        hist.append({"iter": it, "k_app": k_app, "k_av": k_av, "app": app, "av": av, "m_hat_mean": m, "m_hat_std": float(ro.m_hat.std())})
        log.info("balance iter %d k_app=%.3f k_av=%.3f app=%.4f av=%.4f m_hat=%.3f", it, k_app, k_av, app, av, m)
        if abs(m) < 0.03:
            break
        ratio = (app + 1e-4) / (av + 1e-4)
        if ratio < 1:
            k_app *= ratio ** -0.7
        else:
            k_av *= ratio ** 0.7
        k_app, k_av = min(k_app, 8.0), min(k_av, 8.0)
    out = {"km_balance_app": hist[-1]["k_app"], "km_balance_av": hist[-1]["k_av"], "balance_history": hist}
    p = config.BRAIN_DIR / "calibration.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d.update(out)
    p.write_text(json.dumps(d, indent=1, default=str))
    return out


def separability(B: int = 64, noise: float = 0.01) -> dict:
    c = Connectome.load()
    lif = LIF(c, batch=B)
    enc = Encoder(c, B)
    calib = json.loads((config.BRAIN_DIR / "calibration.json").read_text()) if (config.BRAIN_DIR / "calibration.json").exists() else {}
    W = c.W_KM0.clone()
    W[:, c.mbon_app_cols] *= float(calib.get("km_balance_app") or 1.0)
    W[:, c.mbon_av_cols] *= float(calib.get("km_balance_av") or 1.0)
    lif.set_W_KM(W)
    rng = np.random.default_rng(11)
    feats, I = _panel(enc, B, rng)
    lif.set_input(I); lif.set_kc_winners(enc.last_keep); lif.reset(); lif.run(); ro1 = lif.run()
    code1 = (ro1.kc_rates > 0).cpu()
    _, I2 = _panel(enc, B, rng, noise=noise, base=feats)
    lif.set_input(I2); lif.set_kc_winners(enc.last_keep); lif.reset(); lif.run(); ro2 = lif.run()
    code2 = (ro2.kc_rates > 0).cpu()
    inter = code1.float().T @ code1.float()
    sizes = code1.float().sum(0)
    union = sizes[:, None] + sizes[None, :] - inter
    J = inter / union.clamp(min=1)
    off = ~torch.eye(B, dtype=torch.bool)
    same = ((code1 & code2).float().sum(0) / (code1 | code2).float().sum(0).clamp(min=1))
    out = {"jaccard_across": float(J[off].mean()), "jaccard_across_max": float(J[off].max()), "jaccard_same_noisy": float(same.mean()),
           "kc_frac": float(sizes.mean() / c.n_KC), "m_hat_mean": float(ro1.m_hat.mean()), "m_hat_std": float(ro1.m_hat.std()),
           "app": float(ro1.mbon_app_rate.mean()), "av": float(ro1.mbon_av_rate.mean()), "kwta_tau": lif.kwta_tau,
           "k_odor": config.K_ODOR, "identity_scale": config.IDENTITY_SCALE}
    p = config.BRAIN_DIR / "calibration.json"
    d = json.loads(p.read_text()) if p.exists() else {}
    d["separability_v2"] = out
    d["m_hat_baseline"] = out["m_hat_mean"]
    p.write_text(json.dumps(d, indent=1, default=str))
    return out


def main() -> None:
    from ..logging_setup import setup
    setup("balance", to_file=False)
    print(json.dumps(balance_km(), indent=1, default=str))
    print(json.dumps(separability(), indent=1, default=str))


def scale_sweep(mults=(1.0, 0.7, 0.5, 0.35, 0.25, 0.18, 0.125, 0.09), B: int = 32) -> list[dict]:
    """Re-check the global scale under the REAL encoder input (odor + tonic + sensory rows).
    Reports activity, ORN/PN/KC/MBON rates and KC-code separability per scale multiplier of the
    calibrated s. The calibration odor is far weaker than the encoder's input, so its s can be too hot."""
    c = Connectome.load()
    s_cal = float(c.scale_s) if hasattr(c, "scale_s") else None
    calib = json.loads((config.BRAIN_DIR / "calibration.json").read_text())
    s_cal = s_cal or float(calib["s"])
    lif = LIF(c, batch=B)
    enc = Encoder(c, B)
    rng = np.random.default_rng(5)
    feats, I = _panel(enc, B, rng)
    _, I2 = _panel(enc, B, rng, noise=0.01, base=feats)
    out = []
    off = ~torch.eye(B, dtype=torch.bool)
    for m in mults:
        s = s_cal * m
        c.scale(s)
        lif.set_W_KM(c.W_KM0)
        lif.set_input(I); lif.reset(); lif.run(); ro = lif.run()
        code = (ro.kc_rates > 0).cpu().float()
        inter = code.T @ code; sz = code.sum(0); J = inter / (sz[:, None] + sz[None, :] - inter).clamp(min=1)
        lif.set_input(I2); lif.reset(); lif.run(); ro2 = lif.run()
        code2 = (ro2.kc_rates > 0).cpu().float()
        same = ((code * code2).sum(0) / ((code + code2) > 0).float().sum(0).clamp(min=1)).mean()
        pop = ro.pop_rates
        r = {"mult": m, "s": s, "total_pct": float(ro.total_spikes) / lif.ticks / c.N * 100, "orn": float(pop[c.pop_id["ORN_FOOD"]].mean()),
             "pn": float(pop[c.pop_id["ALPN"]].mean()), "kc_frac": float(sz.mean()) / c.n_KC, "app": float(ro.mbon_app_rate.mean()),
             "av": float(ro.mbon_av_rate.mean()), "m_mean": float(ro.m_hat.mean()), "m_std": float(ro.m_hat.std()),
             "J_across": float(J[off].mean()), "J_same": float(same), "dan_pam": float(ro.dan_pam_rate.mean()), "dan_ppl1": float(ro.dan_ppl1_rate.mean())}
        out.append(r)
        print(f"x{m:<5} s={s:.5f} total={r['total_pct']:.2f}%/tick ORN={r['orn']:.3f} PN={r['pn']:.3f} KC={r['kc_frac']:.3f} app={r['app']:.4f} av={r['av']:.4f} "
              f"m={r['m_mean']:+.3f}±{r['m_std']:.3f} J_across={r['J_across']:.3f} J_same={r['J_same']:.3f} PAM={r['dan_pam']:.3f} PPL1={r['dan_ppl1']:.3f}", flush=True)
    d = json.loads((config.BRAIN_DIR / "calibration.json").read_text())
    d["scale_sweep_encoder"] = out
    (config.BRAIN_DIR / "calibration.json").write_text(json.dumps(d, indent=1, default=str))
    return out


def pathway_sweep(s_mults=(0.25, 0.35), gains=(2.0, 4.0, 8.0, 16.0, 32.0), B: int = 32, pathway=("ALPN", "KC")) -> list[dict]:
    """At a sparse global scale, strengthen one pathway (default PN->KC) and measure KC sparsity,
    code separability and MBON rates under the real encoder input."""
    c = Connectome.load()
    calib = json.loads((config.BRAIN_DIR / "calibration.json").read_text())
    s_cal = float(calib["s"])
    lif = LIF(c, batch=B)
    enc = Encoder(c, B)
    rng = np.random.default_rng(5)
    feats, I = _panel(enc, B, rng)
    _, I2 = _panel(enc, B, rng, noise=0.01, base=feats)
    off = ~torch.eye(B, dtype=torch.bool)
    out = []
    for sm in s_mults:
        c.scale(s_cal * sm)
        for g in gains:
            n = c.set_pathway_gain(pathway[0], pathway[1], g)
            lif.set_W_KM(c.W_KM0)
            lif.set_input(I); lif.reset(); lif.run(); ro = lif.run()
            code = (ro.kc_rates > 0).cpu().float()
            inter = code.T @ code; sz = code.sum(0); J = inter / (sz[:, None] + sz[None, :] - inter).clamp(min=1)
            lif.set_input(I2); lif.reset(); lif.run(); ro2 = lif.run()
            code2 = (ro2.kc_rates > 0).cpu().float()
            same = float(((code * code2).sum(0) / ((code + code2) > 0).float().sum(0).clamp(min=1)).mean())
            pop = ro.pop_rates
            r = {"s_mult": sm, "s": s_cal * sm, "gain": g, "edges": n, "total_pct": float(ro.total_spikes) / lif.ticks / c.N * 100,
                 "pn": float(pop[c.pop_id["ALPN"]].mean()), "kc_frac": float(sz.mean()) / c.n_KC, "kc_rate": float(ro.kc_rates.mean()),
                 "app": float(ro.mbon_app_rate.mean()), "av": float(ro.mbon_av_rate.mean()), "m_mean": float(ro.m_hat.mean()),
                 "m_std": float(ro.m_hat.std()), "J_across": float(J[off].mean()), "J_same": same}
            out.append(r)
            print(f"s x{sm} gain {g:<4} edges={n} total={r['total_pct']:.2f}% PN={r['pn']:.3f} KC={r['kc_frac']:.3f} kc_rate={r['kc_rate']:.4f} "
                  f"app={r['app']:.4f} av={r['av']:.4f} m={r['m_mean']:+.3f}±{r['m_std']:.3f} J_across={r['J_across']:.3f} J_same={r['J_same']:.3f}", flush=True)
    d = json.loads((config.BRAIN_DIR / "calibration.json").read_text())
    d["pathway_sweep"] = out
    (config.BRAIN_DIR / "calibration.json").write_text(json.dumps(d, indent=1, default=str))
    return out
