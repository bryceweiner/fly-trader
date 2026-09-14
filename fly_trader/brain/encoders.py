"""Sensory encoders: market features → input currents on the fly's sensory populations.

- Odor (token identity + feature pattern): fixed Gaussian projection of [standardized features; identity
  hash] onto olfactory glomeruli, top-K glomeruli kept, ORNs of each glomerulus driven proportionally
  (the connectome's ORN→PN→KC expansion plus k-WTA then yields the sparse KC code; Dasgupta, Stevens,
  Navlakha 2017 Science).
- Tastes: realized profit → sugar GRNs, loss → bitter GRNs (plus danger → bitter).
- Danger score → pheromone/danger ORNs. Volatility/burstiness → Johnston's organ (wind). Buy/sell
  imbalance → warm/cool thermosensors. Tonic drive to the central complex; hunger from time since entry.
- Dopamine: reward-prediction error → PAM (reward) and PPL1 (punishment) DANs (Bennett et al. 2021).
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
from pathlib import Path

import numpy as np
import torch

from .. import config
from ..market.features import D, FIDX, STATS_BIT

log = logging.getLogger(__name__)

ODOR_NAME = "odor_v1"
STANDARDIZER_NAME = "standardizer_v1"


def identity_vector(mint: str, dim: int = config.IDENTITY_DIM) -> np.ndarray:
    h = hashlib.sha256(mint.encode()).digest()
    bits = np.unpackbits(np.frombuffer(h, dtype=np.uint8))[:dim]
    return (bits.astype(np.float32) * 2 - 1) * config.IDENTITY_SCALE


class Standardizer:
    """Welford running mean/var per feature, updated only where the feature mask bit is set."""

    def __init__(self, d: int = D):
        self.d = d
        self.n = np.zeros(d, dtype=np.float64)
        self.mean = np.zeros(d, dtype=np.float64)
        self.m2 = np.zeros(d, dtype=np.float64)
        self.frozen = False

    def _mask_matrix(self, masks: np.ndarray) -> np.ndarray:
        bits = np.zeros((len(masks), self.d), dtype=bool)
        for i, m in enumerate(masks):
            m = int(m)
            for j in range(self.d):
                if m & (1 << j):
                    bits[i, j] = True
            if m & STATS_BIT:
                bits[i, FIDX["organic_score"]:] = True
        return bits

    def update(self, feats: np.ndarray, masks: np.ndarray) -> None:
        if self.frozen:
            return
        bits = self._mask_matrix(masks)
        for i in range(feats.shape[0]):
            b = bits[i]
            if not b.any():
                continue
            x = feats[i, b]
            self.n[b] += 1
            delta = x - self.mean[b]
            self.mean[b] += delta / self.n[b]
            self.m2[b] += delta * (x - self.mean[b])

    def transform(self, feats: np.ndarray, masks: np.ndarray) -> np.ndarray:
        bits = self._mask_matrix(masks)
        var = np.where(self.n > 1, self.m2 / np.maximum(self.n - 1, 1), 1.0)
        std = np.sqrt(np.maximum(var, 1e-12))
        z = (feats - self.mean) / np.where(self.n > 10, std, 1.0)
        z = np.clip(z, -5, 5)
        z[~bits] = 0.0
        return z.astype(np.float32)

    def state(self) -> dict:
        return {"n": self.n.tolist(), "mean": self.mean.tolist(), "m2": self.m2.tolist()}

    def load_state(self, st: dict) -> None:
        if not st:
            return
        n, mean, m2 = np.array(st["n"]), np.array(st["mean"]), np.array(st["m2"])
        if len(n) == self.d:
            self.n, self.mean, self.m2 = n, mean, m2


class OdorProjection:
    def __init__(self, n_glomeruli: int, d_in: int = D + config.IDENTITY_DIM, seed: int = 20260912,
                 path: Path | None = None):
        self.G = n_glomeruli
        self.d_in = d_in
        self.seed = seed
        self.path = path or (config.ENCODER_DIR / f"{ODOR_NAME}.npz")
        self.P: np.ndarray | None = None
        self.sha256: str | None = None

    def load_or_create(self) -> "OdorProjection":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            z = np.load(self.path)
            if z["P"].shape == (self.G, self.d_in):
                self.P = z["P"].astype(np.float32)
                self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
                return self
            log.warning("odor projection shape mismatch; recreating")
        rng = np.random.default_rng(self.seed)
        self.P = (rng.standard_normal((self.G, self.d_in)) / math.sqrt(self.d_in)).astype(np.float32)
        np.savez(self.path, P=self.P, seed=self.seed, G=self.G, d_in=self.d_in)
        self.sha256 = hashlib.sha256(self.path.read_bytes()).hexdigest()
        return self

    def glomerular_drive(self, z: np.ndarray, ident: np.ndarray, k: int = config.K_ODOR) -> np.ndarray:
        """[B, D] standardized features + [B, 16] identity → [B, G] drive with top-k glomeruli, max 1."""
        x = np.concatenate([z, ident], axis=1).astype(np.float32)
        u = x @ self.P.T
        u = np.maximum(u - u.mean(axis=1, keepdims=True), 0.0)   # Dasgupta 2017 step 1: centre the mean, then rectify
        if k < self.G:
            kth = np.partition(u, self.G - k, axis=1)[:, self.G - k][:, None]
            u = np.where(u >= kth, u, 0.0)
        mx = u.max(axis=1, keepdims=True)
        return np.where(mx > 0, u / np.maximum(mx, 1e-9), 0.0).astype(np.float32)


class Encoder:
    """Builds the [N, B] external-current tensor for one beat."""

    def __init__(self, connectome, batch: int, dan_gain: float | None = None):
        self.c = connectome
        self.B = batch
        self.dev = connectome.device
        self.N = connectome.N
        self.dan_gain = dan_gain if dan_gain is not None else config.I_ORN_MAX
        self.ranges = connectome.pop_ranges
        # glomerulus membership for ORN_FOOD rows: dense [n_orn, G]
        lo, hi = self.ranges["ORN_FOOD"]
        glom = np.asarray(connectome.glomerulus_of_orn)
        self.G = int(glom.max()) + 1 if glom.size and glom.max() >= 0 else 0
        gm = np.zeros((hi - lo, max(self.G, 1)), dtype=np.float32)
        for r, g in enumerate(glom):
            if 0 <= g < self.G:
                gm[r, g] = 1.0
        self.glom_matrix = torch.tensor(gm, device=self.dev)
        self.odor = OdorProjection(max(self.G, 1)).load_or_create()
        self.std = Standardizer()
        # KC odor code (Dasgupta, Stevens, Navlakha 2017 Science): the glomerular pattern is projected through
        # the connectome's own glomerulus -> uniglomerular PN -> KC synapse counts (row-normalised so hub KCs do
        # not dominate) and the top KC_KWTA_FRAC of KCs win. Winners receive a sustained current; the LIF
        # antennal lobe at a uniform synaptic scale does not preserve odor identity (measured: PN-code
        # Jaccard 0.82 across odors), so the projection is computed here and injected.
        self.G2KC = self._build_g2kc()
        self.k_kc = max(1, int(config.KC_KWTA_FRAC * connectome.n_KC))

    def _build_g2kc(self) -> torch.Tensor:
        c = self.c
        import re
        names = [str(x) for x in c.glomerulus_names]
        gidx = {n.replace("ORN_", ""): i for i, n in enumerate(names)}
        a0, a1 = c.pop_ranges["ALPN"]
        k0, k1 = c.pop_ranges["KC"]
        pn_glom = torch.full((a1 - a0,), -1, dtype=torch.int64)
        if c.cell_type is not None:
            for j, t in enumerate(c.cell_type[a0:a1]):
                t = str(t)
                mm = re.match(r"^([A-Za-z0-9+]+)_(ad|l|lv|il|v|vl)?PN", t)
                if mm and not t.startswith("M_") and mm.group(1) in gidx:
                    pn_glom[j] = gidx[mm.group(1)]
        pre, post = c.indices[1], c.indices[0]
        m = (pre >= a0) & (pre < a1) & (post >= k0) & (post < k1)
        g_of_edge = pn_glom[pre[m] - a0]
        ok = g_of_edge >= 0
        rows = (post[m] - k0)[ok]
        cols = g_of_edge[ok]
        vals = c.values_raw[m][ok].abs()
        G2KC = torch.zeros((k1 - k0, max(self.G, 1)), dtype=torch.float32)
        G2KC.index_put_((rows, cols), vals, accumulate=True)
        rs = G2KC.sum(1, keepdim=True)
        G2KC = torch.where(rs > 0, G2KC / rs.clamp(min=1e-6), G2KC)
        self.n_kc_with_glom = int((rs.squeeze(1) > 0).sum())
        return G2KC.to(self.dev)

    def kc_winners(self, drive: np.ndarray, active: np.ndarray) -> torch.Tensor:
        """drive [B, G] glomerular pattern -> bool [n_KC, B] winner mask (top-k KCs per column)."""
        d = self.G2KC @ torch.tensor(drive.T, device=self.dev, dtype=torch.float32)   # [n_KC, B]
        # deterministic tie-break so equal drives do not favour low indices
        tb = torch.arange(d.shape[0], device=self.dev, dtype=torch.float32)[:, None] * 1e-7
        d = d + tb
        top = torch.topk(d, self.k_kc, dim=0, sorted=False).indices
        keep = torch.zeros_like(d, dtype=torch.bool)
        keep.scatter_(0, top, True)
        keep &= (d > 1e-6)
        keep &= torch.tensor(active, device=self.dev)[None, :]
        return keep

    def _fill(self, I: torch.Tensor, name: str, values: torch.Tensor | float) -> None:
        if name not in self.ranges:
            return
        lo, hi = self.ranges[name]
        if hi <= lo:
            return
        if isinstance(values, torch.Tensor):
            I[lo:hi] = values[None, :]
        else:
            I[lo:hi] = values

    def build(self, feats: np.ndarray, masks: np.ndarray, mints: list[str | None], active: np.ndarray,
              sweet: np.ndarray, bitter: np.ndarray, danger: np.ndarray, hunger: np.ndarray,
              dan_rew: np.ndarray, dan_pun: np.ndarray, update_stats: bool = True) -> tuple[torch.Tensor, np.ndarray, np.ndarray]:
        """Returns (I_ext [N,B] on device, glomerular drive [B,G], stim summary [B, 8]); B = rows of feats."""
        B = feats.shape[0]
        if update_stats:
            self.std.update(feats[active], masks[active])
        z = self.std.transform(feats, masks)
        ident = np.stack([identity_vector(m) if m else np.zeros(config.IDENTITY_DIM, np.float32) for m in mints])
        drive = self.odor.glomerular_drive(z, ident)
        drive[~active] = 0.0
        I = torch.zeros((self.N, B), device=self.dev, dtype=torch.float32)
        act = torch.tensor(active.astype(np.float32), device=self.dev)
        # odor: ORNs (innate pathways through the AL / lateral horn) and the KC hash code (winners driven)
        keep = None
        if self.G > 0:
            lo, hi = self.ranges["ORN_FOOD"]
            I[lo:hi] = config.I_ORN_MAX * (self.glom_matrix @ torch.tensor(drive.T, device=self.dev))
            keep = self.kc_winners(drive, active)
            k0, k1 = self.ranges["KC"]
            I[k0:k1] = config.I_KC * keep.float()
        # wind: volatility + burstiness (standardized)
        wind = 1.0 / (1.0 + np.exp(-(z[:, FIDX["rvol_5m"]] + z[:, FIDX["hawkes"]])))
        imb = feats[:, FIDX["imb_5m"]]
        warm = np.clip(imb, 0, 1)
        cool = np.clip(-imb, 0, 1)
        t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32), device=self.dev) * act
        self._fill(I, "ORN_DANGER", config.I_DANGER_MAX * t(danger))
        self._fill(I, "GRN_SWEET", config.G_TASTE * t(np.clip(sweet, 0, 3)))
        self._fill(I, "GRN_BITTER", config.G_TASTE * t(np.clip(bitter, 0, 3)) + 0.05 * t(danger))
        self._fill(I, "MECH_JO", config.I_JO_MAX * t(wind))
        self._fill(I, "THERMO_WARM", config.I_THERMO_MAX * t(warm))
        self._fill(I, "THERMO_COOL", config.I_THERMO_MAX * t(cool))
        self._fill(I, "CX", config.I_CX_TONIC * act)
        self._fill(I, "HUNGER", config.I_HUNGER_MAX * t(np.clip(hunger, 0, 1)))
        self._fill(I, "DAN_PAM", self.dan_gain * t(np.clip(dan_rew, 0, config.DELTA_CLIP) / config.DELTA_CLIP))
        self._fill(I, "DAN_PPL1", self.dan_gain * t(np.clip(dan_pun, 0, config.DELTA_CLIP) / config.DELTA_CLIP))
        stim = np.stack([danger, np.clip(sweet, 0, 3), np.clip(bitter, 0, 3), wind, warm, cool,
                         np.clip(hunger, 0, 1), np.clip(dan_rew, 0, 3) - np.clip(dan_pun, 0, 3)], axis=1).astype(np.float32)
        self.last_keep = keep
        return I, drive, stim

    def state(self) -> dict:
        return {"standardizer": self.std.state(), "odor_sha256": self.odor.sha256, "G": self.G}

    def load_state(self, st: dict | None) -> None:
        if st and "standardizer" in st:
            self.std.load_state(st["standardizer"])
