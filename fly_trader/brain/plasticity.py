"""Dopamine-gated three-factor plasticity at KC→MBON synapses — the only learning in the system.

Rule (Bennett, Philips, Nowotny 2021 Nat Commun 12:2569, eq. for the MV model, with an eligibility
trace over beats and TD-style RPE computed in agent/reward.py):

    E   ← λ_e·E + K_t                       eligibility of each KC per slot (τ = TAU_E_BEATS)
    ΔW  = η · Σ_b E[:,b] ⊗ (s ⊙ d̂_b)·learn_b   s = +1 approach MBON, −1 avoid MBON
    ΔW[ΔW > 0] *= β_pot                     depression-dominant (Hige et al. 2015 Neuron)
    ΔW  *= M_KM                             only synapses present in the connectome
    ‖ΔW‖_F ≤ DW_FROB_MAX                    per-beat cap
    W   ← clip(W + ΔW + κ·(W0 − W), 0, w_max)   slow recovery toward the connectome prior (unsourced)

Reward DANs (d+) firing while a KC is eligible depresses that KC's synapses onto AVOID MBONs
(s=−1 → ΔW<0) and, damped by β_pot, potentiates onto APPROACH MBONs; punishment does the reverse.
One shared W across all slots: the 128 columns are one brain smelling different odors.

Persistence: every beat writes a synapse_updates row; the exact inputs of the update (sparse KC code
per slot, d̂, learn mask, η) are journaled hourly so the whole W trajectory is replayable from the
last snapshot. Snapshots of W every SNAPSHOT_EVERY_BEATS beats.
"""
from __future__ import annotations

import hashlib
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from .. import config

log = logging.getLogger(__name__)


@dataclass
class UpdateStats:
    eta: float
    n_slots: int
    sum_abs: float
    max_abs: float
    frob: float
    frob_capped: bool
    n_pos: int
    n_neg: int
    n_clipped: int
    w_mean: float
    w_min: float
    w_max: float


class Journal:
    """Hourly compressed chunks of (beat_id, kc indices, kc rates, d_hat, learn, eta)."""

    def __init__(self, root: Path | None = None):
        self.root = root or (config.BRAIN_DIR / "journal")
        self.root.mkdir(parents=True, exist_ok=True)
        self.chunk_key: str | None = None
        self.buf: list[dict] = []
        self.path: Path | None = None

    def _key(self, ts: datetime) -> str:
        return ts.strftime("%Y-%m-%d/%H")

    def append(self, beat_id: int, ts: datetime, kc_idx: np.ndarray, kc_slot: np.ndarray, kc_rate: np.ndarray,
               d_hat: np.ndarray, learn: np.ndarray, eta: float) -> Path:
        key = self._key(ts)
        if self.chunk_key is not None and key != self.chunk_key:
            self.flush()
        self.chunk_key = key
        self.path = self.root / f"{key.replace('/', '_')}_w{int(d_hat.shape[0])}.npz"   # width-tagged: a batch change opens a new chunk
        self.buf.append({"beat_id": beat_id, "kc_idx": kc_idx.astype(np.uint16), "kc_slot": kc_slot.astype(np.uint8),
                         "kc_rate": kc_rate.astype(np.float16), "d_hat": d_hat.astype(np.float16),
                         "learn": learn.astype(np.bool_), "eta": np.float32(eta)})
        if len(self.buf) >= 240:  # ~6 minutes at 1.5 s beats; keeps memory and loss-on-crash small
            self.flush(keep_key=True)
        return self.path

    def flush(self, keep_key: bool = False) -> None:
        if not self.buf or self.path is None:
            return
        try:
            self._flush()
        except Exception as e:  # the journal must never take the beat down; keep the data in a side file
            fallback = self.path.with_name(self.path.stem + f"_recover_{int(time.time())}.npz")
            try:
                np.savez_compressed(fallback, **{f"beat_{b['beat_id']}_{k}": v for b in self.buf for k, v in b.items()})
            except Exception:
                pass
            log.error("journal flush failed (%s); buffer saved to %s", type(e).__name__, fallback)
            self.buf = []
            if not keep_key:
                self.chunk_key = None

    def _flush(self, keep_key: bool = False) -> None:
        n = len(self.buf)
        parts = {}
        if self.path.exists():
            old = np.load(self.path, allow_pickle=False)
            parts = {k: old[k] for k in old.files}
        offs = parts.get("offsets", np.zeros(0, dtype=np.int64))
        kc_idx = parts.get("kc_idx", np.zeros(0, dtype=np.uint16))
        kc_slot = parts.get("kc_slot", np.zeros(0, dtype=np.uint8))
        kc_rate = parts.get("kc_rate", np.zeros(0, dtype=np.float16))
        beat_ids = parts.get("beat_ids", np.zeros(0, dtype=np.int64))
        width = int(self.buf[0]["d_hat"].shape[0])
        d_hat = parts.get("d_hat", np.zeros((0, width), dtype=np.float16))
        learn = parts.get("learn", np.zeros((0, width), dtype=np.bool_))
        etas = parts.get("etas", np.zeros(0, dtype=np.float32))
        base = int(offs[-1]) if offs.size else 0
        new_offs = []
        for b in self.buf:
            base += len(b["kc_idx"])
            new_offs.append(base)
        np.savez_compressed(
            self.path,
            offsets=np.concatenate([offs, np.array(new_offs, dtype=np.int64)]),
            kc_idx=np.concatenate([kc_idx] + [b["kc_idx"] for b in self.buf]),
            kc_slot=np.concatenate([kc_slot] + [b["kc_slot"] for b in self.buf]),
            kc_rate=np.concatenate([kc_rate] + [b["kc_rate"] for b in self.buf]),
            beat_ids=np.concatenate([beat_ids, np.array([b["beat_id"] for b in self.buf], dtype=np.int64)]),
            d_hat=np.concatenate([d_hat] + [b["d_hat"][None, :] for b in self.buf]),
            learn=np.concatenate([learn] + [b["learn"][None, :] for b in self.buf]),
            etas=np.concatenate([etas, np.array([b["eta"] for b in self.buf], dtype=np.float32)]),
        )
        self.buf = []
        if not keep_key:
            self.chunk_key = None
        log.debug("journal flushed %d beats to %s", n, self.path)


class Plasticity:
    def __init__(self, W0: torch.Tensor, M: torch.Tensor, s_sign: torch.Tensor, batch: int,
                 eta: float | None = None, journal: Journal | None = None):
        self.dev = W0.device
        self.W0 = W0.clone().float()
        self.W = W0.clone().float()
        self.M = M.to(self.dev).float() if config.PLASTIC_MASK == "connectome" else torch.ones_like(self.W)
        self.s = s_sign.to(self.dev).float()  # [n_MBON]
        self.B = batch
        self.E = torch.zeros((W0.shape[0], batch), device=self.dev)
        self.eta = min(eta if eta is not None else config.ETA, config.ETA_MAX)
        self.lam_e = math.exp(-1.0 / max(config.TAU_E_BEATS, 1e-6))
        self.kappa = config.BEAT_S / (config.KAPPA_HOURS * 3600.0)
        self.w_max = float(config.W_MAX_MULT * W0.max().item()) if W0.numel() else 1.0
        self.frob_max = float(config.DW_FROB_FRAC * torch.linalg.norm(W0).item())
        self.journal = journal or Journal()
        self.last_kc: np.ndarray | None = None

    def reset_columns(self, cols: list[int]) -> None:
        if cols:
            self.E[:, cols] = 0.0

    def step(self, d_hat: torch.Tensor, learn: torch.Tensor) -> UpdateStats:
        """Apply one beat of plasticity using the current eligibility. d_hat, learn: [B] on device."""
        gate = (d_hat * learn.float())  # [B]
        n_slots = int(learn.sum().item())
        if n_slots == 0 or float(gate.abs().sum().item()) == 0.0:
            return UpdateStats(self.eta, 0, 0.0, 0.0, 0.0, False, 0, 0, 0,
                               float(self.W.mean()), float(self.W.min()), float(self.W.max()))
        dW = self.eta * (self.E @ (gate[:, None] * self.s[None, :]))  # [n_KC, n_MBON]
        dW = torch.where(dW > 0, dW * config.BETA_POT, dW) * self.M
        frob = float(torch.linalg.norm(dW).item())
        capped = False
        if frob > self.frob_max > 0:
            dW = dW * (self.frob_max / frob)
            capped = True
        W_new = self.W + dW + self.kappa * (self.W0 - self.W)
        clipped = ((W_new < 0) | (W_new > self.w_max)) & (self.M > 0)
        self.W = W_new.clamp_(0.0, self.w_max)
        return UpdateStats(self.eta, n_slots, float(dW.abs().sum()), float(dW.abs().max()), min(frob, self.frob_max),
                           capped, int((dW > 0).sum()), int((dW < 0).sum()), int(clipped.sum()),
                           float(self.W.mean()), float(self.W.min()), float(self.W.max()))

    def update_eligibility(self, kc_rates: torch.Tensor) -> None:
        self.E = self.lam_e * self.E + kc_rates

    def journal_beat(self, beat_id: int, ts: datetime, kc_rates: torch.Tensor, d_hat: np.ndarray,
                     learn: np.ndarray) -> tuple[str, str]:
        k = kc_rates.detach().cpu().numpy()
        idx, slot = np.nonzero(k)
        path = self.journal.append(beat_id, ts, idx, slot, k[idx, slot], d_hat, learn, self.eta)
        return str(path), ""

    def state_dict(self) -> dict:
        return {"W_KM": self.W.detach().cpu().numpy(), "E": self.E.detach().cpu().numpy(), "eta": self.eta}

    def load_W(self, W: np.ndarray) -> None:
        self.W = torch.tensor(W, device=self.dev, dtype=torch.float32).clamp_(0.0, self.w_max)
