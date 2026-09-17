"""Mushroom-body plasticity: after its bootstrap the fly keeps learning from the market, at the KC→MBON synapses only.

Rule (dopamine-gated three-factor; Hige 2015, Cohn 2015, Bennett 2021; masked like the connectome, FlyModel-style
associative updates, Shen/Dasgupta/Navlakha 2023). A plastic change ``D`` [n_KC, n_MBON] of the KC→MBON weights acts
at the network's last step (``FlyNet.readout``). For a scored minute with KC code ``k`` (presynaptic activity), MBON
activity ``m`` and, 120 minutes later, its realized net return ``r``:

    δ   = clip(clip(r, ±1) − ŷ, ±DELTA_CLIP)                  dopamine: the prediction error (ŷ recomputed with the current D)
    g_j = c_j γ_j (1 − m_j²) / (σ_j · scale)                   how MBON j moves the prediction (∂ŷ/∂D_ij = k_i g_j, exact)
    ΔD  = (α/ν) · M ⊙ Σ_rows w δ k gᵀ                          w = TOP_WEIGHT on minutes scored at/above the line, else 1

which is gradient descent on the weighted squared prediction error: reward (δ > 0) depresses KC→avoid-MBON synapses
(c < 0) and potentiates KC→approach ones; punishment the reverse. ``ν`` (the median squared norm of one row's update,
fixed at bootstrap) makes ``α`` the share of a row's error one update corrects. Changes decay toward the bootstrap weights
with half-life ``τ½`` in event time (dopamine-gated forgetting, Berry 2012), each update is capped at ``STEP_CAP`` of the
bootstrap weights' norm, and ``D`` stays within [−w0, 2·w0] (effective weights within [0, 3·w0]). ``β_pot`` < 1 damps
potentiation (depression-dominant; still a descent direction). ``D = 0`` is the frozen fly (the shadow).

The label arrives two hours after the decision, far beyond any biological eligibility window (seconds: Handler 2019),
so each scored minute leaves a synaptic tag (``PendingTags``: its KC code, MBON pre-activation and descending output)
that is captured when the label resolves (synaptic tag-and-capture, Frey & Morris 1997).

``PlasticBank`` holds ``C`` configurations side by side (the replay's grid; 1 live).
"""
from __future__ import annotations

import bisect
import itertools
import math
from dataclasses import dataclass, field

import numpy as np
import torch

DELTA_CLIP = 0.5
RET_CLIP = 1.0
TOP_WEIGHT = 10.0
STEP_CAP = 1e-3              # ‖ΔD‖_F per update ≤ STEP_CAP · ‖w0‖_F
DAY_S = 86400.0


@dataclass
class Tags:
    """A batch of captured tags: row keys, the parts cached at scoring and each configuration's weight [C, B]."""
    keys: list
    ts: np.ndarray
    y_dn: torch.Tensor      # [B]
    u0: torch.Tensor        # [B, n_MBON]
    k: torch.Tensor         # [B, n_KC] dense KC code
    w: torch.Tensor         # [C, B]
    s: np.ndarray = None    # [B] strategy (dopamine channel) of each tag; None = channel 0

    def __post_init__(self):
        if self.s is None:
            self.s = np.zeros(len(self.keys), dtype=np.int64)

    def __len__(self) -> int:
        return len(self.keys)

    def subset(self, keep: np.ndarray) -> "Tags":
        keep = np.asarray(keep, dtype=bool); i = torch.as_tensor(np.flatnonzero(keep), device=self.y_dn.device)
        return Tags([k for k, m in zip(self.keys, keep) if m], self.ts[keep], self.y_dn[i], self.u0[i], self.k[i], self.w[:, i], self.s[keep])


class PendingTags:
    """Scored minutes waiting for their labels, kept in due order (strategies hold for different times). The KC code is
    stored sparsely (its active cells)."""

    def __init__(self, n_kc: int, k_active: int):
        self.n_kc, self.k_active = n_kc, k_active
        self.q: list = []; self.n = 0; self._seq = itertools.count()

    def __len__(self) -> int:
        return self.n

    def push(self, keys: list, ts: np.ndarray, due: float, y_dn: torch.Tensor, u0: torch.Tensor, k: torch.Tensor, w: torch.Tensor, s=None) -> None:
        """Tags capturable from event time ``due`` on (all rows of one push share it); ``s``: their strategy channel(s)."""
        if len(keys) == 0:
            return
        vals, idx = k.detach().topk(min(self.k_active, k.shape[1]), dim=1)
        ch = np.broadcast_to(np.asarray(0 if s is None else s, dtype=np.int64), (len(keys),)).copy()
        chunk = {"due": float(due), "keys": list(keys), "ts": np.asarray(ts, dtype=np.float64), "y_dn": y_dn.detach().float().cpu(), "s": ch,
                 "u0": u0.detach().float().cpu(), "idx": idx.to(torch.int32).cpu(), "vals": vals.float().cpu(), "w": w.detach().float().cpu()}
        bisect.insort(self.q, (float(due), next(self._seq), chunk))
        self.n += len(keys)

    def pop_due(self, now: float, device=None) -> Tags | None:
        out = []
        while self.q and self.q[0][0] <= now:
            out.append(self.q.pop(0)[2])
        if not out:
            return None
        self.n -= sum(len(c["keys"]) for c in out)
        return self._tags(out, device)

    def clear(self) -> None:
        self.q.clear(); self.n = 0

    def _tags(self, chunks: list[dict], device=None) -> Tags:
        idx = torch.cat([c["idx"] for c in chunks]).long(); vals = torch.cat([c["vals"] for c in chunks])
        k = torch.zeros(len(idx), self.n_kc).scatter_(1, idx, vals)
        dev = device or "cpu"
        return Tags(keys=[key for c in chunks for key in c["keys"]], ts=np.concatenate([c["ts"] for c in chunks]),
                    y_dn=torch.cat([c["y_dn"] for c in chunks]).to(dev), u0=torch.cat([c["u0"] for c in chunks]).to(dev), k=k.to(dev),
                    w=torch.cat([c["w"] for c in chunks], dim=1).to(dev), s=np.concatenate([c["s"] for c in chunks]))


def assign_channels(comp: np.ndarray, km_mass: np.ndarray, c_sign: np.ndarray, n_strategies: int) -> tuple[np.ndarray, np.ndarray]:
    """Each strategy's own dopamine channel: its mushroom-body compartments. ``comp`` [J]: each MBON's compartment (−1: no
    DAN), ``km_mass`` [J]: KC→MBON synapse mass per MBON, ``c_sign`` [J]: readout sign (+1 approach, −1 avoid, 0 other).
    Compartments are split by their MBONs' majority valence and, within each valence, dealt in descending synapse mass to
    the strategy with the least mass so far (deterministic, balanced, every strategy gets both valences when there are
    enough). Returns ``learn`` [S, J] (the strategy's own MBONs: its outcomes change only their KC synapses) and ``read``
    [S, J] (its own + the MBONs no compartment claims, which stay fixed). One strategy: everything, as before; no DAN
    data: every MBON is its own compartment."""
    J = len(c_sign); S = int(n_strategies)
    if S <= 1:
        return np.ones((1, J), bool), np.ones((1, J), bool)
    comp = np.asarray(comp, int) if comp is not None and len(comp) == J else np.full(J, -1)
    if (comp < 0).all():
        comp = np.arange(J)
    learn = np.zeros((S, J), bool)
    ids = sorted(set(comp[comp >= 0].tolist()))
    val = {k: int(np.sign(np.sum(np.sign(c_sign[comp == k])))) for k in ids}
    mass = {k: float(np.asarray(km_mass)[comp == k].sum()) for k in ids}
    for v in (1, -1, 0):
        load = np.zeros(S)
        for k in sorted([k for k in ids if val[k] == v], key=lambda k: (-mass[k], k)):
            s_ = int(np.argmin(load)); learn[s_, comp == k] = True; load[s_] += mass[k]
    free = ~learn.any(0)
    return learn, learn | free[None, :]


def row_weights(scores: torch.Tensor, lines: torch.Tensor) -> torch.Tensor:
    """[C, B]: ``TOP_WEIGHT`` where a configuration scored the row at or above its own line, else 1."""
    return torch.where(scores >= lines[:, None], TOP_WEIGHT, 1.0)


class PlasticBank:
    """Plastic changes ``D`` [C, n_KC, n_MBON] of a FlyNet's KC→MBON weights under ``C`` configurations (α, τ½ in days;
    τ½ = inf: no decay). The net is read, never modified."""

    def __init__(self, net, configs: list[tuple[float, float]], scale: float, nu: float = 1.0, beta_pot: float = 1.0, step_cap: float = STEP_CAP,
                 learn: np.ndarray | None = None, read: np.ndarray | None = None):
        self.net, self.scale, self.nu, self.beta_pot, self.step_cap = net, float(scale), float(nu), float(beta_pot), float(step_cap)
        dev = net.dev
        with torch.no_grad():
            self.w0 = net.w_km().detach().clone()
            self.mask = net.mask.float()
            self.c = net.c.detach().clone(); self.gamma = net.gain[net.mb0:net.mb1].detach().clone()
            mu, sd = net.mbon_stats(); self.mu, self.sigma = mu.detach().clone(), sd.detach().clone()
        self.w0_norm = float(torch.linalg.norm(self.w0))
        self.configs = [(float(a), float(h)) for a, h in configs]
        C = len(self.configs)
        J = self.w0.shape[1]
        self.learn = torch.as_tensor(np.ones((1, J), bool) if learn is None else learn, device=dev)          # [S, J] per-strategy channels
        self.read = torch.as_tensor(np.ones((1, J), bool) if read is None else read, device=dev)
        self.alpha = torch.tensor([a for a, _ in self.configs], device=dev)[:, None].repeat(1, J)            # [C, J]: per-channel rates
        self.half_life_s = torch.tensor([h * DAY_S if math.isfinite(h) else float("inf") for _, h in self.configs], device=dev)[:, None].repeat(1, J)
        self.D = torch.zeros(C, *self.w0.shape, device=dev)
        self.t_last: float | None = None
        self.dev = dev

    @property
    def C(self) -> int:
        return len(self.configs)

    # ---- prediction ----
    def _read(self, s, B: int) -> torch.Tensor:
        """[B, J] the MBONs each row's strategy reads (all MBONs with one strategy)."""
        if self.read.shape[0] == 1 or s is None:
            return self.read[0].float()[None].expand(B, -1)
        return self.read[torch.as_tensor(np.asarray(s), device=self.dev)].float()

    def predict(self, y_dn: torch.Tensor, u0: torch.Tensor, k: torch.Tensor, D: torch.Tensor | None = None, s=None) -> tuple[torch.Tensor, torch.Tensor]:
        """(scores [C, B] in net-return units, MBON activity [C, B, n_MBON]) under each configuration's ``D``; ``s``: each
        row's strategy (its head output is ``y_dn``; it reads its own compartments and the unclaimed ones)."""
        D = self.D if D is None else D
        u = u0[None] + self.gamma * torch.einsum("bi,cij->cbj", k, D * self.mask)
        m = torch.tanh(u)
        return (y_dn[None] + (((m - self.mu) / self.sigma) * self._read(s, len(y_dn))) @ self.c) / self.scale, m

    def frozen(self, y_dn: torch.Tensor, u0: torch.Tensor, s=None) -> torch.Tensor:
        """The shadow: the bootstrap fly's score (D = 0) [B]."""
        return (y_dn + (((torch.tanh(u0) - self.mu) / self.sigma) * self._read(s, len(y_dn))) @ self.c) / self.scale

    def grad_factor(self, m: torch.Tensor, s=None) -> torch.Tensor:
        """g [C, B, n_MBON] with ∂ŷ/∂D_ij = k_i g_j (zero on MBONs the row's strategy does not read)."""
        return self.c * self.gamma * (1.0 - m * m) / (self.sigma * self.scale) * self._read(s, m.shape[1])

    # ---- learning ----
    def decay_to(self, t: float) -> None:
        """Decay toward the bootstrap weights up to event time ``t`` (seconds)."""
        if self.t_last is not None and t > self.t_last:
            f = torch.pow(2.0, -(t - self.t_last) / self.half_life_s)            # [C, J]; inf half-life → 1
            self.D *= f[:, None, :]
        self.t_last = t if self.t_last is None else max(self.t_last, t)

    def estimate_nu(self, y_dn: torch.Tensor, u0: torch.Tensor, k: torch.Tensor) -> float:
        """Median squared norm of one row's update direction at D = 0 (sets the scale ``α`` is measured in)."""
        m = torch.tanh(u0)
        g = self.c * self.gamma * (1.0 - m * m) / (self.sigma * self.scale)          # [B, J]
        sq = ((k * k) @ self.mask * g * g).sum(1)
        v = float(sq.median()) if len(sq) else 0.0
        self.nu = v if v > 0 else 1.0
        return self.nu

    def update(self, tags: Tags, r: torch.Tensor, t: float) -> dict:
        """Capture ``tags`` with realized net returns ``r`` [B] at event time ``t``; returns per-configuration statistics."""
        self.decay_to(t)
        if len(tags) == 0:
            return {"n": 0}
        r = r.to(self.dev).float().clamp(-RET_CLIP, RET_CLIP)
        yhat, m = self.predict(tags.y_dn, tags.u0, tags.k, s=tags.s)
        delta = (r[None] - yhat).clamp(-DELTA_CLIP, DELTA_CLIP)                     # [C, B]
        own = self.learn.float()[torch.as_tensor(tags.s, device=self.dev)] if self.learn.shape[0] > 1 else self.learn[0].float()[None]
        g = self.grad_factor(m, tags.s) * own                                       # [C, B, J]: only the row's own channel learns
        dD = torch.einsum("bi,cbj->cij", tags.k, (tags.w * delta)[:, :, None] * g) * self.mask
        if self.beta_pot != 1.0:
            dD = torch.where(dD > 0, dD * self.beta_pot, dD)
        dD = dD * (self.alpha / self.nu)[:, None, :]
        norms = torch.linalg.norm(dD.flatten(1), dim=1)
        cap = self.step_cap * self.w0_norm
        capped = norms > cap
        dD = dD * torch.where(capped, cap / norms.clamp(min=1e-30), torch.ones_like(norms))[:, None, None]
        D = self.D + dD
        self.D = torch.maximum(torch.minimum(D, 2.0 * self.w0), -self.w0) * self.mask
        return {"n": len(tags), "mean_delta": delta.mean(1).tolist(), "mean_abs_delta": delta.abs().mean(1).tolist(),
                "step": torch.linalg.norm(dD.flatten(1), dim=1).tolist(), "capped": capped.tolist(), "drift": self.drift().tolist()}

    def drift(self, s: int | None = None) -> torch.Tensor:
        """‖D‖_F / ‖w0‖_F per configuration (``s``: over that strategy's own channel only)."""
        if s is None:
            return torch.linalg.norm(self.D.flatten(1), dim=1) / max(self.w0_norm, 1e-30)
        cols = self.learn[s]
        return torch.linalg.norm(self.D[:, :, cols].flatten(1), dim=1) / max(float(torch.linalg.norm(self.w0[:, cols])), 1e-30)

    # ---- state ----
    def state(self) -> dict:
        m = self.mask.bool()
        return {"D": self.D[:, m].cpu(), "t_last": self.t_last, "configs": self.configs, "nu": self.nu, "beta_pot": self.beta_pot, "step_cap": self.step_cap,
                "alpha": self.alpha.cpu(), "half_life_s": self.half_life_s.cpu(), "learn": self.learn.cpu(), "read": self.read.cpu()}

    def load_state(self, s: dict) -> None:
        if [tuple(c) for c in s["configs"]] != self.configs:
            raise ValueError(f"plastic state for configurations {s['configs']} does not match {self.configs}")
        D = torch.zeros_like(self.D); D[:, self.mask.bool()] = s["D"].to(self.dev).float()
        self.D, self.t_last, self.nu = D, s["t_last"], float(s["nu"])
        if "alpha" in s:
            self.alpha, self.half_life_s = s["alpha"].to(self.dev), s["half_life_s"].to(self.dev)
            self.learn, self.read = s["learn"].to(self.dev), s["read"].to(self.dev)
        self.beta_pot, self.step_cap = float(s.get("beta_pot", self.beta_pot)), float(s.get("step_cap", self.step_cap))

    def select(self, i: int) -> "PlasticBank":
        """A single-configuration bank holding configuration ``i``'s current ``D`` (the replay's winner, going live)."""
        b = PlasticBank(self.net, [self.configs[i]], self.scale, self.nu, self.beta_pot, self.step_cap, self.learn.cpu().numpy(), self.read.cpu().numpy())
        b.D = self.D[i:i + 1].clone(); b.t_last = self.t_last
        return b

    def select_per_strategy(self, best: list[int]) -> "PlasticBank":
        """One bank with each strategy's own winning configuration: strategy s's channel columns (D, α, half-life) come from
        configuration ``best[s]``. Exact, because a strategy's value depends only on its own channel."""
        b = self.select(best[0])
        for s_, i in enumerate(best):
            cols = self.learn[s_]
            b.D[0][:, cols] = self.D[i][:, cols]; b.alpha[0, cols] = self.alpha[i, cols]; b.half_life_s[0, cols] = self.half_life_s[i, cols]
        b.configs = [("per-strategy", tuple(best))]
        return b
