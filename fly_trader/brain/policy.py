"""ConnectomePolicy — a FlyGM-style trainable policy whose recurrent core IS the FlyWire connectome.

Following Whole-Brain Connectomic Graph Model (FlyGM, arXiv 2602.17997) and ConnecTorch: every neuron is a
rate unit, messages flow only along real synapses, synaptic weights are initialised from signed synapse
counts with topology and signs frozen (Dale's law), and gradient descent trains the weights plus a learned
sensory encoder (gated into afferent neurons) and a learned motor decoder (read from descending neurons).
Simplification vs FlyGM: scalar state per neuron with per-neuron gain/bias instead of a C-dim state + MLP.

Dynamics per message-passing step (K steps per environment beat):
    h ← tanh( leak·h + g ⊙ (W h) + b + I_in )        W_e = sign_e · softplus(θ_e),  θ_e init from counts·scale
Inputs: x̃ = Enc(obs) ∈ R^{d_enc} → I_in[afferent v] = W_in[v]·x̃ (zero elsewhere).
Outputs: a = Dec(h[descending]) → (mean of the pre-sigmoid action logit, value estimate); action = σ(u).
Propagation uses index_select/index_add (O(edges) memory, differentiable on MPS; 255 ms for 8 steps of
forward+backward at batch 64 on an M4 Max).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import config

AFFERENT_POPS = ("ORN_FOOD", "ORN_DANGER", "GRN_SWEET", "GRN_BITTER", "GRN_OTHER", "MECH_JO", "MECH_BRISTLE",
                 "MECH_OTHER", "THERMO_WARM", "THERMO_COOL", "THERMO_OTHER")
EFFERENT_POP = "DESCENDING"


@dataclass
class PolicyOut:
    mu: torch.Tensor        # [B] pre-sigmoid action mean
    log_std: torch.Tensor   # [B] (broadcast of a learned scalar)
    value: torch.Tensor     # [B]
    h: torch.Tensor         # [N, B] new hidden state


class SubConnectome:
    """A connectome restricted to a set of populations (e.g. everything but the optic lobes)."""

    def __init__(self, c, exclude: tuple[str, ...] = ("VISUAL",)):
        keep = torch.ones(c.N, dtype=torch.bool)
        for name in exclude:
            if name in c.pop_ranges:
                lo, hi = c.pop_ranges[name]
                keep[lo:hi] = False
        new_index = torch.cumsum(keep.to(torch.int64), 0) - 1
        post, pre = c.indices[0], c.indices[1]
        m = keep[post] & keep[pre]
        self.indices = torch.stack([new_index[post[m]], new_index[pre[m]]])
        self.values_raw = c.values_raw[m].clone()
        self.N = int(keep.sum())
        self.pop_ranges = {}
        for name, (lo, hi) in c.pop_ranges.items():
            if name in exclude or hi <= lo:
                continue
            self.pop_ranges[name] = (int(new_index[lo]), int(new_index[hi - 1]) + 1)
        self.device = c.device
        self.s = getattr(c, "s", None)
        self.spectral_radius_raw = getattr(c, "spectral_radius_raw", None)
        self.content_sha256 = getattr(c, "content_sha256", None)
        self.excluded = exclude
        self.keep = keep


class ConnectomePolicy(nn.Module):
    def __init__(self, connectome, obs_dim: int, d_enc: int = 16, k_steps: int = 4, leak: float = 0.5,
                 weight_scale: float | None = None, device=None):
        super().__init__()
        c = connectome
        self.N = c.N
        self.k_steps = k_steps
        self.leak = leak
        dev = torch.device(device) if device else c.device
        self.dev = dev
        idx = c.indices
        self.register_buffer("pre", idx[1].to(dev))
        self.register_buffer("post", idx[0].to(dev))
        vals = c.values_raw.float()
        self.register_buffer("sign", torch.sign(vals).to(dev))
        s = weight_scale if weight_scale is not None else float(getattr(c, "s", 0.0) or 0.0)
        if s <= 0:
            s = 0.99 / float(getattr(c, "spectral_radius_raw", 2788.0))
        mag = (vals.abs() * s).clamp(min=1e-6)
        self.theta = nn.Parameter(torch.log(torch.expm1(mag)).to(dev))          # softplus^-1(|w|)
        self.gain = nn.Parameter(torch.ones(self.N, device=dev))
        self.bias = nn.Parameter(torch.zeros(self.N, device=dev))
        # afferent rows
        rows = []
        for p in AFFERENT_POPS:
            if p in c.pop_ranges:
                lo, hi = c.pop_ranges[p]
                rows.append(torch.arange(lo, hi))
        self.register_buffer("aff_rows", torch.cat(rows).to(dev))
        lo, hi = c.pop_ranges[EFFERENT_POP]
        self.register_buffer("eff_rows", torch.arange(lo, hi).to(dev))
        self.enc = nn.Sequential(nn.Linear(obs_dim, 64), nn.Tanh(), nn.Linear(64, d_enc), nn.Tanh()).to(dev)
        self.w_in = nn.Parameter(torch.randn(len(self.aff_rows), d_enc, device=dev) * (1.0 / math.sqrt(d_enc)))
        n_eff = len(self.eff_rows)
        self.dec = nn.Sequential(nn.Linear(n_eff, 64), nn.Tanh()).to(dev)
        self.pi_head = nn.Linear(64, 1).to(dev)
        self.v_head = nn.Sequential(nn.Linear(64 + obs_dim, 64), nn.Tanh(), nn.Linear(64, 1)).to(dev)
        self.log_std = nn.Parameter(torch.tensor(math.log(0.25), device=dev))
        nn.init.constant_(self.pi_head.bias, -3.0)     # start flat (σ(-3) ≈ 0.05 target exposure): learn WHEN to trade, not when to stop
        self.obs_dim = obs_dim
        self.d_enc = d_enc

    # ---- graph ----
    def weights(self) -> torch.Tensor:
        return self.sign * Fn.softplus(self.theta)

    def propagate(self, h: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        msg = h.index_select(0, self.pre) * w[:, None]                       # [E, B]
        return torch.zeros_like(h).index_add(0, self.post, msg)             # [N, B]

    def init_hidden(self, batch: int) -> torch.Tensor:
        return torch.zeros(self.N, batch, device=self.dev)

    def forward(self, obs: torch.Tensor, h: torch.Tensor, k_steps: int | None = None) -> PolicyOut:
        """obs [B, obs_dim]; h [N, B] → PolicyOut (state advanced k_steps message-passing steps)."""
        B = obs.shape[0]
        x = self.enc(obs)                                                    # [B, d_enc]
        I_aff = self.w_in @ x.T                                              # [n_aff, B]
        I_in = torch.zeros(self.N, B, device=self.dev).index_copy(0, self.aff_rows, I_aff)
        w = self.weights()
        for _ in range(k_steps or self.k_steps):
            h = torch.tanh(self.leak * h + self.gain[:, None] * self.propagate(h, w) + self.bias[:, None] + I_in)
        eff = h.index_select(0, self.eff_rows).T                             # [B, n_eff]
        z = self.dec(eff)
        mu = self.pi_head(z).squeeze(-1)
        value = self.v_head(torch.cat([z, obs], dim=1)).squeeze(-1)
        return PolicyOut(mu=mu, log_std=self.log_std.expand_as(mu), value=value, h=h)

    # ---- action distribution helpers (Gaussian on the logit, squashed by sigmoid) ----
    @staticmethod
    def sample(mu: torch.Tensor, log_std: torch.Tensor, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        std = log_std.exp()
        u = mu if deterministic else mu + std * torch.randn_like(mu)
        logp = -0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)
        return u, logp

    @staticmethod
    def log_prob(mu: torch.Tensor, log_std: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        std = log_std.exp()
        return -0.5 * ((u - mu) / std) ** 2 - log_std - 0.5 * math.log(2 * math.pi)

    @staticmethod
    def entropy(log_std: torch.Tensor) -> torch.Tensor:
        return 0.5 + 0.5 * math.log(2 * math.pi) + log_std

    @staticmethod
    def to_action(u: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(u)

    def load_compatible(self, state_dict: dict) -> None:
        self.load_state_dict(state_dict)

    def param_groups(self, lr_graph: float, lr_heads: float) -> list[dict]:
        graph = [self.theta, self.gain, self.bias]
        heads = [p for n, p in self.named_parameters() if n not in ("theta", "gain", "bias")]
        return [{"params": graph, "lr": lr_graph}, {"params": heads, "lr": lr_heads}]

    def describe(self) -> dict:
        return {"N": self.N, "edges": int(self.theta.numel()), "afferent": int(len(self.aff_rows)), "efferent": int(len(self.eff_rows)),
                "obs_dim": self.obs_dim, "d_enc": self.d_enc, "k_steps": self.k_steps, "leak": self.leak,
                "params": int(sum(p.numel() for p in self.parameters()))}
