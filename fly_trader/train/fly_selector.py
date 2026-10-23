"""The fly: the FlyWire central brain, bootstrapped once by imitating the selector, then left to learn from the market
through its mushroom body (``brain/plastic.py``).

Network (``FlyNet``, flynet-2-mb): the selector's robust-scaled features (``train/scaling.py``) are injected into the
5,460 sensory (afferent) neurons through a learned projection; activity propagates ``K_STEPS`` steps along the real,
signed synapses (leaky tanh rate units, magnitudes initialised from synapse counts, signs and topology fixed, scale.json's
pathway gains applied) and along the mushroom body's KC→MBON synapses, which the connectome build keeps apart from the
sparse graph (the pairs that exist in FlyWire; magnitudes learnable, starting at ``KM_INIT_SCALE`` per synapse so the
MBONs start unsaturated). Kenyon cells keep only their ``KC_ACTIVE`` most active cells at every step (APL-like
winner-take-all; Dasgupta 2017, MacKenzie 2025). Prediction of the net return over the hold (× ``SCALE``) = the
decoder on the normalised descending neurons + a signed linear readout of the normalised MBONs (approach +, avoid −,
the others learned): the path through which plasticity at KC→MBON reaches the output.

``forward_parts`` gives what the plasticity rule needs — the descending-neuron output, the MBONs' pre-activation at the
last step and the KC code that drove it; ``readout`` completes the prediction with a plastic change ``D`` of the KC→MBON
weights at that last step (``D`` = None is the frozen fly). The descending neurons at the last step are computed from
the previous step, so ``D`` reaches the output only through the MBON readout.

Bootstrap (``bootstrap``, the one time the selector teaches): the teacher is the selector the live book trades
(``deployed_teacher``) as refit without its holdout days, so the calibration week is one it never saw; only when none is
deployed, or it has no such fit, is a stack refit on the days before ``S − 8`` (one purge day before the calibration
week), which is also what the replay uses, since there the refit is the honest out-of-sample teacher.
The fly imitates it on the tradable minutes of those days (the teacher's top ``DISTIL_TOP`` plus a random
``DISTIL_SAMPLE``; squared error weighted ``TOP_WEIGHT`` × on the teacher's top 1 %; passes stop when a held-out
slice stops improving by more than its own standard error, so the epoch count comes from the data), both
normalisations are recomputed over ``RECAL_ROWS`` training rows and frozen, and the fly's own buy line and sizing bands
are calibrated (``train/fly_calibrate.py``) on its scores for the seven days before ``S`` whose labels were known by
then — never the selector's line. Reports: the MBONs' share of the score variance, MBON saturation, the slope and rank
correlation of the fly's scores against the teacher's. Snapshots: ``data/brain/policies/fly_<ts>.pt`` + ``brain_snapshots``
kind 'fly_selector'.

Matched mode (opt-in, 2026-10-23; train/matched_pair.py runs it): the teacher is a given stack -- the selector's
holdout-blind fit, to be scored on the holdout, or its full fit, to be deployed -- and the fly learns on exactly the rows
that stack was fit on (``matched_teacher``, ``bootstrap_matched``): no calibration week and no purge day of its own. Its
lines are matched to the teacher's own trade count on those rows, which needs no label, so the holdout stays unseen by
both. Its evidence is the out-of-sample report on the holdout (``oos_report``), then the live race; ``deployable``
reads that report instead of a calibration week. The default bootstrap above is unchanged.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import config
from ..agent import sizing
from ..brain import device as brain_device
from ..brain.connectome import AFFERENT_POPS, EFFERENT_POP, Connectome, SubConnectome
from ..brain.plastic import assign_channels
from ..db.apilog import record_event
from ..db.connection import transaction
from . import fly_calibrate, model_io
from . import progress as prog
from . import selector
from .decisions import HOLD_MIN, DecisionSet, build, day_ords, rank_corr, taken_idx
from .scaling import RobustScaler

log = logging.getLogger(__name__)

SCALE = 20.0                 # the head predicts the net return × 20 (±5 % ≈ ±1)
K_STEPS, LEAK, HIDDEN = 4, 0.5, 128
KC_ACTIVE = 0.10             # share of Kenyon cells left active at every step
KM_INIT_SCALE = 0.006        # KC→MBON weight per synapse: ~22 active KC inputs × ~8 synapses ≈ 1, MBONs unsaturated
C_INIT = 0.1                 # MBON readout coefficient at start (SCALE units per standard deviation of an MBON)
EPOCHS, BATCH = 8, 512       # EPOCHS only caps the runtime; early stopping below picks the count the data supports
TOP_WEIGHT = 10.0            # loss weight of the teacher's top 1 % of training minutes
DISTIL_TOP, DISTIL_SAMPLE = 0.05, 3_000_000
RECAL_ROWS = 200_000
CHECKPOINT = config.BRAIN_DIR / "policies" / "fly_bootstrap.ckpt.pt"     # the live bootstrap's epochs so far (main): a stopped run resumes
VAL_ROWS, VAL_MIN_ROWS = 200_000, 100_000   # distillation rows held out to judge the fit; a smaller slice is too noisy to stop on
MIN_EPOCHS = 2               # never stop on one epoch: it has nothing to be compared against
LR_GRAPH, LR_HEAD = 1e-4, 1e-3
PURGE_DAYS, CALIB_DAYS = 1, 7
DIAG_ROWS = 50_000
GATES = {"mbon_share_min": 0.10, "mbon_saturated_max": 0.20, "slope_min": 0.6, "slope_max": 1.4}
TEACHER_NOTE = "a stack refit on the fly's training days (no deployed selector to copy)"
# the definitions a fly was trained on (the selector's, as its teacher, + the fly's network and plasticity design)
# "calib": how its buy line is picked (config.FLY_CALIB_RULE): a verdict or fly calibrated under another rule is not this one
FLY_VERSION = {**selector.DATA_VERSION, "fly": "flynet-3-ch", "plastic": "mb4-ch", "teacher": "deployed-1", "calib": config.FLY_CALIB_RULE,
               "lines": "per-chain-1"}                    # its own line per (strategy, chain), judged per chain (train/fly_replay.Slots)


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x.clamp(min=1e-6)))


class FlyNet(nn.Module):
    """Scorer on the connectome: features → afferent neurons → K steps along the synapses and the KC→MBON block →
    normalised descending neurons → decoder, + signed readout of the normalised MBONs → predicted net return (× ``SCALE``)."""

    def __init__(self, graph, obs_dim: int, k_steps: int = K_STEPS, leak: float = LEAK, hidden: int = HIDDEN, kc_active: float = KC_ACTIVE, device=None,
                 n_strategies: int = 1, aff_rows=None, scale: float = SCALE):
        """``aff_rows``: the neurons the features enter (default: the memecoin fly's ``AFFERENT_POPS``; the Kalshi fly passes
        the photoreceptors). ``scale``: the unit of the head's output (net return × 20 for memecoins; probability × 1 for Kalshi)."""
        super().__init__()
        dev = torch.device(device) if device else graph.device
        self.N, self.k_steps, self.leak, self.obs_dim, self.hidden, self.kc_active = graph.N, k_steps, leak, obs_dim, hidden, kc_active
        self.scale = float(scale)
        self.register_buffer("pre", graph.indices[1].to(dev)); self.register_buffer("post", graph.indices[0].to(dev))
        vals = graph.values_raw.float(); self.register_buffer("sign", torch.sign(vals).to(dev))
        s = float(getattr(graph, "s", 0.0) or 0.0) or 0.99 / float(getattr(graph, "spectral_radius_raw", None) or 2788.0)
        self.theta = nn.Parameter(_inv_softplus(vals.abs() * s).to(dev))                          # softplus^-1(|w|)
        self.gain = nn.Parameter(torch.ones(self.N, device=dev)); self.bias = nn.Parameter(torch.zeros(self.N, device=dev))
        aff = torch.as_tensor(aff_rows, dtype=torch.long) if aff_rows is not None else torch.cat([torch.arange(*graph.pop_ranges[p]) for p in AFFERENT_POPS if p in graph.pop_ranges])
        self.register_buffer("aff_rows", aff.to(dev)); self.register_buffer("eff_rows", torch.arange(*graph.pop_ranges[EFFERENT_POP]).to(dev))
        # the mushroom body: KC→MBON pairs of the connectome (MBON columns: approach, avoid, other)
        (self.kc0, self.kc1), self.mb0, self.mb1 = graph.pop_ranges["KC"], graph.pop_ranges["MBON_APP"][0], graph.pop_ranges["MBON_OTHER"][1]
        self.n_kc, self.n_mbon = self.kc1 - self.kc0, self.mb1 - self.mb0
        self.k_active = max(1, int(round(kc_active * self.n_kc)))
        M = graph.M_KM.bool(); ki, mj = torch.nonzero(M, as_tuple=True)
        self.register_buffer("mask", M.to(dev)); self.register_buffer("km_kc", ki.to(dev)); self.register_buffer("km_mbon", mj.to(dev))
        self.theta_km = nn.Parameter(_inv_softplus(graph.W_KM0.float()[ki, mj] * KM_INIT_SCALE).to(dev))
        n_app = graph.pop_ranges["MBON_APP"][1] - graph.pop_ranges["MBON_APP"][0]; n_av = graph.pop_ranges["MBON_AV"][1] - graph.pop_ranges["MBON_AV"][0]
        c_sign = torch.zeros(self.n_mbon); c_sign[:n_app] = 1.0; c_sign[n_app:n_app + n_av] = -1.0
        self.register_buffer("c_sign", c_sign.to(dev))
        self.rho = nn.Parameter(torch.full((self.n_mbon,), float(_inv_softplus(torch.tensor(C_INIT))), device=dev))
        self.c_free = nn.Parameter(torch.zeros(self.n_mbon, device=dev))
        n_aff, n_eff = len(self.aff_rows), len(self.eff_rows)
        self.w_in = nn.Parameter(torch.randn(n_aff, obs_dim, device=dev) / obs_dim ** 0.5)       # every feature reaches every sensory neuron
        self.eff_norm = nn.BatchNorm1d(n_eff).to(dev)
        self.mb_norm = nn.BatchNorm1d(self.n_mbon, affine=False).to(dev)
        self.dec = nn.Sequential(nn.Linear(n_eff, hidden), nn.GELU(), nn.Linear(hidden, hidden), nn.GELU()).to(dev)
        # one head per strategy over the shared decoder; strategy s reads its own dopamine channel (mushroom-body
        # compartments, brain/plastic.assign_channels) and the compartments no strategy claims
        self.n_strategies = int(n_strategies)
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(self.n_strategies)]).to(dev)
        comp = getattr(graph, "mbon_compartment", None)
        learn, read = assign_channels(comp if comp is not None else None, graph.W_KM0.float().sum(0).numpy(), c_sign.numpy(), self.n_strategies)
        self.register_buffer("learn", torch.as_tensor(learn).to(dev)); self.register_buffer("read", torch.as_tensor(read).to(dev))
        self.dev = dev

    @property
    def head(self) -> nn.Linear:
        """The first strategy's head (the only one with one strategy)."""
        return self.heads[0]

    @property
    def c(self) -> torch.Tensor:
        """MBON readout coefficients: approach +, avoid −, the others free."""
        return torch.where(self.c_sign != 0, self.c_sign * Fn.softplus(self.rho), self.c_free)

    def batch_rows(self, ceiling: int = 2048, training: bool = False) -> int:
        """Rows one batch may hold on this device: the free memory over the graph's measured per-row cost
        (brain/device.rows_for), never above ``ceiling``."""
        return brain_device.rows_for(int(self.pre.numel()), self.N, self.dev, ceiling=ceiling, training=training)

    def w_km(self) -> torch.Tensor:
        """The KC→MBON weights as a dense [n_KC, n_MBON] matrix (0 where the connectome has no synapse)."""
        W = torch.zeros(self.n_kc, self.n_mbon, device=self.dev)
        W[self.km_kc, self.km_mbon] = Fn.softplus(self.theta_km)
        return W

    def _kwta(self, h: torch.Tensor) -> torch.Tensor:
        kc = torch.relu(h[self.kc0:self.kc1])
        thr = kc.topk(self.k_active, dim=0).values[-1:]
        kc = torch.where(kc >= thr, kc, torch.zeros_like(kc))
        return torch.cat([h[:self.kc0], kc, h[self.kc1:]], 0)

    def _propagate(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(activity after the last step [N, B], KC code that drove it [n_KC, B], MBON pre-activation of it [n_MBON, B])."""
        B = x.shape[0]
        I_in = torch.zeros(self.N, B, device=self.dev).index_copy(0, self.aff_rows, self.w_in @ x.T)
        w = self.sign * Fn.softplus(self.theta); wkm = Fn.softplus(self.theta_km)
        mb_rows, kc_rows = self.mb0 + self.km_mbon, self.kc0 + self.km_kc
        h = torch.zeros(self.N, B, device=self.dev); k = u = None
        for step in range(self.k_steps):
            msg = torch.zeros_like(h).index_add(0, self.post, h.index_select(0, self.pre) * w[:, None])
            msg = msg.index_add(0, mb_rows, h.index_select(0, kc_rows) * wkm[:, None])
            pre_act = self.leak * h + self.gain[:, None] * msg + self.bias[:, None] + I_in
            if step == self.k_steps - 1:
                k, u = h[self.kc0:self.kc1], pre_act[self.mb0:self.mb1]
            h = self._kwta(torch.tanh(pre_act))
        return h, k, u

    def forward_parts_all_h(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``forward_parts_all`` plus every neuron's activity after the last step [B, N] (the console's brain view)."""
        h, k, u = self._propagate(x)
        z = self.dec(self.eff_norm(h.index_select(0, self.eff_rows).T))
        return torch.cat([hd(z) for hd in self.heads], dim=1), u.T, k.T, h.T

    def forward_parts_all(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(every strategy head's decoder output [B, S] × SCALE, MBON pre-activation at the last step [B, n_MBON], KC code [B, n_KC])."""
        return self.forward_parts_all_h(x)[:3]

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(the first head's decoder output [B] × SCALE, MBON pre-activation at the last step [B, n_MBON], KC code [B, n_KC])."""
        y, u, k = self.forward_parts_all(x)
        return y[:, 0], u, k

    def readout(self, y_dn: torch.Tensor, u0: torch.Tensor, k: torch.Tensor, D: torch.Tensor | None = None, s: int = 0) -> torch.Tensor:
        """Strategy ``s``'s prediction × SCALE from the parts; ``D`` [n_KC, n_MBON] is a plastic change of the KC→MBON
        weights at the last step; the strategy reads its own channel's MBONs and the unclaimed ones."""
        u = u0 if D is None else u0 + self.gain[self.mb0:self.mb1] * (k @ (D * self.mask))
        return y_dn + (self.mb_norm(torch.tanh(u)) * self.read[s].float()) @ self.c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(*self.forward_parts(x))

    def forward_all(self, x: torch.Tensor) -> torch.Tensor:
        """Every strategy's prediction × SCALE [B, S]."""
        y, u, k = self.forward_parts_all(x)
        z = self.mb_norm(torch.tanh(u))
        return y + (z[:, None, :] * self.read.float()[None]) @ self.c

    def mbon_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The frozen MBON normalisation (mean, standard deviation) the readout uses in eval mode."""
        return self.mb_norm.running_mean, torch.sqrt(self.mb_norm.running_var + self.mb_norm.eps)

    def param_groups(self, lr_graph: float, lr_head: float) -> list[dict]:
        graph = [self.theta, self.gain, self.bias, self.theta_km]; ids = {id(p) for p in graph}
        return [{"params": graph, "lr": lr_graph}, {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr_head}]


class FlyModel:
    """A trained fly as a trading model: one value per strategy of the selector's stack (``rules``: its fitted trigger,
    high window and hold), each on its own dopamine channel, with the fly's own line and sizing per strategy. The
    single-strategy interface of ``selector.SelectorModel`` (score, threshold, sizing, universe) is the first strategy's."""

    def __init__(self, net: FlyNet, scaler: RobustScaler, cols: list[str], horizon_min: int, threshold: float | None = None, sizing: list | None = None,
                 rules: dict | None = None, lines: dict | None = None, sizings: dict | None = None, combine: str = "score",
                 chain_lines: dict | None = None, chain_sizings: dict | None = None):
        self.net, self.scaler, self.cols, self.horizon_min = net, scaler, list(cols), horizon_min
        self.rules = dict(rules or {"ev": {"thr": {}, "high": None, "hold_min": int(horizon_min)}})
        self.strategies = list(self.rules)
        base = threshold if threshold is not None else selector.MIN_EV
        self.lines = {k: float((lines or {}).get(k, base)) for k in self.strategies}
        self.sizings = {k: list((sizings or {}).get(k, sizing or [])) for k in self.strategies}
        self.combine = combine
        # the other chains' own lines and sizing per strategy (fly_trader/markets: 'rh'); ``lines`` / ``sizings`` are Solana's.
        # One network scores both chains; each chain's line is calibrated on that chain's rows (its teacher budget too).
        self.chain_lines = {ch: {k: float(v) for k, v in d.items()} for ch, d in (chain_lines or {}).items()}
        self.chain_sizings = {ch: {k: list(v) for k, v in d.items()} for ch, d in (chain_sizings or {}).items()}

    def line_for(self, name: str, chain: str | None = "sol") -> float:
        if chain in (None, "sol"):
            return self.lines[name]
        return self.chain_lines.get(chain, {}).get(name, self.lines[name])

    def sizing_for(self, name: str, chain: str | None = "sol") -> list:
        if chain in (None, "sol"):
            return self.sizings[name]
        return self.chain_sizings.get(chain, {}).get(name, self.sizings[name])

    def set_line(self, name: str, chain: str, line: float, sizing_: list) -> None:
        if chain in (None, "sol"):
            self.lines[name], self.sizings[name] = float(line), list(sizing_)
        else:
            self.chain_lines.setdefault(chain, {})[name] = float(line); self.chain_sizings.setdefault(chain, {})[name] = list(sizing_)

    @property
    def threshold(self) -> float:
        return self.lines[self.strategies[0]]

    @threshold.setter
    def threshold(self, v: float) -> None:
        self.lines[self.strategies[0]] = float(v)

    @property
    def sizing(self) -> list:
        return self.sizings[self.strategies[0]]

    @sizing.setter
    def sizing(self, v: list) -> None:
        self.sizings[self.strategies[0]] = list(v or [])

    def hold_s(self, s: int) -> float:
        return float(self.rules[self.strategies[s]]["hold_min"]) * 60.0

    def _x(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(self.scaler.transform(X), device=self.net.dev)

    @torch.no_grad()
    def score_all(self, X: np.ndarray, batch: int | None = None) -> np.ndarray:
        """[B, S] every strategy's predicted net return over its hold (the frozen fly). ``batch`` defaults to what the
        device's memory affords (``FlyNet.batch_rows``)."""
        self.net.eval(); batch = batch or self.net.batch_rows(); out = np.empty((len(X), len(self.strategies)), np.float64)
        for i in range(0, len(X), batch):
            out[i:i + batch] = (self.net.forward_all(self._x(X[i:i + batch])) / self.net.scale).float().cpu().numpy()
        return out

    def score(self, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        """The first strategy's predicted net return over its hold (the frozen fly)."""
        return self.score_all(X, batch)[:, 0]

    @torch.no_grad()
    def parts(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``FlyNet.forward_parts`` on raw feature rows (one batch; eval mode)."""
        self.net.eval()
        return self.net.forward_parts(self._x(X))

    @torch.no_grad()
    def parts_all(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``FlyNet.forward_parts_all``: every head's decoder output [B, S], MBON pre-activation, KC code."""
        return self.parts_all_h(X)[:3]

    @torch.no_grad()
    def parts_all_h(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """``parts_all`` plus the activity of every neuron [B, N], in as many batches as the device's memory asks for
        (``FlyNet.batch_rows``): a busy minute must not exceed a small card."""
        self.net.eval(); b = self.net.batch_rows()
        if len(X) <= b:
            return self.net.forward_parts_all_h(self._x(X))
        parts = [self.net.forward_parts_all_h(self._x(X[i:i + b])) for i in range(0, len(X), b)]
        return tuple(torch.cat([p[j] for p in parts]) for j in range(4))

    def triggers(self, X: np.ndarray, cols: list[str]) -> np.ndarray:
        """[B, S] where each strategy's fitted trigger (the selector's) fires — its candidates."""
        from . import strategies
        X = np.atleast_2d(X); out = np.zeros((len(X), len(self.strategies)), bool)
        for j, name in enumerate(self.strategies):
            r = self.rules[name]
            out[:, j] = (strategies.base_mask(name, X, cols, r.get("high")) & strategies.trigger_mask(name, r.get("thr") or {}, X, cols)) if name in strategies.STRATEGIES else True
        return out

    def universe(self, X: np.ndarray) -> np.ndarray:
        return selector.in_universe(X, self.cols)


def _device() -> torch.device:
    """config.DEVICE resolved (brain/device.py): CUDA, MPS or CPU — the same answer the connectome gets."""
    return brain_device.resolve()


def _graph():
    return SubConnectome(Connectome.load(), exclude=("VISUAL",))


def distil_rows(idx: np.ndarray, tgt: np.ndarray, rng: np.random.Generator, top: float = DISTIL_TOP, sample: int = DISTIL_SAMPLE) -> np.ndarray:
    """Positions (into ``idx``) the fly trains on: every row in the teacher's top ``top`` plus a random ``sample`` of the rest."""
    if len(idx) <= sample:
        return np.arange(len(idx))
    hi = np.flatnonzero(tgt >= np.quantile(tgt, 1.0 - top)); rest = np.setdiff1d(np.arange(len(idx)), hi, assume_unique=True)
    return np.sort(np.r_[hi, rng.choice(rest, min(sample, len(rest)), replace=False)])


@torch.no_grad()
def recalibrate(net: FlyNet, X: np.ndarray, scaler: RobustScaler, batch: int = BATCH) -> None:
    """Both normalisations re-estimated as plain averages over ``X`` (not the last few training batches), then frozen."""
    norms = (net.eff_norm, net.mb_norm)
    for bn in norms:
        bn.reset_running_stats(); bn.momentum = None
    net.train()
    for i in range(0, len(X), batch):
        net(torch.tensor(scaler.transform(X[i:i + batch]), device=net.dev))
    net.eval()


class StackTeacher:
    """The selector's strategy stack (train/strategies.final_models) as the fly's teacher: per strategy, on its candidates,
    the net return it predicts where the stack would trade, and one typical score-spread (MAD) below that strategy's
    line where a filter (win filter, gates, dump veto) blocks it — so the fly learns the final decision, not the raw score."""

    def __init__(self, models: dict, scaler: RobustScaler, cols: list[str]):
        self.models, self.scaler, self.cols = models, scaler, list(cols)
        self.strategies = list(models.get("strategies") or {})
        self.rules = {k: {"thr": dict(v["thr"]), "high": v["high"], "hold_min": int(v["hold_min"])} for k, v in models["strategies"].items()}
        self.lines = {k: float(v["line"]) for k, v in models["strategies"].items()}
        self.threshold = self.lines[self.strategies[0]] if self.strategies else selector.MIN_EV
        self.combine = models.get("combine", "score"); self.eps: dict = {}

    def raw_scores(self, X: np.ndarray, ts, names: list[str]) -> np.ndarray:
        """[B, S] its own score of each row per strategy (NaN off the trigger): the starting score of a fly that scores as
        its teacher plus a learned correction (config.FLY_SCORE = teacher)."""
        return teacher_scores(self.models, X, ts, names, self.cols)

    def targets(self, X: np.ndarray, ts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(targets [B, S] with NaN off each strategy's candidates, allowed [B, S])."""
        from . import strategies
        per = strategies.decide_all(self.models, X, self.cols, ts); T = np.full((len(X), len(self.strategies)), np.nan); A = np.zeros_like(T, dtype=bool)
        for j, k in enumerate(self.strategies):
            d = per[k]; sc = d["score"]; trig = d["trig"]
            if k not in self.eps:
                al = sc[d["allow"] & np.isfinite(sc)]
                self.eps[k] = float(np.median(np.abs(al - np.median(al)))) if len(al) else 0.0
            T[trig, j] = np.where(d["allow"][trig], sc[trig], np.minimum(sc[trig], self.lines[k] - self.eps[k])); A[:, j] = d["allow"]
        return T, A


def teacher_allowed(teacher, X: np.ndarray, ts: np.ndarray, names: list[str], chunk: int = 200_000) -> np.ndarray:
    """[B, S]: where the teacher itself would trade each of the fly's strategies (its final decision: trigger, line and
    every filter) — the selectivity ``fly_calibrate``'s ``match`` rule copies."""
    A = np.zeros((len(X), len(names)), bool)
    for i in range(0, len(X), chunk):
        if isinstance(teacher, StackTeacher):
            _, a = teacher.targets(X[i:i + chunk], ts[i:i + chunk])
            for j, k in enumerate(names):
                if k in teacher.strategies:
                    A[i:i + chunk, j] = a[:, teacher.strategies.index(k)]
        else:
            A[i:i + chunk, :] = (teacher.score(X[i:i + chunk]) >= teacher.threshold)[:, None]
    return A


def _teacher_targets(teacher, ds: DecisionSet, idx: np.ndarray, clip: tuple = (-1.0, 1.0)) -> tuple[np.ndarray, np.ndarray]:
    """[n, S] targets (NaN: not a candidate of that strategy) and loss weights for the rows ``idx``."""
    if isinstance(teacher, StackTeacher):
        T = np.full((len(idx), len(teacher.strategies)), np.nan, np.float32); W = np.ones_like(T)
        for i in range(0, len(idx), 200_000):
            t, a = teacher.targets(ds.X[idx[i:i + 200_000]], ds.ts[idx[i:i + 200_000]])
            T[i:i + 200_000] = t; W[i:i + 200_000] = np.where(a, TOP_WEIGHT, 1.0)
        return np.clip(T, clip[0], clip[1]), W
    tgt = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), 500_000):
        tgt[i:i + 500_000] = teacher.score(ds.X[idx[i:i + 500_000]])
    tgt = np.clip(tgt, -1.0, 1.0)
    return tgt[:, None], np.where(tgt >= np.quantile(tgt, 0.99), TOP_WEIGHT, 1.0).astype(np.float32)[:, None]


@torch.no_grad()
def _val_mse(net: FlyNet, scaler: RobustScaler, ds: DecisionSet, idx: np.ndarray, T: np.ndarray, W: np.ndarray, val: np.ndarray,
             dev: torch.device, batch: int) -> tuple[float, float]:
    """Weighted squared error on distillation rows held out of training, and the standard error of that mean. An epoch
    that buys less than the noise in this estimate has stopped teaching the fly anything, which is where training ends:
    the count comes from the data instead of being chosen. Until 2026-09-21 the bootstrap ran exactly one epoch, leaving
    a per-row error of ~3.9 % net return against a 4.6 % buy line — the ordering near the line was mostly noise."""
    net.eval(); parts = []
    for i in range(0, len(val), batch):
        sl = val[i:i + batch]
        x = torch.tensor(scaler.transform(ds.X[idx[sl]]), device=dev)
        t = torch.tensor(np.nan_to_num(T[sl]) * net.scale, device=dev); m = torch.tensor(np.isfinite(T[sl]), device=dev).float()
        w = torch.tensor(W[sl], device=dev) * m
        parts.append(((w * (net.forward_all(x) - t) ** 2).sum(1) / w.sum(1).clamp(min=1e-9)).cpu())
    net.train()
    e = torch.cat(parts).double() if parts else torch.zeros(1, dtype=torch.float64)
    return float(e.mean()), float(e.std() / max(len(e) ** 0.5, 1.0))


def _resume(path, fp: dict) -> dict | None:
    """The checkpoint at ``path`` when it was written for this same training set (``fp``): training resumes after its last
    epoch. One written for other data is set aside (``.stale``) and training starts fresh."""
    p = Path(path)
    if not p.exists():
        return None
    ck = model_io.torch_load(p)
    if ck.get("fingerprint") != fp:
        stale = p.with_name(p.name + ".stale"); p.replace(stale)
        log.warning("fly checkpoint %s was written for other training data (%s; now %s): set aside as %s, training starts fresh",
                    p.name, ck.get("fingerprint"), fp, stale.name)
        return None
    log.info("fly training resumes after epoch %d from %s", int(ck["epoch"]), p)
    return ck


def _write_checkpoint(path, obj: dict) -> None:
    """``obj`` at ``path``, atomically: a run killed while writing leaves the previous epoch's checkpoint whole."""
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True); tmp = p.with_name(p.name + ".tmp")
    model_io.torch_save(obj, tmp); tmp.replace(p)


def train_fly(ds: DecisionSet, rows: np.ndarray, teacher, epochs: int = EPOCHS, batch: int = BATCH, stop: threading.Event | None = None,
              seed: int = 0, graph=None, device=None, net_kwargs: dict | None = None, model_cls=None, target_clip: tuple = (-1.0, 1.0),
              horizon_min: int | None = None, checkpoint=None, label_known: bool = False) -> tuple[FlyModel | None, dict]:
    """Distil ``teacher`` (a StackTeacher, or anything with ``score`` and ``scaler``) into a FlyNet over the training rows
    ``rows`` (mask): one head per strategy, each learning its strategy's targets on that strategy's candidates.
    ``net_kwargs`` (``aff_rows``, ``scale``), ``model_cls`` and ``target_clip`` let another trading type (kalshi/fly.py) run
    its own fly through this loop; the memecoin defaults are unchanged. ``checkpoint``: a file written after every epoch
    (weights, optimiser, history, the best epoch so far, the random state): a run that dies resumes after its last
    finished epoch, and a finished one goes on past the cap it ran under when ``epochs`` is raised. A file written for
    other training data is set aside, never resumed. ``label_known`` (matched mode): a strategy learns only on rows whose
    label over its hold exists, the rows its teacher's own fit used (strategies.final_models)."""
    brain_device.seed_all(seed); rng = np.random.default_rng(seed); dev = torch.device(device) if device else _device()
    n_s = len(teacher.strategies) if isinstance(teacher, StackTeacher) else 1
    net = FlyNet(graph if graph is not None else _graph(), obs_dim=len(ds.cols), device=dev, n_strategies=n_s, **(net_kwargs or {}))
    model_cls = model_cls or FlyModel
    idx = np.flatnonzero(rows); T, W = _teacher_targets(teacher, ds, idx, target_clip)
    if label_known and isinstance(teacher, StackTeacher):
        for j, k in enumerate(teacher.strategies):
            T[~np.isfinite(_hold_label(ds, int(teacher.rules[k]["hold_min"]))[idx]), j] = np.nan
    keep = np.isfinite(T).any(1); idx, T, W = idx[keep], T[keep], W[keep]
    if not len(idx):
        return None, {"stopped": False, "reason": "no teacher candidates"}
    use = distil_rows(idx, np.nanmax(np.where(np.isfinite(T), T, -np.inf), axis=1), rng)
    val = np.empty(0, use.dtype)
    if epochs > 1 and len(use) // 10 >= VAL_MIN_ROWS:      # a smaller slice cannot tell a real improvement from noise
        n_v = min(VAL_ROWS, len(use) // 10); p0 = rng.permutation(len(use)); val, use = np.sort(use[p0[:n_v]]), np.sort(use[p0[n_v:]])
    with torch.no_grad():
        for j, hd in enumerate(net.heads):
            col = T[:, j]; hd.bias.fill_(float(np.nanmean(col)) * net.scale if np.isfinite(col).any() else 0.0); hd.weight.mul_(0.1)
    batch = min(batch, net.batch_rows(ceiling=batch, training=True))      # a training step keeps every propagation step: ~4x a forward pass
    opt = torch.optim.Adam(net.param_groups(LR_GRAPH, LR_HEAD)); n_b = max(1, len(use) // batch); total = max(1, epochs * n_b); hist = []; t0 = time.time()
    best = best_v = prev_v = early = None; start = 0
    fp = {"training_rows": int(len(idx)), "rows": int(len(use)), "val_rows": int(len(val)), "seed": int(seed), "obs_dim": int(len(ds.cols)),
          "strategies": list(teacher.strategies) if isinstance(teacher, StackTeacher) else []}
    ck = _resume(checkpoint, fp) if checkpoint is not None else None
    if ck is not None:
        net.load_state_dict(ck["state_dict"])
        if ck.get("opt") is not None:
            opt.load_state_dict(ck["opt"])
        if ck.get("rng") is not None:
            rng.bit_generator.state = ck["rng"]
        hist, best, best_v, prev_v, early, start = list(ck["hist"]), ck.get("best"), ck.get("best_v"), ck.get("prev_v"), ck.get("early"), int(ck["epoch"]) + 1
    for ep in range(start, epochs if early is None else start):
        perm = use[rng.permutation(len(use))]; tot = 0.0; net.train()
        for b in range(n_b):
            if stop is not None and stop.is_set():
                return None, {"epochs": hist, "stopped": True}
            sel = perm[b * batch:(b + 1) * batch]
            x = torch.tensor(teacher.scaler.transform(ds.X[idx[sel]]), device=dev)
            t = torch.tensor(np.nan_to_num(T[sel]) * net.scale, device=dev); m = torch.tensor(np.isfinite(T[sel]), device=dev).float()
            w = torch.tensor(W[sel], device=dev) * m
            loss = (w * (net.forward_all(x) - t) ** 2).sum() / w.sum().clamp(min=1e-9)
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tot += loss.item()
            if b % 200 == 0:
                done = ep * n_b + b + 1; here = (ep - start) * n_b + b + 1; el = time.time() - t0      # ``here``: this run's steps (a resumed run's rate)
                prog.update("fly: learning from the selector", done, total, epoch=ep, loss=tot / (b + 1), rows_per_s=here * batch / max(el, 1e-9),
                            eta_s=(total - done) * el / here)
        hist.append({"epoch": ep, "mse": tot / n_b, "secs": time.time() - t0, "rows": min(len(use), n_b * batch)})
        log.info("fly epoch %d: weighted mse %.4f over %d rows (%.0fs)", ep, tot / n_b, min(len(use), n_b * batch), time.time() - t0)
        if len(val):
            vm, vse = _val_mse(net, teacher.scaler, ds, idx, T, W, val, dev, batch)
            hist[-1]["val_mse"], hist[-1]["val_se"] = vm, vse
            log.info("fly epoch %d: held-out weighted mse %.4f +/- %.4f over %d rows", ep, vm, vse, len(val))
            if best_v is None or vm < best_v:
                best_v, best = vm, {k: v.detach().clone() for k, v in net.state_dict().items()}
            gain = None if prev_v is None else prev_v - vm
            stop_now = ep + 1 >= MIN_EPOCHS and gain is not None and gain <= vse
            prev_v = vm
            if stop_now:
                early = f"epoch {ep} moved the held-out error by {gain:+.4f}, inside its standard error {vse:.4f}"
                log.info("fly: stopping early - %s", early)
        if checkpoint is not None:
            _write_checkpoint(checkpoint, {"fingerprint": fp, "epoch": ep, "state_dict": net.state_dict(), "opt": opt.state_dict(), "best": best,
                                           "best_v": best_v, "prev_v": prev_v, "hist": hist, "rng": rng.bit_generator.state, "early": early})
        if early is not None:
            break
    if best is not None:
        net.load_state_dict(best)                          # the epoch that generalised best, not merely the last one
    opt = None; brain_device.empty_cache(dev)                          # the optimiser's state and the autograd workspace go back to the device
    rec = idx[rng.choice(len(idx), min(RECAL_ROWS, len(idx)), replace=False)]
    recalibrate(net, ds.X[np.sort(rec)], teacher.scaler, batch)
    if isinstance(teacher, StackTeacher):
        h0 = teacher.rules[teacher.strategies[0]]["hold_min"]
        fly = model_cls(net, teacher.scaler, ds.cols, horizon_min if horizon_min is not None else int(h0), rules=teacher.rules, combine=teacher.combine)
    else:
        fly = model_cls(net, teacher.scaler, ds.cols, horizon_min if horizon_min is not None else int(ds.horizon_s // 60))
    return fly, {"epochs": hist, "stopped": False, "rows": int(len(use)), "training_rows": int(len(idx)), "top_weight": TOP_WEIGHT,
                 "strategies": fly.strategies, "epochs_run": len(hist), "epoch_cap": int(epochs), "val_rows": int(len(val)),
                 "val_mse": best_v, "early_stop": early}


@torch.no_grad()
def diagnose(fly: FlyModel, X: np.ndarray, teacher_scores: np.ndarray | None = None, batch: int | None = None) -> dict:
    """How much of the score runs through the mushroom body, how saturated the MBONs are, how sparse the KC code is,
    and (given the teacher's scores) how the fly's scores line up with them."""
    net = fly.net; net.eval(); batch = batch or net.batch_rows(); mb, tot, sat, act = [], [], [], []
    for i in range(0, len(X), batch):
        y_dn, u0, k = fly.parts(X[i:i + batch])
        m = net.mb_norm(torch.tanh(u0)) @ net.c
        mb.append(m.cpu()); tot.append((y_dn + m).cpu()); sat.append(u0.abs().cpu()); act.append((k > 0).float().mean(1).cpu())
    mb, tot, sat, act = torch.cat(mb).double(), torch.cat(tot).double(), torch.cat(sat), torch.cat(act)
    out = {"mbon_share": float(mb.var() / tot.var()) if tot.var() > 0 else 0.0, "mbon_saturated": float((sat.median(0).values > 2.0).float().mean()),
           "kc_active": float(act.mean()), "rows": int(len(X))}
    if teacher_scores is not None:
        s = (tot / net.scale).numpy()
        out["slope"] = float(np.polyfit(np.asarray(teacher_scores, dtype=np.float64), s, 1)[0]) if np.std(teacher_scores) > 0 else None
        out["rank_corr"] = rank_corr(s, teacher_scores)
    return out


def gate_check(diag: dict) -> tuple[bool, list[str]]:
    why = []
    if diag["mbon_share"] < GATES["mbon_share_min"]:
        why.append(f"MBONs carry {diag['mbon_share']:.1%} of the score (< {GATES['mbon_share_min']:.0%}): plasticity would move little")
    if diag["mbon_saturated"] > GATES["mbon_saturated_max"]:
        why.append(f"{diag['mbon_saturated']:.0%} of MBONs saturated (> {GATES['mbon_saturated_max']:.0%})")
    sl = diag.get("slope")
    if sl is not None and not (GATES["slope_min"] <= sl <= GATES["slope_max"]):
        why.append(f"score slope {sl:.2f} against the teacher outside {GATES['slope_min']}–{GATES['slope_max']}")
    return not why, why


def _stack_teacher(ds: DecisionSet, train: np.ndarray, stop=None):
    """The selector's strategy stack fitted on the training days only (an honest teacher for the calibration week)."""
    from . import strategies
    sub = ds.subset(train)
    stack = strategies.fit_stack(sub, stop, holdout_days_n=0)   # already only days before S - 8; the replay is its out-of-sample proof
    if not stack.fits:
        return None, stack
    return StackTeacher(strategies.final_models(sub, stack), RobustScaler.fit(sub.X, seed=7), ds.cols), stack


def deployed_teacher(ds: DecisionSet, train: np.ndarray, calib_start: date | None = None) -> tuple[StackTeacher | None, str, int | None]:
    """The selector the live book actually trades, wrapped as the fly's teacher, with the reason and the snapshot id.

    Until 2026-09-21 the bootstrap always refit its own stack on the days before ``S − 8`` and distilled that instead.
    Component selection is unstable enough that the refit was a different trading system: the deployed stack (#60, fit
    through 2026-09-18) traded ``ev`` at line 0.046 over 120 minutes, while the fly's own teacher (through 2026-09-11)
    traded ``capitulation`` at 0.010 over 240 — five of eight components flipped by one extra week of corpus. The fly
    was never a copy of the selector it was being judged against. None (with the reason) falls back to that refit.

    ``calib_start``: the first day of the fly's calibration week. The deployed selector is fit on the whole corpus, that
    week included, so distilling it let the teacher's in-sample fit flatter the fly's line and sizing bands there (fly
    #107: 474 calibration trades at +2.79 %, then its biggest bets lost live). Given ``calib_start``, the teacher is the
    same stack refit without its holdout days (``SelectorModel.blind``: same strategies, settings and lines), used only
    when that fit ends at least ``PURGE_DAYS`` before the week; a selector whose own fit already ends there is used as
    is; otherwise None with the reason, and the bootstrap refits a stack on its training days."""
    from ..agent.selector_session import pinned_snapshot
    sid = pinned_snapshot()
    if sid is None:
        with transaction() as conn:
            r = selector.latest_current(conn)
        sid = int(r["id"]) if r else None
    if sid is None:
        return None, TEACHER_NOTE, None
    m = selector.load_snapshot(sid)
    if m is None:
        return None, f"selector #{sid} could not be loaded; {TEACHER_NOTE}", None
    st = getattr(m, "stack", None) or {}
    if not st.get("strategies"):
        return None, f"selector #{sid} has no strategy stack; {TEACHER_NOTE}", None
    seen = ""
    if calib_start is not None:
        last_ok = calib_start - timedelta(days=PURGE_DAYS + 1)          # the last day a fit may reach, labels spilling over included
        blind = getattr(m, "blind", None) or {}
        through = str(getattr(m, "trained_through", "") or "")
        if blind.get("models", {}).get("strategies") and date.fromisoformat(blind["fit_through"]) <= last_ok:
            st = blind["models"]; seen = f", refit through {blind['fit_through']} (blind to the calibration week from {calib_start})"
        elif through and date.fromisoformat(through) <= last_ok:
            seen = f", fit through {through} (before the calibration week from {calib_start})"
        else:
            fit = blind.get("fit_through") or through or "an unknown day"
            return None, (f"selector #{sid} saw the calibration week from {calib_start} (its fit reaches {fit}"
                          f"{'' if blind else ', and it has no blind models'}); {TEACHER_NOTE}"), None
    missing = sorted({c for v in st["strategies"].values() for c in v["cols"]} - set(ds.cols))
    if missing:
        return None, f"selector #{sid} wants columns the corpus lacks ({', '.join(missing[:4])}); {TEACHER_NOTE}", None
    names = ", ".join(f"{k} (line {v['line']:.4f}, {v['hold_min']} min)" for k, v in st["strategies"].items())
    return StackTeacher(st, RobustScaler.fit(ds.X[np.flatnonzero(train)], seed=7), list(ds.cols)), f"deployed selector #{sid}: {names}{seen}", sid


def row_chains(X: np.ndarray, cols: list[str]) -> np.ndarray:
    """Each row's chain from its ``chain_rh`` input ('sol' where the column is absent or 0)."""
    X = np.atleast_2d(X)
    if "chain_rh" not in cols:
        return np.full(len(X), "sol", dtype=object)
    return np.where(X[:, cols.index("chain_rh")] > 0.5, "rh", "sol").astype(object)


def fly_decide(fly: FlyModel, X: np.ndarray, cols: list[str], values: np.ndarray | None = None) -> dict:
    """The fly's own per-row decision: among the strategies whose trigger fires and whose value clears the fly's line
    for that row's chain, the one with the larger sizing-band Kelly fraction or margin (the teacher's combination rule)."""
    V = fly.score_all(X) if values is None else values; trig = fly.triggers(X, cols); n = len(X)
    strat = np.full(n, None, dtype=object); score = np.full(n, -1.0); hold = np.zeros(n); thr = np.full(n, np.inf); tables = [[] for _ in range(n)]
    key = np.full(n, -np.inf); ch = row_chains(X, cols)
    for j, name in enumerate(fly.strategies):
        v = V[:, j]
        line = np.array([fly.line_for(name, c) for c in ch]) if n else np.zeros(0)
        tabs = [fly.sizing_for(name, c) for c in ch]
        ok = trig[:, j] & (v >= line)
        kv = (np.array([((sizing.band_for(t, m) or {}).get("kelly") or 0.0) for t, m in zip(tabs, v - line)]) + 1e-9 * v
              if fly.combine == "kelly" and any(tabs) else v - line)
        take = ok & (kv > key)
        key = np.where(take, kv, key)
        for i in np.flatnonzero(take):
            strat[i] = name; score[i] = v[i]; hold[i] = fly.rules[name]["hold_min"] * 60.0; thr[i] = line[i]; tables[i] = tabs[i]
    return {"strategy": strat, "score": score, "hold_s": hold, "threshold": thr, "tables": tables}


def bootstrap(ds: DecisionSet, S: date, epochs: int = EPOCHS, stop: threading.Event | None = None, seed: int = 0, graph=None,
              device=None, teacher=None, teacher_note: str | None = None, checkpoint=None) -> tuple[FlyModel | None, dict]:
    """The one time the selector teaches: a fly ready to trade from day ``S`` on, with its own line and sizing per strategy.
    ``teacher``: the live path passes the deployed selector (``deployed_teacher``); left None, the stack is refit on the
    days before ``S − 8`` — which is what the replay wants, since there the refit is the honest out-of-sample teacher (a
    set without the stack's inputs or per-hold labels — synthetic tests — gets the single-model selector).
    ``teacher_note`` records in the snapshot which selector taught this fly. ``checkpoint``: the training's per-epoch
    checkpoint file (``train_fly``)."""
    from .decisions import LEGACY_COLS
    train_end = S - timedelta(days=CALIB_DAYS + PURGE_DAYS)
    train = ds.day < train_end; uni = selector.in_universe(ds.X, ds.cols)
    if not train.any():
        raise RuntimeError(f"no training days before {train_end}")
    stack_info = None
    if teacher is None:
        prog.update("fly: fitting its teacher (the selector's strategy stack, training days only)", 0, 1, force=True)
        if ds.fwd_h and set(LEGACY_COLS) <= set(ds.cols):
            teacher, stack = _stack_teacher(ds, train, stop)
            stack_info = {"components": stack.components, "deployable": stack.deployable, "reason": stack.reason}
            if teacher is None:
                return None, {"S": str(S), "gates_ok": False, "gate_failures": ["the teacher stack has no strategy on the training days"], "teacher_stack": stack_info}
        else:
            teacher = selector.fit(ds, train, seed=7)
    fly, fit_info = train_fly(ds, train & uni, teacher, epochs=epochs, stop=stop, seed=seed, graph=graph, device=device, checkpoint=checkpoint)
    if fly is None:
        return None, {"stopped": True, **fit_info}
    s_epoch = datetime(S.year, S.month, S.day, tzinfo=timezone.utc).timestamp()
    week = (ds.day >= S - timedelta(days=CALIB_DAYS)) & (ds.day < S) & uni
    prog.update("fly: calibrating its own buy lines", 0, 1, force=True)
    wi = np.flatnonzero(week); V = fly.score_all(ds.X[wi]) if len(wi) else np.zeros((0, len(fly.strategies))); trig = fly.triggers(ds.X[wi], ds.cols)
    A = teacher_allowed(teacher, ds.X[wi], ds.ts[wi], fly.strategies) if fly_calibrate.RULE == "match" else None
    per = {}; wch = ds.chains()[wi]
    for chain in sorted(set(wch.tolist()), key=lambda c: c != "sol"):       # Solana first: another chain without rows keeps its lines
        for j, name in enumerate(fly.strategies):
            H = fly.rules[name]["hold_min"]; y = ds.fwd_h.get(H, ds.fwd_pess)
            known = trig[:, j] & (ds.ts[wi] + H * 60.0 + 60.0 <= s_epoch) & (wch == chain)       # labels known at S, this chain's rows
            ci = wi[known]
            target = (len(taken_idx(ds.ts[ci], ds.mint[ci], H * 60.0, np.flatnonzero(A[known, j] & np.isfinite(y[ci])))) if A is not None else None)
            cal = fly_calibrate.calibrate(ds.ts[ci], ds.mint[ci], H * 60.0, V[known, j], y[ci], rule=fly_calibrate.RULE, target_trades=target,
                                          prev_line=fly.lines.get(name) if chain != "sol" else None)
            fly.set_line(name, chain, cal.line, cal.sizing)
            per[name if chain == "sol" else f"{name}@{chain}"] = {"line": cal.line, "trades": cal.trades, "mean": cal.mean, "total": cal.total, "hold_min": H,
                                                                "rule": fly_calibrate.RULE, "teacher_trades": target, "chain": chain}
    d = fly_decide(fly, ds.X[wi], ds.cols, V) if len(wi) else {"strategy": np.array([], dtype=object), "hold_s": np.array([])}
    picked = np.array([x is not None for x in d["strategy"]], bool)
    ret = np.array([ds.fwd_h.get(fly.rules[x]["hold_min"], ds.fwd_pess)[i] if x is not None else np.nan for i, x in zip(wi, d["strategy"])])
    known = picked & (ds.ts[wi] + d["hold_s"] + 60.0 <= s_epoch)
    tr = taken_idx(ds.ts[wi], ds.mint[wi], d["hold_s"], np.flatnonzero(known & np.isfinite(ret))); r = ret[tr]       # trades with a known outcome
    comb = {"trades": int(len(r)), "mean": float(np.mean(r)) if len(r) else None, "total": float(np.sum(r)) if len(r) else 0.0, "per_strategy": per}
    ci = np.flatnonzero(week & (ds.ts + ds.horizon_s + 60.0 <= s_epoch))
    di = ci if len(ci) <= DIAG_ROWS else np.sort(np.random.default_rng(seed).choice(ci, DIAG_ROWS, replace=False))
    t_sc = teacher.score(ds.X[di]) if hasattr(teacher, "score") else (teacher.targets(ds.X[di], ds.ts[di])[0][:, 0] if len(di) else None)
    if t_sc is not None and len(di):
        ok_t = np.isfinite(t_sc); di, t_sc = di[ok_t], t_sc[ok_t]
    diag = diagnose(fly, ds.X[di], t_sc) if len(di) else {"mbon_share": 0.0, "mbon_saturated": 1.0, "kc_active": 0.0, "rows": 0}
    ok, why = gate_check(diag)
    fly.teacher = teacher          # not saved: the replay's match rule reads the teacher's own decisions over its rows
    info = {"S": str(S), "train_through": str(train_end - timedelta(days=1)), "calibration_days": [str(S - timedelta(days=CALIB_DAYS)), str(S - timedelta(days=1))],
            "line": fly.threshold, "sizing": fly.sizing, "lines": fly.lines, "chain_lines": fly.chain_lines, "calibration": comb, "strategies": fly.strategies, "rules": fly.rules,
            "diagnostics": diag, "gates_ok": ok, "gate_failures": why, "fit": fit_info, "teacher": teacher_note or TEACHER_NOTE, "teacher_line": getattr(teacher, "threshold", None),
            "teacher_stack": stack_info}
    log.info("fly bootstrap for %s: strategies %s, lines %s | calibration %d trades, mean %s | MBON share %.1f%% saturated %.0f%% | gates %s",
             S, fly.strategies, {k: round(v, 4) for k, v in fly.lines.items()}, comb["trades"], f"{comb['mean'] * 100:+.2f}%" if comb["mean"] is not None else "-",
             diag["mbon_share"] * 100, diag["mbon_saturated"] * 100, "ok" if ok else "; ".join(why))
    return fly, info


# ---------------------------------------------------------------- matched mode (opt-in): train/matched_pair.py
MATCHED = "matched"
MATCH_CHUNK = 1_000_000      # training rows scored at once for the line match: one chunk's copy of the corpus stays small


def _hold_label(ds: DecisionSet, hold_min: int) -> np.ndarray:
    """The net return over ``hold_min`` (the label the fly's line calibration reads: the hold's own, else the set's)."""
    return ds.fwd_h.get(hold_min, ds.fwd_pess)


def scaler_on(ds: DecisionSet, rows: np.ndarray, seed: int = 7) -> RobustScaler:
    """``RobustScaler.fit(ds.X[rows], seed)`` without copying every row first: the fit keeps a ``FIT_SAMPLE`` sample, drawn
    here the same way, so the result is identical (``deployed_teacher``'s copy of the training rows was ~10 GB)."""
    from .scaling import FIT_SAMPLE
    rows = np.asarray(rows)
    if len(rows) > FIT_SAMPLE:
        rows = rows[np.random.default_rng(seed).choice(len(rows), FIT_SAMPLE, replace=False)]
    return RobustScaler.fit(ds.X[rows], seed=seed)


def matched_teacher(ds: DecisionSet, models: dict, exclude_days: list | None = None, note: str | None = None) -> tuple[StackTeacher, np.ndarray, str]:
    """The teacher of a matched fly: the stack ``models`` itself (strategies.final_models: the holdout-blind fit with
    ``exclude_days``, or the full fit), never the registry's deployed selector or a separate refit, with the rows that
    stack was fit on (``strategies.fit_keep``) and its input scaler fitted on them. (teacher, fit rows, note)."""
    from . import strategies
    st = models.get("strategies") or {}
    if not st:
        raise ValueError("a matched fly needs a teacher stack with at least one strategy")
    missing = sorted({c for v in st.values() for c in v["cols"]} - set(ds.cols))
    if missing:
        raise ValueError(f"the teacher stack wants columns the corpus lacks: {', '.join(missing[:4])}")
    fit = strategies.fit_keep(ds, exclude_days)
    if not fit.any():
        raise ValueError("the teacher stack's fit days hold no row of this set")
    days = sorted(set(ds.day[fit].tolist()))
    names = ", ".join(f"{k} (line {v['line']:.4f}, {v['hold_min']} min)" for k, v in st.items())
    seen = f", blind to {min(exclude_days)}..{max(exclude_days)}" if exclude_days else ""
    note = note or f"matched stack: {names}; fit on {days[0]}..{days[-1]}{seen}"
    return StackTeacher(models, scaler_on(ds, np.flatnonzero(fit)), list(ds.cols)), fit, note


def _match_lines(ds: DecisionSet, fly: FlyModel, teacher: StackTeacher, rows: np.ndarray) -> tuple[dict, int]:
    """Each strategy's line on each chain, set by the match rule (``fly_calibrate.match_line``) over the training rows
    ``rows``: the lowest line at which the fly's own scores take no more trades there than its teacher's decisions do (one
    position per token over the hold). The count needs no label; the in-sample stats at the chosen line are recorded."""
    idx = np.flatnonzero(rows); NS = len(fly.strategies)
    holds = [int(fly.rules[k]["hold_min"]) for k in fly.strategies]; ys = [_hold_label(ds, h) for h in holds]
    trig = np.zeros((len(idx), NS), bool)
    for i in range(0, len(idx), MATCH_CHUNK):
        trig[i:i + MATCH_CHUNK] = fly.triggers(ds.X[idx[i:i + MATCH_CHUNK]], ds.cols)
    ok = trig & np.column_stack([np.isfinite(y[idx]) for y in ys])          # the rows each strategy learned on
    cand = ok.any(1); wi = idx[cand]; ok = ok[cand]
    V = np.empty((len(wi), NS)); A = np.zeros((len(wi), NS), bool); t0 = time.time()
    for i in range(0, len(wi), MATCH_CHUNK):
        sl = wi[i:i + MATCH_CHUNK]
        V[i:i + MATCH_CHUNK] = fly.score_all(ds.X[sl]); A[i:i + MATCH_CHUNK] = teacher_allowed(teacher, ds.X[sl], ds.ts[sl], fly.strategies)
        prog.update("fly: scoring its training rows for the line match", min(i + MATCH_CHUNK, len(wi)), len(wi),
                    eta_s=(len(wi) - i - len(sl)) * (time.time() - t0) / max(i + len(sl), 1))
    ch = ds.chains()[wi]; per = {}
    for chain in sorted(set(ch.tolist()), key=lambda c: c != "sol"):        # Solana first: another chain without rows keeps its lines
        for j, name in enumerate(fly.strategies):
            H = holds[j]; known = ok[:, j] & (ch == chain); ci = wi[known]
            target = len(taken_idx(ds.ts[ci], ds.mint[ci], H * 60.0, np.flatnonzero(A[known, j])))
            cal = fly_calibrate.calibrate(ds.ts[ci], ds.mint[ci], H * 60.0, V[known, j], ys[j][ci], rule="match", target_trades=target,
                                          prev_line=fly.lines.get(name) if chain != "sol" else None)
            fly.set_line(name, chain, cal.line, cal.sizing)
            per[name if chain == "sol" else f"{name}@{chain}"] = {"line": cal.line, "trades": cal.trades, "mean": cal.mean, "total": cal.total, "hold_min": H,
                                                                "rule": "match", "teacher_trades": target, "chain": chain, "rows": int(len(ci))}
    return per, int(len(wi))


def bootstrap_matched(ds: DecisionSet, teacher: StackTeacher, fit: np.ndarray, epochs: int = EPOCHS, stop: threading.Event | None = None, seed: int = 0,
                      graph=None, device=None, teacher_note: str | None = None, checkpoint=None, match_days: int | None = None) -> tuple[FlyModel | None, dict]:
    """A matched fly (``matched_teacher``): taught by ``teacher`` on exactly ``fit`` -- the rows that stack was fit on --
    under the universe filter, each strategy only where its label existed (``train_fly(label_known=True)``), with no
    calibration week and no purge day of its own. Its lines are matched to the teacher's trade count on the same rows
    (``_match_lines``; ``match_days``: on the last N of those days only, since scoring every training candidate costs
    about one forward pass over them). It trades from the day after its last training day (``S``). Its evidence is set
    by the caller (``oos_report``: ``info['oos']``), which ``deployable`` reads; nothing here looks at a later day."""
    uni = selector.in_universe(ds.X, ds.cols); train = np.asarray(fit, bool) & uni
    if not train.any():
        raise RuntimeError("the teacher's fit rows hold no tradable minute")
    od = day_ords(ds)[train]; first, last = date.fromordinal(int(od.min())), date.fromordinal(int(od.max())); S = last + timedelta(days=1)
    fly, fit_info = train_fly(ds, train, teacher, epochs=epochs, stop=stop, seed=seed, graph=graph, device=device, checkpoint=checkpoint, label_known=True)
    if fly is None:
        return None, {"stopped": True, **fit_info}
    prog.update("fly: matching its buy lines to its teacher's trades on its training days", 0, 1, force=True)
    m_first = first if match_days is None else max(first, last - timedelta(days=int(match_days) - 1))
    m_rows = train if match_days is None else train & (day_ords(ds) >= m_first.toordinal())
    per, scored = _match_lines(ds, fly, teacher, m_rows)
    ti = np.flatnonzero(train)
    di = ti if len(ti) <= DIAG_ROWS else np.sort(np.random.default_rng(seed).choice(ti, DIAG_ROWS, replace=False))
    t_sc = teacher.targets(ds.X[di], ds.ts[di])[0][:, 0] if len(di) else np.zeros(0)
    ok_t = np.isfinite(t_sc); di, t_sc = di[ok_t], t_sc[ok_t]
    diag = diagnose(fly, ds.X[di], t_sc) if len(di) else {"mbon_share": 0.0, "mbon_saturated": 1.0, "kc_active": 0.0, "rows": 0}
    ok, why = gate_check(diag)
    fly.teacher = teacher          # not saved: the replay's match rule reads the teacher's own decisions over its rows
    info = {"mode": MATCHED, "S": str(S), "train_days": [str(first), str(last)], "train_through": str(last), "calibration_days": None,
            "line": fly.threshold, "sizing": fly.sizing, "lines": fly.lines, "chain_lines": fly.chain_lines,
            "line_fit": {"rule": "match", "days": [str(m_first), str(last)], "rows": scored, "per_strategy": per}, "calibration": None,
            "strategies": fly.strategies, "rules": fly.rules, "diagnostics": diag, "gates_ok": ok, "gate_failures": why, "fit": fit_info,
            "teacher": teacher_note or "a matched stack", "teacher_line": teacher.threshold, "teacher_stack": None}
    log.info("matched fly through %s: strategies %s, lines %s (matched on %d rows) | MBON share %.1f%% saturated %.0f%% | gates %s", last, fly.strategies,
             {k: round(v, 4) for k, v in fly.lines.items()}, scored, diag["mbon_share"] * 100, diag["mbon_saturated"] * 100, "ok" if ok else "; ".join(why))
    return fly, info


def _book(ds: DecisionSet, picks: np.ndarray, strat: np.ndarray, holds: dict, day0, pool: np.ndarray, trig: dict) -> dict:
    """One book's trades on the holdout (``strategies.score_holdout``'s rules: each pick on its strategy's hold and net
    label, one position per token across the book's strategies), combined and per strategy, each with its random picks:
    the combined book's among ``pool`` on the set's own hold (as score_holdout), a strategy's among its triggered rows of
    ``pool`` on its own hold. ``picks``: row indices; ``strat``: each pick's strategy."""
    from . import strategies
    from .decisions import random_trades, summarize
    n_all = len(ds.y); hold = np.zeros(n_all); ret = np.full(n_all, np.nan, np.float32); who = np.full(n_all, None, dtype=object)
    for name, H in holds.items():
        r = picks[strat == name]
        if len(r):
            hold[r] = H * 60.0; ret[r] = _hold_label(ds, H)[r]; who[r] = name
    tr = taken_idx(ds.ts, ds.mint, hold, picks) if len(picks) else np.zeros(0, int)
    out = {"all": {**strategies.score_trades(ds, tr, ret, day0).dict(), "random_mean": summarize(random_trades(ds, pool, max(1, len(tr)))).get("mean")}}
    for name, H in holds.items():
        ts_ = tr[who[tr] == name]
        rnd = summarize(random_trades(ds, pool & trig[name], max(1, len(ts_)), hold_s=H * 60.0, returns=_hold_label(ds, H)))
        out[name] = {**strategies.score_trades(ds, ts_, ret, day0).dict(), "random_mean": rnd.get("mean"), "hold_min": H}
    return out


def oos_report(ds: DecisionSet, models: dict, fly: FlyModel, days: list) -> dict:
    """The selector's and the fly's picks on ``days`` -- the stack's holdout, which neither the blind stack nor the fly
    matched to it was fit on -- per chain and per strategy, with the selector's own costs and hold rules (``_book``).
    ``selector``: the stack's decisions as it trades them (strategies.decide: triggers, lines, win filters, gates, dump
    veto). ``fly``: the fly's own decisions (``fly_decide``: its scores against its matched lines, on its teacher's
    triggers). With FLY_SCORE=teacher and FLY_TEACHER_GATE the live fly trades exactly the selector's picks until its
    learning vetoes some, so at bootstrap its live book is the selector's; what it then learns is the replay's to show.
    Trades, win rate, profit factor, mean and total net return per trade at fixed size, and the random baseline."""
    from . import strategies
    if not days:
        return {}
    days = sorted(days)
    pool_all = np.isin(day_ords(ds), [d.toordinal() for d in days]) & selector.in_universe(ds.X, ds.cols)
    rows = np.flatnonzero(pool_all); names = list((models.get("strategies") or {}))
    holds = {k: int(v["hold_min"]) for k, v in models["strategies"].items()}
    s_str = np.full(len(rows), None, dtype=object); s_ok = np.zeros(len(rows), bool); f_str = np.full(len(rows), None, dtype=object)
    tr_rows = {k: np.zeros(len(ds.y), bool) for k in names}
    for i in range(0, len(rows), MATCH_CHUNK):
        sl = rows[i:i + MATCH_CHUNK]; X = ds.X[sl]
        d = strategies.decide(models, X, ds.cols, ds.ts[sl]); s_str[i:i + MATCH_CHUNK] = d["strategy"]; s_ok[i:i + MATCH_CHUNK] = d["allow"]
        f_str[i:i + MATCH_CHUNK] = fly_decide(fly, X, ds.cols)["strategy"]
        tg = fly.triggers(X, ds.cols)
        for j, k in enumerate(fly.strategies):
            if k in tr_rows:
                tr_rows[k][sl] = tg[:, j]
    ch = ds.chains(); out = {"days": [str(days[0]), str(days[-1])], "rows": int(len(rows)), "strategies": holds,
                             "note": "fly = its own scores against its matched lines; at bootstrap the live fly (teacher score, teacher gate) trades the selector's picks",
                             "chains": {}}
    for chain in ["all"] + sorted(set(ch[rows].tolist()), key=lambda c: c != "sol"):
        cm = np.ones(len(rows), bool) if chain == "all" else ch[rows] == chain
        pool = pool_all if chain == "all" else pool_all & (ch == chain)
        s_pick = cm & s_ok; f_pick = cm & np.array([x is not None for x in f_str], bool)
        out["chains"][chain] = {"selector": _book(ds, rows[s_pick], s_str[s_pick], holds, days[0], pool, tr_rows),
                                "fly": _book(ds, rows[f_pick], f_str[f_pick], {k: h for k, h in holds.items() if k in fly.strategies}, days[0], pool, tr_rows)}
    return out


def verdict_version() -> dict:
    """What a replay verdict holds for: the bootstrap's definitions (``FLY_VERSION``) and how the fly scores a row
    (config.FLY_SCORE; "own" is what FLY_VERSION alone meant until 2026-10-11, so its verdicts stay valid for it)."""
    return FLY_VERSION if config.FLY_SCORE == "own" else {**FLY_VERSION, "score": config.FLY_SCORE}


def teacher_scores(models: dict, X: np.ndarray, ts, names: list[str], cols: list[str], chunk: int = 200_000) -> np.ndarray:
    """[B, S]: the teacher stack's own score of each row for each of the fly's strategies (its predicted net return; NaN
    where that strategy's trigger does not fire) -- the starting score of the fly when config.FLY_SCORE = teacher."""
    from . import strategies
    X = np.atleast_2d(X); out = np.full((len(X), len(names)), np.nan)
    tsa = np.broadcast_to(np.asarray(ts, dtype=np.float64), (len(X),))
    for i in range(0, len(X), chunk):
        out[i:i + chunk] = teacher_scores_of(strategies.decide_all(models, X[i:i + chunk], cols, tsa[i:i + chunk]), names, len(X[i:i + chunk]))
    return out


def teacher_scores_of(per: dict, names: list[str], n: int) -> np.ndarray:
    """``teacher_scores`` of ``n`` rows from the stack's ``strategies.decide_all`` result on them (no second pass)."""
    out = np.full((n, len(names)), np.nan)
    for j, k in enumerate(names):
        if k in per:
            sc = per[k]["score"]
            out[:, j] = np.where(per[k]["trig"] & np.isfinite(sc), np.clip(sc, -1.0, 1.0), np.nan)
    return out


def is_current(meta: dict | None, version: dict | None = None) -> bool:
    """Trained on exactly the current definitions, the retired corpus build versions set aside (``selector.defs_match``)."""
    return bool(meta) and selector.defs_match(meta.get("data"), version or FLY_VERSION)


def save(fly: FlyModel, metrics: dict, run_id: str | None = None, kind: str = "fly_selector", subdir: str = "policies", version: dict | None = None,
         prefix: str = "fly") -> tuple[Path, int]:
    """``kind`` / ``subdir`` / ``version`` / ``prefix``: another trading type's fly keeps its own snapshots apart (kalshi/fly.py)."""
    root = config.BRAIN_DIR / subdir; root.mkdir(parents=True, exist_ok=True)
    path = root / f"{prefix}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pt"
    n = fly.net
    model_io.torch_save({"state_dict": n.state_dict(), "scaler": fly.scaler.state(), "cols": fly.cols, "threshold": fly.threshold, "sizing": fly.sizing,
                "horizon_min": fly.horizon_min, "rules": fly.rules, "lines": fly.lines, "sizings": fly.sizings, "combine": fly.combine,
                "chain_lines": fly.chain_lines, "chain_sizings": fly.chain_sizings,
                "config": {"k_steps": n.k_steps, "leak": n.leak, "hidden": n.hidden, "obs_dim": n.obs_dim, "kc_active": n.kc_active, "scale": n.scale,
                           "n_strategies": n.n_strategies, "aff_rows": n.aff_rows.cpu().tolist() if kind != "fly_selector" else None}, "metrics": metrics}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction() as conn:
        row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                           (run_id, str(path), sha, kind, json.dumps({**metrics, "data": version or FLY_VERSION}, default=str))).fetchone()
    return path, int(row["id"])


def load(path: str | Path, graph=None, device=None, model_cls=None) -> FlyModel:
    d = model_io.torch_load(path); c = d["config"]
    net = FlyNet(graph if graph is not None else _graph(), obs_dim=c["obs_dim"], k_steps=c["k_steps"], leak=c["leak"], hidden=c["hidden"],
                 kc_active=c["kc_active"], device=device or _device(), n_strategies=c.get("n_strategies", 1), aff_rows=c.get("aff_rows"), scale=c.get("scale", SCALE))
    net.load_state_dict(d["state_dict"]); net.eval()
    for bn in (net.eff_norm, net.mb_norm):
        bn.momentum = None
    return (model_cls or FlyModel)(net, RobustScaler.from_state(d["scaler"]), d["cols"], d["horizon_min"], d["threshold"], d.get("sizing"), rules=d.get("rules"),
                                   lines=d.get("lines"), sizings=d.get("sizings"), combine=d.get("combine", "score"),
                                   chain_lines=d.get("chain_lines"), chain_sizings=d.get("chain_sizings"))


def latest_current(conn, kind: str = "fly_selector", version: dict | None = None) -> dict | None:
    """The newest fly bootstrap trained on the current definitions (``FLY_VERSION``) whose file exists, or None."""
    for r in conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC", (kind,)).fetchall():
        try:
            meta = json.loads(r["note"] or "{}")
        except ValueError:
            continue
        if is_current(meta, version) and Path(r["path"]).exists():
            return {**dict(r), "meta": meta}
    return None


def deployable(meta: dict | None, version: dict | None = None) -> tuple[bool, str]:
    """A bootstrap may race for the seat: current definitions, healthy network (``GATES``) and its own line made money on
    its calibration week over at least the race's own minimum of trades (``config.HANDOVER_MIN_TRADES``); a matched fly
    (``bootstrap_matched``) instead on its out-of-sample report (``info['oos']['evidence']``), over the same minimum. The wallet
    trades the selector's decisions until the fly wins its race (agent/fly_session.py), so a calibration week need not
    prove more than the race will: a quiet week (2026-10-15..21) gave a new fly 88 trades, under the 100 asked before.
    Another trading type (``version``: Kalshi) keeps ``selector.MIN_LINE_TRADES``."""
    if not is_current(meta, version):
        return False, "trained on other definitions"
    if not meta.get("gates_ok"):
        return False, "; ".join(meta.get("gate_failures") or ["network gates failed"])
    if meta.get("mode") == MATCHED:              # a matched fly's evidence: its own picks on the holdout (oos_report), then the race
        ev = (meta.get("oos") or {}).get("evidence") or {}; n = int(ev.get("trades") or 0); m = ev.get("mean")
        if n < config.HANDOVER_MIN_TRADES:
            return False, f"too few out-of-sample trades ({n})"
        if not (isinstance(m, (int, float)) and math.isfinite(m) and m > 0):
            return False, f"its out-of-sample picks lost money ({(m or 0) * 100:+.2f}% per trade)"
        return True, f"its own picks made {m * 100:+.2f}% per trade over {n} out-of-sample trades"
    cal = meta.get("calibration") or {}
    if (cal.get("trades") or 0) < (config.HANDOVER_MIN_TRADES if version in (None, FLY_VERSION) else selector.MIN_LINE_TRADES):
        return False, f"too few calibration trades ({cal.get('trades') or 0})"
    if not (isinstance(cal.get("mean"), (int, float)) and math.isfinite(cal["mean"]) and cal["mean"] > 0):      # NaN never passes
        return False, f"its calibration week lost money ({(cal.get('mean') or 0) * 100:+.2f}% per trade)"
    return True, f"its line made {cal['mean'] * 100:+.2f}% per trade over {cal['trades']} calibration trades"


def latest_deployable(conn, kind: str = "fly_selector", version: dict | None = None) -> dict | None:
    """The newest bootstrap that may trade (``deployable``), or None."""
    for r in conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC", (kind,)).fetchall():
        try:
            meta = json.loads(r["note"] or "{}")
        except ValueError:
            continue
        if deployable(meta, version)[0] and Path(r["path"]).exists():
            return {**dict(r), "meta": meta}
    return None


def main(days: int | None = None, epochs: int | None = None, stop_event: threading.Event | None = None) -> dict:
    """Bootstrap a fly to trade from tomorrow on: taught by the selector the live book trades, on every corpus day but
    the last eight, calibrated on the last seven. ``epochs``: a cap; None leaves the count to the held-out slice."""
    from ..ops.reset import reset_training_stats
    reset_training_stats("fly_selector", reason="fly bootstrap")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("fly: building decision points", 0, 1, force=True)
    from .strategies import HOLDS_MIN
    ds = build(days=days, horizon_min=HOLD_MIN, holds=HOLDS_MIN); S = ds.days[-1] + timedelta(days=1)
    teacher, note, taught_by = deployed_teacher(ds, ds.day < S - timedelta(days=CALIB_DAYS + PURGE_DAYS), calib_start=S - timedelta(days=CALIB_DAYS))
    log.info("fly bootstrap teacher: %s", note)
    fly, info = bootstrap(ds, S, epochs=EPOCHS if epochs is None else epochs, stop=stop_event, teacher=teacher, teacher_note=note, checkpoint=CHECKPOINT)
    if fly is None:
        return {"stopped": True}                     # its checkpoint stays: the next run on the same data resumes it
    info["teacher_snapshot"] = taught_by
    record_event("info", "fly_selector", "fly bootstrap", {k: info.get(k) for k in ("S", "lines", "calibration", "diagnostics", "gates_ok", "gate_failures", "teacher")})
    path, sid = save(fly, info)
    CHECKPOINT.unlink(missing_ok=True)
    prog.update("fly: done", 1, 1, force=True, snapshot_id=sid, line=info["line"], gates_ok=info["gates_ok"], diagnostics=info["diagnostics"])
    return {**info, "snapshot_id": sid, "path": str(path)}
