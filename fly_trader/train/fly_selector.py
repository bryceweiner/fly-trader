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

Bootstrap (``bootstrap``, the one time the selector teaches): the selector is fit on the days before ``S − 8`` (one
purge day before the calibration week), the fly imitates it on the tradable minutes of those days (the teacher's top
``DISTIL_TOP`` plus a random ``DISTIL_SAMPLE``; squared error weighted ``TOP_WEIGHT`` × on the teacher's top 1 %), both
normalisations are recomputed over ``RECAL_ROWS`` training rows and frozen, and the fly's own buy line and sizing bands
are calibrated (``train/fly_calibrate.py``) on its scores for the seven days before ``S`` whose labels were known by
then — never the selector's line. Reports: the MBONs' share of the score variance, MBON saturation, the slope and rank
correlation of the fly's scores against the teacher's. Snapshots: ``data/brain/policies/fly_<ts>.pt`` + ``brain_snapshots``
kind 'fly_selector'.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fn

from .. import config
from ..brain.connectome import AFFERENT_POPS, EFFERENT_POP, Connectome, SubConnectome
from ..db.apilog import record_event
from ..db.connection import transaction
from . import fly_calibrate
from . import progress as prog
from . import selector
from .decisions import HOLD_MIN, DecisionSet, build, rank_corr
from .scaling import RobustScaler

log = logging.getLogger(__name__)

SCALE = 20.0                 # the head predicts the net return × 20 (±5 % ≈ ±1)
K_STEPS, LEAK, HIDDEN = 4, 0.5, 128
KC_ACTIVE = 0.10             # share of Kenyon cells left active at every step
KM_INIT_SCALE = 0.006        # KC→MBON weight per synapse: ~22 active KC inputs × ~8 synapses ≈ 1, MBONs unsaturated
C_INIT = 0.1                 # MBON readout coefficient at start (SCALE units per standard deviation of an MBON)
EPOCHS, BATCH = 1, 512
TOP_WEIGHT = 10.0            # loss weight of the teacher's top 1 % of training minutes
DISTIL_TOP, DISTIL_SAMPLE = 0.05, 3_000_000
RECAL_ROWS = 200_000
LR_GRAPH, LR_HEAD = 1e-4, 1e-3
PURGE_DAYS, CALIB_DAYS = 1, 7
DIAG_ROWS = 50_000
GATES = {"mbon_share_min": 0.10, "mbon_saturated_max": 0.20, "slope_min": 0.6, "slope_max": 1.4}
TEACHER_NOTE = "selector fit on the fly's training days, same configuration"
# the definitions a fly was trained on (the selector's, as its teacher, + the fly's network and plasticity design)
FLY_VERSION = {**selector.DATA_VERSION, "fly": "flynet-2-mb", "plastic": "mb3f-1"}


def _inv_softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.log(torch.expm1(x.clamp(min=1e-6)))


class FlyNet(nn.Module):
    """Scorer on the connectome: features → afferent neurons → K steps along the synapses and the KC→MBON block →
    normalised descending neurons → decoder, + signed readout of the normalised MBONs → predicted net return (× ``SCALE``)."""

    def __init__(self, graph, obs_dim: int, k_steps: int = K_STEPS, leak: float = LEAK, hidden: int = HIDDEN, kc_active: float = KC_ACTIVE, device=None):
        super().__init__()
        dev = torch.device(device) if device else graph.device
        self.N, self.k_steps, self.leak, self.obs_dim, self.hidden, self.kc_active = graph.N, k_steps, leak, obs_dim, hidden, kc_active
        self.register_buffer("pre", graph.indices[1].to(dev)); self.register_buffer("post", graph.indices[0].to(dev))
        vals = graph.values_raw.float(); self.register_buffer("sign", torch.sign(vals).to(dev))
        s = float(getattr(graph, "s", 0.0) or 0.0) or 0.99 / float(getattr(graph, "spectral_radius_raw", None) or 2788.0)
        self.theta = nn.Parameter(_inv_softplus(vals.abs() * s).to(dev))                          # softplus^-1(|w|)
        self.gain = nn.Parameter(torch.ones(self.N, device=dev)); self.bias = nn.Parameter(torch.zeros(self.N, device=dev))
        aff = [torch.arange(*graph.pop_ranges[p]) for p in AFFERENT_POPS if p in graph.pop_ranges]
        self.register_buffer("aff_rows", torch.cat(aff).to(dev)); self.register_buffer("eff_rows", torch.arange(*graph.pop_ranges[EFFERENT_POP]).to(dev))
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
        self.head = nn.Linear(hidden, 1).to(dev)
        self.dev = dev

    @property
    def c(self) -> torch.Tensor:
        """MBON readout coefficients: approach +, avoid −, the others free."""
        return torch.where(self.c_sign != 0, self.c_sign * Fn.softplus(self.rho), self.c_free)

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

    def forward_parts(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(descending-neuron output [B] × SCALE, MBON pre-activation at the last step [B, n_MBON], KC code [B, n_KC])."""
        h, k, u = self._propagate(x)
        y_dn = self.head(self.dec(self.eff_norm(h.index_select(0, self.eff_rows).T))).squeeze(-1)
        return y_dn, u.T, k.T

    def readout(self, y_dn: torch.Tensor, u0: torch.Tensor, k: torch.Tensor, D: torch.Tensor | None = None) -> torch.Tensor:
        """Prediction × SCALE from the parts; ``D`` [n_KC, n_MBON] is a plastic change of the KC→MBON weights at the last step."""
        u = u0 if D is None else u0 + self.gain[self.mb0:self.mb1] * (k @ (D * self.mask))
        return y_dn + self.mb_norm(torch.tanh(u)) @ self.c

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.readout(*self.forward_parts(x))

    def mbon_stats(self) -> tuple[torch.Tensor, torch.Tensor]:
        """The frozen MBON normalisation (mean, standard deviation) the readout uses in eval mode."""
        return self.mb_norm.running_mean, torch.sqrt(self.mb_norm.running_var + self.mb_norm.eps)

    def param_groups(self, lr_graph: float, lr_head: float) -> list[dict]:
        graph = [self.theta, self.gain, self.bias, self.theta_km]; ids = {id(p) for p in graph}
        return [{"params": graph, "lr": lr_graph}, {"params": [p for p in self.parameters() if id(p) not in ids], "lr": lr_head}]


class FlyModel:
    """A trained fly as a trading model: the interface of ``selector.SelectorModel`` (score, universe, threshold, sizing)
    plus ``parts`` for the plasticity rule. ``threshold`` and ``sizing`` are the fly's own calibration."""

    def __init__(self, net: FlyNet, scaler: RobustScaler, cols: list[str], horizon_min: int, threshold: float | None = None, sizing: list | None = None):
        self.net, self.scaler, self.cols, self.horizon_min = net, scaler, list(cols), horizon_min
        self.threshold = threshold if threshold is not None else selector.MIN_EV
        self.sizing = list(sizing or [])

    def _x(self, X: np.ndarray) -> torch.Tensor:
        return torch.tensor(self.scaler.transform(X), device=self.net.dev)

    @torch.no_grad()
    def score(self, X: np.ndarray, batch: int = 2048) -> np.ndarray:
        """Predicted net return over the hold (the frozen fly)."""
        self.net.eval(); out = np.empty(len(X), np.float64)
        for i in range(0, len(X), batch):
            out[i:i + batch] = (self.net(self._x(X[i:i + batch])) / SCALE).float().cpu().numpy()
        return out

    @torch.no_grad()
    def parts(self, X: np.ndarray) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """``FlyNet.forward_parts`` on raw feature rows (one batch; eval mode)."""
        self.net.eval()
        return self.net.forward_parts(self._x(X))

    def universe(self, X: np.ndarray) -> np.ndarray:
        return selector.in_universe(X, self.cols)


def _device() -> torch.device:
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


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


def train_fly(ds: DecisionSet, rows: np.ndarray, teacher, epochs: int = EPOCHS, batch: int = BATCH, stop: threading.Event | None = None,
              seed: int = 0, graph=None, device=None) -> tuple[FlyModel | None, dict]:
    """Distil ``teacher`` (anything with ``score`` and ``scaler``) into a FlyNet over the training rows ``rows`` (mask)."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed); dev = torch.device(device) if device else _device()
    net = FlyNet(graph if graph is not None else _graph(), obs_dim=len(ds.cols), device=dev)
    idx = np.flatnonzero(rows); tgt = np.empty(len(idx), np.float32)
    for i in range(0, len(idx), 500_000):
        tgt[i:i + 500_000] = teacher.score(ds.X[idx[i:i + 500_000]])
    tgt = np.clip(tgt, -1.0, 1.0); wts = np.where(tgt >= np.quantile(tgt, 0.99), TOP_WEIGHT, 1.0).astype(np.float32)
    use = distil_rows(idx, tgt, rng)
    with torch.no_grad():
        net.head.bias.fill_(float(tgt.mean()) * SCALE); net.head.weight.mul_(0.1)
    opt = torch.optim.Adam(net.param_groups(LR_GRAPH, LR_HEAD)); n_b = max(1, len(use) // batch); total = max(1, epochs * n_b); hist = []; t0 = time.time()
    for ep in range(epochs):
        perm = use[rng.permutation(len(use))]; tot = 0.0; net.train()
        for b in range(n_b):
            if stop is not None and stop.is_set():
                return None, {"epochs": hist, "stopped": True}
            sel = perm[b * batch:(b + 1) * batch]
            x = torch.tensor(teacher.scaler.transform(ds.X[idx[sel]]), device=dev)
            t = torch.tensor(tgt[sel] * SCALE, device=dev); w = torch.tensor(wts[sel], device=dev)
            loss = (w * (net(x) - t) ** 2).sum() / w.sum()
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0); opt.step(); tot += loss.item()
            if b % 200 == 0:
                done = ep * n_b + b + 1; el = time.time() - t0
                prog.update("fly: learning from the selector", done, total, epoch=ep, loss=tot / (b + 1), rows_per_s=done * batch / max(el, 1e-9),
                            eta_s=(total - done) * el / done)
        hist.append({"epoch": ep, "mse": tot / n_b, "secs": time.time() - t0, "rows": min(len(use), n_b * batch)})
        log.info("fly epoch %d: weighted mse %.4f over %d rows (%.0fs)", ep, tot / n_b, min(len(use), n_b * batch), time.time() - t0)
    rec = idx[rng.choice(len(idx), min(RECAL_ROWS, len(idx)), replace=False)]
    recalibrate(net, ds.X[np.sort(rec)], teacher.scaler, batch)
    return FlyModel(net, teacher.scaler, ds.cols, int(ds.horizon_s // 60)), {"epochs": hist, "stopped": False, "rows": int(len(use)),
                                                                              "training_rows": int(len(idx)), "top_weight": TOP_WEIGHT}


@torch.no_grad()
def diagnose(fly: FlyModel, X: np.ndarray, teacher_scores: np.ndarray | None = None, batch: int = 2048) -> dict:
    """How much of the score runs through the mushroom body, how saturated the MBONs are, how sparse the KC code is,
    and (given the teacher's scores) how the fly's scores line up with them."""
    net = fly.net; net.eval(); mb, tot, sat, act = [], [], [], []
    for i in range(0, len(X), batch):
        y_dn, u0, k = fly.parts(X[i:i + batch])
        m = net.mb_norm(torch.tanh(u0)) @ net.c
        mb.append(m.cpu()); tot.append((y_dn + m).cpu()); sat.append(u0.abs().cpu()); act.append((k > 0).float().mean(1).cpu())
    mb, tot, sat, act = torch.cat(mb).double(), torch.cat(tot).double(), torch.cat(sat), torch.cat(act)
    out = {"mbon_share": float(mb.var() / tot.var()) if tot.var() > 0 else 0.0, "mbon_saturated": float((sat.median(0).values > 2.0).float().mean()),
           "kc_active": float(act.mean()), "rows": int(len(X))}
    if teacher_scores is not None:
        s = (tot / SCALE).numpy()
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


def bootstrap(ds: DecisionSet, S: date, epochs: int = EPOCHS, stop: threading.Event | None = None, seed: int = 0, graph=None,
              device=None) -> tuple[FlyModel | None, dict]:
    """The one time the selector teaches: a fly ready to trade from day ``S`` on, with its own line and sizing."""
    train_end = S - timedelta(days=CALIB_DAYS + PURGE_DAYS)
    train = ds.day < train_end; uni = selector.in_universe(ds.X, ds.cols)
    if not train.any():
        raise RuntimeError(f"no training days before {train_end}")
    prog.update("fly: fitting its teacher (the selector, same configuration)", 0, 1, force=True)
    teacher = selector.fit(ds, train, seed=7)
    fly, fit_info = train_fly(ds, train & uni, teacher, epochs=epochs, stop=stop, seed=seed, graph=graph, device=device)
    if fly is None:
        return None, {"stopped": True}
    s_epoch = datetime(S.year, S.month, S.day, tzinfo=timezone.utc).timestamp()
    calib = (ds.day >= S - timedelta(days=CALIB_DAYS)) & (ds.day < S) & uni & (ds.ts + ds.horizon_s + 60.0 <= s_epoch)   # labels known at S
    ci = np.flatnonzero(calib)
    prog.update("fly: calibrating its own buy line", 0, 1, force=True)
    sc = fly.score(ds.X[ci]) if len(ci) else np.array([])
    cal = fly_calibrate.calibrate(ds.ts[ci], ds.mint[ci], ds.horizon_s, sc, ds.fwd_pess[ci])
    fly.threshold, fly.sizing = cal.line, cal.sizing
    di = ci if len(ci) <= DIAG_ROWS else np.sort(np.random.default_rng(seed).choice(ci, DIAG_ROWS, replace=False))
    diag = diagnose(fly, ds.X[di], teacher.score(ds.X[di])) if len(di) else {"mbon_share": 0.0, "mbon_saturated": 1.0, "kc_active": 0.0, "rows": 0}
    ok, why = gate_check(diag)
    info = {"S": str(S), "train_through": str(train_end - timedelta(days=1)), "calibration_days": [str(S - timedelta(days=CALIB_DAYS)), str(S - timedelta(days=1))],
            "line": cal.line, "sizing": cal.sizing, "calibration": {"trades": cal.trades, "mean": cal.mean, "total": cal.total, "lines": cal.lines},
            "diagnostics": diag, "gates_ok": ok, "gate_failures": why, "fit": fit_info, "teacher": TEACHER_NOTE, "teacher_line": teacher.threshold}
    log.info("fly bootstrap for %s: line %+.2f%% (%d calibration trades, mean %s) | MBON share %.1f%% saturated %.0f%% KC active %.1f%% slope %s | gates %s",
             S, cal.line * 100, cal.trades, f"{cal.mean * 100:+.2f}%" if cal.mean is not None else "-", diag["mbon_share"] * 100, diag["mbon_saturated"] * 100,
             diag["kc_active"] * 100, f"{diag.get('slope'):.2f}" if diag.get("slope") is not None else "-", "ok" if ok else "; ".join(why))
    return fly, info


def is_current(meta: dict | None) -> bool:
    return bool(meta) and meta.get("data") == FLY_VERSION


def save(fly: FlyModel, metrics: dict, run_id: str | None = None) -> tuple[Path, int]:
    root = config.BRAIN_DIR / "policies"; root.mkdir(parents=True, exist_ok=True)
    path = root / f"fly_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.pt"
    n = fly.net
    torch.save({"state_dict": n.state_dict(), "scaler": fly.scaler.state(), "cols": fly.cols, "threshold": fly.threshold, "sizing": fly.sizing,
                "horizon_min": fly.horizon_min, "config": {"k_steps": n.k_steps, "leak": n.leak, "hidden": n.hidden, "obs_dim": n.obs_dim,
                                                           "kc_active": n.kc_active, "scale": SCALE}, "metrics": metrics}, path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    with transaction() as conn:
        row = conn.execute("INSERT INTO brain_snapshots (run_id, path, sha256, kind, note) VALUES (%s,%s,%s,'fly_selector',%s) RETURNING id",
                           (run_id, str(path), sha, json.dumps({**metrics, "data": FLY_VERSION}, default=str))).fetchone()
    return path, int(row["id"])


def load(path: str | Path, graph=None, device=None) -> FlyModel:
    d = torch.load(path, map_location="cpu", weights_only=False); c = d["config"]
    net = FlyNet(graph if graph is not None else _graph(), obs_dim=c["obs_dim"], k_steps=c["k_steps"], leak=c["leak"], hidden=c["hidden"],
                 kc_active=c["kc_active"], device=device or _device())
    net.load_state_dict(d["state_dict"]); net.eval()
    for bn in (net.eff_norm, net.mb_norm):
        bn.momentum = None
    return FlyModel(net, RobustScaler.from_state(d["scaler"]), d["cols"], d["horizon_min"], d["threshold"], d.get("sizing"))


def latest_current(conn) -> dict | None:
    """The newest fly bootstrap trained on the current definitions (``FLY_VERSION``) whose file exists, or None."""
    for r in conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = 'fly_selector' ORDER BY id DESC").fetchall():
        try:
            meta = json.loads(r["note"] or "{}")
        except ValueError:
            continue
        if is_current(meta) and Path(r["path"]).exists():
            return {**dict(r), "meta": meta}
    return None


def deployable(meta: dict | None) -> tuple[bool, str]:
    """A bootstrap may trade: current definitions, healthy network (``GATES``) and its own line made money over at least
    100 trades on its calibration week."""
    if not is_current(meta):
        return False, "trained on other definitions"
    if not meta.get("gates_ok"):
        return False, "; ".join(meta.get("gate_failures") or ["network gates failed"])
    cal = meta.get("calibration") or {}
    if (cal.get("trades") or 0) < selector.MIN_LINE_TRADES:
        return False, f"too few calibration trades ({cal.get('trades') or 0})"
    if (cal.get("mean") or 0) <= 0:
        return False, f"its calibration week lost money ({(cal.get('mean') or 0) * 100:+.2f}% per trade)"
    return True, f"its line made {cal['mean'] * 100:+.2f}% per trade over {cal['trades']} calibration trades"


def latest_deployable(conn) -> dict | None:
    """The newest bootstrap that may trade (``deployable``), or None."""
    for r in conn.execute("SELECT id, path, note FROM brain_snapshots WHERE kind = 'fly_selector' ORDER BY id DESC").fetchall():
        try:
            meta = json.loads(r["note"] or "{}")
        except ValueError:
            continue
        if deployable(meta)[0] and Path(r["path"]).exists():
            return {**dict(r), "meta": meta}
    return None


def main(days: int | None = None, epochs: int = EPOCHS, stop_event: threading.Event | None = None) -> dict:
    """Bootstrap a fly to trade from tomorrow on: taught on every corpus day but the last eight, calibrated on the last seven."""
    from ..ops.reset import reset_training_stats
    reset_training_stats("fly_selector", reason="fly bootstrap")
    prog.set_stop_event(stop_event); prog.clear()
    prog.update("fly: building decision points", 0, 1, force=True)
    ds = build(days=days, horizon_min=HOLD_MIN); S = ds.days[-1] + timedelta(days=1)
    fly, info = bootstrap(ds, S, epochs=epochs, stop=stop_event)
    if fly is None:
        return {"stopped": True}
    record_event("info", "fly_selector", "fly bootstrap", {k: info[k] for k in ("S", "line", "calibration", "diagnostics", "gates_ok", "gate_failures")})
    path, sid = save(fly, info)
    prog.update("fly: done", 1, 1, force=True, snapshot_id=sid, line=info["line"], gates_ok=info["gates_ok"], diagnostics=info["diagnostics"])
    return {**info, "snapshot_id": sid, "path": str(path)}
