"""Batched discrete leaky integrate-and-fire over the FAFB v783 connectome on torch (MPS or CPU).

One brain, B columns: column b is the same fly smelling slot b's token. Per tick (flybrain
``js/sim-worker.js`` dynamics, batched; Shiu et al. 2024 Nature 634:210 for the sign convention):

    I_syn      = W @ fired                      fixed connectome (sparse COO on MPS, CSR on CPU)
    I_syn[mbon] += W_KM^T @ fired[kc]           plastic KC->MBON path (dense [n_KC, n_MBON])
    active     = refr == 0                      refractory neurons hold their potential
    V          = where(active, leak*V + I_ext + I_syn (+ noise), V)
    refr       = max(refr - 1, 0)
    fire       = active & (V >= theta)
    fire[kc]   &= top-k(V[kc]) per column       k-winners-take-all (Dasgupta, Stevens, Navlakha 2017)
    V[fire] = 0; refr[fire] = refractory; fired = fire
    spike_sum += fired over the last readout_window ticks

Readout (plan section 4): population mean rates through one indicator matmul, KC code = spikes per
readout tick, m_hat = (rho_app - rho_av) / (rho_app + rho_av + 0.05).
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import warnings

import numpy as np
import torch

from .. import config

warnings.filterwarnings("ignore", message="Sparse CSR tensor support is in beta state")
warnings.filterwarnings("ignore", message="Sparse invariant checks are implicitly disabled")

CONNECTOME_DIR = config.BRAIN_DIR / "connectome"
CURRENT_FILE = CONNECTOME_DIR / "current.txt"
SCALE_FILE = CONNECTOME_DIR / "scale.json"
W_CLIP = 1.0        # = theta
M_HAT_EPS = 0.05    # plan section 4
MBON_BLOCK = ("MBON_APP", "MBON_AV", "MBON_OTHER")


def resolve_device(name: str | None = None) -> torch.device:
    """config.DEVICE with automatic fallback to CPU when MPS (or CUDA) is unavailable."""
    name = (name or config.DEVICE or "cpu").lower()
    if name.startswith("mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[lif] MPS unavailable; falling back to cpu", file=sys.stderr)
        return torch.device("cpu")
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        print("[lif] CUDA unavailable; falling back to cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def current_connectome_path() -> Path:
    if not CURRENT_FILE.exists():
        raise FileNotFoundError(f"{CURRENT_FILE} missing: run `fly-trader build-connectome` first")
    name = CURRENT_FILE.read_text().strip()
    path = Path(name) if Path(name).is_absolute() else CONNECTOME_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"connectome {path} named in current.txt does not exist")
    return path


def read_scale_file(connectome_file: str, path: Path = SCALE_FILE) -> dict | None:
    """Calibrated scales if scale.json exists and refers to this artifact (by file name or content hash)."""
    if not path.exists():
        return None
    try:
        d = json.loads(path.read_text())
    except Exception:
        return None
    if d.get("connectome") not in (connectome_file, Path(connectome_file).name):
        return None
    return d


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class Connectome:
    """The fixed connectome resident on the device plus population bookkeeping.

    ``W`` is the sparse [N, N] matrix (row = post, col = pre) with values clip(s * values_raw,
    -w_clip, w_clip); ``W_KM0`` the dense plastic block initialised as clip(s_km * W_KM0_raw, 0, w_max).
    ``scale(s)`` / ``scale_km(s_km)`` rescale in place from the raw synapse counts.
    """

    def __init__(self, path: str | Path | None = None, device: torch.device | str | None = None,
                 s: float | None = None, s_km: float | None = None, use_scale_file: bool = True):
        path = Path(path) if path else current_connectome_path()
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.sha256 = _sha256(path)
        self.content_sha256 = str(z["content_sha256"])
        self.s0 = float(z["s0"])
        self.spectral_radius_raw = float(z["spectral_radius_raw"])
        self.nt_sign_mode = str(z["nt_sign_mode"])
        self.build_config = json.loads(str(z["build_config"]))
        self.source_sha256 = json.loads(str(z["source_sha256"]))
        self.root_ids = z["root_ids"]
        self.nt = z["nt"]
        self.flybrain_group = z["flybrain_group"]
        self.flybrain_group_names = [str(x) for x in z["flybrain_group_names"]]
        self.cell_type = z["cell_type"] if "cell_type" in z.files else None
        scale = read_scale_file(path.name) if use_scale_file else None
        self.scale_file = scale
        if s is None:
            s = float(scale["s"]) if scale and "s" in scale else self.s0
        if s_km is None:
            s_km = float(scale["s_km"]) if scale and "s_km" in scale else self.s0
        self._init(
            indices=z["indices"], values_raw=z["values_raw"], N=int(z["N"]),
            pop_order=[str(x) for x in z["pop_order"]], pop_ranges={k: tuple(v) for k, v in json.loads(str(z["pop_ranges"])).items()},
            W_KM0_raw=z["W_KM0"], M_KM=z["M_KM"], glomerulus_of_orn=z["glomerulus_of_orn"],
            glomerulus_names=[str(x) for x in z["glomerulus_names"]], device=device, s=s, s_km=s_km,
            w_clip=float(z["w_clip"]) if "w_clip" in z.files else W_CLIP, coalesced=True,
        )
        if scale and scale.get("pathway_gains"):
            self.apply_pathway_gains(scale["pathway_gains"])

    @classmethod
    def load(cls, path: str | Path | None = None, device: torch.device | str | None = None, **kw) -> "Connectome":
        """The artifact named in data/brain/connectome/current.txt (or ``path``), scales from scale.json."""
        return cls(path, device=device, **kw)

    @classmethod
    def from_arrays(cls, indices, values_raw, N: int, pop_ranges: dict[str, tuple[int, int]],
                    pop_order: list[str] | None = None, W_KM0=None, M_KM=None, glomerulus_of_orn=None,
                    glomerulus_names: list[str] | None = None, device=None, s: float = 1.0, s_km: float = 1.0,
                    w_clip: float = W_CLIP) -> "Connectome":
        """Explicit tensors (tests, toys). Populations absent from ``pop_ranges`` are empty."""
        self = cls.__new__(cls)
        self.path = None
        self.sha256 = None
        self.content_sha256 = None
        self.s0 = s
        self.spectral_radius_raw = float("nan")
        self.nt_sign_mode = "explicit"
        self.build_config = {}
        self.source_sha256 = {}
        self.root_ids = np.arange(N, dtype=np.int64)
        self.nt = np.array([""] * N)
        self.flybrain_group = np.zeros(N, dtype=np.uint16)
        self.flybrain_group_names = []
        self.cell_type = None
        self.scale_file = None
        from .populations import POPULATION_ORDER
        order = list(pop_order) if pop_order else list(POPULATION_ORDER)
        ranges = dict(pop_ranges)
        cursor = max([b for _, b in ranges.values()], default=0)
        for name in order:
            if name not in ranges:
                ranges[name] = (cursor, cursor)
        indices = np.asarray(indices, dtype=np.int64).reshape(2, -1)
        values_raw = np.asarray(values_raw, dtype=np.float32).reshape(-1)
        n_kc = ranges["KC"][1] - ranges["KC"][0]
        n_mbon = sum(ranges[n][1] - ranges[n][0] for n in MBON_BLOCK if n in ranges)
        if W_KM0 is None:
            W_KM0 = np.zeros((n_kc, n_mbon), dtype=np.float32)
        W_KM0 = np.asarray(W_KM0, dtype=np.float32)
        if M_KM is None:
            M_KM = W_KM0 != 0
        if glomerulus_of_orn is None:
            a, b = ranges["ORN_FOOD"]
            glomerulus_of_orn = np.zeros(b - a, dtype=np.int32)
        self._init(indices=indices, values_raw=values_raw, N=N, pop_order=order, pop_ranges=ranges,
                   W_KM0_raw=W_KM0, M_KM=np.asarray(M_KM, dtype=bool), glomerulus_of_orn=np.asarray(glomerulus_of_orn, dtype=np.int32),
                   glomerulus_names=glomerulus_names or [], device=device, s=s, s_km=s_km, w_clip=w_clip, coalesced=False)
        return self

    # ------------------------------------------------------------------------------------------
    def _init(self, *, indices, values_raw, N, pop_order, pop_ranges, W_KM0_raw, M_KM, glomerulus_of_orn,
              glomerulus_names, device, s, s_km, w_clip, coalesced):
        self.device = device if isinstance(device, torch.device) else resolve_device(device)
        self.N = int(N)
        self.w_clip = float(w_clip)
        self.pop_order = list(pop_order)
        self.pop_ranges = {k: (int(a), int(b)) for k, (a, b) in pop_ranges.items()}
        self.pop_slices = {k: slice(a, b) for k, (a, b) in self.pop_ranges.items()}
        self.pop_id = {k: i for i, k in enumerate(self.pop_order)}
        self.P = len(self.pop_order)
        # sparse fixed connectome: keep raw values on CPU, indices on CPU (int64) for rescaling
        idx = torch.from_numpy(np.ascontiguousarray(indices)).to(torch.int64)
        vals = torch.from_numpy(np.ascontiguousarray(values_raw)).to(torch.float32)
        if not coalesced:
            t = torch.sparse_coo_tensor(idx, vals, (self.N, self.N)).coalesce()
            idx, vals = t.indices().clone(), t.values().clone()
        self.indices = idx
        self.values_raw = vals
        # per-edge pathway gain (CPU); default 1. Lets one pathway (e.g. PN->KC claw synapses) be
        # strengthened relative to the uniform scale s. Persisted in scale.json as pathway_gains.
        self.edge_gain = torch.ones_like(vals)
        self.pathway_gains: dict[str, float] = {}
        self.nnz = int(vals.numel())
        self.s = float(s)
        self.W = self._build_W(self.s)
        # populations of interest
        self.kc_slice = self.pop_slices["KC"]
        self.n_KC = self.kc_slice.stop - self.kc_slice.start
        m0 = self.pop_ranges["MBON_APP"][0]
        m1 = self.pop_ranges["MBON_OTHER"][1]
        assert self.pop_ranges["MBON_APP"][1] == self.pop_ranges["MBON_AV"][0] and \
            self.pop_ranges["MBON_AV"][1] == self.pop_ranges["MBON_OTHER"][0], "MBON block must be contiguous"
        self.mbon_slice = slice(m0, m1)
        self.n_MBON = m1 - m0
        self.mbon_app_slice = self.pop_slices["MBON_APP"]
        self.mbon_av_slice = self.pop_slices["MBON_AV"]
        self.dan_pam_slice = self.pop_slices["DAN_PAM"]
        self.dan_ppl1_slice = self.pop_slices["DAN_PPL1"]
        self.orn_food_slice = self.pop_slices["ORN_FOOD"]
        self.pop_index = {k: torch.arange(a, b, dtype=torch.int64, device=self.device) for k, (a, b) in self.pop_ranges.items()}
        self.kc_idx = self.pop_index["KC"]
        self.mbon_app_idx = self.pop_index["MBON_APP"]
        self.mbon_av_idx = self.pop_index["MBON_AV"]
        self.dan_pam_idx = self.pop_index["DAN_PAM"]
        self.dan_ppl1_idx = self.pop_index["DAN_PPL1"]
        # plastic block
        W_KM0_raw = np.asarray(W_KM0_raw, dtype=np.float32)
        assert W_KM0_raw.shape == (self.n_KC, self.n_MBON), (W_KM0_raw.shape, self.n_KC, self.n_MBON)
        self.W_KM0_raw = torch.from_numpy(np.ascontiguousarray(W_KM0_raw))
        self.M_KM = torch.from_numpy(np.ascontiguousarray(np.asarray(M_KM, dtype=bool))).to(self.device)
        self.has_plastic = self.n_KC > 0 and self.n_MBON > 0
        self.s_km = float(s_km)
        self.w_max = 0.0
        self.W_KM0 = None
        self.scale_km(self.s_km)
        # MBON valence signs for plasticity: +1 approach, -1 avoid, 0 unassigned (relative to the MBON block)
        sign = torch.zeros(self.n_MBON, dtype=torch.float32)
        sign[self.mbon_app_slice.start - m0: self.mbon_app_slice.stop - m0] = 1.0
        sign[self.mbon_av_slice.start - m0: self.mbon_av_slice.stop - m0] = -1.0
        self.mbon_sign = sign.to(self.device)
        # column indices into the MBON dimension of W_KM0 (approach / avoidance) for the plasticity rule
        self.mbon_app_cols = torch.arange(self.mbon_app_slice.start - m0, self.mbon_app_slice.stop - m0,
                                          dtype=torch.int64, device=self.device)
        self.mbon_av_cols = torch.arange(self.mbon_av_slice.start - m0, self.mbon_av_slice.stop - m0,
                                         dtype=torch.int64, device=self.device)
        # glomeruli
        g = np.asarray(glomerulus_of_orn, dtype=np.int64)
        self.glomerulus_names = list(glomerulus_names)
        self.n_glomeruli = max(len(self.glomerulus_names), int(g.max()) + 1 if g.size else 0)
        self.glomerulus_of_orn = np.asarray(glomerulus_of_orn, dtype=np.int32)      # over ORN_FOOD rows, -1 unknown
        g = np.where(g < 0, self.n_glomeruli, g)                                       # unknown -> extra zero row
        self.glomerulus_of_orn_dev = torch.from_numpy(g).to(self.device)
        # population indicator [P, N] with 1/n entries -> mean rates
        rows, cols, vals_g = [], [], []
        for p, name in enumerate(self.pop_order):
            a, b = self.pop_ranges[name]
            if b > a:
                rows.append(np.full(b - a, p, dtype=np.int64))
                cols.append(np.arange(a, b, dtype=np.int64))
                vals_g.append(np.full(b - a, 1.0 / (b - a), dtype=np.float32))
        if rows:
            gi = torch.from_numpy(np.stack([np.concatenate(rows), np.concatenate(cols)]))
            gv = torch.from_numpy(np.concatenate(vals_g))
        else:
            gi = torch.zeros((2, 0), dtype=torch.int64)
            gv = torch.zeros(0, dtype=torch.float32)
        G = torch.sparse_coo_tensor(gi, gv, (self.P, self.N)).coalesce()
        self.group_matrix = self._to_device_sparse(G)
        self.pop_sizes = torch.tensor([self.pop_ranges[n][1] - self.pop_ranges[n][0] for n in self.pop_order], dtype=torch.float32)

    def _to_device_sparse(self, coo_cpu: torch.Tensor) -> torch.Tensor:
        if self.device.type == "cpu":
            return coo_cpu.to_sparse_csr()
        return coo_cpu.to(self.device)

    def set_pathway_gain(self, pre_pop: str, post_pop: str, gain: float, rebuild: bool = True) -> int:
        """Multiply the weights of all edges from pre_pop to post_pop by gain (relative to s)."""
        a0, a1 = self.pop_ranges[pre_pop]
        b0, b1 = self.pop_ranges[post_pop]
        pre, post = self.indices[1], self.indices[0]
        m = (pre >= a0) & (pre < a1) & (post >= b0) & (post < b1)
        self.edge_gain[m] = float(gain)
        self.pathway_gains[f"{pre_pop}->{post_pop}"] = float(gain)
        if rebuild:
            self.W = self._build_W(self.s)
        return int(m.sum())

    def apply_pathway_gains(self, gains: dict[str, float]) -> None:
        for key, g in (gains or {}).items():
            pre_pop, post_pop = key.split("->")
            self.set_pathway_gain(pre_pop, post_pop, g, rebuild=False)
        self.W = self._build_W(self.s)

    def _build_W(self, s: float) -> torch.Tensor:
        vals = torch.clamp(self.values_raw * self.edge_gain * s, -self.w_clip, self.w_clip)
        W = torch.sparse_coo_tensor(self.indices, vals, (self.N, self.N), is_coalesced=True)
        return self._to_device_sparse(W)

    def scale(self, s: float) -> None:
        """Rescale the fixed connectome from the raw synapse counts: values = clip(s * raw, -w_clip, w_clip)."""
        self.s = float(s)
        self.W = self._build_W(self.s)

    def scale_km(self, s_km: float) -> None:
        """W_KM0 = clip(s_km * W_KM0_raw, 0, w_max) with w_max = W_MAX_MULT * max(s_km * W_KM0_raw)."""
        self.s_km = float(s_km)
        scaled = self.W_KM0_raw * self.s_km
        self.w_max = float(config.W_MAX_MULT * scaled.max()) if scaled.numel() else 0.0
        self.W_KM0 = torch.clamp(scaled, 0.0, self.w_max if self.w_max > 0 else float("inf")).to(self.device)

    def slice(self, name: str) -> slice:
        return self.pop_slices[name]

    def n(self, name: str) -> int:
        a, b = self.pop_ranges[name]
        return b - a

    def describe(self) -> str:
        return (f"Connectome N={self.N:,} nnz={self.nnz:,} KC={self.n_KC} MBON={self.n_MBON} device={self.device} "
                f"s={self.s:.4g} s_km={self.s_km:.4g} mode={self.nt_sign_mode} file={self.path.name if self.path else 'explicit'}")


@dataclass
class Readout:
    spike_sum: torch.Tensor        # [N, B] on device, spikes in the readout window
    kc_rates: torch.Tensor         # [n_KC, B] on device, spikes per readout tick
    mbon_app_rate: torch.Tensor    # [B] cpu
    mbon_av_rate: torch.Tensor     # [B] cpu
    dan_pam_rate: torch.Tensor     # [B] cpu
    dan_ppl1_rate: torch.Tensor    # [B] cpu
    pop_rates: torch.Tensor        # [P, B] cpu, mean spikes per tick per neuron
    m_hat: torch.Tensor            # [B] cpu
    v_mean: float
    v_max: float
    total_spikes: float            # over all ticks of the run
    nan_flag: bool
    ticks: int
    readout_window: int
    spikes_per_tick: torch.Tensor | None = None     # [ticks, B] cpu (trace=True)
    kc_spikes_per_tick: torch.Tensor | None = None  # [ticks, B] cpu (trace=True)

    @property
    def kc_active_frac(self) -> torch.Tensor:
        """Fraction of KCs that spiked at least once in the readout window, per column."""
        return (self.kc_rates > 0).float().mean(0).cpu()


class LIF:
    """Batched LIF state over one Connectome. Columns are independent except through W_KM."""

    def __init__(self, connectome: Connectome, batch: int, ticks: int = config.BEAT_TICKS,
                 readout_window: int = config.READOUT_WINDOW, leak: float = config.LEAK, theta: float = config.THETA,
                 refractory: int = config.REFRACTORY, kwta: bool = config.KWTA_ENABLED,
                 kwta_frac: float = config.KC_KWTA_FRAC, noise: float = config.MEMBRANE_NOISE):
        self.conn = connectome
        self.device = connectome.device
        self.N, self.B = connectome.N, int(batch)
        self.ticks, self.readout_window = int(ticks), int(readout_window)
        self.leak, self.theta, self.noise = float(leak), float(theta), float(noise)
        self.refractory = int(refractory)
        assert 0 <= self.refractory <= 127
        self.kwta = bool(kwta) and connectome.n_KC > 0
        self.kwta_frac = float(kwta_frac)
        self.k = max(1, int(self.kwta_frac * connectome.n_KC)) if connectome.n_KC > 0 else 0
        N, B, dev = self.N, self.B, self.device
        self.V = torch.zeros(N, B, dtype=torch.float32, device=dev)
        self.fired = torch.zeros(N, B, dtype=torch.float32, device=dev)
        self.refr = torch.zeros(N, B, dtype=torch.int8, device=dev)
        self.I_ext = torch.zeros(N, B, dtype=torch.float32, device=dev)
        self.spike_sum = torch.zeros(N, B, dtype=torch.float32, device=dev)
        self.W_KM = connectome.W_KM0.clone()
        self._keep = torch.zeros(connectome.n_KC, B, dtype=torch.bool, device=dev)
        # persistent k-WTA: winners are the KCs with the largest low-passed input drive (APL-like sparsening
        # that holds the same KC set for the duration of an odor; Lin et al. 2014 Nat Neurosci, Dasgupta 2017)
        self.kwta_tau = float(getattr(config, "KWTA_TAU_TICKS", 10.0))
        self.kc_drive = torch.zeros(connectome.n_KC, B, dtype=torch.float32, device=dev)
        self.kwta_mode = str(getattr(config, "KWTA_MODE", "drive"))
        self._external_keep: torch.Tensor | None = None
        self._total = torch.zeros((), dtype=torch.float32, device=dev)
        self._refr_fill = self.refractory

    # ---- state -------------------------------------------------------------------------------
    def reset(self, columns: list[int] | torch.Tensor | None = None) -> None:
        """Zero V, fired, refractory counters and spike sums for all or the given columns."""
        if columns is None:
            self.V.zero_(); self.fired.zero_(); self.refr.zero_(); self.spike_sum.zero_(); self.kc_drive.zero_()
            return
        cols = torch.as_tensor(columns, dtype=torch.int64, device=self.device)
        if cols.numel() == 0:
            return
        self.V.index_fill_(1, cols, 0.0)
        self.fired.index_fill_(1, cols, 0.0)
        self.refr.index_fill_(1, cols, 0)
        self.spike_sum.index_fill_(1, cols, 0.0)
        self.kc_drive.index_fill_(1, cols, 0.0)

    def set_kc_winners(self, keep: torch.Tensor | None) -> None:
        """External KC winner mask [n_KC, B] (bool) for the coming beat; None reverts to drive-based k-WTA."""
        self._external_keep = None if keep is None else keep.to(self.device, dtype=torch.bool)

    def set_W_KM(self, W: torch.Tensor | np.ndarray) -> None:
        self.W_KM.copy_(torch.as_tensor(W, dtype=torch.float32, device=self.device))

    # ---- input -------------------------------------------------------------------------------
    def set_input(self, I_ext: torch.Tensor | np.ndarray) -> None:
        self.I_ext.copy_(torch.as_tensor(I_ext, dtype=torch.float32).to(self.device))

    def clear_input(self) -> None:
        self.I_ext.zero_()

    def _pop_values(self, name: str, values) -> torch.Tensor:
        a, b = self.conn.pop_ranges[name]
        v = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if v.dim() == 0:
            v = v.expand(b - a, self.B)
        elif v.dim() == 1:
            assert v.numel() == self.B, f"expected [B={self.B}] got {tuple(v.shape)}"
            v = v.unsqueeze(0).expand(b - a, self.B)
        else:
            assert tuple(v.shape) == (b - a, self.B), f"expected [{b - a}, {self.B}] got {tuple(v.shape)}"
        return v

    def set_population_input(self, name: str, values) -> None:
        """I_ext rows of a population: scalar, [B] (same current for every neuron) or [n, B]."""
        a, b = self.conn.pop_ranges[name]
        if b > a:
            self.I_ext[a:b] = self._pop_values(name, values)

    def add_population_input(self, name: str, values) -> None:
        a, b = self.conn.pop_ranges[name]
        if b > a:
            self.I_ext[a:b] += self._pop_values(name, values)

    def set_rows_input(self, rows: torch.Tensor | np.ndarray, values) -> None:
        """I_ext for arbitrary global row indices: values [B] or [len(rows), B]."""
        r = torch.as_tensor(rows, dtype=torch.int64, device=self.device)
        v = torch.as_tensor(values, dtype=torch.float32, device=self.device)
        if v.dim() == 1:
            v = v.unsqueeze(0).expand(r.numel(), self.B)
        self.I_ext.index_copy_(0, r, v.contiguous())

    def set_glomerulus_input(self, A: torch.Tensor | np.ndarray) -> None:
        """ORN_FOOD currents from a per-glomerulus activation A [G, B] (untyped ORNs get 0)."""
        c = self.conn
        A = torch.as_tensor(A, dtype=torch.float32, device=self.device)
        assert tuple(A.shape) == (c.n_glomeruli, self.B), f"expected [{c.n_glomeruli}, {self.B}] got {tuple(A.shape)}"
        A_ext = torch.cat([A, torch.zeros(1, self.B, dtype=torch.float32, device=self.device)], 0)
        a, b = c.pop_ranges["ORN_FOOD"]
        if b > a:
            self.I_ext[a:b] = A_ext.index_select(0, c.glomerulus_of_orn_dev)

    # ---- dynamics ----------------------------------------------------------------------------
    def _tick(self, accumulate: bool, trace: list | None, kc_trace: list | None) -> None:
        c = self.conn
        I_syn = torch.sparse.mm(c.W, self.fired)                                  # [N, B]
        if c.has_plastic:
            I_syn[c.mbon_slice] += self.W_KM.T @ self.fired[c.kc_slice]           # [n_MBON, B]
        active = self.refr.eq(0)
        V_new = self.V * self.leak
        V_new += self.I_ext
        V_new += I_syn
        if self.noise > 0.0:
            V_new += torch.randn_like(V_new) * self.noise
        self.V = torch.where(active, V_new, self.V)
        self.refr = (self.refr - 1).clamp_(min=0)
        fire = active & (self.V >= self.theta)
        if self.kwta and self.kwta_mode == "external" and self._external_keep is not None:
            keep = self._external_keep
            fire[c.kc_slice] &= keep
            Vkc = self.V[c.kc_slice]
            self.V[c.kc_slice] = torch.where(keep, Vkc, Vkc.clamp(max=0.95 * self.theta))
        elif self.kwta:
            if self.kwta_tau > 0:
                a = 1.0 / self.kwta_tau
                self.kc_drive.mul_(1.0 - a).add_((I_syn[c.kc_slice] + self.I_ext[c.kc_slice]) * a)
                Vk = self.kc_drive
            else:
                Vk = self.V[c.kc_slice]
            top = torch.topk(Vk, self.k, dim=0, sorted=False).indices
            keep = self._keep
            keep.zero_()
            keep.scatter_(0, top, True)
            fire[c.kc_slice] &= keep
            # non-winners are held below threshold (APL inhibition) so stale charge never fires later
            Vkc = self.V[c.kc_slice]
            self.V[c.kc_slice] = torch.where(keep, Vkc, Vkc.clamp(max=0.95 * self.theta))
        self.V.masked_fill_(fire, 0.0)
        self.refr.masked_fill_(fire, self._refr_fill)
        self.fired = fire.to(torch.float32)
        self._total += self.fired.sum()
        if accumulate:
            self.spike_sum += self.fired
        if trace is not None:
            trace.append(self.fired.sum(0))
        if kc_trace is not None:
            kc_trace.append(self.fired[c.kc_slice].sum(0))

    def run(self, ticks: int | None = None, readout_window: int | None = None, trace: bool = False) -> Readout:
        """Advance ``ticks`` ticks (state persists across calls) and read out the last ``readout_window``."""
        ticks = int(ticks if ticks is not None else self.ticks)
        rw = int(min(readout_window if readout_window is not None else self.readout_window, ticks))
        rw = max(rw, 1)
        self.spike_sum.zero_()
        self._total.zero_()
        start = ticks - rw
        tr: list | None = [] if trace else None
        ktr: list | None = [] if trace else None
        for t in range(ticks):
            self._tick(t >= start, tr, ktr)
        return self._readout(ticks, rw, tr, ktr)

    def _readout(self, ticks: int, rw: int, tr: list | None, ktr: list | None) -> Readout:
        c = self.conn
        pop_rates = (torch.sparse.mm(c.group_matrix, self.spike_sum) / rw).cpu()          # [P, B]
        app = pop_rates[c.pop_id["MBON_APP"]]
        av = pop_rates[c.pop_id["MBON_AV"]]
        m_hat = (app - av) / (app + av + M_HAT_EPS)
        finite = torch.isfinite(self.V).all()
        return Readout(
            spike_sum=self.spike_sum, kc_rates=self.spike_sum[c.kc_slice] / rw,
            mbon_app_rate=app, mbon_av_rate=av,
            dan_pam_rate=pop_rates[c.pop_id["DAN_PAM"]], dan_ppl1_rate=pop_rates[c.pop_id["DAN_PPL1"]],
            pop_rates=pop_rates, m_hat=m_hat,
            v_mean=float(self.V.mean()), v_max=float(self.V.max()), total_spikes=float(self._total),
            nan_flag=not bool(finite), ticks=ticks, readout_window=rw,
            spikes_per_tick=torch.stack(tr).cpu() if tr else None,
            kc_spikes_per_tick=torch.stack(ktr).cpu() if ktr else None,
        )


def timing(ticks: int = config.BEAT_TICKS, batch: int = config.SLOTS, connectome: Connectome | None = None,
           device: torch.device | str | None = None, warmup: int = 5, orn_current: float = 0.10) -> dict:
    """ms/tick and s/beat for ``ticks`` x ``batch`` on the device (ORN_FOOD driven so activity is realistic)."""
    conn = connectome or Connectome(device=device)
    lif = LIF(conn, batch, ticks=ticks)
    lif.set_population_input("ORN_FOOD", orn_current)
    lif.run(warmup)
    synchronize(conn.device)
    t0 = time.perf_counter()
    r = lif.run(ticks)
    synchronize(conn.device)
    dt = time.perf_counter() - t0
    return {"device": str(conn.device), "ticks": ticks, "batch": batch, "ms_per_tick": dt / ticks * 1000.0,
            "s_per_beat": dt, "total_spikes": r.total_spikes, "nan_flag": r.nan_flag}
