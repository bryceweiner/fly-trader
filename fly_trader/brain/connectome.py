"""The FlyWire connectome the fly (``train/fly_selector.py``) is built on: the artifact written by ``build-connectome``
(``brain/connectome_build.py``), loaded as signed synapse counts with population ranges, and restricted to a set of
populations by ``SubConnectome``.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .. import config

CONNECTOME_DIR = config.BRAIN_DIR / "connectome"
CURRENT_FILE = CONNECTOME_DIR / "current.txt"
SCALE_FILE = CONNECTOME_DIR / "scale.json"
AFFERENT_POPS = ("ORN_FOOD", "ORN_DANGER", "GRN_SWEET", "GRN_BITTER", "GRN_OTHER", "MECH_JO", "MECH_BRISTLE",
                 "MECH_OTHER", "THERMO_WARM", "THERMO_COOL", "THERMO_OTHER")     # sensory neurons: where the features enter
EFFERENT_POP = "DESCENDING"                                                     # brain-to-body neurons: where the decoder reads


def resolve_device(name: str | None = None) -> torch.device:
    """config.DEVICE (or ``name``) as a device — CPU, CUDA or MPS, with fallback to the CPU — through ``brain/device.py``,
    the one resolver every network shares."""
    from .device import resolve
    return resolve(name)


def current_connectome_path() -> Path:
    if not CURRENT_FILE.exists():
        raise FileNotFoundError(f"{CURRENT_FILE} missing: run `fly-trader build-connectome` first")
    name = CURRENT_FILE.read_text().strip()
    path = Path(name) if Path(name).is_absolute() else CONNECTOME_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"connectome {path} named in current.txt does not exist")
    return path


def _scale_for(connectome_file: str) -> dict | None:
    """The calibrated scale if scale.json exists and refers to this artifact."""
    if not SCALE_FILE.exists():
        return None
    try:
        d = json.loads(SCALE_FILE.read_text())
    except Exception:
        return None
    return d if d.get("connectome") in (connectome_file, Path(connectome_file).name) else None


MB_POPS = ("KC", "MBON_APP", "MBON_AV", "MBON_OTHER")    # the mushroom-body block the build keeps apart from the sparse graph
DAN_POPS = ("DAN_PAM", "DAN_PPL1", "DAN_OTHER")


def mbon_compartments(indices: torch.Tensor, values_raw: torch.Tensor, pop_ranges: dict, cell_type=None) -> tuple[np.ndarray, list[str]]:
    """Each MBON column's compartment: the dopamine cell type (or, without cell types, the DAN population) sending it the
    most synapse mass in the raw graph — before scale.json's gains silence DAN→MBON as fast edges (Aso et al. 2014: a
    compartment = the DAN type innervating that MBON's dendrites). −1 when no DAN reaches the MBON."""
    if not all(p in pop_ranges for p in ("MBON_APP", "MBON_OTHER")):
        return np.zeros(0, int), []
    m0, m1 = pop_ranges["MBON_APP"][0], pop_ranges["MBON_OTHER"][1]
    post, pre = indices[0].numpy(), indices[1].numpy(); w = np.abs(values_raw.numpy())
    dan = np.zeros(len(pre), bool); label = np.full(len(pre), "", dtype=object)
    for pop in DAN_POPS:
        if pop in pop_ranges:
            a, b = pop_ranges[pop]; sel = (pre >= a) & (pre < b); dan |= sel
            label[sel] = np.asarray(cell_type, dtype=object)[pre[sel]] if cell_type is not None else pop
    e = dan & (post >= m0) & (post < m1)
    names = sorted({str(x) or "DAN" for x in label[e]})
    comp = np.full(m1 - m0, -1, int)
    if not e.any():
        return comp, names
    mass = np.zeros((m1 - m0, len(names))); col = {n: i for i, n in enumerate(names)}
    np.add.at(mass, (post[e] - m0, np.array([col[str(x) or "DAN"] for x in label[e]])), w[e])
    has = mass.sum(1) > 0; comp[has] = mass[has].argmax(1)
    return comp, names


def apply_pathway_gains(indices: torch.Tensor, values_raw: torch.Tensor, pop_ranges: dict, gains: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Multiply the edges from population A to population B by ``gains['A->B']``; edges scaled to 0 are dropped.
    scale.json silences the direct DAN→MBON/KC edges: dopamine acts on the mushroom body through KC→MBON plasticity,
    not fast excitation (Hige 2015, Handler 2019)."""
    post, pre = indices[0], indices[1]
    v = values_raw.clone()
    for key, g in (gains or {}).items():
        a, _, b = key.partition("->")
        if a not in pop_ranges or b not in pop_ranges:
            continue
        (a0, a1), (b0, b1) = pop_ranges[a], pop_ranges[b]
        v[(pre >= a0) & (pre < a1) & (post >= b0) & (post < b1)] *= float(g)
    keep = v != 0
    return indices[:, keep], v[keep]


class Connectome:
    """Signed synapse counts ``values_raw`` [E] on edges ``indices`` [2, E] (row = post, col = pre) over ``N`` neurons,
    population ``pop_ranges``, and the weight scale ``s`` (scale.json when it refers to this artifact, else the build's
    s0). The KC→MBON synapses are not in the sparse graph: the build keeps them as the dense block ``W_KM0``
    [n_KC, n_MBON] (synapse counts; MBON columns = MBON_APP, MBON_AV, MBON_OTHER in population order) with the mask
    ``M_KM`` of the pairs that exist. scale.json's ``pathway_gains`` are applied to the sparse graph. Tensors stay on
    the CPU; ``device`` is where models built on it should run."""

    def __init__(self, path: str | Path | None = None, device: torch.device | str | None = None):
        path = Path(path) if path else current_connectome_path()
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.N = int(z["N"])
        self.pop_ranges = {k: (int(a), int(b)) for k, (a, b) in json.loads(str(z["pop_ranges"])).items()}
        scale = _scale_for(path.name)
        self.pathway_gains = dict((scale or {}).get("pathway_gains") or {})
        raw_i = torch.from_numpy(np.ascontiguousarray(z["indices"])).to(torch.int64); raw_v = torch.from_numpy(np.ascontiguousarray(z["values_raw"])).to(torch.float32)
        self.mbon_compartment, self.compartment_names = mbon_compartments(raw_i, raw_v, self.pop_ranges, z["cell_type"] if "cell_type" in z.files else None)
        self.indices, self.values_raw = apply_pathway_gains(torch.from_numpy(np.ascontiguousarray(z["indices"])).to(torch.int64),
                                                            torch.from_numpy(np.ascontiguousarray(z["values_raw"])).to(torch.float32),
                                                            self.pop_ranges, self.pathway_gains)
        self.W_KM0 = torch.from_numpy(np.ascontiguousarray(z["W_KM0"])).to(torch.float32)
        self.M_KM = torch.from_numpy(np.ascontiguousarray(z["M_KM"])).to(torch.bool)
        self.spectral_radius_raw = float(z["spectral_radius_raw"])
        self.content_sha256 = str(z["content_sha256"])
        self.s = float(scale["s"]) if scale and "s" in scale else float(z["s0"])
        self.device = device if isinstance(device, torch.device) else resolve_device(device)

    @classmethod
    def load(cls, path: str | Path | None = None, device: torch.device | str | None = None) -> "Connectome":
        """The artifact named in data/brain/connectome/current.txt (or ``path``)."""
        return cls(path, device=device)


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
        lost = [p for p in MB_POPS if p in c.pop_ranges and not bool(keep[c.pop_ranges[p][0]:c.pop_ranges[p][1]].all())]
        if lost:
            raise ValueError(f"a sub-connectome must keep the whole mushroom body (KC→MBON block); excluded: {lost}")
        self.W_KM0 = getattr(c, "W_KM0", None)          # unchanged: KC and MBON rows keep their order and ranges
        self.mbon_compartment = getattr(c, "mbon_compartment", None); self.compartment_names = getattr(c, "compartment_names", [])
        self.M_KM = getattr(c, "M_KM", None)
        self.pathway_gains = getattr(c, "pathway_gains", {})
        self.excluded = exclude
        self.keep = keep
