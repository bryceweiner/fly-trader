"""The FlyWire connectome the fly (``train/fly_selector.py``) is built on: the artifact written by ``build-connectome``
(``brain/connectome_build.py``), loaded as signed synapse counts with population ranges, and restricted to a set of
populations by ``SubConnectome``.
"""
from __future__ import annotations

import json
import sys
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
    """config.DEVICE with automatic fallback to CPU when MPS (or CUDA) is unavailable."""
    name = (name or config.DEVICE or "cpu").lower()
    if name.startswith("mps"):
        if torch.backends.mps.is_available():
            return torch.device("mps")
        print("[connectome] MPS unavailable; falling back to cpu", file=sys.stderr)
        return torch.device("cpu")
    if name.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(name)
        print("[connectome] CUDA unavailable; falling back to cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device("cpu")


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


class Connectome:
    """Signed synapse counts ``values_raw`` [E] on edges ``indices`` [2, E] (row = post, col = pre) over ``N`` neurons,
    population ``pop_ranges``, and the weight scale ``s`` (scale.json when it refers to this artifact, else the build's
    s0). Tensors stay on the CPU; ``device`` is where models built on it should run."""

    def __init__(self, path: str | Path | None = None, device: torch.device | str | None = None):
        path = Path(path) if path else current_connectome_path()
        z = np.load(path, allow_pickle=False)
        self.path = path
        self.N = int(z["N"])
        self.indices = torch.from_numpy(np.ascontiguousarray(z["indices"])).to(torch.int64)
        self.values_raw = torch.from_numpy(np.ascontiguousarray(z["values_raw"])).to(torch.float32)
        self.pop_ranges = {k: (int(a), int(b)) for k, (a, b) in json.loads(str(z["pop_ranges"])).items()}
        self.spectral_radius_raw = float(z["spectral_radius_raw"])
        self.content_sha256 = str(z["content_sha256"])
        scale = _scale_for(path.name)
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
        self.excluded = exclude
        self.keep = keep
