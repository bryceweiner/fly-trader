"""Where the fly runs: one resolver for CPU, NVIDIA CUDA and Apple MPS, and the batch sizes each can afford.

``config.DEVICE`` names it: ``auto`` (CUDA if present, else MPS, else CPU), ``cpu``, ``cuda`` / ``cuda:1``, ``mps``. A
backend that is named but not available falls back to the CPU with one warning. Everything that builds or loads a
network (``train/fly_selector.py``, ``brain/connectome.py``) resolves through here, so the graph, the parameters and the
plastic bank always share a device.

Batch sizes come from memory, not constants. A forward pass scatters activity across the connectome's ~1.04 M edges
for every row of a batch (``FlyNet._propagate``: ``index_select`` over the edges, then ``index_add``), so its memory is
proportional to ``edges + 2·neurons`` per row. Measured on the FAFB v783 sub-graph (41,756 neurons): 9.5 MB per row
scoring, 40 MB per row training (the autograd graph keeps every step), ~0.5 GB fixed. Unified memory absorbs 2,048 rows
(19.5 GB) without noticing; an 8 GB card cannot. ``rows_for`` sizes a batch from the graph and the free memory, capped
by the caller's ceiling, so the same code runs on a laptop CPU, a 24 GB card and a 128 GB Mac.
"""
from __future__ import annotations

import functools
import logging
import os

import torch

from .. import config

log = logging.getLogger(__name__)

FORWARD_FACTOR, TRAIN_FACTOR = 2.2, 10.0      # bytes per row = 4 · (edges + 2·neurons) · factor (measured 2.1 / 9.3)
FIXED_BYTES = 512 * 2**20                       # allocator and workspace overhead a batch pays once
BUDGET_SHARE = 0.6                              # of the free memory: leave room for the data and the rest of the process
MEMORY_GB_ENV = "DEVICE_MEMORY_GB"              # override the measured free memory (a shared GPU, a container limit)
MIN_ROWS = 32


def _available(kind: str) -> bool:
    if kind == "cuda":
        return torch.cuda.is_available()
    if kind == "mps":
        return bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()
    return kind == "cpu"


@functools.lru_cache(maxsize=None)
def _resolve(name: str) -> torch.device:
    name = (name or "auto").strip().lower()
    if name == "auto":
        for kind in ("cuda", "mps"):
            if _available(kind):
                return torch.device(kind)
        return torch.device("cpu")
    kind = name.split(":")[0]
    if kind in ("cuda", "mps"):
        if _available(kind):
            try:
                dev = torch.device(name)
                if kind == "cuda" and dev.index is not None and dev.index >= torch.cuda.device_count():
                    log.warning("DEVICE=%s but only %d CUDA device(s) present; using cuda:0", name, torch.cuda.device_count())
                    return torch.device("cuda")
                return dev
            except RuntimeError as e:
                log.warning("DEVICE=%s is not a valid device (%s); using the CPU", name, e)
                return torch.device("cpu")
        log.warning("DEVICE=%s but %s is not available on this machine; using the CPU", name, kind.upper())
        return torch.device("cpu")
    if kind != "cpu":
        log.warning("DEVICE=%r is not cpu, cuda, mps or auto; using the CPU", name)
    return torch.device("cpu")


def resolve(name: str | None = None) -> torch.device:
    """The device named by ``name`` (default ``config.DEVICE``), with fallback to the CPU. Cached per name: the fallback
    warning is logged once, and every caller in a process agrees."""
    return _resolve(name if name is not None else config.DEVICE)


def reset() -> None:
    """Forget resolved devices (after ``config.DEVICE`` changes, e.g. a CLI --device flag)."""
    _resolve.cache_clear()


def describe(dev: torch.device | None = None) -> str:
    """Human-readable: what the fly is running on."""
    dev = dev or resolve()
    if dev.type == "cuda":
        i = dev.index if dev.index is not None else torch.cuda.current_device()
        p = torch.cuda.get_device_properties(i)
        return f"cuda:{i} ({p.name}, {p.total_memory / 2**30:.0f} GB)"
    if dev.type == "mps":
        return "mps (Apple GPU, unified memory)"
    return f"cpu ({torch.get_num_threads()} threads)"


def free_bytes(dev: torch.device | None = None) -> int:
    """Memory a batch may draw on: the card's free memory on CUDA, the host's on MPS (unified) and CPU; the
    ``DEVICE_MEMORY_GB`` environment variable overrides the measurement."""
    override = os.environ.get(MEMORY_GB_ENV)
    if override:
        try:
            return int(float(override) * 2**30)
        except ValueError:
            log.warning("%s=%r is not a number; ignoring it", MEMORY_GB_ENV, override)
    dev = dev or resolve()
    if dev.type == "cuda":
        free, _total = torch.cuda.mem_get_info(dev.index if dev.index is not None else torch.cuda.current_device())
        return int(free)
    import psutil
    return int(psutil.virtual_memory().available)


def rows_for(n_edges: int, n_neurons: int, dev: torch.device | None = None, *, ceiling: int, training: bool = False,
             free: int | None = None) -> int:
    """Rows of one batch this device can afford on a graph of ``n_edges`` and ``n_neurons`` — a share of the free
    memory over the measured per-row cost — never more than ``ceiling`` (the caller's own reason for a limit) and
    never fewer than ``MIN_ROWS``."""
    per_row = 4.0 * (n_edges + 2 * n_neurons) * (TRAIN_FACTOR if training else FORWARD_FACTOR)
    budget = (free if free is not None else free_bytes(dev)) * BUDGET_SHARE - FIXED_BYTES
    return int(max(MIN_ROWS, min(ceiling, budget // per_row)))


def seed_all(seed: int) -> None:
    """Seed torch on every backend (``torch.manual_seed`` alone leaves other CUDA devices unseeded)."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def empty_cache(dev: torch.device | None = None) -> None:
    """Return cached device memory between phases (a bootstrap's autograd state before a replay's long loop)."""
    dev = dev or resolve()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps" and hasattr(torch, "mps"):
        torch.mps.empty_cache()
