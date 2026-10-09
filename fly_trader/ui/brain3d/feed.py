"""What the brain view reads: the fly's per-minute activity files, its learned KC→MBON changes and the snapshots to diff
them against — all from disk and the database, cached so the 5 s fragment costs almost nothing."""
from __future__ import annotations

import time
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st

from ...brain import activity, pathways as pw
from ...brain.connectome import CONNECTOME_DIR
from ...db.queries import q, q1
from ...ops.reset import activity_dir, fly_state_dir, kalshi_activity_dir, kalshi_fly_state_dir   # noqa: F401 (re-exported for the pages)

MODES = {"since bootstrap": "bootstrap", "last 24 h": "24h"}
N_PAIRS = 21438


def activity_mtime(d: Path | None = None) -> float:
    d = activity_dir() if d is None else Path(d)
    return d.stat().st_mtime if d.exists() else 0.0


@st.cache_data(ttl=4, show_spinner=False)
def minutes(mtime: float, d: str | None = None) -> list[str]:
    return activity.listing(activity_dir() if d is None else Path(d))


@st.cache_data(ttl=600, max_entries=64, show_spinner=False)
def activity_at(stamp: str, d: str | None = None) -> dict | None:
    p = activity.path_of(activity_dir() if d is None else Path(d), stamp)
    try:
        return activity.load(p)
    except (FileNotFoundError, ValueError, OSError):
        return None


def label(stamp: str) -> str:
    return datetime.strptime(stamp, activity.STAMP).strftime("%d %b %H:%M")


@st.cache_resource(show_spinner=False)
def pairs(connectome_file: str):
    return pw.km_pairs(CONNECTOME_DIR / connectome_file)


@st.cache_resource(show_spinner=False)
def weights(boot_path: str):
    return pw.bootstrap_weights(boot_path)


@st.cache_data(ttl=30, show_spinner=False)
def _pathways(mode: str, boot_id: int, boot_path: str, connectome_file: str, state_mtime: float, ref_path: str | None, mbon_offset: int, state_dir: str | None = None) -> dict:
    kc, mbon, n_app, n_av = pairs(connectome_file); w0, learn, strategies = weights(boot_path)
    state = (fly_state_dir() if state_dir is None else Path(state_dir)) / "state.pt"
    out = {"mode": mode, "key": f"{mode}:{boot_id}:{int(state_mtime)}:{ref_path or ''}", "reference": None, "items": [], "note": None, "changed": 0}
    if not state.exists():
        out["note"] = "the fly has not written its state yet"; return out
    D, bid = pw.bank_D(state)
    if bid is not None and bid != boot_id:
        out["note"] = f"the state on disk belongs to bootstrap #{bid}, not #{boot_id}"; return out
    D_ref = None
    if mode == "24h":
        if ref_path:
            D_ref, _ = pw.bank_D(ref_path); out["reference"] = Path(ref_path).stem.replace("snap_", "")
        else:
            out["note"] = "no snapshot a day old yet: showing the change since bootstrap"
    out["changed"] = int(((D - (D_ref if D_ref is not None else 0)) != 0).sum())
    out["items"] = pw.top_pathways(D, w0, kc, mbon, learn, strategies, n_app, n_av, D_ref=D_ref, mbon_offset=mbon_offset)
    return out


FLY_SNAP_KIND = {"memecoin": "fly_plastic", "kalshi": "kalshi_fly_plastic"}


def pathways(mode_label: str, fly: dict, meta: dict, fly_name: str = "memecoin") -> dict:
    """The pathway payload for the chosen mode, or an empty one with the reason. ``fly_name`` picks the state directory
    and snapshot kind (the Kalshi fly keeps its own); ``meta`` may be the sub-graph or the full geometry (the MBON offset
    is the fly's own sub-graph offset either way, since KC/MBON rows precede every excluded population)."""
    mode = MODES.get(mode_label, "bootstrap"); boot_id = fly.get("bootstrap")
    state_dir = kalshi_fly_state_dir() if fly_name == "kalshi" else fly_state_dir(); snap_kind = FLY_SNAP_KIND.get(fly_name, "fly_plastic")
    empty = {"mode": mode, "key": f"{mode}:none", "reference": None, "items": [], "note": None, "changed": 0}
    if not boot_id:
        return empty
    r = q1("SELECT path FROM brain_snapshots WHERE id = %s", (int(boot_id),))
    if not r or not Path(r["path"]).exists():
        return {**empty, "note": "the fly's bootstrap file is missing"}
    ref = None
    if mode == "24h":
        rows = q("SELECT ts, path, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC LIMIT 400", (snap_kind,))
        ref = pw.reference_snapshot(rows, time.time(), int(boot_id))
    state = state_dir / "state.pt"; mt = state.stat().st_mtime if state.exists() else 0.0
    return _pathways(mode, int(boot_id), r["path"], meta["connectome_file"], mt, ref, int(meta["pop_ranges"]["MBON_APP"][0]), str(state_dir))


def caption(act: dict | None, pth: dict, fly: dict, meta: dict) -> str:
    parts = []
    if act:
        m = datetime.fromtimestamp(act["t"], timezone.utc).strftime("%d %b %H:%M UTC")
        parts.append(f"{m}: mean activity over {act['n_cand']} candidate rows of {act['n_rows']} eligible" if act["n_cand"] else f"{m}: no candidates, mean over all {act['n_rows']} eligible rows")
    drift = fly.get("drift_by_strategy") or {}
    if drift:
        parts.append("drift " + ", ".join(f"{k} {float(v) * 100:.2f}%" for k, v in drift.items()))
    if pth.get("items") or pth.get("changed"):
        parts.append(f"{pth['changed']:,} of {N_PAIRS:,} KC→MBON synapses changed" + (f" since {pth['reference']}" if pth.get("reference") else " since bootstrap") + f"; top {len(pth['items'])} shown")
    elif fly.get("bootstrap"):
        parts.append("no KC→MBON synapse has changed yet")
    if pth.get("note"):
        parts.append(pth["note"])
    frozen = [k for k, v in (fly.get("channels") or {}).items() if v and float(v[0] or 0) == 0.0]
    if frozen:
        parts.append(f"{', '.join(frozen)}: channel does not learn (α = 0), so its pathways never change")
    c = (meta.get("coverage") or {}).get("centroid")
    if c:
        parts.append(f"{c} neurons without a recorded position sit at their population's centroid")
    return " · ".join(parts)
