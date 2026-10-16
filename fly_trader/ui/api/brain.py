"""The fly's brain in 3D (Overview: the memecoin fly; Kalshi overview: the whole brain with both flies).

Payload contract (the React view, console/src/brain): version 1 ``{"version": 1, "geometry": {"url", "meta_url", "sha",
"n"}, "minute", "stamp", "activity_b64": base64 int8[N] | None, "rows", "candidates", "pathways": {"mode", "key",
"reference", "items": [{"kc", "mbon", "delta", "ratio", "strategy", "valence"}], "note", "changed"}, "strategies",
"emphasis", "message", "minutes": [stamps], "running", "caption", "caption_rest"}`` — ``kc``/``mbon`` are sub-graph neuron
indices; ``caption_rest`` is the caption without its minute part (the browser prefixes the minute it shows). Version 2:
the full geometry and ``flies: [{"name", "minute", "stamp", "activity_b64", "n", "rows", "candidates", "pathways",
"strategies", "index": {"url", "n"}, "note", "minutes", "running"}]`` — each fly's vectors are over its own sub-graph
and ``index`` (int32 sub → full) places them in the scene; which fly the shared central brain shows is the browser's choice.

The topics carry the newest minute; playback and the minute slider fetch older minutes as raw int8 from
``/api/brain/activity/<fly>/<stamp>``, and a pinned neuron's day from ``/api/brain/neuron/<fly>/<index>``.
"""
from __future__ import annotations

import base64
import re
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from fastapi import FastAPI, Response
from fastapi.responses import JSONResponse

from ...brain import activity, pathways as pw
from ...brain.connectome import CONNECTOME_DIR
from ...brain.geometry import build_full_geometry, build_geometry
from ...db.queries import q, q1
from ...ops.reset import activity_dir, fly_state_dir, kalshi_activity_dir, kalshi_fly_state_dir
from ..cache import ttl_cache
from ..hub import TopicError, topic
from ..state import kalshi_state, system_state

GEOMETRY_DIR = Path(__file__).resolve().parents[1] / "static" / "brain"
EMPHASIS = ["KC", "MBON_APP", "MBON_AV", "MBON_OTHER"]
MODES = {"bootstrap": "since bootstrap", "24h": "last 24 h"}
N_PAIRS = 21438
FLY_SNAP_KIND = {"memecoin": "fly_plastic", "kalshi": "kalshi_fly_plastic"}
STAMP_RE = re.compile(r"^\d{8}T\d{4}$")


def _dirs(fly: str) -> tuple[Path, Path]:
    if fly == "memecoin":
        return activity_dir(), fly_state_dir()
    if fly == "kalshi":
        return kalshi_activity_dir(), kalshi_fly_state_dir()
    raise TopicError(f"no fly {fly!r}")


# ---------------------------------------------------------------- geometry and files
@ttl_cache(None)
def geometry() -> dict:
    """The memecoin fly's geometry metadata; built into the static directory on first use (idempotent per connectome)."""
    return build_geometry(GEOMETRY_DIR)


@ttl_cache(None)
def geometry_full() -> dict:
    """The whole brain's geometry with the per-fly index maps (built into the static directory on first use)."""
    return build_full_geometry(GEOMETRY_DIR)


def static_url(name: str, sha: str) -> str:
    return f"/static/brain/{name}?v={sha[:12]}"


@ttl_cache(4)
def minutes(d: str) -> list[str]:
    return activity.listing(Path(d))


@ttl_cache(600, maxsize=96)
def activity_at(stamp: str, d: str) -> dict | None:
    try:
        return activity.load(activity.path_of(Path(d), stamp))
    except (FileNotFoundError, ValueError, OSError):
        return None


@ttl_cache(None, maxsize=4)
def pairs(connectome_file: str):
    return pw.km_pairs(CONNECTOME_DIR / connectome_file)


@ttl_cache(None, maxsize=4)
def weights(boot_path: str):
    return pw.bootstrap_weights(boot_path)


@ttl_cache(30, maxsize=16)
def _pathways(mode: str, boot_id: int, boot_path: str, connectome_file: str, state_mtime: float, ref_path: str | None, mbon_offset: int, state_dir: str) -> dict:
    kc, mbon, n_app, n_av = pairs(connectome_file); w0, learn, strategies = weights(boot_path)
    state = Path(state_dir) / "state.pt"
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


def empty_pathways(mode: str) -> dict:
    return {"mode": mode, "key": f"{mode}:none", "reference": None, "items": [], "note": None, "changed": 0}


def pathways(mode: str, fly: dict, meta: dict, fly_name: str = "memecoin") -> dict:
    """The pathway payload for the mode (bootstrap | 24h), or an empty one with the reason. ``meta`` may be the sub-graph
    or the full geometry (the MBON offset is the fly's own sub-graph offset either way, since KC/MBON rows precede every
    excluded population)."""
    boot_id = fly.get("bootstrap"); _, state_dir = _dirs(fly_name); snap_kind = FLY_SNAP_KIND[fly_name]
    if not boot_id:
        return empty_pathways(mode)
    r = q1("SELECT path FROM brain_snapshots WHERE id = %s", (int(boot_id),))
    if not r or not Path(r["path"]).exists():
        return {**empty_pathways(mode), "note": "the fly's bootstrap file is missing"}
    ref = None
    if mode == "24h":
        rows = q("SELECT ts, path, note FROM brain_snapshots WHERE kind = %s ORDER BY id DESC LIMIT 400", (snap_kind,))
        ref = pw.reference_snapshot(rows, time.time(), int(boot_id))
    state = state_dir / "state.pt"; mt = state.stat().st_mtime if state.exists() else 0.0
    return _pathways(mode, int(boot_id), r["path"], meta["connectome_file"], mt, ref, int(meta["pop_ranges"]["MBON_APP"][0]), str(state_dir))


def minute_caption(act: dict | None) -> str | None:
    if not act:
        return None
    m = datetime.fromtimestamp(act["t"], timezone.utc).strftime("%d %b %H:%M UTC")
    return f"{m}: mean activity over {act['n_cand']} candidate rows of {act['n_rows']} eligible" if act["n_cand"] else f"{m}: no candidates, mean over all {act['n_rows']} eligible rows"


def caption_rest(pth: dict, fly: dict, meta: dict) -> str:
    """The caption after its minute part: drift, the learned pathways, notes, channels that cannot learn, coverage. The
    browser puts the minute it shows in front (playback and the slider show other minutes than the topic's newest)."""
    parts = []
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


def caption(act: dict | None, pth: dict, fly: dict, meta: dict) -> str:
    return " · ".join(p for p in (minute_caption(act), caption_rest(pth, fly, meta)) if p)


def _act_fields(act: dict | None, n: int, connectome_file: str) -> tuple[dict, str | None]:
    """The minute's activity as payload fields, or a note when it was recorded on another connectome."""
    if act is None:
        return {"minute": None, "stamp": None, "activity_b64": None, "rows": 0, "candidates": 0}, None
    if len(act["h"]) != n or (act["connectome"] and act["connectome"] != connectome_file):
        return {"minute": None, "stamp": None, "activity_b64": None, "rows": 0, "candidates": 0}, "this minute's activity was recorded on another connectome"
    t = datetime.fromtimestamp(act["t"], timezone.utc)
    return {"minute": t.isoformat(), "stamp": activity.stamp(act["t"]), "activity_b64": base64.b64encode(act["h"].tobytes()).decode(),
            "rows": act["n_rows"], "candidates": act["n_cand"]}, None


def _running(fly: dict) -> bool:
    return bool(fly.get("stage")) and fly.get("stage") != "not trading"


def _mode(params: dict) -> str:
    mode = params.get("mode") or "bootstrap"
    if mode not in MODES:
        raise TopicError(f"mode must be one of {list(MODES)}")
    return mode


# ---------------------------------------------------------------- topics
@topic("brain.memecoin", every=5)
def brain_memecoin(params: dict) -> dict:
    """The memecoin fly's 41,756 neurons lit by its newest scored minute, with the learned KC→MBON pathways."""
    mode = _mode(params); fly = system_state()["fly"]; running = _running(fly)
    try:
        meta = geometry()
    except FileNotFoundError as e:
        return {"version": 1, "unavailable": f"No brain to draw: {e}"}
    d = str(activity_dir()); mins = minutes(d)
    act = activity_at(mins[-1], d) if mins else None
    fields, note = _act_fields(act, meta["n"], meta["connectome_file"])
    pth = pathways(mode, fly, meta) if running else empty_pathways(mode)
    msg = note or (None if mins else ("The fly has not scored a minute yet." if running else "The fly is not running: this is the resting brain."))
    sha = meta["connectome_sha256"]
    return {"version": 1, "geometry": {"url": static_url("geometry.bin", sha), "meta_url": static_url("geometry.json", sha), "sha": sha, "n": meta["n"]},
            **fields, "pathways": pth, "strategies": list((fly.get("channels") or {}).keys()), "emphasis": EMPHASIS, "message": msg,
            "minutes": mins, "running": running, "caption": caption(act, pth, fly, meta), "caption_rest": caption_rest(pth, fly, meta), "fly": "memecoin",
            "fly_detail": {k: fly.get(k) for k in ("drift_by_strategy", "channels", "bootstrap", "learning_frozen")}}


@topic("brain.flies", every=5)
def brain_flies(params: dict) -> dict:
    """The whole brain: the memecoin fly's central brain and other senses, the Kalshi fly's optic lobes, each lit by its own newest minute."""
    mode = _mode(params)
    try:
        meta = geometry_full()
    except FileNotFoundError as e:
        return {"version": 2, "unavailable": f"No brain to draw: {e}"}
    sha = meta["connectome_sha256"]; flies = []
    for name, fly in (("memecoin", system_state()["fly"]), ("kalshi", kalshi_state()["fly"])):
        f = (meta.get("flies") or {}).get(name) or {}
        d = str(_dirs(name)[0]); mins = minutes(d)
        act = activity_at(mins[-1], d) if mins else None
        fields, note = _act_fields(act, int(f.get("n") or 0), meta["connectome_file"])
        pth = pathways(mode, fly, meta, fly_name=name) if _running(fly) else empty_pathways(mode)
        flies.append({"name": name, **fields, "n": int(f.get("n") or 0), "pathways": pth, "strategies": list((fly.get("channels") or {}).keys()),
                      "index": {"url": static_url(f["index"], sha), "n": int(f["n"])} if f else None, "note": f"{name}: {note}" if note else None,
                      "minutes": mins, "running": _running(fly)})
    notes = [f["note"] for f in flies if f.get("note")]
    msg = "; ".join(notes) if notes else (None if any(f["activity_b64"] for f in flies) else "Neither fly has scored a minute yet: this is the resting brain.")
    return {"version": 2, "geometry": {"url": static_url("geometry_full.bin", sha), "meta_url": static_url("geometry_full.json", sha), "sha": sha + ":full", "n": meta["n"]},
            "flies": flies, "emphasis": EMPHASIS, "message": msg,
            "caption": " · ".join(f"{f['name']}: " + (f"{f['minute'][11:16]} UTC, {f['candidates']} candidates of {f['rows']} rows" if f["minute"] else "no minute") for f in flies)}


# ---------------------------------------------------------------- playback and neuron history
_DAYS: dict[str, dict[str, tuple[float, int, np.ndarray]]] = {}       # activity dir → stamp → (t, candidates, activity)
_DAYS_LOCK = threading.Lock()


def day(d: Path) -> list[tuple[float, int, np.ndarray]]:
    """Every minute on disk for one fly, oldest first, kept in memory from the first pinned neuron on (a day of the memecoin
    fly is ~40 MB): each request loads only the minutes written since and drops the pruned ones, so a pin reads one column
    instead of opening every file. The lock makes concurrent pins share the first load."""
    with _DAYS_LOCK:
        have = _DAYS.setdefault(str(d), {}); mins = minutes(str(d)); keep = set(mins)
        for st in [st for st in have if st not in keep]:
            del have[st]
        for st in mins:
            if st not in have:
                try:
                    a = activity.load(activity.path_of(d, st))
                except (FileNotFoundError, ValueError, OSError):
                    continue
                have[st] = (a["t"], a["n_cand"], a["h"])
        return [have[st] for st in mins if st in have]


@ttl_cache(60, maxsize=64)
def neuron_history(fly: str, index: int, newest: str) -> list[dict]:
    """One neuron's activity (−127..127) at every minute on disk (``newest`` keys the cache: it changes once a minute)."""
    return [{"t": datetime.fromtimestamp(t, timezone.utc).isoformat(), "v": int(h[index]), "candidates": n} for t, n, h in day(_dirs(fly)[0]) if 0 <= index < len(h)]


def routes(app: FastAPI) -> None:
    @app.get("/api/brain/activity/{fly}/{stamp}")
    def get_activity(fly: str, stamp: str):
        if fly not in FLY_SNAP_KIND or not STAMP_RE.match(stamp):
            return JSONResponse({"ok": False, "error": "unknown fly or minute"}, status_code=404)
        a = activity_at(stamp, str(_dirs(fly)[0]))
        if a is None:
            return JSONResponse({"ok": False, "error": "no such minute"}, status_code=404)
        return Response(a["h"].tobytes(), media_type="application/octet-stream",
                        headers={"Cache-Control": "private, max-age=86400, immutable", "X-Minute": datetime.fromtimestamp(a["t"], timezone.utc).isoformat(),
                                 "X-Rows": str(a["n_rows"]), "X-Candidates": str(a["n_cand"]), "X-Connectome": a["connectome"] or ""})

    @app.get("/api/brain/neuron/{fly}/{index}")
    def get_neuron(fly: str, index: int):
        if fly not in FLY_SNAP_KIND or index < 0:
            return JSONResponse({"ok": False, "error": "unknown fly or neuron"}, status_code=404)
        mins = minutes(str(_dirs(fly)[0]))
        return {"fly": fly, "index": index, "series": neuron_history(fly, index, mins[-1] if mins else "")}
