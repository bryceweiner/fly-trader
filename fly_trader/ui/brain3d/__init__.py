"""The fly's brain in 3D on the Overview page: a Streamlit custom component (v2) around the standalone ``brain.js``.

Contract (also the website's): the component's ``data`` is
``{"version": 1, "geometry": {"url", "meta_url", "sha", "n"}, "minute": iso | None, "activity_b64": base64 int8[N] |
None, "rows": int, "candidates": int, "pathways": {"mode", "key", "reference", "items": [{"kc", "mbon", "delta",
"ratio", "strategy", "valence"}]}, "strategies": [...], "emphasis": [...], "message": str | None}`` — ``kc``/``mbon``
are sub-graph neuron indices. The geometry (``brain/geometry.py``) is fetched once per connectome from Streamlit's
static directory ``ui/static/brain`` (``server.enableStaticServing``); everything else is small and sent per refresh.

Version 2 (``payload_flies``): the whole brain with both flies overlaid — ``{"version": 2, "geometry": {... the full
geometry ...}, "flies": [{"name", "minute", "activity_b64", "n", "rows", "candidates", "pathways", "strategies",
"index": {"url", "n"}}, ...], "central_owner": "memecoin" | "kalshi", "emphasis", "message"}``. Each fly's activity
vector and pathway indices are over its own sub-graph; ``index`` (int32 sub → full) places them in the scene; neurons
two flies share (the central brain) show ``central_owner``'s activity.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from ...brain.geometry import build_full_geometry, build_geometry

_DIR = Path(__file__).parent
STATIC_DIR = _DIR.parent / "static"
GEOMETRY_DIR = STATIC_DIR / "brain"
EMPHASIS = ["KC", "MBON_APP", "MBON_AV", "MBON_OTHER"]
VERSION = 1
VERSION_FLIES = 2

_BRAIN = st.components.v2.component("fly_brain3d", html='<div class="brain3d"></div>\n', css=(_DIR / "brain.css").read_text(), js=(_DIR / "brain.js").read_text())


@st.cache_resource(show_spinner="Placing the fly's neurons…")
def geometry() -> dict:
    """The geometry metadata; built into the static directory on first use (idempotent per connectome)."""
    return build_geometry(GEOMETRY_DIR)


@st.cache_resource(show_spinner="Placing every neuron of the brain…")
def geometry_full() -> dict:
    """The whole brain's geometry with the per-fly index maps (built into the static directory on first use)."""
    return build_full_geometry(GEOMETRY_DIR)


def static_url(name: str, sha: str) -> str:
    base = (st.get_option("server.baseUrlPath") or "").strip("/")
    return f"/{base + '/' if base else ''}app/static/brain/{name}?v={sha[:12]}"


def payload(meta: dict, act: dict | None, pathways: dict, strategies: list[str], message: str | None = None) -> dict:
    sha = meta["connectome_sha256"]; b64 = None; minute = None; rows = cand = 0
    if act is not None:
        if len(act["h"]) != meta["n"] or (act["connectome"] and act["connectome"] != meta["connectome_file"]):
            message = message or "this minute's activity was recorded on another connectome"
        else:
            from datetime import datetime, timezone
            b64 = base64.b64encode(act["h"].tobytes()).decode(); minute = datetime.fromtimestamp(act["t"], timezone.utc).isoformat()
            rows, cand = act["n_rows"], act["n_cand"]
    return {"version": VERSION, "geometry": {"url": static_url("geometry.bin", sha), "meta_url": static_url("geometry.json", sha), "sha": sha, "n": meta["n"]},
            "minute": minute, "activity_b64": b64, "rows": rows, "candidates": cand, "pathways": pathways, "strategies": list(strategies),
            "emphasis": EMPHASIS, "message": message}


def fly_payload(name: str, meta: dict, act: dict | None, pathways: dict, strategies: list[str]) -> dict:
    """One fly's entry of a version-2 payload; ``meta`` is the full geometry (its ``flies[name]`` gives the sub-graph size and index map)."""
    f = (meta.get("flies") or {}).get(name) or {}; sha = meta["connectome_sha256"]; b64 = None; minute = None; rows = cand = 0; note = None
    if act is not None:
        if len(act["h"]) != f.get("n") or (act["connectome"] and act["connectome"] != meta["connectome_file"]):
            note = f"{name}: this minute's activity was recorded on another connectome"
        else:
            from datetime import datetime, timezone
            b64 = base64.b64encode(act["h"].tobytes()).decode(); minute = datetime.fromtimestamp(act["t"], timezone.utc).isoformat()
            rows, cand = act["n_rows"], act["n_cand"]
    return {"name": name, "minute": minute, "activity_b64": b64, "n": int(f.get("n") or 0), "rows": rows, "candidates": cand, "pathways": pathways, "strategies": list(strategies),
            "index": {"url": static_url(f["index"], sha), "n": int(f["n"])} if f else None, "note": note}


def payload_flies(meta: dict, flies: list[dict], central_owner: str = "kalshi", message: str | None = None) -> dict:
    sha = meta["connectome_sha256"]; notes = [f["note"] for f in flies if f.get("note")]
    return {"version": VERSION_FLIES, "geometry": {"url": static_url("geometry_full.bin", sha), "meta_url": static_url("geometry_full.json", sha), "sha": sha + ":full", "n": meta["n"]},
            "flies": flies, "central_owner": central_owner, "emphasis": EMPHASIS, "message": message or ("; ".join(notes) if notes else None)}


def brain_view(data: dict, *, key: str = "brain3d", height: int = 560):
    return _BRAIN(data=data, key=key, height=height)
