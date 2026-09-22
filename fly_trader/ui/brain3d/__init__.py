"""The fly's brain in 3D on the Overview page: a Streamlit custom component (v2) around the standalone ``brain.js``.

Contract (also the website's): the component's ``data`` is
``{"version": 1, "geometry": {"url", "meta_url", "sha", "n"}, "minute": iso | None, "activity_b64": base64 int8[N] |
None, "rows": int, "candidates": int, "pathways": {"mode", "key", "reference", "items": [{"kc", "mbon", "delta",
"ratio", "strategy", "valence"}]}, "strategies": [...], "emphasis": [...], "message": str | None}`` — ``kc``/``mbon``
are sub-graph neuron indices. The geometry (``brain/geometry.py``) is fetched once per connectome from Streamlit's
static directory ``ui/static/brain`` (``server.enableStaticServing``); everything else is small and sent per refresh.
"""
from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from ...brain.geometry import build_geometry

_DIR = Path(__file__).parent
STATIC_DIR = _DIR.parent / "static"
GEOMETRY_DIR = STATIC_DIR / "brain"
EMPHASIS = ["KC", "MBON_APP", "MBON_AV", "MBON_OTHER"]
VERSION = 1

_BRAIN = st.components.v2.component("fly_brain3d", html='<div class="brain3d"></div>\n', css=(_DIR / "brain.css").read_text(), js=(_DIR / "brain.js").read_text())


@st.cache_resource(show_spinner="Placing the fly's neurons…")
def geometry() -> dict:
    """The geometry metadata; built into the static directory on first use (idempotent per connectome)."""
    return build_geometry(GEOMETRY_DIR)


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


def brain_view(data: dict, *, key: str = "brain3d", height: int = 560):
    return _BRAIN(data=data, key=key, height=height)
