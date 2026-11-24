"""What this install has turned on: the menu and the chain picker read it. And what it runs (``versions``, the strip
along the top of every page, Bryce 2026-11-07: so two installs can be told apart at a glance): the code, read once when the console starts (a later commit changes nothing already loaded), and each chain's
selector and fly as the engines report them, by their number in this install and by their file's sha256, which is the
same on every machine that runs the same model."""
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

from ... import config, markets
from ...db.queries import q
from .. import runtime
from ..hub import topic
from ..state import KALSHI_BOOKS, kalshi_enabled, market_info, setting


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("fly-trader")
    except Exception:
        return "dev"


def _git(*args: str) -> str | None:
    try:
        r = subprocess.run(["git", "-C", str(config.REPO_ROOT), *args], capture_output=True, text=True, timeout=5)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:
        return None


def code_version() -> dict:
    """What code this console runs, in this order: ``FLY_CODE_VERSION`` (a deploy that stamps its own), a release's
    ``release.json`` (Docker installs: its number and commit), the git commit the tree was at (``changed``: tracked files
    differed from it), else the package version. {"label", "commit", "release", "changed", "source", "detail"}."""
    env = os.environ.get("FLY_CODE_VERSION", "").strip()
    if env:
        return {"label": env, "commit": None, "release": None, "changed": False, "source": "FLY_CODE_VERSION", "detail": env}
    rel = Path(config.REPO_ROOT) / "release.json"
    if rel.is_file():
        try:
            r = json.loads(rel.read_text())
            seq, commit = r.get("seq"), str(r.get("commit") or "")[:7] or None
            return {"label": f"release {seq}" + (f" · {commit}" if commit else ""), "commit": commit, "release": seq, "changed": False, "source": "release.json",
                    "detail": f"release {seq}, commit {r.get('commit')}, made {r.get('created_at')}"}
        except Exception:
            pass
    commit = _git("rev-parse", "--short=7", "HEAD")
    if commit:
        changed = bool(_git("status", "--porcelain", "--untracked-files=no"))
        subject = _git("log", "-1", "--format=%cd %s", "--date=format:%Y-%m-%d %H:%M") or ""
        return {"label": commit + (" +changes" if changed else ""), "commit": commit, "release": None, "changed": changed, "source": "git",
                "detail": f"{commit}: {subject}" + (" (with uncommitted changes)" if changed else "")}
    v = _version()
    return {"label": v, "commit": None, "release": None, "changed": False, "source": "package", "detail": f"fly-trader {v}"}


CODE = code_version()


def _model(snapshot_id) -> dict | None:
    if not snapshot_id:
        return None
    r = q("SELECT id, path, sha256 FROM brain_snapshots WHERE id = %s", (int(snapshot_id),))
    if not r:
        return {"id": int(snapshot_id), "sha": None, "file": None}
    return {"id": int(r[0]["id"]), "sha": (r[0]["sha256"] or "")[:8] or None, "file": Path(r[0]["path"]).name if r[0]["path"] else None}


def model_versions() -> list[dict]:
    """Each memecoin chain's selector and fly as its engines last reported them: the fly (fly_status*: ``bootstrap``) and
    the selector that decides its buys, the fly's gate (``gate``; agent/fly_session._follow_deployed), else the selector
    book's (selector_status*: ``snapshot``). After a handover the selector book stops and its last report goes stale (a
    paper-only install, 2026-11-04), while the fly's gate keeps following the deployed selector."""
    out = []
    for m in markets.enabled():
        if m.chain not in ("sol", "rh"):
            continue
        sfx = "" if m.chain == "sol" else f"_{m.chain}"
        sel, _ = setting(f"selector_status{sfx}"); fly, _ = setting(f"fly_status{sfx}")
        out.append({"chain": m.chain, "selector": _model(fly.get("gate") or sel.get("snapshot")), "fly": _model(fly.get("bootstrap"))})
    return out


@topic("meta", every=30)
def meta(params: dict) -> dict:
    return {"pid": os.getpid(), "hosts_workers": runtime.HOSTS_WORKERS, "kalshi": kalshi_enabled(), "rh": bool(config.RH_ENABLED),
            "markets": [market_info(m) for m in markets.enabled()], "kalshi_books": KALSHI_BOOKS,
            "version": _version(), "versions": {"code": CODE, "models": model_versions()}}
