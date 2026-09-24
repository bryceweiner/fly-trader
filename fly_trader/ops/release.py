"""Installing a release's models into a running install (the hosted vault fly's CI/CD; plan phase 7).

A release directory holds ``models/`` (policies, selectors, connectome, wallet_skill) and ``seed/release_seed.json``
(written by tools/make_seed.py): the brain_snapshots rows WITHOUT ids, the replay verdict, the pinned selector (by
sha256) and the skill table's name. The host updater has already verified the signature and every file's hash.

``apply-release`` copies the files into DATA_DIR, reuses a brain_snapshots row with the same (kind, sha256) or inserts a
new one (never the source's id: this server's own hourly ``fly_plastic`` rows share the sequence), upserts the verdict
and the pin, and stamps ``ui_settings['release_applied_at']``; the running engine sees the stamp within 15 s and the
fly and selector switch at once (FlyBook.maybe_reload, SelectorBook.maybe_reload). Refuses a fly that is not
deployable on this code's definitions, so a models-only release can never outrun its code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import shutil
import time
from pathlib import Path

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction

log = logging.getLogger(__name__)


class ReleaseError(RuntimeError):
    pass


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _set(conn, key: str, value) -> None:
    conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                 (key, json.dumps(value, default=str)))


def apply_release(release_dir: Path, bootstrap: bool = False) -> dict:
    """Install the release at ``release_dir``. ``bootstrap``: first boot (also writes autostart if absent)."""
    import sys
    if "fly_trader.train.selector" in sys.modules:       # FLY_VERSION folds in the skill config at import: place files first
        raise ReleaseError("apply-release must run in its own process (fly-trader apply-release), not inside the console")
    rel = Path(release_dir)
    seed = json.loads((rel / "seed" / "release_seed.json").read_text())
    if (rel / "release.json").exists():                    # the signed manifest carries the sequence number and commit
        man = json.loads((rel / "release.json").read_text())
        seed.setdefault("seq", man.get("seq")); seed.setdefault("git_commit", man.get("git_commit"))
    models = rel / "models"
    brain = config.BRAIN_DIR
    # 1. files (copy only what is missing or different; the hashes were verified by the updater, re-checked here)
    placed: dict[str, Path] = {}
    for snap in seed["snapshots"]:
        sub = "policies" if snap["kind"] == "fly_selector" else "selectors"
        src = models / sub / snap["file"]
        if _sha256(src) != snap["sha256"]:
            raise ReleaseError(f"{src.name}: sha256 does not match the release seed")
        dst = brain / sub / snap["file"]; dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists() or _sha256(dst) != snap["sha256"]:
            shutil.copy2(src, dst.with_suffix(dst.suffix + ".part")); dst.with_suffix(dst.suffix + ".part").replace(dst)
        placed[snap["sha256"]] = dst
    cdir = brain / "connectome"; cdir.mkdir(parents=True, exist_ok=True)
    for f in (models / "connectome").glob("*"):
        if not (cdir / f.name).exists() or _sha256(cdir / f.name) != _sha256(f):
            shutil.copy2(f, cdir / f.name)
    from ..train.mature import SKILL_DIR
    SKILL_DIR.mkdir(parents=True, exist_ok=True)
    for f in (models / "wallet_skill").glob("*"):
        if f.name == "TABLE" or (f.suffix == ".parquet" and (SKILL_DIR / f.name).exists()):
            continue
        shutil.copy2(f, SKILL_DIR / f.name)
    # 2. rows (imported only now: the definitions version reads the skill config just placed)
    from ..train import fly_selector
    fly_meta = next((s["note"] for s in seed["snapshots"] if s["kind"] == "fly_selector"), None)
    meta = json.loads(fly_meta) if isinstance(fly_meta, str) else (fly_meta or {})
    ok, why = fly_selector.deployable(meta)
    if not ok:
        raise ReleaseError(f"the release's fly is not deployable on this code: {why}")
    v = seed["verdict"]
    if not v.get("passed") or v.get("data") != fly_selector.FLY_VERSION:
        raise ReleaseError("the release's replay verdict did not pass on this code's definitions")
    ids: dict[str, int] = {}
    with transaction() as conn:
        for snap in seed["snapshots"]:
            r = conn.execute("SELECT id FROM brain_snapshots WHERE kind = %s AND sha256 = %s ORDER BY id DESC LIMIT 1",
                             (snap["kind"], snap["sha256"])).fetchone()
            if r is None:
                note = snap["note"] if isinstance(snap["note"], str) else json.dumps(snap["note"], default=str)
                r = conn.execute("INSERT INTO brain_snapshots (path, sha256, kind, note, promoted_by) VALUES (%s,%s,%s,%s,'release') RETURNING id",
                                 (str(placed[snap["sha256"]]), snap["sha256"], snap["kind"], note)).fetchone()
            ids[snap["sha256"]] = int(r["id"])
        _set(conn, "fly_replay", v)
        pin = seed.get("pinned_selector_sha256")
        if pin:
            _set(conn, "pinned_selector_snapshot", {"id": ids[pin], "why": f"release {seed.get('seq')}", "pinned_at": time.strftime("%Y-%m-%d")})
        if bootstrap and seed.get("autostart"):
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('autostart', %s) ON CONFLICT (key) DO NOTHING", (json.dumps(seed["autostart"]),))
        _set(conn, "release", {"seq": seed.get("seq"), "git_commit": seed.get("git_commit"), "applied_at": int(time.time()),
                               "fly": ids.get(next((s["sha256"] for s in seed["snapshots"] if s["kind"] == "fly_selector"), ""))})
        _set(conn, "release_applied_at", int(time.time()))
    record_event("info", "release", f"release {seed.get('seq')} applied", {"ids": ids})
    try:
        from ..vault import alerts
        alerts.send(f"release {seed.get('seq')} applied: snapshots {sorted(ids.values())}")
    except Exception:
        pass
    return {"seq": seed.get("seq"), "snapshots": ids}


def ready_for_restart() -> tuple[bool, str]:
    """Safe to recreate the container: no order or claim in flight and no settlement between its two phases."""
    with transaction() as conn:
        o = conn.execute("SELECT count(*) AS n FROM orders WHERE book = 'live' AND status IN ('ordered', 'executed') "
                         "AND ts > now() - interval '10 minutes'").fetchone()["n"]
        c = conn.execute("SELECT count(*) AS n FROM vault_claims WHERE status = 'sending'").fetchone()["n"]
        s = conn.execute("SELECT count(*) AS n FROM vault_settlements WHERE status = 'snapshotted'").fetchone()["n"]
    busy = [f"{o} order(s)" if o else "", f"{c} claim payment(s)" if c else "", "a settlement in progress" if s else ""]
    busy = [b for b in busy if b]
    return (not busy, "ready" if not busy else "busy: " + ", ".join(busy))


def health() -> tuple[bool, str]:
    """For the container HEALTHCHECK and the updater: the database answers and the console's workers heartbeat."""
    try:
        with transaction() as conn:
            r = conn.execute("SELECT max(ts) AS t FROM events WHERE ts > now() - interval '15 minutes'").fetchone()
            w = conn.execute("SELECT count(*) AS n FROM processes WHERE stopped_at IS NULL AND cmd[1] = 'thread'").fetchone()["n"]
    except Exception as e:
        return False, f"database: {type(e).__name__}"
    if not w:
        return False, "no worker threads running"
    return True, f"{w} worker threads; last event {r['t']}"
