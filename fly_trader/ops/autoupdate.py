"""Keep a fire-and-forget install current, with SIGNED releases only. Every ``UPDATE_INTERVAL_S`` (an hour) ask Hugging
Face for the head commit of the repo this install came from; when it has moved:

1. fetch its ``release.json`` + ``release.json.sig`` and verify the signature against ``release/allowed_signers``
   of the INSTALLED tree (the release keys this install trusts; only a signed release can change them), with the
   same code the vault server's updater uses (``deploy/fly_update.py``: namespace, principals, strict paths, exactly
   the listed files, each file's size and sha256)
2. refuse a release whose ``seq`` is not larger than the installed one (no rollback to an old signed release)
3. stop the console, keep the current tree beside ``/app``, copy the verified files over it, reinstall exactly
   ``uv.lock`` (``uv sync --frozen``), rerun the bootstrap steps and start the console again; if it is not answering
   within ``HEALTH_WAIT_S`` the previous tree is put back and that revision is never tried again

An unsigned or wrongly signed commit is logged once and never installed. With no release keys in
``release/allowed_signers`` the updater installs nothing. ``docker/entrypoint.sh`` runs this as pid 1 when
AUTO_UPDATE=1; the console is its child and is restarted if it dies. The vault server never runs it (AUTO_UPDATE=0
there: its host updater, deploy/fly_update.py, applies releases with a 24 h hold on code).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

log = logging.getLogger("autoupdate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s autoupdate %(levelname)s %(message)s", stream=sys.stdout)
for _noisy in ("httpx", "huggingface_hub", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

APP = Path(os.environ.get("APP_DIR", "/app"))
PREV = APP.with_name(APP.name + ".prev")
REPO = os.environ.get("HF_REPO", "bryceweiner/fly-trader")
INTERVAL_S = float(os.environ.get("UPDATE_INTERVAL_S", "3600"))
PORT = os.environ.get("PORT", "8501")
HEALTH_WAIT_S = float(os.environ.get("UPDATE_HEALTH_WAIT_S", "300"))
TORCH_INDEX = os.environ.get("TORCH_INDEX", "cpu")
REV_FILE = APP / "data" / "hf_revision"                      # the revision this install runs; on the data volume, survives recreates
BAD_FILE = APP / "data" / "hf_bad_revisions"
RELEASE_FILE = APP / "data" / "hf_release.json"              # {"seq", "manifest_sha256"} of the installed signed release
STAGE = APP / "data" / "updates"                             # verified downloads, before they replace the tree
KEEP_OUT = ("data", "logs", ".venv", ".cache", ".env")        # never part of a backup or a restore: the install's own state
PRINCIPALS = ["bryce", "bryce-recovery"]
NAMESPACE = "fly-trader-release"


class Refused(RuntimeError):
    pass


def head_sha(repo: str = REPO) -> str | None:
    """The repo's head commit on Hugging Face, or None when it cannot be asked (offline, rate-limited)."""
    try:
        from huggingface_hub import HfApi
        return HfApi().model_info(repo).sha
    except Exception as e:
        log.warning("could not ask Hugging Face for %s: %s", repo, e)
        return None


def installed() -> str | None:
    try:
        return REV_FILE.read_text().strip() or None
    except FileNotFoundError:
        return None


def record(sha: str) -> None:
    REV_FILE.parent.mkdir(parents=True, exist_ok=True); REV_FILE.write_text(sha + "\n")


def bad() -> set[str]:
    try:
        return {s for s in BAD_FILE.read_text().split() if s}
    except FileNotFoundError:
        return set()


def mark_bad(sha: str) -> None:
    BAD_FILE.parent.mkdir(parents=True, exist_ok=True)
    with BAD_FILE.open("a") as f:
        f.write(sha + "\n")


def installed_release() -> dict:
    try:
        return json.loads(RELEASE_FILE.read_text())
    except (FileNotFoundError, ValueError):
        return {"seq": 0}


def wants_update(head: str | None, current: str | None, skip: set[str]) -> bool:
    """A new head that is not the installed one and has not already failed."""
    return bool(head) and head != current and head not in skip


def signers() -> Path | None:
    """The installed tree's release keys, or None when it holds none (then nothing is ever installed)."""
    p = APP / "release" / "allowed_signers"
    try:
        lines = [l for l in p.read_text().splitlines() if l.strip() and not l.lstrip().startswith("#")]
    except FileNotFoundError:
        return None
    return p if lines else None


_FU = None                                                   # tests put a stand-in here


def _verifier():
    """deploy/fly_update.py of the INSTALLED tree: the vault server's own release checks."""
    if _FU is not None:
        return _FU
    spec = importlib.util.spec_from_file_location("fly_update_installed", APP / "deploy" / "fly_update.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def fetch_verified(sha: str) -> tuple[dict, Path, bytes, str]:
    """Download and verify the release at ``sha``; returns (manifest, directory, manifest bytes, signer)."""
    keys = signers()
    if keys is None:
        raise Refused("no release keys in release/allowed_signers: this install accepts no update")
    fu = _verifier()
    c = {"repo": REPO, "allowed_signers": str(keys), "principals": PRINCIPALS, "namespace": NAMESPACE, "root": str(STAGE)}
    try:
        m, d, raw, who = fu.fetch(c, sha)                      # signature, manifest, paths, sizes, sha256
    except SystemExit as e:                                   # fly_update reports refusals as SystemExit
        raise Refused(str(e)) from None
    have = int(installed_release().get("seq") or 0)
    if int(m["seq"]) <= have:
        raise Refused(f"release {m['seq']} is not newer than the installed {have}")
    return m, d, raw, who


def sh(*cmd: str, cwd: Path = APP) -> None:
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


def reinstall() -> None:
    sh("uv", "sync", "--frozen", "--no-dev", "--extra", TORCH_INDEX, "-q")


def bootstrap() -> None:
    sh("sh", str(APP / "docker" / "bootstrap.sh"))


def start_console() -> subprocess.Popen:
    log.info("console at http://localhost:%s", PORT)
    return subprocess.Popen(["fly-trader", "ui", "--port", PORT], cwd=str(APP))


def stop_console(p: subprocess.Popen | None, grace_s: float = 90.0) -> None:
    if p is None or p.poll() is not None:
        return
    p.terminate()
    try:
        p.wait(grace_s)
    except subprocess.TimeoutExpired:
        log.warning("console did not stop in %.0fs; killing it", grace_s); p.kill(); p.wait(30)


def healthy(timeout_s: float = HEALTH_WAIT_S) -> bool:
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(5)
    return False


def _code_files() -> list[Path]:
    return [p for p in APP.iterdir() if p.name not in KEEP_OUT]


def backup() -> None:
    """A copy of the install's code, models and seed (not its state) beside it."""
    shutil.rmtree(PREV, ignore_errors=True); PREV.mkdir()
    for p in _code_files():
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, PREV / p.name)


def restore() -> None:
    for p in _code_files():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for p in PREV.iterdir():
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, APP / p.name)


def install_tree(d: Path) -> None:
    """Replace the code, models and seed with exactly the verified release (the install's state is kept)."""
    for p in _code_files():
        shutil.rmtree(p) if p.is_dir() else p.unlink()
    for p in d.iterdir():
        if p.name in KEEP_OUT:
            continue
        (shutil.copytree if p.is_dir() else shutil.copy2)(p, APP / p.name)


def update(sha: str, proc: subprocess.Popen | None) -> subprocess.Popen | None:
    """Verify, stop, replace, reinstall, bootstrap, start; on a failure after the swap put the previous tree back."""
    try:
        m, d, raw, who = fetch_verified(sha)
    except Refused as e:
        log.error("Hugging Face %s @ %s refused: %s", REPO, sha[:12], e)
        mark_bad(sha)
        return proc
    except Exception:
        log.exception("could not fetch %s; will retry", sha[:12])
        return proc
    log.info("release %s (Hugging Face %s), signed by %s: updating", m["seq"], sha[:12], who)
    stop_console(proc); backup()
    try:
        install_tree(d); reinstall(); bootstrap()
        p = start_console()
        if healthy():
            record(sha)
            RELEASE_FILE.write_text(json.dumps({"seq": int(m["seq"]), "manifest_sha256": hashlib.sha256(raw).hexdigest()}) + "\n")
            shutil.rmtree(d, ignore_errors=True)
            log.info("updated to release %s", m["seq"]); return p
        log.error("the new console did not answer within %.0fs", HEALTH_WAIT_S); stop_console(p)
    except Exception:
        log.exception("update to %s failed", sha[:12])
    mark_bad(sha); log.error("restoring the previous install; %s will not be tried again", sha[:12])
    restore(); reinstall(); bootstrap()
    return start_console()


def main() -> None:
    bootstrap()
    if signers() is None:
        log.warning("release/allowed_signers holds no release key: automatic updates are off until it does")
    proc = start_console()
    stopping = {"now": False}

    def _term(signum, frame):
        stopping["now"] = True
    signal.signal(signal.SIGTERM, _term); signal.signal(signal.SIGINT, _term)
    last = 0.0                                                # a new install checks at once: it installs the newest signed release
    while not stopping["now"]:
        time.sleep(5)
        if proc is None or proc.poll() is not None:
            log.error("console exited; restarting it"); proc = start_console(); continue
        if time.time() - last < INTERVAL_S:
            continue
        last = time.time(); head = head_sha()
        if wants_update(head, installed(), bad()):
            proc = update(head, proc)
    stop_console(proc)


if __name__ == "__main__":
    main()
