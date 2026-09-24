"""Keep a fire-and-forget install current. Every ``UPDATE_INTERVAL_S`` (an hour) ask Hugging Face for the head commit
of the model repo this install came from; when it has moved, stop the console, download the new snapshot over ``/app``
(code, models, seed -- the repo is the whole distribution), reinstall the package, rerun the bootstrap steps and start
the console again. The previous tree is kept beside ``/app``: if the new console is not answering within
``HEALTH_WAIT_S`` it is put back, and that revision is never tried again.

``docker/entrypoint.sh`` runs this as pid 1 when AUTO_UPDATE=1; the console is its child and is restarted if it dies.
A restart takes about a minute; the paper books and the fly's learned state live in the database and the data volume,
and open live positions stay in the wallet, sold on schedule by the restarted engine (README: "Stop everything").
"""
from __future__ import annotations

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
for _noisy in ("httpx", "huggingface_hub", "urllib3"):        # the download's every request is not news
    logging.getLogger(_noisy).setLevel(logging.WARNING)

APP = Path(os.environ.get("APP_DIR", "/app"))
PREV = APP.with_name(APP.name + ".prev")
REPO = os.environ.get("HF_REPO", "bryceweiner/fly-trader")
INTERVAL_S = float(os.environ.get("UPDATE_INTERVAL_S", "3600"))
PORT = os.environ.get("PORT", "8501")
HEALTH_WAIT_S = float(os.environ.get("UPDATE_HEALTH_WAIT_S", "300"))
REV_FILE = APP / "data" / "hf_revision"                      # the revision this install runs; on the data volume, survives recreates
BAD_FILE = APP / "data" / "hf_bad_revisions"
KEEP_OUT = ("data", "logs", ".venv", ".cache", ".env")        # never part of a backup or a restore: the install's own state


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


def wants_update(head: str | None, current: str | None, skip: set[str]) -> bool:
    """A new head that is not the installed one and has not already failed."""
    return bool(head) and head != current and head not in skip


def sh(*cmd: str, cwd: Path = APP) -> None:
    subprocess.run(list(cmd), cwd=str(cwd), check=True)


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


def download(sha: str) -> None:
    from huggingface_hub import snapshot_download
    snapshot_download(REPO, revision=sha, local_dir=str(APP))


def update(sha: str, proc: subprocess.Popen | None) -> subprocess.Popen:
    """Stop, replace, reinstall, bootstrap, start; on any failure put the previous tree back and start that."""
    log.info("Hugging Face %s moved to %s: updating", REPO, sha[:12])
    stop_console(proc); backup()
    try:
        download(sha); sh("uv", "pip", "install", "-e", ".", "-q"); bootstrap()
        p = start_console()
        if healthy():
            record(sha); log.info("updated to %s", sha[:12]); return p
        log.error("the new console did not answer within %.0fs", HEALTH_WAIT_S); stop_console(p)
    except Exception:
        log.exception("update to %s failed", sha[:12])
    mark_bad(sha); log.error("restoring the previous install; %s will not be tried again", sha[:12])
    restore(); sh("uv", "pip", "install", "-e", ".", "-q"); bootstrap()
    return start_console()


def main() -> None:
    bootstrap()
    if installed() is None:
        head = head_sha()
        if head:
            record(head); log.info("this install is taken to be Hugging Face %s @ %s", REPO, head[:12])
    proc = start_console()
    stopping = {"now": False}

    def _term(signum, frame):
        stopping["now"] = True
    signal.signal(signal.SIGTERM, _term); signal.signal(signal.SIGINT, _term)
    last = time.time()
    while not stopping["now"]:
        time.sleep(5)
        if proc.poll() is not None:
            log.error("console exited with %s; restarting it", proc.returncode); proc = start_console(); continue
        if time.time() - last < INTERVAL_S:
            continue
        last = time.time(); head = head_sha()
        if wants_update(head, installed(), bad()):
            proc = update(head, proc)
    stop_console(proc)


if __name__ == "__main__":
    main()
