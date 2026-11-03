#!/usr/bin/env python3
"""Fetch and verify a signed release from Hugging Face. The self-updater (fly_trader/ops/autoupdate.py) loads this file from
the installed tree and calls ``fetch`` before it installs anything; tools/publish_release.py runs ``verify`` before it
uploads. Standard library only.

``fetch`` takes one immutable commit of the HF model repo:
 1. fetch ``release.json`` + ``release.json.sig`` and verify the signature with ``ssh-keygen -Y verify``
 2. download exactly the listed paths (strict path check) into a fresh directory, verify size + sha256 of each

    fly_update.py verify <dir> <allowed_signers> [principal ...]
"""
from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path

PATH_RE = re.compile(r"^(?!.*\.\.)(?!/)[A-Za-z0-9_.\-/]+$")
INFRA = ("Dockerfile", "docker-compose.yml", "docker-compose.cuda.yml", "docker-compose.rocm.yml", ".dockerignore", "docker/", "deploy/")
# what models/ and seed/ may hold: nothing a loader could execute (fly_trader/train/model_io.SAFE_SUFFIXES)
MODEL_SUFFIXES = {".pt", ".skops", ".npz", ".json", ".bin", ".parquet", ".txt", ".sha256", ""}
SEED_SUFFIXES = {".json", ".sql"}


def http(url: str, dest: Path | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fly-updater/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if dest is None:
            return r.read()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    return b""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: str) -> str:
    if path.startswith("models/"):
        return "model"
    if path.startswith("seed/"):
        return "seed"
    if path in INFRA or any(path.startswith(p) for p in INFRA if p.endswith("/")):
        return "infra"
    return "code"


def verify_signature(c: dict, manifest: bytes, sig: bytes) -> str:
    """The principal whose key signed ``manifest`` (any of ``principals``), else SystemExit."""
    principals = c.get("principals") or [c["principal"]]
    err = ""
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "release.json.sig"; sp.write_bytes(sig)
        for who in principals:
            r = subprocess.run(["ssh-keygen", "-Y", "verify", "-f", c["allowed_signers"], "-I", who, "-n", c["namespace"], "-s", str(sp)],
                               input=manifest, capture_output=True)
            if r.returncode == 0:
                return who
            err = r.stderr.decode(errors="replace").strip()
    raise SystemExit(f"release signature does NOT verify: {err[:300]}")


def check_manifest(m: dict) -> None:
    if m.get("schema") != 1 or not isinstance(m.get("seq"), int) or not isinstance(m.get("files"), list):
        raise SystemExit("malformed release manifest")
    for f in m["files"]:
        p = f.get("path", "")
        if not PATH_RE.match(p) or p.startswith(".git") or "//" in p:
            raise SystemExit(f"refusing path {p!r}")
        if f.get("class") != classify(p):
            raise SystemExit(f"{p}: class {f.get('class')} does not match {classify(p)}")
        if f["class"] == "model" and Path(p).suffix not in MODEL_SUFFIXES or f["class"] == "seed" and Path(p).suffix not in SEED_SUFFIXES:
            raise SystemExit(f"{p}: file type not allowed in {f['class']}/ (a model or seed must not be able to run code)")
        if not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")) or not isinstance(f.get("size"), int):
            raise SystemExit(f"{p}: bad hash or size")


def verify_dir(d: Path, m: dict) -> None:
    """Exactly the listed files, each with its size and hash (no extra files may ride along into a build)."""
    listed = {f["path"] for f in m["files"]}
    for p in d.rglob("*"):
        if p.is_symlink():
            raise SystemExit(f"symlink in release: {p}")
        rel = p.relative_to(d).as_posix()
        if p.is_file() and rel not in listed and rel not in ("release.json", "release.json.sig"):
            raise SystemExit(f"unlisted file in release: {rel}")
    for f in m["files"]:
        p = d / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
            raise SystemExit(f"{f['path']}: missing or hash/size mismatch")


def fetch(c: dict, sha: str) -> tuple[dict, Path, bytes, str]:
    base = f"https://huggingface.co/{c['repo']}/resolve/{sha}/"
    raw = http(base + "release.json"); sig = http(base + "release.json.sig")
    who = verify_signature(c, raw, sig)
    m = json.loads(raw)
    check_manifest(m)
    dest = Path(c["root"]) / "releases" / str(m["seq"])
    if dest.exists():
        try:                                     # already downloaded (a held or retried release): re-verify, reuse
            if (dest / "release.json").read_bytes() == raw:
                verify_dir(dest, m)
                return m, dest, raw, who
        except (SystemExit, OSError):
            pass
        shutil.rmtree(dest)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    for f in m["files"]:
        out = part / f["path"]; out.parent.mkdir(parents=True, exist_ok=True)
        http(base + f["path"], out, timeout=1800)
    (part / "release.json").write_bytes(raw); (part / "release.json.sig").write_bytes(sig)
    verify_dir(part, m)
    part.rename(dest)
    return m, dest, raw, who


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else ""
    if cmd != "verify":
        raise SystemExit(__doc__)
    d = Path(argv[2]); c = {"allowed_signers": argv[3], "principals": argv[4:] or ["bryce"], "namespace": "fly-trader-release"}
    raw = (d / "release.json").read_bytes()
    verify_signature(c, raw, (d / "release.json.sig").read_bytes())
    m = json.loads(raw); check_manifest(m); verify_dir(d, m)
    print(f"release {m['seq']}: signature and {len(m['files'])} files verify")


if __name__ == "__main__":
    main(sys.argv)
