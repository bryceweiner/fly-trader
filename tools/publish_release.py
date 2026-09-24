"""Publish a signed release of the distribution to Hugging Face (manual, run on the Mac; plan phase 7).

    .venv/bin/python tools/publish_release.py --worktree ../fly-dist --key ~/.ssh/fly_release [--models] [--dry-run]

1. ``--models``: run tools/make_seed.py into the distribution worktree (models/ and seed/ refreshed from this Mac)
2. run the vault + release tests inside the worktree, then commit there (skipped when nothing changed)
3. stage EXACTLY the release files (tracked files of the worktree + the wallet-skill parquet) in a fresh directory
4. write ``release.json`` {schema, seq, prev_sha256, created_at, git_commit, code_digest, fly_version, files} and sign it
   with ``ssh-keygen -Y sign -n fly-trader-release`` (the key never leaves the Mac)
5. verify the staged release with the server's own verifier (deploy/fly_update.py verify)
6. push the worktree's branch to GitHub and upload the staged directory to HF as ONE commit

The server accepts a release only if its signature verifies against /etc/fly/allowed_signers, its seq is larger than
the installed one and ``prev_sha256`` is the installed manifest's hash, so a release must build on the one before.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deploy"))
import fly_update  # noqa: E402  (the server's verifier, reused so both sides agree)


def sh(*args, cwd=None, check=True, capture=True) -> str:
    r = subprocess.run(list(args), cwd=cwd, check=check, capture_output=capture, text=True)
    return (r.stdout or "").strip()


def previous(repo: str) -> tuple[int, str | None]:
    """(seq, sha256 of release.json) of the release currently on HF, or (0, None)."""
    try:
        raw = urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/release.json", timeout=30).read()
    except Exception:
        return 0, None
    return int(json.loads(raw)["seq"]), hashlib.sha256(raw).hexdigest()


def stage(worktree: Path, dest: Path) -> list[dict]:
    files = [p for p in sh("git", "ls-files", "-z", cwd=worktree).split("\0") if p]
    files += [p.relative_to(worktree).as_posix() for p in (worktree / "models" / "wallet_skill").glob("*.parquet")]
    out = []
    for rel in sorted(set(files)):
        src = worktree / rel
        if not src.is_file() or src.is_symlink():
            continue
        if not fly_update.PATH_RE.match(rel):
            raise SystemExit(f"path not allowed in a release: {rel}")
        dst = dest / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        out.append({"path": rel, "sha256": fly_update.sha256_file(dst), "size": dst.stat().st_size, "class": fly_update.classify(rel)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worktree", type=Path, required=True, help="a checkout of the distribution branch")
    ap.add_argument("--key", type=Path, required=True, help="ssh ed25519 private key that signs releases")
    ap.add_argument("--principal", default="bryce")
    ap.add_argument("--repo", default="bryceweiner/fly-trader")
    ap.add_argument("--models", action="store_true", help="refresh models/ and seed/ from this Mac first (make_seed)")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build, sign and verify, but push and upload nothing")
    a = ap.parse_args()
    wt = a.worktree.resolve()
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=wt) != "distribution":
        raise SystemExit("the worktree must be on the distribution branch")
    if a.models:
        sh(sys.executable, str(REPO / "tools" / "make_seed.py"), "--out", str(wt), capture=False)
    if not a.skip_tests:
        sh(sys.executable, "-m", "pytest", "-q", "tests/test_vault_core.py", "tests/test_vault_claims.py", "tests/test_release.py",
           cwd=wt, capture=False)
    if sh("git", "status", "--porcelain", cwd=wt):
        sh("git", "add", "-A", cwd=wt)
        sh("git", "commit", "-m", f"Release {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}", cwd=wt, capture=False)
    commit = sh("git", "rev-parse", "HEAD", cwd=wt)
    seq, prev = previous(a.repo)
    with tempfile.TemporaryDirectory() as d:
        st = Path(d) / "release"; st.mkdir()
        files = stage(wt, st)
        code = hashlib.sha256("".join(f["path"] + f["sha256"] for f in files if f["class"] in ("code", "infra")).encode()).hexdigest()
        fly_version = None
        rs = st / "seed" / "release_seed.json"
        if rs.exists():
            fly_version = next((json.loads(s["note"]).get("data") for s in json.loads(rs.read_text())["snapshots"]
                                if s["kind"] == "fly_selector" and isinstance(s["note"], str)), None)
        man = {"schema": 1, "seq": seq + 1, "prev_sha256": prev, "created_at": int(time.time()), "git_commit": commit,
               "code_digest": code, "fly_version": fly_version, "files": files}
        (st / "release.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")
        sh("ssh-keygen", "-Y", "sign", "-f", str(a.key), "-n", "fly-trader-release", str(st / "release.json"))
        pub = sh("ssh-keygen", "-y", "-f", str(a.key))
        signers = Path(d) / "allowed_signers"; signers.write_text(f"{a.principal} {pub}\n")
        print(sh(sys.executable, str(REPO / "deploy" / "fly_update.py"), "verify", str(st), str(signers), a.principal))
        classes = sorted({f["class"] for f in files})
        print(f"release {seq + 1}: {len(files)} files ({', '.join(classes)}), commit {commit[:10]}, prev {str(prev)[:12]}")
        if a.dry_run:
            print("dry run: nothing pushed or uploaded"); return
        sh("git", "push", "origin", "distribution", cwd=wt, capture=False)
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(folder_path=str(st), repo_id=a.repo, repo_type="model", commit_message=f"release {seq + 1} ({commit[:10]})",
                          delete_patterns=["*"])        # one commit; files no longer in the release are removed
        print(f"uploaded release {seq + 1} to https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
