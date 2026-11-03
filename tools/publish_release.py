"""Publish a signed release of the distribution to Hugging Face (manual).

    .venv/bin/python tools/publish_release.py --worktree ../fly-dist --key fly-release.pub [--models] [--dry-run]

``--key`` is normally a PUBLIC key whose private half lives in ssh-agent (Secretive: Secure Enclave, Touch ID per
signature; later the Ledger's agent), so the signing key never exists as a file. A private key file also works.

1. ``--models``: run tools/make_seed.py into the distribution worktree (models/ and seed/ refreshed from this Mac)
2. run the release tests inside the worktree, then commit there (skipped when nothing changed)
3. stage EXACTLY the release files (tracked files of the worktree + the wallet-skill table named in TABLE) in a fresh
   directory; the signing key must be listed in the worktree's release/allowed_signers (self-hosted installs check it)
4. write ``release.json`` {schema, seq, prev_sha256, created_at, git_commit, code_digest, fly_version, files} and sign it
   with ``ssh-keygen -Y sign -n fly-trader-release`` (the key never leaves the Mac or its Secure Enclave)
5. verify the staged release with the installs' own verifier (deploy/fly_update.py verify)
6. push one snapshot commit to the public GitHub repo and upload the staged directory to HF as ONE commit

An install accepts a release only if its signature verifies against that install's release/allowed_signers and its seq
is larger than the installed one (fly_trader/ops/autoupdate.py).
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
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "deploy"))
import fly_update  # noqa: E402  (the installs' verifier, reused so both sides agree)


# Hugging Face reads a model card's metadata from YAML at the top of README.md; GitHub would show it as a table, so it is
# added to the staged copy only (the manual uploads before signed releases prepended it by hand)
HF_CARD = """---
license: mit
pipeline_tag: reinforcement-learning
library_name: pytorch
tags:
  - solana
  - memecoin
  - pump.fun
  - robinhood-chain
  - trading
  - connectome
  - drosophila
  - flywire
---

"""


def sh(*args, cwd=None, check=True, capture=True) -> str:
    r = subprocess.run(list(args), cwd=cwd, check=check, capture_output=capture, text=True)
    return (r.stdout or "").strip()


def previous(repo: str) -> tuple[int, str | None]:
    """(seq, sha256 of release.json) of the release currently on HF, or (0, None) when HF has none (404). Any other failure
    stops the publish: treating an unreadable release as "none" would publish seq 1 again, which every server refuses."""
    import ssl
    try:
        import certifi                       # python.org's Python on macOS ships without root certificates
        ctx = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        ctx = ssl.create_default_context()
    try:
        raw = urllib.request.urlopen(f"https://huggingface.co/{repo}/resolve/main/release.json", timeout=30, context=ctx).read()
    except urllib.error.HTTPError as e:
        if e.code == 404:
            return 0, None
        raise SystemExit(f"cannot read the current release from Hugging Face: HTTP {e.code}")
    except OSError as e:
        raise SystemExit(f"cannot read the current release from Hugging Face: {e}")
    return int(json.loads(raw)["seq"]), hashlib.sha256(raw).hexdigest()


def push_snapshot(worktree: Path, commit: str, seq: int) -> None:
    """GitHub gets the release's tree as one commit on top of the last release's: the working history stays on this machine."""
    head = sh("git", "ls-remote", "public", "refs/heads/distribution", cwd=worktree).split()
    if head:
        sh("git", "fetch", "-q", "public", "distribution", cwd=worktree)
    snap = sh("git", "commit-tree", f"{commit}^{{tree}}", *(["-p", head[0]] if head else []), "-m", f"Release {seq}", cwd=worktree)
    sh("git", "push", "public", f"{snap}:refs/heads/distribution", cwd=worktree, capture=False)


def stage(worktree: Path, dest: Path) -> list[dict]:
    files = [p for p in sh("git", "ls-files", "-z", cwd=worktree).split("\0") if p]
    table = (worktree / "models" / "wallet_skill" / "TABLE")
    if table.exists():                                  # the one table the release names (git has only its name and sha256)
        name = table.read_text().strip()
        if not (worktree / "models" / "wallet_skill" / name).is_file():
            raise SystemExit(f"models/wallet_skill/{name} (named in TABLE) is not in the worktree: publish with --models, "
                             "or put that file there")
        files.append(f"models/wallet_skill/{name}")
    out = []
    for rel in sorted(set(files)):
        src = worktree / rel
        if not src.is_file() or src.is_symlink() or Path(rel).name.startswith(".git"):
            continue                                    # .gitignore & co.: fly_update refuses them, and HF honours an uploaded .gitignore
        if not fly_update.PATH_RE.match(rel):
            raise SystemExit(f"path not allowed in a release: {rel}")
        dst = dest / rel; dst.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(src, dst)
        if rel == "README.md" and not dst.read_text().startswith("---"):
            dst.write_text(HF_CARD + dst.read_text())
        out.append({"path": rel, "sha256": fly_update.sha256_file(dst), "size": dst.stat().st_size, "class": fly_update.classify(rel)})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--worktree", type=Path, required=True, help="a checkout of the release branch")
    ap.add_argument("--key", type=Path, required=True, help="the release key: a .pub whose private key is in ssh-agent (Secretive), or a private key file")
    ap.add_argument("--principal", default="bryce")
    ap.add_argument("--repo", default="bryceweiner/fly-trader")
    ap.add_argument("--models", action="store_true", help="refresh models/ and seed/ from this Mac first (make_seed)")
    ap.add_argument("--skip-tests", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="build, sign and verify, but push and upload nothing")
    a = ap.parse_args()
    wt = a.worktree.resolve()
    if sh("git", "rev-parse", "--abbrev-ref", "HEAD", cwd=wt) != "release":
        raise SystemExit("the worktree must be on the release branch")
    if a.models:
        sh(sys.executable, str(REPO / "tools" / "make_seed.py"), "--out", str(wt), capture=False)
    if not a.skip_tests:
        sh(sys.executable, "-m", "pytest", "-q", "tests/test_release.py", "tests/test_model_io.py",
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
        print("signing release.json (approve the Touch ID / agent prompt)")
        sh("ssh-keygen", "-Y", "sign", "-f", str(a.key), "-n", "fly-trader-release", str(st / "release.json"))
        pub = a.key.read_text().strip() if a.key.suffix == ".pub" else sh("ssh-keygen", "-y", "-f", str(a.key))
        trusted = (wt / "release" / "allowed_signers")
        material = pub.split()[1] if len(pub.split()) > 1 else pub
        if not trusted.exists() or material not in trusted.read_text():
            raise SystemExit("this key is not in the distribution's release/allowed_signers: self-hosted installs would refuse the "
                             "release. Add it there (and commit) first")
        signers = Path(d) / "allowed_signers"; signers.write_text(f'{a.principal} namespaces="fly-trader-release" {pub}\n')
        print(sh(sys.executable, str(REPO / "deploy" / "fly_update.py"), "verify", str(st), str(signers), a.principal))
        classes = sorted({f["class"] for f in files})
        print(f"release {seq + 1}: {len(files)} files ({', '.join(classes)}), commit {commit[:10]}, prev {str(prev)[:12]}")
        print(f"manifest sha256 (a new server's EXPECT_MANIFEST): {hashlib.sha256((st / 'release.json').read_bytes()).hexdigest()}")
        if a.dry_run:
            print("dry run: nothing pushed or uploaded"); return
        push_snapshot(wt, commit, seq + 1)
        from huggingface_hub import HfApi
        api = HfApi()
        api.upload_folder(folder_path=str(st), repo_id=a.repo, repo_type="model", commit_message=f"release {seq + 1} ({commit[:10]})",
                          delete_patterns=["*"])        # one commit; files no longer in the release are removed
        print(f"uploaded release {seq + 1} to https://huggingface.co/{a.repo}")


if __name__ == "__main__":
    main()
