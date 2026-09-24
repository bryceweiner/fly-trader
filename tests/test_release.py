"""Signed releases: what the server's updater (deploy/fly_update.py) accepts and refuses."""
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / "deploy"))
import fly_update as fu  # noqa: E402


@pytest.fixture()
def signer(tmp_path):
    key = tmp_path / "k"
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(key)], check=True)
    signers = tmp_path / "allowed_signers"
    signers.write_text("bryce " + (tmp_path / "k.pub").read_text())
    return key, {"allowed_signers": str(signers), "principal": "bryce", "namespace": "fly-trader-release"}


def make_release(d: Path, key: Path, files: dict, seq=1, prev=None) -> bytes:
    d.mkdir(parents=True, exist_ok=True)
    entries = []
    for rel, data in files.items():
        p = d / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(data)
        entries.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "class": fu.classify(rel)})
    raw = json.dumps({"schema": 1, "seq": seq, "prev_sha256": prev, "files": entries}, sort_keys=True).encode()
    (d / "release.json").write_bytes(raw)
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "fly-trader-release", str(d / "release.json")], check=True, capture_output=True)
    return raw


FILES = {"fly_trader/x.py": b"print(1)\n", "models/policies/f.pt": b"\x00" * 64, "Dockerfile": b"FROM x\n", "seed/release_seed.json": b"{}"}


def test_good_release_verifies(signer, tmp_path):
    key, c = signer
    raw = make_release(tmp_path / "r", key, FILES)
    fu.verify_signature(c, raw, (tmp_path / "r" / "release.json.sig").read_bytes())
    m = json.loads(raw); fu.check_manifest(m); fu.verify_dir(tmp_path / "r", m)


def test_tampered_extra_and_foreign_key_refused(signer, tmp_path):
    key, c = signer
    raw = make_release(tmp_path / "r", key, FILES)
    m = json.loads(raw)
    (tmp_path / "r" / "fly_trader" / "x.py").write_bytes(b"import os; os.system('evil')\n")
    with pytest.raises(SystemExit):
        fu.verify_dir(tmp_path / "r", m)
    (tmp_path / "r" / "fly_trader" / "x.py").write_bytes(FILES["fly_trader/x.py"])
    (tmp_path / "r" / "sitecustomize.py").write_text("evil")
    with pytest.raises(SystemExit):
        fu.verify_dir(tmp_path / "r", m)
    with pytest.raises(SystemExit):                            # manifest edited after signing
        fu.verify_signature(c, raw.replace(b'"seq": 1', b'"seq": 9'), (tmp_path / "r" / "release.json.sig").read_bytes())
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "other")], check=True)
    raw2 = make_release(tmp_path / "r2", tmp_path / "other", FILES)
    with pytest.raises(SystemExit):
        fu.verify_signature(c, raw2, (tmp_path / "r2" / "release.json.sig").read_bytes())


def test_bad_paths_and_classes(signer):
    for p in ("../etc/passwd", "/abs", "a/../b", ".git/config"):
        with pytest.raises(SystemExit):
            fu.check_manifest({"schema": 1, "seq": 1, "files": [{"path": p, "sha256": "0" * 64, "size": 1, "class": "code"}]})
    with pytest.raises(SystemExit):                            # infra disguised as code
        fu.check_manifest({"schema": 1, "seq": 1, "files": [{"path": "docker/entrypoint.sh", "sha256": "0" * 64, "size": 1, "class": "code"}]})


def test_changed_classes_decide_the_path():
    old = {"files": [{"path": "models/a.pt", "sha256": "1"}, {"path": "fly_trader/x.py", "sha256": "2"}, {"path": "Dockerfile", "sha256": "3"}]}
    new = {"files": [{"path": "models/a.pt", "sha256": "9"}, {"path": "fly_trader/x.py", "sha256": "2"}, {"path": "Dockerfile", "sha256": "3"}]}
    assert fu.changed_classes(old, new) == {"model"}
    new["files"][2]["sha256"] = "4"
    assert fu.changed_classes(old, new) == {"model", "infra"}


def test_rollback_and_fork_refused(tmp_path, monkeypatch):
    c = {"root": str(tmp_path), "project": "t"}
    (tmp_path / "releases" / "5").mkdir(parents=True)
    old = {"schema": 1, "seq": 5, "files": []}
    raw_old = json.dumps(old).encode(); (tmp_path / "releases" / "5" / "release.json").write_bytes(raw_old)
    st = {"seq": 5, "manifest_sha256": hashlib.sha256(raw_old).hexdigest(), "image": 5}
    with pytest.raises(SystemExit, match="not newer"):
        fu.apply(c, st, {"schema": 1, "seq": 4, "prev_sha256": st["manifest_sha256"], "files": []}, tmp_path, b"x", "sha")
    with pytest.raises(SystemExit, match="chain"):
        fu.apply(c, st, {"schema": 1, "seq": 6, "prev_sha256": "f" * 64, "files": []}, tmp_path, b"x", "sha")


def test_infra_change_waits_for_approval(tmp_path, monkeypatch):
    c = {"root": str(tmp_path), "project": "t"}
    (tmp_path / "releases" / "1").mkdir(parents=True)
    old = {"schema": 1, "seq": 1, "files": [{"path": "Dockerfile", "sha256": "a"}]}
    raw_old = json.dumps(old).encode(); (tmp_path / "releases" / "1" / "release.json").write_bytes(raw_old)
    st = {"seq": 1, "manifest_sha256": hashlib.sha256(raw_old).hexdigest(), "image": 1}
    monkeypatch.setattr(fu, "deploy_code", lambda *a, **k: pytest.fail("must not deploy infra without approval"))
    new = {"schema": 1, "seq": 2, "prev_sha256": st["manifest_sha256"], "files": [{"path": "Dockerfile", "sha256": "b"}]}
    fu.apply(c, st, new, tmp_path, b"raw", "hf2")
    assert json.loads((tmp_path / "state.json").read_text())["pending"]["seq"] == 2
