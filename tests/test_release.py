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
    signers.write_text('bryce namespaces="fly-trader-release" ' + (tmp_path / "k.pub").read_text())
    return key, {"allowed_signers": str(signers), "principals": ["bryce", "bryce-recovery"], "namespace": "fly-trader-release"}


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


def installed(tmp_path, seq=1, files=None, **conf):
    """An install at ``seq`` whose manifest lists ``files``; returns (conf, state, manifest sha)."""
    c = {"root": str(tmp_path), "project": "t", **conf}
    (tmp_path / "releases" / str(seq)).mkdir(parents=True, exist_ok=True)
    raw = json.dumps({"schema": 1, "seq": seq, "files": files or [{"path": "fly_trader/x.py", "sha256": "a"}]}).encode()
    (tmp_path / "releases" / str(seq) / "release.json").write_bytes(raw)
    sha = hashlib.sha256(raw).hexdigest()
    return c, {"seq": seq, "manifest_sha256": sha, "image": seq}, sha


def test_recovery_principal_verifies_and_namespace_is_enforced(signer, tmp_path):
    key, c = signer
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "rec")], check=True)
    with open(c["allowed_signers"], "a") as f:
        f.write('bryce-recovery namespaces="fly-trader-release" ' + (tmp_path / "rec.pub").read_text())
    raw = make_release(tmp_path / "r", tmp_path / "rec", FILES)
    assert fu.verify_signature(c, raw, (tmp_path / "r" / "release.json.sig").read_bytes()) == "bryce-recovery"
    (tmp_path / "m").write_bytes(raw)                                # same key, another namespace: refused
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(key), "-n", "file", str(tmp_path / "m")], check=True, capture_output=True)
    with pytest.raises(SystemExit):
        fu.verify_signature({**c, "namespace": "file"}, raw, (tmp_path / "m.sig").read_bytes())


def test_models_and_seed_cannot_carry_code():
    for p in ("models/selectors/s.joblib", "models/policies/p.pkl", "models/x.py", "seed/evil.py"):
        cls = fu.classify(p)
        with pytest.raises(SystemExit, match="not allowed"):
            fu.check_manifest({"schema": 1, "seq": 1, "files": [{"path": p, "sha256": "0" * 64, "size": 1, "class": cls}]})
    fu.check_manifest({"schema": 1, "seq": 1, "files": [{"path": "models/selectors/s.skops", "sha256": "0" * 64, "size": 1, "class": "model"},
                                                       {"path": "models/wallet_skill/TABLE", "sha256": "0" * 64, "size": 1, "class": "model"}]})


def test_first_release_must_be_pinned(tmp_path, monkeypatch):
    c = {"root": str(tmp_path), "project": "t"}
    monkeypatch.setattr(fu, "deploy_code", lambda *a, **k: "ok")
    m = {"schema": 1, "seq": 1, "files": []}
    with pytest.raises(SystemExit, match="pins none"):
        fu.apply(c, {"seq": 0}, m, tmp_path, b"raw", "hf1")
    with pytest.raises(SystemExit, match="not the pinned"):
        fu.apply({**c, "expect_manifest_sha256": "0" * 64}, {"seq": 0}, m, tmp_path, b"raw", "hf1")
    st = {"seq": 0}
    fu.apply({**c, "expect_manifest_sha256": hashlib.sha256(b"raw").hexdigest()}, st, m, tmp_path, b"raw", "hf1")
    assert st["seq"] == 1 and st["image"] == 1


def test_code_release_is_held_24h_then_applies(tmp_path, monkeypatch):
    c, st, prev = installed(tmp_path)
    deployed = []
    monkeypatch.setattr(fu, "deploy_code", lambda c_, st_, m, d, infra: deployed.append(m["seq"]) or "ok")
    new = {"schema": 1, "seq": 2, "prev_sha256": prev, "files": [{"path": "fly_trader/x.py", "sha256": "b"}]}
    fu.apply(c, st, new, tmp_path, b"raw2", "hf2", now=1000.0)
    assert deployed == [] and st["pending"]["due_at"] == 1000.0 + 86400
    fu.apply(c, st, new, tmp_path, b"raw2", "hf2", now=1000.0 + 86399)
    assert deployed == []
    fu.apply(c, st, new, tmp_path, b"raw2", "hf2", now=1000.0 + 86400)
    assert deployed == [2] and st["seq"] == 2 and st["pending"] is None


def test_approve_applies_code_at_once_but_infra_only_with_it(tmp_path, monkeypatch):
    c, st, prev = installed(tmp_path, files=[{"path": "Dockerfile", "sha256": "a"}])
    deployed = []
    monkeypatch.setattr(fu, "deploy_code", lambda c_, st_, m, d, infra: deployed.append((m["seq"], infra)) or "ok")
    new = {"schema": 1, "seq": 2, "prev_sha256": prev, "files": [{"path": "Dockerfile", "sha256": "b"}]}
    fu.apply(c, st, new, tmp_path, b"raw2", "hf2", now=0.0)
    fu.apply(c, st, new, tmp_path, b"raw2", "hf2", now=10 * 86400.0)      # long past the delay: infra still waits
    assert deployed == []
    fu.apply(c, st, new, tmp_path, b"raw2", "hf2", approved=True, now=10 * 86400.0)
    assert deployed == [(2, True)]


def test_newer_release_restarts_the_delay(tmp_path, monkeypatch):
    c, st, prev = installed(tmp_path)
    monkeypatch.setattr(fu, "deploy_code", lambda *a, **k: pytest.fail("held"))
    r2 = {"schema": 1, "seq": 2, "prev_sha256": prev, "files": [{"path": "fly_trader/x.py", "sha256": "b"}]}
    fu.apply(c, st, r2, tmp_path, b"raw2", "hf2", now=0.0)
    r3 = {"schema": 1, "seq": 3, "prev_sha256": hashlib.sha256(b"raw2").hexdigest(), "files": [{"path": "fly_trader/x.py", "sha256": "c"}]}
    fu.apply(c, st, r3, tmp_path, b"raw3", "hf3", now=86000.0)             # chains from the held one
    assert st["pending"]["seq"] == 3 and st["pending"]["due_at"] == 86000.0 + 86400


def test_veto(tmp_path, monkeypatch):
    c, st, prev = installed(tmp_path)
    monkeypatch.setattr(fu, "conf", lambda: c)
    monkeypatch.setattr(fu, "deploy_code", lambda *a, **k: pytest.fail("vetoed"))
    r2 = {"schema": 1, "seq": 2, "prev_sha256": prev, "files": [{"path": "fly_trader/x.py", "sha256": "evil"}]}
    fu.apply(c, st, r2, tmp_path, b"raw2", "hf2", now=0.0)
    fu.veto(2)
    st = fu.load_state(c)
    assert st["pending"] is None and st["hf_sha_rejected"] == "hf2" and hashlib.sha256(b"raw2").hexdigest() in st["vetoed"]
    with pytest.raises(SystemExit):
        fu.veto(2)
    # the next real release (built on top of the vetoed one on HF) is still accepted -- and held again
    r3 = {"schema": 1, "seq": 3, "prev_sha256": hashlib.sha256(b"raw2").hexdigest(), "files": [{"path": "fly_trader/x.py", "sha256": "good"}]}
    fu.apply(c, st, r3, tmp_path, b"raw3", "hf3", now=1.0)
    assert st["pending"]["seq"] == 3


def test_failed_deploy_is_rejected_not_retried(tmp_path, monkeypatch):
    c, st, prev = installed(tmp_path)
    monkeypatch.setattr(fu, "deploy_code", lambda *a, **k: "failed")
    r2 = {"schema": 1, "seq": 2, "prev_sha256": prev, "files": [{"path": "fly_trader/x.py", "sha256": "b"}]}
    with pytest.raises(SystemExit, match="healthy"):
        fu.apply(c, st, r2, tmp_path, b"raw2", "hf2", approved=True, now=0.0)


def test_run_holds_then_applies_when_due(tmp_path, monkeypatch):
    c, st, prev = installed(tmp_path, repo="x/y")
    fu.save_state(c, st)
    monkeypatch.setattr(fu, "conf", lambda: c)
    monkeypatch.setattr(fu, "http", lambda url, *a, **k: json.dumps({"sha": "hf2"}).encode())
    r2 = {"schema": 1, "seq": 2, "prev_sha256": prev, "files": [{"path": "fly_trader/x.py", "sha256": "b"}]}
    fetched = []
    monkeypatch.setattr(fu, "fetch", lambda c_, sha: fetched.append(sha) or (r2, tmp_path, b"raw2", "bryce"))
    deployed = []
    monkeypatch.setattr(fu, "deploy_code", lambda *a, **k: deployed.append(1) or "ok")
    fu.run(now=0.0)
    fu.run(now=3600.0)                      # held: not even fetched again
    assert fetched == ["hf2"] and deployed == []
    fu.run(now=86400.0)
    assert deployed == [1] and fu.load_state(c)["seq"] == 2
