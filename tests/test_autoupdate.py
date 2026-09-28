"""The self-updater: a new head that is neither the installed revision nor a failed one is an update; revisions and
failures are remembered on the data volume; nothing else about the install is a backup's business."""
from pathlib import Path

from fly_trader.ops import autoupdate as au


def test_wants_update_only_for_a_new_head_that_has_not_failed():
    assert au.wants_update("b" * 40, "a" * 40, set())
    assert not au.wants_update("a" * 40, "a" * 40, set())                  # already installed
    assert not au.wants_update("b" * 40, "a" * 40, {"b" * 40})            # tried and rolled back
    assert not au.wants_update(None, "a" * 40, set())                      # Hugging Face unreachable: nothing to do
    assert au.wants_update("b" * 40, None, set())                          # nothing recorded yet


def test_revisions_and_failures_live_on_the_data_volume(tmp_path, monkeypatch):
    monkeypatch.setattr(au, "APP", tmp_path); monkeypatch.setattr(au, "REV_FILE", tmp_path / "data" / "hf_revision"); monkeypatch.setattr(au, "BAD_FILE", tmp_path / "data" / "hf_bad_revisions")
    assert au.installed() is None and au.bad() == set()
    au.record("c" * 40); assert au.installed() == "c" * 40
    au.mark_bad("d" * 40); au.mark_bad("e" * 40); assert au.bad() == {"d" * 40, "e" * 40}


def test_backup_and_restore_leave_the_installs_state_alone(tmp_path, monkeypatch):
    app = tmp_path / "app"; app.mkdir(); prev = tmp_path / "app.prev"
    monkeypatch.setattr(au, "APP", app); monkeypatch.setattr(au, "PREV", prev)
    (app / "fly_trader").mkdir(); (app / "fly_trader" / "x.py").write_text("v1"); (app / "seed").mkdir(); (app / "seed" / "seed.sql").write_text("v1")
    for keep in au.KEEP_OUT:
        (app / keep).mkdir() if not keep.startswith(".env") else (app / keep).write_text("SECRET=1")
    (app / "data" / "state.pt").write_text("learned")
    au.backup()
    assert (prev / "fly_trader" / "x.py").read_text() == "v1" and not (prev / "data").exists() and not (prev / ".env").exists()
    (app / "fly_trader" / "x.py").write_text("v2 broken"); (app / "fly_trader" / "new.py").write_text("v2"); (app / "data" / "state.pt").write_text("learned more")
    au.restore()
    assert (app / "fly_trader" / "x.py").read_text() == "v1" and not (app / "fly_trader" / "new.py").exists()   # code back, additions gone
    assert (app / "data" / "state.pt").read_text() == "learned more" and (app / ".env").read_text() == "SECRET=1"   # state untouched


class FakeFU:
    def __init__(self, seq):
        self.seq = seq

    def fetch(self, c, sha):
        assert c["principals"] == ["bryce", "bryce-recovery"] and c["namespace"] == "fly-trader-release"
        return {"seq": self.seq}, Path(c["root"]), b"{}", "bryce"


def test_updates_need_release_keys_and_a_newer_signed_release(tmp_path, monkeypatch):
    import pytest
    monkeypatch.setattr(au, "APP", tmp_path); monkeypatch.setattr(au, "RELEASE_FILE", tmp_path / "data" / "hf_release.json")
    monkeypatch.setattr(au, "STAGE", tmp_path / "data" / "updates")
    (tmp_path / "release").mkdir(); (tmp_path / "release" / "allowed_signers").write_text("# no keys yet\n")
    with pytest.raises(au.Refused, match="no release keys"):
        au.fetch_verified("a" * 40)
    (tmp_path / "release" / "allowed_signers").write_text('bryce namespaces="fly-trader-release" ssh-ed25519 AAAA\n')
    monkeypatch.setattr(au, "_FU", FakeFU(3))
    assert au.fetch_verified("a" * 40)[0]["seq"] == 3
    (tmp_path / "data").mkdir(exist_ok=True); au.RELEASE_FILE.write_text('{"seq": 3}')
    with pytest.raises(au.Refused, match="not newer"):
        au.fetch_verified("b" * 40)


def test_a_refused_release_is_never_retried(tmp_path, monkeypatch):
    monkeypatch.setattr(au, "BAD_FILE", tmp_path / "bad")
    monkeypatch.setattr(au, "fetch_verified", lambda sha: (_ for _ in ()).throw(au.Refused("signature does NOT verify")))
    assert au.update("c" * 40, None) is None and "c" * 40 in au.bad()


def test_the_real_verifier_refuses_an_unsigned_release(tmp_path, monkeypatch):
    """The installed deploy/fly_update.py checks the signature: a manifest signed by another key is refused."""
    import subprocess
    import pytest
    from pathlib import Path as P
    app = tmp_path / "app"; (app / "deploy").mkdir(parents=True); (app / "release").mkdir()
    (app / "deploy" / "fly_update.py").write_text((P(__file__).parents[1] / "deploy" / "fly_update.py").read_text())
    for k in ("good", "evil"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / k)], check=True)
    (app / "release" / "allowed_signers").write_text('bryce namespaces="fly-trader-release" ' + (tmp_path / "good.pub").read_text())
    manifest = tmp_path / "release.json"; manifest.write_text('{"schema": 1, "seq": 9, "files": []}')
    subprocess.run(["ssh-keygen", "-Y", "sign", "-f", str(tmp_path / "evil"), "-n", "fly-trader-release", str(manifest)], check=True, capture_output=True)
    monkeypatch.setattr(au, "APP", app); monkeypatch.setattr(au, "STAGE", tmp_path / "stage"); monkeypatch.setattr(au, "_FU", None)
    fu = au._verifier()
    blobs = {"release.json": manifest.read_bytes(), "release.json.sig": (tmp_path / "release.json.sig").read_bytes()}
    monkeypatch.setattr(fu, "http", lambda url, dest=None, timeout=0: blobs[url.rsplit("/", 1)[1]])
    monkeypatch.setattr(au, "_FU", fu)
    with pytest.raises(au.Refused, match="does NOT verify"):
        au.fetch_verified("d" * 40)
