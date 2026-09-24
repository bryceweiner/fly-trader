"""The self-updater: a new head that is neither the installed revision nor a failed one is an update; revisions and
failures are remembered on the data volume; nothing else about the install is a backup's business."""
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
