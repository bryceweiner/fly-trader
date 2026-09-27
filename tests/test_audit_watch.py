"""The host's key-read watcher: which audit events are the signer's own and which are alerted."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[1] / "deploy"))
import fly_audit_watch as aw  # noqa: E402

RAW = """type=SYSCALL msg=audit(1790000000.100:5): arch=c000003e syscall=257 success=yes exit=3 a0=ffffff9c items=1 ppid=9 pid=10 auid=4294967295 uid=175538 gid=175536 euid=175538 comm="python" exe="/usr/local/bin/python3.13" key="fly_keys"
type=PATH msg=audit(1790000000.100:5): item=0 name="/run/secrets/bot_key"
type=SYSCALL msg=audit(1790000001.100:6): arch=c000003e syscall=257 success=yes exit=3 auid=1000 uid=0 gid=0 comm="cat" exe="/usr/bin/cat" key="fly_keys"
type=SYSCALL msg=audit(1790000002.100:7): arch=c000003e syscall=257 success=yes exit=3 auid=4294967295 uid=0 gid=0 comm="aide" exe="/usr/bin/aide" key="fly_keys"
type=SYSCALL msg=audit(1790000003.100:8): arch=c000003e syscall=257 success=yes exit=3 auid=4294967295 uid=175537 gid=175536 comm="python" exe="/usr/local/bin/python3.13" key="fly_keys"
"""


def test_only_the_signer_and_aide_are_quiet():
    bad = aw.suspicious(aw.events(RAW), signer_uid=165536 + 10002)
    assert [(e["exe"], e["uid"]) for e in bad] == [("/usr/bin/cat", "0"), ("/usr/local/bin/python3.13", "175537")]   # a shell, and the brain's uid
