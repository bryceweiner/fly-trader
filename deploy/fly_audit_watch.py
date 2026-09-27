#!/usr/bin/env python3
"""Tell the operator when anything but the signer reads a wallet key (runs on the HOST, root, every minute).

auditd logs every access to /srv/fly/secrets (key ``fly_keys``, bootstrap-host.sh). The signer container's Python is
the only reader there should ever be (plus AIDE's daily integrity scan); a read by anything else -- a shell, cat, a
process in the brain container -- is the earliest sign of a compromise, so it goes to Telegram at once, with the
program, user and time. ``ausearch --checkpoint`` makes each event reported once.

    fly_audit_watch.py run            (the systemd timer)
    fly_audit_watch.py test           (send a test message)
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

CONF = Path("/etc/fly/update.json")                  # the updater's Telegram settings
CHECKPOINT = Path("/var/lib/fly/audit.checkpoint")
SIGNER_EXE = {"/usr/local/bin/python3.13", "/usr/local/bin/python3"}   # the signer image's interpreter
ALLOWED_EXE = {"/usr/bin/aide", "/usr/sbin/aide"}                        # AIDE hashes every file daily
SIGNER_UID_OFFSET = 10002                            # the signer container's user; on the host: dockremap base + this

FIELD = re.compile(r'(\w+)=("[^"]*"|\S+)')


def remap_base() -> int:
    for line in Path("/etc/subuid").read_text().splitlines():
        name, start, _ = line.split(":")
        if name == "dockremap":
            return int(start)
    return -1


def events(raw: str) -> list[dict]:
    """SYSCALL records of ``ausearch --raw`` output as dicts (pure)."""
    out = []
    for line in raw.splitlines():
        if not line.startswith("type=SYSCALL"):
            continue
        out.append({k: v.strip('"') for k, v in FIELD.findall(line)})
    return out


def suspicious(evs: list[dict], signer_uid: int) -> list[dict]:
    """Reads that are not the signer's interpreter as the signer's user, nor AIDE (pure)."""
    bad = []
    for e in evs:
        exe, uid = e.get("exe", ""), int(e.get("uid", "-1")) if e.get("uid", "").lstrip("-").isdigit() else -1
        if exe in SIGNER_EXE and uid == signer_uid:
            continue
        if exe in ALLOWED_EXE and uid == 0:
            continue
        bad.append(e)
    return bad


def alert(text: str) -> None:
    c = json.loads(CONF.read_text()) if CONF.exists() else {}
    tok, chat = c.get("telegram_bot_token"), c.get("telegram_chat_id")
    print("ALERT", text, flush=True)
    if tok and chat:
        body = json.dumps({"chat_id": chat, "text": "[fly host] " + text}).encode()
        urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=10)


def run() -> int:
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(["ausearch", "-k", "fly_keys", "--raw", "--checkpoint", str(CHECKPOINT), "--start", "checkpoint"],
                       capture_output=True, text=True)
    if r.returncode not in (0, 1):                   # 1: no matches; 10/11/12: checkpoint problems -> start fresh next run
        CHECKPOINT.unlink(missing_ok=True)
        return 0
    bad = suspicious(events(r.stdout), remap_base() + SIGNER_UID_OFFSET)
    if bad:
        lines = [f"{e.get('exe')} (uid {e.get('uid')}, comm {e.get('comm')}, syscall {e.get('syscall')})" for e in bad[:10]]
        alert(f"WALLET KEY READ by something other than the signer ({len(bad)} event(s)):\n" + "\n".join(lines)
              + "\nIf this was not you: revoke L1 and L2 in the owner wallet now, then investigate.")
    return len(bad)


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "run"
    if cmd == "test":
        alert("audit watch test message")
    else:
        run()
