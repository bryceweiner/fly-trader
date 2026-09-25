"""Encrypted off-box backups: the wallet key (once) and the vault ledger (nightly), encrypted with ``age`` to Bryce's
public key (``VAULT_BACKUP_RECIPIENT``: an age1… or ssh-ed25519 key) and uploaded to a private HF storage bucket with a
token scoped to that bucket. Plaintext never leaves the server; only the holder of the private key can decrypt:

    age -d -i ~/.ssh/id_ed25519 wallet-key.age > key.txt
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

from .. import config
from . import alerts, state

log = logging.getLogger(__name__)
LEDGER_TABLES = ("vault_flows", "vault_settlements", "vault_allocations", "vault_claims", "vault_events", "vault_kv",
                 "positions", "fills", "orders", "wallet_events")


def configured() -> bool:
    return bool(config.VAULT_BACKUP_RECIPIENT and config.VAULT_BACKUP_BUCKET and config.VAULT_BACKUP_HF_TOKEN and shutil.which("age"))


def _encrypt(plain: bytes, out: Path) -> None:
    with tempfile.NamedTemporaryFile("w", delete=False) as f:
        f.write(config.VAULT_BACKUP_RECIPIENT.strip() + "\n"); rfile = f.name
    try:
        subprocess.run(["age", "-R", rfile, "-o", str(out)], input=plain, check=True, capture_output=True)
    finally:
        os.unlink(rfile)


def _upload(files: list[tuple[Path, str]]) -> None:
    from huggingface_hub import batch_bucket_files
    batch_bucket_files(config.VAULT_BACKUP_BUCKET, add=[(str(p), remote) for p, remote in files], token=config.VAULT_BACKUP_HF_TOKEN)


def key_once() -> bool:
    """Back up the wallet key the first time only; later runs are no-ops."""
    if state.get("backup_key_done") or not configured():
        return False
    from ..chain.keys import load_keypair
    from . import payout
    import base58
    kp = load_keypair()
    with tempfile.TemporaryDirectory() as d:
        files = []
        for name, k in (("trading", kp), ("payout", payout.load_keypair())):
            out = Path(d) / f"{name}-key.age"
            _encrypt(base58.b58encode(bytes(k)), out)
            files.append((out, f"keys/{name}-{k.pubkey()}.age"))
        _upload(files)
    state.put("backup_key_done", {"ts": int(time.time()), "pubkey": str(kp.pubkey())})
    alerts.send(f"wallet key backup written (encrypted) for {kp.pubkey()}")
    return True


def ledger_nightly() -> bool:
    last = int((state.get("backup_ledger") or {}).get("ts") or 0)
    if not configured() or time.time() - last < 20 * 3600:
        return False
    url = config.DATABASE_URL
    args = ["pg_dump", "-Fc", *[a for t in LEDGER_TABLES for a in ("-t", t)], url]
    dump = subprocess.run(args, check=True, capture_output=True).stdout
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "ledger.dump.age"
        _encrypt(dump, out)
        day = time.strftime("%Y-%m-%d", time.gmtime())
        _upload([(out, f"ledger/{day}.dump.age")])
    state.put("backup_ledger", {"ts": int(time.time()), "bytes": len(dump)})
    return True
