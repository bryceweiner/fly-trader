"""Key parsing accepts both on-disk forms; failures never leak secret material; wallet creation is
tested only against a temporary .env (the real bot wallet is never created here)."""
import json
import os
import stat

import base58
import pytest
from solders.keypair import Keypair

from fly_trader.chain import keys


def test_base58_and_json_array_forms_agree():
    kp = Keypair()
    b58 = base58.b58encode(bytes(kp)).decode()
    arr = json.dumps(list(bytes(kp)))
    assert keys.bot_pubkey(b58) == keys.bot_pubkey(arr) == str(kp.pubkey())
    assert keys.load_keypair(b58).pubkey() == kp.pubkey()
    assert len(b58) in (87, 88)  # what the logging scrubber redacts


def test_bad_key_errors_never_contain_material():
    marker = "SECRETMARKER"
    with pytest.raises(keys.WalletKeyError) as ei:
        keys.load_keypair(marker + "0OIl-not-base58")
    assert marker not in str(ei.value)

    with pytest.raises(keys.WalletKeyError) as ei:
        keys.load_keypair(json.dumps([7] * 63))
    assert "7, 7" not in str(ei.value) and "[7" not in str(ei.value)
    assert "63" in str(ei.value)  # the length is safe to report

    kp = Keypair()
    b58 = base58.b58encode(bytes(kp)).decode()
    truncated = b58[:-3]
    with pytest.raises(keys.WalletKeyError) as ei:
        keys.load_keypair(truncated)
    assert b58[:16] not in str(ei.value)


def test_missing_key(monkeypatch):
    monkeypatch.delenv("BOT_PRIVATE_KEY", raising=False)
    with pytest.raises(keys.WalletKeyError, match="not set"):
        keys.load_keypair()
    assert keys.key_available() is False


def test_create_wallet_refuses_when_env_file_has_key(tmp_path, monkeypatch):
    monkeypatch.delenv("BOT_PRIVATE_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("FOO=1\nexport BOT_PRIVATE_KEY=placeholder\n")
    with pytest.raises(keys.WalletExists):
        keys.create_wallet(env_path=env)
    assert env.read_text().count("BOT_PRIVATE_KEY") == 1


def test_create_wallet_into_temporary_env(tmp_path, monkeypatch, db_conn):
    monkeypatch.delenv("BOT_PRIVATE_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_bytes(b"FOO=1")  # no trailing newline, like the operator's file
    pubkey = keys.create_wallet(env_path=env)
    text = env.read_text()
    assert text.startswith("FOO=1\nBOT_PRIVATE_KEY=") and text.endswith("\n")
    assert stat.S_IMODE(env.stat().st_mode) == 0o600
    line = next(l for l in text.splitlines() if l.startswith("BOT_PRIVATE_KEY="))
    assert keys.bot_pubkey(line.split("=", 1)[1]) == pubkey
    assert "BOT_PRIVATE_KEY" not in os.environ
    with pytest.raises(keys.WalletExists):
        keys.create_wallet(env_path=env)
    n = db_conn.execute("SELECT count(*) AS n FROM wallet_events WHERE kind = 'created' AND pubkey = %s", (pubkey,)).fetchone()["n"]
    assert n == 1


def test_show_wallet_uses_rpc_and_records_snapshot(monkeypatch, db_conn):
    kp = Keypair()
    monkeypatch.setattr(keys, "load_keypair", lambda raw=None: kp)

    class Rpc:
        def get_balance(self, pk):
            assert pk == str(kp.pubkey())
            return 1_234_567_890

    line = keys.show_wallet(rpc=Rpc())
    assert str(kp.pubkey()) in line and "1.234567890" in line
    row = db_conn.execute("SELECT detail FROM wallet_events WHERE kind = 'balance_snapshot' AND pubkey = %s",
                          (str(kp.pubkey()),)).fetchone()
    assert row["detail"]["lamports"] == 1_234_567_890
