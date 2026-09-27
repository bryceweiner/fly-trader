"""Squads v4 multisig: the treasury's spending limits, built by hand from the program's Anchor IDL (v2.1.0,
Squads-Protocol/v4 ``sdk/multisig/idl/squads_multisig_program.json``); stdlib + solders only, so the signer image needs
nothing else.

The treasury is vault 0 of a multisig whose only member is Bryce's wallet (Solflare now, the Ledger later). Two
spending limits let the server's hot keys move SOL out of it without a vote, and nothing more:

  L1  member = trading key, destinations = [trading wallet], period Day  -> the trading float refills itself
  L2  member = payout key,  destinations = [] (any holder),  period Week -> claims

``spendingLimitUse`` checks on chain that the signer is a member of the limit, that the destination is allowed and that
the amount fits what is left of the period; the owner removes a limit (or the whole server's access) from his phone.
"""
from __future__ import annotations

import hashlib
import struct

from solders.instruction import AccountMeta, Instruction
from solders.pubkey import Pubkey

PROGRAM_ID = Pubkey.from_string("SQDS4ep65T869zMMBKyuUq6aD6EgTu8psMjkvj52pCf")
SYSTEM_PROGRAM = Pubkey.from_string("11111111111111111111111111111111")
SEED_PREFIX, SEED_MULTISIG, SEED_VAULT, SEED_SPENDING_LIMIT = b"multisig", b"multisig", b"vault", b"spending_limit"
PERIODS = ("OneTime", "Day", "Week", "Month")
PERIOD_S = {"OneTime": None, "Day": 86400, "Week": 7 * 86400, "Month": 30 * 86400}


def _disc(kind: str, name: str) -> bytes:
    return hashlib.sha256(f"{kind}:{name}".encode()).digest()[:8]


IX_SPENDING_LIMIT_USE = _disc("global", "spending_limit_use")
ACC_SPENDING_LIMIT = _disc("account", "SpendingLimit")
ACC_MULTISIG = _disc("account", "Multisig")


def multisig_pda(create_key: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([SEED_PREFIX, SEED_MULTISIG, bytes(create_key)], PROGRAM_ID)[0]


def vault_pda(multisig: Pubkey, index: int = 0) -> Pubkey:
    return Pubkey.find_program_address([SEED_PREFIX, bytes(multisig), SEED_VAULT, bytes([index])], PROGRAM_ID)[0]


def spending_limit_pda(multisig: Pubkey, create_key: Pubkey) -> Pubkey:
    return Pubkey.find_program_address([SEED_PREFIX, bytes(multisig), SEED_SPENDING_LIMIT, bytes(create_key)], PROGRAM_ID)[0]


def spending_limit_use_sol(multisig: Pubkey, member: Pubkey, spending_limit: Pubkey, vault: Pubkey, destination: Pubkey,
                           lamports: int, memo: str | None = None) -> Instruction:
    """Move ``lamports`` SOL from ``vault`` to ``destination`` under ``spending_limit`` (``member`` signs)."""
    if lamports <= 0:
        raise ValueError("amount must be positive")
    data = IX_SPENDING_LIMIT_USE + struct.pack("<QB", int(lamports), 9)
    if memo is None:
        data += b"\x00"
    else:
        m = memo.encode(); data += b"\x01" + struct.pack("<I", len(m)) + m
    accts = [AccountMeta(multisig, False, False), AccountMeta(member, True, False), AccountMeta(spending_limit, False, True),
             AccountMeta(vault, False, True), AccountMeta(destination, False, True), AccountMeta(SYSTEM_PROGRAM, False, False)]
    # Anchor's absent optional accounts (mint, vault/destination token accounts, token program) are the program id
    accts += [AccountMeta(PROGRAM_ID, False, False)] * 4
    return Instruction(PROGRAM_ID, data, accts)


class _Reader:
    def __init__(self, b: bytes):
        self.b, self.i = b, 0

    def take(self, n: int) -> bytes:
        if self.i + n > len(self.b):
            raise ValueError("account data too short")
        out = self.b[self.i:self.i + n]; self.i += n
        return out

    def u8(self) -> int:
        return self.take(1)[0]

    def u16(self) -> int:
        return struct.unpack("<H", self.take(2))[0]

    def u32(self) -> int:
        return struct.unpack("<I", self.take(4))[0]

    def u64(self) -> int:
        return struct.unpack("<Q", self.take(8))[0]

    def i64(self) -> int:
        return struct.unpack("<q", self.take(8))[0]

    def key(self) -> Pubkey:
        return Pubkey.from_bytes(self.take(32))

    def keys(self) -> list[Pubkey]:
        return [self.key() for _ in range(self.u32())]


def decode_spending_limit(data: bytes) -> dict:
    if bytes(data[:8]) != ACC_SPENDING_LIMIT:
        raise ValueError("not a Squads SpendingLimit account")
    r = _Reader(bytes(data)); r.take(8)
    d = {"multisig": r.key(), "create_key": r.key(), "vault_index": r.u8(), "mint": r.key(), "amount": r.u64()}
    p = r.u8()
    if p >= len(PERIODS):
        raise ValueError("unknown period")
    d.update({"period": PERIODS[p], "remaining": r.u64(), "last_reset": r.i64(), "bump": r.u8(), "members": r.keys(), "destinations": r.keys()})
    return d


def decode_multisig(data: bytes) -> dict:
    if bytes(data[:8]) != ACC_MULTISIG:
        raise ValueError("not a Squads Multisig account")
    r = _Reader(bytes(data)); r.take(8)
    d = {"create_key": r.key(), "config_authority": r.key(), "threshold": r.u16(), "time_lock": r.u32(), "transaction_index": r.u64(),
         "stale_transaction_index": r.u64()}
    d["rent_collector"] = r.key() if r.u8() else None
    d["bump"] = r.u8()
    d["members"] = [(r.key(), r.u8()) for _ in range(r.u32())]      # (key, permission mask: 1 initiate, 2 vote, 4 execute)
    return d


def remaining_now(limit: dict, now: int) -> int:
    """What ``spendingLimitUse`` would allow right now (the program resets whole periods lazily, on use)."""
    secs = PERIOD_S[limit["period"]]
    if secs is not None and now - limit["last_reset"] > secs:
        return int(limit["amount"])
    return int(limit["remaining"])


def encode_spending_limit(d: dict) -> bytes:
    """Account bytes for tests and the local demo (the inverse of ``decode_spending_limit``)."""
    b = ACC_SPENDING_LIMIT + bytes(d["multisig"]) + bytes(d["create_key"]) + bytes([d.get("vault_index", 0)]) + bytes(d.get("mint", SYSTEM_PROGRAM))
    b += struct.pack("<QBQqB", d["amount"], PERIODS.index(d["period"]), d["remaining"], d["last_reset"], d.get("bump", 255))
    for k in ("members", "destinations"):
        b += struct.pack("<I", len(d[k])) + b"".join(bytes(x) for x in d[k])
    return b
