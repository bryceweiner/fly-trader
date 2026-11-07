"""Known-good contract templates of plain v4 coins (Bryce, 2026-11-05: "build a bytecode template for proper coins based on
known good launches and then integrate that as a filter which rejects any coin that deviates. We might need multiple
templates but if at least 5 coins w/ diff creators share the same bytecode and don't rug those are 'known good'").

A coin's template is its runtime code reduced to a skeleton (a minimal proxy's implementation in its place): the compiler's
CBOR metadata trailer stripped, every PUSH5..PUSH32 operand masked (addresses, immutables, constructor-set constants),
PUSH1..PUSH4 kept (jump targets, selectors). Coins with one skeleton were compiled from one source with other constants.

Built 2026-11-06 (data/scratch/v4_templates_20261105) from the code of 13,753 plain v4 coins (every coin listed, every coin
the swap scan saw trade at a tradeable age). A template is known good when:
- at least 5 of its coins from 5 different creators did not rug: 24 or more 15-minute buckets clearing the live floors at
  age 6 h or more (the swap scan to 10-28, the coins' own minutes after);
- none of its coins is a known scam (ANT 0x36c2861a, 0x493f2e7b) or was refused by the contract check for its code
  (upgradeable proxy, trading switch, built-in tax);
- its code does not delegate (an EIP-1967 proxy's bytecode says nothing of the code it runs).
Nine templates pass: 8,733 of the coins (63.5 %). Every plain v4 trade the books ever made (both scams and one other coin,
23 live and 6 paper trades) was outside them. A template that reaches the rule later is added here by a rebuild.

``check`` (rh/v4_universe.run_checks) refuses an accepted coin whose template is not known good, before the creator
probe; ``retemplate`` re-reads the code of coins accepted before this check and refuses those that deviate.
"""
from __future__ import annotations

import hashlib

# template (sha256 of the skeleton) -> (coins, creators, creators of coins that did not rug), as built
KNOWN_GOOD: dict[str, tuple[int, int, int]] = {
    "f6bdb52a97894128aef0d23ef09150a04f0659a4531cd52cb7e55a945c1b624e": (6566, 3170, 10),
    "49af139e742f766f92f5e0d502fd8b6b97eebf39e2a6662b703c3499faa9f400": (1679, 1287, 202),
    "0e7adef37012d6fbc674846e564af41274f892e5df77ac672abc6da6976e1dae": (380, 232, 16),
    "16bb4db25b3d67731d9a8cbcf551f7efa3a1ad8be030781ccb624429035eb0b3": (36, 36, 12),
    "00962c1adc4a1286685c2950b1b18f87728acc237ed62ef3f5bc66ddf80e9f5f": (23, 10, 7),
    "dd104d00d426716e90090bd228507d2bc3b463d8cf9afb6b5cdd46f28b56675b": (18, 9, 6),
    "13c9e24213c0be99d8c5a21bd07e3e5234bc815429a5817003185400fd3ac9f7": (12, 11, 5),
    "db5d484c62c275f6671246bd6e6023ef86ab2758060adaee8c47942155e1a554": (12, 12, 5),
    "7667934fcd575b13a87627f540231b8ab7b03c174b0a2d959ec47ba9ef73e38d": (7, 7, 5),
}
DEVIATES = "its contract matches no known-good template"


def strip_meta(b: bytes) -> bytes:
    """``b`` without the compiler's CBOR metadata trailer (its length in the last two bytes), if it has one."""
    if len(b) > 2:
        n = int.from_bytes(b[-2:], "big")
        if 0 < n < len(b) - 2 and b[-n - 2] in (0xa1, 0xa2, 0xa3):
            return b[:-n - 2]
    return b


def skeleton(b: bytes) -> bytes:
    """The opcodes of ``b`` with PUSH1..PUSH4 operands kept and PUSH5..PUSH32 operands dropped."""
    out = bytearray(); i = 0
    while i < len(b):
        op = b[i]; out.append(op)
        if 0x60 <= op <= 0x7f:
            n = op - 0x5f
            if n <= 4:
                out += b[i + 1:i + 1 + n]
            i += 1 + n
        else:
            i += 1
    return bytes(out)


def template_id(code_hex: str | None) -> str | None:
    """The template of runtime code ``code_hex`` (0x-prefixed), None for no code."""
    if not isinstance(code_hex, str) or len(code_hex) <= 2:
        return None
    return hashlib.sha256(skeleton(strip_meta(bytes.fromhex(code_hex[2:] if code_hex.startswith("0x") else code_hex)))).hexdigest()


def known(template: str | None) -> bool:
    return template in KNOWN_GOOD
