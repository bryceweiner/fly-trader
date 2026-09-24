"""Cheap syntax checks for claim inputs. Nothing here is cryptographic: the fly verifies signatures."""
from __future__ import annotations

import re

_EVM = re.compile(r"0x[0-9a-fA-F]{40}")
_NONCE = re.compile(r"[0-9a-f]{32}")
_EVM_SIG = re.compile(r"0x(?:[0-9a-fA-F]{2}){65,1024}")
B58_ALPHABET = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
_B58_INDEX = {c: i for i, c in enumerate(B58_ALPHABET)}


def b58decode(s: object, max_chars: int) -> "bytes | None":
    """Bitcoin-alphabet base58 to bytes, or None. Leading '1's are zero bytes."""
    if not isinstance(s, str) or not s or len(s) > max_chars:
        return None
    n = 0
    for c in s:
        i = _B58_INDEX.get(c)
        if i is None:
            return None
        n = n * 58 + i
    zeros = len(s) - len(s.lstrip("1"))
    body = n.to_bytes((n.bit_length() + 7) // 8, "big") if n else b""
    return b"\x00" * zeros + body


def evm_address(s: object) -> "str | None":
    """Lowercase address, or None. Any case is accepted: EIP-55 needs keccak, which stdlib lacks."""
    if isinstance(s, str) and _EVM.fullmatch(s):
        return s.lower()
    return None


def sol_address(s: object) -> bool:
    b = b58decode(s, 44)
    return b is not None and len(b) == 32


def sol_signature(s: object) -> bool:
    b = b58decode(s, 88)
    return b is not None and len(b) == 64


def evm_signature(s: object) -> bool:
    return isinstance(s, str) and _EVM_SIG.fullmatch(s) is not None


def nonce(s: object) -> bool:
    return isinstance(s, str) and _NONCE.fullmatch(s) is not None
