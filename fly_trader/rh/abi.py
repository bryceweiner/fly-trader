"""Solidity ABI encoding and decoding: every type the RH trading path meets (uintN/intN, address, bool, bytesN, bytes,
string, T[], T[k], nested tuples), for calldata we build (UniversalRouter, Permit2, ERC-20, V4Quoter) and calldata we
must check before signing (a KyberSwap build). No web3 dependency, like fly_trader/vault/evm.py whose keccak it reuses.

Types are canonical strings: ``"uint256"``, ``"address[]"``, ``"(address,uint24,int24,address)"``,
``"(address,(bytes32,uint256)[])"``. Addresses decode to lowercase 0x-hex; bytes to ``bytes``; ints to ``int``.
"""
from __future__ import annotations

from ..vault.evm import keccak256

WORD = 32


# ---------------------------------------------------------------- type parsing
def _split_top(s: str) -> list[str]:
    """Split a comma list at depth 0: ``"a,(b,c)[],d"`` → ``["a", "(b,c)[]", "d"]``."""
    out, depth, cur = [], 0, []
    for ch in s:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    if cur or out:
        out.append("".join(cur))
    return [x.strip() for x in out if x.strip()]


def _array_suffix(t: str) -> tuple[str, int | None] | None:
    """``"T[]"`` → (T, None), ``"T[3]"`` → (T, 3); not an array → None."""
    if not t.endswith("]"):
        return None
    i = t.rindex("[")
    inner, n = t[:i], t[i + 1:-1]
    return inner, (int(n) if n else None)


def _tuple_parts(t: str) -> list[str] | None:
    if t.startswith("(") and t.endswith(")"):
        return _split_top(t[1:-1])
    return None


def is_dynamic(t: str) -> bool:
    arr = _array_suffix(t)
    if arr is not None:
        inner, n = arr
        return n is None or is_dynamic(inner)
    parts = _tuple_parts(t)
    if parts is not None:
        return any(is_dynamic(p) for p in parts)
    return t in ("bytes", "string")


def head_size(t: str) -> int:
    if is_dynamic(t):
        return WORD
    arr = _array_suffix(t)
    if arr is not None:
        return arr[1] * head_size(arr[0])
    parts = _tuple_parts(t)
    if parts is not None:
        return sum(head_size(p) for p in parts)
    return WORD


# ---------------------------------------------------------------- encoding
def _enc_int(t: str, v: int) -> bytes:
    v = int(v)
    if t.startswith("uint"):
        bits = int(t[4:] or 256)
        if not 0 <= v < 2 ** bits:
            raise ValueError(f"{v} out of range for {t}")
        return v.to_bytes(WORD, "big")
    bits = int(t[3:] or 256)
    if not -(2 ** (bits - 1)) <= v < 2 ** (bits - 1):
        raise ValueError(f"{v} out of range for {t}")
    return (v % 2 ** 256).to_bytes(WORD, "big")


def _enc_address(v: str) -> bytes:
    h = v[2:] if v.startswith("0x") else v
    if len(h) != 40:
        raise ValueError(f"not an address: {v!r}")
    return bytes(12) + bytes.fromhex(h)


def _pad_right(b: bytes) -> bytes:
    return b + bytes((-len(b)) % WORD)


def encode_single(t: str, v) -> bytes:
    arr = _array_suffix(t)
    if arr is not None:
        inner, n = arr
        vals = list(v)
        if n is not None and len(vals) != n:
            raise ValueError(f"{t} needs {n} values, got {len(vals)}")
        body = encode([inner] * len(vals), vals)
        return (len(vals).to_bytes(WORD, "big") if n is None else b"") + body
    parts = _tuple_parts(t)
    if parts is not None:
        return encode(parts, list(v))
    if t == "address":
        return _enc_address(v)
    if t == "bool":
        return (1 if v else 0).to_bytes(WORD, "big")
    if t.startswith("uint") or t.startswith("int"):
        return _enc_int(t, v)
    if t == "bytes":
        b = bytes(v)
        return len(b).to_bytes(WORD, "big") + _pad_right(b)
    if t == "string":
        b = v.encode()
        return len(b).to_bytes(WORD, "big") + _pad_right(b)
    if t.startswith("bytes"):
        n = int(t[5:]); b = bytes(v)
        if len(b) != n:
            raise ValueError(f"{t} needs {n} bytes, got {len(b)}")
        return _pad_right(b)
    raise ValueError(f"unsupported ABI type {t!r}")


def encode(types: list[str], values: list) -> bytes:
    """``abi.encode(values...)`` for ``types`` (heads then tails)."""
    if len(types) != len(values):
        raise ValueError(f"{len(types)} types, {len(values)} values")
    heads, tails = [], []
    offset = sum(head_size(t) for t in types)
    for t, v in zip(types, values):
        if is_dynamic(t):
            enc = encode_single(t, v)
            heads.append(offset.to_bytes(WORD, "big")); tails.append(enc); offset += len(enc)
        else:
            heads.append(encode_single(t, v))
    return b"".join(heads) + b"".join(tails)


def selector(signature: str) -> bytes:
    return keccak256(signature.encode())[:4]


def signature_types(signature: str) -> list[str]:
    """``"swap(address,(uint256,bytes))"`` → ``["address", "(uint256,bytes)"]``."""
    return _split_top(signature[signature.index("(") + 1:signature.rindex(")")])


def encode_call(signature: str, *values) -> bytes:
    return selector(signature) + encode(signature_types(signature), list(values))


def encode_packed_path(parts: list[tuple[str, object]]) -> bytes:
    """``abi.encodePacked`` for (type, value) pairs of static types (a Uniswap v3 path: address, uint24, address, …)."""
    out = b""
    for t, v in parts:
        if t == "address":
            out += bytes.fromhex(v[2:])
        elif t.startswith("uint"):
            out += int(v).to_bytes(int(t[4:]) // 8, "big")
        else:
            raise ValueError(f"unsupported packed type {t!r}")
    return out


# ---------------------------------------------------------------- decoding
def _word(data: bytes, pos: int) -> bytes:
    if pos + WORD > len(data):
        raise ValueError("ABI data too short")
    return data[pos:pos + WORD]


def decode_single(t: str, data: bytes, pos: int):
    """Decode the value of type ``t`` whose head is at ``pos`` within ``data`` (the enclosing tuple's encoding)."""
    if is_dynamic(t):
        off = int.from_bytes(_word(data, pos), "big")
        return _decode_at(t, data, off)
    return _decode_at(t, data, pos)


def _decode_at(t: str, data: bytes, pos: int):
    arr = _array_suffix(t)
    if arr is not None:
        inner, n = arr
        if n is None:
            n = int.from_bytes(_word(data, pos), "big"); pos += WORD
        if n > 1_000_000:
            raise ValueError("ABI array length absurd")
        return decode([inner] * n, data[pos:])
    parts = _tuple_parts(t)
    if parts is not None:
        return tuple(decode(parts, data[pos:]))
    w = _word(data, pos)
    if t == "address":
        if any(w[:12]):
            raise ValueError("dirty address word")
        return "0x" + w[12:].hex()
    if t == "bool":
        v = int.from_bytes(w, "big")
        if v > 1:
            raise ValueError("dirty bool word")
        return bool(v)
    if t.startswith("uint"):
        v = int.from_bytes(w, "big"); bits = int(t[4:] or 256)
        if v >= 2 ** bits:
            raise ValueError(f"dirty {t} word")
        return v
    if t.startswith("int"):
        v = int.from_bytes(w, "big", signed=True); bits = int(t[3:] or 256)
        if not -(2 ** (bits - 1)) <= v < 2 ** (bits - 1):
            raise ValueError(f"dirty {t} word")
        return v
    if t in ("bytes", "string"):
        n = int.from_bytes(w, "big")
        if pos + WORD + n > len(data):
            raise ValueError("ABI bytes run past the data")
        b = data[pos + WORD:pos + WORD + n]
        return b.decode() if t == "string" else b
    if t.startswith("bytes"):
        return w[:int(t[5:])]
    raise ValueError(f"unsupported ABI type {t!r}")


def decode(types: list[str], data: bytes) -> list:
    out, pos = [], 0
    for t in types:
        out.append(decode_single(t, data, pos)); pos += head_size(t)
    return out


def decode_call(signature: str, calldata: bytes) -> list:
    """Decode calldata of ``signature``; refuses another selector."""
    if calldata[:4] != selector(signature):
        raise ValueError(f"calldata is not {signature.split('(')[0]}")
    return decode(signature_types(signature), calldata[4:])


def event_topic(signature: str) -> str:
    return "0x" + keccak256(signature.encode()).hex()
