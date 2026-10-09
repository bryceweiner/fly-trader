"""EIP-1559 (type-2) transactions for Robinhood Chain: RLP, the signing hash, secp256k1 signatures and sender recovery.

Signed with ``coincurve`` (libsecp256k1: deterministic RFC 6979 nonces, low-s) over keccak(0x02 ‖ rlp(fields)); the
wire form is 0x02 ‖ rlp(fields + [yParity, r, s]) and the hash is keccak of it. Checked against viem's serializer
(tests/vectors/evm_tx_v1.json) — the repo carries no web3/eth_account next to a hot key.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import coincurve

from ..vault.evm import keccak256

SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141


# ---------------------------------------------------------------- RLP
def _int_bytes(n: int) -> bytes:
    if n < 0:
        raise ValueError("RLP cannot encode a negative integer")
    return b"" if n == 0 else n.to_bytes((n.bit_length() + 7) // 8, "big")


def rlp_encode(item) -> bytes:
    """RLP of bytes / int / nested lists (ints as minimal big-endian, 0 as the empty string)."""
    if isinstance(item, int):
        item = _int_bytes(item)
    if isinstance(item, (bytes, bytearray)):
        b = bytes(item)
        if len(b) == 1 and b[0] < 0x80:
            return b
        return _len_prefix(len(b), 0x80) + b
    if isinstance(item, (list, tuple)):
        body = b"".join(rlp_encode(x) for x in item)
        return _len_prefix(len(body), 0xC0) + body
    raise TypeError(f"cannot RLP-encode {type(item).__name__}")


def _len_prefix(n: int, offset: int) -> bytes:
    if n < 56:
        return bytes([offset + n])
    ln = _int_bytes(n)
    return bytes([offset + 55 + len(ln)]) + ln


def rlp_decode(data: bytes):
    item, rest = _rlp_item(data)
    if rest:
        raise ValueError("trailing RLP bytes")
    return item


def _rlp_item(d: bytes):
    if not d:
        raise ValueError("empty RLP")
    p = d[0]
    if p < 0x80:
        return d[:1], d[1:]
    if p < 0xB8:
        n = p - 0x80; return d[1:1 + n], d[1 + n:]
    if p < 0xC0:
        ll = p - 0xB7; n = int.from_bytes(d[1:1 + ll], "big"); return d[1 + ll:1 + ll + n], d[1 + ll + n:]
    if p < 0xF8:
        n = p - 0xC0; body, rest = d[1:1 + n], d[1 + n:]
    else:
        ll = p - 0xF7; n = int.from_bytes(d[1:1 + ll], "big"); body, rest = d[1 + ll:1 + ll + n], d[1 + ll + n:]
    out = []
    while body:
        x, body = _rlp_item(body); out.append(x)
    return out, rest


# ---------------------------------------------------------------- EIP-1559
def _addr_bytes(a: str | None) -> bytes:
    if a is None or a == "":
        return b""                                  # contract creation (never used here)
    h = a[2:] if a.startswith("0x") else a
    if len(h) != 40:
        raise ValueError(f"not an address: {a!r}")
    return bytes.fromhex(h)


@dataclass(frozen=True)
class Eip1559Tx:
    chain_id: int
    nonce: int
    max_priority_fee_per_gas: int
    max_fee_per_gas: int
    gas: int
    to: str
    value: int = 0
    data: bytes = b""
    access_list: list = field(default_factory=list)

    def fields(self) -> list:
        al = [[_addr_bytes(a), [bytes.fromhex(k[2:]) for k in keys]] for a, keys in self.access_list]
        return [self.chain_id, self.nonce, self.max_priority_fee_per_gas, self.max_fee_per_gas, self.gas, _addr_bytes(self.to),
                self.value, bytes(self.data), al]

    def signing_hash(self) -> bytes:
        return keccak256(b"\x02" + rlp_encode(self.fields()))


@dataclass(frozen=True)
class SignedTx:
    tx: Eip1559Tx
    y_parity: int
    r: int
    s: int
    raw: bytes

    @property
    def hash(self) -> str:
        return "0x" + keccak256(self.raw).hex()

    @property
    def raw_hex(self) -> str:
        return "0x" + self.raw.hex()


def address_of(private_key: bytes) -> str:
    """The lowercase 0x address of a 32-byte secp256k1 private key."""
    pub = coincurve.PrivateKey(private_key).public_key.format(compressed=False)[1:]
    return "0x" + keccak256(pub)[-20:].hex()


def sign(tx: Eip1559Tx, private_key: bytes) -> SignedTx:
    if tx.max_priority_fee_per_gas > tx.max_fee_per_gas:
        raise ValueError("priority fee above the max fee")
    sig = coincurve.PrivateKey(private_key).sign_recoverable(tx.signing_hash(), hasher=None)
    r, s, v = int.from_bytes(sig[:32], "big"), int.from_bytes(sig[32:64], "big"), sig[64]
    if s > SECP256K1_N // 2:                         # libsecp256k1 already normalises; refuse rather than send a malleable one
        raise ValueError("high-s signature")
    raw = b"\x02" + rlp_encode(tx.fields() + [v, r, s])
    return SignedTx(tx, v, r, s, raw)


def decode_raw(raw: bytes) -> SignedTx:
    """Parse a signed type-2 transaction (the inverse of ``sign``)."""
    if not raw or raw[0] != 0x02:
        raise ValueError("not a type-2 transaction")
    f = rlp_decode(raw[1:])
    if len(f) != 12:
        raise ValueError("a type-2 transaction has 12 fields")
    i = lambda b: int.from_bytes(b, "big")
    al = [("0x" + a.hex(), ["0x" + k.hex() for k in keys]) for a, keys in f[8]]
    tx = Eip1559Tx(chain_id=i(f[0]), nonce=i(f[1]), max_priority_fee_per_gas=i(f[2]), max_fee_per_gas=i(f[3]), gas=i(f[4]),
                   to="0x" + f[5].hex(), value=i(f[6]), data=bytes(f[7]), access_list=al)
    return SignedTx(tx, i(f[9]), i(f[10]), i(f[11]), raw)


def recover_sender(stx: SignedTx) -> str:
    sig = stx.r.to_bytes(32, "big") + stx.s.to_bytes(32, "big") + bytes([stx.y_parity])
    pub = coincurve.PublicKey.from_signature_and_message(sig, stx.tx.signing_hash(), hasher=None).format(compressed=False)[1:]
    return "0x" + keccak256(pub)[-20:].hex()
