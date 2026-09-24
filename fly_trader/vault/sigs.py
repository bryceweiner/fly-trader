"""Claim signature checks: EIP-191 ``personal_sign`` (ecrecover, EIP-1271 for contract wallets) and ed25519.

EOA path: the 65-byte r||s||v signature must be canonical (low-s, v in 27/28 or 0/1) and recover to the claimed
address. Contract-wallet path (a Safe on Robinhood Chain): only when the address has code, ``isValidSignature`` is
called over the RPC with a gas cap and must return the EIP-1271 magic value. Solana: ed25519 over the UTF-8 text.
"""
from __future__ import annotations

import base58
import coincurve
from solders.pubkey import Pubkey
from solders.signature import Signature

from . import evm

SECP256K1_N = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141
EIP1271_MAGIC = "0x1626ba7e"
EIP1271_SELECTOR = evm.selector("isValidSignature(bytes32,bytes)")
EIP1271_GAS = 200_000


class SignatureInvalid(ValueError):
    """The signature does not verify; the message names why without echoing inputs."""


def eip191_hash(text: str) -> bytes:
    msg = text.encode("utf-8")
    return evm.keccak256(b"\x19Ethereum Signed Message:\n" + str(len(msg)).encode() + msg)


def _split(sig_hex: str) -> tuple[bytes, int]:
    s = sig_hex[2:] if sig_hex.startswith("0x") else sig_hex
    try:
        raw = bytes.fromhex(s)
    except ValueError:
        raise SignatureInvalid("evm signature is not hex") from None
    if len(raw) != 65:
        raise SignatureInvalid("evm signature must be 65 bytes")
    v = raw[64]
    if v in (27, 28):
        v -= 27
    if v not in (0, 1):
        raise SignatureInvalid("evm signature v must be 27/28 or 0/1")
    s_int = int.from_bytes(raw[32:64], "big")
    r_int = int.from_bytes(raw[:32], "big")
    if not (0 < r_int < SECP256K1_N) or not (0 < s_int <= SECP256K1_N // 2):
        raise SignatureInvalid("evm signature is not canonical (high s)")
    return raw[:64], v


def recover(text: str, sig_hex: str) -> str:
    """EIP-55 address that signed ``text`` with personal_sign."""
    rs, v = _split(sig_hex)
    try:
        pub = coincurve.PublicKey.from_signature_and_message(rs + bytes([v]), eip191_hash(text), hasher=None)
    except Exception:
        raise SignatureInvalid("evm signature does not recover") from None
    return evm.to_checksum("0x" + evm.keccak256(pub.format(compressed=False)[1:])[-20:].hex())


def verify_evm(text: str, sig_hex: str, address: str, rpc: "evm.EvmRpc | None" = None) -> str:
    """'eoa' or 'eip1271' when valid; raises SignatureInvalid otherwise."""
    want = evm.to_checksum(address)
    try:
        if recover(text, sig_hex) == want:
            return "eoa"
    except SignatureInvalid:
        if rpc is None:
            raise
    if rpc is None:
        raise SignatureInvalid("evm signature is from another address")
    try:
        has_code = rpc.code(want) not in ("0x", "0x0", "")
    except Exception:
        raise SignatureInvalid("could not check the address for contract code") from None
    if not has_code:
        raise SignatureInvalid("evm signature is from another address")
    try:
        sig = bytes.fromhex(sig_hex[2:] if sig_hex.startswith("0x") else sig_hex)
    except ValueError:
        raise SignatureInvalid("evm signature is not hex") from None
    if not 65 <= len(sig) <= 1024:
        raise SignatureInvalid("evm signature length out of range")
    data = (EIP1271_SELECTOR + evm.encode_bytes32(eip191_hash(text)) + evm.encode_uint(64) + evm.encode_dynamic_bytes(sig))
    try:
        out = rpc.eth_call(want, data, gas=EIP1271_GAS)
    except Exception:
        raise SignatureInvalid("contract wallet rejected the signature") from None
    if not out or out[:10].lower() != EIP1271_MAGIC:
        raise SignatureInvalid("contract wallet rejected the signature")
    return "eip1271"


def verify_sol(text: str, sig_b58: str, pubkey_b58: str) -> None:
    try:
        raw = base58.b58decode(sig_b58)
        pk = Pubkey.from_string(pubkey_b58)
    except Exception:
        raise SignatureInvalid("solana signature or address is malformed") from None
    if len(raw) != 64:
        raise SignatureInvalid("solana signature must be 64 bytes")
    if not pk.is_on_curve():
        raise SignatureInvalid("solana address is not a wallet (off curve)")
    if not Signature(raw).verify(pk, text.encode("utf-8")):
        raise SignatureInvalid("solana signature does not verify")


# ---- test/rehearsal helpers (never used with a real key) ----
def sign_evm(text: str, private_key: bytes) -> str:
    rs_v = coincurve.PrivateKey(private_key).sign_recoverable(eip191_hash(text), hasher=None)
    return "0x" + (rs_v[:64] + bytes([rs_v[64] + 27])).hex()


def evm_address(private_key: bytes) -> str:
    pub = coincurve.PrivateKey(private_key).public_key.format(compressed=False)
    return evm.to_checksum("0x" + evm.keccak256(pub[1:])[-20:].hex())
