"""Sign only our own signer slot of a Jupiter-built VersionedTransaction (plan §9).

The multi-signer trap (flywheel utils/transaction_signing.py:99-205): ``VersionedTransaction(msg,
[keypair])`` demands keypairs for EVERY required signer and replaces all signatures, which pushed
that code into rewriting account keys and the header — Jupiter then rejects the transaction (and a
rewritten fee payer would make us pay for someone else's message). Here the message, header and fee
payer are never touched: we locate our pubkey among the static account keys, require its index to
be below ``num_required_signatures``, sign the versioned message bytes, and drop the signature into
that slot while keeping any signatures already present (RFQ makers pre-sign their slot).

Verified 2026-09-12: a Swap v2 ``/order`` transaction with a taker is v0 with
``num_required_signatures=1``, the taker as static key 0 (fee payer), 4 address-table lookups and no
pre-filled signatures.
"""
from __future__ import annotations

import base64

from solders.message import to_bytes_versioned
from solders.signature import Signature
from solders.transaction import VersionedTransaction


class NotASigner(ValueError):
    """Our pubkey is absent from the static keys or sits outside the signer range."""


def _as_tx(tx) -> VersionedTransaction:
    if isinstance(tx, VersionedTransaction):
        return tx
    if isinstance(tx, str):
        tx = base64.b64decode(tx)
    return VersionedTransaction.from_bytes(bytes(tx))


def signer_index(tx, pubkey) -> int:
    """Index of ``pubkey`` among the static account keys, verified to be a required signer."""
    tx = _as_tx(tx)
    msg = tx.message
    keys = list(msg.account_keys)
    if pubkey not in keys:
        raise NotASigner(f"{pubkey} is not among the transaction's static account keys")
    idx = keys.index(pubkey)
    n_req = msg.header.num_required_signatures
    if idx >= n_req:
        raise NotASigner(f"{pubkey} is account {idx} but only the first {n_req} account(s) sign")
    return idx


def sign_transaction_b64(tx_b64: str, keypair) -> tuple[str, int]:
    """Return ``(base64 signed transaction, our signer index)``. Message/header/fee payer untouched."""
    tx = _as_tx(tx_b64)
    msg = tx.message
    idx = signer_index(tx, keypair.pubkey())
    n_req = msg.header.num_required_signatures
    sigs = list(tx.signatures)
    if len(sigs) < n_req:
        sigs.extend([Signature.default()] * (n_req - len(sigs)))
    sigs[idx] = keypair.sign_message(to_bytes_versioned(msg))
    signed = VersionedTransaction.populate(msg, sigs)
    return base64.b64encode(bytes(signed)).decode(), idx


def transaction_id(tx) -> str:
    """The transaction signature (first signature) of a signed transaction."""
    return str(_as_tx(tx).signatures[0])


def static_program_ids(tx) -> list[str]:
    """Program ids of the compiled instructions that resolve to STATIC account keys, unique, in order.
    Program ids that live in address lookup tables are not resolved in v1 (see ``n_lookup_programs``)."""
    msg = _as_tx(tx).message
    keys = [str(k) for k in msg.account_keys]
    out: list[str] = []
    for ci in msg.instructions:
        i = ci.program_id_index
        if i < len(keys) and keys[i] not in out:
            out.append(keys[i])
    return out


def n_lookup_programs(tx) -> int:
    """How many compiled instructions reference a program id from an address lookup table."""
    msg = _as_tx(tx).message
    n_static = len(msg.account_keys)
    return sum(1 for ci in msg.instructions if ci.program_id_index >= n_static)
