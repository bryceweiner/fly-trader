"""Signing touches only our signature slot: message, header and fee payer stay byte-identical."""
import base64

import pytest
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from fly_trader.chain.signing import (NotASigner, n_lookup_programs, sign_transaction_b64, signer_index,
                                      static_program_ids, transaction_id)

SYSTEM = "11111111111111111111111111111111"


def _message(payer: Keypair, extra_signer: Keypair | None = None) -> MessageV0:
    ixs = [transfer(TransferParams(from_pubkey=payer.pubkey(), to_pubkey=Pubkey.new_unique(), lamports=1000))]
    if extra_signer is not None:
        ixs.append(Instruction(Pubkey.from_string(SYSTEM), b"\x00",
                               [AccountMeta(extra_signer.pubkey(), True, True), AccountMeta(Pubkey.new_unique(), False, True)]))
    return MessageV0.try_compile(payer.pubkey(), ixs, [], Hash.new_unique())


def _unsigned_b64(msg: MessageV0, sigs: list[Signature] | None = None) -> str:
    n = msg.header.num_required_signatures
    tx = VersionedTransaction.populate(msg, sigs or [Signature.default()] * n)
    return base64.b64encode(bytes(tx)).decode()


def test_sign_with_fee_payer_verifies_and_leaves_message_untouched():
    payer = Keypair()
    msg = _message(payer)
    signed_b64, idx = sign_transaction_b64(_unsigned_b64(msg), payer)
    assert idx == 0
    tx = VersionedTransaction.from_bytes(base64.b64decode(signed_b64))
    assert tx.signatures[0].verify(payer.pubkey(), to_bytes_versioned(tx.message))
    assert tx.verify_with_results() == [True]
    assert to_bytes_versioned(tx.message) == to_bytes_versioned(msg)
    assert tx.message.header == msg.header
    assert tx.message.account_keys[0] == payer.pubkey()
    assert transaction_id(signed_b64) == str(tx.signatures[0])


def test_keypair_that_is_not_a_signer_raises():
    payer, stranger = Keypair(), Keypair()
    with pytest.raises(NotASigner):
        sign_transaction_b64(_unsigned_b64(_message(payer)), stranger)


def test_present_but_non_signer_account_raises():
    payer, dest = Keypair(), Keypair()
    ix = transfer(TransferParams(from_pubkey=payer.pubkey(), to_pubkey=dest.pubkey(), lamports=1))
    msg = MessageV0.try_compile(payer.pubkey(), [ix], [], Hash.new_unique())
    assert dest.pubkey() in list(msg.account_keys)
    with pytest.raises(NotASigner, match="only the first 1"):
        signer_index(_unsigned_b64(msg), dest.pubkey())


def test_existing_other_signature_is_preserved():
    payer, other = Keypair(), Keypair()
    msg = _message(payer, other)
    assert msg.header.num_required_signatures == 2
    other_idx = list(msg.account_keys).index(other.pubkey())
    other_sig = other.sign_message(to_bytes_versioned(msg))
    sigs = [Signature.default()] * 2
    sigs[other_idx] = other_sig
    signed_b64, idx = sign_transaction_b64(_unsigned_b64(msg, sigs), payer)
    tx = VersionedTransaction.from_bytes(base64.b64decode(signed_b64))
    assert idx == 0 and other_idx == 1
    assert tx.signatures[other_idx] == other_sig
    assert tx.verify_with_results() == [True, True]


def test_second_signer_slot_can_be_signed_alone():
    payer, other = Keypair(), Keypair()
    msg = _message(payer, other)
    signed_b64, idx = sign_transaction_b64(_unsigned_b64(msg), other)
    tx = VersionedTransaction.from_bytes(base64.b64decode(signed_b64))
    assert idx == 1
    assert tx.signatures[0] == Signature.default()  # fee payer slot untouched
    assert tx.signatures[1].verify(other.pubkey(), to_bytes_versioned(msg))


def test_static_program_ids():
    payer = Keypair()
    b64 = _unsigned_b64(_message(payer))
    assert static_program_ids(b64) == [SYSTEM]
    assert n_lookup_programs(b64) == 0
