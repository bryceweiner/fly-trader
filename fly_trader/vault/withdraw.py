"""Principal withdrawal: the operator takes deposited SOL back out, only to one of his funding addresses.

    cap = min(D - Wd + min(0, R - A),  N - reserved - gas)

The first term is his principal net of any loss not yet earned back (so he never takes the lockers' profit or makes
them fund his loss twice); the second keeps owed SOL, open positions and the gas reserve untouched. The flow row is
written before broadcast and the transfer's fee is charged to the withdrawal, not to the lockers.
"""
from __future__ import annotations

import base64
import logging

from solders.hash import Hash
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from .. import config
from ..chain.cluster_guard import assert_vault_signing_allowed
from ..db.apilog import record_event
from ..db.connection import transaction
from . import nav, settle, walletlock

log = logging.getLogger(__name__)
FEE = 5000


def cap(conn, native: int, token_acct: int = 0) -> dict:
    cost, _ = settle.open_positions(conn)
    f = settle.flow_totals(conn, 2**62)
    r = settle.realized(native, token_acct, cost, f["deposits"], f["withdrawals"], f["payouts"])
    a = settle.allocated_total(conn)
    principal = f["deposits"] - f["withdrawals"] + min(0, r - a)
    liquid = native - nav.reserved_lamports(conn) - int(round(config.GAS_RESERVE_SOL * config.LAMPORTS_PER_SOL)) - FEE
    return {"principal": principal, "liquid": liquid, "cap": max(0, min(principal, liquid)), "realized": r, "allocated": a}


def withdraw(lamports: int, to: str, rpc, keypair, wait=None) -> dict:
    from ..execution.broker_live import await_confirmation
    if to not in set(config.FUNDING_ADDRESSES):
        raise ValueError("withdrawals go only to an address in FUNDING_ADDRESSES")
    assert_vault_signing_allowed()
    wait = wait or await_confirmation
    wallet = str(keypair.pubkey())
    with walletlock.exclusive(timeout_s=180):
        native = rpc.get_balance(wallet)
        with transaction() as conn:
            c = cap(conn, native)
        if lamports <= 0 or lamports > c["cap"]:
            raise ValueError(f"amount must be 1..{c['cap']} lamports (principal {c['principal']}, liquid {c['liquid']})")
        bh = rpc.get_latest_blockhash()
        msg = MessageV0.try_compile(keypair.pubkey(), [transfer(TransferParams(from_pubkey=keypair.pubkey(), to_pubkey=Pubkey.from_string(to),
                                                                             lamports=int(lamports)))], [], Hash.from_string(bh["blockhash"]))
        tx = VersionedTransaction(msg, [keypair])
        sig = str(tx.signatures[0])
        with transaction() as conn:                           # known before it can land: the flow scanner never mistakes it
            conn.execute("INSERT INTO vault_flows (signature, slot, block_time, direction, kind, counterparty, lamports, fee_lamports, classified_by, note) "
                         "VALUES (%s, 0, now(), 'out', 'withdrawal', %s, %s, %s, 'sender', 'principal withdrawal (pending)')",
                         (sig, to, int(lamports), FEE))
        rpc.send_transaction(base64.b64encode(bytes(tx)).decode())
        status, info = wait(rpc, sig, bh["last_valid_block_height"])
        with transaction() as conn:
            if status == "confirmed":
                conn.execute("UPDATE vault_flows SET slot = %s, note = 'principal withdrawal' WHERE signature = %s AND kind = 'withdrawal'",
                             (int((info or {}).get("slot") or 0), sig))
            else:
                conn.execute("DELETE FROM vault_flows WHERE signature = %s AND kind = 'withdrawal'", (sig,))
    record_event("info" if status == "confirmed" else "error", "vault", f"principal withdrawal {status}",
                 {"lamports": lamports, "to": to, "signature_prefix": sig[:12]})
    return {"status": status, "signature": sig, "lamports": lamports, "cap_before": c["cap"]}
