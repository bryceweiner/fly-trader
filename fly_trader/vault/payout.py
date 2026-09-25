"""The payout wallet: claims are paid from a second wallet, so a holder who claims never learns the trading wallet.

After each settlement the fly moves the SOL now owed to lockers from the trading wallet to the payout wallet, plus a
small buffer for claim fees ("sweep"). Every claim is paid from the payout wallet. Following that one weekly transfer
back still leads to the trading wallet, but it takes deliberate on-chain digging instead of a glance at a wallet app.

Accounting treats both wallets as one: R, NAV and the performance index count their SOL together, so a sweep is
neither profit nor loss. The trading bankroll reserves only the part of what is owed that has not been swept yet.
"""
from __future__ import annotations

import base64
import logging
from pathlib import Path

from solders.hash import Hash
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from .. import config
from ..chain.cluster_guard import assert_vault_signing_allowed
from ..db.apilog import record_event
from ..db.connection import transaction
from . import alerts, state, walletlock

log = logging.getLogger(__name__)
FEE = 5000


def load_keypair():
    """The payout key: its own file (``PAYOUT_PRIVATE_KEY_FILE``), made on the server like the trading key."""
    from ..chain.keys import _keypair_from
    path = config.PAYOUT_PRIVATE_KEY_FILE
    if not path:
        raise RuntimeError("PAYOUT_PRIVATE_KEY_FILE is not set")
    return _keypair_from(Path(path).read_text(encoding="utf-8"))


def cached_balance() -> int:
    """The payout wallet's lamports as the vault worker last read them (0 until then: reserves more, never less)."""
    return int((state.get("payout_balance") or {}).get("lamports") or 0)


def refresh_balance(rpc, pubkey: str) -> int:
    lam = int(rpc.get_balance(pubkey))
    state.put("payout_balance", {"lamports": lam, "pubkey": pubkey})
    return lam


def owed_total(conn) -> int:
    """Allocated to lockers and not yet paid out (in-flight claims included: their SOL has not left yet)."""
    r = conn.execute("SELECT COALESCE((SELECT sum(allocated) FROM vault_settlements WHERE status = 'allocated'), 0) - "
                     "COALESCE((SELECT sum(lamports) FROM vault_flows WHERE kind = 'claim'), 0) AS r").fetchone()
    return max(0, int(r["r"] or 0))


def sweep_amount(owed: int, payout_lamports: int, trading_liquid: int) -> int:
    """How much to move now: enough for everything owed plus the fee buffer, never more than the trading wallet can
    spare above its gas reserve. Pure."""
    want = max(0, int(owed) + int(config.PAYOUT_FEE_BUFFER_LAMPORTS) - int(payout_lamports))
    if want < int(config.PAYOUT_MIN_SWEEP_LAMPORTS) and payout_lamports >= owed:
        return 0                                          # topping up the fee buffer by dust is not worth a transaction
    return max(0, min(want, int(trading_liquid)))


def sweep(rpc, trading_kp, payout_pubkey: str, wait=None) -> dict:
    """Move owed SOL to the payout wallet. Under the wallet lock; the flow row is written before broadcast so the
    trading wallet's scanner knows the transfer is ours."""
    from ..execution.broker_live import await_confirmation
    assert_vault_signing_allowed()
    wait = wait or await_confirmation
    with walletlock.exclusive(timeout_s=180):
        payout_bal = refresh_balance(rpc, payout_pubkey)
        native = int(rpc.get_balance(str(trading_kp.pubkey())))
        with transaction() as conn:
            owed = owed_total(conn)
        liquid = native - int(round(config.GAS_RESERVE_SOL * config.LAMPORTS_PER_SOL)) - FEE
        amount = sweep_amount(owed, payout_bal, liquid)
        if amount <= 0:
            if payout_bal < owed:
                alerts.send(f"payout wallet short: owes {owed} lamports, holds {payout_bal}, trading wallet cannot spare more",
                            key="payout_short", cooldown_s=6 * 3600)
            return {"swept": 0, "owed": owed, "payout": payout_bal}
        bh = rpc.get_latest_blockhash()
        msg = MessageV0.try_compile(trading_kp.pubkey(), [transfer(TransferParams(from_pubkey=trading_kp.pubkey(),
                                                                                  to_pubkey=Pubkey.from_string(payout_pubkey), lamports=amount))],
                                    [], Hash.from_string(bh["blockhash"]))
        tx = VersionedTransaction(msg, [trading_kp])
        sig = str(tx.signatures[0])
        with transaction() as conn:
            conn.execute("INSERT INTO vault_flows (signature, slot, block_time, direction, kind, counterparty, lamports, fee_lamports, classified_by, note) "
                         "VALUES (%s, 0, now(), 'out', 'sweep', %s, %s, %s, 'sender', 'owed SOL to the payout wallet (pending)')",
                         (sig, payout_pubkey, amount, FEE))
        try:
            rpc.send_transaction(base64.b64encode(bytes(tx)).decode())
        except Exception as e:
            log.warning("sweep send failed: %s", type(e).__name__)
        status, info = wait(rpc, sig, bh["last_valid_block_height"])
        with transaction() as conn:
            if status == "confirmed":
                conn.execute("UPDATE vault_flows SET slot = %s, note = 'owed SOL to the payout wallet' WHERE signature = %s AND kind = 'sweep'",
                             (int((info or {}).get("slot") or 0), sig))
            else:
                conn.execute("DELETE FROM vault_flows WHERE signature = %s AND kind = 'sweep'", (sig,))
        refresh_balance(rpc, payout_pubkey)
    record_event("info" if status == "confirmed" else "warning", "vault", f"sweep to payout wallet {status}",
                 {"lamports": amount, "owed": owed, "signature_prefix": sig[:12]})
    return {"swept": amount if status == "confirmed" else 0, "status": status, "owed": owed}
