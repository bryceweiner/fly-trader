"""Withdraw SOL from the bot wallet to one of the operator's own wallets (System → Safety & wallet in the console).

- Only to a wallet listed in FUNDING_ADDRESSES and only on mainnet; never on the hosted vault fly, whose SOL leaves
  through the treasury (cluster_guard.assert_withdraw_allowed). No LIVE_ENABLED: a withdrawal takes money out of the
  fly's reach and never trades.
- What is left is 0 or at least the rent-exempt minimum, and while live positions are open at least the gas reserve
  (their exits pay fees from it). An empty destination must receive at least the rent-exempt minimum.
- It holds the live broker's in-flight lock, so a swap's balance delta never contains a withdrawal.
- The signed transaction is recorded (wallet_events 'withdrawal', status 'signed') before it is broadcast; its outcome is
  added to that row. A confirmed withdrawal re-bases the live book's kill-switch peak (agent/rails.record_withdrawal):
  money taken out is not a drawdown.
"""
from __future__ import annotations

import base64
import logging
from decimal import Decimal

from psycopg.types.json import Jsonb
from solders.hash import Hash
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from .. import config
from ..agent import rails
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import scrub
from ..markets import SOL
from .cluster_guard import assert_withdraw_allowed
from .rpc import RpcError, RpcTransportError

log = logging.getLogger(__name__)
LAMPORTS = config.LAMPORTS_PER_SOL
FEE_LAMPORTS = 5_000                 # one signature, no priority fee
RENT_EXEMPT_MIN = 890_880            # a system account without data: a balance below it (other than 0) is refused
LOCK_WAIT_S = 120.0                  # a swap holds the broker's lock until it confirms or expires


class WithdrawRefused(RuntimeError):
    """Nothing was signed; the message says why."""


def destinations() -> list[str]:
    """The wallets SOL may go to: the valid Solana addresses in FUNDING_ADDRESSES."""
    out = []
    for a in config.FUNDING_ADDRESSES:
        try:
            Pubkey.from_string(a)
            out.append(a)
        except ValueError:
            log.warning("FUNDING_ADDRESSES: %r is not a Solana address", a)
    return out


def limits(rpc, me: str, to: str | None = None) -> dict:
    """Fresh reads: the wallet's lamports, what must stay (the gas reserve while live positions are open), the most that
    can go out, and the destination's lamports."""
    balance = int(rpc.get_balance(me))
    with transaction() as conn:
        n_open = int(conn.execute("SELECT count(*) AS n FROM positions WHERE book = %s AND status = 'open'", (SOL.live_book,)).fetchone()["n"])
    keep = max(int(config.GAS_RESERVE_SOL * LAMPORTS), RENT_EXEMPT_MIN) if n_open else 0
    return {"balance": balance, "fee": FEE_LAMPORTS, "keep": keep, "n_open": n_open, "max": max(0, balance - FEE_LAMPORTS - keep),
            "to_balance": int(rpc.get_balance(to)) if to else None}


def plan(lim: dict, sol: float | None) -> tuple[int, str | None]:
    """(lamports to send, why not or None) for ``sol`` SOL, or everything withdrawable when ``sol`` is None."""
    lam = lim["max"] if sol is None else int(Decimal(str(sol)) * LAMPORTS)
    if lam <= 0:
        return 0, "nothing to withdraw" if sol is None else "enter an amount"
    if lam > lim["max"]:
        return lam, (f"at most {lim['max'] / LAMPORTS:.6f} SOL can go out"
                     + (f": {lim['keep'] / LAMPORTS:g} SOL stays for the fees of {lim['n_open']} open live position(s)" if lim["keep"] else " after the fee"))
    left = lim["balance"] - lam - lim["fee"]
    if 0 < left < RENT_EXEMPT_MIN:
        return lam, f"Solana refuses to leave {left / LAMPORTS:.6f} SOL behind: leave at least {RENT_EXEMPT_MIN / LAMPORTS:g} SOL or withdraw everything"
    if lim.get("to_balance") == 0 and lam < RENT_EXEMPT_MIN:
        return lam, f"an empty wallet must receive at least {RENT_EXEMPT_MIN / LAMPORTS:g} SOL"
    return lam, None


def withdraw(sol: float | None, to: str, rpc=None, keypair=None) -> dict:
    """Send ``sol`` SOL (None: everything withdrawable) from the bot wallet to ``to``. Returns {status, signature, lamports,
    to, slot}; status 'confirmed', 'failed_on_chain' (only the fee moved) or 'expired' (nothing moved). Raises
    WithdrawRefused, cluster_guard.SigningRefused or keys.WalletKeyError when nothing was signed."""
    from ..execution.broker_live import _INFLIGHT, await_confirmation
    assert_withdraw_allowed(to)
    if keypair is None:
        from .keys import load_keypair
        keypair = load_keypair()
    if rpc is None:
        from .rpc import HttpSolanaRpc
        rpc = HttpSolanaRpc()
    me = str(keypair.pubkey())
    if not _INFLIGHT.acquire(timeout=LOCK_WAIT_S):
        raise WithdrawRefused("a swap is still in flight: try again in a minute")
    try:
        lam, why = plan(limits(rpc, me, to), sol)
        if why:
            raise WithdrawRefused(why)
        bh = rpc.get_latest_blockhash()
        ix = transfer(TransferParams(from_pubkey=keypair.pubkey(), to_pubkey=Pubkey.from_string(to), lamports=lam))
        tx = VersionedTransaction(MessageV0.try_compile(keypair.pubkey(), [ix], [], Hash.from_string(bh["blockhash"])), [keypair])
        sig = str(tx.signatures[0])
        detail = {"to": to, "lamports": lam, "sol": lam / LAMPORTS, "signature": sig, "last_valid_block_height": bh["last_valid_block_height"], "status": "signed"}
        with transaction() as conn:                                     # recorded before the network ever sees it
            eid = conn.execute("INSERT INTO wallet_events (kind, pubkey, detail) VALUES ('withdrawal', %s, %s) RETURNING id", (me, Jsonb(detail))).fetchone()["id"]
        try:
            rpc.send_transaction(base64.b64encode(bytes(tx)).decode())
        except RpcError as e:                                           # refused by the node (preflight): never broadcast
            with transaction() as conn:
                conn.execute("UPDATE wallet_events SET detail = detail || %s WHERE id = %s", (Jsonb({"status": "rejected", "error": scrub(str(e))[:300]}), eid))
            raise WithdrawRefused(f"the network refused it: {scrub(str(e))[:200]}") from None
        except RpcTransportError as e:                                  # it may have gone out: its status decides
            log.warning("withdrawal %s: send failed in transport (%s); polling its status", sig, e)
        status, st = await_confirmation(rpc, sig, bh["last_valid_block_height"])
        slot = int((st or {}).get("slot") or 0)
        with transaction() as conn:
            conn.execute("UPDATE wallet_events SET detail = detail || %s WHERE id = %s", (Jsonb({"status": status, "slot": slot}), eid))
            if status == "confirmed":
                rails.record_withdrawal(conn, {"sol": lam / LAMPORTS, "to": to, "signature": sig})
    finally:
        _INFLIGHT.release()
    record_event("info" if status == "confirmed" else "warning", "wallet", f"withdrawal of {lam / LAMPORTS:.6f} SOL to {to}: {status}",
                 {"signature": sig, "slot": slot})
    return {"status": status, "signature": sig, "lamports": lam, "to": to, "slot": slot}
