"""Withdraw ETH from the RH bot wallet to one of the operator's own addresses (System → Safety & wallet in the console).

The Solana withdrawal's rules (chain/withdraw.py), in ETH:
- only to an address listed in RH_FUNDING_ADDRESSES, never on the hosted vault fly, and every key and chain check of
  rh/guard.py but RH_LIVE_ENABLED (guard.check_withdrawal);
- what is left covers, while live positions or base lots are open, the gas reserve (their exits pay gas from it);
- the wallet's lock (rh/wallet.py: one transaction in flight per address) is held from the balance read to the booking,
  so the executor and the reconciler never see a withdrawal half done;
- the transfer is an ``rh_txs`` row of kind 'transfer', committed before broadcast. Whichever path records its receipt
  (this one, ``settle_pending``, or the executor's recovery) books it once (``book``): the ETH sent becomes a
  'withdrawal' wallet flow, so rh/accounting.reconcile still finds the chain equal to the books, and circuit 3's
  kill-switch peak re-bases at it;
- a withdrawal still without a receipt is settled before the next one (rh/exec.settle_tx): mined, superseded at its
  nonce, rebroadcast or cancelled.
"""
from __future__ import annotations

from decimal import Decimal

from .. import config
from ..agent import rails
from ..db.apilog import record_event
from ..db.connection import transaction
from ..markets import RH, RH_CIRCUIT
from ..vault.evm import is_address
from . import guard

WEI = 10 ** 18
RECEIPT_WAIT_S = 120.0


class WithdrawRefused(RuntimeError):
    """Nothing was signed; the message says why."""


def destinations() -> list[str]:
    """The addresses ETH may go to: the valid EVM addresses in RH_FUNDING_ADDRESSES, lowercase."""
    return [a.lower() for a in config.RH_FUNDING_ADDRESSES if is_address(a)]


def wallet(to: str, rpc=None):
    """The RH bot wallet, for a withdrawal to ``to`` (guard.check_withdrawal)."""
    from .rpc import RhRpc
    from .wallet import RhWallet
    rpc = rpc or RhRpc()
    key, addr = guard.check_withdrawal(rpc, to)
    return RhWallet(rpc, key, addr)


def limits(w, to: str) -> dict:
    """Fresh reads: the wallet's wei, the transfer's gas and fee cap, what must stay (the gas reserve while live positions or
    base lots are open), and the most that can go out with the fee counted at its cap."""
    balance = w.balance(); mx, tip = w.fees(); gas = w.estimate(to, b"", 0)
    with transaction() as conn:
        n_open = int(conn.execute("SELECT (SELECT count(*) FROM positions WHERE book = %s AND status = 'open') + "
                                  "(SELECT count(*) FROM rh_base_lots WHERE status = 'open') AS n", (RH.live_book,)).fetchone()["n"])
    keep = int(config.RH_GAS_RESERVE_ETH * WEI) if n_open else 0
    return {"balance": balance, "gas": gas, "max_fee": mx, "tip": tip, "fee": gas * mx, "keep": keep, "n_open": n_open,
            "max": max(0, balance - gas * mx - keep)}


def plan(lim: dict, eth: float | None) -> tuple[int, str | None]:
    """(wei to send, why not or None) for ``eth`` ETH, or everything withdrawable when ``eth`` is None."""
    wei = lim["max"] if eth is None else int(Decimal(str(eth)) * WEI)
    if wei <= 0:
        return 0, "nothing to withdraw" if eth is None else "enter an amount"
    if wei > lim["max"]:
        return wei, (f"at most {lim['max'] / WEI:.6f} ETH can go out"
                     + (f": {lim['keep'] / WEI:g} ETH stays for the gas of {lim['n_open']} open live position(s)" if lim["keep"] else " after the gas"))
    return wei, None


def book(conn, tx_id: int, ok: bool) -> None:
    """A withdrawal's receipt, in the transaction that records it (rh/wallet.RhWallet.record_receipt): the row is marked
    applied and, if it went through, its ETH is booked as a 'withdrawal' wallet flow and circuit 3's peak re-based. Once."""
    tx = conn.execute("UPDATE rh_txs SET applied_at = now() WHERE id = %s AND applied_at IS NULL RETURNING value_wei, to_addr, hash, block",
                      (tx_id,)).fetchone()
    if tx is None or not ok:
        return
    conn.execute("INSERT INTO rh_wallet_flows (block, direction, kind, wei, note) VALUES (%s, 'out', 'withdrawal', %s, %s)",
                 (tx["block"], int(tx["value_wei"]), f"to {tx['to_addr']} ({tx['hash']})"))
    rails.record_withdrawal(conn, {"eth": int(tx["value_wei"]) / WEI, "to": tx["to_addr"], "hash": tx["hash"]}, RH_CIRCUIT)


def settle_pending(w) -> list[str]:
    """Settle the wallet's withdrawals that have no receipt yet (a receipt books them). Returns the hashes still in flight.
    Under the wallet's lock."""
    from .exec import settle_tx
    q = ("SELECT * FROM rh_txs WHERE from_addr = %s AND kind = 'transfer' AND applied_at IS NULL AND status IN ('signed', 'sent') "
         "ORDER BY nonce, id")
    with transaction() as conn:
        rows = [dict(r) for r in conn.execute(q, (w.address,)).fetchall()]
    for tx in rows:
        settle_tx(w, tx)
    with transaction() as conn:
        return [r["hash"] for r in conn.execute(q, (w.address,)).fetchall()]


def withdraw(eth: float | None, to: str, rpc=None, w=None) -> dict:
    """Send ``eth`` ETH (None: everything withdrawable) from the RH bot wallet to ``to``. Returns {status, hash, wei, to,
    block}; status 'mined_ok', 'reverted' (only the gas moved) or 'pending' (no receipt yet: it is booked when one is
    recorded). Raises WithdrawRefused, guard.SigningRefused, wallet.FeeTooHigh or wallet.SendFailed when nothing went out."""
    to = to.lower()
    if w is None:
        w = wallet(to, rpc)
    else:
        guard.check_withdrawal(w.rpc, to)                              # the same checks for a wallet built elsewhere
    with w.lock:
        if left := settle_pending(w):
            raise WithdrawRefused(f"an earlier withdrawal ({left[0]}) has not landed yet: try again in a few minutes")
        lim = limits(w, to)
        wei, why = plan(lim, eth)
        if why:
            raise WithdrawRefused(why)
        s = w.send(to=to, data=b"", value=wei, kind="transfer", gas=lim["gas"], fees=(lim["max_fee"], lim["tip"]))
        rc = w.wait(s["hash"], RECEIPT_WAIT_S)
        if rc is None:
            return {"status": "pending", "hash": s["hash"], "wei": wei, "to": to, "block": None}
        with transaction() as conn:
            res = w.record_receipt(conn, s["id"], rc)                  # books it
    status = "mined_ok" if res["ok"] else "reverted"
    record_event("info" if res["ok"] else "warning", "rh_wallet", f"withdrawal of {wei / WEI:.6f} ETH to {to}: {status}", {"hash": s["hash"]})
    return {"status": status, "hash": s["hash"], "wei": wei, "to": to, "block": res["block"]}
