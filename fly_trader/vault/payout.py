"""The treasury and the trading float (docs/vault/SPEC.md "Custody").

Everything beyond a small float lives in a Squads v4 treasury (the multisig's vault 0) owned by Bryce's wallet. The
server's keys can take SOL out of it only through two on-chain spending limits: L1 lets the trading key refill ITS
OWN wallet (per day), L2 lets the payout key pay holders (per week, claims.py). A stolen server can therefore lose at
most the float plus what is left of those two limits, until the owner revokes them from his phone.

Every minute ``rebalance`` keeps the trading wallet's SOL near ``TREASURY_FLOAT_SOL``: a top-up (treasury -> trading,
L1) when it is low, a sweep (trading -> treasury) when it holds more, or when the treasury could not cover what is
owed. The signer (fly_trader/signer) builds and signs both; the flow row is written before broadcast so the scanner
knows the transfer is ours.

Accounting treats trading wallet + treasury as one book: R, NAV and the performance index count their SOL together,
so neither move is profit, loss or a flow. The trading bankroll holds back only the owed SOL the treasury lacks.
"""
from __future__ import annotations

import logging

from .. import config
from ..chain.cluster_guard import assert_vault_signing_allowed
from ..db.apilog import record_event
from ..db.connection import transaction
from ..signer.client import SignerUnavailable
from ..signer.policy import PolicyError
from . import alerts, state, walletlock

log = logging.getLogger(__name__)
FEE = 5000
TREASURY_RENT = 890_880            # the treasury PDA is a 0-byte system account: it keeps this much


def cached_balance() -> int:
    """The treasury's lamports as the vault worker last read them (0 until then: reserves more, never less)."""
    return int((state.get("treasury_balance") or {}).get("lamports") or 0)


def refresh_balance(rpc, treasury: str) -> int:
    lam = int(rpc.get_balance(treasury))
    state.put("treasury_balance", {"lamports": lam})
    return lam


def owed_total(conn) -> int:
    """Allocated to lockers and not yet paid out (in-flight claims included: their SOL has not left yet)."""
    r = conn.execute("SELECT COALESCE((SELECT sum(allocated) FROM vault_settlements WHERE status = 'allocated'), 0) - "
                     "COALESCE((SELECT sum(lamports) FROM vault_flows WHERE kind = 'claim'), 0) AS r").fetchone()
    return max(0, int(r["r"] or 0))


def plan(trading: int, treasury: int, owed: int, float_target: int, gas_reserve: int, min_move: int) -> tuple[str | None, int]:
    """What to move now. Pure. The treasury must hold everything owed first; then the trading wallet is brought to
    ``float_target``. Returns ('sweep' | 'topup' | None, lamports)."""
    spare_trading = max(0, trading - gas_reserve - FEE)                  # what trading can give without touching gas
    short = max(0, owed + TREASURY_RENT - treasury)                      # the treasury cannot pay what it owes
    excess = max(0, trading - float_target)
    give = min(spare_trading, max(excess, short))
    if give >= min_move or (short > 0 and give > 0):
        return "sweep", give
    want = float_target - trading
    available = max(0, treasury - TREASURY_RENT - owed)                  # never lend out what holders are owed
    take = min(want, available)
    if want >= min_move and take >= min_move:
        return "topup", take
    return None, 0


def rebalance(rpc, signer, trading: str, treasury: str, wait=None) -> dict:
    """One round under the wallet lock: plan, ask the signer, record, broadcast, confirm."""
    from ..execution.broker_live import await_confirmation
    assert_vault_signing_allowed()
    wait = wait or await_confirmation
    L = config.LAMPORTS_PER_SOL
    with walletlock.exclusive(timeout_s=180):
        tb = refresh_balance(rpc, treasury)
        native = int(rpc.get_balance(trading))
        with transaction() as conn:
            owed = owed_total(conn)
        target = 0 if state.get("panic") else int(config.TREASURY_FLOAT_SOL * L)   # panic: everything but gas goes back
        action, amount = plan(native, tb, owed, target, int(config.GAS_RESERVE_SOL * L), int(config.TOPUP_MIN_SOL * L))
        if action is None:
            return {"action": None, "trading": native, "treasury": tb, "owed": owed}
        try:
            built = signer.call("topup" if action == "topup" else "return_float", lamports=amount)
        except PolicyError as e:
            if action == "topup" and e.code == "cap":
                alerts.send(f"the trading float cannot refill: {e}", key="topup_cap", cooldown_s=6 * 3600)
            elif action == "topup" and e.code == "liquidity":
                alerts.send("the treasury has nothing left to lend the trading float", key="topup_liquidity", cooldown_s=6 * 3600)
            else:
                alerts.send(f"treasury {action} refused by the signer: {e}", key=f"{action}_refused", cooldown_s=3600)
            return {"action": action, "refused": str(e), "code": e.code}
        except SignerUnavailable as e:
            log.warning("treasury %s: signer unavailable: %s", action, e)
            return {"action": action, "refused": str(e), "code": "error"}
        if native < 2 * FEE:                             # the trading wallet pays its own top-up's fee
            alerts.send("the trading wallet has no SOL for fees, so it cannot refill itself from the treasury: send it 0.05 SOL "
                        "(from a funding address)", key="trading_no_fee", cooldown_s=6 * 3600)
            return {"action": action, "refused": "no SOL for the fee", "code": "liquidity"}
        sig = built["signature"]
        direction, counterparty, note = (("in", treasury, "float refill from the treasury (L1)") if action == "topup"
                                         else ("out", treasury, "float excess to the treasury"))
        with transaction() as conn:
            conn.execute("INSERT INTO vault_flows (signature, slot, block_time, direction, kind, counterparty, lamports, fee_lamports, classified_by, note) "
                         "VALUES (%s, 0, now(), %s, %s, %s, %s, %s, 'sender', %s)",
                         (sig, direction, action, counterparty, amount, FEE, note + " (pending)"))
        try:
            rpc.send_transaction(built["tx"])
        except Exception as e:
            log.warning("treasury %s send failed: %s", action, type(e).__name__)
        status, info = wait(rpc, sig, int(built["last_valid_block_height"]))
        with transaction() as conn:
            if status == "confirmed":
                conn.execute("UPDATE vault_flows SET slot = %s, note = %s WHERE signature = %s AND kind = %s",
                             (int((info or {}).get("slot") or 0), note, sig, action))
            else:
                conn.execute("DELETE FROM vault_flows WHERE signature = %s AND kind = %s", (sig, action))
        refresh_balance(rpc, treasury)
    record_event("info" if status == "confirmed" else "warning", "vault", f"treasury {action} {status}",
                 {"lamports": amount, "owed": owed, "signature_prefix": sig[:12]})
    return {"action": action, "lamports": amount if status == "confirmed" else 0, "status": status, "owed": owed}
