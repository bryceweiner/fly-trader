"""Principal withdrawal: the operator takes deposited SOL back out of the treasury, from his own wallet in the Squads
app (the server's keys cannot: the treasury's spending limits only refill the trading float and pay holders).

    cap = min(D - Wd + min(0, R - A),  treasury - owed - rent)

The first term is his principal net of any loss not yet earned back (so he never takes the lockers' profit or makes
them fund his loss twice); the second leaves what holders are owed in the treasury. The flow scanner books the
transfer as a withdrawal when it sees it (a multisig member signed it) and alerts; a withdrawal to an address outside
``FUNDING_ADDRESSES`` alerts louder.
"""
from __future__ import annotations

from .. import config
from . import nav, payout, settle


def cap(conn, trading_native: int, treasury_native: int, token_acct: int = 0) -> dict:
    cost, _ = settle.open_positions(conn)
    f = settle.flow_totals(conn, 2**62)
    native = int(trading_native) + int(treasury_native)
    r = settle.realized(native, token_acct, cost, f["deposits"], f["withdrawals"], f["payouts"])
    a = settle.allocated_total(conn)
    principal = f["deposits"] - f["withdrawals"] + min(0, r - a)
    liquid = int(treasury_native) - nav.reserved_lamports(conn) - payout.TREASURY_RENT
    return {"principal": principal, "treasury_liquid": liquid, "cap": max(0, min(principal, liquid)), "realized": r, "allocated": a}


def instructions(c: dict, treasury: str) -> str:
    sol = c["cap"] / config.LAMPORTS_PER_SOL
    to = ", ".join(config.FUNDING_ADDRESSES) or "(set FUNDING_ADDRESSES)"
    return (f"You may withdraw up to {sol:.6f} SOL (principal {c['principal'] / config.LAMPORTS_PER_SOL:.6f}, "
            f"treasury free {c['treasury_liquid'] / config.LAMPORTS_PER_SOL:.6f}).\n"
            f"In the Squads app (app.squads.so) with your owner wallet: open the multisig, vault 0 ({treasury}),\n"
            f"Send -> SOL -> at most {sol:.6f} -> to one of your funding addresses: {to}. Approve and execute.\n"
            "The fly books it as a withdrawal when it lands (and tells you on Telegram).")
