"""The treasury's on-chain setup, checked every minute (docs/vault/SPEC.md "Custody").

What must hold, and what the operator is told when it does not or when it changes:

* the multisig's members are the owner's wallets only: neither server key may ever be a member (a member could vote
  itself any power; the server gets only the two spending limits)
* L1: SOL, per day, member = the trading key, the ONLY destination = the trading wallet
* L2: SOL, per week, member = the payout key (any destination: holders)

A limit that disappears is the owner revoking it (or someone who took the owner wallet): the signer then refuses and
this alerts once. Every change to members, threshold or limits is reported with before and after.
"""
from __future__ import annotations

import logging

from solders.pubkey import Pubkey

from .. import config
from ..signer import squads
from . import alerts, state

log = logging.getLogger(__name__)


def treasury_address() -> str | None:
    return str(squads.vault_pda(Pubkey.from_string(config.VAULT_MULTISIG), 0)) if config.VAULT_MULTISIG else None


def _account(rpc, addr: str) -> bytes | None:
    info = rpc.call("getAccountInfo", [addr, {"encoding": "base64", "commitment": "confirmed"}])
    v = (info or {}).get("value")
    if not v or v.get("owner") != str(squads.PROGRAM_ID):
        return None
    import base64
    return base64.b64decode(v["data"][0])


def read(rpc) -> dict:
    """The current setup, as plain JSON (for state, alerts and the console)."""
    out: dict = {"multisig": config.VAULT_MULTISIG, "treasury": treasury_address()}
    data = _account(rpc, config.VAULT_MULTISIG)
    if data is None:
        out["error"] = "the multisig account is missing or not a Squads account"
        return out
    ms = squads.decode_multisig(data)
    out.update({"threshold": ms["threshold"], "time_lock": ms["time_lock"], "members": sorted(str(k) for k, _ in ms["members"]),
                "config_authority": str(ms["config_authority"])})
    for name, addr in (("L1", config.VAULT_LIMIT_TRADING), ("L2", config.VAULT_LIMIT_PAYOUT)):
        d = _account(rpc, addr) if addr else None
        if d is None:
            out[name] = None
            continue
        lim = squads.decode_spending_limit(d)
        out[name] = {"address": addr, "multisig": str(lim["multisig"]), "mint": str(lim["mint"]), "amount": lim["amount"], "period": lim["period"],
                     "remaining": lim["remaining"], "last_reset": lim["last_reset"], "members": [str(k) for k in lim["members"]],
                     "destinations": [str(k) for k in lim["destinations"]]}
    return out


def problems(c: dict, trading: str, payout: str) -> list[str]:
    """What is wrong with the setup (pure)."""
    if c.get("error"):
        return [c["error"]]
    bad = []
    for who, key in (("trading", trading), ("payout", payout)):
        if key in c.get("members", []):
            bad.append(f"the {who} key is a MEMBER of the treasury multisig: remove it (the server must only hold spending limits)")
    if str(c.get("config_authority")) not in ("11111111111111111111111111111111", "None"):
        bad.append(f"the multisig has a config authority ({c['config_authority']}): it can change members without a vote")
    l1, l2 = c.get("L1"), c.get("L2")
    if l1 is None:
        bad.append("L1 (the trading float's daily refill) is missing: the float cannot refill")
    else:
        if l1["multisig"] != c["multisig"] or l1["mint"] != "11111111111111111111111111111111":
            bad.append("L1 is not a SOL limit on this multisig")
        if l1["members"] != [trading]:
            bad.append(f"L1 members are {l1['members']}, expected only the trading key")
        if l1["destinations"] != [trading]:
            bad.append("L1 must have the trading wallet as its ONLY destination (else a stolen trading key could send treasury SOL anywhere)")
    if l2 is None:
        bad.append("L2 (weekly claims) is missing: claims will wait")
    else:
        if l2["multisig"] != c["multisig"] or l2["mint"] != "11111111111111111111111111111111":
            bad.append("L2 is not a SOL limit on this multisig")
        if l2["members"] != [payout]:
            bad.append(f"L2 members are {l2['members']}, expected only the payout key")
    return bad


def changes(before: dict | None, now: dict) -> list[str]:
    """Human-readable differences between two readings (pure)."""
    if not before:
        return []
    out = []
    for k, label in (("members", "multisig members"), ("threshold", "threshold"), ("time_lock", "time lock"), ("config_authority", "config authority")):
        if before.get(k) != now.get(k):
            out.append(f"{label}: {before.get(k)} -> {now.get(k)}")
    for name in ("L1", "L2"):
        b, n = before.get(name), now.get(name)
        if (b is None) != (n is None):
            out.append(f"{name} {'removed' if n is None else 'created'}")
        elif b and n:
            for k in ("amount", "period", "members", "destinations"):
                if b.get(k) != n.get(k):
                    fmt = (lambda v: f"{v / config.LAMPORTS_PER_SOL:g} SOL") if k == "amount" else str
                    out.append(f"{name} {k}: {fmt(b.get(k))} -> {fmt(n.get(k))}")
    return out


def check(rpc, trading: str, payout: str) -> dict:
    """Read, compare with the last reading, alert on problems and changes; stores the reading in vault_kv."""
    c = read(rpc)
    before = state.get("custody")
    for line in changes(before, c):
        alerts.send(f"treasury setup changed: {line}. If you did not do this in the Squads app, revoke L1 and L2 now.")
    bad = problems(c, trading, payout)
    if bad:
        alerts.send("treasury setup: " + "; ".join(bad), key="custody_problem", cooldown_s=6 * 3600)
    state.put("custody", c)
    return {**c, "problems": bad}


def owners() -> set[str]:
    """The multisig members as last read (their outbound treasury transfers are principal withdrawals)."""
    return set((state.get("custody") or {}).get("members") or [])
