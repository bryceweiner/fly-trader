"""$FLY vault: the trading wallet's ledger (deposits, profit, claims), settlements, the claims queue, halts and the
Robinhood Chain index. Only meaningful on the hosted vault fly (VAULT_ENABLED): every topic answers
``{"enabled": False, "message": ...}`` and every action refuses while it is off.

Two writes, both the operator's: clearing a halt (after checking its reasons) and relabelling one wallet flow. Principal
withdrawals stay CLI-only (`fly-trader vault withdraw`)."""
from __future__ import annotations

import re

from ... import config
from ...db.queries import q, q1
from ...vault import flows, state
from ..hub import ActionError, action, topic
from ..state import jv

OFF = "The vault is off on this machine (VAULT_ENABLED is not set). It runs on the hosted vault fly."
KINDS = ("deposit", "profit", "internal")
SIGNATURE = re.compile(r"^[1-9A-HJ-NP-Za-km-z]{32,100}$")      # a Solana transaction signature (base58)


def _off() -> dict | None:
    return None if config.VAULT_ENABLED else {"enabled": False, "message": OFF}


def _require_on() -> None:
    if not config.VAULT_ENABLED:
        raise ActionError(OFF)


def solscan_tx() -> str:
    from ...ops.wallets import explorer
    return explorer("sol", "tx", "")


@topic("vault.summary", every=15)
def summary(params: dict) -> dict:
    """The halt banner and the six ledger numbers (lamports; plus profit received and the last 180 NAV marks for the charts),
    with how far the wallet scan and the RH index have got."""
    if off := _off():
        return off
    h = state.halted()
    f = q1("SELECT COALESCE(sum(lamports) FILTER (WHERE kind='deposit'),0) d, COALESCE(sum(lamports+fee_lamports) FILTER (WHERE kind='withdrawal'),0) w, "
           "COALESCE(sum(lamports) FILTER (WHERE kind='claim'),0) p, COALESCE(sum(lamports) FILTER (WHERE kind='profit'),0) g FROM vault_flows") or {}
    a = q1("SELECT COALESCE(sum(allocated),0) a FROM vault_settlements WHERE status='allocated'") or {}
    marks = q("SELECT ts, wealth FROM (SELECT ts, wealth FROM vault_nav ORDER BY ts DESC LIMIT 180) n ORDER BY ts")
    nav = marks[-1] if marks else {}
    scan = flows.cursor()
    ix = q1("SELECT slot, detail, updated_at FROM vault_scan WHERE name='rh_vault'") or {}
    allocated, paid = int(a.get("a") or 0), int(f.get("p") or 0)
    return {"enabled": True, "halted": {"reasons": [str(r) for r in h["reasons"]]} if h else None,
            "nav": int(nav.get("wealth") or 0), "nav_at": nav.get("ts"), "nav_spark": [int(r["wealth"]) for r in marks],
            "deposits": int(f.get("d") or 0), "withdrawn": int(f.get("w") or 0), "profit": int(f.get("g") or 0),
            "allocated": allocated, "paid": paid, "owed": max(0, allocated - paid),
            "scan_slot": scan.get("slot"), "rh_block": ix.get("slot"), "rh_updated_at": ix.get("updated_at"), "impl": jv(ix.get("detail")).get("impl")}


@topic("vault.tables", every=30)
def tables(params: dict) -> dict:
    """Settlements (last 52 weeks), claims (last 100), wallet flows (last 200) and holders, amounts in SOL."""
    if off := _off():
        return off
    return {"enabled": True, "tx_url": solscan_tx(),
            "settlements": q("SELECT id, period_end, status, realized/1e9 AS realized_sol, pot/1e9 AS pot_sol, allocated/1e9 AS allocated_sol, "
                             "carried/1e9 AS carried_sol, earners, inputs_sha256 FROM vault_settlements ORDER BY period_end DESC LIMIT 52"),
            "claims": q("SELECT id, relay_id, evm, sol, status, reason, lamports/1e9 AS sol_amount, tx_signature, updated_at FROM vault_claims ORDER BY id DESC LIMIT 100"),
            "flows": q("SELECT id, block_time, kind, direction, counterparty, lamports/1e9 AS sol_amount, classified_by, signature, note FROM vault_flows ORDER BY id DESC LIMIT 200"),
            "holders": q("SELECT evm, allocated/1e9 AS allocated_sol, claimed/1e9 AS claimed_sol, owed/1e9 AS owed_sol FROM vault_accounts ORDER BY owed DESC")}


@action("vault.resume")
def resume(body: dict) -> dict:
    """Clear a halt (vault_kv 'halt' and 'panic'), as `fly-trader vault resume`. Refused when nothing is halted, so a stale
    page cannot clear a halt raised after the operator last looked."""
    _require_on()
    seen = body.get("reasons")
    from ...db.connection import transaction
    with transaction() as conn:                       # compare and clear in one transaction: a halt raised meanwhile is never cleared
        r = conn.execute("SELECT value FROM vault_kv WHERE key = 'halt' FOR UPDATE").fetchone()
        h = r["value"] if r and isinstance(r["value"], dict) else None
        if not h or not h.get("reasons"):
            raise ActionError("The vault is not halted: there is nothing to resume.")
        if not isinstance(seen, list) or not seen:
            raise ActionError("reasons: the halt reasons this page showed are required", 400)
        if [str(x) for x in seen] != [str(x) for x in h["reasons"]]:
            raise ActionError("The halt reasons changed since this page showed them: " + "; ".join(map(str, h["reasons"])) + ". Check them and resume again.")
        for key in ("halt", "panic"):                 # what vault/state.resume clears
            state.put(key, None, conn=conn)
    return {"message": "Vault halt cleared. New entries stay paused until you resume them (Safety page, or fly-trader resume-entries)."}


@action("vault.reclassify")
def reclassify(body: dict) -> dict:
    """Relabel one transaction's flow: deposit | profit | internal (vault/flows.reclassify). A change after its week
    settled only moves future weeks."""
    _require_on()
    sig, kind = body.get("signature"), body.get("kind")
    if not isinstance(sig, str) or not sig.strip():
        raise ActionError("Enter the transaction signature.", 400)
    sig = sig.strip()
    if not SIGNATURE.match(sig):
        raise ActionError("That is not a Solana transaction signature (base58, no spaces).", 400)
    if kind not in KINDS:
        raise ActionError("The label must be deposit, profit or internal.", 400)
    try:
        n = flows.reclassify(sig, kind)
    except (LookupError, ValueError) as e:
        raise ActionError(str(e)) from e
    return {"message": f"{n} row(s) changed", "changed": n}
