"""Weekly settlement: how much the lockers earned and who gets it (plan "Core accounting"; all integers in lamports).

    R   = N + K + C - D + Wd + P        realized profit since the wallet's first lamport
    pot = max(0, R - A)                 A = everything ever allocated: losses and dust carry forward by construction
    out = min(pot, N - (A - P) - gas)   never allocate SOL that is not liquid (the rest stays in the pot)

N native SOL, K lamports in token accounts that hold no open position (wSOL, rent of empty accounts), C cost of open
positions, D deposits, Wd principal withdrawals (+ their fees), P claims paid. Open positions sit at cost, so only
closed trades move R, and every fee, failed transaction, rent refund, sweep and gift is counted exactly once.

Two phases, because the two chains finalize at different speeds:
1. **snapshot** at the period end T1: the wallet (finalized, consistent with our own fills) and the flows up to its slot.
2. **allocate** once Robinhood Chain has finalized past T1: time-weighted earning balances over [T0, T1) from FlyVault
   events, then whole lamports per holder, floored.
"""
from __future__ import annotations

import hashlib
import json
import logging
import time
from datetime import datetime, timezone

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from . import state

log = logging.getLogger(__name__)
LAMPORTS = config.LAMPORTS_PER_SOL
FINALITY_WAIT_S = 120.0


# ---------------------------------------------------------------- pure math
def period_end_at_or_before(t: float, period_s: int | None = None, epoch: int | None = None) -> int:
    p = int(period_s or config.VAULT_PERIOD_S)
    e = int(config.VAULT_EPOCH if epoch is None else epoch)
    return e + ((int(t) - e) // p) * p


def realized(native: int, token_acct: int, open_cost: int, deposits: int, withdrawals: int, payouts: int) -> int:
    return int(native) + int(token_acct) + int(open_cost) - int(deposits) + int(withdrawals) + int(payouts)


def weights(start: dict[str, int], events: list[tuple[int, str, int]], t0: int, t1: int) -> dict[str, int]:
    """Token-seconds each account earned in [t0, t1). ``start``: earning balance at t0; ``events``: (time, account,
    balance after) in chain order within the window. Balances are integers in wei, so the result is exact."""
    bal = {a: int(b) for a, b in start.items() if int(b) > 0}
    since = {a: t0 for a in bal}
    out: dict[str, int] = {}
    for t, a, after in events:
        t = min(max(int(t), t0), t1)
        if a in bal:
            out[a] = out.get(a, 0) + bal[a] * (t - since[a])
        after = int(after)
        if after > 0:
            bal[a], since[a] = after, t
        else:
            bal.pop(a, None); since.pop(a, None)
    for a, b in bal.items():
        out[a] = out.get(a, 0) + b * (t1 - since[a])
    return {a: w for a, w in out.items() if w > 0}


def allocate(amount: int, w: dict[str, int]) -> dict[str, int]:
    """Floor(amount * w_i / W) per account; the remainder (< one lamport per account) stays in the pot."""
    total = sum(w.values())
    if amount <= 0 or total <= 0:
        return {}
    out = {a: (int(amount) * wi) // total for a, wi in sorted(w.items())}
    return {a: v for a, v in out.items() if v > 0}


def plan(realized_: int, allocated_before: int, payouts: int, native: int, gas_lamports: int) -> dict:
    pot = max(0, int(realized_) - int(allocated_before))
    reserved = int(allocated_before) - int(payouts)
    liquid = int(native) - reserved - int(gas_lamports)
    return {"pot": pot, "reserved": reserved, "liquid_cap": liquid, "amount": max(0, min(pot, liquid))}


# ---------------------------------------------------------------- wallet snapshot
def token_account_lamports(accounts: list[dict], open_mints: set[str]) -> int:
    """K: wSOL (its lamports are SOL) plus the rent of empty token accounts, which the hourly close returns. Accounts
    holding tokens are left out: an open position's cost (rent included) is in C, a dead bag's counts once swept."""
    k = 0
    for a in accounts:
        if a.get("is_native") or a.get("mint") == config.WSOL_MINT:
            k += int(a.get("lamports") or 0)
        elif int(a.get("amount") or 0) == 0 and a.get("mint") not in open_mints:
            k += int(a.get("lamports") or 0)
    return k


def open_positions(conn) -> tuple[int, set[str]]:
    rows = conn.execute("SELECT mint, cost_sol FROM positions WHERE book = %s AND status = 'open'", (config.VAULT_BOOK,)).fetchall()
    return int(round(sum(float(r["cost_sol"] or 0.0) for r in rows) * LAMPORTS)), {r["mint"] for r in rows}


def last_own_slot(conn) -> int:
    r = conn.execute("SELECT GREATEST((SELECT max(slot) FROM fills WHERE book = 'live'), "
                     "(SELECT max(slot) FROM vault_flows WHERE kind IN ('claim', 'withdrawal', 'sweep')), "
                     "(SELECT max((detail->>'slot')::bigint) FROM wallet_events WHERE kind = 'ata_closed' AND detail ? 'slot')) AS s").fetchone()
    return int(r["s"] or 0)


def flow_totals(conn, through_slot: int) -> dict:
    r = conn.execute(
        "SELECT COALESCE(sum(lamports) FILTER (WHERE kind = 'deposit'), 0) AS d, "
        "COALESCE(sum(lamports + fee_lamports) FILTER (WHERE kind = 'withdrawal'), 0) AS wd, "
        "COALESCE(sum(lamports) FILTER (WHERE kind = 'claim'), 0) AS p, "
        "COALESCE(sum(lamports) FILTER (WHERE kind = 'profit'), 0) AS gifts "
        "FROM vault_flows WHERE slot <= %s", (int(through_slot),)).fetchone()
    return {"deposits": int(r["d"]), "withdrawals": int(r["wd"]), "payouts": int(r["p"]), "gifts": int(r["gifts"])}


def snapshot(rpc, wallet: str, wait_s: float = FINALITY_WAIT_S, payout_wallet: str | None = None) -> dict:
    """The wallet at a finalized slot S that already contains every transaction our own code sent, read while no
    swap, claim or withdrawal can run, plus every flow up to S."""
    from . import flows, walletlock
    deadline = time.monotonic() + wait_s
    while True:
        with transaction() as conn:
            last = last_own_slot(conn)
        fin = int(rpc.call("getSlot", [{"commitment": "finalized"}]))
        if fin >= last:
            with walletlock.exclusive(timeout_s=wait_s):
                with transaction() as conn:
                    if last_own_slot(conn) > fin:
                        continue                                  # a fill landed while we waited for the lock
                    cost, mints = open_positions(conn)
                native, slot = rpc.get_balance_ctx(wallet, commitment="finalized", min_context_slot=max(last, 1))
                if payout_wallet:                         # the payout wallet is part of the same book (vault/payout.py)
                    native += rpc.get_balance_ctx(payout_wallet, commitment="finalized", min_context_slot=max(last, 1))[0]
                accts = rpc.get_token_accounts_by_owner(wallet, commitment="finalized")
            break
        if time.monotonic() > deadline:
            raise TimeoutError(f"finalized slot {fin} still behind our last transaction {last}")
        time.sleep(2.0)
    k = token_account_lamports(accts, mints)
    while flows.scanned_through() < slot:
        if time.monotonic() > deadline + wait_s:
            raise TimeoutError(f"flow scan has not reached slot {slot}")
        flows.scan(rpc, wallet)
        if flows.scanned_through() < slot:
            time.sleep(3.0)
    with transaction() as conn:
        f = flow_totals(conn, slot)
    return {"slot": slot, "at": int(time.time()), "native": native, "token_acct": k, "open_cost": cost, **f}


# ---------------------------------------------------------------- the weekly job
def _ts(t: int) -> datetime:
    return datetime.fromtimestamp(int(t), tz=timezone.utc)


def allocated_total(conn) -> int:
    return int(conn.execute("SELECT COALESCE(sum(allocated), 0) AS a FROM vault_settlements WHERE status = 'allocated'").fetchone()["a"])


def due(now: float | None = None) -> tuple[int, int] | None:
    """(T0, T1) to settle next: T1 is the newest period end after the vault started; T0 is where the last settlement
    ended (the first settlement covers the one period before T1). Normally that is one period. If periods were missed
    (worker down, snapshot timeouts), it is one window over all of them: the profit is cumulative, so weighting it over
    the whole window is exact, whereas settling only the newest period would hand the missed weeks' profit to that
    period's holders."""
    now = time.time() if now is None else now
    start = int(state.get("vault_started_at") or 0)
    t1 = period_end_at_or_before(now)
    if not start or t1 <= start:
        return None
    with transaction() as conn:
        last = conn.execute("SELECT max(period_end) AS e FROM vault_settlements").fetchone()["e"]
    t0 = int(last.timestamp()) if last is not None else t1 - int(config.VAULT_PERIOD_S)
    return (t0, t1) if t0 < t1 else None


def paper() -> bool:
    return config.VAULT_BOOK != "live"


def paper_snapshot() -> dict:
    """A paper book has no wallet: R is its closed-trade profit since the vault started (fees are inside each trade),
    the latest wealth mark stands in for the wallet, and nothing flows in or out."""
    start = int(state.get("vault_started_at") or 0)
    with transaction() as conn:
        r = conn.execute("SELECT COALESCE(sum(realized_sol), 0) AS s FROM positions WHERE book = %s AND status = 'closed' AND closed_at >= to_timestamp(%s)",
                         (config.VAULT_BOOK, start)).fetchone()
        m = conn.execute("SELECT sol_free FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (config.VAULT_BOOK,)).fetchone()
        cost, _ = open_positions(conn)
        paid = int(conn.execute("SELECT COALESCE(sum(lamports), 0) AS p FROM vault_flows WHERE kind = 'claim'").fetchone()["p"])
    realized_ = int(round(float(r["s"]) * LAMPORTS))
    native = int(round(float(m["sol_free"]) * LAMPORTS)) if m else 0
    # realized() adds back payouts; here R is already the book's profit, so present it as native-only terms
    return {"slot": 0, "at": int(time.time()), "native": native, "token_acct": 0, "open_cost": cost, "deposits": 0, "withdrawals": 0,
            "payouts": paid, "realized_override": realized_}


def take_snapshot(rpc, wallet: str, t0: int, t1: int, payout_wallet: str | None = None) -> int:
    snap = paper_snapshot() if paper() else snapshot(rpc, wallet, payout_wallet=payout_wallet)
    r = snap["realized_override"] if "realized_override" in snap else \
        realized(snap["native"], snap["token_acct"], snap["open_cost"], snap["deposits"], snap["withdrawals"], snap["payouts"])
    with transaction() as conn:
        sid = conn.execute(
            "INSERT INTO vault_settlements (period_start, period_end, status, snapshot_slot, snapshot_at, native, token_acct, open_cost, "
            "deposits, withdrawals, payouts, realized) VALUES (%s,%s,'snapshotted',%s,%s,%s,%s,%s,%s,%s,%s,%s) "
            "ON CONFLICT (period_end) DO NOTHING RETURNING id",
            (_ts(t0), _ts(t1), snap["slot"], _ts(snap["at"]), snap["native"], snap["token_acct"], snap["open_cost"],
             snap["deposits"], snap["withdrawals"], snap["payouts"], r)).fetchone()
        if sid is None:
            return 0
        conn.execute("UPDATE vault_flows SET settlement_id = %s WHERE settlement_id IS NULL AND slot <= %s", (sid["id"], snap["slot"]))
    log.info("settlement snapshot for %s: R %d lamports at slot %d", _ts(t1).isoformat(), r, snap["slot"])
    return int(sid["id"])


def event_inputs(conn, chain_id: int, t0: int, t1: int) -> tuple[dict[str, int], list[tuple[int, str, int]]]:
    start = {r["account"]: int(r["locked_after"]) for r in conn.execute(
        "SELECT DISTINCT ON (account) account, locked_after FROM vault_events WHERE chain_id = %s AND locked_after IS NOT NULL "
        "AND block_time < %s ORDER BY account, block_number DESC, log_index DESC", (chain_id, t0)).fetchall()}
    evs = [(int(r["block_time"]), r["account"], int(r["locked_after"])) for r in conn.execute(
        "SELECT block_time, account, locked_after FROM vault_events WHERE chain_id = %s AND locked_after IS NOT NULL "
        "AND block_time >= %s AND block_time < %s ORDER BY block_number, log_index", (chain_id, t0, t1)).fetchall()]
    return start, evs


def allocate_settlement(sid: int, index_state: dict) -> dict:
    """Phase 2 for snapshot ``sid``; ``index_state`` comes from rh_index (finalized through time, implementation)."""
    with transaction() as conn:
        s = conn.execute("SELECT * FROM vault_settlements WHERE id = %s FOR UPDATE", (sid,)).fetchone()
        if s is None or s["status"] != "snapshotted":
            return {"skipped": True}
        t0, t1 = int(s["period_start"].timestamp()), int(s["period_end"].timestamp())
        if int(index_state.get("through_time") or 0) < t1:
            return {"waiting": "robinhood chain has not finalized the period end yet"}
        impl, codehash = index_state.get("impl"), index_state.get("impl_codehash")
        allowed = {h.lower() for h in config.VAULT_IMPL_CODEHASHES}
        if allowed and (codehash or "").lower() not in allowed:
            state.halt(f"FlyVault implementation {impl} has an unapproved code hash {codehash}", {"impl": impl, "codehash": codehash})
            return {"halted": True}
        start, evs = event_inputs(conn, config.RH_CHAIN_ID, t0, t1)
        w = weights(start, evs, t0, t1)
        before = allocated_total(conn)
        paid = int(conn.execute("SELECT COALESCE(sum(lamports), 0) AS p FROM vault_flows WHERE kind = 'claim' AND slot <= %s",
                                (s["snapshot_slot"],)).fetchone()["p"])
        p = plan(int(s["realized"]), before, paid, int(s["native"]), int(round(config.GAS_RESERVE_SOL * LAMPORTS)))
        alloc = allocate(p["amount"], w)
        total = sum(alloc.values())
        inputs = {"period": [t0, t1], "snapshot": {k: int(s[k]) for k in ("snapshot_slot", "native", "token_acct", "open_cost",
                                                                             "deposits", "withdrawals", "payouts", "realized")},
                  "allocated_before": before, "plan": p, "weights": {a: str(v) for a, v in sorted(w.items())},
                  "allocations": dict(sorted(alloc.items())), "rh_block_end": index_state.get("through_block"), "impl": impl}
        digest = hashlib.sha256(json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for a, lam in alloc.items():
            conn.execute("INSERT INTO vault_allocations (settlement_id, evm, weight, lamports) VALUES (%s,%s,%s,%s)", (sid, a, w[a], lam))
        conn.execute(
            "UPDATE vault_settlements SET status = 'allocated', allocated_before = %s, pot = %s, liquid_cap = %s, allocated = %s, "
            "carried = %s, total_weight = %s, earners = %s, rh_block_end = %s, impl = %s, impl_codehash = %s, inputs_sha256 = %s, "
            "allocated_at = now() WHERE id = %s",
            (before, p["pot"], p["liquid_cap"], total, p["pot"] - total, sum(w.values()), len(alloc),
             index_state.get("through_block"), impl, codehash, digest, sid))
        state.put(f"settlement_inputs:{sid}", inputs, conn=conn)
    record_event("info", "vault", "settlement allocated", {"id": sid, "pot": p["pot"], "allocated": total, "earners": len(alloc)})
    return {"allocated": total, "pot": p["pot"], "earners": len(alloc), "carried": p["pot"] - total, "sha256": digest}


def run_once(rpc, wallet: str, index_state: dict, now: float | None = None, payout_wallet: str | None = None) -> dict:
    """Advance settlement by at most one step per phase; safe to call every minute."""
    if state.halted():
        return {"halted": True}
    out: dict = {}
    d = due(now)
    if d:
        sid = take_snapshot(rpc, wallet, *d, payout_wallet=payout_wallet)
        out["snapshot"] = sid
    with transaction() as conn:
        pend = conn.execute("SELECT id FROM vault_settlements WHERE status = 'snapshotted' ORDER BY period_end").fetchall()
    for r in pend:
        res = allocate_settlement(int(r["id"]), index_state)
        out[f"allocate:{r['id']}"] = res
        if "allocated" in res:
            from . import alerts
            alerts.send(f"settlement #{r['id']}: {res['allocated'] / LAMPORTS:.4f} SOL to {res['earners']} holders "
                        f"(pot {res['pot'] / LAMPORTS:.4f}, carried {res['carried'] / LAMPORTS:.4f})")
        else:
            break                                                 # settle in period order
    return out
