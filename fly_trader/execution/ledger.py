"""Positions and fills bookkeeping per book (live, paper_selector).

Paper cash is derived, never stored: cash = CAPITAL_SOL + Σ realized_sol(all) − Σ cost_sol(open).
realized_sol accumulates per position: each partial sale books proceeds − the cost it retires (cost_sol
shrinks by that part) and the close adds proceeds − the remaining cost, so an open position may carry
realized P&L from partial sales.
Live cash is the on-chain SOL balance. Every open/close writes positions rows; every fill writes a
fills row (verified_by = 'balance_delta' for live, 'model' for paper).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

from .. import config

log = logging.getLogger(__name__)


def open_positions(conn, book: str) -> list[dict]:
    """Open positions of ``book`` with what trading them needs; ``decision_ts`` is the entry decision's minute (its ``ts``,
    the minute end a paper position opens at): a live position's hold runs from it, not from its fill 10–20 s later."""
    rows = conn.execute(
        """SELECT p.*, COALESCE(t.decimals, CASE WHEN p.chain = 'rh' THEN 18 END) AS decimals, t.graduated_at,
                  COALESCE(wp.program_label, CASE WHEN p.chain = 'rh' THEN 'Pons v4' END) AS program_label,
                  COALESCE(p.hold_s, (d.detail->>'hold_s')::float) AS hold_s, COALESCE(p.strategy, d.detail->>'strategy') AS strategy,
                  d.ts AS decision_ts
           FROM positions p LEFT JOIN tokens t ON t.mint = p.mint
           LEFT JOIN watch_pools wp ON wp.pool = p.pool
           LEFT JOIN decisions d ON d.id = p.entry_decision_id
           WHERE p.book = %s AND p.status = 'open' ORDER BY p.opened_at""",
        (book,),
    ).fetchall()
    return [dict(r) for r in rows]


def paper_cash(conn, book: str) -> float:
    r = conn.execute(
        """SELECT COALESCE(sum(realized_sol), 0) AS realized,
                  COALESCE(sum(cost_sol) FILTER (WHERE status = 'open'), 0) AS open_cost
           FROM positions WHERE book = %s""",
        (book,),
    ).fetchone()
    from ..markets import for_book
    return for_book(book).capital() + float(r["realized"]) - float(r["open_cost"])        # the book's own market's capital (SOL / ETH)


def record_fill(conn, *, order_id: int | None, book: str, mint: str, side: str, token_delta: int,
                sol_delta_lamports: int, price_sol: float, fee_lamports: int, verified_by: str,
                signature: str | None = None, slot: int | None = None, ts: datetime | None = None,
                pre: dict | None = None, post: dict | None = None) -> int:
    row = conn.execute(
        """INSERT INTO fills (order_id, book, ts, signature, slot, mint, side, token_delta, sol_delta_lamports,
             price_sol, fee_lamports, verified_by, pre_snapshot, post_snapshot)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
        (order_id, book, ts or datetime.now(timezone.utc), signature, slot, mint, side, token_delta, sol_delta_lamports,
         price_sol, fee_lamports, verified_by, json.dumps(pre) if pre else None, json.dumps(post) if post else None),
    ).fetchone()
    return int(row["id"])


def _chain_of(book: str) -> str:
    from ..markets import for_book
    return for_book(book).chain


def open_position(conn, *, book: str, mint: str, pool: str | None, qty_raw: int, cost_sol: float,
                  entry_price: float, decision_id: int | None, fees_sol: float, ts: datetime | None = None,
                  hold_s: float | None = None, strategy: str | None = None, deposit_sol: float = 0.0) -> int:
    """``deposit_sol``: what the buy put into the mint's token account (its rent, a first live buy), kept apart from
    ``cost_sol``: the account's close returns it (``refund_deposits``)."""
    ts = ts or datetime.now(timezone.utc)
    existing = conn.execute("SELECT id, qty, cost_sol, entry_price FROM positions WHERE book=%s AND mint=%s AND status='open'",
                            (book, mint)).fetchone()
    if existing:  # add to an open position: average the entry
        q0, c0 = int(existing["qty"]), float(existing["cost_sol"])
        q1, c1 = q0 + qty_raw, c0 + cost_sol
        p1 = (float(existing["entry_price"] or entry_price) * q0 + entry_price * qty_raw) / max(q1, 1)
        conn.execute("UPDATE positions SET qty=%s, cost_sol=%s, entry_price=%s, fees_sol=fees_sol+%s, last_mark_price=%s, last_mark_ts=%s, "
                     "deposit_sol = NULLIF(COALESCE(deposit_sol, 0) + %s, 0) WHERE id=%s",
                     (q1, c1, p1, fees_sol, entry_price, ts, deposit_sol, existing["id"]))
        return int(existing["id"])
    row = conn.execute(
        """INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, peak_price,
             last_mark_price, last_mark_ts, fees_sol, status, last_swap_ts, hold_s, strategy, chain, deposit_sol)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'open',%s,%s,%s,%s,%s) RETURNING id""",
        (book, mint, pool, ts, decision_id, qty_raw, cost_sol, entry_price, entry_price, entry_price, ts, fees_sol, ts, hold_s, strategy, _chain_of(book),
         deposit_sol or None),
    ).fetchone()
    return int(row["id"])


def refund_deposits(conn, book: str, mint: str) -> float:
    """The mint's token-account deposits not yet returned (paid by the buy that opened the account, whichever position
    of the mint closes it), marked returned; their total in SOL. 0 when none was booked apart (a position bought before
    deposits were: its cost holds the rent, so the whole refund is its own)."""
    rows = conn.execute("UPDATE positions SET deposit_refunded = true WHERE book = %s AND mint = %s AND COALESCE(deposit_sol, 0) > 0 AND NOT deposit_refunded "
                        "RETURNING deposit_sol", (book, mint)).fetchall()
    return sum(float(r["deposit_sol"]) for r in rows)


def deposits_held(conn, book: str) -> float:
    """SOL the book's token accounts hold as deposits (their rent) until their close returns it: part of its wealth."""
    r = conn.execute("SELECT COALESCE(sum(deposit_sol), 0) AS s FROM positions WHERE book = %s AND COALESCE(deposit_sol, 0) > 0 AND NOT deposit_refunded",
                     (book,)).fetchone()
    return float(r["s"])


def realize_partial(conn, *, position_id: int, qty_raw: int, cost_part: float, proceeds_sol: float, fees_sol: float,
                    price: float, ts: datetime | None = None) -> float | None:
    """Shrink an open position by qty_raw / cost_part and book proceeds − cost_part into realized_sol.
    Returns this sale's realized P&L, or None when the position is no longer open (nothing booked)."""
    ts = ts or datetime.now(timezone.utc)
    realized = proceeds_sol - cost_part
    r = conn.execute(
        """UPDATE positions SET qty = qty - %s, cost_sol = cost_sol - %s, realized_sol = COALESCE(realized_sol, 0) + %s,
             fees_sol = fees_sol + %s, last_mark_price=%s, last_mark_ts=%s WHERE id=%s AND status='open' RETURNING id""",
        (qty_raw, cost_part, realized, fees_sol, price, ts, position_id)).fetchone()
    return realized if r else None


def close_position(conn, *, position_id: int, exit_price: float, proceeds_sol: float, fees_sol: float,
                   decision_id: int | None, forced_kind: str | None, ts: datetime | None = None) -> float | None:
    """Close an open position; realized_sol accumulates (earlier partial sales + proceeds − remaining cost).
    Returns this sale's realized P&L, or None when the position was not open (already closed: nothing booked)."""
    ts = ts or datetime.now(timezone.utc)
    r = conn.execute(
        """UPDATE positions SET status='closed', closed_at=%s, exit_decision_id=%s, exit_price=%s,
             realized_sol=COALESCE(realized_sol, 0) + %s - cost_sol, fees_sol=fees_sol+%s, forced_exit_kind=%s,
             last_mark_price=%s, last_mark_ts=%s WHERE id=%s AND status='open' RETURNING %s - cost_sol AS realized""",
        (ts, decision_id, exit_price, proceeds_sol, fees_sol, forced_kind, exit_price, ts, position_id, proceeds_sol),
    ).fetchone()
    if r is None:
        log.warning("close_position: position %s is not open; nothing booked", position_id)
        return None
    return float(r["realized"])


def credit(conn, position_id: int, sol: float) -> None:
    """Book ``sol`` that reached the wallet after a position closed (its token account's rent refunded, a dead-bag's tokens
    swept) onto that position's realized P&L."""
    conn.execute("UPDATE positions SET realized_sol = COALESCE(realized_sol, 0) + %s WHERE id = %s", (sol, position_id))


def last_dead_bag(conn, book: str, mint: str) -> int | None:
    """The most recent position of ``mint`` written off as a dead-bag in ``book`` (its later sweeps' proceeds are its)."""
    r = conn.execute("SELECT id FROM positions WHERE book = %s AND mint = %s AND forced_exit_kind = 'dead_bag' ORDER BY closed_at DESC NULLS LAST, id DESC LIMIT 1",
                     (book, mint)).fetchone()
    return int(r["id"]) if r else None


def mark_positions(conn, book: str, prices: dict[str, float], last_swaps: dict[str, float], ts: datetime,
                   dt_s: float = 0.0) -> None:
    """Mark to the last price; track the peak; accrue satiety = Σ max(unrealized return, 0) × dt while in profit."""
    rows = conn.execute("SELECT id, mint, peak_price, entry_price, satiety FROM positions WHERE book=%s AND status='open'", (book,)).fetchall()
    for r in rows:
        p = prices.get(r["mint"])
        if p is None:
            continue
        peak = max(float(r["peak_price"] or 0.0), p)
        entry = float(r["entry_price"] or 0.0)
        u = (p / entry - 1.0) if entry > 0 else 0.0
        sat = float(r["satiety"] or 0.0) + max(u, 0.0) * dt_s
        ls = last_swaps.get(r["mint"])
        conn.execute("UPDATE positions SET last_mark_price=%s, last_mark_ts=%s, peak_price=%s, satiety=%s, "
                     "last_swap_ts=COALESCE(%s, last_swap_ts) WHERE id=%s",
                     (p, ts, peak, sat, datetime.fromtimestamp(ls, tz=timezone.utc) if ls else None, r["id"]))


def verify_fills() -> None:
    """Reconcile live positions against on-chain token balances; prints mismatches, writes wallet_events."""
    from ..chain.rpc import HttpSolanaRpc
    from ..db.connection import transaction
    from ..signer import client
    rpc = HttpSolanaRpc()
    pub = client.get().call("pubkeys")["trading"]
    accounts = rpc.get_token_accounts_by_owner(pub)
    onchain = {a["mint"]: int(a["amount"]) for a in accounts}
    with transaction() as conn:
        pos = open_positions(conn, "live")
        mismatches = []
        for p in pos:
            have = onchain.get(p["mint"], 0)
            if abs(have - int(p["qty"])) > max(1, int(p["qty"]) // 1000):
                mismatches.append({"mint": p["mint"], "db_qty": int(p["qty"]), "onchain": have})
        stray = [m for m, a in onchain.items() if a > 0 and m not in {p["mint"] for p in pos} and m != config.WSOL_MINT]
        conn.execute("INSERT INTO wallet_events (kind, pubkey, detail) VALUES ('verify_fills', %s, %s)",
                     (pub, json.dumps({"mismatches": mismatches, "stray_mints": stray, "n_positions": len(pos)})))
    print(json.dumps({"positions": len(pos), "mismatches": mismatches, "stray_mints": stray}, indent=1))
