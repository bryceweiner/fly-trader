"""Positions and fills bookkeeping per book (live, paper_free, paper_mirror).

Paper cash is derived, never stored: cash = CAPITAL_SOL + Σ realized_sol(closed) − Σ cost_sol(open).
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
    rows = conn.execute(
        """SELECT p.*, t.decimals, t.graduated_at, wp.program_label
           FROM positions p LEFT JOIN tokens t ON t.mint = p.mint
           LEFT JOIN watch_pools wp ON wp.pool = p.pool
           WHERE p.book = %s AND p.status = 'open' ORDER BY p.opened_at""",
        (book,),
    ).fetchall()
    return [dict(r) for r in rows]


def paper_cash(conn, book: str) -> float:
    r = conn.execute(
        """SELECT COALESCE(sum(realized_sol) FILTER (WHERE status = 'closed'), 0) AS realized,
                  COALESCE(sum(cost_sol) FILTER (WHERE status = 'open'), 0) AS open_cost
           FROM positions WHERE book = %s""",
        (book,),
    ).fetchone()
    return config.CAPITAL_SOL + float(r["realized"]) - float(r["open_cost"])


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


def open_position(conn, *, book: str, mint: str, pool: str | None, qty_raw: int, cost_sol: float,
                  entry_price: float, decision_id: int | None, fees_sol: float, ts: datetime | None = None) -> int:
    ts = ts or datetime.now(timezone.utc)
    existing = conn.execute("SELECT id, qty, cost_sol, entry_price FROM positions WHERE book=%s AND mint=%s AND status='open'",
                            (book, mint)).fetchone()
    if existing:  # add to an open position: average the entry
        q0, c0 = int(existing["qty"]), float(existing["cost_sol"])
        q1, c1 = q0 + qty_raw, c0 + cost_sol
        p1 = (float(existing["entry_price"] or entry_price) * q0 + entry_price * qty_raw) / max(q1, 1)
        conn.execute("UPDATE positions SET qty=%s, cost_sol=%s, entry_price=%s, fees_sol=fees_sol+%s, last_mark_price=%s, last_mark_ts=%s WHERE id=%s",
                     (q1, c1, p1, fees_sol, entry_price, ts, existing["id"]))
        return int(existing["id"])
    row = conn.execute(
        """INSERT INTO positions (book, mint, pool, opened_at, entry_decision_id, qty, cost_sol, entry_price, peak_price,
             last_mark_price, last_mark_ts, fees_sol, status, last_swap_ts)
           VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,'open',%s) RETURNING id""",
        (book, mint, pool, ts, decision_id, qty_raw, cost_sol, entry_price, entry_price, entry_price, ts, fees_sol, ts),
    ).fetchone()
    return int(row["id"])


def close_position(conn, *, position_id: int, exit_price: float, proceeds_sol: float, fees_sol: float,
                   decision_id: int | None, forced_kind: str | None, ts: datetime | None = None) -> float:
    ts = ts or datetime.now(timezone.utc)
    p = conn.execute("SELECT cost_sol FROM positions WHERE id=%s", (position_id,)).fetchone()
    realized = proceeds_sol - float(p["cost_sol"])
    conn.execute(
        """UPDATE positions SET status='closed', closed_at=%s, exit_decision_id=%s, exit_price=%s, realized_sol=%s,
             fees_sol=fees_sol+%s, forced_exit_kind=%s, last_mark_price=%s, last_mark_ts=%s WHERE id=%s""",
        (ts, decision_id, exit_price, realized, fees_sol, forced_kind, exit_price, ts, position_id),
    )
    return realized


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
    from ..chain import keys
    from ..chain.rpc import HttpSolanaRpc
    from ..db.connection import transaction
    rpc = HttpSolanaRpc()
    pub = keys.bot_pubkey()
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
