"""The live RH book's accounting (book ``live_rh``, amounts in ETH): what a finished swap intent books, the wallet's
wealth, and the reconciler.

What an intent books (every figure measured, never assumed):
- ETH moved by a swap = the native balance delta around the intent (one transaction in flight per wallet, so the delta
  is that intent's) plus the gas its transactions paid (``rh_txs.fee_wei``); token amounts = balanceOf deltas, checked
  against the receipt's Transfer logs;
- a buy opens a position whose ``cost_sol`` (ETH) is the ETH spent + the gas of every transaction of the intent
  (approvals and reverted attempts included) + the basis of a consumed base lot; ``gas_q`` keeps the gas part;
- a buy that produced no tokens is booked as an aborted position (qty 0, closed at once): realized = −gas − base loss;
- a sale's proceeds are the ETH received − the gas of the exit intent; realized = proceeds − cost;
- two-leg routes (rh/exec.py fallback) pass through ``rh_base_lots``: an entry leg's base is a lot consumed by the
  meme buy; an exit leg's base is a lot valued at the mark when the position closes, and its liquidation books the
  difference onto that same position;
- every leg is an ``rh_legs`` row in raw units (``asset`` 'eth' = native ETH).

``wealth``: native ETH − reserved + open positions at the ETH close net of the Pons exit cost + open lots at their base
mark. ``reconcile`` compares the chain with the books: token balances vs open positions and lots; the native balance vs
the last mark plus every booked native flow since (legs, gas, recorded payouts/deposits). Tokens missing, or native ETH
leaving unexplained, pause entries on circuit 3 (unknown outbound also trips its kill switch).
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

from ..db.apilog import record_event
from ..execution import ledger
from ..market.exit_cost import PONS_LABEL, exit_cost_fraction
from ..markets import RH, RH_CIRCUIT
from . import abi

log = logging.getLogger(__name__)
BOOK = RH.live_book
NATIVE = "eth"                                                   # rh_legs / rh_base_lots asset name of native ETH
WEI = 10 ** 18
TRANSFER = abi.event_topic("Transfer(address,address,uint256)").lower()
DUST_ETH = 1e-6                                                  # reconciliation tolerance on native ETH


def _addr_topic(a: str) -> str:
    return "0x" + "0" * 24 + a.lower()[2:]


def transfer_delta(receipt: dict | None, token: str, owner: str) -> int:
    """Net ``token`` units into ``owner`` from a receipt's Transfer logs."""
    if not receipt:
        return 0
    t, me = token.lower(), _addr_topic(owner)
    n = 0
    for lg in receipt.get("logs") or []:
        tp = [x.lower() for x in lg.get("topics") or []]
        if lg.get("address", "").lower() != t or len(tp) < 3 or tp[0] != TRANSFER:
            continue
        v = int(lg.get("data") or "0x0", 16)
        n += v if tp[2] == me else 0
        n -= v if tp[1] == me else 0
    return n


def intent_gas_wei(conn, intent_id: int) -> int:
    r = conn.execute("SELECT COALESCE(sum(fee_wei), 0) AS g FROM rh_txs WHERE intent_id = %s AND fee_wei IS NOT NULL", (intent_id,)).fetchone()
    return int(r["g"])


def add_leg(conn, *, intent_id: int, tx_id: int | None, position_id: int | None, leg_no: int, asset_in: str, asset_out: str,
            amount_in: int, amount_out: int, eth_in: float | None, eth_out: float | None, verified_by: str = "balance_delta") -> None:
    conn.execute("INSERT INTO rh_legs (intent_id, tx_id, position_id, leg_no, asset_in, asset_out, amount_in_raw, amount_out_raw, eth_in, eth_out, verified_by) "
                 "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                 (intent_id, tx_id, position_id, leg_no, asset_in, asset_out, int(amount_in), int(amount_out), eth_in, eth_out, verified_by))


# ---- positions
def book_buy(conn, *, intent_id: int, token: str, pool: str | None, quote_asset: str, eth_spent_wei: int, tokens_raw: int, decision_id: int | None,
             ts: datetime, hold_s: float | None, strategy: str | None, lot_basis_eth: float = 0.0, decimals: int = 18) -> tuple[int, float]:
    """Book a finished buy intent. Returns (position id, realized) — realized is 0 for an opened position and −cost for an
    aborted one (no tokens)."""
    gas = intent_gas_wei(conn, intent_id)
    spent = eth_spent_wei / WEI + lot_basis_eth
    cost = spent + gas / WEI
    if tokens_raw <= 0:                                          # nothing bought: the gas and any base loss are the loss
        pid = ledger.open_position(conn, book=BOOK, mint=token, pool=pool, qty_raw=0, cost_sol=cost, entry_price=0.0, decision_id=decision_id,
                                   fees_sol=gas / WEI, ts=ts, hold_s=hold_s, strategy=strategy)
        conn.execute("UPDATE positions SET quote_asset = %s, gas_q = %s WHERE id = %s", (quote_asset, gas / WEI, pid))
        realized = ledger.close_position(conn, position_id=pid, exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=None, forced_kind="aborted", ts=ts)
        return pid, float(realized or -cost)
    price = spent / (tokens_raw / 10 ** decimals)
    pid = ledger.open_position(conn, book=BOOK, mint=token, pool=pool, qty_raw=tokens_raw, cost_sol=cost, entry_price=price, decision_id=decision_id,
                               fees_sol=gas / WEI, ts=ts, hold_s=hold_s, strategy=strategy)
    conn.execute("UPDATE positions SET quote_asset = %s, gas_q = %s WHERE id = %s", (quote_asset, gas / WEI, pid))
    conn.execute("UPDATE rh_intents SET position_id = %s WHERE id = %s", (pid, intent_id))
    conn.execute("UPDATE rh_legs SET position_id = %s WHERE intent_id = %s", (pid, intent_id))
    return pid, 0.0


def book_sell(conn, *, intent_id: int, position: dict, eth_in_wei: int, tokens_sold_raw: int, decision_id: int | None, ts: datetime,
              forced_kind: str | None = None, lot_value_eth: float = 0.0) -> float | None:
    """Book a finished sell intent on ``position``: proceeds = ETH in (+ the mark value of an exit lot) − the intent's gas.
    A partial fill shrinks the position pro rata. Returns this sale's realized P&L (None: position not open)."""
    gas = intent_gas_wei(conn, intent_id)
    proceeds = eth_in_wei / WEI + lot_value_eth - gas / WEI
    qty = int(position["qty"]); dec = int(position.get("decimals") or 18)
    conn.execute("UPDATE positions SET gas_q = gas_q + %s WHERE id = %s", (gas / WEI, int(position["id"])))
    conn.execute("UPDATE rh_intents SET position_id = %s WHERE id = %s", (int(position["id"]), intent_id))
    if tokens_sold_raw <= 0:                                     # nothing sold: only the gas is lost, charged to the position
        conn.execute("UPDATE positions SET cost_sol = cost_sol + %s, fees_sol = fees_sol + %s WHERE id = %s AND status = 'open'",
                     (gas / WEI, gas / WEI, int(position["id"])))
        return -gas / WEI
    px = (eth_in_wei / WEI + lot_value_eth) / (tokens_sold_raw / 10 ** dec)
    if tokens_sold_raw < qty:
        frac = tokens_sold_raw / qty
        return ledger.realize_partial(conn, position_id=int(position["id"]), qty_raw=tokens_sold_raw, cost_part=float(position["cost_sol"]) * frac,
                                      proceeds_sol=proceeds, fees_sol=gas / WEI, price=px, ts=ts)
    return ledger.close_position(conn, position_id=int(position["id"]), exit_price=px, proceeds_sol=proceeds, fees_sol=gas / WEI,
                                 decision_id=decision_id, forced_kind=forced_kind, ts=ts)


def charge_gas(conn, intent_id: int, position_id: int | None) -> float:
    """A failed intent's gas: onto its open position's cost when it has one (an exit that reverted), else the book's
    overhead as an aborted zero-quantity position (a failed entry). Returns the ETH charged."""
    gas = intent_gas_wei(conn, intent_id) / WEI
    if gas <= 0:
        return 0.0
    if position_id is not None:
        conn.execute("UPDATE positions SET cost_sol = cost_sol + %s, fees_sol = fees_sol + %s, gas_q = gas_q + %s WHERE id = %s",
                     (gas, gas, gas, position_id))
        return gas
    it = conn.execute("SELECT token, decision_id FROM rh_intents WHERE id = %s", (intent_id,)).fetchone()
    now = datetime.now(timezone.utc)
    pid = ledger.open_position(conn, book=BOOK, mint=it["token"] or "gas", pool=None, qty_raw=0, cost_sol=gas, entry_price=0.0, decision_id=None,
                               fees_sol=gas, ts=now)
    conn.execute("UPDATE positions SET gas_q = %s WHERE id = %s", (gas, pid))
    ledger.close_position(conn, position_id=pid, exit_price=0.0, proceeds_sol=0.0, fees_sol=0.0, decision_id=None, forced_kind="aborted", ts=now)
    return gas


# ---- base lots (two-leg fallback, stray balances)
def open_lot(conn, *, asset: str, qty_raw: int, basis_eth: float, source: str, position_id: int | None = None, intent_id: int | None = None) -> int:
    return int(conn.execute("INSERT INTO rh_base_lots (asset, qty_raw, basis_eth, position_id, intent_id, source) VALUES (%s,%s,%s,%s,%s,%s) RETURNING id",
                            (asset.lower(), int(qty_raw), float(basis_eth), position_id, intent_id, source)).fetchone()["id"])


def consume_lot(conn, lot_id: int, position_id: int | None) -> float:
    r = conn.execute("UPDATE rh_base_lots SET status = 'consumed', position_id = COALESCE(%s, position_id), closed_at = now() WHERE id = %s AND status = 'open' "
                     "RETURNING basis_eth", (position_id, lot_id)).fetchone()
    return float(r["basis_eth"]) if r else 0.0


def liquidate_lot(conn, lot_id: int, proceeds_eth: float, fee_eth: float) -> float:
    """Close a lot sold for ``proceeds_eth`` (after ``fee_eth`` gas). The difference to its basis lands on the position it
    came from (an exit lot) — so a two-leg sale books exactly what the two legs returned. Returns that difference."""
    r = conn.execute("UPDATE rh_base_lots SET status = 'liquidated', proceeds_eth = %s, fee_eth = %s, closed_at = now() WHERE id = %s AND status = 'open' "
                     "RETURNING basis_eth, position_id", (proceeds_eth, fee_eth, lot_id)).fetchone()
    if r is None:
        return 0.0
    diff = proceeds_eth - fee_eth - float(r["basis_eth"])
    if r["position_id"] is not None:
        conn.execute("UPDATE positions SET realized_sol = COALESCE(realized_sol, 0) + %s, gas_q = gas_q + %s WHERE id = %s", (diff, fee_eth, r["position_id"]))
    return diff


def open_lots(conn) -> list[dict]:
    return [dict(r) for r in conn.execute("SELECT * FROM rh_base_lots WHERE status = 'open' ORDER BY id").fetchall()]


# ---- marks
def base_mark(conn, asset: str, stale_after_s: float = 600.0) -> tuple[float | None, bool]:
    """(ETH per whole unit of ``asset``, stale). Native ETH is 1 and never stale."""
    if asset in (NATIVE, "0x0000000000000000000000000000000000000000"):
        return 1.0, False
    r = conn.execute("SELECT ts, price_eth, stale FROM rh_base_prices WHERE asset = %s AND price_eth IS NOT NULL ORDER BY ts DESC LIMIT 1",
                     (asset.lower(),)).fetchone()
    if r is None:
        return None, True
    age = (datetime.now(timezone.utc) - r["ts"]).total_seconds()
    return float(r["price_eth"]), bool(r["stale"]) or age > stale_after_s


def wealth(conn, native_wei: int, reserved_wei: int, prices: dict, resqs: dict, fees: dict) -> dict:
    """The live RH book's wealth in ETH (native − reserved + positions net of exit cost + lots at their base mark)."""
    gross_v = exit_cost = exposure = 0.0; n = 0
    for p in ledger.open_positions(conn, BOOK):
        px = prices.get(p["mint"]) or float(p.get("last_mark_price") or p["entry_price"] or 0.0)
        g = int(p["qty"]) / 10 ** int(p.get("decimals") or 18) * px
        gross_v += g; exposure += float(p["cost_sol"]); n += 1
        exit_cost += g * exit_cost_fraction(g, resqs.get(p["mint"]), None, PONS_LABEL, fees.get(p["mint"]))
    lots_v = 0.0
    for lot in open_lots(conn):
        dec = _decimals(conn, lot["asset"])
        mark, _ = base_mark(conn, lot["asset"])
        lots_v += int(lot["qty_raw"]) / 10 ** dec * (mark if mark is not None else 0.0)
    native = native_wei / WEI; reserved = reserved_wei / WEI
    return {"native": native, "reserved": reserved, "positions": gross_v - exit_cost, "exit_cost": exit_cost, "lots": lots_v, "exposure": exposure, "n_open": n,
            "wealth": native - reserved + gross_v - exit_cost + lots_v}


def _decimals(conn, asset: str) -> int:
    r = conn.execute("SELECT decimals FROM rh_assets WHERE asset = %s", (asset.lower(),)).fetchone()
    return int(r["decimals"]) if r else 18


# ---- reconciliation
def booked_native_since(conn, since: datetime) -> int:
    """Native wei the books say moved since ``since``: legs out of/into native ETH, all gas paid, recorded wallet flows."""
    legs = conn.execute("SELECT COALESCE(sum(CASE WHEN asset_out = %s THEN amount_out_raw ELSE 0 END), 0) - "
                        "COALESCE(sum(CASE WHEN asset_in = %s THEN amount_in_raw ELSE 0 END), 0) AS d FROM rh_legs WHERE ts > %s",
                        (NATIVE, NATIVE, since)).fetchone()["d"]
    gas = conn.execute("SELECT COALESCE(sum(fee_wei), 0) AS g FROM rh_txs WHERE fee_wei IS NOT NULL AND fee_at > %s", (since,)).fetchone()["g"]
    flows = conn.execute("SELECT COALESCE(sum(CASE WHEN direction = 'in' THEN wei ELSE -wei END), 0) AS f FROM rh_wallet_flows "
                         "WHERE ts > %s AND kind <> 'unknown_outbound' AND kind <> 'unexplained_in'", (since,)).fetchone()["f"]
    return int(legs) - int(gas) + int(flows)


def reconcile(conn, wallet, now: datetime | None = None) -> dict:
    """Chain vs books. Records a wallet mark; returns {ok, native_gap_eth, token_gaps}. Tokens short of the books or
    native ETH gone unexplained pause circuit 3 (unexplained outbound also trips its kill switch); an unexplained native
    increase is recorded as a deposit."""
    now = now or datetime.now(timezone.utc)
    before = now - timedelta(microseconds=1)                     # flows found now belong before this mark, never after it
    native = wallet.balance()
    gaps = []
    held: dict[str, int] = {}
    for p in ledger.open_positions(conn, BOOK):
        held[p["mint"].lower()] = held.get(p["mint"].lower(), 0) + int(p["qty"])
    for lot in open_lots(conn):
        if lot["asset"] != NATIVE:
            held[lot["asset"]] = held.get(lot["asset"], 0) + int(lot["qty_raw"])
    for token, want in held.items():
        have = wallet.balance_of(token)
        if have < want:
            gaps.append({"token": token, "books": want, "chain": have})
    last = conn.execute("SELECT ts, native_wei FROM rh_wallet_marks ORDER BY ts DESC LIMIT 1").fetchone()
    gap_eth = 0.0
    if last is None:                                             # the first check: what the wallet holds was funded
        conn.execute("INSERT INTO rh_wallet_flows (ts, direction, kind, wei, note) VALUES (%s, 'in', 'deposit', %s, 'reconciler: opening balance')", (before, native))
    elif last["native_wei"] is not None:
        expected = int(last["native_wei"]) + booked_native_since(conn, last["ts"])
        gap_eth = (native - expected) / WEI
        if gap_eth > DUST_ETH:
            conn.execute("INSERT INTO rh_wallet_flows (ts, direction, kind, wei, note) VALUES (%s, 'in', 'deposit', %s, 'reconciler: unexplained increase')",
                         (before, native - expected))
        elif gap_eth < -DUST_ETH:
            conn.execute("INSERT INTO rh_wallet_flows (ts, direction, kind, wei, note) VALUES (%s, 'out', 'unknown_outbound', %s, 'reconciler')", (before, expected - native))
    ok = not gaps and gap_eth >= -DUST_ETH
    conn.execute("INSERT INTO rh_wallet_marks (ts, native_wei, consistent) VALUES (%s, %s, %s) ON CONFLICT (ts) DO NOTHING", (now, native, ok))
    if not ok:
        detail = {"token_gaps": gaps, "native_gap_eth": gap_eth}
        conn.execute("UPDATE circuit_state SET entries_paused = true, updated_at = now() WHERE id = %s", (RH_CIRCUIT,))
        if gap_eth < -DUST_ETH:
            conn.execute("UPDATE circuit_state SET kill_switch = true, kill_reason = %s, updated_at = now() WHERE id = %s",
                         (f"{-gap_eth:.6f} ETH left the RH wallet unexplained", RH_CIRCUIT))
        conn.execute("INSERT INTO circuit_events (kind, detail) VALUES ('rh_reconcile', %s)", (json.dumps({**detail, "circuit": RH_CIRCUIT}, default=str),))
        record_event("error", "rh_live", "RH wallet does not match the books: entries paused", detail)
    return {"ok": ok, "native_gap_eth": gap_eth, "token_gaps": gaps, "native_eth": native / WEI}
