"""PaperBroker: the same decisions executed at model prices.

Fill price = last tape price; cost = the real fees (pool fee by market cap, Jupiter 10 bps, network fee) + constant-product impact
of the position size against the pool's quote reserve (market/exit_cost.py). Deviation from the plan's
"first tape price ≥ 2 s after the decision" rule: fills are immediate at the last price; the
live-vs-mirror gap measures the difference.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import config
from ..market.exit_cost import exit_cost_fraction, fee_fraction, impact_fraction
from . import ledger

log = logging.getLogger(__name__)


@dataclass
class PaperFill:
    ok: bool
    position_id: int | None
    fill_id: int | None
    price: float
    sol_delta: float
    token_delta: int
    fee_sol: float
    reason: str = ""


class PaperBroker:
    def __init__(self, book: str):
        assert book == "paper_selector" or book.startswith("replay_"), book   # the selector's book; replay_* for tests
        self.book = book

    def buy(self, conn, *, decision_id: int | None, mint: str, pool: str | None, size_sol: float, price: float | None,
            res_quote_sol: float | None, mcap_sol: float | None, decimals: int, program_label: str | None, pool_fee: float | None = None,
            ts: datetime | None = None) -> PaperFill:
        """Fill at ``price`` pushed up by the constant-product impact, paying the real fees (market/exit_cost.py): the pool
        fee the stream reports (``pool_fee``) or the schedule's rate for ``mcap_sol``, Jupiter's platform fee, the network fee."""
        ts = ts or datetime.now(timezone.utc)
        if not price or price <= 0:
            return PaperFill(False, None, None, 0.0, 0.0, 0, 0.0, "no price")
        if res_quote_sol is None or res_quote_sol <= 0:
            return PaperFill(False, None, None, price, 0.0, 0, 0.0, "no liquidity estimate")
        fee_frac = fee_fraction(size_sol, mcap_sol, pool_fee)
        impact = impact_fraction(size_sol, res_quote_sol, program_label)
        if impact > config.EXECUTABILITY_MAX_IMPACT:
            # a router would not fill this either; refuse like the live executability check
            return PaperFill(False, None, None, price, 0.0, 0, 0.0, f"impact {impact:.2f} > {config.EXECUTABILITY_MAX_IMPACT}")
        eff_price = price * (1.0 + impact)  # buying pushes the price up against us
        tokens = size_sol * (1.0 - fee_frac) / eff_price
        qty_raw = int(tokens * (10 ** decimals))
        if qty_raw <= 0:
            return PaperFill(False, None, None, price, 0.0, 0, 0.0, "zero quantity")
        fee_sol = size_sol * fee_frac
        pid = ledger.open_position(conn, book=self.book, mint=mint, pool=pool, qty_raw=qty_raw, cost_sol=size_sol,
                                   entry_price=eff_price, decision_id=decision_id, fees_sol=fee_sol, ts=ts)
        fid = ledger.record_fill(conn, order_id=None, book=self.book, mint=mint, side="buy", token_delta=qty_raw,
                                 sol_delta_lamports=-int(size_sol * config.LAMPORTS_PER_SOL), price_sol=eff_price,
                                 fee_lamports=int(fee_sol * config.LAMPORTS_PER_SOL), verified_by="model", ts=ts)
        return PaperFill(True, pid, fid, eff_price, -size_sol, qty_raw, fee_sol)

    def sell(self, conn, *, position: dict, decision_id: int | None, price: float | None, res_quote_sol: float | None,
             mcap_sol: float | None, program_label: str | None, forced_kind: str | None, pool_fee: float | None = None,
             ts: datetime | None = None, fraction: float = 1.0) -> PaperFill:
        ts = ts or datetime.now(timezone.utc)
        price = price or float(position.get("last_mark_price") or position.get("entry_price") or 0.0)
        fraction = float(min(1.0, max(0.0, fraction)))
        qty_raw = int(int(position["qty"]) * fraction) if fraction < 0.999 else int(position["qty"])
        decimals = int(position.get("decimals") or 6)
        gross = qty_raw / (10 ** decimals) * price
        c = exit_cost_fraction(gross, res_quote_sol, mcap_sol, program_label, pool_fee)
        proceeds = gross * (1.0 - c)
        fee_sol = gross * fee_fraction(gross, mcap_sol, pool_fee)
        if fraction < 0.999:   # partial: shrink the position, realise the sold part
            realized = ledger.realize_partial(conn, position_id=int(position["id"]), qty_raw=qty_raw,
                                              cost_part=float(position["cost_sol"]) * fraction, proceeds_sol=proceeds,
                                              fees_sol=fee_sol, price=price, ts=ts)
            if realized is None:
                return PaperFill(False, int(position["id"]), None, price, 0.0, 0, 0.0, "position not open")
            fid = ledger.record_fill(conn, order_id=None, book=self.book, mint=position["mint"], side="sell", token_delta=-qty_raw,
                                     sol_delta_lamports=int(proceeds * config.LAMPORTS_PER_SOL), price_sol=price * (1.0 - c),
                                     fee_lamports=int(fee_sol * config.LAMPORTS_PER_SOL), verified_by="model", ts=ts)
            return PaperFill(True, int(position["id"]), fid, price * (1.0 - c), proceeds, -qty_raw, fee_sol, reason=f"partial {fraction:.2f} realized {realized:+.5f} SOL")
        realized = ledger.close_position(conn, position_id=int(position["id"]), exit_price=price * (1.0 - c),
                                         proceeds_sol=proceeds, fees_sol=fee_sol, decision_id=decision_id,
                                         forced_kind=forced_kind, ts=ts)
        if realized is None:
            return PaperFill(False, int(position["id"]), None, price, 0.0, 0, 0.0, "position not open")
        fid = ledger.record_fill(conn, order_id=None, book=self.book, mint=position["mint"], side="sell", token_delta=-qty_raw,
                                 sol_delta_lamports=int(proceeds * config.LAMPORTS_PER_SOL), price_sol=price * (1.0 - c),
                                 fee_lamports=int(fee_sol * config.LAMPORTS_PER_SOL), verified_by="model", ts=ts)
        return PaperFill(True, int(position["id"]), fid, price * (1.0 - c), proceeds, -qty_raw, fee_sol,
                         reason=f"realized {realized:+.5f} SOL")
