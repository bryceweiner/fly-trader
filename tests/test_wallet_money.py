"""A live book's money comes from its wallet, not its trades (ops/wallets.money): realized P&L is what the wallet holds
with open positions at cost, less the amount it started with, so it keeps what the trade ledger never sees (pump.fun's fee share to
holders on Solana, ETH lost outside a trade on Robinhood Chain). A withdrawal is money taken out, not a loss. The Overview and the Trades page show that figure for the live
book and the closed trades' own for a paper book; RH's sizing reasons speak ETH."""
from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from psycopg.types.json import Jsonb

from fly_trader import markets
from fly_trader.agent.sizing import size_position
from fly_trader.db.connection import transaction
from fly_trader.ops import wallets
from fly_trader.ui.api import overview, trades as trades_api
from tests.test_console_overview import MINT_A, MINT_B, _ok, _state, quiet

SOL = SimpleNamespace(chain="sol", live_book="test_money_sol")
RH = SimpleNamespace(chain="rh", live_book="test_money_rh")
OWNER = "TestMoneyOwner1111111111111111111111111111"
TABLE = [{"lo": 0.0, "n": 100, "mean": 0.1, "win": 0.6, "kelly": 0.4}]


@pytest.fixture
def ledger_rows():
    """Marks, positions, flows, withdrawals and lots of a test alone; every other row of those tables back afterwards."""
    with transaction() as c:
        flows = c.execute("DELETE FROM rh_wallet_flows RETURNING block, ts, direction, kind, wei, note").fetchall()
        events = c.execute("DELETE FROM wallet_events WHERE kind = 'withdrawal' RETURNING ts, kind, pubkey, detail").fetchall()
    lots = []
    yield lots
    books = [SOL.live_book, RH.live_book]
    with transaction() as c:
        c.execute("DELETE FROM wealth_marks WHERE book = ANY(%s)", (books,)); c.execute("DELETE FROM positions WHERE book = ANY(%s)", (books,))
        c.execute("DELETE FROM rh_wallet_flows"); c.execute("DELETE FROM wallet_events WHERE kind = 'withdrawal'")
        for i in lots:
            c.execute("DELETE FROM rh_base_lots WHERE id = %s", (i,))
        for f in flows:
            c.execute("INSERT INTO rh_wallet_flows (block, ts, direction, kind, wei, note) VALUES (%s, %s, %s, %s, %s, %s)", (f["block"], f["ts"], f["direction"], f["kind"], f["wei"], f["note"]))
        for e in events:
            c.execute("INSERT INTO wallet_events (ts, kind, pubkey, detail) VALUES (%s, %s, %s, %s)", (e["ts"], e["kind"], e["pubkey"], Jsonb(e["detail"])))


def _mark(book: str, beat: int, wealth: float, positions_value: float = 0.0, exposure: float = 0.0) -> None:
    with transaction() as c:
        c.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, wealth, exposure, n_open) VALUES (%s, %s, %s, %s, %s, %s, %s, 0)",
                  (beat, book, datetime(2100, 1, 1, 0, abs(beat) % 60, tzinfo=timezone.utc), wealth - positions_value, positions_value, wealth, exposure))


def _position(book: str, mint: str, status: str = "closed", realized: float | None = None, cost: float = 0.1, deposit: float | None = None) -> None:
    with transaction() as c:
        c.execute("INSERT INTO positions (book, mint, status, realized_sol, cost_sol, closed_at, deposit_sol) VALUES (%s, %s, %s, %s, %s, %s, %s)",
                  (book, mint, status, realized, cost, datetime.now(timezone.utc) if status == "closed" else None, deposit))


def _flow(direction: str, kind: str, eth: float) -> None:
    with transaction() as c:
        c.execute("INSERT INTO rh_wallet_flows (direction, kind, wei, note) VALUES (%s, %s, %s, 'test')", (direction, kind, int(round(eth * 1e18))))


def _withdrawal(sol: float, status: str) -> None:
    with transaction() as c:
        c.execute("INSERT INTO wallet_events (kind, pubkey, detail) VALUES ('withdrawal', %s, %s)", (OWNER, Jsonb({"sol": sol, "lamports": int(sol * 1e9), "status": status})))


def test_nothing_before_the_first_mark(ledger_rows):
    assert wallets.money(SOL) is None and wallets.money(RH) is None


def test_solana_realized_is_the_wallet_not_the_trades(ledger_rows):
    """The case that was misreported: 5.000011 SOL paid in, 5.031983 in the wallet, flat, trades +0.000792."""
    _mark(SOL.live_book, -91001, 5.000011); _mark(SOL.live_book, -91002, 5.031983)
    m = wallets.money(SOL)
    assert m["start"] == pytest.approx(5.000011) and m["realized"] == pytest.approx(0.031972)


def test_an_open_position_counts_at_cost(ledger_rows):
    """Its loss on paper is not realized, and the rent its token account holds is still the wallet's."""
    _mark(SOL.live_book, -91001, 5.0)
    _mark(SOL.live_book, -91002, 4.93, positions_value=0.0315, exposure=0.1)             # 0.1 SOL of coin worth 0.03, 0.0015 of rent: cash 4.8985
    _position(SOL.live_book, MINT_A, status="open", deposit=0.0015)
    assert wallets.money(SOL)["realized"] == pytest.approx(0.0)


def test_a_solana_withdrawal_is_not_a_loss(ledger_rows):
    _mark(SOL.live_book, -91001, 5.0); _mark(SOL.live_book, -91002, 4.2)
    _withdrawal(1.0, "confirmed"); _withdrawal(9.0, "rejected"); _withdrawal(9.0, "signed")
    m = wallets.money(SOL)
    assert m["start"] == pytest.approx(5.0) and m["realized"] == pytest.approx(0.2)


def test_rh_starts_from_its_first_mark_whatever_flowed_since(ledger_rows):
    """0.225 opened the wallet: the starting amount stays that, through 0.06 lost and a 0.0254 top-up."""
    _flow("in", "deposit", 0.225); _flow("out", "unknown_outbound", 0.06); _flow("in", "deposit", 0.0254)
    _mark(RH.live_book, -91001, 0.225); _mark(RH.live_book, -91002, 0.2098)
    m = wallets.money(RH)
    assert m["start"] == pytest.approx(0.225) and m["realized"] == pytest.approx(-0.0152)


def test_rh_withdrawals_and_payouts_are_not_losses(ledger_rows):
    _flow("in", "deposit", 0.2); _flow("out", "withdrawal", 0.05); _flow("out", "payout", 0.01)
    _mark(RH.live_book, -91001, 0.2); _mark(RH.live_book, -91002, 0.145)
    m = wallets.money(RH)
    assert m["start"] == pytest.approx(0.2) and m["realized"] == pytest.approx(0.005)


def test_rh_a_base_lot_is_held_at_its_basis(ledger_rows):
    """A sale that left a base token: the money is the wallet's until it is swapped back, at what it cost."""
    _flow("in", "deposit", 0.2)
    _mark(RH.live_book, -91001, 0.2); _mark(RH.live_book, -91002, 0.2, positions_value=0.03, exposure=0.0)
    with transaction() as c:
        cols = {r["column_name"]: r["is_nullable"] for r in c.execute("SELECT column_name, is_nullable FROM information_schema.columns WHERE table_name = 'rh_base_lots'")}
        vals = {"asset": "0xbase", "qty_raw": 1, "basis_eth": 0.03, "source": "stray", "status": "open"}
        assert set(vals) <= set(cols)
        ledger_rows.append(c.execute(f"INSERT INTO rh_base_lots ({', '.join(vals)}) VALUES ({', '.join(['%s'] * len(vals))}) RETURNING id", list(vals.values())).fetchone()["id"])
    assert wallets.money(RH)["realized"] == pytest.approx(0.0)


def _live_state(mk, monkeypatch):
    monkeypatch.setattr(overview, "system_state", lambda m: _state(market=mk, live_money=True))
    monkeypatch.setattr(overview, "narrative", lambda s: "words")


def test_overview_shows_the_wallet_for_the_live_book(console, monkeypatch):
    _live_state(markets.SOL, monkeypatch)
    monkeypatch.setattr(overview.wallets, "money", lambda mk: {"start": 5.000011, "realized": 0.031972})
    with quiet():
        d = _ok(console.topic("overview.summary", chain="sol"))
    assert d["live"] and d["capital"] == 5.000011 and d["pnl"] == pytest.approx(0.031972)
    assert d["wealth_help"].endswith("started at 5.00001 SOL.")
    assert not d["pnl_spark"] or d["pnl_spark"][-1] == pytest.approx(0.031972)                  # the line ends at the figure shown


def test_overview_keeps_a_paper_book_on_its_trades(console):
    with quiet():
        d = _ok(console.topic("overview.summary", chain="sol"))
    assert not d["live"] and d["capital"] == markets.SOL.capital()


def test_trades_page_shows_the_wallet_for_the_live_book_only(console, monkeypatch):
    monkeypatch.setattr(trades_api, "money", lambda mk: {"start": 5.0, "realized": 0.032})
    live = _ok(console.topic("trades.summary", chain="sol", book="live"))["metrics"]
    paper = _ok(console.topic("trades.summary", chain="sol", book="paper_selector"))["metrics"]
    assert live["realized"] == pytest.approx(0.032) and paper["realized"] == paper["pnl"]


def test_sizing_reasons_speak_the_chains_coin(monkeypatch):
    from fly_trader import config
    monkeypatch.setattr(config, "KELLY_FRACTION", 0.125)
    sol, why_sol = size_position(0.2, 0.1, TABLE, 5.0, 5.0, 100.0)
    rh, why_rh = size_position(0.2, 0.1, TABLE, 0.2, 0.2, 100.0, market=markets.RH)
    assert sol > 0 and "SOL" in why_sol and "ETH" not in why_sol
    assert rh > 0 and "ETH" in why_rh and "SOL" not in why_rh
    _, flat = size_position(0.2, 0.1, TABLE, 0.2, 0.2, 100.0, market=markets.RH, flat=True)
    assert "ETH" in flat and "SOL" not in flat
    _, fixed = size_position(0.2, 0.1, None, 0.2, 0.2, 100.0, market=markets.RH)
    assert "ETH" in fixed and "SOL" not in fixed
