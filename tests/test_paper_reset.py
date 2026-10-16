"""Live money arriving: the paper race books start again from the wallet's funds (ops/reset.reset_paper_books) while what
the fly learned and the live books stay, and with LIVE_WITHOUT_HANDOVER the wallet trades without waiting for the seat."""
from types import SimpleNamespace

import pytest

from fly_trader import config
from fly_trader.agent import fly_session
from fly_trader.db.connection import transaction
from fly_trader.ops import reset

BOOKS = ("paper_selector", "paper_fly", "paper_rh_selector", "paper_rh_fly", "live", "paper_other")


@pytest.fixture
def seeded():
    with transaction() as conn:
        for t in ("positions", "wealth_marks", "fills", "orders", "book_state"):
            conn.execute(f"DELETE FROM {t} WHERE book = ANY(%s)", (list(BOOKS),))
        ids = {}
        for b in BOOKS:
            d = conn.execute("INSERT INTO decisions (ts, kind, mint, book_targets, detail) VALUES (now(), 'selector_enter', 'M', %s, %s) RETURNING id",
                             ([b], '{"book": "%s"}' % b)).fetchone()["id"]
            ids[b] = d
            conn.execute("INSERT INTO positions (book, mint, status, opened_at, cost_sol, qty, entry_price, entry_decision_id) VALUES (%s, 'M', 'open', now(), 0.1, 1, 0.1, %s)", (b, d))
            conn.execute("INSERT INTO wealth_marks (ts, book, wealth, beat_id) VALUES (now(), %s, 5.0, 0)", (b,))
            conn.execute("INSERT INTO book_state (book, halted, reason) VALUES (%s, true, 'test') ON CONFLICT (book) DO UPDATE SET halted = true", (b,))
        conn.execute("INSERT INTO fly_updates (hour, n, drift) VALUES (date_trunc('hour', now()) - interval '777 days', 1, 0.1) ON CONFLICT DO NOTHING")
    yield ids
    with transaction() as conn:
        for t in ("positions", "wealth_marks", "book_state"):
            conn.execute(f"DELETE FROM {t} WHERE book = ANY(%s)", (list(BOOKS),))
        conn.execute("DELETE FROM decisions WHERE id = ANY(%s)", (list(ids.values()),))
        conn.execute("DELETE FROM fly_updates WHERE hour = date_trunc('hour', now()) - interval '777 days'")


def _count(table, book):
    with transaction() as conn:
        return conn.execute(f"SELECT count(*) AS n FROM {table} WHERE book = %s", (book,)).fetchone()["n"]


def test_reset_paper_books_clears_only_the_paper_race_books(seeded, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path)
    r = reset.reset_paper_books(("sol", "rh"), reason="test")
    assert r["books"] == ["paper_selector", "paper_fly", "paper_rh_selector", "paper_rh_fly"] and (tmp_path / r["archived_to"].split("/")[-1] / "positions.csv").exists()
    for b in r["books"]:
        assert _count("positions", b) == 0 and _count("wealth_marks", b) == 0 and _count("book_state", b) == 0
    for b in ("live", "paper_other"):                                          # the live book and every other book stay
        assert _count("positions", b) == 1 and _count("wealth_marks", b) == 1
    with transaction() as conn:
        kept = {r["id"] for r in conn.execute("SELECT id FROM decisions WHERE id = ANY(%s)", (list(seeded.values()),)).fetchall()}
        learned = conn.execute("SELECT count(*) AS n FROM fly_updates WHERE hour = date_trunc('hour', now()) - interval '777 days'").fetchone()["n"]
    assert kept == {seeded["live"], seeded["paper_other"]} and learned == 1     # what the fly learned is untouched


def test_one_chain_alone(seeded, tmp_path, monkeypatch):
    monkeypatch.setattr(config, "PG_ARCHIVE_DIR", tmp_path)
    reset.reset_paper_books(("rh",), reason="test", archive=False)
    assert _count("positions", "paper_rh_fly") == 0 and _count("positions", "paper_fly") == 1


@pytest.mark.parametrize("live,flag,seat,trades", [(True, True, None, True), (True, False, None, False), (True, False, {"at": "x"}, True),
                                                    (False, True, {"at": "x"}, False)])
def test_the_wallet_trades_without_the_seat_when_asked(monkeypatch, live, flag, seat, trades):
    monkeypatch.setattr(config, "LIVE_WITHOUT_HANDOVER", flag)
    monkeypatch.setattr(fly_session.handover, "state", lambda conn, chain: seat)
    book = SimpleNamespace(live=live, chain="sol")
    assert fly_session.FlyBook.trades_live(book, None) is trades
