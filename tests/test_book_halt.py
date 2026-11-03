"""A paper race book's own kill switch: a drawdown halts that book only, never the other book or the live circuit."""
from fly_trader import config
from fly_trader.agent import rails
from fly_trader.db.connection import transaction


def test_book_drawdown_halts_only_that_book(db_conn, monkeypatch):
    monkeypatch.setattr(config, "KILL_SWITCH_DRAWDOWN", 0.30)
    with transaction() as conn:
        conn.execute("DELETE FROM book_state WHERE book IN ('paper_fly','paper_selector')")
        kill_before = rails.load_circuit(conn).kill_switch
        assert not rails.check_book_drawdown(conn, "paper_fly", 4.0, 5.0)              # -20 %: fine
        assert rails.check_book_drawdown(conn, "paper_fly", 3.4, 5.0)                  # -32 %: halted
        assert rails.check_book_drawdown(conn, "paper_fly", 5.0, 5.0)                  # stays halted until cleared
        assert not rails.book_halted(conn, "paper_selector")
        assert rails.load_circuit(conn).kill_switch == kill_before                      # the live circuit is untouched
    rails.clear_book_halt("paper_fly")
    with transaction() as conn:
        assert not rails.book_halted(conn, "paper_fly")
        conn.execute("DELETE FROM book_state WHERE book = 'paper_fly'")
