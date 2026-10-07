"""The console with two memecoin chains: the Robinhood Chain page is in the Memecoins section, and the pages that follow
the chosen chain (overview, trades, Robinhood Chain, safety) render without an exception on either chain."""
import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "fly_trader" / "ui"


def test_rh_page_is_a_memecoin_page():
    tree = ast.parse((UI / "app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pages" for t in node.targets):
            sections = {k.value: [c.args[0].value for c in v.elts] for k, v in zip(node.value.keys, node.value.values)}
    assert "app_pages/rh.py" in sections["Memecoins"]


@pytest.mark.parametrize("page,chain", [("rh.py", None), ("overview.py", "Solana"), ("overview.py", "Robinhood Chain"), ("trades.py", None), ("safety.py", None)])
def test_pages_render_on_both_chains(db_conn, monkeypatch, page, chain):
    import streamlit as st
    from fly_trader import config
    from fly_trader.ops import wallets
    monkeypatch.setattr(config, "RH_ENABLED", True)
    st.cache_data.clear()                                                   # the Safety page's wallet reads: no network in tests
    monkeypatch.setattr(wallets, "summary", lambda c: {"chain": c, "unit": "SOL" if c == "sol" else "ETH", "address": None, "native": None, "error": None,
                                                       "n_open": 0, "positions": 0.0, "marked_at": None, "price_usd": None, "price_at": None,
                                                       "native_usd": None, "positions_usd": None, "total_usd": None})
    at = AppTest.from_file(str(UI / "app_pages" / page), default_timeout=60)
    if chain:
        at.session_state["chain"] = chain
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "trades.py":
        at.selectbox(key="trades_book").set_value("paper_rh_fly") if "paper_rh_fly" in at.selectbox(key="trades_book").options else None
        at.run()
        assert not at.exception, [e.value for e in at.exception]


def test_trades_page_lists_each_chains_books(db_conn, monkeypatch):
    """The Trades page follows the chain: each chain's books are offered before their first trade, in every view."""
    from fly_trader import config, markets
    monkeypatch.setattr(config, "RH_ENABLED", True)
    for m in (markets.SOL, markets.RH):
        at = AppTest.from_file(str(UI / "app_pages" / "trades.py"), default_timeout=60)
        at.session_state["chain"] = m.name
        at.run()
        assert list(m.books) == at.selectbox(key="trades_book").options[:3] and at.selectbox(key="trades_book").value == m.selector_book
        for view in ("Decision log", "Fills"):
            at.segmented_control(key="trades_view").set_value(view).run()
            assert not at.exception, [e.value for e in at.exception]
