"""The console with two memecoin chains: the Robinhood Chain page follows the memecoin pages, and the pages that follow
the chosen chain (overview, trades, Robinhood Chain, safety) render without an exception on either chain."""
import ast
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

UI = Path(__file__).resolve().parents[1] / "fly_trader" / "ui"


def test_rh_page_is_a_memecoin_page():
    """The distribution's console has one flat page list (no Kalshi sections): the RH page follows the memecoin pages."""
    tree = ast.parse((UI / "app.py").read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pages" for t in node.targets):
            pages = [c.args[0].value for c in node.value.elts]
    assert pages.index("app_pages/rh.py") == pages.index("app_pages/data.py") + 1


@pytest.mark.parametrize("page,chain", [("rh.py", None), ("overview.py", "Solana"), ("overview.py", "Robinhood Chain"), ("trades.py", None), ("safety.py", None)])
def test_pages_render_on_both_chains(db_conn, monkeypatch, page, chain):
    from fly_trader import config
    monkeypatch.setattr(config, "RH_ENABLED", True)
    at = AppTest.from_file(str(UI / "app_pages" / page), default_timeout=60)
    if chain:
        at.session_state["chain"] = chain
    at.run()
    assert not at.exception, [e.value for e in at.exception]
    if page == "trades.py":
        at.selectbox(key="trades_book").set_value("paper_rh_fly") if "paper_rh_fly" in at.selectbox(key="trades_book").options else None
        at.run()
        assert not at.exception, [e.value for e in at.exception]
