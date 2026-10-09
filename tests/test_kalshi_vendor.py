"""The vendored Kalshi stack is better_bot's code line for line, minus the sports sections: a fresh run of the vendoring
script reproduces the files exactly (skipped where better_bot is not checked out), and the fee model is unchanged."""
import importlib.util
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "tools" / "vendor_kalshi.py"


def _script():
    spec = importlib.util.spec_from_file_location("vendor_kalshi", SCRIPT); mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def test_vendored_files_match_a_fresh_strip():
    mod = _script()
    if not mod.DEFAULT_SOURCE.exists():
        pytest.skip("better_bot source not available")
    for name, text in mod.render(mod.DEFAULT_SOURCE).items():
        assert (mod.DEST / name).read_text() == text, name


def test_sports_symbols_are_gone_and_fee_model_is_intact():
    from fly_trader.kalshi.vendor import kalshi_client as kc, book_sizing, venue_kalshi
    for name in ("LEAGUE_TO_KALSHI_SERIES", "match_event_for_matchup", "get_market_odds_sync", "sync_kalshi_team_abbrevs_sync", "kalshi_league_token"):
        assert not hasattr(kc, name), name
    assert "team_matching" not in sys.modules
    assert kc.kalshi_fee_rate_cents(50) == pytest.approx(1.75) and kc.kalshi_fee_rate_cents(90) == pytest.approx(0.63)
    assert kc.effective_price_cents(90) == pytest.approx(90.63)
    assert kc.kalshi_trading_fee_cents(50, 100) == 175
    book = book_sizing.parse_kalshi_orderbook({"orderbook_fp": {"yes_dollars": [["0.40", "10"]], "no_dollars": [["0.55", "20"]]}}, "T")
    assert book.best_ask("yes") == 45 and book.best_ask("no") == 60
    assert venue_kalshi.KalshiVenue().payout_cents(3) == 300
