"""Vendor better_bot's Kalshi client stack into ``fly_trader/kalshi/vendor/`` verbatim, minus the sports sections.

    .venv/bin/python tools/vendor_kalshi.py [--source /Users/bryce/Documents/code/better_bot/code] [--check]

Every file is copied line for line; the only edits are the removal of whole top-level (or class-level) definitions named
in ``DROP`` — the sports-league matching, abbreviation sync and per-event odds lookups that need better_bot's ESPN tables
— together with the comment block directly above each and the ``team_matching`` import. Relative imports stay as they
are: the shims beside the copies (``config``, ``gate_register``, ``database``, ``db_connection``, ``prediction_utils``)
resolve them. ``--check`` exits 1 when the vendored files differ from a fresh run (tests/test_kalshi_vendor.py).
"""
from __future__ import annotations

import argparse
import ast
import sys
from pathlib import Path

DEFAULT_SOURCE = Path("/Users/bryce/Documents/code/better_bot/code")
DEST = Path(__file__).resolve().parents[1] / "fly_trader" / "kalshi" / "vendor"

FILES = ("kalshi_client.py", "book_sizing.py", "venue_register.py", "venue_kalshi.py", "kalshi_stream.py")
DROP: dict[str, set[str]] = {
    "kalshi_client.py": {
        "_INDIVIDUAL_SPORT_SLUGS", "LEAGUE_TO_KALSHI_SERIES", "_league_series_cache", "series_for_league",
        "ADDITIONAL_GAME_SERIES", "series_tickers_for", "_NAME_SUFFIXES", "_KALSHI_TEAM_ALIASES", "_expand_alias",
        "_surname", "_title_contains_team", "_MarketProxy", "_EventProxy", "_events_cache", "_CACHE_TTL", "_EMPTY_ODDS",
        "_fetch_sports_events", "_fetch_one_series", "clear_events_cache", "_match_market_to_winner",
        "_event_ticker_date", "match_event_for_matchup", "_get_market_odds", "get_market_odds_for_event",
        "get_market_odds_sync", "kalshi_league_token", "_resolve_title_part", "_sync_kalshi_team_abbrevs",
        "sync_kalshi_team_abbrevs_sync", "_get_league_events", "get_league_events_sync",
    },
    "venue_kalshi.py": {"series_for_league", "series_for_slug", "tradable_leagues", "market_for_event", "cached_odds"},
}
DROP_IMPORTS = {"kalshi_client.py": {"team_matching"}}
HEADER = "# Vendored verbatim from better_bot code/{name} by tools/vendor_kalshi.py; sports sections removed. Do not edit.\n"


def _names(node: ast.AST) -> set[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return {node.name}
    if isinstance(node, ast.Assign):
        return {t.id for t in node.targets if isinstance(t, ast.Name)}
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return {node.target.id}
    return set()


def _span(node: ast.AST, lines: list[str]) -> tuple[int, int]:
    """1-based inclusive line span of ``node`` plus the comment block directly above it."""
    start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
    end = node.end_lineno
    while start > 1 and lines[start - 2].lstrip().startswith("#"):
        start -= 1
    return start, end


def strip(name: str, text: str) -> str:
    drop, drop_imports = DROP.get(name, set()), DROP_IMPORTS.get(name, set())
    lines = text.splitlines(keepends=True)
    tree = ast.parse(text)
    spans: list[tuple[int, int]] = []
    for node in tree.body:
        if _names(node) & drop:
            spans.append(_span(node, lines))
        elif isinstance(node, ast.ImportFrom) and node.module in drop_imports:
            spans.append((node.lineno, node.end_lineno))
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if _names(sub) & drop:
                    spans.append(_span(sub, lines))
    keep = [True] * len(lines)
    for a, b in spans:
        for i in range(a - 1, b):
            keep[i] = False
    out = "".join(l for l, k in zip(lines, keep) if k)
    while "\n\n\n\n" in out:
        out = out.replace("\n\n\n\n", "\n\n\n")
    ast.parse(out)                                 # the stripped module must still parse
    return HEADER.format(name=name) + out


def render(source: Path) -> dict[str, str]:
    return {name: strip(name, (source / name).read_text()) for name in FILES}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE); ap.add_argument("--check", action="store_true")
    a = ap.parse_args(argv)
    files = render(a.source)
    if a.check:
        bad = [n for n, t in files.items() if not (DEST / n).exists() or (DEST / n).read_text() != t]
        print("vendored files differ: " + ", ".join(bad) if bad else "vendored files match")
        return 1 if bad else 0
    DEST.mkdir(parents=True, exist_ok=True)
    for n, t in files.items():
        (DEST / n).write_text(t); print(f"wrote {DEST / n} ({len(t.splitlines())} lines)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
