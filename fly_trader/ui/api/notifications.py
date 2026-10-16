"""The notifications bell: what an operator should not miss — warnings and errors from the event log, the rails' circuit
events (kill switch and breaker trips, halts, pauses, resets) and every live-money fill — as one newest-first list."""
from __future__ import annotations

from ... import markets
from ...db.queries import q
from ..hub import topic
from ..state import kalshi_enabled, labels

N_EVENTS, N_CIRCUIT, N_FILLS = 50, 30, 30
CIRCUIT = {1: "Solana", 2: "Kalshi", 3: "Robinhood Chain"}


def _circuit_text(kind: str, d: dict) -> tuple[str, str]:
    """A circuit_events row as (level, sentence)."""
    c = CIRCUIT.get(d.get("circuit"), "Solana" if d.get("circuit") is None else f"circuit {d.get('circuit')}")
    book = d.get("book")
    if kind == "tripped":
        return "error", f"{c} circuit breaker tripped after {d.get('fail_count', '?')} failures: {d.get('reason') or 'no reason recorded'}"
    if kind == "kill_switch":
        return "error", f"{c} kill switch tripped: wealth {float(d['wealth']):.4f} vs peak {float(d['peak']):.4f}; no new entries" if d.get("wealth") is not None \
            else f"{c} kill switch tripped: no new entries"
    if kind == "book_halt":
        return "error", f"{book}: drawdown halt, entries blocked" + (f" (wealth {float(d['wealth']):.4f} vs peak {float(d['peak']):.4f})" if d.get("wealth") is not None else "")
    if kind == "book_resumed":
        return "info", f"{book}: drawdown halt cleared"
    if kind == "reset":
        return "info", f"{c} circuit reset" + (", kill switch cleared (peak re-based)" if d.get("kill") else "")
    if kind == "paused":
        return "warning", f"{c}: new entries paused by the operator"
    if kind == "resumed":
        return "info", f"{c}: new entries resumed"
    if kind == "withdrawal":
        return "info", f"{c}: withdrawal recorded; the kill switch's peak is re-based"
    if kind == "training_reset":
        return "info", f"Paper history reset ({d.get('reason') or 'no reason recorded'})"
    if kind == "gap_pause":
        return "warning", f"{c} live mirror paused: the live book drifted from its paper twin" + (f" ({d['arm']} arm)" if d.get("arm") else "")
    if kind == "unknown_fill_pause":
        return "error", f"Unknown live fill ({d.get('side') or '?'} {d.get('mint') or '?'}): live trading paused until checked"
    if kind == "rh_reconcile":
        return "warning", "Robinhood Chain books reconciled against the chain"
    return "info", f"{c}: {kind.replace('_', ' ')}"


@topic("notifications.recent", every=2)
def recent(params: dict) -> dict:
    items = []
    # warnings and errors; the rails' own (source 'rails') arrive as circuit events below, so they are not listed twice
    for r in q("""SELECT id, ts, level, source, message FROM events WHERE level IN ('warning', 'error') AND COALESCE(source, '') <> 'rails'
                  ORDER BY id DESC LIMIT %s""", (N_EVENTS,)):
        items.append({"id": f"e:{r['id']}", "ts": r["ts"], "level": r["level"], "source": r["source"] or "system", "text": r["message"]})
    for r in q("SELECT id, ts, kind, detail FROM circuit_events ORDER BY id DESC LIMIT %s", (N_CIRCUIT,)):
        level, text = _circuit_text(r["kind"], r["detail"] if isinstance(r["detail"], dict) else {})
        items.append({"id": f"c:{r['id']}", "ts": r["ts"], "level": level, "source": "rails", "text": text})
    fills = q("SELECT id, ts, book, side, mint, sol_delta_lamports, signature FROM fills WHERE book LIKE 'live%%' ORDER BY id DESC LIMIT %s", (N_FILLS,))
    names = labels([r["mint"] for r in fills])
    for r in fills:
        m = markets.for_book(r["book"]); amt = abs((r["sol_delta_lamports"] or 0) / 1e9)
        items.append({"id": f"f:{r['id']}", "ts": r["ts"], "level": "live", "source": r["book"], "chain": m.chain, "mint": r["mint"],
                      "text": f"Live {'buy' if r['side'] == 'buy' else 'sell'} on {m.name}: {names.get(r['mint'], r['mint'])} for {amt:,.4f} {m.unit}"})
    if kalshi_enabled():
        for r in q("SELECT trade_id, ts, book, ticker, side, action, price_cents, count FROM kalshi_fills ORDER BY ts DESC LIMIT %s", (N_FILLS,)):
            items.append({"id": f"k:{r['trade_id']}", "ts": r["ts"], "level": "live", "source": r["book"] or "live_kalshi", "ticker": r["ticker"],
                          "text": f"Live Kalshi fill: {r['action'] or 'buy'} {float(r['count'] or 0):,.0f} {str(r['side']).upper()} at {float(r['price_cents'] or 0):.0f}¢ on {r['ticker']}"})
    items.sort(key=lambda r: (r["ts"].timestamp(), r["id"]), reverse=True)
    return {"items": items}
