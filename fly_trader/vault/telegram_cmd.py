"""Two Telegram commands for the operator, read by polling (outbound only: the server still accepts no connection).

A message counts only when it comes from ``TELEGRAM_CHAT_ID`` AND from the user ``TELEGRAM_ADMIN_USER_ID``; anything
else is ignored. A hijacked Telegram account can therefore at worst stop the fly and pull its float back to the
treasury -- never move SOL anywhere else, never resume it (resume is SSH-only).

  /status   halt and kill-switch state, the float, the treasury, what is owed, what L1/L2 have left, the release
  /panic    halt the vault (claims, settlement, entries), tell the signer to refuse buys, top-ups and claims, trip
            the kill switch with liquidation, and return the float to the treasury as positions sell
"""
from __future__ import annotations

import logging
import time

import httpx

from .. import config
from . import alerts, state

log = logging.getLogger(__name__)
HELP = "commands: /status, /panic (halts everything, sells, returns the float to the treasury; resume only over SSH)"


def _updates(offset: int) -> list[dict]:
    r = httpx.get(f"https://api.telegram.org/bot{config.TELEGRAM_BOT_TOKEN}/getUpdates",
                  params={"offset": offset, "timeout": 0, "allowed_updates": '["message"]'}, timeout=10)
    r.raise_for_status()
    return (r.json() or {}).get("result") or []


def authorised(msg: dict) -> bool:
    chat, sender = str((msg.get("chat") or {}).get("id")), str((msg.get("from") or {}).get("id"))
    return bool(config.TELEGRAM_CHAT_ID and config.TELEGRAM_ADMIN_USER_ID) and chat == str(config.TELEGRAM_CHAT_ID) \
        and sender == str(config.TELEGRAM_ADMIN_USER_ID)


def status_text(signer=None) -> str:
    from ..db.connection import transaction
    from . import nav, payout
    L = config.LAMPORTS_PER_SOL
    with transaction() as conn:
        owed = payout.owed_total(conn)
        k = conn.execute("SELECT kill_switch, entries_paused FROM circuit_state WHERE id = 1").fetchone() or {}
        rel = conn.execute("SELECT value FROM ui_settings WHERE key = 'release'").fetchone()
        last = conn.execute("SELECT wealth FROM vault_nav ORDER BY ts DESC LIMIT 1").fetchone()
    c = state.get("custody") or {}
    lim = lambda n: (f"{(c[n]['remaining']) / L:.3f}/{c[n]['amount'] / L:g} SOL per {c[n]['period'].lower()}" if c.get(n) else "missing")   # noqa: E731
    sig = ""
    if signer is not None:
        try:
            s = signer.call("status")
            sig = f"\nsigner: {'PANIC' if s.get('panic') else 'ok'}, buys 24 h {s.get('buys_24h_lamports', 0) / L:.3f} SOL"
        except Exception as e:
            sig = f"\nsigner: UNREACHABLE ({type(e).__name__})"
    h = state.halted()
    return (f"{'HALTED: ' + '; '.join(h.get('reasons', [])) if h else 'running'}; kill switch {'ON' if k.get('kill_switch') else 'off'}"
            f"{', entries paused' if k.get('entries_paused') else ''}\n"
            f"book {(int(last['wealth']) / L) if last else 0:.4f} SOL, treasury {payout.cached_balance() / L:.4f} SOL, owed {owed / L:.4f} SOL\n"
            f"L1 {lim('L1')}; L2 {lim('L2')}\n"
            f"release {((rel or {}).get('value') or {}).get('seq', '?')}{sig}")


def panic(signer, who: str = "telegram") -> str:
    from ..db.connection import transaction
    reason = f"PANIC from {who}"
    state.halt(reason, {"at": int(time.time())})
    state.put("panic", {"at": int(time.time()), "by": who})
    out = ["vault halted (claims, settlement, entries)"]
    try:
        signer.call("panic", reason=reason)
        out.append("signer refuses buys, top-ups and claims")
    except Exception as e:
        out.append(f"signer NOT reached ({type(e).__name__}): revoke L1 and L2 in the owner wallet now")
    with transaction() as conn:
        conn.execute("UPDATE circuit_state SET kill_switch = true, kill_reason = %s, updated_at = now() WHERE id = 1", (reason,))
    out.append("kill switch on: open positions are sold within a minute, then the float goes back to the treasury")
    return "; ".join(out) + ". Resume only over SSH: fly-trader vault resume, reset-circuit --kill, and the signer's clear-panic."


def poll(signer) -> int:
    """Handle new commands; returns how many were obeyed. Never raises into the worker loop's caller beyond its job."""
    if not (config.TELEGRAM_BOT_TOKEN and config.TELEGRAM_CHAT_ID):
        return 0
    offset = int(state.get("telegram_offset") or 0)
    n = 0
    for u in _updates(offset):
        offset = max(offset, int(u["update_id"]) + 1)
        state.put("telegram_offset", offset)                   # each update is handled at most once
        msg = u.get("message") or {}
        text = (msg.get("text") or "").strip().split("@")[0].lower()
        if not text.startswith("/"):
            continue
        if not authorised(msg):
            log.warning("telegram command %r from an unauthorised chat/user ignored", text[:20])
            continue
        if text == "/panic":
            alerts.send(panic(signer))
        elif text == "/status":
            alerts.send(status_text(signer))
        else:
            alerts.send(HELP)
            continue
        n += 1
    return n


def heartbeat(signer, now: float | None = None) -> bool:
    """The daily all-quiet message at 09:00 UTC (a silent day then means something is wrong)."""
    now = time.time() if now is None else now
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    if time.gmtime(now).tm_hour < 9 or state.get("heartbeat_day") == day:
        return False
    state.put("heartbeat_day", day)
    alerts.send("daily report\n" + status_text(signer))
    return True


def deadman_ping() -> None:
    """healthchecks.io-style ping: its own Telegram integration alerts when these stop (the box or this worker died)."""
    if config.HEALTHCHECK_URL:
        httpx.get(config.HEALTHCHECK_URL, timeout=10)
