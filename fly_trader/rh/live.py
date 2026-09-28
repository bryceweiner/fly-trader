"""The Robinhood Chain fly's live book (``live_rh``): mirrors the paper RH fly's minute on the ETH wallet once it holds
the RH seat (handover_rh) and signing is allowed (rh/guard.py). The executor and accounting live in rh/exec.py and
rh/accounting.py."""
from __future__ import annotations

from .. import config


def minute(book, ctx, st: dict) -> dict:
    """One minute of the live RH mirror for ``book`` (agent/fly_session.FlyBook on chain 'rh')."""
    missing = config.rh_live_prerequisites_missing()
    if missing:
        return {"stage": "live gated: " + ", ".join(missing)}
    if book.mirror is None:
        from .exec import RhLiveMirror
        book.mirror = RhLiveMirror(book.H)
    return book.mirror.minute(ctx, run_id=book.run_id, beat_id=st["beat_id"], entries=st["entries"])
