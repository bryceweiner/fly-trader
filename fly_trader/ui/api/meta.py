"""What this install has turned on: the menu and the chain picker read it."""
from __future__ import annotations

import os

from ... import config, markets
from .. import runtime
from ..hub import topic
from ..state import KALSHI_BOOKS, kalshi_enabled, market_info


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("fly-trader")
    except Exception:
        return "dev"


@topic("meta", every=30)
def meta(params: dict) -> dict:
    return {"pid": os.getpid(), "hosts_workers": runtime.HOSTS_WORKERS, "kalshi": kalshi_enabled(), "rh": bool(config.RH_ENABLED),
            "vault": bool(config.VAULT_ENABLED), "markets": [market_info(m) for m in markets.enabled()], "kalshi_books": KALSHI_BOOKS,
            "version": _version()}
