"""Shim for better_bot's gate register: fly-trader has no capability gates; the client is always allowed to run."""
from __future__ import annotations


def can_use(*_args, **_kwargs) -> bool:
    return True
