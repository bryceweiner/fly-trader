"""Operator commands (pause / resume / roll back) reach the running fly at its next minute; one queued while the engine was
down is dropped instead of landing whenever the engine next starts."""
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import pytest

from fly_trader.agent import fly_session as memecoin
from fly_trader.kalshi import fly_session as kalshi


class _Conn:
    def __init__(self, value):
        self.value, self.deleted = value, False

    def execute(self, sql, params=None):
        if sql.startswith("DELETE"):
            self.deleted = True
        return SimpleNamespace(fetchone=lambda: {"value": self.value} if self.value is not None and sql.startswith("SELECT") else None)


@pytest.mark.parametrize("mod", [memecoin, kalshi])
@pytest.mark.parametrize("age_s,applied", [(30, True), (memecoin.COMMAND_MAX_AGE_S + 60, False)])
def test_a_stale_command_is_dropped(mod, age_s, applied, monkeypatch):
    monkeypatch.setattr(mod, "record_event", lambda *a, **k: None)
    at = (datetime.now(timezone.utc) - timedelta(seconds=age_s)).isoformat()
    conn = _Conn(json.dumps({"cmd": "pause", "at": at}))
    book = SimpleNamespace(learning_frozen=False, brain=None)
    mod_cls = memecoin.FlyBook if mod is memecoin else kalshi.KalshiFlyBook
    mod_cls._commands(book, conn, 0.0)
    assert conn.deleted and book.learning_frozen is applied


def test_resume_lifts_every_freeze(monkeypatch):
    """The rollback limit freezes one strategy's learning (FlyBook.frozen); the operator's resume lifts that too."""
    monkeypatch.setattr(memecoin, "record_event", lambda *a, **k: None)
    conn = _Conn(json.dumps({"cmd": "resume", "at": datetime.now(timezone.utc).isoformat()}))
    book = SimpleNamespace(learning_frozen=True, frozen=["ev"], rollbacks={"ev": [1.0, 2.0, 3.0]}, brain=None)
    memecoin.FlyBook._commands(book, conn, 0.0)
    assert book.learning_frozen is False and book.frozen == [] and book.rollbacks == {}


def test_a_frozen_strategy_learns_nothing_and_the_others_keep_learning():
    import numpy as np
    import torch
    from fly_trader.brain.plastic import Tags
    calls = []
    bank = SimpleNamespace(update=lambda tags, r, now, pathway: calls.append(("update", list(tags.s), r.tolist())) or {"ok": True},
                           decay_to=lambda now: calls.append(("decay",)))
    me = SimpleNamespace(names=["ev", "capitulation"], frozen=["ev"], learning_frozen=False, bank=bank, brain=None)
    tags = Tags([("a",), ("b",), ("c",)], np.zeros(3), torch.zeros(3), torch.zeros(3, 2), torch.zeros(3, 4), torch.ones(1, 3), np.array([0, 1, 0]))
    assert memecoin.FlyBook._teach(me, tags, np.array([0.1, 0.2, 0.3]), 1.0, "outcome") == {"ok": True}
    assert calls == [("update", [1], [pytest.approx(0.2)])]                                  # capitulation's row only
    calls.clear()
    assert memecoin.FlyBook._teach(me, tags.subset(np.array([True, False, True])), np.array([0.1, 0.3]), 2.0, "outcome") is None
    assert calls == [("decay",)]                                                             # ev alone: its memory decays, learns nothing
    calls.clear(); me.learning_frozen = True
    assert memecoin.FlyBook._teach(me, tags, np.array([0.1, 0.2, 0.3]), 3.0, "outcome") is None and calls == [("decay",)]


def test_another_chains_book_follows_the_memorys_owner_and_leaves_it_the_commands(monkeypatch):
    """The RH book teaches the Solana book's memory: that book's freezes hold for it, and only that book takes commands
    (2026-10-18: the RH book's minute took a resume first, and the frozen Solana book never saw it)."""
    import numpy as np
    import torch
    from fly_trader.brain.plastic import Tags
    monkeypatch.setattr(memecoin, "record_event", lambda *a, **k: None)
    owner = SimpleNamespace(learning_frozen=True, frozen=[])
    conn = _Conn(json.dumps({"cmd": "resume", "at": datetime.now(timezone.utc).isoformat()}))
    calls = []
    rh = SimpleNamespace(brain=owner, learning_frozen=False, frozen=[], rollbacks={}, names=["ev", "capitulation"],
                         bank=SimpleNamespace(update=lambda *a, **k: calls.append("update"), decay_to=lambda now: calls.append("decay")))
    memecoin.FlyBook._commands(rh, conn, 0.0)
    assert not conn.deleted and owner.learning_frozen is True                               # left for the Solana book
    tags = Tags([("a",)], np.zeros(1), torch.zeros(1), torch.zeros(1, 2), torch.zeros(1, 4), torch.ones(1, 1), np.array([1]))
    assert memecoin.FlyBook._teach(rh, tags, np.array([0.1]), 1.0, "outcome") is None and calls == ["decay"]   # its owner is paused
    owner.learning_frozen, owner.frozen = False, ["capitulation"]; calls.clear()
    assert memecoin.FlyBook._teach(rh, tags, np.array([0.1]), 2.0, "outcome") is None and calls == ["decay"]   # and its freezes hold

