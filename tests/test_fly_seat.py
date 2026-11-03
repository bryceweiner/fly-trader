"""An untested fly never inherits a seat (2026-10-23): a bootstrap other than the one this chain's state was learned under
starts its own race and clears the chain's handover, and a seat recorded for another fly is cleared at start. Until then
a new bootstrap kept the old race and seat, so an install whose fly had won handed a new fly live money at once."""
from types import SimpleNamespace

from fly_trader import markets
from fly_trader.agent import fly_session, handover
from fly_trader.db.connection import transaction


def _book(boot_id, chain="sol", race=1_790_000_000.0):
    b = SimpleNamespace(boot_id=boot_id, chain=chain, market=markets.for_chain(chain), race_started_at=race, has_seat=True)
    b._new_race = lambda old, seat=None: fly_session.FlyBook._new_race(b, old, seat)
    return b


def _seat(chain, detail):
    with transaction() as conn:
        if detail is None:
            handover.clear(conn, chain)
        else:
            handover.record(conn, detail, chain)


def _held(chain):
    with transaction() as conn:
        return handover.state(conn, chain)


def test_a_seat_won_by_another_fly_is_cleared_and_the_new_fly_races(db_conn):
    try:
        for chain in ("sol", "rh"):
            _seat(chain, {"bootstrap": 157, "at": "2026-10-01"}); b = _book(561, chain)
            fly_session.FlyBook._own_seat(b)
            assert _held(chain) is None and b.race_started_at is None and b.has_seat is False
    finally:
        _seat("sol", None); _seat("rh", None)


def test_a_seat_won_by_this_fly_or_set_by_hand_stays(db_conn):
    try:
        _seat("sol", {"bootstrap": 561}); b = _book(561)
        fly_session.FlyBook._own_seat(b)
        assert _held("sol") == {"bootstrap": 561} and b.race_started_at == 1_790_000_000.0
        _seat("sol", {"why": "set by hand"}); b = _book(561)
        fly_session.FlyBook._own_seat(b)
        assert _held("sol") == {"why": "set by hand"} and b.race_started_at == 1_790_000_000.0
    finally:
        _seat("sol", None)


def test_a_new_bootstrap_starts_its_own_race_and_gives_the_seat_back(db_conn):
    try:
        _seat("sol", {"why": "set by hand"}); b = _book(561)
        fly_session.FlyBook._new_race(b, 157)                       # adopting #561 over a state learned under #157
        assert _held("sol") is None and b.race_started_at is None and b.has_seat is False
    finally:
        _seat("sol", None)
