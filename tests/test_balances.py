from datetime import datetime, timezone

from fly_trader.chain.balances import Snapshot, compute_delta, snapshot_balances

TS = datetime(2026, 9, 12, tzinfo=timezone.utc)


def test_compute_delta_over_union_of_mints():
    pre = Snapshot(100, {"A": 5, "B": 7}, TS)
    post = Snapshot(90, {"A": 9, "C": 1}, TS)
    assert compute_delta(pre, post) == {"lamports_delta": -10, "token_deltas": {"A": 4, "B": -7, "C": 1}}
    assert compute_delta(pre, pre) == {"lamports_delta": 0, "token_deltas": {"A": 0, "B": 0}}


def test_snapshot_sums_accounts_per_mint_and_keeps_decimals():
    class Rpc:
        def get_balance(self, pk):
            return 42

        def get_token_accounts_by_owner(self, pk):
            return [{"address": "x", "mint": "M", "amount": 5, "decimals": 6, "program": "p1"},
                    {"address": "y", "mint": "M", "amount": 7, "decimals": 6, "program": "p2"},
                    {"address": "z", "mint": "N", "amount": 0, "decimals": 9, "program": "p1"}]

    snap = snapshot_balances(Rpc(), "owner")
    assert snap.lamports == 42 and snap.tokens == {"M": 12, "N": 0} and snap.decimals == {"M": 6, "N": 9}
    assert snap.ts.tzinfo is not None and snap.token("missing") == 0
    js = snap.to_json()
    assert js["lamports"] == 42 and js["tokens"]["M"] == 12 and js["decimals"]["N"] == 9 and js["ts"].endswith("+00:00")
