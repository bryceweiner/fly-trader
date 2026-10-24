from datetime import datetime, timezone

from fly_trader import config
from fly_trader.chain.balances import Snapshot, compute_delta, snapshot_balances, tx_hop_deposits

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


def test_the_rent_held_by_each_mints_token_accounts_is_a_deposit_wrapped_sol_is_not():
    class Rpc:
        def get_balance(self, pk):
            return 42

        def get_token_accounts_by_owner(self, pk):
            return [{"address": "x", "mint": "M", "amount": 5, "decimals": 6, "lamports": 2_039_280},
                    {"address": "w", "mint": "So11111111111111111111111111111111111111112", "amount": 9, "decimals": 9, "lamports": 9_002_039_280, "is_native": True}]

    snap = snapshot_balances(Rpc(), "owner")
    assert snap.deposits == {"M": 2_039_280} and snap.to_json()["deposits"] == {"M": 2_039_280}


def test_a_transactions_hop_deposits_are_the_rent_it_put_into_the_owners_accounts_of_other_tokens():
    """The rent a multi-hop route put into the owner's accounts for its intermediate tokens, from the transaction: not
    the traded mint's account, not wrapped SOL, not another wallet's accounts."""
    def bal(i, mint, owner, amount="0"):
        return {"accountIndex": i, "mint": mint, "owner": owner, "uiTokenAmount": {"amount": amount}}
    tx = {"transaction": {"message": {"accountKeys": [{"pubkey": "ME"}, {"pubkey": "a1"}, {"pubkey": "a2"}, {"pubkey": "a3"}, {"pubkey": "a4"}]}},
          "meta": {"preBalances": [10**9, 0, 0, 2_039_280, 0], "postBalances": [10**9 - 53_000_000, 1_513_840, 1_488_440, 2_039_280, 2_039_280],
                   "preTokenBalances": [bal(3, "USDC", "SOMEONE", "9")],
                   "postTokenBalances": [bal(1, "TOKEN", "ME", "700"), bal(2, "USDC", "ME"), bal(3, "USDC", "SOMEONE", "9"),
                                         bal(4, config.WSOL_MINT, "ME")]}}
    assert tx_hop_deposits(tx, "ME", "TOKEN") == {"USDC": 1_488_440}
    assert tx_hop_deposits(tx, "ME", "USDC") == {"TOKEN": 1_513_840}
    assert tx_hop_deposits({"meta": {}}, "ME", "TOKEN") is None
