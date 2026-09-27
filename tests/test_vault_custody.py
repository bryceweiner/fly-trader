"""The treasury's flows and on-chain setup: owner withdrawals, stolen-key spends, and what custody.check reports."""
from fly_trader.vault import custody, flows

T, P, TREASURY, OWNER, HOLDER, FUND, THIEF = "Trading1", "Payout1", "Treasury1", "Owner1", "Holder1", "Fund1", "Thief1"
SYS = "11111111111111111111111111111111"


def tx(sig, keys, pre, post, err=None):
    return {"slot": 9, "blockTime": 1, "transaction": {"signatures": [sig], "message": {"accountKeys": [{"pubkey": k, "signer": s} for k, s in keys]}},
            "meta": {"err": err, "fee": 5000, "preBalances": pre, "postBalances": post}}


def test_treasury_flows():
    ours = {T, P}
    # an owner withdrawal from the treasury (the Squads app: the owner signs, the vault PDA pays)
    f = flows.classify(tx("w", [(OWNER, True), (TREASURY, False), (FUND, False)], [10, 50, 0], [5, 30, 20]), TREASURY, False, {FUND}, ours, {OWNER})
    assert (f.kind, f.direction, f.lamports, f.counterparty) == ("withdrawal", "out", 20, FUND)
    # SOL leaving the treasury with nobody we know signing: halt
    f = flows.classify(tx("x", [(THIEF, True), (TREASURY, False)], [10, 50], [40, 20]), TREASURY, False, set(), ours, {OWNER})
    assert f.kind == "anomaly"
    # a stolen payout key using L2 for a transaction our code never made: halt
    f = flows.classify(tx("y", [(T, False), (P, True), (TREASURY, False), (THIEF, False)], [9, 0, 50, 0], [9, 0, 30, 20]), TREASURY, False, set(), ours, {OWNER})
    assert f.kind == "unknown_outbound"
    # our own claim payment (fee payer trading, co-signed by payout) is known: nothing to record
    assert flows.classify(tx("z", [(T, True), (P, True), (TREASURY, False), (HOLDER, False)], [9, 0, 50, 0], [8, 0, 30, 20]), TREASURY, True, set(), ours, {OWNER}) is None
    # a deposit into the treasury from a funding address
    f = flows.classify(tx("d", [(FUND, True), (TREASURY, False)], [100, 0], [50, 50]), TREASURY, False, {FUND}, ours, {OWNER})
    assert (f.kind, f.lamports) == ("deposit", 50)


def reading(**over):
    c = {"multisig": "MS", "treasury": TREASURY, "threshold": 1, "time_lock": 0, "members": [OWNER], "config_authority": SYS,
         "L1": {"address": "L1", "multisig": "MS", "mint": SYS, "amount": 2 * 10**9, "period": "Day", "remaining": 2 * 10**9, "last_reset": 0,
                "members": [T], "destinations": [T]},
         "L2": {"address": "L2", "multisig": "MS", "mint": SYS, "amount": 10**9, "period": "Week", "remaining": 10**9, "last_reset": 0,
                "members": [P], "destinations": []}}
    c.update(over)
    return c


def test_custody_problems():
    assert custody.problems(reading(), T, P) == []
    assert any("MEMBER" in x for x in custody.problems(reading(members=[OWNER, T]), T, P))
    bad = reading(); bad["L1"] = {**bad["L1"], "destinations": []}
    assert any("ONLY destination" in x for x in custody.problems(bad, T, P))
    assert any("L2" in x and "missing" in x for x in custody.problems(reading(L2=None), T, P))
    assert any("config authority" in x for x in custody.problems(reading(config_authority=THIEF), T, P))


def test_custody_changes_are_reported():
    b = reading()
    n = reading(); n["L2"] = {**n["L2"], "amount": 2 * 10**9}
    assert custody.changes(b, n) == ["L2 amount: 1 SOL -> 2 SOL"]
    assert custody.changes(b, reading(L1=None)) == ["L1 removed"]
    assert custody.changes(b, reading(members=[OWNER, THIEF])) == [f"multisig members: {[OWNER]} -> {[OWNER, THIEF]}"]
    assert custody.changes(None, b) == []


# ------------------------------------------------------------------ the second RPC provider
class Check:
    def __init__(self, balances=None, tx=None):
        self.balances, self.tx = balances or {}, tx

    def get_balance_ctx(self, a, commitment="finalized", min_context_slot=None):
        return self.balances[a], min_context_slot

    def get_transaction(self, sig):
        return self.tx


def test_settlement_needs_both_providers_to_agree(monkeypatch):
    import pytest
    from fly_trader.chain import rpc as chain_rpc
    from fly_trader.vault import settle
    monkeypatch.setattr(settle.time, "sleep", lambda s: None)
    monkeypatch.setattr(chain_rpc, "check_rpc", lambda: Check({T: 7, TREASURY: 3}))
    settle.agree_with_check_rpc([T, TREASURY], 10, 99)                       # same book: fine
    with pytest.raises(settle.RpcDisagree):
        settle.agree_with_check_rpc([T, TREASURY], 11, 99)                   # the primary reports a lamport more
    monkeypatch.setattr(chain_rpc, "check_rpc", lambda: None)
    settle.agree_with_check_rpc([T], 11, 99)                                 # no second provider configured: as before


def test_halt_needs_the_second_provider_to_see_it_too(monkeypatch):
    from fly_trader.chain import rpc as chain_rpc
    halted = []
    monkeypatch.setattr(flows, "_halt", lambda f: halted.append(f))
    real = tx("x", [(THIEF, True), (TREASURY, False)], [10, 50], [40, 20])
    f = flows.classify(real, TREASURY, False, set(), {T, P}, {OWNER})
    monkeypatch.setattr(chain_rpc, "check_rpc", lambda: Check(tx=None))      # the check provider has no such transaction
    assert flows._confirmed_elsewhere(chain_rpc.check_rpc(), f, TREASURY, set(), {T, P}, {OWNER}) is False
    monkeypatch.setattr(chain_rpc, "check_rpc", lambda: Check(tx=real))
    assert flows._confirmed_elsewhere(chain_rpc.check_rpc(), f, TREASURY, set(), {T, P}, {OWNER}) is True
