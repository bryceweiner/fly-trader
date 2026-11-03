"""The signer: what it signs and what it refuses (fly_trader/signer), offline with a fake RPC."""
import base64
import json
import struct
from pathlib import Path

import pytest
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.signature import Signature
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from fly_trader.signer import client, core, policy
from fly_trader.signer.ledger import Ledger
from fly_trader.signer.policy import JUPITER_V6, NATIVE_MINT, TOKEN, PolicyError, Snapshot

FIX = json.loads((Path(__file__).parent / "fixtures" / "ultra_orders.json").read_text())
TRADING = Keypair.from_seed(bytes([7]) * 32)
ME = TRADING.pubkey()
MEME = Keypair.from_seed(bytes([10]) * 32).pubkey()
OTHER = Keypair.from_seed(bytes([11]) * 32).pubkey()
BH = "11111111111111111111111111111111"


def token_data(mint: Pubkey, owner: Pubkey, amount: int) -> bytes:
    return bytes(mint) + bytes(owner) + struct.pack("<Q", amount) + bytes(165 - 72)


def mint_data(decimals: int = 6) -> bytes:
    return bytes(44) + bytes([decimals]) + bytes(37)


class FakeRpc:
    def __init__(self):
        self.acc: dict[Pubkey, dict] = {}
        self.post: dict[Pubkey, dict | None] = {}
        self.sim_err = None
        self.prices = {}

    def account(self, k, commitment="confirmed"):
        return self.acc.get(k)

    def accounts(self, keys, commitment="confirmed"):
        return [self.acc.get(k) for k in keys]

    def token_accounts(self, owner, program):
        return [k for k, a in self.acc.items() if a["owner"] == program and bytes(a["data"][32:64]) == bytes(owner)]

    def blockhash(self):
        return BH, 150

    def simulate(self, tx, addresses):
        out = []
        for a in addresses:
            v = self.post[a] if a in self.post else self.acc.get(a)
            out.append(None if v is None else {"lamports": v["lamports"], "owner": str(v["owner"]), "data": [base64.b64encode(v["data"]).decode(), "base64"]})
        return {"err": self.sim_err, "logs": [], "accounts": out}

    def usd_prices(self, mints):
        return {m: self.prices[m] for m in mints if m in self.prices}


def signer(rpc=None, **cfg) -> core.Signer:
    return core.Signer(TRADING, rpc or FakeRpc(), core.SignerConfig(**cfg), Ledger())


def swap_tx(extra: list[Instruction] | None = None, price_micro: int = 1000, route_accounts=None) -> str:
    wsol = policy.ata(ME, NATIVE_MINT)
    ixs = [Instruction(policy.COMPUTE_BUDGET, bytes([2]) + struct.pack("<I", 300_000), []),
           Instruction(policy.COMPUTE_BUDGET, bytes([3]) + struct.pack("<Q", price_micro), []),
           transfer(TransferParams(from_pubkey=ME, to_pubkey=wsol, lamports=50_000_000)),
           Instruction(TOKEN, bytes([17]), [AccountMeta(wsol, False, True)]),
           Instruction(JUPITER_V6, bytes(8), route_accounts or [AccountMeta(ME, True, True), AccountMeta(wsol, False, True)]),
           Instruction(TOKEN, bytes([9]), [AccountMeta(wsol, False, True), AccountMeta(ME, False, True), AccountMeta(ME, True, False)]),
           *(extra or [])]
    msg = MessageV0.try_compile(ME, ixs, [], Hash.from_string(BH))
    return base64.b64encode(bytes(VersionedTransaction.populate(msg, [Signature.default()] * msg.header.num_required_signatures))).decode()


def wallet(rpc: FakeRpc, lamports=3 * 10**9, meme=0):
    rpc.acc[ME] = {"lamports": lamports, "owner": policy.SYSTEM, "data": b""}
    rpc.acc[MEME] = {"lamports": 1_461_600, "owner": TOKEN, "data": mint_data(6)}
    rpc.acc[NATIVE_MINT] = {"lamports": 1_000_000, "owner": TOKEN, "data": mint_data(9)}
    if meme:
        rpc.acc[policy.ata(ME, MEME)] = {"lamports": 2_039_280, "owner": TOKEN, "data": token_data(MEME, ME, meme)}


def after_buy(rpc: FakeRpc, spent=50_005_000, got=1_000_000, to=None):
    """Post-state of a buy: the wallet paid ``spent`` (incl. the new ATA's rent) and ``got`` tokens arrived at ``to``."""
    rpc.post[ME] = {**rpc.acc[ME], "lamports": rpc.acc[ME]["lamports"] - spent - (2_039_280 if to is None else 0)}
    dst = to or policy.ata(ME, MEME)
    rpc.post[dst] = {"lamports": 2_039_280, "owner": TOKEN, "data": token_data(MEME, ME if to is None else OTHER, got)}


# ------------------------------------------------------------------ static rules
def test_real_ultra_orders_pass_the_static_rules():
    tables = {k: [Pubkey.from_string(a) for a in v] for k, v in FIX["lookup_tables"].items()}
    for o in FIX["orders"]:
        tx = VersionedTransaction.from_bytes(base64.b64decode(o["transaction"]))
        chk = policy.check_swap_static(tx, Pubkey.from_string(o["taker"]), lambda k: tables[str(k)], 5_000_000)
        assert chk.signer_index == 0


@pytest.mark.parametrize("bad, why", [
    ([transfer(TransferParams(from_pubkey=ME, to_pubkey=OTHER, lamports=1))], "System"),
    ([Instruction(TOKEN, bytes([3]) + struct.pack("<Q", 5), [AccountMeta(policy.ata(ME, MEME), False, True), AccountMeta(OTHER, False, True), AccountMeta(ME, True, False)])], "token instruction 3"),
    ([Instruction(TOKEN, bytes([4]) + struct.pack("<Q", 5), [AccountMeta(policy.ata(ME, MEME), False, True), AccountMeta(OTHER, False, False), AccountMeta(ME, True, False)])], "token instruction 4"),
    ([Instruction(TOKEN, bytes([9]), [AccountMeta(policy.ata(ME, MEME), False, True), AccountMeta(OTHER, False, True), AccountMeta(ME, True, False)])], "close"),
    ([Instruction(policy.MEMO, b"hi", [])], "not one a swap may call"),
    ([Instruction(policy.ATA, bytes([1]), [AccountMeta(ME, True, True), AccountMeta(OTHER, False, True), AccountMeta(OTHER, False, False)])], "another wallet"),
])
def test_static_rules_refuse(bad, why):
    tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx(bad)))
    with pytest.raises(PolicyError, match=why):
        policy.check_swap_static(tx, ME, lambda k: [], 5_000_000)


def test_priority_fee_cap_and_signer_role():
    tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx(price_micro=10**9)))
    with pytest.raises(PolicyError, match="priority fee"):
        policy.check_swap_static(tx, ME, lambda k: [], 5_000_000)
    tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx()))
    with pytest.raises(PolicyError, match="does not sign"):
        policy.check_swap_static(tx, OTHER, lambda k: [], 5_000_000)


# ------------------------------------------------------------------ effects
def snap(lamports, **tok):
    return Snapshot(lamports=lamports, tokens={k: v for k, v in tok.items()})


def test_effects_buy_and_sell():
    a = policy.ata(ME, MEME)
    pre = Snapshot(10**9, {}); post = Snapshot(10**9 - 50_000_000 - 2_039_280 - 5000, {a: (MEME, 700, 2_039_280)})
    assert policy.check_swap_effects(pre, post, in_mint=NATIVE_MINT, out_mint=MEME, in_amount=50_000_000, min_out=600, slack_lamports=20_000) == {"sol": -50_005_000, "token": 700}
    with pytest.raises(PolicyError, match="below the minimum"):                           # tokens went elsewhere
        policy.check_swap_effects(pre, Snapshot(post.lamports + 2_039_280, {}), in_mint=NATIVE_MINT, out_mint=MEME, in_amount=50_000_000, min_out=600, slack_lamports=20_000)
    with pytest.raises(PolicyError, match="spends"):
        policy.check_swap_effects(pre, Snapshot(post.lamports - 10**8, post.tokens), in_mint=NATIVE_MINT, out_mint=MEME, in_amount=50_000_000, min_out=600, slack_lamports=20_000)
    b = Keypair().pubkey(); other_mint = Keypair().pubkey()
    with pytest.raises(PolicyError, match="another holding"):
        policy.check_swap_effects(Snapshot(10**9, {b: (other_mint, 5, 2_039_280)}), Snapshot(post.lamports, {**post.tokens, b: (other_mint, 0, 2_039_280)}),
                                  in_mint=NATIVE_MINT, out_mint=MEME, in_amount=50_000_000, min_out=600, slack_lamports=20_000)
    pre = Snapshot(10**9, {a: (MEME, 700, 2_039_280)})
    post = Snapshot(10**9 + 40_000_000, {a: (MEME, 0, 2_039_280)})
    assert policy.check_swap_effects(pre, post, in_mint=MEME, out_mint=NATIVE_MINT, in_amount=700, min_out=39_000_000, slack_lamports=20_000)["sol"] == 40_000_000
    with pytest.raises(PolicyError, match="sells"):
        policy.check_swap_effects(pre, post, in_mint=MEME, out_mint=NATIVE_MINT, in_amount=500, min_out=39_000_000, slack_lamports=20_000)
    with pytest.raises(PolicyError, match="below the minimum"):
        policy.check_swap_effects(pre, Snapshot(10**9 + 10, post.tokens), in_mint=MEME, out_mint=NATIVE_MINT, in_amount=700, min_out=39_000_000, slack_lamports=20_000)


def test_value_floor():
    with pytest.raises(PolicyError, match="value"):
        policy.check_value("buy", -10**9, 80 * 10**6, 6, 1.0, 150.0, 0.6)              # 1 SOL ($150) for $80 of tokens
    assert policy.check_value("buy", -10**9, 140 * 10**6, 6, 1.0, 150.0, 0.6) == pytest.approx(140 / 150)
    assert policy.check_value("sell", 10**9, 100 * 10**6, 6, None, 150.0, 0.6) is None  # no price: the other checks stand


# ------------------------------------------------------------------ sign_swap end to end (fake RPC)
def test_sign_swap_signs_a_good_buy_and_refuses_diverted_output():
    rpc = FakeRpc(); wallet(rpc); after_buy(rpc)
    s = signer(rpc)
    out = s.sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    tx = VersionedTransaction.from_bytes(base64.b64decode(out["tx"]))
    assert tx.verify_with_results() == [True] and out["simulated"]["token"] == 1_000_000
    rpc = FakeRpc(); wallet(rpc); after_buy(rpc, to=Keypair().pubkey())                   # the route pays another wallet
    with pytest.raises(PolicyError, match="below the minimum"):
        signer(rpc).sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)


def test_a_multi_hop_buys_account_for_its_intermediate_token_is_the_wallets_not_spent():
    """A route through an intermediate token (SOL -> USDC -> the token) opens the wallet's account for it with an ATA
    instruction: its rent stays the wallet's (their close returns it), so the buy does not overspend (such buys were
    refused 2026-10-23..24). The same rent without the account opened in the transaction, or in an account there that is
    not the wallet's, is spent."""
    usdc = Keypair.from_seed(bytes([15]) * 32).pubkey(); hop = policy.ata(ME, usdc); rent = 1_488_440
    open_hop = Instruction(policy.ATA, bytes([1]), [AccountMeta(ME, True, True), AccountMeta(hop, False, True), AccountMeta(ME, False, False),
                                                    AccountMeta(usdc, False, False), AccountMeta(policy.SYSTEM, False, False), AccountMeta(TOKEN, False, False)])

    def chain(owner):
        rpc = FakeRpc(); wallet(rpc); after_buy(rpc)
        rpc.post[ME] = {**rpc.post[ME], "lamports": rpc.post[ME]["lamports"] - rent}
        rpc.post[hop] = {"lamports": rent, "owner": TOKEN, "data": token_data(usdc, owner, 0)}
        return rpc
    tx = VersionedTransaction.from_bytes(base64.b64decode(swap_tx([open_hop])))
    assert policy.check_swap_static(tx, ME, lambda k: [], 5_000_000).opens == [hop]
    out = signer(chain(ME)).sign_swap(swap_tx([open_hop]), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    assert out["simulated"] == {"sol": -50_005_000, "token": 1_000_000}
    with pytest.raises(PolicyError, match="spends"):
        signer(chain(ME)).sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    with pytest.raises(PolicyError, match="spends"):
        signer(chain(OTHER)).sign_swap(swap_tx([open_hop]), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)


def test_buy_caps():
    rpc = FakeRpc(); wallet(rpc); after_buy(rpc)
    s = signer(rpc, max_buy_lamports=10_000_000)
    with pytest.raises(PolicyError, match="per-trade cap"):
        s.sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    s = signer(rpc, daily_buy_lamports=120_000_000)
    s.sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    s.sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    with pytest.raises(PolicyError, match="daily") as e:
        s.sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    assert e.value.code == "cap"


def test_only_the_served_methods_can_be_called():
    c = client.LocalSigner(signer())
    assert c.call("pubkeys") == {"trading": str(ME)}
    with pytest.raises(PolicyError, match="unknown method"):
        c.call("_built")
