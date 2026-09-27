"""The signer: what it signs and what it refuses (fly_trader/signer), offline with a fake RPC."""
import base64
import json
import socket
import struct
import threading
import time
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

from fly_trader.signer import client, core, policy, squads
from fly_trader.signer.ledger import Ledger
from fly_trader.signer.policy import JUPITER_V6, NATIVE_MINT, TOKEN, PolicyError, Snapshot

FIX = json.loads((Path(__file__).parent / "fixtures" / "ultra_orders.json").read_text())
TRADING, PAYOUT, OWNER = Keypair.from_seed(bytes([7]) * 32), Keypair.from_seed(bytes([8]) * 32), Keypair.from_seed(bytes([9]) * 32)
ME = TRADING.pubkey()
MEME = Keypair.from_seed(bytes([10]) * 32).pubkey()
OTHER = Keypair.from_seed(bytes([11]) * 32).pubkey()
MULTISIG = squads.multisig_pda(Keypair.from_seed(bytes([12]) * 32).pubkey())
TREASURY = squads.vault_pda(MULTISIG, 0)
L1 = squads.spending_limit_pda(MULTISIG, Keypair.from_seed(bytes([13]) * 32).pubkey())
L2 = squads.spending_limit_pda(MULTISIG, Keypair.from_seed(bytes([14]) * 32).pubkey())
BH = "11111111111111111111111111111111"


def token_data(mint: Pubkey, owner: Pubkey, amount: int) -> bytes:
    return bytes(mint) + bytes(owner) + struct.pack("<Q", amount) + bytes(165 - 72)


def mint_data(decimals: int = 6) -> bytes:
    return bytes(44) + bytes([decimals]) + bytes(37)


def limit(member, dests, amount=2 * 10**9, remaining=2 * 10**9, period="Day", last_reset=None):
    return {"lamports": 3_000_000, "owner": squads.PROGRAM_ID, "data": squads.encode_spending_limit(
        {"multisig": MULTISIG, "create_key": Keypair().pubkey(), "amount": amount, "period": period, "remaining": remaining,
         "last_reset": int(time.time()) if last_reset is None else last_reset, "members": [member], "destinations": dests})}


class FakeRpc:
    def __init__(self):
        self.acc: dict[Pubkey, dict] = {}
        self.post: dict[Pubkey, dict | None] = {}
        self.statuses: dict[str, dict | None] = {}
        self.height = 100
        self.sim_err = None
        self.prices = {}

    def account(self, k, commitment="confirmed"):
        return self.acc.get(k)

    def accounts(self, keys, commitment="confirmed"):
        return [self.acc.get(k) for k in keys]

    def balance(self, k, commitment="confirmed"):
        return (self.acc.get(k) or {"lamports": 0})["lamports"]

    def token_accounts(self, owner, program):
        return [k for k, a in self.acc.items() if a["owner"] == program and bytes(a["data"][32:64]) == bytes(owner)]

    def blockhash(self):
        return BH, 150

    def block_height(self):
        return self.height

    def signature_status(self, sig):
        return self.statuses.get(sig)

    def simulate(self, tx, addresses):
        out = []
        for a in addresses:
            v = self.post[a] if a in self.post else self.acc.get(a)
            out.append(None if v is None else {"lamports": v["lamports"], "owner": str(v["owner"]), "data": [base64.b64encode(v["data"]).decode(), "base64"]})
        return {"err": self.sim_err, "logs": [], "accounts": out}

    def usd_prices(self, mints):
        return {m: self.prices[m] for m in mints if m in self.prices}


def signer(rpc=None, **cfg) -> core.Signer:
    c = core.SignerConfig(multisig=str(MULTISIG), limit_trading=str(L1), limit_payout=str(L2), **cfg)
    return core.Signer(TRADING, PAYOUT, rpc or FakeRpc(), c, Ledger())


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


class Direct:
    """The brain's view of a signer (``call``) over a Signer object, as client.LocalSigner does."""

    def __init__(self, s):
        self.s, self.chain = s, s.rpc

    def call(self, method, **p):
        return getattr(self.s, method)(**p)


def vault_signer(l2_remaining=10**9):
    """A real signer over a fake chain: a funded treasury and a weekly L2 with ``l2_remaining`` left (other tests)."""
    chain = FakeRpc(); wallet(chain)
    treasury(chain, l2=limit(PAYOUT.pubkey(), [], amount=10**9, remaining=l2_remaining, period="Week"))
    return Direct(signer(chain))


# ------------------------------------------------------------------ squads
def test_spending_limit_use_layout():
    ix = squads.spending_limit_use_sol(MULTISIG, PAYOUT.pubkey(), L2, TREASURY, OTHER, 123, "fly-vault-claim:9")
    d = bytes(ix.data)
    assert d[:8] == squads.IX_SPENDING_LIMIT_USE and struct.unpack("<QB", d[8:17]) == (123, 9)
    assert d[17] == 1 and d[22:] == b"fly-vault-claim:9" and struct.unpack("<I", d[18:22])[0] == len(b"fly-vault-claim:9")
    metas = [(m.pubkey, m.is_signer, m.is_writable) for m in ix.accounts]
    assert metas[:5] == [(MULTISIG, False, False), (PAYOUT.pubkey(), True, False), (L2, False, True), (TREASURY, False, True), (OTHER, False, True)]
    assert [m[0] for m in metas[6:]] == [squads.PROGRAM_ID] * 4                 # Anchor's "None" optional accounts
    assert squads.decode_spending_limit(limit(TRADING.pubkey(), [ME])["data"])["destinations"] == [ME]


def test_remaining_resets_by_period():
    lim = {"period": "Day", "amount": 10, "remaining": 1, "last_reset": 1000}
    assert squads.remaining_now(lim, 1000 + 86400) == 1 and squads.remaining_now(lim, 1000 + 86401) == 10
    assert squads.remaining_now({**lim, "period": "OneTime"}, 10**10) == 1


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


def test_buy_caps_rate_and_panic():
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
    s = signer(rpc); s.panic("test")
    with pytest.raises(PolicyError, match="panic"):
        s.sign_swap(swap_tx(), str(NATIVE_MINT), str(MEME), 50_000_000, 900_000)
    with pytest.raises(PolicyError, match="panic"):
        s.topup(10**8)


# ------------------------------------------------------------------ treasury: top-up, return, claims
def treasury(rpc: FakeRpc, lamports=10 * 10**9, l1=None, l2=None):
    rpc.acc[TREASURY] = {"lamports": lamports, "owner": policy.SYSTEM, "data": b""}
    rpc.acc[L1] = l1 or limit(ME, [ME])
    rpc.acc[L2] = l2 or limit(PAYOUT.pubkey(), [], amount=10**9, remaining=10**9, period="Week")


def test_topup_uses_l1_only_to_the_trading_wallet():
    rpc = FakeRpc(); wallet(rpc); treasury(rpc)
    out = signer(rpc).topup(10**9)
    msg = VersionedTransaction.from_bytes(base64.b64decode(out["tx"])).message
    keys = list(msg.account_keys); ix = msg.instructions[0]
    assert keys[ix.program_id_index] == squads.PROGRAM_ID
    assert [keys[i] for i in bytes(ix.accounts)][:5] == [MULTISIG, ME, L1, TREASURY, ME]
    rpc.acc[L1] = limit(ME, [])                                                            # owner forgot the destination lock
    with pytest.raises(PolicyError, match="only destination"):
        signer(rpc).topup(10**9)
    rpc.acc[L1] = limit(ME, [ME], remaining=10**8)
    with pytest.raises(PolicyError, match="too little left") as e:
        signer(rpc).topup(10**9)
    assert e.value.code == "cap"
    del rpc.acc[L1]                                                                        # revoked from the owner's phone
    with pytest.raises(PolicyError, match="revoked") as e:
        signer(rpc).topup(10**8)
    assert e.value.code == "cap"
    rpc.acc[L1] = limit(ME, [ME]); rpc.acc[TREASURY]["lamports"] = 10**8
    with pytest.raises(PolicyError) as e:
        signer(rpc).topup(10**9)
    assert e.value.code == "liquidity"


def test_return_float_goes_to_the_treasury_only():
    rpc = FakeRpc(); wallet(rpc); treasury(rpc)
    msg = VersionedTransaction.from_bytes(base64.b64decode(signer(rpc).return_float(10**9)["tx"])).message
    keys = list(msg.account_keys)
    assert TREASURY in keys and OTHER not in keys


def test_claims_pay_once_per_id():
    rpc = FakeRpc(); wallet(rpc); treasury(rpc)
    s = signer(rpc)
    first = s.pay_claim("7", str(OTHER), 10**8)
    tx = VersionedTransaction.from_bytes(base64.b64decode(first["tx"]))
    assert tx.verify_with_results() == [True, True] and list(tx.message.account_keys)[:2] == [ME, PAYOUT.pubkey()]
    with pytest.raises(PolicyError) as e:                                                  # may still land
        s.pay_claim("7", str(OTHER), 10**8)
    assert e.value.code == "in_flight"
    with pytest.raises(PolicyError, match="already signed"):                               # same id, another wallet
        s.pay_claim("7", str(Keypair().pubkey()), 10**8)
    rpc.statuses[first["signature"]] = {"err": None, "confirmationStatus": "finalized"}
    with pytest.raises(PolicyError) as e:
        s.pay_claim("7", str(OTHER), 10**8)
    assert e.value.code == "paid"
    rpc.statuses.clear(); rpc.height = 151                                                 # expired unlanded: re-sign
    s.pay_claim("7", str(OTHER), 10**8)
    assert s.ledger.claim("7")["attempts"] == 2
    with pytest.raises(PolicyError, match="this week") as e:                               # over the weekly cap
        s.pay_claim("8", str(OTHER), 2 * 10**9)
    assert e.value.code == "cap"


# ------------------------------------------------------------------ the socket
def test_socket_round_trip(tmp_path):
    from fly_trader.signer import server
    rpc = FakeRpc(); wallet(rpc); treasury(rpc)
    s = signer(rpc)
    path = str(Path("/tmp") / f"fly-signer-test-{time.time_ns()}.sock")               # AF_UNIX paths must be short
    th = threading.Thread(target=server.serve, args=(s, path), daemon=True); th.start()
    for _ in range(100):
        if Path(path).exists():
            break
        time.sleep(0.02)
    c = client.SocketSigner(path)
    assert c.call("pubkeys")["trading"] == str(ME)
    with pytest.raises(PolicyError) as e:
        c.call("topup", lamports=10**12)
    assert e.value.code == "policy"
    with pytest.raises(PolicyError, match="unknown method"):
        c.call("clear_panic")                                                              # never served
    raw = socket.socket(socket.AF_UNIX); raw.connect(path); raw.sendall(b"not json\n")
    assert json.loads(raw.makefile().readline())["ok"] is False
    Path(path).unlink(missing_ok=True)
