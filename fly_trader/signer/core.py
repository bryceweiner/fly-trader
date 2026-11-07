"""The signer: every swap and token-account close goes through here. The Jupiter swap is built elsewhere; the signer
checks it against policy (policy.py) and keeps its own ledger of what it signed (ledger.py).

What each method will sign:
  sign_swap     a Jupiter (Metis) swap between SOL and one token that the static rules and a simulation accept; buys
                capped per trade, per minute and per day
  close_atas    empty token accounts of the trading wallet, rent back to it
"""
from __future__ import annotations

import base64
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

from solders.address_lookup_table_account import AddressLookupTable
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from . import policy
from .ledger import Ledger
from .policy import NATIVE_MINT, TOKEN, TOKEN_2022, PolicyError
from .rpc import Rpc

LAMPORTS = 1_000_000_000
SIM_MAX_ACCOUNTS = 20                # the RPC caps accounts per simulation (30, or fewer: it has said 27)
_IO = ThreadPoolExecutor(max_workers=6, thread_name_prefix="signer-io")    # a swap's reads, side by side


@dataclass
class SignerConfig:
    max_buy_lamports: int = int(1.0 * LAMPORTS)
    daily_buy_lamports: int = int(10 * LAMPORTS)
    swaps_per_min: int = 6
    max_priority_lamports: int = 5_000_000
    min_value_ratio: float = 0.6
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, env=os.environ) -> "SignerConfig":
        f = lambda k, d: float(env.get(k) or d)                                    # noqa: E731
        return cls(max_buy_lamports=int(f("SIGNER_MAX_BUY_SOL", 1.0) * LAMPORTS),   # a safety bound, not sizing
                   daily_buy_lamports=int(f("SIGNER_DAILY_BUY_SOL", 10) * LAMPORTS), swaps_per_min=int(f("SIGNER_SWAPS_PER_MIN", 6)),
                   max_priority_lamports=int(f("SIGNER_MAX_PRIORITY_SOL", 0.005) * LAMPORTS), min_value_ratio=f("SIGNER_MIN_VALUE_RATIO", 0.6))


def _b64(tx: VersionedTransaction) -> str:
    return base64.b64encode(bytes(tx)).decode()


class Signer:
    def __init__(self, trading: Keypair, rpc: Rpc, cfg: SignerConfig, ledger: Ledger | None = None):
        self.trading, self.rpc, self.cfg = trading, rpc, cfg
        self.ledger = ledger or Ledger()

    # ------------------------------------------------------------------ helpers
    def _built(self, payer: Keypair, ixs: list[Instruction], signers: list[Keypair]) -> dict:
        bh, lvbh = self.rpc.blockhash()
        msg = MessageV0.try_compile(payer.pubkey(), ixs, [], Hash.from_string(bh))
        tx = VersionedTransaction(msg, signers)
        return {"tx": _b64(tx), "signature": str(tx.signatures[0]), "last_valid_block_height": lvbh, "blockhash": bh}

    def _resolve_alt(self, table: Pubkey) -> list[Pubkey]:
        acc = self.rpc.account(table)
        if acc is None:
            raise PolicyError(f"lookup table {table} not found")
        return list(AddressLookupTable.deserialize(acc["data"]).addresses)

    def _snapshot(self, accounts: list[dict | None], addrs: list[Pubkey]) -> policy.Snapshot:
        """The trading wallet's lamports and the token accounts among ``addrs`` that are its own (their authority)."""
        me = self.trading.pubkey()
        tokens = {}
        for a, acc in zip(addrs[1:], accounts[1:]):
            if acc and acc["owner"] in (TOKEN, TOKEN_2022) and len(acc["data"]) >= 72 and policy.token_owner(acc["data"]) == me:
                tokens[a] = (policy.token_mint(acc["data"]), policy.token_amount(acc["data"]), acc["lamports"])
        return policy.Snapshot(lamports=accounts[0]["lamports"] if accounts[0] else 0, tokens=tokens)

    # ------------------------------------------------------------------ methods
    def pubkeys(self) -> dict:
        return {"trading": str(self.trading.pubkey())}

    def sign_swap(self, tx: str, in_mint: str, out_mint: str, in_amount: int, min_out: int) -> dict:
        me = self.trading.pubkey()
        in_m, out_m = Pubkey.from_string(in_mint), Pubkey.from_string(out_mint)
        buy = in_m == NATIVE_MINT
        now = time.time()
        if len(self.ledger.swaps_since(now - 60)) >= self.cfg.swaps_per_min:
            raise PolicyError("swap rate limit", code="cap")
        if buy:
            if in_amount > self.cfg.max_buy_lamports:
                raise PolicyError(f"buy of {in_amount} lamports is above the per-trade cap {self.cfg.max_buy_lamports}")
            spent = sum(r["in_amount"] for r in self.ledger.swaps_since(now - 86400) if r["direction"] == "buy")
            if spent + in_amount > self.cfg.daily_buy_lamports:
                raise PolicyError("daily buy volume cap reached", code="cap")
        vt = VersionedTransaction.from_bytes(base64.b64decode(tx))
        chk = policy.check_swap_static(vt, me, self._resolve_alt, self.cfg.max_priority_lamports)
        # every token account the trading wallet owns, plus those this swap opens for it: the in and out tokens' and a
        # multi-hop route's for its intermediate tokens, whose rent stays the wallet's (their close returns it) and is not
        # spent (the reads side by side: a live order waits on this signature)
        tok = out_m if buy else in_m
        f_mints = _IO.submit(self.rpc.accounts, [in_m, out_m])
        f_spl, f_t22 = _IO.submit(self.rpc.token_accounts, me, TOKEN), _IO.submit(self.rpc.token_accounts, me, TOKEN_2022)
        f_prices = _IO.submit(self.rpc.usd_prices, [str(tok), str(NATIVE_MINT)])
        mint_acc = f_mints.result()
        progs = {m: (a["owner"] if a else TOKEN) for m, a in zip((in_m, out_m), mint_acc)}
        addrs = [me] + f_spl.result() + f_t22.result()
        for a in [policy.ata(me, m, progs[m]) for m in (in_m, out_m)] + chk.opens + chk.writable:   # writable: accounts the route opens
            if a not in addrs:
                addrs.append(a)
        f_pre = _IO.submit(self.rpc.accounts, addrs)
        f_sims = [_IO.submit(self.rpc.simulate, tx, addrs[i:i + SIM_MAX_ACCOUNTS]) for i in range(0, len(addrs), SIM_MAX_ACCOUNTS)]   # in chunks: the RPC caps accounts per simulation
        pre_accs = f_pre.result()
        pre = self._snapshot(pre_accs, addrs)
        post_accs: list[dict | None] = []
        for f in f_sims:
            sim = f.result()
            if sim.get("err") is not None:
                raise PolicyError(f"the swap fails in simulation: {sim['err']} {(sim.get('logs') or [])[-3:]}", code="in_flight")
            post_accs += [None if a is None else {"lamports": a["lamports"], "owner": Pubkey.from_string(a["owner"]), "data": base64.b64decode(a["data"][0])}
                          for a in sim["accounts"]]
        post = self._snapshot(post_accs, addrs)
        vault_rent = policy.rent_opened_for_others(pre_accs, post_accs)          # a pool's fee vault opened at our expense
        slack = policy.BASE_FEE_LAMPORTS * vt.message.header.num_required_signatures + chk.priority_lamports + 10_000 + vault_rent
        eff = policy.check_swap_effects(pre, post, in_mint=in_m, out_mint=out_m, in_amount=int(in_amount), min_out=int(min_out), slack_lamports=slack)
        tok_acc = mint_acc[1] if buy else mint_acc[0]
        prices = f_prices.result()
        ratio = policy.check_value("buy" if buy else "sell", eff["sol"], abs(eff["token"]), tok_acc["data"][44] if tok_acc else 0,
                                   prices.get(str(tok)), prices.get(str(NATIVE_MINT)), self.cfg.min_value_ratio)
        sigs = list(vt.signatures)
        sigs[chk.signer_index] = self.trading.sign_message(to_bytes_versioned(vt.message))
        signed = VersionedTransaction.populate(vt.message, sigs)
        sig = str(sigs[chk.signer_index])
        self.ledger.record_swap("buy" if buy else "sell", in_mint, out_mint, int(in_amount), -eff["sol"], sig)
        return {"tx": _b64(signed), "signature": sig, "simulated": eff, "value_ratio": ratio, "vault_rent": vault_rent}

    def close_atas(self, accounts: list[str]) -> list[dict]:
        me = self.trading.pubkey()
        keys = [Pubkey.from_string(a) for a in accounts]
        ixs = []
        for k, acc in zip(keys, self.rpc.accounts(keys)):
            if not acc or acc["owner"] not in (TOKEN, TOKEN_2022) or len(acc["data"]) < 72:
                continue
            if Pubkey.from_bytes(acc["data"][32:64]) != me or policy.token_amount(acc["data"]) != 0:
                raise PolicyError(f"{k} is not an empty token account of the trading wallet")
            ixs.append(Instruction(acc["owner"], bytes([9]), [AccountMeta(k, False, True), AccountMeta(me, False, True), AccountMeta(me, True, False)]))
        return [self._built(self.trading, ixs[i:i + 10], [self.trading]) for i in range(0, len(ixs), 10)]


SERVED = ("pubkeys", "sign_swap", "close_atas")
