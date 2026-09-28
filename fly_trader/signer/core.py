"""The signer: the only code on the server that holds the two hot keys. The brain asks it for signed transactions and
broadcasts them itself (it persists the signature first, as before); the signer builds every transaction except the
Jupiter swap itself, checks each against policy (policy.py) and the treasury's on-chain spending limits (squads.py),
and keeps its own ledger (ledger.py). It never trusts the brain's database.

What each method will sign:
  sign_swap     a Jupiter (Metis) swap between SOL and one token that the static rules and a simulation accept; buys
                capped per trade, per minute and per day; none while panicking
  topup         treasury -> trading under L1 (destination locked to the trading wallet by the owner)
  return_float  trading -> treasury (a fixed destination)
  pay_claim     treasury -> a holder under L2, once per claim id (re-signed only after the earlier one provably failed
                or expired), within what L2 has left this week
  close_atas    empty token accounts of the trading wallet, rent back to it
  backup_keys   both keys age-encrypted to the operator's public key (the plaintext never leaves this process)
  panic         from now on only sells, return_float and closes (``clear-panic`` is a CLI command inside the container)
"""
from __future__ import annotations

import base64
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field

from solders.address_lookup_table_account import AddressLookupTable
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.keypair import Keypair
from solders.message import MessageV0, to_bytes_versioned
from solders.pubkey import Pubkey
from solders.system_program import TransferParams, transfer
from solders.transaction import VersionedTransaction

from . import policy, squads
from .ledger import Ledger
from .policy import NATIVE_MINT, TOKEN, TOKEN_2022, PolicyError
from .rpc import Rpc

RENT_EXEMPT_MIN = 890_880          # a 0-byte system account: the treasury PDA must keep this much
LAMPORTS = 1_000_000_000
SIM_MAX_ACCOUNTS = 20                # the RPC caps accounts per simulation (30, or fewer: it has said 27)


@dataclass
class SignerConfig:
    multisig: str | None = None                  # the Squads multisig (the treasury is its vault 0)
    limit_trading: str | None = None             # L1 spending-limit account
    limit_payout: str | None = None              # L2 spending-limit account
    max_buy_lamports: int = int(1.0 * LAMPORTS)
    daily_buy_lamports: int = int(10 * LAMPORTS)
    swaps_per_min: int = 6
    max_priority_lamports: int = 5_000_000
    min_value_ratio: float = 0.6
    topup_max_lamports: int = int(2 * LAMPORTS)
    backup_recipient: str | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, env=os.environ) -> "SignerConfig":
        f = lambda k, d: float(env.get(k) or d)                                    # noqa: E731
        return cls(multisig=env.get("VAULT_MULTISIG") or None, limit_trading=env.get("VAULT_LIMIT_TRADING") or None,
                   limit_payout=env.get("VAULT_LIMIT_PAYOUT") or None,
                   max_buy_lamports=int(f("SIGNER_MAX_BUY_SOL", 1.0) * LAMPORTS),   # a safety bound, not sizing
                   daily_buy_lamports=int(f("SIGNER_DAILY_BUY_SOL", 10) * LAMPORTS), swaps_per_min=int(f("SIGNER_SWAPS_PER_MIN", 6)),
                   max_priority_lamports=int(f("SIGNER_MAX_PRIORITY_SOL", 0.005) * LAMPORTS), min_value_ratio=f("SIGNER_MIN_VALUE_RATIO", 0.6),
                   topup_max_lamports=int(f("TREASURY_FLOAT_SOL", 2.0) * LAMPORTS), backup_recipient=env.get("VAULT_BACKUP_RECIPIENT") or None)


def _b64(tx: VersionedTransaction) -> str:
    return base64.b64encode(bytes(tx)).decode()


class Signer:
    def __init__(self, trading: Keypair, payout: Keypair | None, rpc: Rpc, cfg: SignerConfig, ledger: Ledger | None = None):
        if payout is not None and payout.pubkey() == trading.pubkey():
            raise ValueError("the payout key must differ from the trading key")
        self.trading, self.payout, self.rpc, self.cfg = trading, payout, rpc, cfg
        self.ledger = ledger or Ledger()
        self.multisig = Pubkey.from_string(cfg.multisig) if cfg.multisig else None
        self.treasury = squads.vault_pda(self.multisig, 0) if self.multisig else None

    # ------------------------------------------------------------------ helpers
    def _built(self, payer: Keypair, ixs: list[Instruction], signers: list[Keypair]) -> dict:
        bh, lvbh = self.rpc.blockhash()
        msg = MessageV0.try_compile(payer.pubkey(), ixs, [], Hash.from_string(bh))
        tx = VersionedTransaction(msg, signers)
        return {"tx": _b64(tx), "signature": str(tx.signatures[0]), "last_valid_block_height": lvbh, "blockhash": bh}

    def _limit(self, which: str, member: Pubkey) -> tuple[Pubkey, dict]:
        addr = self.cfg.limit_trading if which == "L1" else self.cfg.limit_payout
        if not (self.multisig and addr):
            raise PolicyError(f"{which} is not configured (VAULT_MULTISIG / VAULT_LIMIT_*)")
        key = Pubkey.from_string(addr)
        acc = self.rpc.account(key)
        if acc is None or acc["owner"] != squads.PROGRAM_ID:
            raise PolicyError(f"{which} ({addr}) is not a Squads account: revoked?", code="cap")
        lim = squads.decode_spending_limit(acc["data"])
        if lim["multisig"] != self.multisig or lim["vault_index"] != 0 or lim["mint"] != squads.SYSTEM_PROGRAM:
            raise PolicyError(f"{which} is not a SOL limit on this treasury")
        if member not in lim["members"]:
            raise PolicyError(f"{which} does not list this key as a member: revoked?", code="cap")
        return key, lim

    def _treasury_spendable(self) -> int:
        return max(0, self.rpc.balance(self.treasury) - RENT_EXEMPT_MIN)

    def _resolve_alt(self, table: Pubkey) -> list[Pubkey]:
        acc = self.rpc.account(table)
        if acc is None:
            raise PolicyError(f"lookup table {table} not found")
        return list(AddressLookupTable.deserialize(acc["data"]).addresses)

    def _snapshot(self, accounts: list[dict | None], addrs: list[Pubkey]) -> policy.Snapshot:
        tokens = {}
        for a, acc in zip(addrs[1:], accounts[1:]):
            if acc and acc["owner"] in (TOKEN, TOKEN_2022) and len(acc["data"]) >= 72:
                tokens[a] = (policy.token_mint(acc["data"]), policy.token_amount(acc["data"]), acc["lamports"])
        return policy.Snapshot(lamports=accounts[0]["lamports"] if accounts[0] else 0, tokens=tokens)

    # ------------------------------------------------------------------ methods
    def pubkeys(self) -> dict:
        return {"trading": str(self.trading.pubkey()), "payout": str(self.payout.pubkey()) if self.payout else None,
                "treasury": str(self.treasury) if self.treasury else None, "multisig": self.cfg.multisig}

    def sign_swap(self, tx: str, in_mint: str, out_mint: str, in_amount: int, min_out: int) -> dict:
        me = self.trading.pubkey()
        in_m, out_m = Pubkey.from_string(in_mint), Pubkey.from_string(out_mint)
        buy = in_m == NATIVE_MINT
        if buy and self.ledger.flag("panic"):
            raise PolicyError("panic: no buys until cleared on the server")
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
        # every token account the trading wallet owns, plus the two this swap may open
        mint_acc = self.rpc.accounts([in_m, out_m])
        progs = {m: (a["owner"] if a else TOKEN) for m, a in zip((in_m, out_m), mint_acc)}
        addrs = [me] + self.rpc.token_accounts(me, TOKEN) + self.rpc.token_accounts(me, TOKEN_2022)
        for m in (in_m, out_m):
            a = policy.ata(me, m, progs[m])
            if a not in addrs:
                addrs.append(a)
        pre = self._snapshot(self.rpc.accounts(addrs), addrs)
        post_accs: list[dict | None] = []
        for i in range(0, len(addrs), SIM_MAX_ACCOUNTS):          # in chunks: the RPC caps accounts per simulation
            sim = self.rpc.simulate(tx, addrs[i:i + SIM_MAX_ACCOUNTS])
            if sim.get("err") is not None:
                raise PolicyError(f"the swap fails in simulation: {sim['err']} {(sim.get('logs') or [])[-3:]}", code="in_flight")
            post_accs += [None if a is None else {"lamports": a["lamports"], "owner": Pubkey.from_string(a["owner"]), "data": base64.b64decode(a["data"][0])}
                          for a in sim["accounts"]]
        post = self._snapshot(post_accs, addrs)
        slack = policy.BASE_FEE_LAMPORTS * vt.message.header.num_required_signatures + chk.priority_lamports + 10_000
        eff = policy.check_swap_effects(pre, post, in_mint=in_m, out_mint=out_m, in_amount=int(in_amount), min_out=int(min_out), slack_lamports=slack)
        tok = out_m if buy else in_m
        tok_acc = mint_acc[1] if buy else mint_acc[0]
        prices = self.rpc.usd_prices([str(tok), str(NATIVE_MINT)])
        ratio = policy.check_value("buy" if buy else "sell", eff["sol"], abs(eff["token"]), tok_acc["data"][44] if tok_acc else 0,
                                   prices.get(str(tok)), prices.get(str(NATIVE_MINT)), self.cfg.min_value_ratio)
        sigs = list(vt.signatures)
        sigs[chk.signer_index] = self.trading.sign_message(to_bytes_versioned(vt.message))
        signed = VersionedTransaction.populate(vt.message, sigs)
        sig = str(sigs[chk.signer_index])
        self.ledger.record_swap("buy" if buy else "sell", in_mint, out_mint, int(in_amount), -eff["sol"], sig)
        return {"tx": _b64(signed), "signature": sig, "simulated": eff, "value_ratio": ratio}

    def topup(self, lamports: int) -> dict:
        if self.ledger.flag("panic"):
            raise PolicyError("panic: no top-ups")
        lamports = int(lamports)
        if not 0 < lamports <= self.cfg.topup_max_lamports:
            raise PolicyError(f"top-up of {lamports} is outside (0, {self.cfg.topup_max_lamports}]")
        me = self.trading.pubkey()
        key, lim = self._limit("L1", me)
        if lim["destinations"] != [me]:
            raise PolicyError("L1 must be locked to the trading wallet as its only destination (owner: recreate it)")
        if squads.remaining_now(lim, int(time.time())) < lamports:
            raise PolicyError("L1 has too little left today", code="cap")
        if self._treasury_spendable() < lamports:
            raise PolicyError("the treasury holds too little", code="liquidity")
        ix = squads.spending_limit_use_sol(self.multisig, me, key, self.treasury, me, lamports, "fly-topup")
        return self._built(self.trading, [ix], [self.trading])

    def return_float(self, lamports: int) -> dict:
        lamports = int(lamports)
        if lamports <= 0 or self.treasury is None:
            raise PolicyError("nothing to return, or no treasury configured")
        if self.rpc.balance(self.trading.pubkey()) < lamports + 10_000:
            raise PolicyError("the trading wallet holds less than that")
        ix = transfer(TransferParams(from_pubkey=self.trading.pubkey(), to_pubkey=self.treasury, lamports=lamports))
        return self._built(self.trading, [ix], [self.trading])

    def pay_claim(self, claim_id: str, dest: str, lamports: int) -> dict:
        if self.ledger.flag("panic"):
            raise PolicyError("panic: claims are halted", code="cap")
        if self.payout is None:
            raise PolicyError("no payout key")
        claim_id, lamports, to = str(claim_id), int(lamports), Pubkey.from_string(dest)
        if lamports <= 0:
            raise PolicyError("a claim must be positive")
        prev = self.ledger.claim(claim_id)
        if prev is not None:
            if prev["dest"] != dest or prev["lamports"] != lamports:
                raise PolicyError(f"claim {claim_id} was already signed for {prev['lamports']} lamports to {prev['dest']}")
            st = self.rpc.signature_status(prev["signature"])
            if st is not None and st.get("err") is None and st.get("confirmationStatus") in ("confirmed", "finalized"):
                raise PolicyError(f"claim {claim_id} is already paid ({prev['signature']})", code="paid")
            if st is None and self.rpc.block_height() <= prev["last_valid_block_height"]:
                raise PolicyError(f"claim {claim_id}'s earlier payment may still land", code="in_flight")
        key, lim = self._limit("L2", self.payout.pubkey())
        if lim["destinations"] and to not in lim["destinations"]:
            raise PolicyError("L2 does not allow this destination")
        if squads.remaining_now(lim, int(time.time())) < lamports:
            raise PolicyError("L2 has too little left this week: raise it from the owner wallet", code="cap")
        if self._treasury_spendable() < lamports:
            raise PolicyError("the treasury holds too little", code="liquidity")
        ix = squads.spending_limit_use_sol(self.multisig, self.payout.pubkey(), key, self.treasury, to, lamports, f"fly-vault-claim:{claim_id}")
        out = self._built(self.trading, [ix], [self.trading, self.payout])     # trading pays the fee; payout uses L2
        self.ledger.record_claim(claim_id, dest, lamports, out["signature"], out["last_valid_block_height"])
        return out

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

    def backup_keys(self) -> list[dict]:
        rcpt = self.cfg.backup_recipient
        if not rcpt:
            raise PolicyError("no backup recipient (VAULT_BACKUP_RECIPIENT)")
        import base58
        out = []
        with tempfile.NamedTemporaryFile("w", delete=False) as f:
            f.write(rcpt.strip() + "\n"); rfile = f.name
        try:
            for name, kp in (("trading", self.trading), ("payout", self.payout)):
                if kp is None:
                    continue
                enc = subprocess.run(["age", "-R", rfile], input=base58.b58encode(bytes(kp)), check=True, capture_output=True).stdout
                out.append({"name": name, "pubkey": str(kp.pubkey()), "age_b64": base64.b64encode(enc).decode()})
        finally:
            os.unlink(rfile)
        return out

    def panic(self, reason: str = "") -> dict:
        self.ledger.set_flag("panic", reason or "panic")
        return {"panic": True}

    def clear_panic(self) -> dict:                     # CLI only (not served on the socket)
        self.ledger.set_flag("panic", None)
        return {"panic": False}

    def status(self) -> dict:
        now = time.time()
        return {"panic": self.ledger.flag("panic"), "swaps_last_min": len(self.ledger.swaps_since(now - 60)),
                "buys_24h_lamports": sum(r["in_amount"] for r in self.ledger.swaps_since(now - 86400) if r["direction"] == "buy"),
                **self.pubkeys()}


SERVED = ("pubkeys", "sign_swap", "topup", "return_float", "pay_claim", "close_atas", "backup_keys", "panic", "status")
