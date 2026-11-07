"""What the signer will sign for a swap. Two layers, both required:

1. Static: every top-level instruction of the Jupiter transaction is one we expect from a Metis route
   (tools: ComputeBudget, System wrap, SPL Token sync/close, ATA create, Jupiter v6), each with the trading wallet in
   the only roles it may have; the priority fee is capped. Program ids are always static keys in a v0 message, so a
   lookup table can hide an account but never a program.
2. Simulated: the signer simulates the transaction itself (its own RPC) and compares the trading wallet's SOL and every
   token account it owns before and after (those the transaction opens for it included: a multi-hop route's token
   accounts for its intermediate tokens hold their rent for the wallet). A buy may spend at most the ordered SOL (+ fees) and must deliver
   the bought token to the trading wallet; a sell may spend at most the ordered tokens and must return SOL to it; no
   other holding may shrink. So a route that pays someone else, drains another holding or overspends is refused
   whatever the brain claimed. One cost to others is allowed: the rent of an empty token account the route opens inside
   itself for a program's vault (a pool's fee vault opened at the buyer's expense; ``rent_opened_for_others``).

Pure functions only: the RPC calls live in core.py, which passes their results in here.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field

from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

SYSTEM = Pubkey.from_string("11111111111111111111111111111111")
COMPUTE_BUDGET = Pubkey.from_string("ComputeBudget111111111111111111111111111111")
TOKEN = Pubkey.from_string("TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA")
TOKEN_2022 = Pubkey.from_string("TokenzQdBNbLqP5VEhdkAS6EPFLC1PHnBqCXEpPxuEb")
ATA = Pubkey.from_string("ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL")
JUPITER_V6 = Pubkey.from_string("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4")
NATIVE_MINT = Pubkey.from_string("So11111111111111111111111111111111111111112")
MEMO = Pubkey.from_string("MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr")
SWAP_PROGRAMS = frozenset({SYSTEM, COMPUTE_BUDGET, TOKEN, TOKEN_2022, ATA, JUPITER_V6})
# the Jupiter routers the broker asks for (Metis only): every other router brings programs we have not pinned
EXCLUDE_ROUTERS = "jupiterz,dflow,okx"
ATA_RENT_LAMPORTS = 2_039_280
BASE_FEE_LAMPORTS = 5_000
RENT_PER_BYTE_MAX = 6_960            # lamports per byte (+128 of overhead) a rent exemption cost before 2026; it is lower now
OPENED_FOR_OTHERS_MAX = 2            # empty vault accounts a swap may open at the wallet's expense


class PolicyError(RuntimeError):
    """``code``: 'policy' (refused: never retry as is), 'cap' (a limit's period is used up: wait), 'in_flight' (the swap
    failed in simulation: retry with a fresh route)."""

    def __init__(self, msg: str, code: str = "policy"):
        super().__init__(msg)
        self.code = code


def ata(owner: Pubkey, mint: Pubkey, token_program: Pubkey = TOKEN) -> Pubkey:
    return Pubkey.find_program_address([bytes(owner), bytes(token_program), bytes(mint)], ATA)[0]


def account_keys(msg: MessageV0, resolve_alt) -> list[Pubkey]:
    """Static keys, then every lookup table's writable, then readonly addresses (the v0 message order).
    ``resolve_alt(table) -> list[Pubkey]`` reads a lookup table (the signer's own RPC)."""
    keys = list(msg.account_keys)
    lookups = list(msg.address_table_lookups)
    tables = {lk.account_key: resolve_alt(lk.account_key) for lk in lookups}
    try:
        keys += [tables[lk.account_key][i] for lk in lookups for i in bytes(lk.writable_indexes)]
        keys += [tables[lk.account_key][i] for lk in lookups for i in bytes(lk.readonly_indexes)]
    except IndexError as e:
        raise PolicyError("a lookup table index is out of range") from e
    return keys


@dataclass
class SwapCheck:
    signer_index: int
    fee_payer: Pubkey
    priority_lamports: int
    creates_atas: int
    opens: list[Pubkey] = field(default_factory=list)   # the token accounts its ATA instructions open for the trading wallet
    writable: list[Pubkey] = field(default_factory=list)  # every account the transaction may write (lookup tables resolved)


def check_swap_static(tx: VersionedTransaction, trading: Pubkey, resolve_alt, max_priority_lamports: int) -> SwapCheck:
    msg = tx.message
    if not isinstance(msg, MessageV0) and not hasattr(msg, "account_keys"):
        raise PolicyError("not a transaction message")
    static = list(msg.account_keys)
    n_sign = msg.header.num_required_signatures
    try:
        idx = static.index(trading)
    except ValueError:
        raise PolicyError("the trading wallet does not sign this transaction") from None
    if idx >= n_sign:
        raise PolicyError("the trading wallet is not a required signer")
    keys = account_keys(msg, resolve_alt) if hasattr(msg, "address_table_lookups") else static
    wsol = ata(trading, NATIVE_MINT)
    cu_limit, cu_price, creates, opens = 200_000, 0, 0, []
    for ix in msg.instructions:
        prog = static[ix.program_id_index]
        if prog not in SWAP_PROGRAMS:
            raise PolicyError(f"program {prog} is not one a swap may call")
        data = bytes(ix.data); acc = [keys[i] for i in bytes(ix.accounts)]
        tag = data[0] if data else None
        if prog == COMPUTE_BUDGET:
            if tag == 2 and len(data) >= 5:
                cu_limit = struct.unpack("<I", data[1:5])[0]
            elif tag == 3 and len(data) >= 9:
                cu_price = struct.unpack("<Q", data[1:9])[0]
            elif tag not in (1, 4):                        # heap frame, loaded-accounts limit: harmless
                raise PolicyError(f"compute-budget instruction {tag} not expected")
        elif prog == SYSTEM:
            if tag != 2 or len(acc) < 2:                    # only Transfer: wrapping SOL into our own wSOL account
                raise PolicyError("only a SOL wrap into the trading wallet's wSOL account may use the System program")
            if acc[0] != trading or acc[1] != wsol:
                raise PolicyError("a System transfer that is not the trading wallet's own SOL wrap")
        elif prog in (TOKEN, TOKEN_2022):
            if tag == 17:                                    # SyncNative
                continue
            if tag == 9 and len(acc) >= 3:                   # CloseAccount(account, destination, owner)
                if acc[1] != trading or acc[2] != trading:
                    raise PolicyError("a token account close that does not return its rent to the trading wallet")
                continue
            raise PolicyError(f"token instruction {tag} is not part of a swap (transfer, approve, authority changes are refused)")
        elif prog == ATA:
            if tag not in (None, 0, 1) or len(acc) < 3 or acc[2] != trading:   # Create / CreateIdempotent for our wallet
                raise PolicyError("an associated-token-account instruction for another wallet")
            creates += 1
            if acc[1] not in opens:                          # (payer, the account, wallet, mint, ...)
                opens.append(acc[1])
        # JUPITER_V6: the route itself; what it does to our balances is checked by simulation
    prio = cu_limit * cu_price // 1_000_000
    if prio > max_priority_lamports:
        raise PolicyError(f"priority fee {prio} lamports is above the cap {max_priority_lamports}")
    return SwapCheck(signer_index=idx, fee_payer=static[0], priority_lamports=prio, creates_atas=creates, opens=opens,
                     writable=writable_keys(msg, keys))


def writable_keys(msg, keys: list[Pubkey]) -> list[Pubkey]:
    """The accounts of ``keys`` (``account_keys`` order) the message may write: the writable signers, the writable
    non-signers, then each lookup table's writable addresses."""
    h = msg.header; ns = len(msg.account_keys)
    n_sign, ro_sign, ro_unsigned = h.num_required_signatures, h.num_readonly_signed_accounts, h.num_readonly_unsigned_accounts
    out = [k for i, k in enumerate(keys[:ns]) if i < n_sign - ro_sign or n_sign <= i < ns - ro_unsigned]
    n_alt_w = sum(len(bytes(lk.writable_indexes)) for lk in getattr(msg, "address_table_lookups", []))
    return out + list(keys[ns:ns + n_alt_w])


def token_amount(data: bytes | None) -> int:
    """SPL token account amount (Token and Token-2022 share the base layout); a closed account holds 0."""
    if not data or len(data) < 72:
        return 0
    return struct.unpack("<Q", bytes(data[64:72]))[0]


def token_mint(data: bytes) -> Pubkey:
    return Pubkey.from_bytes(bytes(data[0:32]))


def token_owner(data: bytes) -> Pubkey:
    """The wallet a token account belongs to (its authority)."""
    return Pubkey.from_bytes(bytes(data[32:64]))


@dataclass
class Snapshot:
    """The trading wallet's lamports and each of its token accounts: address -> (mint, amount, lamports)."""
    lamports: int
    tokens: dict


def rent_opened_for_others(pre_accounts: list[dict | None], post_accounts: list[dict | None]) -> int:
    """Lamports a swap put into token accounts it opened for a program's vault, in the simulation (``pre_accounts`` and
    ``post_accounts`` side by side, as ``Rpc.accounts`` reads them). Pump.fun's AMM opens a fee vault's empty wSOL
    account at the buyer's expense: 34Ct2K, 2026-11-06, 1,488,440 lamports to an account of 5AmqLBHH (a program address),
    and the signer refused the buy twice as an overspend. Counted: an account that did not exist, a token account whose
    authority is a program address (off the curve: no wallet's), holding no tokens and no more lamports than the rent
    exemption of its size; at most ``OPENED_FOR_OTHERS_MAX``. Rent to a wallet, or into an account that receives
    tokens or SOL, is not."""
    rents = []
    for a0, a1 in zip(pre_accounts, post_accounts):
        if (a0 and a0["lamports"]) or not a1 or a1["owner"] not in (TOKEN, TOKEN_2022) or len(a1["data"]) < 72:
            continue
        if token_owner(a1["data"]).is_on_curve() or token_amount(a1["data"]) != 0:
            continue
        if 0 < a1["lamports"] <= (128 + len(a1["data"])) * RENT_PER_BYTE_MAX:
            rents.append(a1["lamports"])
    return sum(sorted(rents)[:OPENED_FOR_OTHERS_MAX])


def check_swap_effects(pre: Snapshot, post: Snapshot, *, in_mint: Pubkey, out_mint: Pubkey, in_amount: int, min_out: int,
                       slack_lamports: int) -> dict:
    """The balance changes a simulation produced, against what was ordered. Returns the deltas; raises PolicyError."""
    if (in_mint == NATIVE_MINT) == (out_mint == NATIVE_MINT):
        raise PolicyError("only SOL <-> token swaps are signed")
    if in_amount <= 0 or min_out <= 0:
        raise PolicyError("an order needs a positive amount and a positive minimum out")
    sol = post.lamports - pre.lamports
    by_mint: dict[Pubkey, int] = {}
    for addr in set(pre.tokens) | set(post.tokens):
        mint = (pre.tokens.get(addr) or post.tokens.get(addr))[0]
        a0 = (pre.tokens.get(addr) or (mint, 0, 0))
        a1 = (post.tokens.get(addr) or (mint, 0, 0))
        if mint == NATIVE_MINT:                               # wrapped SOL counts as SOL: by lamports
            sol += a1[2] - a0[2]
        else:
            by_mint[mint] = by_mint.get(mint, 0) + (a1[1] - a0[1])
            sol += a1[2] - a0[2]                               # rent of token accounts opened or closed
    for mint, d in by_mint.items():
        if d < 0 and mint != in_mint:
            raise PolicyError(f"the swap would reduce another holding ({mint})")
    if in_mint == NATIVE_MINT:                                 # buy
        got = by_mint.get(out_mint, 0)
        if -sol > in_amount + slack_lamports:
            raise PolicyError(f"the swap spends {-sol} lamports, more than the {in_amount} ordered (+{slack_lamports} fees and rent)")
        if got < min_out:
            raise PolicyError(f"the trading wallet would receive {got} of the token, below the minimum {min_out}")
        return {"sol": sol, "token": got}
    spent = -by_mint.get(in_mint, 0)                           # sell
    if spent > in_amount:
        raise PolicyError(f"the swap sells {spent} tokens, more than the {in_amount} ordered")
    if sol < min_out - slack_lamports:
        raise PolicyError(f"the trading wallet would receive {sol} lamports, below the minimum {min_out}")
    return {"sol": sol, "token": -spent}


def check_value(direction: str, sol_moved: int, token_moved: int, token_decimals: int, token_usd: float | None, sol_usd: float | None,
                min_ratio: float) -> float | None:
    """Value out / value in at independent prices (the signer's own Jupiter price query); None if a price is missing.
    Refuses a swap that gives away more than ``1 - min_ratio`` of its value (a self-dealing route into a pool the
    attacker priced)."""
    if not token_usd or not sol_usd or token_moved <= 0 or sol_moved == 0:
        return None
    tok_value = token_moved / 10 ** token_decimals * token_usd
    sol_value = abs(sol_moved) / 1e9 * sol_usd
    ratio = tok_value / sol_value if direction == "buy" else sol_value / tok_value
    if ratio < min_ratio:
        raise PolicyError(f"the swap returns {ratio:.0%} of the value it gives (floor {min_ratio:.0%})")
    return ratio
