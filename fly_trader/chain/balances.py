"""Wallet balance snapshots and deltas (pattern: flywheel utils/balance_tracker.py:41-80).

A fill is verified ONLY by comparing a snapshot taken before the order with one taken after
confirmation (plan §9). SOL is tracked as lamports of the wallet itself; SPL balances are raw
integer amounts summed per mint across all accounts under both token programs (Jupiter wraps and
unwraps WSOL inside the transaction, so a SOL leg shows up as a lamport delta, not a WSOL delta).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from .. import config


@dataclass
class Snapshot:
    lamports: int
    tokens: dict[str, int]
    ts: datetime
    decimals: dict[str, int] = field(default_factory=dict)

    def token(self, mint: str) -> int:
        return int(self.tokens.get(mint, 0))

    def to_json(self) -> dict:
        return {"lamports": int(self.lamports), "tokens": {m: int(a) for m, a in self.tokens.items()},
                "decimals": {m: int(d) for m, d in self.decimals.items()}, "ts": self.ts.isoformat()}


def snapshot_balances(rpc, pubkey) -> Snapshot:
    lamports = int(rpc.get_balance(str(pubkey)))
    tokens: dict[str, int] = {}
    decimals: dict[str, int] = {}
    for acct in rpc.get_token_accounts_by_owner(str(pubkey)):
        mint = acct["mint"]
        tokens[mint] = tokens.get(mint, 0) + int(acct["amount"])
        if acct.get("decimals") is not None:
            decimals[mint] = int(acct["decimals"])
    return Snapshot(lamports=lamports, tokens=tokens, ts=config.utcnow(), decimals=decimals)


def compute_delta(pre: Snapshot, post: Snapshot) -> dict:
    """``{"lamports_delta": int, "token_deltas": {mint: post - pre}}`` over every mint seen in either snapshot."""
    mints = set(pre.tokens) | set(post.tokens)
    return {
        "lamports_delta": int(post.lamports) - int(pre.lamports),
        "token_deltas": {m: int(post.tokens.get(m, 0)) - int(pre.tokens.get(m, 0)) for m in sorted(mints)},
    }


def tx_deltas(tx: dict, owner: str, mint: str) -> tuple[int, int] | None:
    """(lamports, raw tokens of ``mint``) the transaction moved for ``owner``, wSOL counted as lamports. Unlike two
    balance snapshots this ignores anything else that landed in between (a deposit, a claim payout). None if unreadable."""
    try:
        keys = tx["transaction"]["message"]["accountKeys"]
        meta = tx["meta"]
        idx = next(i for i, k in enumerate(keys) if (k.get("pubkey") if isinstance(k, dict) else str(k)) == owner)
        lam = int(meta["postBalances"][idx]) - int(meta["preBalances"][idx])

        def tok(rows, m):
            return sum(int(r["uiTokenAmount"]["amount"]) for r in rows or [] if r.get("owner") == owner and r.get("mint") == m)
        pre, post = meta.get("preTokenBalances"), meta.get("postTokenBalances")
        lam += tok(post, config.WSOL_MINT) - tok(pre, config.WSOL_MINT)
        return lam, tok(post, mint) - tok(pre, mint)
    except (KeyError, TypeError, ValueError, StopIteration):
        return None
