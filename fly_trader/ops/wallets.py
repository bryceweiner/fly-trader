"""The bot's two memecoin wallets as the console shows them (System → Safety & wallet): each chain's address, native
balance and open live positions, in the chain's coin and in USD (SOL from Jupiter, ETH from a KyberSwap quote into USDG:
vault/prices.py), their withdrawals, and the withdrawal itself (chain/withdraw.py, rh/withdraw.py)."""
from __future__ import annotations

from datetime import datetime, timezone

from .. import config
from ..chain import withdraw as sol_withdraw
from ..chain.cluster_guard import SigningRefused
from ..chain.keys import WalletKeyError
from ..db.connection import transaction
from ..logging_setup import scrub
from ..markets import MARKETS
from ..rh import guard as rh_guard, withdraw as rh_withdraw
from ..rh.wallet import FeeTooHigh, SendFailed
from ..vault import prices

# a withdrawal that raises one of these sent nothing
REFUSALS = (sol_withdraw.WithdrawRefused, SigningRefused, WalletKeyError, rh_withdraw.WithdrawRefused, rh_guard.SigningRefused, FeeTooHigh, SendFailed)
SETTINGS = {"sol": "FUNDING_ADDRESSES", "rh": "RH_FUNDING_ADDRESSES"}


def explorer(chain: str, kind: str, value: str) -> str:
    """The block explorer's page of an address (``kind`` 'address') or a transaction ('tx')."""
    if chain == "sol":
        return f"https://solscan.io/{'account' if kind == 'address' else 'tx'}/{value}"
    base = "https://explorer.testnet.chain.robinhood.com" if config.RH_TESTNET else "https://robinhoodchain.blockscout.com"
    return f"{base}/{kind}/{value}"


def address(chain: str) -> str | None:
    """The bot wallet's address: Solana's from its key (else the last one created), RH's pinned RH_BOT_ADDRESS."""
    if chain == "rh":
        return (config.RH_BOT_ADDRESS or "").lower() or None
    from ..chain.keys import bot_pubkey
    try:
        return bot_pubkey()
    except WalletKeyError:
        with transaction() as conn:
            r = conn.execute("SELECT pubkey FROM wallet_events WHERE kind = 'created' ORDER BY id DESC LIMIT 1").fetchone()
        return r["pubkey"] if r else None


def _positions(book: str) -> dict:
    """The live book's open positions, valued at its last wealth mark (net of exit cost), or at cost before any mark."""
    with transaction() as conn:
        r = conn.execute("SELECT count(*) AS n, COALESCE(sum(cost_sol), 0) AS cost FROM positions WHERE book = %s AND status = 'open'", (book,)).fetchone()
        m = conn.execute("SELECT ts, positions_value FROM wealth_marks WHERE book = %s ORDER BY ts DESC LIMIT 1", (book,)).fetchone() if r["n"] else None
    if m is None or m["positions_value"] is None:
        return {"n_open": int(r["n"]), "positions": float(r["cost"]), "marked_at": None}
    return {"n_open": int(r["n"]), "positions": float(m["positions_value"]), "marked_at": m["ts"]}


def summary(chain: str) -> dict:
    """One chain's bot wallet: {chain, unit, address, native, n_open, positions, marked_at, price_usd, price_at, native_usd,
    positions_usd, total_usd, error}. Amounts in the chain's coin; a USD figure is None without a price. Never raises: a
    failed read is ``error``."""
    spec = MARKETS[chain]
    out = {"chain": chain, "unit": spec.unit, "address": None, "native": None, "error": None, **_positions(spec.live_book)}
    try:
        out["address"] = address(chain)
        if out["address"] and chain == "sol":
            from ..chain.rpc import HttpSolanaRpc
            out["native"] = HttpSolanaRpc().get_balance(out["address"]) / config.LAMPORTS_PER_SOL
        elif out["address"]:
            from ..rh.rpc import RhRpc
            out["native"] = RhRpc().get_balance(out["address"]) / rh_withdraw.WEI
    except Exception as e:                                              # noqa: BLE001 — the page shows it, the rest still renders
        out["error"] = f"{type(e).__name__}: {scrub(str(e))[:200]}"
    px, at = prices.sol_usd() if chain == "sol" else prices.eth_usd()
    usd = (lambda v: v * px if px and v is not None else None)
    out.update(price_usd=px or None, price_at=datetime.fromtimestamp(at, timezone.utc) if at else None, native_usd=usd(out["native"]),
               positions_usd=usd(out["positions"]), total_usd=usd(None if out["native"] is None else out["native"] + out["positions"]))
    return out


def destinations(chain: str) -> list[str]:
    return sol_withdraw.destinations() if chain == "sol" else rh_withdraw.destinations()


def withdraw_blockers(chain: str) -> list[str]:
    """Why the console cannot withdraw from ``chain``'s bot wallet (empty: it can). Reads no chain."""
    out = []
    if config.VAULT_ENABLED:
        out.append("the vault fly's money leaves through its treasury and claims, not from the console")
    if chain == "sol":
        if config.SOLANA_CLUSTER != "mainnet-beta":
            out.append(f"SOLANA_CLUSTER is {config.SOLANA_CLUSTER}, the bot wallet is on mainnet-beta")
        from ..chain.keys import key_available
        if not key_available():
            out.append("no bot key in this process (BOT_PRIVATE_KEY)")
    else:
        try:
            rh_guard.pinned_key()
        except rh_guard.SigningRefused as e:
            out.append(f"no usable RH key: {e}")
    if not destinations(chain):
        out.append(f"list your own wallet in {SETTINGS[chain]} (.env), then restart the console")
    return out


def limits(chain: str, to: str) -> dict:
    """Fresh chain reads for a withdrawal to ``to``: the balance, the fee, what must stay and the most that can go out."""
    if chain == "sol":
        from ..chain.keys import bot_pubkey
        from ..chain.rpc import HttpSolanaRpc
        return sol_withdraw.limits(HttpSolanaRpc(), bot_pubkey(), to)
    return rh_withdraw.limits(rh_withdraw.wallet(to), to.lower())


def plan(chain: str, lim: dict, amount: float | None) -> tuple[int, str | None]:
    """(base units to send, why not or None) for ``amount`` of the coin, or everything withdrawable when None."""
    return (sol_withdraw if chain == "sol" else rh_withdraw).plan(lim, amount)


def withdraw(chain: str, amount: float | None, to: str) -> dict:
    """Send ``amount`` of the chain's coin (None: everything withdrawable) to ``to``, one of your own wallets. Returns
    {status, tx, amount, to}: status 'confirmed', 'failed' (only the fee moved), 'expired' (nothing moved) or 'pending'
    (sent, no receipt yet). Raises one of ``REFUSALS`` when nothing was sent."""
    if chain == "sol":
        r = sol_withdraw.withdraw(amount, to)
        return {"status": {"failed_on_chain": "failed"}.get(r["status"], r["status"]), "tx": r["signature"],
                "amount": r["lamports"] / config.LAMPORTS_PER_SOL, "to": to}
    r = rh_withdraw.withdraw(amount, to)
    return {"status": {"mined_ok": "confirmed", "reverted": "failed"}.get(r["status"], r["status"]), "tx": r["hash"],
            "amount": r["wei"] / rh_withdraw.WEI, "to": r["to"]}


def history(chain: str, addr: str | None, limit: int = 20) -> list[dict]:
    """The wallet's latest withdrawals: {when, amount, to, status, tx}."""
    if not addr:
        return []
    with transaction() as conn:
        if chain == "sol":
            rows = conn.execute("SELECT ts AS \"when\", (detail->>'sol')::float AS amount, detail->>'to' AS \"to\", detail->>'status' AS status, "
                                "detail->>'signature' AS tx FROM wallet_events WHERE kind = 'withdrawal' AND pubkey = %s ORDER BY id DESC LIMIT %s",
                                (addr, limit)).fetchall()
        else:
            rows = conn.execute("SELECT created_at AS \"when\", value_wei / 1e18 AS amount, to_addr AS \"to\", status, hash AS tx FROM rh_txs "
                                "WHERE kind = 'transfer' AND from_addr = %s ORDER BY id DESC LIMIT %s", (addr.lower(), limit)).fetchall()
    return [dict(r) for r in rows]
