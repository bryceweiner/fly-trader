"""The RH bot wallet: nonces, fees, signing, persist-before-broadcast, receipts, exact ERC-20 approvals.

Rules (docs: plan "Accounting / Idempotency"):
- one transaction in flight per wallet (a lock around send→receipt), so a native-ETH balance delta belongs to exactly
  one transaction;
- the signed transaction (nonce, raw bytes, hash) is written to ``rh_txs`` and committed BEFORE it is broadcast, so a
  crash after broadcast can always be recovered (rebroadcast the same bytes, or cancel at that nonce);
- nonce = max(the chain's pending count, the highest nonce we signed that is not dropped/replaced + 1);
- fee = 2 × base fee + tip, never above ``RH_MAX_FEE_GWEI`` per gas (a spike refuses rather than overpays);
- approvals are exact (never 2**256-1), skipped when the allowance already covers the amount, reset to 0 first only for a
  token that refuses to change a non-zero allowance.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time

from .. import config
from ..db.connection import transaction
from ..vault.evm import EvmRpcError
from . import abi, guard
from .tx import Eip1559Tx, sign

log = logging.getLogger(__name__)

GAS_MARGIN_NUM, GAS_MARGIN_DEN, GAS_MARGIN_ADD = 13, 10, 10_000     # estimate × 1.3 + 10k: Orbit estimates include the L1 part
RECEIPT_POLL_S = 1.0
MAX_UINT = 2 ** 256 - 1


class FeeTooHigh(RuntimeError):
    pass


class WalletLock:
    """One RH address, one transaction in flight — across threads (trading intents, vault ETH payouts) and processes
    (CLI tools): a re-entrant thread lock plus a Postgres advisory lock held while the outermost holder has it. The
    executor's balance deltas are only that intent's because nothing else can send from the address meanwhile."""

    def __init__(self, address: str):
        self.key = int.from_bytes(hashlib.sha256(("rh-wallet:" + address.lower()).encode()).digest()[:8], "big", signed=True)
        self._t = threading.RLock(); self._depth = 0; self._conn = None

    def acquire(self, blocking: bool = True, timeout: float = -1) -> bool:
        if not self._t.acquire(blocking, timeout):
            return False
        if self._depth == 0:
            from ..db.connection import connect
            conn = connect(autocommit=True)
            deadline = None if timeout is None or timeout < 0 else time.monotonic() + timeout
            while not conn.execute("SELECT pg_try_advisory_lock(%s) AS ok", (self.key,)).fetchone()["ok"]:
                if not blocking or (deadline is not None and time.monotonic() > deadline):
                    conn.close(); self._t.release()
                    return False
                time.sleep(0.2)
            self._conn = conn
        self._depth += 1
        return True

    def release(self) -> None:
        self._depth -= 1
        if self._depth == 0 and self._conn is not None:
            try:
                self._conn.execute("SELECT pg_advisory_unlock(%s)", (self.key,))
            finally:
                self._conn.close(); self._conn = None
        self._t.release()

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


_LOCKS: dict[str, WalletLock] = {}
_LOCKS_MU = threading.Lock()


def wallet_lock(address: str) -> WalletLock:
    with _LOCKS_MU:
        return _LOCKS.setdefault(address.lower(), WalletLock(address))


class SendFailed(RuntimeError):
    pass


def _calldata(d) -> bytes:
    if isinstance(d, (bytes, bytearray)):
        return bytes(d)
    return bytes.fromhex(d[2:] if d.startswith("0x") else d)


class RhWallet:
    def __init__(self, rpc, key: bytes | None = None, address: str | None = None):
        self.rpc = rpc
        if key is None:
            key, address = guard.check(rpc)
        self._key = key
        self.address = (address or "").lower()
        self.lock = wallet_lock(self.address)

    # ---- chain reads
    def balance(self) -> int:
        return self.rpc.get_balance(self.address)

    def balance_of(self, token: str, owner: str | None = None, block: str = "latest") -> int:
        return abi.decode(["uint256"], bytes.fromhex(self.rpc.eth_call(token, "0x" + abi.encode_call("balanceOf(address)", owner or self.address).hex(), block)[2:]))[0]

    def allowance(self, token: str, spender: str, owner: str | None = None) -> int:
        data = abi.encode_call("allowance(address,address)", owner or self.address, spender)
        return abi.decode(["uint256"], bytes.fromhex(self.rpc.eth_call(token, "0x" + data.hex())[2:]))[0]

    def decimals(self, token: str) -> int:
        return abi.decode(["uint8"], bytes.fromhex(self.rpc.eth_call(token, "0x" + abi.selector("decimals()").hex())[2:]))[0]

    # ---- fees, gas, nonces
    def fees(self) -> tuple[int, int]:
        """(max fee, tip) per gas in wei."""
        cap = int(config.RH_MAX_FEE_GWEI * 1e9)
        base = self.rpc.base_fee(); tip = self.rpc.max_priority_fee()
        if base + tip > cap:
            raise FeeTooHigh(f"base fee {base / 1e9:.3f} gwei + tip over the {config.RH_MAX_FEE_GWEI} gwei cap")
        mx = min(2 * base + tip, cap)
        return mx, min(tip, mx)

    def estimate(self, to: str, data: bytes, value: int = 0) -> int:
        g = self.rpc.estimate_gas({"from": self.address, "to": to, "data": "0x" + data.hex(), "value": hex(value)})
        return g * GAS_MARGIN_NUM // GAS_MARGIN_DEN + GAS_MARGIN_ADD

    def simulate(self, to: str, data: bytes, value: int = 0, block: str = "latest") -> str:
        """eth_call as this wallet; raises EvmRpcError on a revert (checked before anything is signed)."""
        return self.rpc.call("eth_call", [{"from": self.address, "to": to, "data": "0x" + data.hex(), "value": hex(value)}, block])

    def next_nonce(self, conn) -> int:
        chain = self.rpc.tx_count(self.address, "pending")
        r = conn.execute("SELECT max(nonce) AS n FROM rh_txs WHERE from_addr = %s AND status NOT IN ('dropped', 'replaced')", (self.address,)).fetchone()
        return max(chain, (int(r["n"]) + 1) if r and r["n"] is not None else 0)

    # ---- send
    def send(self, *, to: str, data=b"", value: int = 0, kind: str, intent_id: int | None = None, position_id: int | None = None,
             gas: int | None = None, nonce: int | None = None, fees: tuple[int, int] | None = None, router: str | None = None,
             quote: dict | None = None, build: dict | None = None, on_persist=None) -> dict:
        """Sign, persist, broadcast. Returns {id, hash, nonce, gas, max_fee}. Raises SendFailed (the row is marked dropped,
        so its nonce is reused) or FeeTooHigh (nothing signed). ``on_persist(conn, tx_id)`` runs inside the transaction
        that persists the signed row, so a caller's own link to it (a claim's payout) is committed before broadcast too."""
        data = _calldata(data)
        with self.lock:
            gas = gas or self.estimate(to, data, value)
            mx, tip = fees or self.fees()
            with transaction() as conn:
                n = self.next_nonce(conn) if nonce is None else nonce
                stx = sign(Eip1559Tx(chain_id=self.rpc.chain_id, nonce=n, max_priority_fee_per_gas=tip, max_fee_per_gas=mx, gas=gas,
                                     to=to.lower(), value=value, data=data), self._key)
                row = conn.execute(
                    "INSERT INTO rh_txs (intent_id, position_id, kind, from_addr, nonce, hash, raw, to_addr, value_wei, data_sha256, gas_limit, "
                    "max_fee_wei, max_prio_wei, router, quote, build) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) "
                    # the same fields re-signed after a failed broadcast give the same hash: revive that row, never a second one
                    "ON CONFLICT (hash) DO UPDATE SET status = CASE WHEN rh_txs.status = 'dropped' THEN 'signed' ELSE rh_txs.status END, "
                    "error = NULL, updated_at = now() RETURNING id",
                    (intent_id, position_id, kind, self.address, n, stx.hash, stx.raw_hex, to.lower(), value, hashlib.sha256(data).hexdigest(), gas,
                     mx, tip, router, json.dumps(quote) if quote else None, json.dumps(build) if build else None)).fetchone()
                if on_persist is not None:
                    on_persist(conn, int(row["id"]))
            tid = int(row["id"])                                              # committed: a crash from here on is recoverable
            try:
                self.rpc.send_raw(stx.raw_hex)
            except EvmRpcError as e:
                msg = (e.message or "").lower()
                if "already known" in msg or "known transaction" in msg:
                    pass                                                      # the node has it: same hash, carry on
                else:
                    with transaction() as conn:
                        conn.execute("UPDATE rh_txs SET status = 'dropped', error = %s, updated_at = now() WHERE id = %s", (str(e)[:300], tid))
                    raise SendFailed(str(e)) from None
            with transaction() as conn:
                conn.execute("UPDATE rh_txs SET status = 'sent', updated_at = now() WHERE id = %s AND status = 'signed'", (tid,))
            return {"id": tid, "hash": stx.hash, "nonce": n, "gas": gas, "max_fee": mx}

    def wait(self, tx_hash: str, timeout_s: float = 120.0, confirmations: int | None = None) -> dict | None:
        """The receipt once it has ``confirmations`` blocks on top (``RH_CONFIRMATIONS``), or None at the timeout."""
        conf = config.RH_CONFIRMATIONS if confirmations is None else confirmations
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            rc = self.rpc.receipt(tx_hash)
            if rc is not None and rc.get("blockNumber"):
                if self.rpc.block_number("latest") - int(rc["blockNumber"], 16) >= conf:
                    return rc
            time.sleep(RECEIPT_POLL_S)
        return None

    @staticmethod
    def record_receipt(conn, tx_id: int, rc: dict) -> dict:
        """Store a receipt's outcome on its rh_txs row. Returns {ok, fee_wei, gas_used, block}."""
        gas_used = int(rc["gasUsed"], 16); price = int(rc.get("effectiveGasPrice") or "0x0", 16); fee = gas_used * price
        ok = int(rc.get("status", "0x0"), 16) == 1
        conn.execute("UPDATE rh_txs SET status = %s, block = %s, block_hash = %s, gas_used = %s, eff_gas_price_wei = %s, fee_wei = %s, "
                     "fee_at = COALESCE(fee_at, clock_timestamp()), updated_at = now() "
                     "WHERE id = %s", ("mined_ok" if ok else "reverted", int(rc["blockNumber"], 16), rc.get("blockHash"), gas_used, price, fee, tx_id))
        return {"ok": ok, "fee_wei": fee, "gas_used": gas_used, "block": int(rc["blockNumber"], 16)}

    def send_and_wait(self, timeout_s: float = 120.0, **kw) -> dict:
        """``send`` then ``wait`` then ``record_receipt``. Returns the send dict plus {ok, fee_wei, gas_used, block, receipt};
        ``receipt`` None means no receipt yet (the row stays 'sent' for recovery)."""
        with self.lock:
            s = self.send(**kw)
            rc = self.wait(s["hash"], timeout_s)
            if rc is None:
                return {**s, "ok": None, "receipt": None}
            with transaction() as conn:
                out = self.record_receipt(conn, s["id"], rc)
            return {**s, **out, "receipt": rc}

    def cancel(self, nonce: int, prev_max_fee: int) -> dict:
        """Replace whatever sits at ``nonce`` with a 0-value self-transfer at ≥ 1.25 × its fee (the node's replacement rule)."""
        mx, tip = self.fees()
        mx = max(mx, prev_max_fee * 5 // 4 + 1); tip = min(max(tip, 1), mx)
        return self.send(to=self.address, data=b"", value=0, kind="cancel", gas=21_000, nonce=nonce, fees=(mx, tip))

    # ---- ERC-20 approvals: exact, never unlimited
    def approve_exact(self, token: str, spender: str, amount: int, *, intent_id: int | None = None, position_id: int | None = None) -> dict | None:
        """Make ``spender``'s allowance at least ``amount`` with an approval of exactly ``amount``. None if it already is."""
        if amount <= 0 or amount >= MAX_UINT // 2:
            raise ValueError(f"refusing an approval of {amount}: exact amounts only, never unlimited")
        with self.lock:
            cur = self.allowance(token, spender)
            if cur >= amount:
                return None
            data = abi.encode_call("approve(address,uint256)", spender, amount)
            if cur > 0:
                try:
                    self.simulate(token, data)
                except EvmRpcError:                                           # a token that refuses non-zero → non-zero changes
                    z = self.send_and_wait(to=token, data=abi.encode_call("approve(address,uint256)", spender, 0), kind="approve",
                                           intent_id=intent_id, position_id=position_id)
                    if not z.get("ok"):
                        raise SendFailed(f"resetting the allowance of {token} failed")
            r = self.send_and_wait(to=token, data=data, kind="approve", intent_id=intent_id, position_id=position_id)
            if not r.get("ok"):
                raise SendFailed(f"approve {token} → {spender} {'reverted' if r.get('ok') is False else 'has no receipt'}")
            return r
