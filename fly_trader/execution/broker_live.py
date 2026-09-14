"""Live execution through Jupiter Swap API v2 with balance-verified fills (plan §9).

Per attempt (one in-flight order per process, guarded by a module lock):
  assert_signing_allowed() -> pre = snapshot_balances() -> GET /order (taker = bot, excludeRouters=
  jupiterz) -> assert signatureFeePayer == taker -> orders row (status 'ordered') -> sign only our
  signer slot -> POST /execute -> orders row updated -> poll getSignatureStatuses every 1 s until
  confirmed or getBlockHeight() > lastValidBlockHeight (no other timeout) -> post = snapshot ->
  fill iff the input balance decreased AND the output balance increased -> fills row.

Jupiter's execute status is recorded but never trusted alone: 'Failed' with moved balances is a
fill (verified_by='balance_delta_despite_failed'); 'Success' without movement is a fills row with
verified_by='unverified' plus an events row after a getTransaction check. On execute code -1000
(failed to land) or a slippage error with NO balance movement the order is retried with
slippage + step_bps up to max_slippage_bps; each attempt is its own orders row.

orders.status words: ordered | order_error | sign_error | executed | execute_error | confirmed |
failed_on_chain | expired | unsent.   fills.verified_by: balance_delta |
balance_delta_despite_failed | unverified.   price_sol = |sol delta| / 1e9 / (|token delta| /
10^decimals) — an all-in effective price (transaction fee, priority fee and any ATA rent are inside
the SOL delta; fee_lamports carries the estimate so the ledger must not add it again).
Every write is committed immediately so a crash mid-poll leaves an auditable row for verify-fills.
"""
from __future__ import annotations

import base64
import logging
import re
import threading
import time
from dataclasses import dataclass

from psycopg.types.json import Jsonb
from solders.hash import Hash
from solders.instruction import AccountMeta, Instruction
from solders.message import MessageV0
from solders.pubkey import Pubkey
from solders.transaction import VersionedTransaction

from .. import config
from ..chain.balances import Snapshot, compute_delta, snapshot_balances
from ..chain.cluster_guard import assert_signing_allowed
from ..chain.jupiter_swap import JupiterError, JupiterSwap
from ..chain.keys import load_keypair
from ..chain.rpc import TOKEN_2022_PROGRAM_ID, TOKEN_PROGRAM_ID, HttpSolanaRpc
from ..chain.signing import NotASigner, n_lookup_programs, sign_transaction_b64, static_program_ids, transaction_id
from ..db.apilog import record_event
from ..db.connection import connect, transaction
from ..logging_setup import scrub

log = logging.getLogger(__name__)

_INFLIGHT = threading.Lock()  # one in-flight order at a time (plan §9)

CONFIRM_POLL_S = 1.0
BLOCKHASH_VALIDITY_BLOCKS = 150  # fallback horizon when /order carries no lastValidBlockHeight
CLOSE_ATA_BATCH = 10
SYSTEM_PROGRAM_ID = "11111111111111111111111111111111"
COMPUTE_BUDGET_PROGRAM_ID = "ComputeBudget111111111111111111111111111111"
ASSOCIATED_TOKEN_PROGRAM_ID = "ATokenGPvbdGVxr1b2hvZbsiqW5xWH25efTNsLJA8knL"
JUPITER_PROGRAM_IDS = ("JUP6LkbZbjS1jKKwapdHNy74zcZ3tLUZoi5QNyVTaV4",)
KNOWN_PROGRAM_IDS = frozenset({SYSTEM_PROGRAM_ID, COMPUTE_BUDGET_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID,
                               TOKEN_PROGRAM_ID, TOKEN_2022_PROGRAM_ID, *JUPITER_PROGRAM_IDS})
RETRY_CODES = frozenset({-1000})  # aggregator: failed to land
_SLIPPAGE_RE = re.compile(r"slippage|0x1771|\b6001\b", re.IGNORECASE)  # Jupiter SlippageToleranceExceeded = 6001


@dataclass
class FillResult:
    ok: bool
    order_id: int | None
    fill_id: int | None
    signature: str | None
    status: str | None
    code: int | None
    error: str | None
    token_delta: int
    lamports_delta: int
    price_sol: float | None
    attempts: int


def _int_or_none(v) -> int | None:
    try:
        return int(v) if v is not None else None
    except (TypeError, ValueError):
        return None


def await_confirmation(rpc, signature: str | None, last_valid_block_height: int | None) -> tuple[str, dict | None]:
    """Poll until the signature is confirmed/finalized, fails on chain, or the chain's block height
    passes ``last_valid_block_height`` (the transaction can then never land). Sleeps 1 s between
    polls; there is deliberately no other timeout."""
    if not signature:
        return "unsent", None
    if last_valid_block_height is None:
        last_valid_block_height = rpc.get_block_height() + BLOCKHASH_VALIDITY_BLOCKS
    while True:
        st = (rpc.get_signature_statuses([signature]) or [None])[0]
        if st:
            if st.get("err") is not None:
                return "failed_on_chain", st
            if st.get("confirmationStatus") in ("confirmed", "finalized"):
                return "confirmed", st
        if rpc.get_block_height() > last_valid_block_height:
            return "expired", st
        time.sleep(CONFIRM_POLL_S)


class LiveBroker:
    def __init__(self, rpc=None, jup=None, keypair=None):
        self._rpc = rpc
        self._jup = jup
        self._keypair = keypair
        self._labels: dict[str, str] | None = None

    @property
    def rpc(self):
        if self._rpc is None:
            self._rpc = HttpSolanaRpc()
        return self._rpc

    @property
    def jup(self):
        if self._jup is None:
            self._jup = JupiterSwap()
        return self._jup

    @property
    def keypair(self):
        if self._keypair is None:
            self._keypair = load_keypair()
        return self._keypair

    @property
    def pubkey(self) -> str:
        return str(self.keypair.pubkey())

    @property
    def labels(self) -> dict[str, str]:
        if self._labels is None:
            try:
                self._labels = dict(self.jup.program_id_to_label())
            except Exception as e:
                log.warning("program-id-to-label unavailable: %s", type(e).__name__)
                self._labels = {}
        return self._labels

    # ---- public ----
    def swap(self, conn, *, decision_id: int | None, book: str, side: str, mint: str, amount_in: int,
             slippage_bps: int, max_slippage_bps: int, step_bps: int) -> FillResult:
        """Buy (lamports in -> ``mint``) or sell (raw token units in -> SOL) with balance verification."""
        assert_signing_allowed()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")
        amount_in = int(amount_in)
        if amount_in <= 0:
            raise ValueError("amount_in must be positive")
        if side == "buy":
            input_mint, output_mint = config.WSOL_MINT, mint
        else:
            input_mint, output_mint = mint, config.WSOL_MINT
        with _INFLIGHT:
            slip = int(slippage_bps)
            attempt = 0
            while True:
                attempt += 1
                result, retryable = self._attempt(
                    conn, decision_id=decision_id, book=book, side=side, mint=mint, input_mint=input_mint,
                    output_mint=output_mint, amount_in=amount_in, slippage_bps=slip, attempt=attempt)
                result.attempts = attempt
                if result.ok or not retryable or step_bps <= 0 or slip + step_bps > max_slippage_bps:
                    if retryable and not result.ok:
                        log.warning("order %s not retried: slippage ladder exhausted at %d bps", result.order_id, slip)
                    return result
                slip += int(step_bps)
                log.warning("retrying %s %s after %s (code %s) with slippage %d bps",
                            side, mint, result.status, result.code, slip)

    # ---- one attempt ----
    def _attempt(self, conn, *, decision_id, book, side, mint, input_mint, output_mint, amount_in,
                 slippage_bps, attempt) -> tuple[FillResult, bool]:
        taker = self.pubkey
        pre = snapshot_balances(self.rpc, taker)
        request = {"inputMint": input_mint, "outputMint": output_mint, "amount": str(amount_in), "taker": taker,
                   "slippageBps": slippage_bps, "excludeRouters": "jupiterz"}
        base = dict(decision_id=decision_id, book=book, attempt=attempt, side=side, input_mint=input_mint,
                    output_mint=output_mint, amount_in=amount_in, slippage_bps=slippage_bps, request=request)

        def fail(status, error, code=None, order_response=None, order_id=None):
            if order_id is None:
                order_id = self._insert_order(conn, **base, order_response=order_response, status=status,
                                              error_code=code, error=error)
            else:
                self._update_order(conn, order_id, status=status, error_code=code, error=error)
            log.error("%s %s attempt %d: %s: %s", side, mint, attempt, status, error)
            return FillResult(False, order_id, None, None, status, code, error, 0, 0, None, attempt), False

        # 1. order
        try:
            order = self.jup.order(input_mint, output_mint, amount_in, taker=taker, slippage_bps=slippage_bps)
        except JupiterError as e:
            return fail("order_error", scrub(str(e)))
        err_code = _int_or_none(order.get("errorCode"))
        tx_b64 = order.get("transaction")
        if err_code is not None or not tx_b64:
            msg = order.get("errorMessage") or order.get("error") or "no transaction in /order response"
            return fail("order_error", scrub(str(msg)), err_code, order)
        fee_payer = order.get("signatureFeePayer")
        if fee_payer and fee_payer != taker:
            return fail("order_error", f"signatureFeePayer {fee_payer} != taker", None, order)
        lvbh = _int_or_none(order.get("lastValidBlockHeight"))
        program_ids = static_program_ids(tx_b64)
        unknown = [p for p in program_ids if p not in KNOWN_PROGRAM_IDS and p not in self.labels]
        if unknown:
            log.warning("order references unknown static program ids: %s", unknown)
        n_lookup = n_lookup_programs(tx_b64)
        if n_lookup:
            log.warning("%d instruction(s) use program ids from lookup tables (unresolved in v1)", n_lookup)
        order_id = self._insert_order(
            conn, **base, order_response=order, request_id=order.get("requestId"), router=order.get("router"),
            fee_bps=_int_or_none(order.get("feeBps")), program_ids=program_ids, lvbh=lvbh, status="ordered")

        # 2. sign (our slot only)
        try:
            signed_b64, _idx = sign_transaction_b64(tx_b64, self.keypair)
        except (NotASigner, ValueError) as e:
            return fail("sign_error", f"{type(e).__name__}: {scrub(str(e))}", None, order_id=order_id)
        tx_sig = transaction_id(signed_b64)

        # 3. execute
        exec_req = {"requestId": order.get("requestId"), "lastValidBlockHeight": lvbh, "signedTransaction": signed_b64}
        ex: dict | None
        try:
            ex = self.jup.execute(signed_b64, order.get("requestId"), lvbh)
        except JupiterError as e:
            ex = None
            code = None
            error = scrub(str(e))
            signature = tx_sig  # the transaction may still have been broadcast: poll it anyway
            self._update_order(conn, order_id, execute_request=exec_req, execute_response=None, signature=signature,
                               status="execute_error", error_code=None, error=error)
        else:
            code = _int_or_none(ex.get("code"))
            error = ex.get("error")
            signature = ex.get("signature") or tx_sig
            if ex.get("signature") and ex["signature"] != tx_sig:
                log.warning("execute signature %s differs from our transaction id %s", ex["signature"], tx_sig)
            self._update_order(conn, order_id, execute_request=exec_req, execute_response=ex, signature=signature,
                               status="executed", error_code=code, error=error, latency_ms=ex.get("_latency_ms"))

        # 4. confirm (or expire)
        chain_status, st = await_confirmation(self.rpc, signature, lvbh)
        self._update_order(conn, order_id, status=chain_status)

        # 5. verify by balance deltas
        post = snapshot_balances(self.rpc, taker)
        delta = compute_delta(pre, post)
        lam = int(delta["lamports_delta"])
        tok = int(delta["token_deltas"].get(mint, 0))
        verified = (lam < 0 and tok > 0) if side == "buy" else (tok < 0 and lam > 0)
        jup_status = (ex or {}).get("status")
        slot = _int_or_none((ex or {}).get("slot")) or _int_or_none((st or {}).get("slot"))
        if verified:
            verified_by = "balance_delta" if jup_status == "Success" else "balance_delta_despite_failed"
            decimals = self._decimals(conn, mint, pre, post)
            price = (abs(lam) / config.LAMPORTS_PER_SOL) / (abs(tok) / 10 ** decimals) if decimals is not None else None
            fee = self._fee_lamports(side, lam, amount_in, ex, order)
            fill_id = self._insert_fill(conn, order_id, book, signature, slot, mint, side, tok, lam, price, fee,
                                        order.get("platformFee"), verified_by, pre, post)
            log.info("fill %s: %s %s token_delta=%d lamports_delta=%d price_sol=%s (%s)",
                     fill_id, side, mint, tok, lam, price, verified_by)
            return FillResult(True, order_id, fill_id, signature, chain_status, code, error, tok, lam, price, attempt), False
        if jup_status == "Success" or chain_status == "confirmed":
            tx_info = None
            try:
                tx_info = self.rpc.get_transaction(signature)
            except Exception as e:
                log.warning("getTransaction failed for %s: %s", signature, type(e).__name__)
            meta = (tx_info or {}).get("meta") or {}
            record_event("warning", "broker_live",
                         "fill unverified: execute reported success but balances did not move as expected",
                         {"order_id": order_id, "signature": signature, "side": side, "mint": mint,
                          "lamports_delta": lam, "token_delta": tok, "chain_status": chain_status,
                          "jupiter_status": jup_status, "tx_found": tx_info is not None,
                          "tx_err": meta.get("err"), "tx_fee": meta.get("fee"), "tx_slot": (tx_info or {}).get("slot")})
            fill_id = self._insert_fill(conn, order_id, book, signature, slot, mint, side, tok, lam, None, None,
                                        order.get("platformFee"), "unverified", pre, post)
            return FillResult(False, order_id, fill_id, signature, chain_status, code, error, tok, lam, None, attempt), False
        retryable = (code in RETRY_CODES) or bool(error and _SLIPPAGE_RE.search(str(error)))
        log.error("%s %s attempt %d not filled: chain=%s jupiter=%s code=%s error=%s lamports_delta=%d token_delta=%d",
                  side, mint, attempt, chain_status, jup_status, code, error, lam, tok)
        return FillResult(False, order_id, None, signature, chain_status, code, error, tok, lam, None, attempt), retryable

    # ---- helpers ----
    def _decimals(self, conn, mint: str, pre: Snapshot, post: Snapshot) -> int | None:
        d = post.decimals.get(mint, pre.decimals.get(mint))
        if d is not None:
            return int(d)
        try:
            row = conn.execute("SELECT decimals FROM tokens WHERE mint = %s", (mint,)).fetchone()
            if row and row["decimals"] is not None:
                return int(row["decimals"])
        except Exception:
            conn.rollback()
        try:
            return self.rpc.get_mint_decimals(mint)
        except Exception:
            return None

    @staticmethod
    def _fee_lamports(side: str, lam: int, amount_in: int, ex: dict | None, order: dict) -> int | None:
        """Buy: lamports beyond the swap input (tx fee + priority fee + ATA rent). Sell: SOL the route
        produced that did not reach the wallet. Approximations from execute amounts, else the order's
        fee fields."""
        ex = ex or {}
        if side == "buy":
            total_in = _int_or_none(ex.get("totalInputAmount")) or amount_in
            return abs(lam) - total_in
        total_out = _int_or_none(ex.get("totalOutputAmount"))
        if total_out is not None:
            return total_out - lam
        sig_fee = _int_or_none(order.get("signatureFeeLamports")) or 0
        prio = _int_or_none(order.get("prioritizationFeeLamports")) or 0
        return sig_fee + prio

    def _insert_order(self, conn, *, decision_id, book, attempt, side, input_mint, output_mint, amount_in,
                      slippage_bps, request, order_response=None, request_id=None, router=None, fee_bps=None,
                      program_ids=None, lvbh=None, status, error_code=None, error=None) -> int:
        row = conn.execute(
            "INSERT INTO orders (decision_id, book, attempt, side, input_mint, output_mint, amount_in, slippage_bps, "
            "request, order_response, request_id, router, fee_bps, program_ids, last_valid_block_height, status, "
            "error_code, error) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (decision_id, book, attempt, side, input_mint, output_mint, amount_in, slippage_bps, Jsonb(request),
             Jsonb(order_response) if order_response is not None else None, request_id, router, fee_bps,
             list(program_ids) if program_ids is not None else None, lvbh, status, error_code, error),
        ).fetchone()
        conn.commit()
        return int(row["id"])

    def _update_order(self, conn, order_id: int, **fields) -> None:
        json_cols = {"execute_request", "execute_response", "order_response"}
        cols, vals = [], []
        for k, v in fields.items():
            cols.append(f"{k} = %s")
            vals.append(Jsonb(v) if k in json_cols and v is not None else v)
        conn.execute(f"UPDATE orders SET {', '.join(cols)} WHERE id = %s", (*vals, order_id))
        conn.commit()

    def _insert_fill(self, conn, order_id, book, signature, slot, mint, side, token_delta, sol_delta_lamports,
                     price_sol, fee_lamports, platform_fee, verified_by, pre: Snapshot, post: Snapshot) -> int:
        row = conn.execute(
            "INSERT INTO fills (order_id, book, signature, slot, mint, side, token_delta, sol_delta_lamports, price_sol, "
            "fee_lamports, platform_fee, verified_by, pre_snapshot, post_snapshot) "
            "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id",
            (order_id, book, signature, slot, mint, side, token_delta, sol_delta_lamports, price_sol, fee_lamports,
             Jsonb(platform_fee) if platform_fee is not None else None, verified_by,
             Jsonb(pre.to_json()), Jsonb(post.to_json())),
        ).fetchone()
        conn.commit()
        return int(row["id"])


def swap_smoke(sol: float) -> None:
    """Operator smoke test (plan phase 3.12): SOL -> USDC then USDC -> SOL through the live broker.
    Requires the full live gate (LIVE_ENABLED=1 etc.). Prints both FillResults and the net lamports."""
    assert_signing_allowed()
    lamports = int(round(float(sol) * config.LAMPORTS_PER_SOL))
    broker = LiveBroker()
    print(f"swap-smoke: wallet {broker.pubkey}: {sol} SOL -> USDC -> SOL")
    with connect() as conn:
        r1 = broker.swap(conn, decision_id=None, book="live", side="buy", mint=config.USDC_MINT, amount_in=lamports,
                         slippage_bps=config.SLIPPAGE_ENTRY_BPS, max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS,
                         step_bps=config.SLIPPAGE_STEP_BPS)
        print(f"leg 1 (buy USDC): {r1}")
        net = r1.lamports_delta
        if r1.ok and r1.token_delta > 0:
            r2 = broker.swap(conn, decision_id=None, book="live", side="sell", mint=config.USDC_MINT,
                             amount_in=r1.token_delta, slippage_bps=config.SLIPPAGE_EXIT_BPS,
                             max_slippage_bps=config.SLIPPAGE_FORCED_BPS, step_bps=config.SLIPPAGE_STEP_BPS)
            print(f"leg 2 (sell USDC): {r2}")
            net += r2.lamports_delta
        else:
            print("leg 2 skipped: leg 1 was not a verified fill")
    print(f"net lamports delta: {net} ({net / config.LAMPORTS_PER_SOL:.9f} SOL)")


def close_empty_atas(rpc=None, keypair=None, url: str | None = None, batch: int = CLOSE_ATA_BATCH) -> dict:
    """Close every zero-balance token account owned by the bot (rent back to the wallet).
    SPL Token / Token-2022 CloseAccount = instruction index 9, accounts [account, destination, owner].
    One v0 transaction per ``batch`` accounts; each is recorded as wallet_events(kind='ata_closed')."""
    assert_signing_allowed()
    kp = keypair or load_keypair()
    owner = kp.pubkey()
    owner_s = str(owner)
    rpc = rpc or HttpSolanaRpc()
    empties = [a for a in rpc.get_token_accounts_by_owner(owner_s) if int(a["amount"]) == 0]
    result = {"pubkey": owner_s, "candidates": len(empties), "closed": 0, "signatures": [], "failed": []}
    if not empties:
        return result
    for i in range(0, len(empties), batch):
        chunk = empties[i:i + batch]
        accounts = [a["address"] for a in chunk]
        ixs = [Instruction(Pubkey.from_string(a["program"]), bytes([9]),
                           [AccountMeta(Pubkey.from_string(a["address"]), False, True),
                            AccountMeta(owner, False, True),
                            AccountMeta(owner, True, False)]) for a in chunk]
        bh = rpc.get_latest_blockhash()
        msg = MessageV0.try_compile(owner, ixs, [], Hash.from_string(bh["blockhash"]))
        tx = VersionedTransaction(msg, [kp])
        tx_b64 = base64.b64encode(bytes(tx)).decode()
        try:
            sig = rpc.send_transaction(tx_b64)
        except Exception as e:
            err = f"{type(e).__name__}: {scrub(str(e))}"
            result["failed"].append({"accounts": accounts, "error": err})
            record_event("error", "close_empty_atas", "close transaction rejected", {"accounts": accounts, "error": err})
            continue
        status, _st = await_confirmation(rpc, sig, bh["last_valid_block_height"])
        detail = {"accounts": accounts, "signature": sig, "status": status}
        try:
            with transaction(url) as conn:
                conn.execute("INSERT INTO wallet_events (kind, pubkey, detail) VALUES ('ata_closed', %s, %s)",
                             (owner_s, Jsonb(detail)))
        except Exception as e:
            log.warning("wallet_events insert failed: %s", type(e).__name__)
        if status == "confirmed":
            result["closed"] += len(chunk)
            result["signatures"].append(sig)
        else:
            result["failed"].append(detail)
    log.info("close_empty_atas: closed %d of %d (%d tx)", result["closed"], len(empties), len(result["signatures"]))
    return result
