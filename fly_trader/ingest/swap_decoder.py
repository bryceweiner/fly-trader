"""DEX-agnostic swap decoder from pre/post token balances of a pool's two vaults.

Meteora emits swap events via ``emit_cpi!`` (inner instructions, not logs) and every DEX has its own
borsh layout, so instead of per-program decoding we watch the pool's two vault token accounts and
read their balance deltas out of ``meta.preTokenBalances`` / ``meta.postTokenBalances``. This works
identically for Helius ``transactionSubscribe`` notifications (encoding jsonParsed, transactionDetails
full) and for ``getTransaction`` (jsonParsed) results, which are used as test fixtures.

Vault ownership observed on mainnet (2026-09-12, recent transactions of live pools; pool structs
decoded from ``getAccountInfo`` where noted):

* PumpSwap ``pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA``: ``pool_base_token_account`` /
  ``pool_quote_token_account`` (pool struct offsets 8+1+2+32*4 and +32) are token accounts whose
  ``owner`` is the **pool account itself** (verified on 7 pools by struct decode and on 100+ live
  pools by an on-chain audit; base side is usually Token-2022 ``TokenzQd…``, quote side
  ``Tokenkeg…``). The quote is NOT always SOL: Jupiter's ``graduatedPool`` for ``baton`` is a PumpSwap
  pool quoted in PUMP (``pumpCmXq…``) and one live pool was USDC-quoted, so the quote mint must be
  learned from the balances, not assumed. Trap: arbitrage-bot transactions list the PUMP-quoted pool
  together with a Meteora DLMM baton/WSOL pool (``BN7CfsGm…``) whose vaults look like a perfect
  base+WSOL pair — which is why pool-owned token accounts are treated as definitive (no fallthrough)
  and, when the DEX program is known, only its own instructions may teach vaults.
* Jupiter's ``graduatedPool`` is not always a PumpSwap pool: 6 of ~200 live pump.fun graduations
  (older tokens such as GRIFFAIN) are Raydium v4 pools, so ``program_label`` from discovery cannot
  be trusted for the vault rule; the learner does not depend on it.
* Raydium CPMM ``CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C``: vaults are owned by the global CPMM
  authority ``GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL`` — **not** the pool account.
* Raydium v4 ``675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8``: vaults owned by the AMM authority
  ``5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1``.
* Meteora DLMM ``LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo``: ``reserve_x``/``reserve_y`` are owned by
  the lb_pair (the pool account).
* Meteora DAMM v2 ``cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG``: vaults owned by the global pool
  authority ``HLnpSz9h2S4hiLQ43rnSD9XkcUThA7B8hQMKmDaiTLcC``, which holds *many* pools' vaults; a
  routed transaction therefore shows that owner with 3+ mints.

Because three of the five DEXes use a shared authority, ``learn_vaults`` narrows candidates to the
token accounts referenced by the instruction(s) that also reference the pool (top-level or inner),
then picks the owner that holds exactly one base-mint and one quote-mint account there. The pure
"single non-signer owner with two mints" heuristic from VOC is the final fallback.

Account indices in token balances index ``message.accountKeys``; with jsonParsed encoding that
list already contains the lookup-table-loaded addresses (``source: "lookupTable"``), verified on
v0 transactions with ``accountIndex`` beyond the static keys. For plain ``json`` encoding the
loaded addresses are appended (writable then readonly) as the plan states.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timezone

from .. import config
from .tape import TapeRow

log = logging.getLogger(__name__)

WSOL_MINT, USDC_MINT, USDT_MINT = config.WSOL_MINT, config.USDC_MINT, config.USDT_MINT
QUOTE_MINTS: dict[str, int] = config.QUOTE_MINTS
# Prefer the stable side as quote when both sides are quote mints (SOL/USDC pools).
QUOTE_PRIORITY = {USDC_MINT: 3, USDT_MINT: 2, WSOL_MINT: 1}

# Jupiter /program-id-to-label names (fetched 2026-09-12) for the DEX programs the watch list uses. Reference only:
# capture passes watch_pools.program_id and never infers it from a label (see the Raydium v4 note above).
PROGRAM_IDS_BY_LABEL: dict[str, str] = {
    "Pump.fun Amm": "pAMMBay6oceH9fJKBRHGP5D4bD4sWpmSwMn52FMfXEA",
    "Raydium CP": "CPMMoo8L3F4NbTegBCKVNunggL7H1ZpdTHKxQB5qKP1C",
    "Raydium": "675kPX9MHTjS2zt1qfr1NYHuzeLXfQM9H24wFSUt1Mp8",
    "Meteora DLMM": "LBUZKhRxPF3XUpBCjp4YzTKgLccjZhTSDM9YuVaPwxo",
    "Meteora DAMM v2": "cpamdpZCGKUy5JxQXB4dcpGPiikHawvSWAd6mEn1sGG",
}

KNOWN_VAULT_AUTHORITIES: dict[str, str] = {
    "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1": "Raydium",          # v4 AMM authority
    "GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL": "Raydium CP",       # CPMM authority
    "HLnpSz9h2S4hiLQ43rnSD9XkcUThA7B8hQMKmDaiTLcC": "Meteora DAMM v2",  # pool authority
}


@dataclass(slots=True)
class PoolVaults:
    pool: str
    mint: str                 # base mint
    quote_mint: str
    base_vault: str
    quote_vault: str
    base_decimals: int
    quote_decimals: int
    program_label: str | None = None


@dataclass(slots=True)
class _Balance:
    index: int
    pubkey: str
    mint: str
    owner: str | None
    amount: int
    decimals: int


# ---- shape handling --------------------------------------------------------------------------
def unwrap(tx: dict) -> tuple[dict, dict, str | None, int | None, int | None, int | None]:
    """Return ``(transaction, meta, signature, slot, tx_index, block_time)`` for either shape.

    Accepts a full websocket message (``{"params": {"result": ...}}``), a notification ``result``
    (``{"transaction": {"transaction", "meta"}, "signature", "slot", "transactionIndex"}``) or a
    ``getTransaction`` result (``{"transaction": {"signatures", "message"}, "meta", "slot", ...}``).
    """
    if "params" in tx and isinstance(tx["params"], dict):
        tx = tx["params"].get("result") or {}
    inner = tx.get("transaction") or {}
    if isinstance(inner, dict) and "meta" in inner and "transaction" in inner:
        # transactionSubscribe notification result
        meta = inner.get("meta") or {}
        txn = inner.get("transaction") or {}
        sig = tx.get("signature") or ((txn.get("signatures") or [None])[0])
        return txn, meta, sig, _int(tx.get("slot")), _int(tx.get("transactionIndex")), None
    meta = tx.get("meta") or {}
    txn = inner if isinstance(inner, dict) else {}
    sig = (txn.get("signatures") or [None])[0]
    return txn, meta, sig, _int(tx.get("slot")), _int(tx.get("transactionIndex")), _int(tx.get("blockTime"))


def _int(v) -> int | None:
    try:
        return None if v is None else int(v)
    except (TypeError, ValueError):
        return None


def account_keys(txn: dict, meta: dict) -> tuple[list[str], list[bool]]:
    """Pubkeys in balance-index order and a parallel signer flag list."""
    msg = txn.get("message") or {}
    raw = msg.get("accountKeys") or []
    keys: list[str] = []
    signers: list[bool] = []
    if raw and isinstance(raw[0], dict):
        for k in raw:
            keys.append(k.get("pubkey"))
            signers.append(bool(k.get("signer")))
        return keys, signers
    # plain json encoding: static keys then loaded addresses (writable, readonly)
    n_sig = int(((msg.get("header") or {}).get("numRequiredSignatures")) or 0)
    for i, k in enumerate(raw):
        keys.append(k)
        signers.append(i < n_sig)
    loaded = meta.get("loadedAddresses") or {}
    for k in (loaded.get("writable") or []) + (loaded.get("readonly") or []):
        keys.append(k)
        signers.append(False)
    return keys, signers


def _balances(meta: dict, which: str, keys: list[str]) -> dict[int, _Balance]:
    out: dict[int, _Balance] = {}
    for b in meta.get(which) or []:
        try:
            idx = int(b["accountIndex"])
            ui = b.get("uiTokenAmount") or {}
            amount = int(ui.get("amount") or 0)
            decimals = int(ui.get("decimals") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        pubkey = keys[idx] if 0 <= idx < len(keys) else None
        if pubkey is None:
            continue
        out[idx] = _Balance(idx, pubkey, b.get("mint"), b.get("owner"), amount, decimals)
    return out


def _instruction_scopes(txn: dict, meta: dict, pool: str, program_id: str | None = None) -> list[set[str]]:
    """Account sets of every instruction (top-level and inner) that references ``pool``.

    Smallest first: a router's top-level instruction lists every account of every hop (other pools'
    vaults included), while the DEX's own inner instruction lists just this pool's accounts. When
    ``program_id`` is known, that program's instructions come before all others.
    """
    scopes: list[tuple[int, int, frozenset[str]]] = []
    msg = txn.get("message") or {}
    groups = [msg.get("instructions") or []]
    for grp in meta.get("innerInstructions") or []:
        groups.append(grp.get("instructions") or [])
    seen: set[frozenset[str]] = set()
    for ixs in groups:
        for ix in ixs:
            accts = ix.get("accounts") if isinstance(ix, dict) else None
            if not accts or not isinstance(accts[0], str) or pool not in accts:
                continue
            fs = frozenset(accts)
            if fs in seen:
                continue
            seen.add(fs)
            rank = 0 if (program_id and ix.get("programId") == program_id) else 1
            scopes.append((rank, len(fs), fs))
    scopes.sort(key=lambda t: (t[0], t[1]))
    return [set(fs) for _r, _n, fs in scopes]


def _has_pubkey_instructions(txn: dict, meta: dict) -> bool:
    """True for jsonParsed-style shapes where instructions list account pubkeys (not indices)."""
    msg = txn.get("message") or {}
    groups = [msg.get("instructions") or []]
    for grp in meta.get("innerInstructions") or []:
        groups.append(grp.get("instructions") or [])
    for ixs in groups:
        for ix in ixs:
            accts = ix.get("accounts") if isinstance(ix, dict) else None
            if accts and isinstance(accts[0], str):
                return True
    return False


def _programs_touching(txn: dict, meta: dict, scope: set[str]) -> set[str]:
    """Program ids of the instructions whose account set equals ``scope``."""
    out: set[str] = set()
    msg = txn.get("message") or {}
    groups = [msg.get("instructions") or []]
    for grp in meta.get("innerInstructions") or []:
        groups.append(grp.get("instructions") or [])
    for ixs in groups:
        for ix in ixs:
            accts = ix.get("accounts") if isinstance(ix, dict) else None
            if accts and isinstance(accts[0], str) and set(accts) == scope:
                out.add(ix.get("programId"))
    return out


def first_signer(txn: dict, meta: dict) -> str | None:
    keys, signers = account_keys(txn, meta)
    for k, s in zip(keys, signers):
        if s:
            return k
    return keys[0] if keys else None


# ---- vault learning --------------------------------------------------------------------------
def _pick(entries: list[_Balance], pool: str, mint: str | None, quotes: dict[str, int]):
    """From one owner's token accounts return ``(base, quote)`` balances or None if not exactly one each."""
    quote_accts = [e for e in entries if e.mint in quotes]
    if mint is not None:
        base_accts = [e for e in entries if e.mint == mint]
        quote_accts = [e for e in quote_accts if e.mint != mint]
    else:
        base_accts = [e for e in entries if e.mint not in quotes]
        if not base_accts and len(quote_accts) == 2 and quote_accts[0].mint != quote_accts[1].mint:
            # both sides are quote mints (e.g. SOL/USDC): the lower-priority one is the base
            a, b = sorted(quote_accts, key=lambda e: QUOTE_PRIORITY.get(e.mint, 0))
            base_accts, quote_accts = [a], [b]
    # distinct accounts (the same account can appear in pre and post lists)
    base_accts = list({e.pubkey: e for e in base_accts}.values())
    quote_accts = list({e.pubkey: e for e in quote_accts}.values())
    if len(base_accts) != 1 or len(quote_accts) != 1:
        return None
    if mint is None and len({e.mint for e in entries}) != 2:
        return None
    return base_accts[0], quote_accts[0]


def learn_vaults(tx: dict, pool: str, quote_mints: dict[str, int] | None = None, *,
                 mint: str | None = None, program_label: str | None = None,
                 program_id: str | None = None) -> PoolVaults | None:
    """Identify the pool's base/quote vault token accounts from one transaction touching ``pool``.

    Order of evidence: (1) token accounts owned by the pool address itself (PumpSwap, Meteora DLMM);
    (2) among the token accounts referenced by the instructions that reference the pool, the owner
    holding exactly one base-mint and one quote-mint account (Raydium v4/CPMM and Meteora DAMM v2
    authorities; known authorities win ties); (3) the VOC heuristic: the single non-signer owner
    holding exactly two mints, one of them a quote. ``mint`` (the watch-list base mint) makes the
    match strict; ``quote_mints`` defaults to WSOL/USDC/USDT with their decimals; ``program_id``
    (the DEX program, when the watch list knows it) ranks that program's instructions first.
    Returns None when the transaction is ambiguous — the caller simply tries the next one.
    """
    quotes = dict(QUOTE_MINTS if quote_mints is None else quote_mints)
    txn, meta, _sig, _slot, _idx, _bt = unwrap(tx)
    keys, signer_flags = account_keys(txn, meta)
    if pool not in keys:
        return None
    signers = {k for k, s in zip(keys, signer_flags) if s}
    post = _balances(meta, "postTokenBalances", keys)
    pre = _balances(meta, "preTokenBalances", keys)
    entries: dict[str, _Balance] = {}
    for b in list(pre.values()) + list(post.values()):
        entries[b.pubkey] = b  # post wins
    cands = [e for e in entries.values() if e.owner and e.owner not in signers]
    if not cands:
        return None

    def build(base: _Balance, quote: _Balance) -> PoolVaults:
        return PoolVaults(pool=pool, mint=base.mint, quote_mint=quote.mint, base_vault=base.pubkey,
                          quote_vault=quote.pubkey, base_decimals=base.decimals,
                          quote_decimals=quote.decimals, program_label=program_label)

    # (1) owned by the pool itself. Definitive: when the pool owns token accounts those ARE its vaults,
    # so an unrecognised quote (a PumpSwap pool quoted in PUMP) yields None rather than falling
    # through to some neighbouring pool's accounts (an arbitrage bot lists both pools' vaults).
    own = [e for e in cands if e.owner == pool]
    if own:
        picked = _pick(own, pool, mint, quotes)
        return build(*picked) if picked else None

    # (2) the smallest instruction that references the pool AND carries vault-like token accounts.
    # Only that one scope is judged: a larger (router) scope only adds other pools' vaults, and a
    # pool whose quote is unrecognised (e.g. an EMBER/MET DAMM v2 pool) must yield None, not a
    # neighbouring pool's accounts. When the DEX program is known, only its own instructions
    # count — a transaction that merely lists the pool without the DEX executing teaches nothing.
    scopes = _instruction_scopes(txn, meta, pool, program_id)
    if program_id:
        scopes = [s for s in scopes if program_id in _programs_touching(txn, meta, s)]
    parsed_shape = _has_pubkey_instructions(txn, meta)
    for scope in scopes:
        by_owner: dict[str, list[_Balance]] = {}
        for e in cands:
            if e.pubkey in scope:
                by_owner.setdefault(e.owner, []).append(e)
        informative = any(e.mint == mint for es in by_owner.values() for e in es) if mint else bool(by_owner)
        if not informative:
            continue
        hits = []
        for owner, es in by_owner.items():
            picked = _pick(es, pool, mint, quotes)
            if picked:
                hits.append((owner, picked))
        if len(hits) > 1:
            known = [h for h in hits if h[0] in KNOWN_VAULT_AUTHORITIES]
            hits = known if len(known) == 1 else hits
        return build(*hits[0][1]) if len(hits) == 1 else None
    if parsed_shape:
        return None  # instructions were inspectable and none taught the vaults: do not guess

    # (3) VOC fallback, only when the encoding carries no pubkey instruction lists (plain "json"):
    # one non-signer owner with exactly two mints, one a quote
    by_owner = {}
    for e in cands:
        by_owner.setdefault(e.owner, []).append(e)
    hits = []
    for owner, es in by_owner.items():
        mints = {e.mint for e in es}
        if len(mints) != 2 or not (mints & set(quotes)):
            continue
        if mint is not None and mint not in mints:
            continue
        picked = _pick(es, pool, mint, quotes)
        if picked:
            hits.append((owner, picked))
    if len(hits) > 1:
        known = [h for h in hits if h[0] in KNOWN_VAULT_AUTHORITIES]
        hits = known if len(known) == 1 else hits
    if len(hits) == 1:
        return build(*hits[0][1])
    return None


# ---- decoding --------------------------------------------------------------------------------
def build_vault_index(pools: dict[str, PoolVaults]) -> dict[str, PoolVaults]:
    """vault pubkey -> PoolVaults, for O(#token balances) lookups per transaction."""
    idx: dict[str, PoolVaults] = {}
    for pv in pools.values():
        if pv.base_vault and pv.quote_vault:
            idx[pv.base_vault] = pv
            idx[pv.quote_vault] = pv
    return idx


def decode_transaction(tx: dict, pools: dict[str, PoolVaults], *, vault_index: dict[str, PoolVaults] | None = None,
                       ts: datetime | None = None) -> list[TapeRow]:
    """One TapeRow per watched pool whose vaults changed in ``tx``. Failed transactions yield nothing.

    Δq > 0 and Δb < 0 → buy (+1); Δq < 0 and Δb > 0 → sell (−1); anything else with a non-zero delta
    (same sign, or one side only) → lp/other (0) with absolute amounts. ``price_quote`` is quote per
    base in the pool's quote units; ``price_sol`` is the same number only when the quote is WSOL.
    ``ts`` is the receipt time for websocket rows (notifications carry no blockTime); RPC results use
    their ``blockTime``.
    """
    txn, meta, sig, slot, tx_index, block_time = unwrap(tx)
    if meta.get("err") is not None:
        return []
    keys, signer_flags = account_keys(txn, meta)
    if not keys:
        return []
    index = vault_index if vault_index is not None else build_vault_index(pools)
    post = _balances(meta, "postTokenBalances", keys)
    pre = _balances(meta, "preTokenBalances", keys)
    touched: dict[str, PoolVaults] = {}
    for b in list(post.values()) + list(pre.values()):
        pv = index.get(b.pubkey)
        if pv is not None and pv.pool in pools:
            touched[pv.pool] = pv
    if not touched:
        return []
    by_key_post = {b.pubkey: b for b in post.values()}
    by_key_pre = {b.pubkey: b for b in pre.values()}
    signer = None
    for k, s in zip(keys, signer_flags):
        if s:
            signer = k
            break
    if signer is None:
        signer = keys[0]
    if block_time is not None:
        row_ts = datetime.fromtimestamp(block_time, tz=timezone.utc)
    else:
        row_ts = ts or datetime.now(timezone.utc)
    rows: list[TapeRow] = []
    for pool, pv in touched.items():
        b_post, b_pre = by_key_post.get(pv.base_vault), by_key_pre.get(pv.base_vault)
        q_post, q_pre = by_key_post.get(pv.quote_vault), by_key_pre.get(pv.quote_vault)
        if b_post is None and b_pre is None and q_post is None and q_pre is None:
            continue
        res_base = b_post.amount if b_post else 0
        res_quote = q_post.amount if q_post else 0
        d_base = res_base - (b_pre.amount if b_pre else 0)
        d_quote = res_quote - (q_pre.amount if q_pre else 0)
        if d_base == 0 and d_quote == 0:
            continue
        if d_quote > 0 and d_base < 0:
            side = 1
        elif d_quote < 0 and d_base > 0:
            side = -1
        else:
            side = 0
        amount_base, amount_quote = abs(d_base), abs(d_quote)
        bdec = b_post.decimals if b_post else (b_pre.decimals if b_pre else pv.base_decimals)
        qdec = q_post.decimals if q_post else (q_pre.decimals if q_pre else pv.quote_decimals)
        price_quote = None
        if amount_base > 0 and amount_quote > 0:
            price_quote = (amount_quote / 10 ** qdec) / (amount_base / 10 ** bdec)
        price_sol = price_quote if pv.quote_mint == WSOL_MINT else None
        rows.append(TapeRow(
            ts=row_ts, slot=slot or 0, sig=sig or "", tx_index=tx_index if tx_index is not None else -1,
            pool=pool, mint=pv.mint, side=side, amount_base=amount_base, amount_quote=amount_quote,
            price_sol=price_sol, signer=signer, res_base=res_base, res_quote=res_quote,
            program_label=pv.program_label, price_quote=price_quote, quote_mint=pv.quote_mint,
        ))
    return rows
