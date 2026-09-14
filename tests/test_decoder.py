"""swap_decoder against real mainnet transactions (tests/fixtures, fetched 2026-09-12 via Helius
getTransaction jsonParsed and trimmed to the fields the decoder reads).

Expectations in fixtures/manifest.json were computed independently of the decoder: vault addresses
come from the pool accounts themselves (PumpSwap pool struct / byte search of the pool data) and the
amounts from the raw pre/post balances of those vault accounts.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from fly_trader.ingest import swap_decoder as sd
from fly_trader.ingest.swap_decoder import PoolVaults
from fly_trader.ingest.tape import COLUMNS, TapeRow

FX = Path(__file__).parent / "fixtures"
MANIFEST = json.loads((FX / "manifest.json").read_text())
SWAP_FIXTURES = ["pumpswap_buy", "pumpswap_sell", "raydium_cpmm_sell", "raydium_cpmm_buy",
                 "raydium_v4_buy", "raydium_v4_sell", "meteora_dlmm_buy"]


def load(name: str) -> dict:
    return json.loads((FX / MANIFEST["fixtures"][name]["file"]).read_text())


def pool_of(name: str) -> dict:
    return MANIFEST["pools"][MANIFEST["fixtures"][name]["pool_key"]]


def expect(name: str) -> dict:
    return MANIFEST["fixtures"][name]["expect"]


def vaults_of(name: str) -> PoolVaults:
    p = pool_of(name)
    return PoolVaults(pool=p["pool"], mint=p["mint"], quote_mint=p["quote_mint"], base_vault=p["base_vault"],
                      quote_vault=p["quote_vault"], base_decimals=p["base_decimals"],
                      quote_decimals=p["quote_decimals"], program_label=p["program_label"])


def all_pools() -> dict[str, PoolVaults]:
    out = {}
    for key, p in MANIFEST["pools"].items():
        out[p["pool"]] = PoolVaults(pool=p["pool"], mint=p["mint"], quote_mint=p["quote_mint"],
                                    base_vault=p["base_vault"], quote_vault=p["quote_vault"],
                                    base_decimals=p["base_decimals"], quote_decimals=p["quote_decimals"],
                                    program_label=p["program_label"])
    return out


def as_notification(tx: dict, subscription: int = 7) -> dict:
    """Wrap a getTransaction result in the transactionSubscribe envelope (shape verified live)."""
    return {"jsonrpc": "2.0", "method": "transactionNotification", "params": {
        "subscription": subscription,
        "result": {"transaction": {"transaction": tx["transaction"], "meta": tx["meta"], "version": tx.get("version")},
                   "signature": tx["transaction"]["signatures"][0], "slot": tx["slot"],
                   "transactionIndex": tx.get("transactionIndex")}}}


# ---- vault learning ---------------------------------------------------------------------------
@pytest.mark.parametrize("name", SWAP_FIXTURES + ["meteora_dlmm_lp", "pumpswap_failed"])
def test_learn_vaults_matches_pool_account(name):
    p = pool_of(name)
    pv = sd.learn_vaults(load(name), p["pool"], mint=p["mint"], program_id=p["program_id"],
                         program_label=p["program_label"])
    assert pv is not None, name
    assert (pv.base_vault, pv.quote_vault) == (p["base_vault"], p["quote_vault"])
    assert pv.mint == p["mint"]
    assert pv.quote_mint == p["quote_mint"]
    assert (pv.base_decimals, pv.quote_decimals) == (p["base_decimals"], p["quote_decimals"])
    assert pv.program_label == p["program_label"]


@pytest.mark.parametrize("name", ["pumpswap_buy", "raydium_cpmm_sell", "raydium_v4_buy", "meteora_dlmm_buy"])
def test_learn_vaults_without_hints(name):
    """The plan's minimal call: only the pool address. Base mint is inferred as the non-quote side."""
    p = pool_of(name)
    pv = sd.learn_vaults(load(name), p["pool"])
    assert pv is not None
    assert (pv.mint, pv.base_vault, pv.quote_vault) == (p["mint"], p["base_vault"], p["quote_vault"])


def test_learn_vaults_pumpswap_vaults_owned_by_pool():
    """PumpSwap: both vault token accounts have owner == pool (Token-2022 base, Tokenkeg quote)."""
    p = pool_of("pumpswap_buy")
    tx = load("pumpswap_buy")
    keys = [a["pubkey"] for a in tx["transaction"]["message"]["accountKeys"]]
    owners = {keys[b["accountIndex"]]: b["owner"] for b in tx["meta"]["postTokenBalances"]}
    assert owners[p["base_vault"]] == p["pool"]
    assert owners[p["quote_vault"]] == p["pool"]


@pytest.mark.parametrize("name,authority", [
    ("raydium_cpmm_sell", "GpMZbSM2GgvTKHJirzeGfMFoaZ8UR2X7F4v8vHTvxFbL"),
    ("raydium_v4_buy", "5Q544fKrFoe6tsEbD7S8EmxGTJYAKtTVhAW5Q5pge4j1"),
])
def test_raydium_vaults_owned_by_authority_not_pool(name, authority):
    p = pool_of(name)
    tx = load(name)
    keys = [a["pubkey"] for a in tx["transaction"]["message"]["accountKeys"]]
    owners = {keys[b["accountIndex"]]: b["owner"] for b in tx["meta"]["postTokenBalances"]}
    assert owners[p["base_vault"]] == authority != p["pool"]
    assert owners[p["quote_vault"]] == authority
    assert authority in sd.KNOWN_VAULT_AUTHORITIES


def test_learn_vaults_pool_absent_returns_none():
    other = pool_of("raydium_cpmm_sell")
    assert sd.learn_vaults(load("pumpswap_buy"), other["pool"], mint=other["mint"]) is None


def test_learn_vaults_unknown_quote_returns_none():
    """A pool whose quote is not WSOL/USDC/USDT must not be learned from a neighbouring pool's vaults."""
    p = pool_of("pumpswap_buy")
    quotes = {"EPjFWdd5AufqSSqeM2qN1xzybapC8G4sEGGkZwyTDt1v": 6}  # not a quote this pool uses
    assert sd.learn_vaults(load("pumpswap_buy"), p["pool"], quotes, mint=p["mint"]) is None


def test_pump_quoted_pool_never_learns_neighbouring_dlmm_vaults():
    """Real arbitrage-bot tx: Jupiter's graduatedPool for 'baton' is a PumpSwap pool quoted in PUMP, and
    the same tx lists the Meteora DLMM baton/WSOL pool's vaults. The live daemon once learned those
    DLMM vaults for the PumpSwap pool; this pins the fix (pool-owned vaults are definitive)."""
    name = "pumpswap_pump_quoted_bot"
    p, e, tx = pool_of(name), expect(name), load(name)
    keys = [a["pubkey"] for a in tx["transaction"]["message"]["accountKeys"]]
    assert p["pool"] in keys and e["dlmm_base_vault"] in keys and e["dlmm_quote_vault"] in keys  # the trap is present
    for kw in ({}, {"mint": p["mint"]}, {"mint": p["mint"], "program_id": p["program_id"]}):
        assert sd.learn_vaults(tx, p["pool"], **kw) is None, kw
    # with PUMP allowed as a quote the pool's own vaults are found (they are owned by the pool)
    quotes = dict(sd.QUOTE_MINTS, **{p["quote_mint"]: 6})
    pv = sd.learn_vaults(tx, p["pool"], quotes, mint=p["mint"], program_id=p["program_id"])
    assert (pv.base_vault, pv.quote_vault, pv.quote_mint) == (p["base_vault"], p["quote_vault"], p["quote_mint"])
    assert (pv.base_decimals, pv.quote_decimals) == (6, 6)
    # the bot never traded through the PumpSwap pool: no row
    assert sd.decode_transaction(tx, {p["pool"]: vaults_of(name)}) == []


@pytest.mark.parametrize("name", ["raydium_v4_buy", "raydium_cpmm_sell"])
def test_program_id_restricts_learning_to_that_programs_instructions(name):
    p = pool_of(name)
    wrong = sd.PROGRAM_IDS_BY_LABEL["Pump.fun Amm"]
    assert sd.learn_vaults(load(name), p["pool"], mint=p["mint"], program_id=wrong) is None
    pv = sd.learn_vaults(load(name), p["pool"], mint=p["mint"], program_id=p["program_id"])
    assert (pv.base_vault, pv.quote_vault) == (p["base_vault"], p["quote_vault"])


def test_program_ids_by_label_cover_fixture_dexes():
    for key, p in MANIFEST["pools"].items():
        assert sd.PROGRAM_IDS_BY_LABEL[p["program_label"]] == p["program_id"], key


# ---- decoding ---------------------------------------------------------------------------------
@pytest.mark.parametrize("name", SWAP_FIXTURES)
def test_decode_swap_matches_vault_deltas(name):
    p, e = pool_of(name), expect(name)
    rows = sd.decode_transaction(load(name), {p["pool"]: vaults_of(name)})
    assert len(rows) == 1
    r = rows[0]
    assert isinstance(r, TapeRow)
    assert r.side == e["side"] and r.side in (1, -1)
    assert (r.amount_base, r.amount_quote) == (e["amount_base"], e["amount_quote"])
    assert r.amount_base > 0 and r.amount_quote > 0
    assert (r.res_base, r.res_quote) == (e["res_base"], e["res_quote"])
    assert r.signer == e["signer"]
    assert (r.slot, r.sig, r.tx_index) == (e["slot"], e["sig"], e["tx_index"])
    assert (r.pool, r.mint, r.quote_mint, r.program_label) == (p["pool"], p["mint"], p["quote_mint"], p["program_label"])
    assert r.ts == datetime.fromtimestamp(e["block_time"], tz=timezone.utc)
    assert r.price_quote == pytest.approx(e["price_quote"], rel=1e-12)
    assert 0 < r.price_quote < 10  # memecoin priced in SOL/USDT: sane magnitude
    if p["quote_mint"] == sd.WSOL_MINT:
        assert r.price_sol == pytest.approx(e["price_sol"], rel=1e-12)
    else:
        assert r.price_sol is None


def test_pumpswap_buy_is_side_plus_one_with_sane_price():
    rows = sd.decode_transaction(load("pumpswap_buy"), {pool_of("pumpswap_buy")["pool"]: vaults_of("pumpswap_buy")})
    r = rows[0]
    assert r.side == 1
    assert r.amount_quote / 1e9 == pytest.approx(2.176599144)      # 2.18 SOL paid into the pool
    assert r.amount_base / 1e6 == pytest.approx(4958.580297)       # tokens left the pool
    assert 1e-5 < r.price_sol < 1e-2                                # ~0.000439 SOL per token
    assert r.price_sol == r.price_quote


def test_pumpswap_sell_is_side_minus_one():
    r = sd.decode_transaction(load("pumpswap_sell"), {pool_of("pumpswap_sell")["pool"]: vaults_of("pumpswap_sell")})[0]
    assert r.side == -1
    assert r.amount_base == expect("pumpswap_sell")["amount_base"] > 0
    assert r.amount_quote == expect("pumpswap_sell")["amount_quote"] > 0


def test_failed_transaction_is_skipped():
    tx = load("pumpswap_failed")
    assert tx["meta"]["err"] is not None
    p = pool_of("pumpswap_failed")
    keys = [a["pubkey"] for a in tx["transaction"]["message"]["accountKeys"]]
    assert p["base_vault"] in keys and p["quote_vault"] in keys  # the vaults are referenced, err is the reason
    assert sd.decode_transaction(tx, {p["pool"]: vaults_of("pumpswap_failed")}) == []


def test_lp_event_is_side_zero_with_absolute_amounts():
    e = expect("meteora_dlmm_lp")
    r = sd.decode_transaction(load("meteora_dlmm_lp"), {pool_of("meteora_dlmm_lp")["pool"]: vaults_of("meteora_dlmm_lp")})[0]
    assert r.side == 0
    assert (r.amount_base, r.amount_quote) == (e["amount_base"], 0)
    assert r.price_quote is None and r.price_sol is None
    assert r.res_base == e["res_base"] and r.res_quote == e["res_quote"]


def test_usdt_quoted_pool_has_price_quote_but_no_price_sol():
    r = sd.decode_transaction(load("meteora_dlmm_buy"), {pool_of("meteora_dlmm_buy")["pool"]: vaults_of("meteora_dlmm_buy")})[0]
    assert r.quote_mint == sd.USDT_MINT
    assert r.price_sol is None
    assert r.price_quote == pytest.approx(expect("meteora_dlmm_buy")["price_quote"])


def test_notification_shape_decodes_like_rpc_shape():
    tx = load("pumpswap_buy")
    p = pool_of("pumpswap_buy")
    pools = {p["pool"]: vaults_of("pumpswap_buy")}
    receipt = datetime(2026, 9, 12, 14, 0, tzinfo=timezone.utc)
    rpc_row = sd.decode_transaction(tx, pools)[0]
    ws_rows = sd.decode_transaction(as_notification(tx), pools, ts=receipt)
    assert len(ws_rows) == 1
    w = ws_rows[0]
    assert w.ts == receipt  # notifications carry no blockTime: receipt time is used
    for f in ("slot", "sig", "tx_index", "side", "amount_base", "amount_quote", "res_base", "res_quote",
              "signer", "price_sol", "pool", "mint"):
        assert getattr(w, f) == getattr(rpc_row, f), f
    # the same envelope also learns vaults
    pv = sd.learn_vaults(as_notification(tx), p["pool"], mint=p["mint"])
    assert (pv.base_vault, pv.quote_vault) == (p["base_vault"], p["quote_vault"])


def test_untouched_pool_yields_nothing():
    other = pool_of("raydium_cpmm_sell")
    assert sd.decode_transaction(load("pumpswap_buy"), {other["pool"]: vaults_of("raydium_cpmm_sell")}) == []


@pytest.mark.parametrize("name", SWAP_FIXTURES)
def test_vault_index_over_many_pools_attributes_to_the_right_pool(name):
    pools = all_pools()
    index = sd.build_vault_index(pools)
    assert len(index) == 2 * len(pools)
    rows = sd.decode_transaction(load(name), pools, vault_index=index)
    assert [r.pool for r in rows] == [pool_of(name)["pool"]]


def test_taperow_shapes():
    r = sd.decode_transaction(load("raydium_v4_sell"), {pool_of("raydium_v4_sell")["pool"]: vaults_of("raydium_v4_sell")})[0]
    d = r.as_dict()
    assert set(COLUMNS) <= set(d) and {"price_quote", "quote_mint"} <= set(d)
    assert len(r.db_values()) == len(COLUMNS)
    assert d["side"] == -1
