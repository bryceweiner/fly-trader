"""Our own Solana feed (ingest/helius_feed.py): real mainnet events decode to exactly what PumpAPI recorded for the same
transactions (its archive and stream rows, 2026-10-19), cut-off transactions are fetched whole and hold their minute,
and a pool first seen is read for its mints."""
import asyncio
import base64

import base58
import pytest

from fly_trader.ingest import helius_feed as hf

# a PumpSwap buy and sale of slot 458694520 and PumpAPI's archive rows for them
BUY = ("Z/RSHyz1d3eGl9VqAAAAAIj5ZwgAAAAAqqXIEgAAAAB65OpfAAAAAKqlyBIAAAAAzmqF+Z4CAAC/BTd7iwUAAEz31REAAAAAFAAAAAAAAADHIQkAAAAAAAUAAAAAAAAAckgCAAAAAAATGd8RAAAAAPep4xEAAAAAifzy+5hplKLOHT9q2Tb+TxSpBNzCeZz7Pmv2nnmhbFEtIeVOaIW2bFBTdyLO22Vs1eHt9BkHA6J/YELevIFIkf4QRY1FrZ1msENG97lcJnNh/BBwppkdCs0WmjZpVnvItKGQL7P7SVfIOgXW3BfCYdjZaqFWVT9Vk5eM3vqhqJ9gjMwd/OlhtDt3nBkVBabi079F1aTbRhitdsgtYXVFNV0OMm1Matf/fY9TP8MJCFS9H+8ykQhAD19V155Lh4BRUTVB2kCi9fIkH75mB+5QbNLVbq42Z/0HfaQX9zT19/YFAAAAAAAAAHJIAgAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAiPlnCAAAAAADAAAAYnV5AAAAAAAAAAAAAAAAAAAAAIgTAAAAAAAAOSQBAAAAAADLcQQYBAAAAAAAAAAAAAAAAQDMlal8jQMAAAAAAAAAAAAAAAAAAAAAAFO4GQAAAAAA")
BUY_POOL, BUY_MINT, BUY_TRADER = "AHeZiehLfP7cX8qYNoT1SM45CMdN3XGzMH2vfUq738gG", "pHW26J5oCRgr591wU5BbydvbF4p3zm64QEGDx6spump", "43BKnn3gnGrTXs5qF9HeVrQh75vWTNdEmieauXMcVHtt"
SELL = ("Pi83CqUD3CqGl9VqAAAAAJPqVt0AAAAAAAAAAAAAAAAwQX6kFQAAAAAAAAAAAAAAytzUCQcOAABoWmwBBgEAAJLFZRAAAAAAFAAAAAAAAABDZQgAAAAAAAUAAAAAAAAAURkCAAAAAABPYF0QAAAAAKi4UxAAAAAAsU390QxXuS1uUKlpEHVf6XfCg74gnJZ4fNRov+OPgTbDi8nUVaTTa0tTIxRl88AcduaFFW/e4MQd50RYk1bHz5JMmcXoGR8GqtVjsjXTlvvTnOFCdf2dlefsHOEcgR2n+Q6xsQpP5B/0VSlpfe70EDc2zPKid5fjuhZFHzJlzdpgjMwd/OlhtDt3nBkVBabi079F1aTbRhitdsgtYXVFNV0OMm1Matf/fY9TP8MJCFS9H+8ykQhAD19V155Lh4BRnLLOUvi9AAAGoxgoi0+Pfm7HRlXRyYE9Qlkx07mqScgSAAAAAAAAAFaOBwAAAAAAAAAAAAAAAAAAAAAAAAAAAIgTAAAAAAAAqAwBAAAAAADIQR4YBAAAAAAAAAAAAAAAAVMcintCjQMAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
SELL_POOL, SELL_MINT, SELL_TRADER = "Cw8936xr1GHo16nuqi5yVjEbdTCC4ioCSB86bbPePZjs", "wtfdg6xwKYfEnoa16JArycTZ612gzgMGshbBqnwpump", "EAL8aSRU8Skxp4CJbb4cGb2oZAEwp3n3rCqZr4Vg1B1x"
# a mayhem-mode create with its creator's buy (tx 3bTz3igqcf…, PumpAPI: dev 0.075075987 SOL, 2678514.694875 tokens, supply 2e9)
CREATE = ["G3KpTd7rY3YQAAAARnJlbmNoIEdva3UgSGVybwYAAABQSUVSUkVDAAAAaHR0cHM6Ly9pcGZzLmlvL2lwZnMvUW1TWXhQdlFxNlVrcVhFODRGemRlaVdpaTZoZ1F1NGcyd2d1VFJ1V0doU0tiMfrYJ+DKN6sCWEt5a/qLgscDXx8HPpai4Hiu/DF5b4vfws5A43OfnUSsmlb0W0dSbZzQ7+bmeqi5xn65J/cOnFM6o1ByRC0oNGE55GdIgqw3qciOEDVpvj2L2towErn7czqjUHJELSg0YTnkZ0iCrDepyI4QNWm+PYva2jASuftz4MjVagAAAAAAENhH488DAACsI/wGAAAAAHjF+1HRAgAAgMakfo0DAAbd9uHudY/eGEJdvORszdq2GvxNg7kNJ/69+SjYoYv8AQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAACsI/wGAAAAAAAAAAAAAAAAAA==",
          "vdt/007mYe762CfgyjerAlhLeWv6i4LHA18fBz6WouB4rvwxeW+L35OReQQAAAAA27rqo28CAAABOqNQckQtKDRhOeRnSIKsN6nIjhA1ab49i9raMBK5+3PgyNVqAAAAAJM9nQAHAAAAJVXto3PNAwCTkXkEAAAAACW92lfizgIANYRTYlYJT5EoGRJ++k5Ea0MzcheT0Th2+arb89x9C19fAAAAAAAAAAbiCgAAAAAAOqNQckQtKDRhOeRnSIKsN6nIjhA1ab49i9raMBK5+3MeAAAAAAAAAMxvAwAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAEgAAAGJ1eV9leGFjdF9xdW90ZV9pbgEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAJOReQQAAAAAkz2dAAcAAACTkXkEAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="]
CREATE_MINT, CREATE_SIGNER = "HtC6SDTy3H2qd2PSd6EzqzozhLvFZevq7dynJpMepump", "4wu5nFAjsUBwgPZrdtJyAK4pzRcbEQKdNCVgcUih53vi"
# a boosted migration's self-CPI events (tx 2qht5gndpy…, logs cut off; PumpAPI: quoteInPool 67.405853768)
MIGRATION = [(hf.AMM, "H3QSeD5qWuG9CPkfpRpuTjuJzX59XP4Edypm8yL3riZjzD9RjRnR2teTirRCoNPgJAr4u4DwDfp56ujNAFBMXKeJrPJ71T9ZuM23oDovzfiBFX5de3Rdf3FHhvRaesw83DaQiGcpvZncbmsym757H1r2nr3Y3SwK8STNCS4ju2VAFrx9dheF6WXDF8Zw8QFmQ81yoUgFULidRKRP4haTMV1FghzSXC14xEJfmUprgpHu5dqv3wPpaT9iEpeEHb6wpyAiX1fEcyhhnUsFDAEk7euS6mG5zFDrq6hGbsgk4ymiP49STdbTMD8jbij2w3LisV83fY2jqkqgiswv7L8MnhVYYRDJw461vbdUZtcQXCwqKZghZCP6JipLVxY5LPBWQoE2p1KVh3zRSgBRVbt5BGqvoU3wgABwEh1dNjhvuqwpLU3ie37Fa5fcDwPz3zN8uK6c1DsqV3GaNK2CBJqMYaVcAmhL8S51y"),
             (hf.AMM, "DjRuqDRQjw1SYyq7taVPcKfC3tY5iwM45ZzwpZp3tYqdWtp61siCjMmztrmchVczif6hqMqbQLXUdevUNhvPenofzaa9XrRSyYx9iUuxXmRyfBmwuy8M7DgLZeegWfv1n6TtZtP7PL9mU3gicvGSDWpW2zSp7VXDnxCFa2WDaVJ7CzeGJZCRUs3GTWgfBpGPaun39"),
             (hf.PUMP, "28AUVybe3Jro3a5o6vV4ukD6DAhUER3Z8TxgXHRjvKCN3L2UnjK2d7x88aAjX2DcDDnMjZnKyxAhV9AzyF2UanY3bYnsCxkaHNGuqXxoNwH65V69ip1zaBdQkzimdRGPtyY7TA3roj6V4FHdD2yGNdos7DiCSgeeo5r4AkNZ78iZDYn4GFXkr4DT3fAt3BsrXcSWiXfJGehSevL2H2mhWZKRmgtX2S83bXDreBzLEsNR3jQ3SjF2VWm6ArVvDbBA4cQ7D9sreYeNti7dVh8WxFqHBMpoy")]
MIG_PAYER = "H8yC3CGKrNL2i7pS3FB6Z51ENeLeCVtknYP6JEtXRrEw"


def _logs(*payloads, truncated=False):
    return [f"Program data: {p}" for p in payloads] + (["Log truncated"] if truncated else [])


def _tx(payer, cpi):
    """A getTransaction (json) result holding ``cpi`` [(program, base58 data)] as one instruction's inner instructions."""
    keys = [payer, hf.AMM, hf.PUMP]
    return {"slot": 9, "transaction": {"message": {"accountKeys": keys}},
            "meta": {"loadedAddresses": {}, "innerInstructions": [{"index": 0, "instructions": [{"programIdIndex": keys.index(p), "data": d} for p, d in cpi]}]}}


def _note(sig, logs, slot=5):
    return {"params": {"result": {"context": {"slot": slot}, "value": {"signature": sig, "err": None, "logs": logs}}}}


def test_a_pumpswap_buy_and_sale_decode_to_pumpapis_trades():
    pools = {BUY_POOL: (BUY_MINT, hf.WSOL), SELL_POOL: (SELL_MINT, hf.WSOL)}
    (b,), unknown = hf.build("sigB", 458694520, hf.log_events(_logs(BUY)), pools)
    assert not unknown and (b["action"], b["pool"], b["mint"], b["poolId"], b["quoteMint"]) == ("buy", "pump-amm", BUY_MINT, BUY_POOL, hf.WSOL)
    assert b["breakdown"] == [{"action": "buy", "trader": BUY_TRADER, "quoteAmount": 0.299235148}]
    assert b["quoteInPool"] == pytest.approx(6096.925031691, abs=1e-9) and b["price"] == pytest.approx(0.0021218601665281655, rel=1e-12)
    (s,), _ = hf.build("sigS", 458694520, hf.log_events(_logs(SELL)), pools)
    assert s["breakdown"] == [{"action": "sell", "trader": SELL_TRADER, "quoteAmount": 0.27510517}]
    assert s["quoteInPool"] == pytest.approx(1125.03020463, abs=1e-9) and s["price"] == pytest.approx(7.406539583265785e-05, rel=1e-12)
    assert b["timestamp"] % 1000 == pytest.approx(900) and hf.minute_of(hf.log_events(_logs(BUY))) == int(b["timestamp"] / 1000 // 60) * 60


def test_a_trade_in_a_pool_not_known_yet_waits_for_its_mints():
    out, unknown = hf.build("sigB", 1, hf.log_events(_logs(BUY)), {})
    assert out == [] and unknown == {BUY_POOL}


def test_a_create_carries_its_creators_buy_and_a_mayhem_coins_whole_supply():
    out, _ = hf.build("sigC", 458752890, hf.log_events(_logs(*CREATE)), {})
    c, curve = out[0], out[1:]
    assert (c["action"], c["pool"], c["mint"], c["txSigner"], c["name"], c["symbol"]) == ("create", "pump", CREATE_MINT, CREATE_SIGNER, "French Goku Hero", "PIERRE")
    assert (c["quoteAmount"], c["initialBuy"], c["supply"], c["mayhemMode"]) == (pytest.approx(0.075075987), pytest.approx(2678514.694875), 2e9, True)
    assert [(e["action"], e["pool"], e["mint"], e["breakdown"][0]["trader"]) for e in curve] == [("buy", "pump", CREATE_MINT, CREATE_SIGNER)]


def test_a_migration_is_read_from_its_inner_instructions_with_its_boosted_reserve():
    evs = hf.cpi_events(_tx(MIG_PAYER, MIGRATION))
    assert [n for n, _ in evs] == ["CreatePoolEvent", "InitBoostEvent", "CompletePumpAmmMigrationEvent"]
    (m,), _ = hf.build("sigM", 458751013, evs, {}, payer=MIG_PAYER)
    assert (m["action"], m["pool"], m["mint"], m["poolId"], m["txSigner"]) == (
        "migrate", "pump-amm", "tmVbsN1ojRb5wtatFF1dVy1VoMep1g4YzRwnuoPpump", "Dte7pq6rxNDc1vF8zhwzKHfFXCAFchs7KRF47jQWSsee", MIG_PAYER)
    assert m["quoteInPool"] == pytest.approx(67.405853768)                          # the rq0 feature: the real reserve after the boost
    (alone,), _ = hf.build("sigM", 458751013, evs[:1], {})                          # its pool creation alone would read as a custom pool
    assert alone["action"] == "createPool" and alone["poolCreatedBy"] != "pump"


class _Rpc:
    def __init__(self, txs=None, accounts=None):
        self.txs, self.accounts, self.calls = txs or {}, accounts or {}, []

    def batch(self, calls):
        self.calls.append([m for m, _ in calls])
        return [self.txs.get(p[0]) for _, p in calls]

    def call(self, method, params):
        self.calls.append(method)
        return {"value": [self.accounts.get(k) for k in params[0]]}


def test_cut_off_logs_are_never_read_the_transaction_is_fetched_whole_and_holds_its_minute():
    seen = []
    whole = _tx("PAYER", [(hf.AMM, base58.b58encode(hf.EVENT_IX_TAG + base64.b64decode(BUY)).decode())])
    feed = hf.HeliusFeed(lambda evs, src: seen.extend((e["action"], e["mint"], src) for e in evs), rpc=_Rpc(txs={"sigX": whole}))
    feed.pools.map[BUY_POOL] = (BUY_MINT, hf.WSOL)
    feed.on_logs(hf.AMM, _note("sigX", _logs(BUY, truncated=True)))
    m = hf.minute_of(hf.log_events(_logs(BUY)))
    assert seen == [] and "sigX" in feed.fetching and feed.pending == {m: 1}       # nothing from the cut-off logs; its minute waits
    asyncio.run(feed._fetch_round())
    assert seen == [("buy", BUY_MINT, "helius")] and not feed.fetching and not feed.pending and feed.wm > 0


def test_a_pool_first_seen_is_read_for_its_mints_before_its_trade_goes_out():
    seen = []
    data = bytearray(300); data[43:75] = base58.b58decode(BUY_MINT); data[75:107] = base58.b58decode(hf.WSOL)
    rpc = _Rpc(accounts={BUY_POOL: {"owner": hf.AMM, "data": [base64.b64encode(bytes(data)).decode(), "base64"]}})
    feed = hf.HeliusFeed(lambda evs, src: seen.extend(e["mint"] for e in evs), rpc=rpc)
    feed.on_logs(hf.AMM, _note("sigB", _logs(BUY)))
    assert seen == [] and len(feed.waiting) == 1 and sum(feed.pending.values()) == 1
    asyncio.run(feed._resolve_round())
    assert seen == [BUY_MINT] and feed.pools.map[BUY_POOL] == (BUY_MINT, hf.WSOL) and not feed.waiting and not feed.pending
