"""The signer's own Solana JSON-RPC client (its own provider: SIGNER_RPC_URL, not the brain's), and Jupiter's public
price endpoint for the value check. httpx only."""
from __future__ import annotations

import base64

import httpx
from solders.pubkey import Pubkey

PRICE_URL = "https://lite-api.jup.ag/price/v3"


class RpcError(RuntimeError):
    pass


class Rpc:
    def __init__(self, url: str, timeout: float = 20.0):
        if not url:
            raise RpcError("the signer needs its own RPC URL (SIGNER_RPC_URL)")
        self.url, self.http = url, httpx.Client(timeout=timeout)

    def call(self, method: str, params: list):
        r = self.http.post(self.url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params})
        r.raise_for_status()
        d = r.json()
        if d.get("error"):
            raise RpcError(f"{method}: {d['error']}")
        return d.get("result")

    def accounts(self, keys: list[Pubkey], commitment: str = "confirmed") -> list[dict | None]:
        out: list[dict | None] = []
        for i in range(0, len(keys), 100):
            v = self.call("getMultipleAccounts", [[str(k) for k in keys[i:i + 100]], {"encoding": "base64", "commitment": commitment}])["value"]
            out += [None if a is None else {"lamports": a["lamports"], "owner": Pubkey.from_string(a["owner"]), "data": base64.b64decode(a["data"][0])}
                    for a in v]
        return out

    def account(self, key: Pubkey, commitment: str = "confirmed") -> dict | None:
        return self.accounts([key], commitment)[0]

    def balance(self, key: Pubkey, commitment: str = "confirmed") -> int:
        return int(self.call("getBalance", [str(key), {"commitment": commitment}])["value"])

    def token_accounts(self, owner: Pubkey, program: Pubkey) -> list[Pubkey]:
        v = self.call("getTokenAccountsByOwner", [str(owner), {"programId": str(program)}, {"encoding": "base64", "commitment": "confirmed",
                                                                                           "dataSlice": {"offset": 0, "length": 0}}])["value"]
        return [Pubkey.from_string(a["pubkey"]) for a in v]

    def blockhash(self) -> tuple[str, int]:
        v = self.call("getLatestBlockhash", [{"commitment": "confirmed"}])["value"]
        return v["blockhash"], int(v["lastValidBlockHeight"])

    def block_height(self) -> int:
        return int(self.call("getBlockHeight", [{"commitment": "confirmed"}]))

    def signature_status(self, sig: str) -> dict | None:
        return self.call("getSignatureStatuses", [[sig], {"searchTransactionHistory": True}])["value"][0]

    def simulate(self, tx_b64: str, addresses: list[Pubkey]) -> dict:
        return self.call("simulateTransaction", [tx_b64, {"sigVerify": False, "replaceRecentBlockhash": True, "encoding": "base64",
                                                          "commitment": "processed",
                                                          "accounts": {"encoding": "base64", "addresses": [str(a) for a in addresses]}}])["value"]

    def usd_prices(self, mints: list[str]) -> dict[str, float]:
        try:
            r = self.http.get(PRICE_URL, params={"ids": ",".join(mints)}, timeout=5)
            r.raise_for_status()
            return {m: float(v["usdPrice"]) for m, v in (r.json() or {}).items() if v and v.get("usdPrice")}
        except Exception:
            return {}
