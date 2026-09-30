"""Log reads from Envio HyperSync (``ENVIO_API_TOKEN``): the chain's full history in bulk, instead of the public RPC's
~1 eth_getLogs a second under a 10,000-log cap. ``HyperRpc`` is an ``RhRpc`` whose ``get_logs_multi`` asks HyperSync and
returns the same log dicts as eth_getLogs (plus ``blockTimestamp``, which the RPC leaves at 0x0). Blocks HyperSync has
not reached yet are read from the RPC.

Query API (measured 2026-09-30): POST {url}/query {from_block, to_block (exclusive), logs: [{address, topics}],
field_selection} → {data: [{logs, blocks}], next_block, archive_height}; a response may stop early at ``next_block``.
"""
from __future__ import annotations

import threading
import time

import httpx

from .. import config
from ..db.apilog import record_api_call
from .rpc import RhRpc

LOG_FIELDS = ["block_number", "log_index", "transaction_hash", "block_hash", "address", "data", "topic0", "topic1", "topic2", "topic3"]
TRIES = 8
_PACE_LOCK = threading.Lock()
_LAST = [0.0]


def _pace() -> None:
    """At most ``RH_HYPERSYNC_RPM`` queries a minute across threads (the free tier is fair-use; Starter 100 rpm)."""
    gap = 60.0 / max(1.0, float(config.RH_HYPERSYNC_RPM))
    with _PACE_LOCK:
        wait = _LAST[0] + gap - time.monotonic()
        if wait > 0:
            time.sleep(wait)
        _LAST[0] = time.monotonic()


class HyperSyncError(RuntimeError):
    pass


def _topics(topics: list | None) -> list[list[str]]:
    out = []
    for t in topics or []:
        out.append([] if t is None else ([x.lower() for x in t] if isinstance(t, (list, tuple)) else [t.lower()]))
    return out


def to_rpc_log(g: dict, ts: dict[int, int]) -> dict:
    b = int(g["block_number"])
    return {"address": (g.get("address") or "").lower(), "topics": [g[k] for k in ("topic0", "topic1", "topic2", "topic3") if g.get(k)],
            "data": g.get("data") or "0x", "blockNumber": hex(b), "logIndex": hex(int(g["log_index"])), "transactionHash": g.get("transaction_hash"),
            "blockHash": g.get("block_hash"), "blockTimestamp": hex(ts[b]) if b in ts else "0x0"}


class HyperSync:
    def __init__(self, token: str, url: str | None = None, client: httpx.Client | None = None):
        self.url = (url or config.RH_HYPERSYNC_URL).rstrip("/")
        self.http = client or httpx.Client(timeout=120.0, headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"})
        self.height = 0

    def _post(self, body: dict) -> dict:
        for i in range(TRIES):
            _pace()
            t0 = time.monotonic(); status = None; err = None; retry_after = None
            try:
                r = self.http.post(f"{self.url}/query", json=body); status = r.status_code
                if r.status_code == 200:
                    return r.json()
                err = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in (429, 500, 502, 503, 504):
                    raise HyperSyncError(err)
                ra = r.headers.get("retry-after")
                retry_after = float(ra) if ra and ra.replace(".", "", 1).isdigit() else None
            except httpx.HTTPError as e:
                err = type(e).__name__
            finally:
                record_api_call("rh_hypersync", "/query", "POST", status, int((time.monotonic() - t0) * 1000), err is None, err)
            time.sleep(retry_after if retry_after is not None else min(60.0, 5.0 * 2 ** i))       # a rate-limit window resets within a minute
        raise HyperSyncError(f"HyperSync failed {TRIES} times: {err}")

    def archive_height(self) -> int:
        r = self.http.get(f"{self.url}/height"); r.raise_for_status()
        self.height = int(r.json()["height"])
        return self.height

    def get_logs(self, addresses: list[str] | None, lo: int, hi: int, topics: list | None = None) -> list[dict]:
        """Logs in blocks lo..hi (inclusive), eth_getLogs-shaped, in chain order."""
        sel: dict = {}
        if addresses:
            sel["address"] = [a.lower() for a in addresses]
        if topics:
            sel["topics"] = _topics(topics)
        out, ts, frm = [], {}, lo
        while frm <= hi:
            d = self._post({"from_block": frm, "to_block": hi + 1, "logs": [sel], "field_selection": {"log": LOG_FIELDS, "block": ["number", "timestamp"]}})
            for part in d.get("data") or []:
                for b in part.get("blocks") or []:
                    ts[int(b["number"])] = int(b["timestamp"], 16) if isinstance(b["timestamp"], str) else int(b["timestamp"])
                out += part.get("logs") or []
            nxt = int(d.get("next_block") or hi + 1)
            if nxt <= frm:
                raise HyperSyncError(f"HyperSync made no progress at block {frm}")
            frm = nxt
        logs = [to_rpc_log(g, ts) for g in out]
        logs.sort(key=lambda x: (int(x["blockNumber"], 16), int(x["logIndex"], 16)))
        return logs


class HyperRpc(RhRpc):
    """An RhRpc whose log reads come from HyperSync up to its archive height, and from the RPC beyond it."""

    def __init__(self, hs: HyperSync, **kw):
        super().__init__(**kw)
        self.hs = hs

    def get_logs_multi(self, addresses, from_block: int, to_block: int, topics=None) -> list[dict]:
        top = self.hs.height if to_block <= self.hs.height else self.hs.archive_height()
        if to_block <= top:
            return self.hs.get_logs(addresses, from_block, to_block, topics)
        out = self.hs.get_logs(addresses, from_block, top, topics) if from_block <= top else []
        return out + super().get_logs_multi(addresses, max(from_block, top + 1), to_block, topics)
