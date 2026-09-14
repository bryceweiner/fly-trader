"""Historical corpus puller (Supervisor thread ``corpus``).

Enumerates every pump.fun graduation by paging the migration authority's succeeded transactions through
Helius ``getTransactionsForAddress`` (newest first, back to ``CORPUS_DAYS``), then, per token and newest
first, pulls from pump.fun's own swap-api:

* candles in SOL: 1-minute for the first ``CORPUS_CANDLE_1M_H`` hours after graduation (plus the last 30
  minutes of the bonding curve), then 5-minute until ``CORPUS_CANDLE_5M_D`` days (2–4 requests per token);
* tokens are pulled only once they are ``CORPUS_MIN_AGE_H`` old, so the first hours are complete;
* per-trade rows (wallet, side, SOL, tokens, price) for the first ``CORPUS_TRADES_H`` hours, for a deterministic
  ``CORPUS_TRADES_SAMPLE`` share of the tokens that kept trading at least ``CORPUS_MIN_LIFE_H`` and graduated within
  ``CORPUS_TRADES_DAYS`` (trades cost up to 100 pages per token; candles cost one to three).

Registry: ``corpus_tokens``. Data: ``data/corpus/candles/<mint>.parquet`` and ``data/corpus/trades/<mint>.parquet``
(append-only: a token already on disk is never rewritten). Progress: ``ui_settings['corpus_status']``.
The swap-api throttles bursts (Cloudflare 1015 after ~20 requests at 1/s), so requests are paced adaptively.
Measured 2026-09-13: ~36 graduations/hour; candle volume in SOL matches the tape within 0.5 %.
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import httpx
import pyarrow as pa
import pyarrow.parquet as pq

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup

log = logging.getLogger(__name__)
WSOL = "So11111111111111111111111111111111111111112"
PUMP_PROGRAM = "6EF8rrecthR5Dkzon8Nwu78hRvfCKubJ14M5uBEwF6P"
SLOTS_PER_S = 2.5
CANDLE_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("interval", pa.string()), ("open", pa.float64()),
                           ("high", pa.float64()), ("low", pa.float64()), ("close", pa.float64()), ("volume_sol", pa.float64())])
TRADE_SCHEMA = pa.schema([("ts", pa.timestamp("ms", tz="UTC")), ("slot_index", pa.string()), ("tx", pa.string()), ("wallet", pa.string()),
                          ("side", pa.int8()), ("program", pa.string()), ("price_sol", pa.float64()), ("sol", pa.float64()), ("tokens", pa.float64())])


class Throttled(Exception):
    pass


class Status:
    """ui_settings['corpus_status'] writer (throttled to one write per 3 s unless forced)."""

    def __init__(self):
        self.s: dict = {"started_at": datetime.now(timezone.utc).isoformat(), "req_ok": 0, "req_429": 0, "candles_rows": 0, "trades_rows": 0,
                        "tokens_pulled": 0, "with_trades": 0}
        self.lock = threading.Lock(); self.last = 0.0; self.t0 = time.time()

    def update(self, force: bool = False, **kv) -> None:
        with self.lock:
            self.s.update(kv); self.s["updated_at"] = datetime.now(timezone.utc).isoformat()
            hours = (time.time() - self.t0) / 3600
            self.s["tokens_per_h"] = self.s["tokens_pulled"] / hours if hours > 0.01 else 0.0
            if not force and time.time() - self.last < 3.0:
                return
            self.last = time.time(); payload = dict(self.s)
        try:
            with transaction() as conn:
                conn.execute("INSERT INTO ui_settings (key, value) VALUES ('corpus_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (json.dumps(payload, default=str),))
        except Exception:
            log.debug("corpus status write failed", exc_info=True)

    def bump(self, key: str, n: int = 1) -> None:
        with self.lock:
            self.s[key] = self.s.get(key, 0) + n


class Pacer:
    """Adaptive spacing between swap-api requests: back off 0.5 s per throttle, creep back after 200 clean calls."""

    def __init__(self, pace_s: float, floor_s: float):
        self.pace = pace_s; self.floor = floor_s; self.last = 0.0; self.clean = 0

    def wait(self, stop: threading.Event | None) -> None:
        dt = time.time() - self.last
        if dt < self.pace:
            _sleep(self.pace - dt, stop)
        self.last = time.time()

    def ok(self) -> None:
        self.clean += 1
        if self.clean >= 200:
            self.clean = 0; self.pace = max(self.floor, self.pace - 0.1)

    def throttled(self, stop: threading.Event | None, retry_after: float | None) -> None:
        self.clean = 0; self.pace = min(10.0, self.pace + 0.5)
        _sleep(max(60.0, retry_after or 0.0), stop)


def _sleep(seconds: float, stop: threading.Event | None) -> None:
    end = time.time() + seconds
    while time.time() < end:
        if stop is not None and stop.is_set():
            return
        time.sleep(min(1.0, end - time.time()))


class SwapApi:
    def __init__(self, pacer: Pacer, status: Status, stop: threading.Event | None):
        self.c = httpx.Client(base_url=config.CORPUS_SWAP_API, headers={"User-Agent": "Mozilla/5.0", "Accept": "application/json"}, timeout=30)
        self.pacer, self.status, self.stop = pacer, status, stop

    def get(self, path: str, params: dict):
        for attempt in range(6):
            if self.stop is not None and self.stop.is_set():
                raise InterruptedError
            self.pacer.wait(self.stop)
            try:
                r = self.c.get(path, params=params)
            except httpx.HTTPError as e:
                log.warning("swap-api %s: %s", path, e); _sleep(5.0 * (attempt + 1), self.stop); continue
            if r.status_code == 200:
                self.pacer.ok(); self.status.bump("req_ok"); return r.json()
            if r.status_code == 429:
                self.status.bump("req_429"); ra = r.headers.get("retry-after")
                log.info("swap-api throttled; pace -> %.1f s", self.pacer.pace + 0.5)
                self.pacer.throttled(self.stop, float(ra) if ra and ra.isdigit() else None); continue
            if r.status_code >= 500:
                _sleep(5.0 * (attempt + 1), self.stop); continue
            raise RuntimeError(f"swap-api {path} HTTP {r.status_code}: {r.text[:160]}")
        raise RuntimeError(f"swap-api {path}: gave up after retries")

    def candles(self, mint: str, interval: str, created_ms: int, before_ms: int | None = None) -> list[dict]:
        p = {"interval": interval, "limit": 1000, "createdTs": created_ms, "currency": "SOL"}
        if before_ms is not None:
            p["beforeTs"] = before_ms
        j = self.get(f"/v2/coins/{mint}/candles", p)
        return j if isinstance(j, list) else (j.get("candles") or [])

    def trades(self, mint: str, cursor: str | None) -> tuple[list[dict], str | None]:
        p = {"limit": 100}
        if cursor:
            p["cursor"] = cursor
        j = self.get(f"/v2/coins/{mint}/trades", p)
        return j.get("trades") or [], (j.get("pagination") or {}).get("nextCursor")


class Helius:
    def __init__(self, stop: threading.Event | None):
        if not config.HELIUS_API_KEY:
            raise RuntimeError("HELIUS_API_KEY missing")
        self.c = httpx.Client(timeout=120); self.stop = stop; self.last = 0.0

    def gtfa(self, address: str, opts: dict) -> dict:
        for attempt in range(5):
            dt = time.time() - self.last
            if dt < 0.3:
                time.sleep(0.3 - dt)
            self.last = time.time()
            try:
                r = self.c.post(f"{config.HELIUS_HTTP}?api-key={config.HELIUS_API_KEY}",
                                json={"jsonrpc": "2.0", "id": 1, "method": "getTransactionsForAddress", "params": [address, opts]})
                j = r.json()
            except (httpx.HTTPError, ValueError) as e:
                log.warning("helius gTFA: %s", e); _sleep(3.0 * (attempt + 1), self.stop); continue
            if "result" in j and j["result"] is not None:
                return j["result"]
            err = (j.get("error") or {}).get("message", "unknown error")
            if "rate" in err.lower() or r.status_code == 429:
                _sleep(5.0 * (attempt + 1), self.stop); continue
            raise RuntimeError(f"helius gTFA: {err}")
        raise RuntimeError("helius gTFA: gave up after retries")


# ---------------------------------------------------------------- enumeration
def _graduations_from_page(data: list[dict]) -> list[tuple[str, int, str, int]]:
    """(mint, blockTime, signature, slot) for every succeeded migration transaction on the page."""
    out = []
    for tx in data:
        msg = tx["transaction"]["message"]; keys = msg.get("accountKeys") or []
        progs = set()
        for ins in msg.get("instructions") or []:
            pid = ins.get("programId")
            if pid is None and "programIdIndex" in ins and ins["programIdIndex"] < len(keys):
                k = keys[ins["programIdIndex"]]; pid = k["pubkey"] if isinstance(k, dict) else k
            progs.add(pid)
        if PUMP_PROGRAM not in progs:
            continue
        mints = {b.get("mint") for b in ((tx.get("meta") or {}).get("postTokenBalances") or [])} - {WSOL, None}
        if len(mints) != 1:
            continue
        sig = tx["transaction"]["signatures"][0] if tx["transaction"].get("signatures") else None
        out.append((mints.pop(), int(tx["blockTime"]), sig, int(tx["slot"])))
    return out


def _upsert_graduations(rows: list[tuple[str, int, str, int]]) -> int:
    if not rows:
        return 0
    with transaction() as conn:
        before = conn.execute("SELECT count(*) AS n FROM corpus_tokens").fetchone()["n"]
        conn.cursor().executemany(
            "INSERT INTO corpus_tokens (mint, graduated_at, grad_sig, grad_slot) VALUES (%s, to_timestamp(%s), %s, %s) "
            "ON CONFLICT (mint) DO UPDATE SET graduated_at = LEAST(corpus_tokens.graduated_at, EXCLUDED.graduated_at), "
            "grad_slot = LEAST(corpus_tokens.grad_slot, EXCLUDED.grad_slot), updated_at = now()", rows)
        after = conn.execute("SELECT count(*) AS n FROM corpus_tokens").fetchone()["n"]
    return int(after - before)


def _enum_state() -> dict:
    with transaction() as conn:
        r = conn.execute("SELECT value FROM ui_settings WHERE key = 'corpus_enum'").fetchone()
    v = r["value"] if r else {}
    return v if isinstance(v, dict) else json.loads(v or "{}")


def _save_enum_state(state: dict) -> None:
    with transaction() as conn:
        conn.execute("INSERT INTO ui_settings (key, value) VALUES ('corpus_enum', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                     (json.dumps(state, default=str),))


def enumerate_graduations(hel: Helius, status: Status, stop: threading.Event | None, max_pages: int = 20) -> int:
    """One enumeration round: (1) newest pages until a page adds nothing new, (2) up to ``max_pages`` of backfill
    from the persisted pagination token until ``CORPUS_DAYS`` is reached. Returns rows added."""
    addr = config.CORPUS_MIGRATION_AUTHORITY
    pull_before = datetime.fromisoformat(config.CORPUS_PULL_BEFORE).replace(tzinfo=timezone.utc)
    horizon = int(min(datetime.now(timezone.utc) - timedelta(days=config.CORPUS_DAYS), pull_before - timedelta(days=30)).timestamp())
    state = _enum_state(); added = 0
    # (1) incremental: newest first, stop when a full page adds nothing
    token = None
    for _ in range(500):          # keeps walking after an outage until a page adds nothing
        res = hel.gtfa(addr, {"limit": 100, "transactionDetails": "full", "sortOrder": "desc", "filters": {"status": "succeeded"},
                              **({"paginationToken": token} if token else {})})
        rows = _graduations_from_page(res.get("data") or []); n = _upsert_graduations(rows); added += n
        token = res.get("paginationToken")
        status.update(stage="enumerating (new graduations)", enum_added=added)
        if n == 0 or not token or (stop is not None and stop.is_set()):
            break
    # (2) backfill toward the horizon from where the last round stopped
    if not state.get("backfill_done"):
        token = state.get("token"); oldest = state.get("oldest_bt")
        if token is None and oldest is None:
            # first round: start a fresh newest-first walk that will proceed past the incremental pages
            token = None
        for _ in range(max_pages):
            if stop is not None and stop.is_set():
                break
            res = hel.gtfa(addr, {"limit": 100, "transactionDetails": "full", "sortOrder": "desc", "filters": {"status": "succeeded"},
                                  **({"paginationToken": token} if token else {})})
            data = res.get("data") or []
            rows = _graduations_from_page(data); added += _upsert_graduations(rows)
            token = res.get("paginationToken")
            bts = [int(t["blockTime"]) for t in data if t.get("blockTime")]
            if bts:
                oldest = min(bts)
            state.update({"token": token, "oldest_bt": oldest})
            status.update(stage="enumerating (backfill)", backfill_through=datetime.fromtimestamp(oldest, timezone.utc).isoformat() if oldest else None,
                          backfill_target=datetime.fromtimestamp(horizon, timezone.utc).date().isoformat(), enum_added=added)
            if not token or (oldest is not None and oldest < horizon):
                state["backfill_done"] = True; break
        _save_enum_state(state)
    status.update(backfill_done=bool(state.get("backfill_done")), force=True)
    return added


# ---------------------------------------------------------------- per-token pull
def _write_atomic(table: pa.Table, path: Path) -> None:
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp, compression="zstd"); tmp.replace(path)


def _candle_rows(cands: list[dict], interval: str) -> list[dict]:
    return [{"ts": datetime.fromtimestamp(int(c["timestamp"]) / 1000, timezone.utc), "interval": interval, "open": float(c["open"]),
             "high": float(c["high"]), "low": float(c["low"]), "close": float(c["close"]), "volume_sol": float(c["volume"])} for c in cands]


def _page_back(api: SwapApi, mint: str, interval: str, created_ms: int, before_ms: int, until_ms: int, max_pages: int) -> list[dict]:
    """Candles with timestamp in [until_ms, before_ms), paging backwards."""
    out, before = [], before_ms
    for _ in range(max_pages):
        page = api.candles(mint, interval, created_ms, before)
        if not page:
            break
        out.extend(c for c in page if until_ms <= int(c["timestamp"]) < before_ms)
        lo = min(int(c["timestamp"]) for c in page)
        if lo <= until_ms or len(page) < 1000:
            break
        before = lo
    return out


def pull_token(api: SwapApi, row: dict, status: Status) -> dict:
    """Two to four swap-api requests per token: 1-minute candles for the first CORPUS_CANDLE_1M_H hours (plus the last
    30 curve minutes), then the newest 5-minute page (which also tells how long the token lived), paged back to the
    1-minute window and capped at CORPUS_CANDLE_5M_D days. Trades (first CORPUS_TRADES_H hours) for a deterministic
    CORPUS_TRADES_SAMPLE share of tokens that lived ≥ CORPUS_MIN_LIFE_H and graduated within CORPUS_TRADES_DAYS."""
    mint: str = row["mint"]; grad: datetime = row["graduated_at"]
    g_ms = int(grad.timestamp() * 1000); created_ms = g_ms - 86_400_000
    cdir = config.CORPUS_DIR / "candles"; tdir = config.CORPUS_DIR / "trades"
    cpath, tpath = cdir / f"{mint}.parquet", tdir / f"{mint}.parquet"
    upd: dict = {"status": "done", "candles_1m": 0, "candles_5m": 0, "trades": 0, "trades_through": None, "candle_path": None, "trade_path": None}
    start_ms = g_ms - 30 * 60_000; win1_end = g_ms + int(config.CORPUS_CANDLE_1M_H * 3.6e6); win2_end = g_ms + int(config.CORPUS_CANDLE_5M_D * 86_400_000)
    one_m = _page_back(api, mint, "1m", created_ms, win1_end, start_ms, max_pages=3)
    newest5 = api.candles(mint, "5m", created_ms)
    stamps = [int(c["timestamp"]) for c in one_m] + [int(c["timestamp"]) for c in newest5]
    if not stamps:
        upd["status"] = "empty"; return upd
    last_ms = max(stamps); life_h = (last_ms - g_ms) / 3.6e6; upd["life_h"] = life_h
    five_m = [c for c in newest5 if win1_end <= int(c["timestamp"]) < win2_end]
    if newest5 and len(newest5) >= 1000 and min(int(c["timestamp"]) for c in newest5) > win1_end:
        lo5 = min(int(c["timestamp"]) for c in newest5)
        pages = min(20, int((min(lo5, win2_end) - win1_end) / (1000 * 300_000)) + 2)     # enough pages to reach the 1-minute window's end
        five_m += _page_back(api, mint, "5m", created_ms, min(lo5, win2_end), win1_end, max_pages=pages)
        five_m = [c for c in five_m if int(c["timestamp"]) < win2_end]
    rows = _candle_rows(one_m, "1m") + _candle_rows(five_m, "5m"); upd["candles_1m"] = len(one_m); upd["candles_5m"] = len(five_m)
    if rows and not cpath.exists():
        _write_atomic(pa.Table.from_pylist(rows, schema=CANDLE_SCHEMA), cpath)
    upd["candle_path"] = str(cpath) if cpath.exists() else None
    status.bump("candles_rows", len(rows))
    recent = grad >= datetime.now(timezone.utc) - timedelta(days=config.CORPUS_TRADES_DAYS)
    sampled = (int.from_bytes(hashlib.sha1(mint.encode()).digest()[:4], "big") / 2**32) < config.CORPUS_TRADES_SAMPLE
    if life_h >= config.CORPUS_MIN_LIFE_H and recent and sampled and not tpath.exists():
        end_ms = min(g_ms + int(config.CORPUS_TRADES_H * 3.6e6), last_ms + 60_000)
        slot_t = int(row["grad_slot"] or 0) + int((end_ms - g_ms) / 1000 * SLOTS_PER_S)
        cursor: str | None = f"{slot_t:010d}000000000000-{end_ms}"
        trows: list[dict] = []; pages = 0; oldest_seen = end_ms
        while cursor and pages < config.CORPUS_TRADES_MAX_PAGES:
            trades, cursor = api.trades(mint, cursor); pages += 1
            if not trades:
                break
            for t in trades:
                ts = datetime.fromisoformat(t["timestamp"].replace("Z", "+00:00")); ms = int(ts.timestamp() * 1000)
                oldest_seen = min(oldest_seen, ms)
                if ms < start_ms:
                    continue
                trows.append({"ts": ts, "slot_index": t.get("slotIndexId"), "tx": t.get("tx"), "wallet": t.get("userAddress"),
                              "side": 1 if t.get("type") == "buy" else -1, "program": t.get("program"),
                              "price_sol": float(t.get("priceSol") or 0.0), "sol": float(t.get("amountSol") or 0.0), "tokens": float(t.get("baseAmount") or 0.0)})
            if oldest_seen < start_ms:
                break
        if trows:
            _write_atomic(pa.Table.from_pylist(trows, schema=TRADE_SCHEMA), tpath)
            upd.update(trades=len(trows), trades_through=datetime.fromtimestamp(end_ms / 1000, timezone.utc), trade_path=str(tpath))
            status.bump("trades_rows", len(trows)); status.bump("with_trades")
    return upd


def _next_pending(limit: int = 50) -> list[dict]:
    with transaction() as conn:
        return conn.execute("SELECT mint, graduated_at, grad_slot FROM corpus_tokens WHERE status = 'pending' AND graduated_at < now() - %s * interval '1 hour' "
                            "AND graduated_at < %s::timestamptz ORDER BY graduated_at DESC LIMIT %s", (config.CORPUS_MIN_AGE_H, config.CORPUS_PULL_BEFORE, limit)).fetchall()


def _mark(mint: str, upd: dict) -> None:
    with transaction() as conn:
        conn.execute("UPDATE corpus_tokens SET status = %s, life_h = %s, candles_1m = %s, candles_5m = %s, trades = %s, trades_through = %s, "
                     "candle_path = %s, trade_path = %s, last_error = %s, updated_at = now() WHERE mint = %s",
                     (upd.get("status", "done"), upd.get("life_h"), upd.get("candles_1m"), upd.get("candles_5m"), upd.get("trades"), upd.get("trades_through"),
                      upd.get("candle_path"), upd.get("trade_path"), upd.get("last_error"), mint))


def counts() -> dict:
    with transaction() as conn:
        rows = conn.execute("SELECT status, count(*) AS n FROM corpus_tokens GROUP BY status").fetchall()
    return {r["status"]: int(r["n"]) for r in rows}


# ---------------------------------------------------------------- worker entry
def main(stop_event: threading.Event | None = None) -> None:
    setup("corpus")
    status = Status(); stop = stop_event
    (config.CORPUS_DIR / "candles").mkdir(parents=True, exist_ok=True); (config.CORPUS_DIR / "trades").mkdir(parents=True, exist_ok=True)
    pacer = Pacer(config.CORPUS_PACE_S, config.CORPUS_PACE_MIN_S); api = SwapApi(pacer, status, stop); hel = Helius(stop)
    record_event("info", "corpus", "corpus puller started", {"days": config.CORPUS_DAYS, "trades_h": config.CORPUS_TRADES_H, "pace_s": pacer.pace})
    log.info("corpus puller started (days=%d)", config.CORPUS_DAYS)
    from ..train import corpus_features
    builder = threading.Thread(target=corpus_features.build_loop, args=(stop,), name="corpus-features", daemon=True); builder.start()
    try:
        while not (stop is not None and stop.is_set()):
            try:
                added = enumerate_graduations(hel, status, stop)
                if added:
                    log.info("enumerated %d new graduations", added)
            except InterruptedError:
                break
            except Exception as e:
                log.exception("enumeration failed"); record_event("error", "corpus", f"enumeration failed: {type(e).__name__}: {e}")
                status.update(stage="enumeration error", last_error=str(e)[:200], force=True); _sleep(30, stop)
            pending = _next_pending()
            c = counts(); status.update(stage="pulling" if pending else "idle", pending=c.get("pending", 0), done=c.get("done", 0), empty=c.get("empty", 0),
                                        errors=c.get("error", 0), enumerated=sum(c.values()), pace_s=round(pacer.pace, 2), force=True)
            if not pending:
                _sleep(60, stop); continue
            for row in pending:
                if stop is not None and stop.is_set():
                    break
                try:
                    upd = pull_token(api, row, status)
                except InterruptedError:
                    break
                except Exception as e:
                    log.warning("pull %s failed: %s", row["mint"], e); upd = {"status": "error", "last_error": f"{type(e).__name__}: {e}"[:300]}
                _mark(row["mint"], upd); status.bump("tokens_pulled")
                c = counts()
                status.update(stage="pulling", last_mint=row["mint"], last_graduated=row["graduated_at"].isoformat(), pending=c.get("pending", 0), done=c.get("done", 0),
                              empty=c.get("empty", 0), errors=c.get("error", 0), enumerated=sum(c.values()), pace_s=round(pacer.pace, 2),
                              eta_h=(c.get("pending", 0) / status.s["tokens_per_h"]) if status.s.get("tokens_per_h") else None)
    finally:
        builder.join(30.0)
        status.update(stage="stopped", force=True)
        record_event("info", "corpus", "corpus puller stopped", {k: status.s.get(k) for k in ("tokens_pulled", "candles_rows", "trades_rows", "req_ok", "req_429")})
        log.info("corpus puller stopped")
