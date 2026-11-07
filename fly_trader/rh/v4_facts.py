"""A plain v4 coin's launch facts (2026-11-02): what the universe's creator rules and the model's launch inputs read in
``rh_meta``. A coin without them never trades: rh/meta.build_v4 leaves them unknown and train/selector.in_universe refuses
an unknown launch. The definitions are the ones selector #791 trained on (2,469 coins of the v4 history), from the coin's
Transfer logs and code sizes:

- the launch block L: the first block in which a contract that ever held ``BIG`` of the supply (or the PoolManager) pays
  tokens to anyone but another such holder, the mint aside. That is a launch-transaction buy on a factory's Uniswap v3 pool
  or curve, or the first buy of a coin seeded by hand into a v4 pool;
- the holdings at L's end: the PoolManager, the zero and dead addresses and the token itself are nobody's; a contract that
  paid at least ``MARKET_PAYEES`` addresses in the pulled window is a market (a pool, a curve); everything else is owned, a
  contract that pays nobody included, so it fails closed. A wallet: code of at most ``WALLET_CODE`` bytes (an EIP-7702
  delegation is 23, a clone 45);
- the creator: the sender of the mint transaction. dev_share = dev_hold_share = its share, bundle_share = the rest owned
  (those holders go to rh_insiders), grad_hhi = the concentration of every holding but the markets';
- graduated_at = L's time (the age floor counts from it), ttg_min = mint to L, rq0 = the quote reserve in ETH at the first
  swap of the coin's registered pools, curve_known = 1, its creator's history as rh/meta._history.

``fly-trader rh-v4-facts`` is a child of the stream at the lowest CPU priority (rh/stream.facts_apart), never in a minute's
path, its HyperSync calls in a slot of each minute behind live's (``_slot``) and the header endpoint, not the transaction
endpoint live orders use. Each run takes the coins that clear the floors now and near the universe's age, busiest first; the engine reads their
row within ``META_TTL_S``. A coin without a launch is tried again after ``RETRY_H``, and stays out until it has one.
"""
from __future__ import annotations

import fcntl
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from .. import config
from ..db.connection import transaction
from . import hypersync, index, meta, prices, v4, v4_universe
from .rpc import logs_rpc

log = logging.getLogger(__name__)
ZERO = "0x" + "00" * 20
DEAD = {ZERO, "0x000000000000000000000000000000000000dead"}
WALLET_CODE = 100               # bytes of code a wallet can have
MARKET_PAYEES = 3               # distinct addresses a contract paid in the window, to count as a market
MIN_SHARE = 0.002               # a holding below this share of the supply is nobody's in particular
BIG = 0.05                      # only an address that ever holds this share can be a pool or a curve, so only those are looked up
AFTER_INIT = 600                # blocks of Transfers past the first pool's Initialize (a minute)
AFTER_L = 300                   # and past L
WINDOWS = (36_000, 360_000, 3_600_000)   # blocks past a pool's start a later launch or first swap is looked for in (~1, 10, 100 h), never up to the head: a busy coin's whole history
EARLY_H = 1.0                   # a coin is taken this long before the universe's 6 h
MAX_COINS = 20                  # coins a run
RUN_S = 480.0                   # no new coin is started after this long
RETRY_H = 24.0                  # a coin that had no launch is tried again after this long
CODE_BATCH = 8
SLOT_S = (8.0, 20.0)            # a HyperSync call starts this far into a minute: the minute that ended is written, its live orders out
SLOT_CALLS = 2                  # at most this many a minute: since 2026-11-02 18:01 HyperSync answers 10 a minute, and the live tail takes the rest
SLOT_WAIT_S = 300.0             # a call that finds no slot this long raises: its coin is tried again next run
LAG_BLOCKS = 120                # the live index this far behind needs every call: no run
_SLOT = {"minute": None, "n": 0}
LOCK = config.DATA_DIR / ".rh_v4_facts.lock"


class Codes:
    """Code sizes, each address read once. An address that cannot be read raises: its coin is tried again next run."""

    def __init__(self, rpc):
        self.rpc = rpc; self.size: dict[str, int] = {ZERO: 0}

    def need(self, addrs) -> None:
        pm = config.V4_POOL_MANAGER.lower(); todo = sorted({a for a in addrs if a not in self.size and a != pm})
        for i in range(0, len(todo), CODE_BATCH):
            ch = todo[i:i + CODE_BATCH]
            for a, v in zip(ch, self.rpc.batch([("eth_getCode", [a, "latest"]) for a in ch])):
                if not isinstance(v, str):
                    raise RuntimeError(f"no code size for {a}: {v!r}"[:200])
                self.size[a] = (len(v) - 2) // 2

    def wallet(self, a: str) -> bool:
        return a != config.V4_POOL_MANAGER.lower() and self.size.get(a, 0) <= WALLET_CODE


def launch(token: str, rows: list[tuple], codes: Codes) -> dict | None:
    """L and the holdings at its end from a coin's Transfers in order, (block, tx, src, dst, value), all lower case; L is
    None when none was found in ``rows``. None when nothing was minted."""
    pm = config.V4_POOL_MANAGER.lower(); token = token.lower()
    minted = sum(float(r[4]) for r in rows if r[2] == ZERO)
    if minted <= 0:
        return None
    first = next(r for r in rows if r[2] == ZERO)
    out = {"minted": minted, "mint_block": first[0], "mint_tx": first[1], "L": None}
    payees: dict[str, set] = defaultdict(set)
    for _b, _t, s, d, _v in rows:
        if s != ZERO:
            payees[s].add(d)
    run: dict[str, float] = defaultdict(float); big = {pm}
    for b, _t, s, d, v in rows:
        v = float(v); run[s] -= v; run[d] += v
        if s == ZERO or v <= 0 or d in DEAD or d == token:
            continue
        if run[d] / minted >= BIG and d not in big:
            codes.need([d]); big.add(d)
        if s in big and not codes.wallet(s) and d not in big:
            out["L"] = b; break
    if out["L"] is None:
        return out
    bal: dict[str, float] = defaultdict(float)
    for b, _t, s, d, v in rows:
        if b > out["L"]:
            break
        bal[s] -= float(v); bal[d] += float(v)
    hold = {a: v for a, v in bal.items() if a != pm and a not in DEAD and a != token and v / minted >= MIN_SHARE}
    codes.need(hold)
    market = {a for a in hold if not codes.wallet(a) and len(payees[a]) >= MARKET_PAYEES}
    owned = {a: v / minted for a, v in hold.items() if a not in market}
    rest = [v for a, v in bal.items() if v > 0 and a != pm and a not in DEAD and a != token and a not in market]
    total = sum(rest)
    out.update(owned=owned, hhi=sum((v / total) ** 2 for v in rest) if rest else None)
    return out


def _transfers(rpc, token: str, lo: int, hi: int) -> list[tuple]:
    return [(int(x["blockNumber"], 16), x["transactionHash"], "0x" + x["topics"][1][-40:].lower(), "0x" + x["topics"][2][-40:].lower(),
             int(x["data"], 16) if x["data"] not in ("0x", "") else 0)
            for x in index._order(index._get_logs(rpc, [token], lo, hi, [[index.TRANSFER]])) if len(x.get("topics") or ()) == 3]


def find_launch(rpc, codes: Codes, token: str, init: int, head: int) -> dict | None:
    """``launch`` on the coin's Transfers from block 0 to a minute past its first pool (``init``), further out while no
    launch shows, and to ``AFTER_L`` past L: the window the payee count and the history's holdings saw."""
    hi = min(head, init + AFTER_INIT); rows = _transfers(rpc, token, 0, hi); f = launch(token, rows, codes)
    for w in WINDOWS:
        if f is None or f["L"] is not None or hi >= head:
            break
        to = min(head, init + w)
        if to > hi:
            rows += _transfers(rpc, token, hi + 1, to); hi = to; f = launch(token, rows, codes)
    if f is not None and f["L"] is not None and f["L"] + AFTER_L > hi and hi < head:
        to = min(head, f["L"] + AFTER_L); rows += _transfers(rpc, token, hi + 1, to); f = launch(token, rows, codes)
    return f


def first_swap(rpc, pools: list[str], lo: int, head: int) -> dict | None:
    """The first Swap of ``pools`` within ``WINDOWS`` of block ``lo``, or None: rq0 stays unknown."""
    frm = lo
    for w in WINDOWS:
        to = min(head, lo + w)
        if to >= frm:
            logs = index._get_logs(rpc, [config.V4_POOL_MANAGER], frm, to, [[v4.TOPIC["Swap"]], pools])
            if logs:
                return v4.decode(index._order(logs)[0])
        frm = to + 1
    return None


def block_time(rpc, block: int) -> datetime:
    src = getattr(rpc, "blocks", rpc)
    return datetime.fromtimestamp(int(src.call("eth_getBlockByNumber", [hex(block), False])["timestamp"], 16), timezone.utc)


def facts(rpc, codes: Codes, c: dict, head: int) -> dict:
    """A coin's rh_meta facts from the chain, or {"why": ...} when it has no launch. ``c``: {"token", "first_block", "pools":
    [{"pool_id", "init_block", "quote_asset", "token_is_0", "qdec"}]}. No database."""
    token = c["token"].lower(); f = find_launch(rpc, codes, token, int(c["first_block"]), head)
    if f is None:
        return {"why": "nothing minted in its Transfers"}
    if f["L"] is None:
        return {"why": "no launch found"}
    src = getattr(rpc, "blocks", rpc); tx = src.tx_by_hash(f["mint_tx"])
    creator = ((tx or {}).get("from") or "").lower() or None
    if creator is None:
        raise RuntimeError(f"mint transaction {f['mint_tx']} of {token} not found")
    g, created = block_time(rpc, f["L"]), block_time(rpc, f["mint_block"])
    own = f["owned"]; dev = float(own.get(creator, 0.0)); total = sum(own.values()); supply = f["minted"] / 1e18
    out = {"graduated_at": g, "create_ts": created, "creator": creator, "supply": supply, "ttg_min": max(0.0, (g - created).total_seconds() / 60.0), "dev_sol": 0.0,
           "dev_tokens": dev * supply, "dev_share": dev, "bundle_share": max(0.0, total - dev), "dev_hold_share": dev, "dev_sold_frac": 0.0, "grad_hhi": f["hhi"],
           "curve_known": 1.0, "owned": own, "rq0": None}
    start = min((int(p["init_block"]) for p in c["pools"] if p.get("init_block") is not None), default=int(c["first_block"]))
    s = first_swap(rpc, [p["pool_id"] for p in c["pools"]], start, head) if c["pools"] else None
    if s is not None:
        p = next(x for x in c["pools"] if x["pool_id"] == s["pool_id"])
        out["first_swap"] = {"at": block_time(rpc, s["block"]), "quote_asset": p["quote_asset"],
                             "reserve": v4.quote_reserve(int(s["sqrt_price_x96"]), int(s["liquidity"]), bool(p["token_is_0"]), int(p["qdec"]))}
    return out


FIELDS = ("graduated_at", "create_ts", "creator", "supply", "ttg_min", "dev_sol", "dev_tokens", "dev_share", "bundle_share", "dev_hold_share", "dev_sold_frac", "grad_hhi",
          "curve_known", "rq0", "prior_launches", "prior_grads", "prior_known", "prior_rug_share", "prior_moon_share", "own_dd60", "own_max60")


def write(conn, token: str, f: dict) -> bool:
    """``facts`` into the coin's rh_meta row and rh_insiders, with its creator's history and its first hour where the
    minutes hold it. A row that has its facts stays as it is."""
    row = {k: f.get(k) for k in FIELDS}; s = f.get("first_swap")
    if s is not None:
        base = prices.base_eth(conn, s["quote_asset"], s["at"])[0]
        row["rq0"] = s["reserve"] * base if base is not None else None
    row.update(meta._history(conn, f["creator"], f["create_ts"], f["graduated_at"], token))
    o = meta.outcomes(conn, token, f["graduated_at"])
    if o is not None:
        row["own_dd60"], row["own_max60"] = o
    n = conn.execute(f"UPDATE rh_meta SET {', '.join(k + ' = %s' for k in FIELDS)}, updated_at = now() WHERE mint = %s AND source = 'v4' AND ttg_min IS NULL",
                     [row.get(k) for k in FIELDS] + [token]).rowcount
    if n:
        for w in f["owned"]:
            conn.execute("INSERT INTO rh_insiders (mint, wallet, kind) VALUES (%s, %s, %s) ON CONFLICT DO NOTHING", (token, w, "dev" if w == f["creator"] else "bundle"))
    return bool(n)


def note_failure(conn, token: str, why: str) -> None:
    conn.execute("""UPDATE rh_v4_tokens SET detail = coalesce(detail, '{}'::jsonb) || jsonb_build_object('facts', jsonb_build_object('at', now(), 'why', %s::text)) WHERE token = %s""", (why, token))


def due(conn, limit: int = MAX_COINS, now: datetime | None = None) -> list[dict]:
    """The plain coins without facts, busiest first: past the universe's age less ``EARLY_H``, near the floors over the last half hour. Their registered pools."""
    now = now or datetime.now(timezone.utc); res, vol = v4_universe.floors()
    rows = conn.execute("""WITH a AS (SELECT mint, max(resq_eth) AS resq, sum(buy_eth + sell_eth) AS vol FROM rh_minutes WHERE ts >= %(since)s GROUP BY 1)
                           SELECT m.mint AS token, v.first_block FROM a JOIN rh_meta m ON m.mint = a.mint JOIN rh_v4_tokens v ON v.token = m.mint
                           WHERE m.source = 'v4' AND m.ttg_min IS NULL AND a.resq >= %(res)s AND a.vol >= %(vol)s AND v.first_block IS NOT NULL AND m.graduated_at <= %(aged)s
                             AND (v.detail -> 'facts' IS NULL OR (v.detail -> 'facts' ->> 'at')::timestamptz <= %(retry)s)
                           ORDER BY a.vol DESC LIMIT %(n)s""",
                        {"since": now - timedelta(minutes=30), "res": res, "vol": vol, "aged": now - timedelta(hours=6.0 - EARLY_H), "retry": now - timedelta(hours=RETRY_H), "n": limit}).fetchall()
    out = []
    for r in rows:
        ps = conn.execute("""SELECT p.pool_id, p.init_block, p.quote_asset, p.token_is_0, coalesce(a.decimals, 18) AS qdec FROM rh_pools p LEFT JOIN rh_assets a ON a.asset = p.quote_asset
                             WHERE p.token = %s AND p.kind = 'v4'""", (r["token"],)).fetchall()
        if ps:
            out.append({"token": r["token"], "first_block": int(r["first_block"]), "pools": [dict(p) for p in ps]})
    return out


def live_trouble(conn) -> str | None:
    """Why a run waits for the next, or None: HyperSync failing other than by its 429 cap in the last 5 minutes (an outage:
    live logs need every call), or the live index behind or silent. The cap's 429s come every minute; ``_slot`` keeps a
    run's calls behind live's."""
    n = int(conn.execute("""SELECT count(*) AS n FROM api_calls WHERE service = 'rh_hypersync' AND ts > now() - interval '5 minutes'
                            AND ok IS FALSE AND status IS DISTINCT FROM 429""").fetchone()["n"])
    if n:
        return f"{n} HyperSync failures in 5 minutes"
    r = conn.execute("SELECT value FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()
    s = (r["value"] if isinstance(r["value"], dict) else json.loads(r["value"])) if r else {}
    age = time.time() - datetime.fromisoformat(s["updated_at"]).timestamp() if s.get("updated_at") else None
    if age is None or age > 180.0:
        return "the stream is silent"
    if s.get("mode") != "live" or (s.get("behind_blocks") or 0) > LAG_BLOCKS:
        return f"the live index is {s.get('behind_blocks')} blocks behind"
    return None


def _minute_written(end: int) -> bool:
    """The stream has written the minute ending at ``end`` (``flushed_through``: the newest written minute's start, as the engine reads it)."""
    with transaction() as conn:
        r = conn.execute("SELECT value->>'flushed_through' AS ft FROM ui_settings WHERE key = 'rh_stream_status'").fetchone()
    return bool(r and r["ft"]) and datetime.fromisoformat(r["ft"]).timestamp() + 60.0 >= end


def _slot(clock=time.time, sleep=time.sleep) -> None:
    """Each HyperSync call of a run waits here (in place of hypersync._pace, in this process only): ``SLOT_S`` into a
    minute, once the stream has written the minute that ended, ``SLOT_CALLS`` a minute at most. A 429's retry waits here
    too, so for a later slot."""
    t0 = clock()
    while True:
        t = clock(); m = int(t // 60); s = t - m * 60
        if t - t0 > SLOT_WAIT_S:
            raise TimeoutError(f"no HyperSync slot in {SLOT_WAIT_S:.0f} s")
        if _SLOT["minute"] != m:
            _SLOT.update(minute=m, n=0)
        if s < SLOT_S[0]:
            sleep(SLOT_S[0] - s)
        elif s > SLOT_S[1] or _SLOT["n"] >= SLOT_CALLS:
            sleep(60.0 - s + SLOT_S[0])
        elif not _minute_written(m * 60):
            sleep(1.0)
        else:
            _SLOT["n"] += 1
            return


def run_once(limit: int = MAX_COINS) -> dict:
    hypersync._pace = _slot                     # this process's HyperSync calls are all a run's: each takes the slot
    out = {"due": 0, "written": 0, "no_launch": 0, "failed": 0}
    with transaction() as conn:
        why = live_trouble(conn)
        if why:
            log.info("rh v4 facts: no run, %s", why)
            return {**out, "paused": why}
        coins = due(conn, limit); cur = conn.execute("SELECT block FROM rh_scan WHERE name = %s", (index.CURSOR,)).fetchone()
    out["due"] = len(coins)
    if not coins or not cur:
        return out
    rpc = logs_rpc(); codes = Codes(getattr(rpc, "blocks", rpc)); head = int(cur["block"]); start = time.monotonic()
    for c in coins:
        t0 = time.monotonic()
        if t0 - start > RUN_S:
            break
        try:
            f = facts(rpc, codes, c, head)
            with transaction() as conn:
                if "why" in f:
                    note_failure(conn, c["token"], f["why"]); out["no_launch"] += 1
                elif write(conn, c["token"], f):
                    out["written"] += 1
            log.info("rh v4 facts %s: %s (%.1f s)", c["token"][:10], f.get("why") or {k: f.get(k) for k in ("dev_share", "bundle_share", "ttg_min")}, time.monotonic() - t0)
        except Exception as e:                  # the next run tries it again: nothing was written
            out["failed"] += 1; log.warning("rh v4 facts %s failed: %s: %s", c["token"][:10], type(e).__name__, str(e)[:200])
    return out


def run_locked(limit: int = MAX_COINS) -> dict:
    """``run_once`` under a file lock, one run at a time (``fly-trader rh-v4-facts``)."""
    with open(LOCK, "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return {"busy": True}
        return run_once(limit)

