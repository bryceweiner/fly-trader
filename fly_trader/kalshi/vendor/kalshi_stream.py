# Vendored verbatim from better_bot code/kalshi_stream.py by tools/vendor_kalshi.py; sports sections removed. Do not edit.
"""The two broadcast channels: who wants which combo, and what the board is doing.

Kalshi broadcasts every RFQ any member creates to every authenticated
subscriber, and ``rfq_created`` carries ``mve_selected_legs`` -- so the demand
side of the combo market is public, structure and size included, before anyone
quotes it. That is the training set for a combo pricing model and the map of
what would be worth quoting, and it is the one input that cannot be recovered
retroactively: nothing on the REST API remembers an RFQ that was never filled.

**What is NOT public, and the correction that matters.** Only ``rfq_created``
and ``rfq_deleted`` reach every subscriber. ``quote_created``,
``quote_accepted`` and ``quote_executed`` go solely to the RFQ's creator and the
quoting maker, so this feed shows demand and never the prices makers answered
with. Reading it as "we can see the whole RFQ book" would be wrong.

**It is a firehose, and that shapes everything here.** Measured over a 40-second
window on 2026-08-09: 30,570 ``rfq_created``, 30,161 ``rfq_deleted`` and 12,989
multivariate lifecycle messages -- roughly 760 RFQs a second, created and
cancelled in near-equal numbers by bots probing for prices. Persisting raw
messages would write ~65 million rows a day of almost pure churn. So this
aggregates in memory by STRUCTURE -- the sorted set of ``(market_ticker,
side)`` legs -- and flushes counts periodically. The same combo requested five
hundred times is one row with a count of five hundred, which is also the more
useful shape: it says what the market keeps asking for.

Endpoint note, because the published docs are wrong about it: the per-channel
URLs in Kalshi's websocket reference (``wss://external-api-ws.kalshi.com/
communications``) return HTTP 404. Both channels are served on the ordinary v2
socket and selected with a ``subscribe`` command, which is what
:data:`WS_PATH` and :func:`subscribe_commands` implement.

Run it with ``scripts/kalshi_stream.py``. It is deliberately NOT wired into the
daily cycle: the cycle is a batch job that exits, and a feed whose whole value
is continuity does not belong inside one.
"""

import asyncio
import json
import logging
import time

from .config import KALSHI_API_KEY_ID
from .prediction_utils import utc_now
from .db_connection import get_connection
from .kalshi_client import _kalshi_available, _load_private_key

logger = logging.getLogger(__name__)

#: The v2 socket, not the per-channel URLs the docs advertise -- those 404.
WS_HOST = "wss://api.elections.kalshi.com"
WS_PATH = "/trade-api/ws/v2"

#: Both broadcast channels. ``communications`` carries the RFQ flow;
#: ``multivariate_market_lifecycle`` announces combo markets as they are created
#: and settled, which is how a ticker becomes visible intraday rather than at
#: the next daily listing walk.
CHANNELS = ("communications", "multivariate_market_lifecycle")

#: Lifecycle events worth recording. The channel is dominated by
#: ``close_date_updated`` churn -- 12,989 messages in 40 seconds, nearly all of
#: it re-stamping close times -- and none of that says anything we act on.
LIFECYCLE_EVENTS = frozenset({"created", "determined", "settled"})

#: How often the in-memory aggregate is written. Long enough that the firehose
#: collapses into a few hundred rows, short enough that a crash loses minutes
#: rather than a session.
FLUSH_SECONDS = 60

#: Reconnect backoff, doubling to this ceiling. A long-lived socket WILL drop.
RECONNECT_MIN_SECONDS = 2
RECONNECT_MAX_SECONDS = 120


def subscribe_commands(channels=CHANNELS):
    """The subscribe frames to send once the socket is open."""
    return [{"id": index, "cmd": "subscribe", "params": {"channels": [channel]}}
            for index, channel in enumerate(channels, start=1)]


def structure_fingerprint(legs):
    """A stable id for a combo's shape, order-independent.

    The RFQ id is useless as a key -- every probe mints a new one -- and the
    market ticker is nearly as bad, since a combo market is created per request.
    What repeats, and what is worth counting, is the SET of legs.
    """
    return "|".join(sorted(
        f"{leg.get('market_ticker')}:{leg.get('side') or 'yes'}"
        for leg in legs))


def _number(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


class DemandAggregate:
    """In-memory tally of requested combo structures, flushed periodically.

    Deliberately not a cache in front of the database: at ~760 RFQs a second an
    upsert per message would spend the whole budget on write amplification for
    rows that are re-written milliseconds later.
    """

    def __init__(self):
        self.structures = {}
        self.channel_counts = {}

    def note_message(self, message_type):
        self.channel_counts[message_type] = (
            self.channel_counts.get(message_type, 0) + 1)

    def note_rfq(self, msg):
        """Record one ``rfq_created``. Ignores RFQs that are not combos.

        A single-market RFQ is somebody asking for a price on one contract; it
        carries no joint structure and is not what this feed is for.
        """
        legs = msg.get("mve_selected_legs") or []
        if len(legs) < 2:
            return False
        fingerprint = structure_fingerprint(legs)
        entry = self.structures.get(fingerprint)
        if entry is None:
            entry = self.structures[fingerprint] = {
                "fingerprint": fingerprint,
                "collection_ticker": msg.get("mve_collection_ticker"),
                "num_legs": len(legs),
                "legs_json": json.dumps(
                    [{"market_ticker": leg.get("market_ticker"),
                      "event_ticker": leg.get("event_ticker"),
                      "side": leg.get("side") or "yes"} for leg in legs],
                    sort_keys=True),
                # A combo whose legs all name one game is the population the
                # dependence experiment cares about; carrying the flag here
                # saves re-deriving it from legs_json later.
                "is_same_game": int(len({leg.get("event_ticker")
                                         for leg in legs}) == 1),
                "rfq_count": 0,
                "total_contracts": 0.0,
                "total_target_cost": 0.0,
            }
        entry["rfq_count"] += 1
        entry["total_contracts"] += _number(msg.get("contracts_fp"))
        entry["total_target_cost"] += _number(msg.get("target_cost_dollars"))
        return True

    @property
    def is_empty(self):
        return not self.structures and not self.channel_counts

    def drain(self):
        """Take everything accumulated so far and reset."""
        structures = list(self.structures.values())
        counts = dict(self.channel_counts)
        self.structures = {}
        self.channel_counts = {}
        return structures, counts


def flush(aggregate, conn=None):
    """Persist one window. Returns ``{structures, messages}``.

    Counts ADD on conflict rather than replace, so a structure seen across many
    windows accumulates one total instead of keeping only the last window's.
    """
    structures, counts = aggregate.drain()
    if not structures and not counts:
        return {"structures": 0, "messages": 0}
    owned = conn is None
    conn = conn or get_connection()
    try:
        seen_at = utc_now()
        conn.executemany(
            """INSERT INTO kalshi_rfq_demand
                   (fingerprint, collection_ticker, num_legs, legs_json,
                    is_same_game, rfq_count, total_contracts,
                    total_target_cost, first_seen, last_seen)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(fingerprint) DO UPDATE SET
                   rfq_count = rfq_count + excluded.rfq_count,
                   total_contracts = total_contracts + excluded.total_contracts,
                   total_target_cost =
                       total_target_cost + excluded.total_target_cost,
                   last_seen = excluded.last_seen""",
            [(s["fingerprint"], s["collection_ticker"], s["num_legs"],
              s["legs_json"], s["is_same_game"], s["rfq_count"],
              s["total_contracts"], s["total_target_cost"], seen_at, seen_at)
             for s in structures])
        conn.executemany(
            """INSERT INTO kalshi_stream_stats
                   (window_end, message_type, message_count)
               VALUES (?, ?, ?)""",
            [(seen_at, name, count) for name, count in sorted(counts.items())])
        conn.commit()
        return {"structures": len(structures), "messages": sum(counts.values())}
    finally:
        if owned:
            conn.close()


def _auth_headers():
    """Handshake headers, signed by the SDK's own signer.

    Reused rather than reimplemented: the signature is over
    ``timestamp + METHOD + path`` with RSA-PSS, and a hand-rolled second copy
    would be one upstream change away from silently failing to authenticate.
    """
    from kalshi_python_async.auth import KalshiAuth

    private_key = _load_private_key()
    if not (KALSHI_API_KEY_ID and private_key):
        return None
    return KalshiAuth(KALSHI_API_KEY_ID, private_key).create_auth_headers(
        "GET", WS_PATH)


async def _consume(aggregate, *, deadline=None, flush_seconds=FLUSH_SECONDS,
                   conn=None, channels=CHANNELS):
    """One connection's lifetime: subscribe, aggregate, flush on a timer."""
    import websockets

    headers = _auth_headers()
    if headers is None:
        raise RuntimeError("Kalshi websocket needs KALSHI_API_KEY_ID and a "
                           "readable KALSHI_PRIVATE_KEY_PATH")
    async with websockets.connect(WS_HOST + WS_PATH,
                                  additional_headers=headers,
                                  open_timeout=15,
                                  max_queue=None) as socket:
        for command in subscribe_commands(channels):
            await socket.send(json.dumps(command))
        logger.info("kalshi stream: subscribed to %s", ", ".join(channels))
        next_flush = time.monotonic() + flush_seconds
        while deadline is None or time.monotonic() < deadline:
            timeout = max(0.5, next_flush - time.monotonic())
            try:
                raw = await asyncio.wait_for(socket.recv(), timeout)
            except asyncio.TimeoutError:
                raw = None
            if raw is not None:
                handle(aggregate, raw)
            if time.monotonic() >= next_flush:
                written = flush(aggregate, conn=conn)
                if written["messages"]:
                    logger.info("kalshi stream: %s", written)
                next_flush = time.monotonic() + flush_seconds


def handle(aggregate, raw):
    """Route one frame into the aggregate. Malformed frames are counted, not raised."""
    try:
        message = json.loads(raw)
    except (TypeError, ValueError):
        aggregate.note_message("unparseable")
        return
    message_type = message.get("type")
    payload = message.get("msg") or {}
    if message_type == "multivariate_market_lifecycle":
        event_type = payload.get("event_type")
        if event_type not in LIFECYCLE_EVENTS:
            return
        aggregate.note_message(f"lifecycle:{event_type}")
        return
    if message_type == "rfq_created":
        aggregate.note_message(
            "rfq_created:combo" if aggregate.note_rfq(payload)
            else "rfq_created:single")
        return
    if message_type in ("rfq_deleted", "subscribed", "error"):
        aggregate.note_message(message_type)


async def stream(duration_seconds=None, *, flush_seconds=FLUSH_SECONDS,
                 conn=None, channels=CHANNELS):
    """Listen until ``duration_seconds`` elapses, reconnecting on drop.

    ``None`` runs forever, which is the intended production mode.
    """
    if not _kalshi_available():
        raise RuntimeError("Kalshi is not configured; nothing to subscribe to")
    aggregate = DemandAggregate()
    deadline = None if duration_seconds is None else (
        time.monotonic() + duration_seconds)
    backoff = RECONNECT_MIN_SECONDS
    try:
        while deadline is None or time.monotonic() < deadline:
            try:
                await _consume(aggregate, deadline=deadline,
                               flush_seconds=flush_seconds, conn=conn,
                               channels=channels)
                backoff = RECONNECT_MIN_SECONDS
            except asyncio.CancelledError:
                raise
            except Exception as exc:                            # noqa: BLE001
                # Any drop is survivable and expected on a socket meant to stay
                # up for days. Flush first so the window's counts are not lost
                # to the reconnect.
                flush(aggregate, conn=conn)
                logger.warning("kalshi stream: %s: %s; reconnecting in %ss",
                               type(exc).__name__, exc, backoff)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, RECONNECT_MAX_SECONDS)
    finally:
        # Whatever the reason for stopping -- deadline, Ctrl-C, cancellation --
        # the partial window is worth more written than discarded.
        flush(aggregate, conn=conn)


def top_structures(limit=25, *, same_game_only=False, conn=None):
    """The most-requested combo structures seen so far.

    What the market keeps asking to be priced, which is where quoting would
    matter and which structures the dependence experiment should cover first.
    """
    owned = conn is None
    conn = conn or get_connection()
    try:
        return [dict(row) for row in conn.execute(
            f"""SELECT fingerprint, collection_ticker, num_legs, is_same_game,
                       rfq_count, total_contracts, first_seen, last_seen
                  FROM kalshi_rfq_demand
                 {'WHERE is_same_game = 1' if same_game_only else ''}
                 ORDER BY rfq_count DESC LIMIT ?""", (limit,))]
    finally:
        if owned:
            conn.close()
