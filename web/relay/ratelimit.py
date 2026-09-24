"""Token buckets in SQLite, shared by every uWSGI process. The caller holds a write transaction."""
from __future__ import annotations

import ipaddress

HOUR = 3600
# name -> (capacity, refill period in seconds); SPEC §3.
LIMITS = {
    "challenge": (30, HOUR),
    "claim_ip": (10, HOUR),
    "claim_evm": (5, HOUR),
    "read": (600, HOUR),
}


def take(conn, name: str, who: str, now: float) -> float:
    """Spend one token. Returns 0 when allowed, else the seconds until one is available."""
    capacity, period = LIMITS[name]
    key = "%s:%s" % (name, who)
    rate = capacity / float(period)
    row = conn.execute("SELECT tokens, updated FROM buckets WHERE key = ?", (key,)).fetchone()
    tokens = float(capacity) if row is None else min(capacity, row[0] + max(0.0, now - row[1]) * rate)
    if tokens < 1.0:
        return (1.0 - tokens) / rate
    conn.execute("INSERT OR REPLACE INTO buckets(key, tokens, updated) VALUES (?, ?, ?)", (key, tokens - 1.0, now))
    return 0.0


def purge(conn, now: float) -> None:
    """A bucket idle for a whole period is full again, which is what a missing row means."""
    longest = max(p for _, p in LIMITS.values())
    conn.execute("DELETE FROM buckets WHERE updated < ?", (now - longest,))


def ip_key(ip: str) -> str:
    """IPv6 clients are bucketed by /64: one subscriber usually owns a whole /64."""
    ip = (ip or "").strip()
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return ip[:64] or "unknown"
    if addr.version == 4:
        return str(addr)
    if addr.ipv4_mapped is not None:
        return str(addr.ipv4_mapped)
    return str(ipaddress.ip_network("%s/64" % addr, strict=False))
