"""The trading wallet's NAV marks and its flow-adjusted performance index (plan "Core accounting": kill switch).

    I_t = I_{t-1} * (W_t - F_t) / W_{t-1}

W is the wallet's marked wealth (owed SOL included, so a settlement moves nothing) and F every external flow that
landed since the previous mark (deposits and gifts in, withdrawals and claims out). A deposit or a payout therefore
neither lifts the peak nor looks like a loss. Flows are attached by their on-chain time, and the peak only uses marks
at least ``PEAK_MIN_AGE_S`` old, so a flow the scanner books a minute late is in place before its mark can count.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from .. import config

PEAK_MIN_AGE_S = 600
WINDOW_DAYS = 30
LAMPORTS = config.LAMPORTS_PER_SOL


def reserved_lamports(conn) -> int:
    """Owed and not yet paid: everything allocated minus claims paid (in-flight claims are still in the wallet)."""
    r = conn.execute("SELECT COALESCE((SELECT sum(allocated) FROM vault_settlements WHERE status = 'allocated'), 0) - "
                     "COALESCE((SELECT sum(lamports) FROM vault_flows WHERE kind = 'claim'), 0) AS r").fetchone()
    return max(0, int(r["r"] or 0))


def mark(conn, ts: datetime, native: int, positions_value: int, exit_cost: int, token_acct: int = 0,
         slot: int | None = None, consistent: bool = True) -> None:
    wealth = int(native) + int(token_acct) + int(positions_value) - int(exit_cost)
    conn.execute(
        "INSERT INTO vault_nav (ts, slot, native, token_acct, positions_value, exit_cost, wealth, reserved, consistent) "
        "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (ts) DO UPDATE SET native = EXCLUDED.native, token_acct = EXCLUDED.token_acct, "
        "positions_value = EXCLUDED.positions_value, exit_cost = EXCLUDED.exit_cost, wealth = EXCLUDED.wealth, reserved = EXCLUDED.reserved, "
        "consistent = EXCLUDED.consistent",
        (ts, slot, int(native), int(token_acct), int(positions_value), int(exit_cost), wealth, reserved_lamports(conn), bool(consistent)))


def index_series(marks: list[tuple[float, int]], flows: list[tuple[float, int]]) -> list[tuple[float, float]]:
    """Pure: marks (ts, wealth) in time order, flows (ts, signed lamports). Returns (ts, index) starting at 1.0."""
    out: list[tuple[float, float]] = []
    fi, prev_w, idx = 0, None, 1.0
    flows = sorted(flows)
    for ts, w in marks:
        f = 0
        while fi < len(flows) and flows[fi][0] <= ts:
            if prev_w is not None:
                f += flows[fi][1]
            fi += 1
        if prev_w is not None and prev_w > 0:
            idx *= max(0.0, (w - f) / prev_w)
        prev_w = w
        out.append((ts, idx))
    return out


def drawdown(series: list[tuple[float, float]], now: float) -> dict:
    if not series:
        return {"value": 1.0, "peak": 1.0, "drawdown": 0.0}
    eligible = [i for t, i in series if t <= now - PEAK_MIN_AGE_S] or [series[0][1]]
    cur, peak = series[-1][1], max(max(eligible), series[0][1])
    return {"value": cur, "peak": peak, "drawdown": (1.0 - cur / peak) if peak > 0 else 0.0}


def load(conn, since: datetime | None = None) -> list[tuple[float, float]]:
    since = max(since or datetime.fromtimestamp(0, timezone.utc), datetime.now(timezone.utc) - timedelta(days=WINDOW_DAYS))
    marks = [(r["ts"].timestamp(), int(r["wealth"])) for r in conn.execute(
        "SELECT ts, wealth FROM vault_nav WHERE ts >= %s AND consistent ORDER BY ts", (since,)).fetchall()]
    flows = [(r["t"], int(r["f"])) for r in conn.execute(
        "SELECT extract(epoch FROM block_time)::float AS t, CASE WHEN direction = 'in' THEN lamports ELSE -(lamports + fee_lamports) END AS f "
        "FROM vault_flows WHERE block_time >= %s AND kind IN ('deposit', 'profit', 'withdrawal', 'claim')", (since,)).fetchall()]
    return index_series(marks, flows)


def status(conn, since: datetime | None = None, now: float | None = None) -> dict:
    now = datetime.now(timezone.utc).timestamp() if now is None else now
    return drawdown(load(conn, since), now)
