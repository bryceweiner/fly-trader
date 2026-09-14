"""Slot assignment: held tokens pinned, then a rotation (not a ranking) over the activity floor.

Operator decision: candidates need ≥ ACTIVITY_FLOOR_SOL_3H of swap volume in 3 h (VOC measured that
an activity *ranking* kept 9 % of oracle winners while a small floor kept > 75 %). Never-visited
tokens (fresh graduations) fill first, then the least-recently visited. A slot keeps its token for
DWELL_BEATS and while m̂ ≥ θ_sell so a building signal is not evicted.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .. import config


@dataclass
class Slot:
    mint: str | None = None
    dwell: int = 0


@dataclass
class SlotManager:
    n: int
    slots: list[Slot] = field(default_factory=list)
    visits: dict[str, float] = field(default_factory=dict)  # mint -> last visit ts
    visit_counts: dict[str, int] = field(default_factory=dict)

    def __post_init__(self):
        if not self.slots:
            self.slots = [Slot() for _ in range(self.n)]

    def load_visits(self, rows: list[dict]) -> None:
        for r in rows:
            if r.get("last_visit_ts") is not None:
                self.visits[r["mint"]] = r["last_visit_ts"].timestamp()
                self.visit_counts[r["mint"]] = int(r.get("visits") or 0)

    def assign(self, now: float, held: set[str], candidates: dict[str, float], m_hat_by_mint: dict[str, float]) -> list[str | None]:
        """candidates: mint -> 3h activity in SOL; m_hat_by_mint: mint -> last valence z-score."""
        current = {s.mint: i for i, s in enumerate(self.slots) if s.mint}
        eligible = {m for m, a in candidates.items() if a >= config.ACTIVITY_FLOOR_SOL_3H} | set(held)
        # 1. evict: not held, dwell over, and (m̂ < θ_sell or no longer eligible)
        for s in self.slots:
            if s.mint is None:
                continue
            s.dwell += 1
            if s.mint in held:
                continue
            m = m_hat_by_mint.get(s.mint, 0.0)
            if s.mint not in eligible or (s.dwell >= config.DWELL_BEATS and m < config.Z_SELL) or \
               (s.dwell >= 3 * config.DWELL_BEATS):
                s.mint, s.dwell = None, 0
        # 2. pin held tokens not yet in a slot
        placed = {s.mint for s in self.slots if s.mint}
        for m in held:
            if m not in placed:
                free = next((s for s in self.slots if s.mint is None), None)
                if free is None:  # steal the oldest non-held slot
                    free = max((s for s in self.slots if s.mint not in held), key=lambda s: s.dwell, default=None)
                if free is not None:
                    free.mint, free.dwell = m, 0
                    placed.add(m)
        # 3. fill free slots: never-visited first, then least recently visited
        pool = [m for m in eligible if m not in placed]
        pool.sort(key=lambda m: (self.visits.get(m, -1.0), m))
        for s in self.slots:
            if s.mint is None and pool:
                s.mint, s.dwell = pool.pop(0), 0
        for s in self.slots:
            if s.mint:
                self.visits[s.mint] = now
                self.visit_counts[s.mint] = self.visit_counts.get(s.mint, 0) + 1
        return [s.mint for s in self.slots]

    def dwell_of(self) -> list[int]:
        return [s.dwell for s in self.slots]

    def persist(self, conn, now_dt) -> None:
        rows = [(m, now_dt, self.visit_counts.get(m, 0)) for m in {s.mint for s in self.slots if s.mint}]
        if rows:
            conn.cursor().executemany(
                "INSERT INTO slot_visits (mint, last_visit_ts, visits) VALUES (%s,%s,%s) "
                "ON CONFLICT (mint) DO UPDATE SET last_visit_ts = EXCLUDED.last_visit_ts, visits = EXCLUDED.visits",
                rows,
            )
