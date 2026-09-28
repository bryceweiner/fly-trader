"""Block → minute without one lookup per block: every block's minute is exact, whatever the block rate does."""
import random

from fly_trader.rh import blocktime, scan


class Chain:
    def __init__(self, n, seed=0):
        rnd = random.Random(seed); t = 1_790_000_000; self.ts = []
        for i in range(n):
            t += rnd.choice([0, 0, 0, 1, 1, 2, 5, 30]) if i % 5000 else 400        # bursts, idle gaps, a long pause
            self.ts.append(t)
        self.calls = 0

    def batch(self, calls):
        self.calls += 1
        return [{"timestamp": hex(self.ts[int(p[0], 16)])} for _, p in calls]


def test_every_minute_is_exact(monkeypatch):
    monkeypatch.setattr(scan, "PAUSE_S", 0.0)
    c = Chain(30_000); lo, hi = 1000, 29_000
    wanted = list(range(lo, hi + 1, 7)) + [lo, hi]
    got = blocktime.times(c, lo, hi, wanted)
    assert all(int(got[b] // 60) == c.ts[b] // 60 for b in wanted)
    assert all(abs(got[b] - c.ts[b]) < 60 for b in wanted)
    assert c.calls < 250                                                       # vs ~4000 per-block lookups
