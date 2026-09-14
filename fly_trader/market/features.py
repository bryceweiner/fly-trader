"""Per-token rolling state from the swap tape and the beat-time feature vector.

Feature definitions follow VOC dexlp/env/obs_spec.py (multi-timeframe momentum / realized vol /
imbalance / toxicity, Hawkes burstiness, drawdown, overextension, exit cost) restricted to the
token-intrinsic subset; LP-position slots are dropped. Every feature is standardized downstream
(Welford running moments in the encoder), so scales only need to be consistent.

Data structures: append-only Python lists with prefix sums, so each window statistic is O(log n)
via bisect; unique-signer counts use sliding counters with eviction pointers. Lists are compacted
when they hold more than twice the 3 h window.
"""
from __future__ import annotations

import math
from bisect import bisect_left
from collections import Counter
from dataclasses import dataclass, field

from .. import config
from .exit_cost import exit_cost_fraction

WINDOWS = {"1m": 60.0, "5m": 300.0, "15m": 900.0, "1h": 3600.0, "3h": 10800.0}
HAWKES_BETA = 1.0 / 60.0  # per second; intensity decays with a 1-minute time constant
EWMA_1H_TAU_S = 3600.0  # 1-hour time constant of the price EWMA (continuous time, seconds)

FEATURES: list[str] = [
    "ret_1m", "ret_5m", "ret_15m", "ret_1h", "ret_3h",
    "rvol_5m", "rvol_15m", "rvol_1h", "rvol_3h",
    "imb_1m", "imb_5m", "imb_15m", "imb_1h",
    "logvol_1m", "logvol_5m", "logvol_15m", "logvol_1h",
    "logn_1m", "logn_5m", "logn_15m", "logn_1h",
    "logsigners_15m", "logsigners_1h",
    "hawkes", "log_since_last", "log_liquidity_sol", "log_age_h",
    "dd_1h", "dd_3h", "overext_1h", "vpin_15m", "mom_decel", "vol_divergence", "exit_cost_0p1",
    "organic_score", "log_holders", "log_liq_usd", "top_holders_pct", "dev_balance_pct",
    "is_sus", "is_verified", "net_buyers_1h", "holder_change_1h", "price_change_24h", "token2022",
]
D = len(FEATURES)
FIDX = {n: i for i, n in enumerate(FEATURES)}
STATS_FEATURES = FEATURES[FIDX["organic_score"]:]
STATS_BIT = 1 << 62  # beyond any feature index; feature_mask is a bigint


@dataclass
class TokenMeta:
    mint: str
    pool: str | None = None
    program_label: str | None = None
    graduated_at: float | None = None  # epoch seconds
    token2022: bool = False
    stats: dict | None = None  # latest token_stats row (dict) or None


@dataclass
class TokenState:
    mint: str
    ts: list[float] = field(default_factory=list)
    logp: list[float] = field(default_factory=list)
    cum_vol: list[float] = field(default_factory=lambda: [0.0])
    cum_buy: list[float] = field(default_factory=lambda: [0.0])
    cum_sq: list[float] = field(default_factory=lambda: [0.0])
    signers: list[int] = field(default_factory=list)
    hawkes: float = 0.0
    last_ts: float | None = None
    last_price: float | None = None
    last_res_quote_sol: float | None = None
    ewma_1h: float | None = None
    high_1h_cache: tuple[float, float] | None = None
    sig_ptr: dict[str, int] = field(default_factory=lambda: {"15m": 0, "1h": 0})
    sig_cnt: dict[str, Counter] = field(default_factory=lambda: {"15m": Counter(), "1h": Counter()})
    n_since_compact: int = 0

    # -- ingestion --
    def append(self, ts: float, price: float, sol_vol: float, is_buy: bool, signer: str | None,
               res_quote_sol: float | None) -> None:
        if price is None or price <= 0 or not math.isfinite(price):
            return
        lp = math.log(price)
        if self.last_ts is not None:
            dt = max(0.0, ts - self.last_ts)
            self.hawkes = self.hawkes * math.exp(-HAWKES_BETA * dt) + 1.0
            dlp = lp - self.logp[-1]
            self.cum_sq.append(self.cum_sq[-1] + dlp * dlp)
            w = 1.0 - math.exp(-dt / EWMA_1H_TAU_S)
            self.ewma_1h = (1 - w) * self.ewma_1h + w * price if self.ewma_1h is not None else price
        else:
            self.hawkes = 1.0
            self.cum_sq.append(0.0)
            self.ewma_1h = price
        self.ts.append(ts)
        self.logp.append(lp)
        self.cum_vol.append(self.cum_vol[-1] + sol_vol)
        self.cum_buy.append(self.cum_buy[-1] + (sol_vol if is_buy else 0.0))
        h = hash(signer) if signer else 0
        self.signers.append(h)
        for k in self.sig_cnt:
            self.sig_cnt[k][h] += 1
        self.last_ts = ts
        self.last_price = price
        if res_quote_sol is not None and res_quote_sol > 0:
            self.last_res_quote_sol = res_quote_sol
        self.n_since_compact += 1
        if self.n_since_compact > 20000 and len(self.ts) > 2 * self._count_since(ts - WINDOWS["3h"]):
            self._compact(ts - WINDOWS["3h"])

    def _count_since(self, t0: float) -> int:
        return len(self.ts) - bisect_left(self.ts, t0)

    def _compact(self, t0: float) -> None:
        i = bisect_left(self.ts, t0)
        if i <= 0:
            self.n_since_compact = 0
            return
        base_vol, base_buy, base_sq = self.cum_vol[i], self.cum_buy[i], self.cum_sq[i]
        for k in self.sig_ptr:                     # entries dropped before the window pointer reached them still count: retire them
            cnt = self.sig_cnt[k]
            for j in range(self.sig_ptr[k], i):
                h = self.signers[j]; cnt[h] -= 1
                if cnt[h] <= 0:
                    del cnt[h]
        self.ts = self.ts[i:]
        self.logp = self.logp[i:]
        self.signers = self.signers[i:]
        self.cum_vol = [x - base_vol for x in self.cum_vol[i:]]
        self.cum_buy = [x - base_buy for x in self.cum_buy[i:]]
        self.cum_sq = [x - base_sq for x in self.cum_sq[i:]]
        for k in self.sig_ptr:
            self.sig_ptr[k] = max(0, self.sig_ptr[k] - i)
        self.n_since_compact = 0

    # -- queries --
    def _advance_signers(self, key: str, now: float) -> None:
        t0 = now - WINDOWS[key]
        ptr = self.sig_ptr[key]
        cnt = self.sig_cnt[key]
        while ptr < len(self.ts) and self.ts[ptr] < t0:
            h = self.signers[ptr]
            cnt[h] -= 1
            if cnt[h] <= 0:
                del cnt[h]
            ptr += 1
        self.sig_ptr[key] = ptr

    def volume_since(self, t0: float) -> float:
        i = bisect_left(self.ts, t0)
        return self.cum_vol[-1] - self.cum_vol[i]

    def rvol(self, now: float, window_s: float = 900.0) -> float:
        """Realized volatility (root of summed squared log price changes) over the window."""
        i = bisect_left(self.ts, now - window_s)
        return math.sqrt(max(0.0, self.cum_sq[-1] - self.cum_sq[i])) if self.ts else 0.0

    def price_at(self, t: float) -> float | None:
        """Last price at or before t (None if no swap that early)."""
        i = bisect_left(self.ts, t + 1e-9)
        if i == 0:
            return None
        return math.exp(self.logp[i - 1])

    def features(self, now: float, meta: TokenMeta) -> tuple[list[float], int]:
        f = [0.0] * D
        mask = 0
        n = len(self.ts)
        if n == 0 or self.last_price is None:
            return f, mask
        lp_now = self.logp[-1]
        p_now = self.last_price
        idx = {k: bisect_left(self.ts, now - w) for k, w in WINDOWS.items()}
        avail = {k: idx[k] < n for k in WINDOWS}
        # returns: price at window start vs now (needs a price before the window start)
        for k in ("1m", "5m", "15m", "1h", "3h"):
            i = idx[k]
            if i > 0:
                f[FIDX[f"ret_{k}"]] = lp_now - self.logp[i - 1]
                mask |= 1 << FIDX[f"ret_{k}"]
            elif n > 1:
                f[FIDX[f"ret_{k}"]] = lp_now - self.logp[0]  # partial history; still informative
                mask |= 1 << FIDX[f"ret_{k}"]
        for k in ("5m", "15m", "1h", "3h"):
            i = idx[k]
            if avail[k]:
                f[FIDX[f"rvol_{k}"]] = math.sqrt(max(0.0, self.cum_sq[-1] - self.cum_sq[i]))
                mask |= 1 << FIDX[f"rvol_{k}"]
        for k in ("1m", "5m", "15m", "1h"):
            i = idx[k]
            vol = self.cum_vol[-1] - self.cum_vol[i]
            buy = self.cum_buy[-1] - self.cum_buy[i]
            cnt = n - i
            f[FIDX[f"imb_{k}"]] = (2 * buy - vol) / (vol + 1e-9) if vol > 0 else 0.0
            f[FIDX[f"logvol_{k}"]] = math.log1p(vol)
            f[FIDX[f"logn_{k}"]] = math.log1p(cnt)
            mask |= (1 << FIDX[f"imb_{k}"]) | (1 << FIDX[f"logvol_{k}"]) | (1 << FIDX[f"logn_{k}"])
        for k in ("15m", "1h"):
            self._advance_signers(k, now)
            f[FIDX[f"logsigners_{k}"]] = math.log1p(len(self.sig_cnt[k]))
            mask |= 1 << FIDX[f"logsigners_{k}"]
        dt = max(0.0, now - self.last_ts)
        f[FIDX["hawkes"]] = self.hawkes * math.exp(-HAWKES_BETA * dt)
        f[FIDX["log_since_last"]] = math.log1p(dt)
        mask |= (1 << FIDX["hawkes"]) | (1 << FIDX["log_since_last"])
        liq = self.last_res_quote_sol
        if liq is not None:
            f[FIDX["log_liquidity_sol"]] = math.log1p(2.0 * liq)
            mask |= 1 << FIDX["log_liquidity_sol"]
        age_h = None
        if meta.graduated_at is not None:
            age_h = max(0.0, (now - meta.graduated_at) / 3600.0)
            f[FIDX["log_age_h"]] = math.log1p(age_h)
            mask |= 1 << FIDX["log_age_h"]
        # drawdown from window highs; overextension vs EWMA
        for k in ("1h", "3h"):
            i = idx[k]
            if avail[k]:
                hi = max(self.logp[i:]) if n - i <= 5000 else max(self.logp[i:i + 5000] + self.logp[-1:])
                f[FIDX[f"dd_{k}"]] = lp_now - hi
                mask |= 1 << FIDX[f"dd_{k}"]
        if self.ewma_1h:
            f[FIDX["overext_1h"]] = p_now / self.ewma_1h - 1.0
            mask |= 1 << FIDX["overext_1h"]
        # VPIN-like toxicity: mean |imbalance| over five 3-minute buckets in the last 15 min
        i15 = idx["15m"]
        if avail["15m"]:
            tox, nb = 0.0, 0
            for b in range(5):
                t_a, t_b = now - 900 + 180 * b, now - 900 + 180 * (b + 1)
                ia, ib = bisect_left(self.ts, t_a), bisect_left(self.ts, t_b)
                v = self.cum_vol[ib] - self.cum_vol[ia]
                if v > 0:
                    bv = self.cum_buy[ib] - self.cum_buy[ia]
                    tox += abs(2 * bv - v) / v
                    nb += 1
            f[FIDX["vpin_15m"]] = tox / nb if nb else 0.0
            mask |= 1 << FIDX["vpin_15m"]
        f[FIDX["mom_decel"]] = f[FIDX["ret_1m"]] - f[FIDX["ret_5m"]] / 5.0
        f[FIDX["vol_divergence"]] = math.log1p((self.cum_vol[-1] - self.cum_vol[idx["5m"]]) / 5.0) - \
            math.log1p((self.cum_vol[-1] - self.cum_vol[idx["1h"]]) / 60.0)
        mask |= (1 << FIDX["mom_decel"]) | (1 << FIDX["vol_divergence"])
        f[FIDX["exit_cost_0p1"]] = exit_cost_fraction(config.MAX_POSITION_SOL, liq, age_h, meta.program_label)
        mask |= 1 << FIDX["exit_cost_0p1"]
        # Jupiter token stats
        st = meta.stats
        if st:
            f[FIDX["organic_score"]] = (st.get("organic_score") or 0.0) / 100.0
            f[FIDX["log_holders"]] = math.log1p(st.get("holder_count") or 0)
            f[FIDX["log_liq_usd"]] = math.log1p(st.get("liquidity_usd") or 0.0)
            f[FIDX["top_holders_pct"]] = float(st.get("top_holders_pct") or 0.0)
            f[FIDX["dev_balance_pct"]] = float(st.get("dev_balance_pct") or 0.0)
            f[FIDX["is_sus"]] = 1.0 if st.get("is_sus") else 0.0
            f[FIDX["is_verified"]] = 1.0 if st.get("is_verified") else 0.0
            s1 = (st.get("stats") or {}).get("stats1h") or {}
            s24 = (st.get("stats") or {}).get("stats24h") or {}
            f[FIDX["net_buyers_1h"]] = float(s1.get("numNetBuyers") or 0.0)
            f[FIDX["holder_change_1h"]] = float(s1.get("holderChange") or 0.0)
            f[FIDX["price_change_24h"]] = float(s24.get("priceChange") or 0.0)
            mask |= STATS_BIT
        f[FIDX["token2022"]] = 1.0 if meta.token2022 else 0.0
        return f, mask


class FeatureBank:
    """All watched tokens' rolling states; the runner and the replay feed both drive it."""

    def __init__(self):
        self.states: dict[str, TokenState] = {}
        self.pool_to_mint: dict[str, str] = {}
        self.sol_usd: float | None = None

    def state(self, mint: str) -> TokenState:
        st = self.states.get(mint)
        if st is None:
            st = TokenState(mint)
            self.states[mint] = st
        return st

    def ingest_tape_row(self, row: dict) -> None:
        """row: a swap_tape dict (ts datetime or epoch, side, amount_quote, price_sol/price_quote, signer, res_quote, mint, pool)."""
        side = row.get("side") or 0
        if side == 0:
            return
        mint = row.get("mint") or self.pool_to_mint.get(row.get("pool"))
        if not mint:
            return
        ts = row["ts"]
        ts = ts.timestamp() if hasattr(ts, "timestamp") else float(ts)
        price = row.get("price_sol")
        aq = row.get("amount_quote") or 0
        rq = row.get("res_quote")
        qdec = row.get("quote_decimals")
        if price is None:
            pq = row.get("price_quote")
            if pq is not None and self.sol_usd:
                price = pq / self.sol_usd
        if price is None:
            return
        if qdec is None or qdec == 9:
            sol_vol = float(aq) / config.LAMPORTS_PER_SOL
            res_q = float(rq) / config.LAMPORTS_PER_SOL if rq is not None else None
        else:
            usd = float(aq) / (10 ** qdec)
            sol_vol = usd / self.sol_usd if self.sol_usd else 0.0
            res_q = (float(rq) / (10 ** qdec)) / self.sol_usd if (rq is not None and self.sol_usd) else None
        self.state(mint).append(ts, float(price), sol_vol, side > 0, row.get("signer"), res_q)

    def activity_sol(self, mint: str, now: float, window_s: float = 10800.0) -> float:
        st = self.states.get(mint)
        return st.volume_since(now - window_s) if st else 0.0

    def last_price(self, mint: str) -> float | None:
        st = self.states.get(mint)
        return st.last_price if st else None

    def last_swap_ts(self, mint: str) -> float | None:
        st = self.states.get(mint)
        return st.last_ts if st else None

    def rvol(self, mint: str, now: float, window_s: float = 900.0) -> float:
        st = self.states.get(mint)
        return st.rvol(now, window_s) if st else 0.0

    def res_quote_sol(self, mint: str) -> float | None:
        st = self.states.get(mint)
        return st.last_res_quote_sol if st else None
