"""Selector session (``BRAIN_MODE=selector``): the strategy running on the live stream.

Every UTC minute, once the stream has written it (``pumpstream_status.flushed_through``), the minute's rich candle per
mint (``pump_minutes``: open/high/low/close, SOL volume by side, buy/sell counts, distinct traders, quote reserve; the
same fields and filters ``train/mature.py`` builds from the archive) is fed to the live feature engine as two
synthetic trades, exactly as in training. Graduation time and creation/creator facts come from ``corpus_meta``, the
same source training uses. Rows that pass the eligibility gate (pool ≥ 20 SOL, 15-minute volume ≥ 5 SOL, no scale break —
a close 50× from the previous one or a reserve above 100,000 SOL — since the mint's state began) are scored
by the deployed selector (``brain_snapshots`` kind 'selector') with the feature vector assembled by name in the
model's column order; scores at or above its threshold open a position sized from the model's measured certainty
(``agent/sizing.py``: a fraction of the growth-optimal bet for the score's band, of the bankroll minus the gas reserve) in book
``paper_selector``, held ``horizon_min`` minutes, then sold at the last traded price (the training label's exit).
Every score, entry and exit is a ``decisions`` row; marks and wealth go to ``wealth_marks`` per minute with the same
conventions as the other books. Kill switch, pause and reserve rails apply to entries; no entries or exits happen
while the stream is stale; missed minutes are replayed through the features without trading.
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
import uuid
from collections import deque
from datetime import datetime, timezone

import numpy as np

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from . import sizing
from ..execution import ledger
from ..execution.broker_paper import PaperBroker
from ..market.exit_cost import exit_cost_fraction
from ..market.features import FEATURES, FIDX, TokenMeta, TokenState
from ..train.corpus_meta import FEATURE_COLS as META_COLS
from ..train.decisions import EXTRA_COLS, MIN_RESQ_SOL, MIN_VOL_15M_SOL, X_COLS
from ..train.mature import MASKED

log = logging.getLogger(__name__)
BOOK = "paper_selector"
WARMUP_MIN = 1440            # training warms each day on the previous day's candles
IDLE_EVICT_S = 86400         # training has no history for a mint idle since before the previous day
STREAM_STALE_S = 180         # no entries or exits when the stream's newest complete minute is older than this
META_TTL_S = 600
MODEL_CHECK_S = 600          # how often the session looks for a newer deployable model (train/pipeline.py)


class MintState:
    __slots__ = ("st", "meta", "hist", "decimals", "pool", "program_label", "graduated_at", "meta_row", "prev_close", "broken")

    def __init__(self, mint: str, decimals: int, pool: str | None, program_label: str | None, graduated_at: float | None, meta_row: dict | None):
        self.st = TokenState(mint); self.meta = TokenMeta(mint=mint, program_label=program_label or "Pump.fun Amm", graduated_at=graduated_at)
        self.hist: deque = deque(maxlen=200)       # (t_end, n_trades, n_traders) for the trailing hour
        self.decimals, self.pool, self.program_label, self.graduated_at, self.meta_row = decimals, pool, program_label, graduated_at, meta_row
        self.prev_close: float | None = None; self.broken = False


class NoModel(RuntimeError):
    """No selector has qualified to trade (current data and a profitable backtest after costs)."""


class SelectorSession:
    def __init__(self, live: bool = False, horizon_min: int | None = None):
        from ..train import selector as sel
        self.model = sel.load_latest()
        if self.model is None:
            raise NoModel("no model has qualified to trade yet (trained on the current data, with a backtest that made money after costs and beat random picks)")
        missing = [c for c in self.model.cols if c not in X_COLS]
        if missing:
            raise RuntimeError(f"selector expects features the live engine does not produce: {missing}")
        if list(self.model.cols) != list(X_COLS):
            log.warning("selector was trained on %d of the %d live columns; scoring by name", len(self.model.cols), len(X_COLS))
        self.horizon_s = (horizon_min or self.model.horizon_min) * 60
        self.live = live
        if live:
            log.warning("live execution is not wired to the selector session yet; trading the paper book only")
        self.broker = PaperBroker(BOOK)
        self.states: dict[str, MintState] = {}
        self.run_id = str(uuid.uuid4()); self.beat_no = 0; self.last_minute: float | None = None
        self.meta_cache: dict[str, dict] = {}; self.last_sweep = time.time()
        with transaction() as conn:
            sid = sel.latest_current(conn)["id"]                  # the snapshot load_latest returned
            self.snapshot_id = sid
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,'selector',%s,%s,'running')",
                         (self.run_id, json.dumps({"threshold": self.model.threshold, "horizon_min": self.model.horizon_min, "book": BOOK}, default=str), sid))
        record_event("info", "selector", "selector session started", {"run_id": self.run_id, "snapshot": sid, "threshold": self.model.threshold, "horizon_min": self.model.horizon_min})
        log.info("selector session: snapshot %d threshold %.4f horizon %d min", sid, self.model.threshold, self.model.horizon_min)

    def maybe_reload(self) -> bool:
        """Switch to a newer deployable model without a restart (the paper book and open positions carry on)."""
        import joblib
        from ..train import selector as sel
        with transaction() as conn:
            r = sel.latest_current(conn)
        if not r or r["id"] == getattr(self, "snapshot_id", None):
            return False
        m = joblib.load(r["path"])
        missing = [c for c in m.cols if c not in X_COLS]
        if missing:
            log.error("model #%d expects features the live engine does not produce: %s; keeping #%s", r["id"], missing, self.snapshot_id)
            return False
        old = self.snapshot_id; self.model = m; self.snapshot_id = r["id"]
        with transaction() as conn:
            conn.execute("UPDATE runs SET brain_snapshot_id = %s, config = %s WHERE run_id = %s",
                         (r["id"], json.dumps({"threshold": m.threshold, "horizon_min": m.horizon_min, "book": BOOK}, default=str), self.run_id))
        record_event("info", "selector", f"switched to model #{r['id']}", {"from": old, "to": r["id"], "threshold": m.threshold})
        log.info("switched from model #%s to #%d (threshold %.4f)", old, r["id"], m.threshold)
        return True

    # ---- lookups ----
    def _meta(self, conn, mint: str) -> dict | None:
        """``corpus_meta`` row (graduation time + creation/creator features), cached for 10 minutes: rows appear at
        graduation and the archive rebuild later replaces stream values, so the cache is refreshed."""
        hit = self.meta_cache.get(mint)
        if hit is None or time.time() - hit["at"] > META_TTL_S:
            r = conn.execute("SELECT graduated_at, " + ", ".join(META_COLS) + " FROM corpus_meta WHERE mint = %s", (mint,)).fetchone()
            hit = self.meta_cache[mint] = {"row": dict(r) if r else None, "at": time.time()}
        return hit["row"]

    def _state(self, conn, mint: str, pool: str | None, program_label: str | None) -> MintState:
        s = self.states.get(mint)
        if s is None:
            t = conn.execute("SELECT decimals FROM tokens WHERE mint = %s", (mint,)).fetchone()
            s = MintState(mint, int((t or {}).get("decimals") or 6), pool, program_label, None, None)
            self.states[mint] = s
        row = self._meta(conn, mint)
        s.meta_row = row
        g = row["graduated_at"].timestamp() if row and row.get("graduated_at") else None
        if g != s.graduated_at:                     # same source as training: corpus_meta.graduated_at
            s.graduated_at = g; s.meta.graduated_at = g
        return s

    def _last_resq(self, conn, mint: str) -> float | None:
        s = self.states.get(mint)
        if s is not None and s.st.last_res_quote_sol:
            return s.st.last_res_quote_sol
        r = conn.execute("SELECT resq_sol FROM pump_minutes WHERE mint = %s AND resq_sol IS NOT NULL ORDER BY ts DESC LIMIT 1", (mint,)).fetchone()
        return float(r["resq_sol"]) if r else None

    def _age_h(self, mint: str, now_s: float) -> float | None:
        s = self.states.get(mint)
        return (now_s - s.graduated_at) / 3600.0 if s is not None and s.graduated_at else None

    # ---- one minute ----
    def _aggregate(self, conn, m0: datetime, m1: datetime) -> dict[str, dict]:
        if config.SELECTOR_SOURCE == "stream":       # PumpAPI minutes: every SOL-quoted PumpSwap pump.fun token, same fields as the archive
            rows = conn.execute("SELECT mint, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol FROM pump_minutes WHERE ts = %s", (m0,)).fetchall()
            return {r["mint"]: {"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "buy": r["buy_sol"] or 0.0, "sell": r["sell_sol"] or 0.0,
                                "nb": r["n_buys"] or 0, "ns": r["n_sells"] or 0, "n_traders": int(r["n_traders"] or 0), "resq": r["resq_sol"],
                                "pool": r["pool_id"], "program_label": "Pump.fun Amm"} for r in rows if r["close"]}
        rows = conn.execute("""SELECT s.mint, s.pool, s.ts, s.side, s.amount_quote, s.price_sol, s.signer, s.res_quote, wp.program_label
                               FROM swap_tape s LEFT JOIN watch_pools wp ON wp.pool = s.pool
                               WHERE s.ts >= %s AND s.ts < %s AND s.side <> 0 AND s.price_sol > 0 ORDER BY s.ts, s.id""", (m0, m1)).fetchall()
        agg: dict[str, dict] = {}
        for r in rows:
            a = agg.setdefault(r["mint"], {"open": None, "high": -1.0, "low": math.inf, "close": None, "buy": 0.0, "sell": 0.0, "nb": 0, "ns": 0, "traders": set(),
                                           "resq": None, "pool": r["pool"], "program_label": r["program_label"]})
            p = float(r["price_sol"]); sol = float(r["amount_quote"] or 0) / config.LAMPORTS_PER_SOL
            a["open"] = p if a["open"] is None else a["open"]; a["high"] = max(a["high"], p); a["low"] = min(a["low"], p); a["close"] = p
            if int(r["side"]) == 1:
                a["buy"] += sol; a["nb"] += 1
            else:
                a["sell"] += sol; a["ns"] += 1
            if r["signer"]:
                a["traders"].add(r["signer"])
            if r["res_quote"]:
                a["resq"] = float(r["res_quote"]) / config.LAMPORTS_PER_SOL
        for a in agg.values():
            a["n_traders"] = len(a.pop("traders"))
        return agg

    def _features(self, conn, mint: str, a: dict, t_end: float) -> tuple[np.ndarray, dict]:
        s = self._state(conn, mint, a["pool"], a["program_label"])
        st, price, resq = s.st, float(a["close"]), a["resq"]
        if (s.prev_close and not (1 / 50 <= price / s.prev_close <= 50)) or (resq is not None and resq > 1e5):
            s.broken = True                          # train/decisions.py: a scale break makes the mint ineligible from that minute on
        s.prev_close = price
        if a["buy"] > 0:
            st.append(t_end - 2e-3, price, a["buy"], True, None, resq)
        if a["sell"] > 0 or a["buy"] <= 0:
            st.append(t_end - 1e-3, price, a["sell"], False, None, resq)
        s.hist.append((t_end, float(a["nb"] + a["ns"]), float(a["n_traders"])))
        f, _mask = st.features(t_end, s.meta)
        ht = np.array([h[0] for h in s.hist]); hn = np.array([h[1] for h in s.hist]); htr = np.array([h[2] for h in s.hist])
        for k, w in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)):
            f[FIDX[f"logn_{k}"]] = math.log1p(float(hn[ht > t_end - w].sum()))
        for name in MASKED:
            f[FIDX[name]] = 0.0
        if s.graduated_at is None:
            f[FIDX["log_age_h"]] = 0.0
        hod = ((t_end - 60.0) % 86400) / 3600.0                     # minute start, as in train/decisions.py
        meta = s.meta_row or {}
        extra = {"traders_15m": float(htr[ht > t_end - 900].sum()), "traders_1h": float(htr[ht > t_end - 3600].sum()), "n_trades_1m": float(a["nb"] + a["ns"]),
                 "hod_s": math.sin(2 * math.pi * hod / 24), "hod_c": math.cos(2 * math.pi * hod / 24), "age_known": 1.0 if s.graduated_at is not None else 0.0,
                 "meta_known": 1.0 if meta.get("ttg_min") is not None else 0.0}
        by_name = {**{n: float(f[i]) for i, n in enumerate(FEATURES)}, **extra,
                   **{c: float(meta.get(c)) if meta.get(c) is not None else 0.0 for c in META_COLS}}
        x = np.nan_to_num(np.asarray([by_name.get(c, 0.0) for c in self.model.cols], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        info = {"price": price, "resq": resq, "logvol_15m": f[FIDX["logvol_15m"]], "age_h": (t_end - s.graduated_at) / 3600 if s.graduated_at else None,
                "decimals": s.decimals, "pool": s.pool, "program_label": s.program_label, "broken": s.broken}
        return x, info

    def _stream_through(self) -> float | None:
        with transaction() as conn:
            r = conn.execute("SELECT value->>'flushed_through' AS ft FROM ui_settings WHERE key = 'pumpstream_status'").fetchone()
        return datetime.fromisoformat(r["ft"]).timestamp() if r and r["ft"] else None

    def ready_through(self, m1_epoch: float) -> float:
        """End of the newest minute (at most ``m1``) the stream has written; the tape source is always current. A minute is
        never read before it is written, so a lagging stream delays minutes instead of feeding them empty or partial."""
        if config.SELECTOR_SOURCE != "stream":
            return m1_epoch
        ft = self._stream_through()
        return min(m1_epoch, ft + 60.0) if ft is not None else float("-inf")

    def stream_fresh(self, m0_epoch: float) -> bool:
        if config.SELECTOR_SOURCE != "stream":
            return True
        ft = self._stream_through()
        return ft is not None and m0_epoch - ft <= STREAM_STALE_S

    def warm_up(self, m1_epoch: float, minutes: int = WARMUP_MIN) -> int:
        """Feed the last ``minutes`` closed minutes through the feature states without trading."""
        n = 0
        with transaction() as conn:
            for k in range(minutes, 0, -1):
                t1 = m1_epoch - 60 * k
                agg = self._aggregate(conn, datetime.fromtimestamp(t1 - 60, timezone.utc), datetime.fromtimestamp(t1, timezone.utc))
                for mint, a in agg.items():
                    self._features(conn, mint, a, t1); n += 1
        log.info("warm-up: %d token-minutes over the last %d minutes (%d tokens)", n, minutes, len(self.states))
        return n

    def _sweep(self, now_s: float, held: set[str]) -> None:
        if now_s - self.last_sweep < 3600:
            return
        self.last_sweep = now_s
        idle = [m for m, s in self.states.items() if m not in held and (s.st.last_ts is None or s.st.last_ts < now_s - IDLE_EVICT_S)]
        for m in idle:
            self.states.pop(m, None); self.meta_cache.pop(m, None)
        if idle:
            log.info("evicted %d mint states idle for more than %d h", len(idle), IDLE_EVICT_S // 3600)

    def run_minute(self, m1_epoch: float, trade: bool = True) -> dict:
        m1 = datetime.fromtimestamp(m1_epoch, timezone.utc); m0 = datetime.fromtimestamp(m1_epoch - 60, timezone.utc); now = datetime.now(timezone.utc)
        with transaction() as conn:
            agg = self._aggregate(conn, m0, m1)
            xs, infos, mints = [], [], []
            for mint, a in agg.items():
                x, info = self._features(conn, mint, a, m1_epoch)
                if not info["broken"] and info["resq"] is not None and info["resq"] >= MIN_RESQ_SOL and info["logvol_15m"] >= math.log1p(MIN_VOL_15M_SOL):
                    xs.append(x); infos.append(info); mints.append(mint)
            scores = self.model.score(np.stack(xs)) if xs else np.array([])
            summary = {"minute": m1.isoformat(), "mints_traded": len(agg), "eligible": len(mints), "picks": int((scores >= self.model.threshold).sum()) if len(scores) else 0,
                       "score_p99": float(np.percentile(scores, 99)) if len(scores) else None, "score_max": float(scores.max()) if len(scores) else None}
            if not trade:
                return summary
            if not self.stream_fresh(m1_epoch - 60):
                status = {**summary, "stage": "stream stale: holding (no entries or exits)", "threshold": self.model.threshold, "updated_at": now.isoformat()}
                conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (json.dumps(status, default=str),))
                return status
            self.beat_no += 1
            beat = conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active, notes) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                                (self.run_id, m1, self.beat_no, len(mints), json.dumps({"minute": m1.isoformat(), "mints_traded": len(agg)}))).fetchone()["id"]
            prices = {m: float(a["close"]) for m, a in agg.items()}; resqs = {m: a["resq"] for m, a in agg.items()}
            # exits: positions past the horizon, at this minute's price or the last traded price (the label's exit)
            n_exit = 0
            for p in ledger.open_positions(conn, BOOK):
                held = (m1 - p["opened_at"]).total_seconds()
                if held < self.horizon_s:
                    continue
                did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'selector_exit',%s,false,%s,%s) RETURNING id",
                                   (beat, self.run_id, m1, p["mint"], p["pool"], float(p["cost_sol"]), f"held {held/60:.0f} min", json.dumps({"book": BOOK}))).fetchone()["id"]
                self.broker.sell(conn, position=p, decision_id=did, price=prices.get(p["mint"]), res_quote_sol=resqs.get(p["mint"]) or self._last_resq(conn, p["mint"]),
                                 age_hours=self._age_h(p["mint"], m1_epoch), program_label=p.get("program_label"), forced_kind=None, ts=m1)
                n_exit += 1
            # entries: scores at/above threshold, one position per mint, rails
            opens_now = ledger.open_positions(conn, BOOK); open_mints = {p["mint"] for p in opens_now}
            cash = ledger.paper_cash(conn, BOOK); n_enter = 0; picks = []
            bankroll = cash + sum(float(p["cost_sol"]) for p in opens_now)         # wealth at cost: the sizing base (agent/sizing.py)
            table = getattr(self.model, "sizing", None) or []
            circuit = conn.execute("SELECT kill_switch, entries_paused FROM circuit_state WHERE id = 1").fetchone()
            blocked = "kill switch" if circuit and circuit["kill_switch"] else ("paused" if circuit and circuit["entries_paused"] else None)
            for m, i, sc in zip(mints, infos, scores):
                if sc < self.model.threshold:
                    continue
                picks.append((m, float(sc)))
                size, why = sizing.size_position(float(sc), self.model.threshold, table, bankroll, cash, i["resq"])
                if m in open_mints or blocked or size <= 0:
                    rail = blocked or ("held" if m in open_mints else "sizing")
                    conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail) VALUES (%s,%s,%s,%s,%s,'blocked',%s,%s,false,%s,%s,%s)",
                                 (beat, self.run_id, m1, m, i["pool"], float(sc), size, rail, why if rail == "sizing" else "selector pick not taken",
                                  json.dumps({"score": float(sc), "threshold": self.model.threshold, "sizing": why})))
                    continue
                did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, reason, book_targets, detail) VALUES (%s,%s,%s,%s,%s,'selector_enter',%s,%s,false,%s,%s,%s) RETURNING id",
                                   (beat, self.run_id, m1, m, i["pool"], float(sc), size, f"score {sc:.3f} ≥ {self.model.threshold:.3f}; {why}", [BOOK],
                                    json.dumps({"score": float(sc), "threshold": self.model.threshold, "size_sol": size, "sizing": why, "resq": i["resq"], "age_h": i["age_h"], "price": i["price"]}))).fetchone()["id"]
                fill = self.broker.buy(conn, decision_id=did, mint=m, pool=i["pool"], size_sol=size, price=i["price"], res_quote_sol=i["resq"],
                                       age_hours=i["age_h"], decimals=i["decimals"], program_label=i["program_label"], ts=m1)
                if fill.ok:
                    n_enter += 1; open_mints.add(m); cash -= size
                else:
                    conn.execute("UPDATE decisions SET rail = %s, reason = %s WHERE id = %s", ("paper_fill", fill.reason, did))
            # marks and wealth (conventions of agent/reward.py: positions net of exit cost, exposure at cost, drawdown ≥ 0)
            ledger.mark_positions(conn, BOOK, prices, {}, m1)
            opens = ledger.open_positions(conn, BOOK); gross_v = 0.0; exit_cost = 0.0; exposure = 0.0
            for p in opens:
                gross = int(p["qty"]) / 10 ** int(p.get("decimals") or 6) * float(p.get("last_mark_price") or p["entry_price"])
                rq = resqs.get(p["mint"]) or self._last_resq(conn, p["mint"])
                gross_v += gross; exposure += float(p["cost_sol"])
                exit_cost += gross * exit_cost_fraction(gross, rq, self._age_h(p["mint"], m1_epoch), p.get("program_label"))
            cash = ledger.paper_cash(conn, BOOK); positions_net = gross_v - exit_cost; wealth = cash + positions_net
            peak_row = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s", (BOOK,)).fetchone(); peak = max(float(peak_row["pk"] or 0.0), wealth)
            conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                         (beat, BOOK, m1, cash, positions_net, exit_cost, wealth, peak, (1.0 - wealth / peak) if peak > 0 else 0.0, exposure, len(opens)))
            self._sweep(m1_epoch, open_mints)
            status = {**summary, "entered": n_enter, "exited": n_exit, "open": len(opens), "cash": cash, "wealth": wealth, "threshold": self.model.threshold,
                      "top_scores": sorted(picks, key=lambda x: -x[1])[:5], "latency_s": (now - m1).total_seconds(), "updated_at": datetime.now(timezone.utc).isoformat()}
            status["picks"] = len(picks)
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (json.dumps(status, default=str),))
        return status

    def finish(self) -> None:
        with transaction() as conn:
            conn.execute("UPDATE runs SET ended_at = now(), status = 'finished' WHERE run_id = %s", (self.run_id,))
        record_event("info", "selector", "selector session stopped", {"run_id": self.run_id, "minutes": self.beat_no})


def _wait_for_model(stop_event: threading.Event | None, live: bool) -> "SelectorSession | None":
    """Start a session as soon as a model qualifies (the training pipeline produces one); until then say so and wait."""
    while not (stop_event is not None and stop_event.is_set()):
        try:
            return SelectorSession(live=live)
        except NoModel as e:
            try:
                with transaction() as conn:
                    conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                                 (json.dumps({"stage": "waiting for a model", "detail": str(e), "updated_at": datetime.now(timezone.utc).isoformat()}),))
            except Exception:
                log.debug("selector status write failed", exc_info=True)
            log.info("trading engine waiting: %s", e)
            (stop_event or threading.Event()).wait(MODEL_CHECK_S)
    return None


def main(stop_event: threading.Event | None = None, live: bool = False) -> None:
    s = _wait_for_model(stop_event, live)
    if s is None:
        return
    m_start = math.floor(time.time() / 60) * 60
    s.warm_up(m_start); s.last_minute = m_start - 60; last_reload = time.time()
    try:
        while not (stop_event is not None and stop_event.is_set()):
            now = time.time(); m1 = math.floor(now / 60) * 60
            if now - last_reload >= MODEL_CHECK_S:
                last_reload = now
                try:
                    s.maybe_reload()
                except Exception:
                    log.exception("model reload check failed")
            if m1 > s.last_minute and now - m1 >= 4.0 and (upto := s.ready_through(m1)) > s.last_minute:   # only minutes the stream has written
                for minute in range(int(s.last_minute) + 60, int(upto) + 1, 60):
                    try:
                        st = s.run_minute(float(minute), trade=(minute == int(m1)))   # missed minutes update features only
                        if minute == int(m1) and s.beat_no % 10 == 0 and "wealth" in st:
                            log.info("minute %s: traded %d eligible %d picks %d entered %d exited %d open %d wealth %.3f", st["minute"], st["mints_traded"], st["eligible"],
                                     st["picks"], st["entered"], st["exited"], st["open"], st["wealth"])
                    except Exception:
                        log.exception("selector minute %s failed", minute); record_event("error", "selector", f"minute failed: {minute}")
                s.last_minute = upto
            time.sleep(0.5)
    finally:
        s.finish()
