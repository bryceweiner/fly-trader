"""Selector session (``BRAIN_MODE=selector``): the strategy running on the live tape.

Every UTC minute, two seconds after it closes, the minute's swaps for every watched pool are aggregated per mint
into the same rich candle ``train/mature.py`` builds from the archive (open/high/low/close, SOL volume by side,
buy/sell counts, distinct traders, quote reserve) and fed to the live feature engine as two synthetic trades,
exactly as in training. Rows that pass the eligibility gate (pool ≥ 20 SOL, 15-minute volume ≥ 5 SOL) are scored
by the deployed selector (``brain_snapshots`` kind 'selector'); scores at or above its threshold open a
``MAX_POSITION_SOL`` position in book ``paper_selector`` (and the live book once live execution is wired to this
session), held ``horizon_min`` minutes then sold at the pool price. Every score, entry and exit is a ``decisions``
row; marks and wealth go to ``wealth_marks`` per minute. Kill switch, pause and reserve rails apply to entries.
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
from ..execution import ledger
from ..execution.broker_paper import PaperBroker
from ..market.exit_cost import exit_cost_fraction
from ..market.features import D, FEATURES, FIDX, TokenMeta, TokenState
from ..train.corpus_meta import FEATURE_COLS as META_COLS
from ..train.decisions import EXTRA_COLS, MIN_RESQ_SOL, MIN_VOL_15M_SOL, X_COLS
from ..train.mature import MASKED

log = logging.getLogger(__name__)
BOOK = "paper_selector"


class MintState:
    __slots__ = ("st", "meta", "hist", "decimals", "pool", "program_label", "graduated_at", "meta_row")

    def __init__(self, mint: str, decimals: int, pool: str | None, program_label: str | None, graduated_at: float | None, meta_row: dict | None):
        self.st = TokenState(mint); self.meta = TokenMeta(mint=mint, program_label=program_label or "Pump.fun Amm", graduated_at=graduated_at)
        self.hist: deque = deque(maxlen=200)       # (t_end, n_trades, n_traders) for the trailing hour
        self.decimals, self.pool, self.program_label, self.graduated_at, self.meta_row = decimals, pool, program_label, graduated_at, meta_row


class SelectorSession:
    def __init__(self, live: bool = False, horizon_min: int | None = None):
        from ..train import selector as sel
        self.model = sel.load_latest()
        if self.model is None:
            raise RuntimeError("no selector snapshot; run training (regimen 'selector') first")
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
        self.meta_cache: dict[str, dict | None] = {}; self.meta_refreshed = 0.0
        with transaction() as conn:
            sid = conn.execute("SELECT id FROM brain_snapshots WHERE kind = 'selector' ORDER BY id DESC LIMIT 1").fetchone()["id"]
            conn.execute("INSERT INTO runs (run_id, kind, config, brain_snapshot_id, status) VALUES (%s,'selector',%s,%s,'running')",
                         (self.run_id, json.dumps({"threshold": self.model.threshold, "horizon_min": self.model.horizon_min, "book": BOOK}, default=str), sid))
        record_event("info", "selector", "selector session started", {"run_id": self.run_id, "snapshot": sid, "threshold": self.model.threshold, "horizon_min": self.model.horizon_min})
        log.info("selector session: snapshot %d threshold %.4f horizon %d min", sid, self.model.threshold, self.model.horizon_min)

    # ---- lookups ----
    def _state(self, conn, mint: str, pool: str | None, program_label: str | None) -> MintState:
        s = self.states.get(mint)
        if s is None:
            t = conn.execute("SELECT decimals, graduated_at FROM tokens WHERE mint = %s", (mint,)).fetchone()
            g = t["graduated_at"].timestamp() if t and t["graduated_at"] else None
            if g is None:
                ct = conn.execute("SELECT graduated_at FROM corpus_tokens WHERE mint = %s", (mint,)).fetchone()
                g = ct["graduated_at"].timestamp() if ct and ct["graduated_at"] else None
            s = MintState(mint, int((t or {}).get("decimals") or 6), pool, program_label, g, self._meta(conn, mint))
            self.states[mint] = s
        return s

    def _meta(self, conn, mint: str) -> dict | None:
        """corpus_meta row for the mint; missing rows are retried every 10 minutes (the stream writes them at graduation)."""
        hit = self.meta_cache.get(mint)
        if hit is None or (hit["row"] is None and time.time() - hit["at"] > 600):
            r = conn.execute("SELECT " + ", ".join(META_COLS) + " FROM corpus_meta WHERE mint = %s", (mint,)).fetchone()
            hit = self.meta_cache[mint] = {"row": dict(r) if r else None, "at": time.time()}
        return hit["row"]

    # ---- one minute ----
    def _aggregate(self, conn, m0: datetime, m1: datetime) -> dict[str, dict]:
        if config.SELECTOR_SOURCE == "stream":       # PumpAPI minutes: every SOL-quoted PumpSwap pump.fun token, same fields as the archive
            rows = conn.execute("SELECT mint, pool_id, open, high, low, close, buy_sol, sell_sol, n_buys, n_sells, n_traders, resq_sol FROM pump_minutes WHERE ts = %s", (m0,)).fetchall()
            return {r["mint"]: {"open": r["open"], "high": r["high"], "low": r["low"], "close": r["close"], "buy": r["buy_sol"] or 0.0, "sell": r["sell_sol"] or 0.0,
                                "nb": r["n_buys"] or 0, "ns": r["n_sells"] or 0, "traders": set(range(int(r["n_traders"] or 0))), "resq": r["resq_sol"],
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
        return agg

    def _features(self, conn, mint: str, a: dict, t_end: float) -> tuple[np.ndarray, dict]:
        s = self._state(conn, mint, a["pool"], a["program_label"])
        if s.meta_row is None:
            s.meta_row = self._meta(conn, mint)
        st, price, resq = s.st, float(a["close"]), a["resq"]
        if a["buy"] > 0:
            st.append(t_end - 2e-3, price, a["buy"], True, None, resq)
        if a["sell"] > 0 or a["buy"] <= 0:
            st.append(t_end - 1e-3, price, a["sell"], False, None, resq)
        s.hist.append((t_end, float(a["nb"] + a["ns"]), float(len(a["traders"]))))
        f, mask = st.features(t_end, s.meta)
        ht = np.array([h[0] for h in s.hist]); hn = np.array([h[1] for h in s.hist]); htr = np.array([h[2] for h in s.hist])
        for k, w in (("1m", 60), ("5m", 300), ("15m", 900), ("1h", 3600)):
            f[FIDX[f"logn_{k}"]] = math.log1p(float(hn[ht > t_end - w].sum()))
        for name in MASKED:
            f[FIDX[name]] = 0.0
        if s.graduated_at is None:
            f[FIDX["log_age_h"]] = 0.0
        hod = (t_end % 86400) / 3600.0
        extra = {"traders_15m": float(htr[ht > t_end - 900].sum()), "traders_1h": float(htr[ht > t_end - 3600].sum()), "n_trades_1m": float(a["nb"] + a["ns"]),
                 "hod_s": math.sin(2 * math.pi * hod / 24), "hod_c": math.cos(2 * math.pi * hod / 24), "age_known": 1.0 if s.graduated_at is not None else 0.0,
                 "meta_known": 1.0 if (s.meta_row and s.meta_row.get("ttg_min") is not None) else 0.0}
        meta = s.meta_row or {}
        by_name = {**{n: float(f[i]) for i, n in enumerate(FEATURES)}, **extra, **{c: float(meta.get(c) if meta.get(c) is not None else 0.0) for c in META_COLS}}
        x = np.nan_to_num(np.asarray([by_name.get(c, 0.0) for c in self.model.cols], dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        info = {"price": price, "resq": resq, "logvol_15m": f[FIDX["logvol_15m"]], "age_h": (t_end - s.graduated_at) / 3600 if s.graduated_at else None,
                "decimals": s.decimals, "pool": s.pool, "program_label": s.program_label}
        return x, info

    def minute_ready(self, m0_epoch: float, max_wait_s: float = 25.0) -> bool:
        """True once the stream has flushed minute ``m0`` (or the source is the tape); gives up after ``max_wait_s`` past the close."""
        if config.SELECTOR_SOURCE != "stream":
            return True
        with transaction() as conn:
            r = conn.execute("SELECT value->>'flushed_through' AS ft FROM ui_settings WHERE key = 'pumpstream_status'").fetchone()
        ft = r["ft"] if r else None
        if ft and datetime.fromisoformat(ft).timestamp() >= m0_epoch:
            return True
        return time.time() - (m0_epoch + 60) >= max_wait_s

    def warm_up(self, m1_epoch: float, minutes: int = 180) -> int:
        """Feed the last ``minutes`` closed minutes through the feature states without trading (windows up to 3 h)."""
        n = 0
        with transaction() as conn:
            for k in range(minutes, 0, -1):
                t1 = m1_epoch - 60 * k
                agg = self._aggregate(conn, datetime.fromtimestamp(t1 - 60, timezone.utc), datetime.fromtimestamp(t1, timezone.utc))
                for mint, a in agg.items():
                    self._features(conn, mint, a, t1); n += 1
        log.info("warm-up: %d token-minutes over the last %d minutes (%d tokens)", n, minutes, len(self.states))
        return n

    def run_minute(self, m1_epoch: float, trade: bool = True) -> dict:
        m1 = datetime.fromtimestamp(m1_epoch, timezone.utc); m0 = datetime.fromtimestamp(m1_epoch - 60, timezone.utc); now = datetime.now(timezone.utc)
        with transaction() as conn:
            agg = self._aggregate(conn, m0, m1)
            xs, infos, mints = [], [], []
            for mint, a in agg.items():
                x, info = self._features(conn, mint, a, m1_epoch)
                if info["resq"] is not None and info["resq"] >= MIN_RESQ_SOL and info["logvol_15m"] >= math.log1p(MIN_VOL_15M_SOL):
                    xs.append(x); infos.append(info); mints.append(mint)
            scores = self.model.score(np.stack(xs)) if xs else np.array([])
            if not trade:
                return {"minute": m1.isoformat(), "mints_traded": len(agg), "eligible": len(mints), "picks": int((scores >= self.model.threshold).sum()) if len(scores) else 0,
                        "score_p99": float(np.percentile(scores, 99)) if len(scores) else None, "score_max": float(scores.max()) if len(scores) else None}
            self.beat_no += 1
            beat = conn.execute("INSERT INTO beats (run_id, ts, beat_no, n_slots_active, notes) VALUES (%s,%s,%s,%s,%s) RETURNING id",
                                (self.run_id, m1, self.beat_no, len(mints), json.dumps({"minute": m1.isoformat(), "mints_traded": len(agg)}))).fetchone()["id"]
            # exits: positions past the horizon
            prices = {m: i["price"] for m, i in zip(mints, infos)}; resqs = {m: i["resq"] for m, i in zip(mints, infos)}
            for m, a in agg.items():
                prices.setdefault(m, float(a["close"])); resqs.setdefault(m, a["resq"])
            n_exit = 0
            for p in ledger.open_positions(conn, BOOK):
                held = (m1 - p["opened_at"]).total_seconds()
                if held >= self.horizon_s:
                    did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, size_sol, forced, reason, detail) VALUES (%s,%s,%s,%s,%s,'selector_exit',%s,false,%s,%s) RETURNING id",
                                       (beat, self.run_id, m1, p["mint"], p["pool"], float(p["cost_sol"]), f"held {held/60:.0f} min", json.dumps({"book": BOOK}))).fetchone()["id"]
                    age_h = (m1_epoch - p["graduated_at"].timestamp()) / 3600 if p.get("graduated_at") else None
                    st_ = self.states.get(p["mint"]); last_rq = st_.st.last_res_quote_sol if st_ else None
                    fill = self.broker.sell(conn, position=p, decision_id=did, price=prices.get(p["mint"]), res_quote_sol=resqs.get(p["mint"]) or last_rq,
                                            age_hours=age_h, program_label=p.get("program_label"), forced_kind=None, ts=m1)
                    n_exit += 1
            # entries: scores at/above threshold, one position per mint, rails
            open_mints = {p["mint"] for p in ledger.open_positions(conn, BOOK)}
            cash = ledger.paper_cash(conn, BOOK); n_enter = 0; picks = []
            circuit = conn.execute("SELECT kill_switch, entries_paused FROM circuit_state WHERE id = 1").fetchone()
            blocked = "kill switch" if circuit and circuit["kill_switch"] else ("paused" if circuit and circuit["entries_paused"] else None)
            for m, i, sc in zip(mints, infos, scores):
                if sc < self.model.threshold:
                    continue
                picks.append((m, float(sc)))
                if m in open_mints or blocked or cash < config.MAX_POSITION_SOL + config.GAS_RESERVE_SOL:
                    conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, rail, reason, detail) VALUES (%s,%s,%s,%s,%s,'blocked',%s,%s,false,%s,%s,%s)",
                                 (beat, self.run_id, m1, m, i["pool"], float(sc), config.MAX_POSITION_SOL, blocked or ("held" if m in open_mints else "cash"), "selector pick not taken", json.dumps({"score": float(sc), "threshold": self.model.threshold})))
                    continue
                did = conn.execute("INSERT INTO decisions (beat_id, run_id, ts, mint, pool, kind, m_hat, size_sol, forced, reason, book_targets, detail) VALUES (%s,%s,%s,%s,%s,'selector_enter',%s,%s,false,%s,%s,%s) RETURNING id",
                                   (beat, self.run_id, m1, m, i["pool"], float(sc), config.MAX_POSITION_SOL, f"score {sc:.3f} ≥ {self.model.threshold:.3f}", [BOOK],
                                    json.dumps({"score": float(sc), "threshold": self.model.threshold, "resq": i["resq"], "age_h": i["age_h"], "price": i["price"]}))).fetchone()["id"]
                fill = self.broker.buy(conn, decision_id=did, mint=m, pool=i["pool"], size_sol=config.MAX_POSITION_SOL, price=i["price"], res_quote_sol=i["resq"],
                                       age_hours=i["age_h"], decimals=i["decimals"], program_label=i["program_label"], ts=m1)
                if fill.ok:
                    n_enter += 1; open_mints.add(m); cash -= config.MAX_POSITION_SOL
                else:
                    conn.execute("UPDATE decisions SET rail = %s, reason = %s WHERE id = %s", ("paper_fill", fill.reason, did))
            # marks and wealth
            ledger.mark_positions(conn, BOOK, prices, {}, m1)
            opens = ledger.open_positions(conn, BOOK); value = 0.0; exit_cost = 0.0
            for p in opens:
                gross = int(p["qty"]) / 10 ** int(p.get("decimals") or 6) * float(p.get("last_mark_price") or p["entry_price"])
                st_ = self.states.get(p["mint"]); rq = resqs.get(p["mint"]) or (st_.st.last_res_quote_sol if st_ else None)
                value += gross; exit_cost += gross * exit_cost_fraction(gross, rq, (m1_epoch - p["graduated_at"].timestamp()) / 3600 if p.get("graduated_at") else None, p.get("program_label"))
            cash = ledger.paper_cash(conn, BOOK); wealth = cash + value - exit_cost                      # marked net of the modelled exit cost
            peak_row = conn.execute("SELECT max(wealth) AS pk FROM wealth_marks WHERE book = %s", (BOOK,)).fetchone(); peak = max(float(peak_row["pk"] or 0.0), wealth)
            conn.execute("INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)",
                         (beat, BOOK, m1, cash, value, exit_cost, wealth, peak, (wealth / peak - 1.0) if peak > 0 else 0.0, value, len(opens)))
            status = {"minute": m1.isoformat(), "mints_traded": len(agg), "eligible": len(mints), "picks": len(picks), "entered": n_enter, "exited": n_exit, "open": len(opens),
                      "cash": cash, "wealth": wealth, "threshold": self.model.threshold, "top_scores": sorted(picks, key=lambda x: -x[1])[:5],
                      "score_p99": float(np.percentile(scores, 99)) if len(scores) else None, "latency_s": (now - m1).total_seconds(), "updated_at": datetime.now(timezone.utc).isoformat()}
            conn.execute("INSERT INTO ui_settings (key, value) VALUES ('selector_status', %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()", (json.dumps(status, default=str),))
        return status

    def finish(self) -> None:
        with transaction() as conn:
            conn.execute("UPDATE runs SET ended_at = now(), status = 'finished' WHERE run_id = %s", (self.run_id,))
        record_event("info", "selector", "selector session stopped", {"run_id": self.run_id, "minutes": self.beat_no})


def main(stop_event: threading.Event | None = None, live: bool = False) -> None:
    s = SelectorSession(live=live)
    m_start = math.floor(time.time() / 60) * 60
    s.warm_up(m_start); s.last_minute = m_start - 60
    try:
        while not (stop_event is not None and stop_event.is_set()):
            now = time.time(); m1 = math.floor(now / 60) * 60
            if s.last_minute is None:
                s.last_minute = m1 - 60
            if m1 > s.last_minute and now - m1 >= 4.0 and s.minute_ready(m1 - 60):   # the minute closed and the stream has written it
                for minute in range(int(s.last_minute) + 60, int(m1) + 1, 60):
                    try:
                        st = s.run_minute(float(minute))
                        if s.beat_no % 10 == 0:
                            log.info("minute %s: traded %d eligible %d picks %d entered %d exited %d open %d wealth %.3f", st["minute"], st["mints_traded"], st["eligible"], st["picks"], st["entered"], st["exited"], st["open"], st["wealth"])
                    except Exception:
                        log.exception("selector minute %s failed", minute); record_event("error", "selector", f"minute failed: {minute}")
                s.last_minute = m1
            time.sleep(0.5)
    finally:
        s.finish()
