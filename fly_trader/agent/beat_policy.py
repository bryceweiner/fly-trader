"""PolicySession: the live beat driven by the trained connectome policy (BRAIN_MODE=policy).

Reuses Session's ingest/slots/features/brokers/rails/persistence; replaces the spiking brain, the KC→MBON
learner and the reflex/threshold decisions with: obs → ConnectomePolicy → target exposure per slot → trades
per book. Reward per slot = net change of the position's marked value plus cash flows, in percent of
MAX_POSITION_SOL (the same reward the policy was trained on), persisted for later online training.
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..execution import ledger
from ..ingest import tape
from ..market.features import TokenMeta
from . import rails
from .beat import Group, Session, SessionOptions, _git_sha
from .policy_driver import PolicyBrain
from .reward import mark_book

log = logging.getLogger(__name__)


class PolicySession(Session):
    def __init__(self, connectome, opts: SessionOptions):
        super().__init__(connectome, opts, corpus=None, split=None)
        self.prev_values: dict[str, dict[str, float]] = {}
        self.brain: PolicyBrain | None = None

    def start(self) -> None:
        off = 0
        for g in self.groups:
            g.offset = off; off += g.n
            g.prev_slot_mints = [None] * g.n
        self.brain = PolicyBrain(self.c, self.B)
        with transaction() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, kind, git_sha, config, connectome_sha256, brain_snapshot_id, status) VALUES (%s,%s,%s,%s,%s,%s,'running')",
                (self.run_id, self.opts.run_kind, _git_sha(), json.dumps(config.summary(), default=str), self.c.sha256, self.brain.snapshot_id))
            for g in self.groups:
                g.slots.load_visits([dict(r) for r in conn.execute("SELECT mint, last_visit_ts, visits FROM slot_visits").fetchall()])
                g.tape_last_id = tape.latest_id(conn)
                self.refresh_meta(g, conn, time.time(), force=True)
        self.started = True

    def save_snapshot(self, conn, kind: str, beat_id: int | None, note: str | None = None) -> int:  # no online learning yet
        return self.brain.snapshot_id or 0

    def finish(self, status: str = "done", metrics: dict | None = None) -> None:
        with transaction() as conn:
            conn.execute("UPDATE runs SET ended_at = now(), status = %s, metrics = %s WHERE run_id = %s",
                         (status, json.dumps(metrics or {}, default=str), self.run_id))
        for g in self.groups:
            if g.exec_worker is not None:
                g.exec_worker.stop()

    # ---------- the beat ----------
    def run_beat(self, now_wall: float) -> dict:
        assert self.started
        t_start = time.perf_counter()
        self.beat_no += 1
        now = now_wall
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
        g = self.groups[0]
        with transaction() as conn:
            rows = tape.tail(conn, g.tape_last_id)
            g.clock = now
            self._pre_brain(g, conn, now, rows)
            pre = g.pre
            slot_mints, active, flows, pending, prices = pre["slot_mints"], pre["active"], pre["flows"], pre["pending"], pre["prices"]
            if pre["changed_cols"]:
                self.brain.reset_columns(pre["changed_cols"])
            # portfolio observation from the reward-source book (live if it holds the token, else the free book)
            n = g.n
            pos_frac = np.zeros(n, np.float32); unreal = np.zeros(n, np.float32); held_b = np.zeros(n, np.float32)
            for i, m in enumerate(slot_mints):
                if not m:
                    continue
                src = "live" if ("live" in g.books and m in g.books["live"].held) else g.free_book
                pos = next((p for p in g.books[src].positions if p["mint"] == m), None) if src else None
                if pos:
                    pos_frac[i] = min(1.0, float(pos.get("cost_sol") or 0.0) / config.MAX_POSITION_SOL)
                    if prices.get(m) and pos.get("entry_price"):
                        unreal[i] = prices[m] / float(pos["entry_price"]) - 1.0
                    held_b[i] = (now - pos["opened_at"].timestamp()) / config.BEAT_S
            t_gpu = time.perf_counter()
            obs = self.brain.observe(pre["feats"], pre["masks"], active, pos_frac, unreal, held_b)
            target, value, mu = self.brain.act(obs, active)
            gpu_ms = int((time.perf_counter() - t_gpu) * 1000)
            # rails for live/mirror increases
            live_blocks: list[str] = []
            if g.apply_rails:
                if g.exec_worker is not None:
                    self._refresh_live_balance(g)
                sol_free_live = g.sol_free_live if g.exec_worker is not None else ledger.paper_cash(conn, "paper_mirror" if "paper_mirror" in g.books else g.free_book)
                live_blocks = rails.entry_blocks(conn, feed_age_s=pre["feed_age"], sol_free=sol_free_live, context_ready=g.ctx.ready(now),
                                                 brain_ok=True, edge=1.0)
            pre["blocks"] = live_blocks
            warm = (now - (g.started_at or now)) >= config.WARMUP_S
            if g.started_at is None:
                g.started_at = now
            dec_rows = list(pre["forced_decisions"])
            slot_dec: dict[int, tuple[str, float, float]] = {}
            n_trades = 0
            for i, m in enumerate(slot_mints):
                if not m or not active[i] or m in pending or not warm:
                    continue
                meta = g.meta.get(m) or TokenMeta(m)
                a = float(target[i])
                for b, bs in g.books.items():
                    pos = next((p for p in bs.positions if p["mint"] == m), None)
                    cur = min(1.0, float(pos.get("cost_sol") or 0.0) / config.MAX_POSITION_SOL) if pos else 0.0
                    delta = a - cur
                    if abs(delta) < config.POLICY_MIN_TRADE_FRAC and not (a < 0.05 and pos):
                        continue
                    blocked = (b in ("live", "paper_mirror")) and delta > 0 and (live_blocks or not getattr(meta, "tradable", True))
                    if delta > 0 and not blocked:
                        size = delta * config.MAX_POSITION_SOL
                        if b == "live":
                            ok = self._enter_live(g, SimpleNamespace(mint=m, size_sol=size), meta)
                        else:
                            ok = self._enter(g, conn, b, m, size, now_dt, flows, allow_add=bool(pos))
                        if ok:
                            n_trades += 1
                            kind = "resize" if pos else "enter"
                            dec_rows.append({"slot": i, "mint": m, "pool": meta.pool, "kind": kind, "m_hat": a, "size_sol": size, "forced": False,
                                             "reason": f"target {a:.2f} from {cur:.2f}", "book_targets": [b], "detail": {"book": b, "value": float(value[i]), "mu": float(mu[i])}})
                            slot_dec[i] = (kind, size, None)
                    elif delta < 0 and pos:
                        frac = 1.0 if a < 0.05 else min(1.0, -delta / max(cur, 1e-9))
                        row = self._reduce(g, conn, b, pos, frac, now_dt, flows, a)
                        if row:
                            n_trades += 1
                            dec_rows.append(row); slot_dec[i] = (row["kind"], row.get("size_sol") or 0.0, None)
                    elif delta > 0 and blocked and b == "live":
                        dec_rows.append({"slot": i, "mint": m, "pool": meta.pool, "kind": "blocked", "rail": ",".join(live_blocks) or "not_tradable",
                                         "reason": "live increase blocked", "book_targets": ["live"], "forced": False, "m_hat": a})
            # marks + rewards
            marks, reward_rows = [], []
            values_by_book: dict[str, dict[str, float]] = {}
            for b, bs in g.books.items():
                bs.positions = ledger.open_positions(conn, b)
                bs.held = {p["mint"] for p in bs.positions}
                sol_free = g.sol_free_live if b == "live" else ledger.paper_cash(conn, b)
                res_q = {p["mint"]: g.bank.res_quote_sol(p["mint"]) for p in bs.positions}
                ages = {p["mint"]: self._age_h(g, p["mint"], now) for p in bs.positions}
                labels = {p["mint"]: (g.meta[p["mint"]].program_label if p["mint"] in g.meta else None) for p in bs.positions}
                for p in bs.positions:
                    p["decimals"] = p.get("decimals") or getattr(g.meta.get(p["mint"]), "decimals", 6)
                wm, per_mint = mark_book(sol_free, bs.positions, prices, res_q, ages, labels)
                values_by_book[b] = per_mint
                if b == "live":
                    peak, _ = rails.update_peak(conn, wm.wealth)
                else:
                    peak = max(g.peak.get(b, wm.wealth), wm.wealth); g.peak[b] = peak
                dd = 1.0 - wm.wealth / peak if peak > 0 else 0.0
                marks.append({"book": b, "sol_free": wm.sol_free, "positions_value": wm.positions_value, "exit_cost": wm.exit_cost,
                              "wealth": wm.wealth, "peak": peak, "drawdown": dd, "exposure": wm.exposure, "n_open": wm.n_open})
            for i, m in enumerate(slot_mints):
                if not m:
                    continue
                src = "live" if ("live" in g.books and m in g.books["live"].held) else g.free_book
                prev = self.prev_values.get(src, {}).get(m, 0.0)
                cur_v = values_by_book.get(src, {}).get(m, 0.0)
                cf = flows.get(src, {}).get(m, 0.0)
                if prev == 0.0 and cur_v == 0.0 and cf == 0.0:
                    continue
                r = 100.0 * (cur_v - prev + cf) / config.MAX_POSITION_SOL
                reward_rows.append({"slot": i, "book": src, "mint": m, "r_slot": r, "r_global": 0.0, "r_tilde": r, "m_prev": float(value[i]),
                                    "m_now": float(target[i]), "delta": 0.0, "source": src})
            for b in g.books:
                self.prev_values[b] = dict(values_by_book[b])
            slot_rows = []
            dwell = g.slots.dwell_of()
            for i, m in enumerate(slot_mints):
                if not m:
                    continue
                dk = slot_dec.get(i)
                slot_rows.append({"slot": i, "mint": m, "pool": g.meta[m].pool if m in g.meta else None, "dwell_beats": dwell[i],
                                  "features": pre["feats"][i].tolist(), "feature_mask": int(pre["masks"][i]), "danger": float(pre["danger"][i]),
                                  "portfolio": [float(pos_frac[i]), float(unreal[i]), float(held_b[i]), 0.0, 0.0], "glomeruli": [], "stim": [],
                                  "kc_active_frac": 0.0, "m_hat": float(target[i]), "rho_app": float(value[i]), "rho_av": float(mu[i]),
                                  "dan_rew": 0.0, "dan_pun": 0.0, "delta_in": 0.0,
                                  "decision_kind": dk[0] if dk else None, "decision_size": dk[1] if dk else None, "softmax_p": None})
            total_ms = int((time.perf_counter() - t_start) * 1000)
            beat = {"beat_no": self.beat_no, "sim_ts": None, "tape_last_id": g.tape_last_id, "n_slots_active": int(active.sum()),
                    "n_held_live": len(g.books["live"].held) if "live" in g.books else 0,
                    "n_held_paper": len(g.books[g.free_book].held) if g.free_book else 0, "ticks": 0, "gpu_ms": gpu_ms, "total_ms": total_ms,
                    "feed_age_ms": int(pre["feed_age"] * 1000) if pre["feed_age"] is not None else None,
                    "notes": {"mode": "policy", "snapshot": self.brain.snapshot_id, "live": {"active": int(active.sum()), "trades": n_trades,
                              "blocks": live_blocks, "warm": warm, "mean_target": float(target[active].mean()) if active.any() else 0.0,
                              "mean_value": float(value[active].mean()) if active.any() else 0.0, "rows": pre["rows"]}}}
            beat_id, _ = self.writer.write(conn, run_id=self.run_id, ts=now_dt, beat=beat, slots=slot_rows if self.opts.persist_slots else [],
                                           activity=None, decisions=dec_rows, rewards=reward_rows, marks=marks, synapse=None)
            g.slots.persist(conn, now_dt)
        return {"beat_id": beat_id, "active": int(active.sum()), "gpu_ms": gpu_ms, "total_ms": total_ms, "mean_mbon": 0.0, "kc": 0.0,
                "blocks": live_blocks, "groups": {"live": beat["notes"]["live"]}, "decisions": n_trades, "forced": len(pre["forced_decisions"])}

    def _reduce(self, g: Group, conn, book: str, pos: dict, frac: float, now_dt: datetime, flows: dict, target: float) -> dict | None:
        mint = pos["mint"]
        meta = g.meta.get(mint) or TokenMeta(mint)
        kind = "exit" if frac >= 0.999 else "resize"
        if book == "live":
            from ..execution.worker import ExecRequest
            req = ExecRequest(decision_id=None, mint=mint, pool=pos.get("pool"), side="sell", amount_in=int(int(pos["qty"]) * frac),
                              slippage_bps=config.SLIPPAGE_EXIT_BPS, max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS,
                              decimals=int(pos.get("decimals") or 6), forced_kind=None, position_id=int(pos["id"]))
            if not g.exec_worker.submit(req):
                return None
            return {"slot": None, "mint": mint, "pool": pos.get("pool"), "kind": kind, "m_hat": target, "size_sol": float(pos.get("cost_sol") or 0.0) * frac,
                    "forced": False, "reason": f"target {target:.2f}", "book_targets": [book], "detail": {"book": book, "fraction": frac}}
        fr = g.books[book].broker.sell(conn, position=pos, decision_id=None, price=g.bank.last_price(mint), res_quote_sol=g.bank.res_quote_sol(mint),
                                       age_hours=self._age_h(g, mint, now_dt.timestamp()), program_label=meta.program_label, forced_kind=None,
                                       ts=now_dt, fraction=frac)
        if not fr.ok:
            return None
        flows[book][mint] = flows[book].get(mint, 0.0) + fr.sol_delta
        if frac >= 0.999:
            g.books[book].held.discard(mint)
        return {"slot": None, "mint": mint, "pool": pos.get("pool"), "kind": kind, "m_hat": target, "size_sol": float(pos.get("cost_sol") or 0.0) * frac,
                "forced": False, "reason": f"target {target:.2f} {fr.reason}", "book_targets": [book], "detail": {"book": book, "fraction": frac}}
