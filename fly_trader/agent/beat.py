"""One beat, one brain, several column groups.

The LIF batch is the concatenation of column groups: the *live* group (pump.fun slots fed by the
swap tape, paper books + optional live book) and any number of *replay* groups (corpus tokens fed
by a ReplayFeed on a simulated clock, one paper book each). All columns share the same plastic
KC→MBON matrix, so the fly learns from live outcomes and replayed history at once.

Order per beat: per group → drain execution results, ingest feed, refresh meta, positions and
forced exits, slot assignment, features, encoder currents; then one LIF run over all columns;
then per group → readout slice, decisions, rails, execution per book, wealth marks, rewards and
RPE (used next beat); then one plasticity step (using last beat's RPE, eligibility buffered one
beat), journal, and one transaction persisting everything.
"""
from __future__ import annotations

import json
import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import torch

from .. import config
from ..brain import checkpoint
from ..brain.encoders import Encoder
from ..brain.lif import LIF
from ..brain.plasticity import Plasticity
from ..db.apilog import record_event
from ..db.connection import transaction
from ..db.writer import BeatWriter
from ..execution import ledger
from ..execution.broker_paper import PaperBroker
from ..ingest import tape
from ..market.context_window import ContextWindow
from ..market.danger import danger_score
from ..market.features import D, FeatureBank, TokenMeta
from . import rails
from .decide import DecisionState, decide
from .innate import EdgeTracker, eligible, innate_score
from .reward import RewardTracker, mark_book
from .slots import SlotManager

log = logging.getLogger(__name__)

BRAIN_RATE_MIN, BRAIN_RATE_MAX = 0.02, 0.8


@dataclass
class SessionOptions:
    run_kind: str = "live"           # live | pretrain
    ticks: int = config.BEAT_TICKS
    learn: bool = True
    persist_slots: bool = True
    initial_snapshot: int | None = None


@dataclass
class BookState:
    broker: object
    held: set[str] = field(default_factory=set)
    positions: list[dict] = field(default_factory=list)


@dataclass
class Group:
    name: str
    n: int
    offset: int = 0
    sim: bool = False
    books: dict[str, BookState] = field(default_factory=dict)
    free_book: str | None = None
    bank: FeatureBank = field(default_factory=FeatureBank)
    ctx: ContextWindow = field(default_factory=ContextWindow)
    slots: SlotManager | None = None
    dstate: DecisionState = field(default_factory=DecisionState)
    rt: RewardTracker = field(default_factory=RewardTracker)
    meta: dict[str, TokenMeta] = field(default_factory=dict)
    meta_refreshed: float = 0.0
    pending_delta: dict[str, tuple[float, str]] = field(default_factory=dict)
    realized_pulse: dict[str, float] = field(default_factory=dict)   # mint -> realized return of a trade closed this beat
    last_mark_ts: float | None = None
    edge: EdgeTracker = field(default_factory=EdgeTracker)
    last_edge: float | None = None
    last_edge_n: int = 0
    started_at: float | None = None
    peak: dict[str, float] = field(default_factory=dict)      # book -> highest wealth since this run started
    prev_slot_mints: list = field(default_factory=list)
    clock: float = 0.0
    feed: object = None              # ReplayFeed for replay groups
    tape_last_id: int = 0
    exec_worker: object = None
    rpc: object = None
    pubkey: str | None = None
    sol_free_live: float = 0.0
    apply_rails: bool = False
    # per-beat scratch
    pre: dict = field(default_factory=dict)


class Session:
    def __init__(self, connectome, opts: SessionOptions, corpus: str | None = None, split: dict | None = None):
        self.c = connectome
        self.opts = opts
        self.dev = connectome.device
        self.run_id = str(uuid.uuid4())
        self.groups: list[Group] = []
        self.writer = BeatWriter()
        self.rng = np.random.default_rng(int(time.time()))
        self.beat_no = 0
        self.snapshot_beat = 0
        self.k_prev: torch.Tensor | None = None
        self.corpus, self.split = corpus, split
        self.lif = None
        self.enc = None
        self.plast = None
        self.started = False

    # ---------- groups ----------
    def add_live_group(self, n: int, books: tuple[str, ...] = ("paper_free", "paper_mirror"), live: bool = False) -> Group:
        g = Group(name="live", n=n, sim=False, apply_rails=True)
        for b in books:
            g.books[b] = BookState(PaperBroker(b))
        g.free_book = next((b for b in books if b != "paper_mirror"), None)
        if live:
            from ..chain import keys
            from ..chain.rpc import HttpSolanaRpc
            from ..execution.broker_live import LiveBroker
            from ..execution.worker import ExecutionWorker
            g.rpc = HttpSolanaRpc()
            g.pubkey = keys.bot_pubkey()
            g.exec_worker = ExecutionWorker(LiveBroker(rpc=g.rpc))
            g.books["live"] = BookState(None)
        g.slots = SlotManager(n)
        self.groups.append(g)
        return g

    def add_replay_group(self, n: int, feed, book: str, name: str = "replay") -> Group:
        g = Group(name=name, n=n, sim=True, feed=feed, apply_rails=False)
        g.books[book] = BookState(PaperBroker(book))
        g.free_book = book
        g.meta = feed.build_meta()
        g.meta_refreshed = float("inf")
        for m in g.meta.values():
            if m.pool:
                g.bank.pool_to_mint[m.pool] = m.mint
        g.clock = feed.t0
        g.slots = SlotManager(n)
        self.groups.append(g)
        return g

    @property
    def B(self) -> int:
        return sum(g.n for g in self.groups)

    def start(self) -> None:
        off = 0
        for g in self.groups:
            g.offset = off
            off += g.n
            g.prev_slot_mints = [None] * g.n
        self.lif = LIF(self.c, batch=self.B, ticks=self.opts.ticks)
        calib = self._load_calibration()
        self.enc = Encoder(self.c, self.B, dan_gain=calib.get("dan_gain"))
        self.dan_rate_max = float(calib.get("dan_rate_max_above_baseline") or calib.get("dan_rate_max") or 0.5)
        self.dan_base_pam = float(calib.get("dan_baseline_pam") or 0.0)
        self.dan_base_ppl1 = float(calib.get("dan_baseline_ppl1") or 0.0)
        self.dan_measured = (config.DAN_MODE == "measured") and self.dan_rate_max > 0.05
        self.m_hat_baseline = float(calib.get("m_hat_baseline") or 0.0)
        s_sign = torch.zeros(self.c.W_KM0.shape[1], device=self.dev)
        s_sign[self.c.mbon_app_cols] = 1.0
        s_sign[self.c.mbon_av_cols] = -1.0
        W0 = self.c.W_KM0.clone()
        k_app, k_av = float(calib.get("km_balance_app") or 1.0), float(calib.get("km_balance_av") or 1.0)
        if k_app != 1.0 or k_av != 1.0:
            W0[:, self.c.mbon_app_cols] *= k_app
            W0[:, self.c.mbon_av_cols] *= k_av
        self.plast = Plasticity(W0, self.c.M_KM, s_sign, self.B)
        self._restore_brain()
        self.lif.set_W_KM(self.plast.W)
        with transaction() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, kind, git_sha, config, connectome_sha256, encoder_sha256, corpus, split, status) "
                "VALUES (%s,%s,%s,%s,%s,%s,%s,%s,'running')",
                (self.run_id, self.opts.run_kind, _git_sha(), json.dumps(config.summary(), default=str), self.c.sha256,
                 self.enc.odor.sha256, self.corpus, json.dumps(self.split or {}, default=str)),
            )
            for g in self.groups:
                if not g.sim:
                    g.slots.load_visits([dict(r) for r in conn.execute("SELECT mint, last_visit_ts, visits FROM slot_visits").fetchall()])
                    g.tape_last_id = tape.latest_id(conn)
                    self.refresh_meta(g, conn, time.time(), force=True)
        self.started = True

    # ---------- setup helpers ----------
    @staticmethod
    def _load_calibration() -> dict:
        p = config.BRAIN_DIR / "calibration.json"
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                return {}
        return {}

    def _restore_brain(self) -> None:
        with transaction() as conn:
            sid = self.opts.initial_snapshot
            if sid is None:
                sid = checkpoint.consume_pending(conn) or checkpoint.live_id(conn)
        if sid is not None:
            try:
                W, enc_state, row = checkpoint.load_snapshot(sid)
                snap_sha = (enc_state or {}).get("connectome_sha256")
                if snap_sha and snap_sha != self.c.content_sha256:
                    log.warning("snapshot %s was made for another connectome build; starting from the prior", sid)
                    record_event("warn", "runner", "snapshot refused: connectome mismatch", {"snapshot_id": sid})
                elif W.shape == tuple(self.plast.W.shape):
                    self.plast.load_W(W)
                    self.enc.load_state(enc_state)
                    log.info("restored brain snapshot %s (%s)", sid, row.get("kind"))
                    record_event("info", "runner", "brain snapshot restored", {"snapshot_id": sid})
                    return
                else:
                    log.warning("snapshot %s shape mismatch; starting from the connectome prior", sid)
            except Exception as e:
                log.warning("snapshot %s not loadable (%s); starting from the connectome prior", sid, e)
        log.info("brain starts from the connectome prior W_KM0")

    def save_snapshot(self, conn, kind: str, beat_id: int | None, note: str | None = None) -> int:
        enc_state = dict(self.enc.state(), connectome_sha256=self.c.content_sha256)
        sid, path, sha = checkpoint.save_snapshot(self.plast.W.detach().cpu().numpy(), enc_state, kind=kind,
                                                 run_id=self.run_id, beat_id=beat_id, note=note, conn=conn)
        if self.opts.run_kind == "live":
            checkpoint.set_live(sid, conn)
        return sid

    # ---------- meta ----------
    def refresh_meta(self, g: Group, conn, now: float, force: bool = False) -> None:
        if g.sim or (not force and now - g.meta_refreshed < 60.0):
            return
        g.meta_refreshed = now
        rows = conn.execute(
            """SELECT wp.pool, wp.mint, wp.program_label, wp.tradable, t.decimals, t.graduated_at, t.token_program,
                      (SELECT row_to_json(s) FROM (SELECT organic_score, holder_count, liquidity_usd, top_holders_pct, dev_balance_pct,
                              is_sus, is_verified, stats FROM token_stats ts WHERE ts.mint = wp.mint ORDER BY ts DESC LIMIT 1) s) AS stats
               FROM watch_pools wp LEFT JOIN tokens t ON t.mint = wp.mint WHERE wp.active"""
        ).fetchall()
        meta: dict[str, TokenMeta] = {}
        for r in rows:
            st = r["stats"]
            if isinstance(st, str):
                st = json.loads(st)
            m = TokenMeta(mint=r["mint"], pool=r["pool"], program_label=r["program_label"],
                          graduated_at=r["graduated_at"].timestamp() if r["graduated_at"] else None,
                          token2022=(r["token_program"] or "").startswith("Tokenz"), stats=st)
            m.decimals = int(r["decimals"] or 6)  # type: ignore[attr-defined]
            m.tradable = bool(r["tradable"])       # type: ignore[attr-defined]
            meta[r["mint"]] = m
            g.bank.pool_to_mint[r["pool"]] = r["mint"]
        for bs in g.books.values():
            for m in bs.held:
                if m not in meta and m in g.meta:
                    meta[m] = g.meta[m]
        g.meta = meta

    @staticmethod
    def _age_h(g: Group, mint: str, now: float) -> float | None:
        m = g.meta.get(mint)
        return (now - m.graduated_at) / 3600.0 if (m and m.graduated_at) else None

    # ---------- the beat ----------
    def run_beat(self, now_wall: float) -> dict:
        assert self.started
        t_start = time.perf_counter()
        self.beat_no += 1
        now_dt = datetime.fromtimestamp(now_wall, tz=timezone.utc)
        with transaction() as conn:
            # ---- phase 1: per group, everything before the brain ----
            I_parts, keep_parts, changed_all = [], [], []
            for g in self.groups:
                if g.sim:
                    g.clock += config.BEAT_S_SIM
                    rows = g.feed.rows_until(g.clock)
                    t_g = g.clock
                else:
                    g.clock = now_wall
                    rows = tape.tail(conn, g.tape_last_id)
                    t_g = now_wall
                self._pre_brain(g, conn, t_g, rows)
                pre = g.pre
                I_g, drive, stim = self.enc.build(pre["feats"], pre["masks"], pre["slot_mints"], pre["active"], pre["sweet"],
                                                  pre["bitter"], pre["danger"], pre["hunger"], pre["dan_rew"], pre["dan_pun"],
                                                  update_stats=self.opts.learn)
                pre["drive"], pre["stim"] = drive, stim
                I_parts.append(I_g)
                keep_parts.append(self.enc.last_keep)
                changed_all.extend(g.offset + i for i in pre["changed_cols"])
            # ---- phase 2: one brain ----
            t_gpu = time.perf_counter()
            if changed_all:
                self.lif.reset(changed_all)
                self.plast.reset_columns(changed_all)
                if self.k_prev is not None:
                    self.k_prev[:, changed_all] = 0.0
            self.lif.set_input(torch.cat(I_parts, dim=1))
            self.lif.set_kc_winners(torch.cat(keep_parts, dim=1) if all(k is not None for k in keep_parts) else None)
            ro = self.lif.run(self.opts.ticks)
            gpu_ms = int((time.perf_counter() - t_gpu) * 1000)
            m_hat_all = ro.m_hat.detach().cpu().numpy() - self.m_hat_baseline
            app_all, av_all = ro.mbon_app_rate.detach().cpu().numpy(), ro.mbon_av_rate.detach().cpu().numpy()
            dpam_all, dppl_all = ro.dan_pam_rate.detach().cpu().numpy(), ro.dan_ppl1_rate.detach().cpu().numpy()
            kc_frac_all = ro.kc_active_frac.detach().cpu().numpy()
            active_all = np.concatenate([g.pre["active"] for g in self.groups])
            mean_mbon = float(np.mean((app_all + av_all)[active_all]) if active_all.any() else 0.0)
            brain_ok = (BRAIN_RATE_MIN <= mean_mbon <= BRAIN_RATE_MAX) and not ro.nan_flag
            # ---- phase 3: per group, decisions / execution / marks / rewards ----
            d_hat_all = np.zeros(self.B, dtype=np.float32)
            learn_all = np.zeros(self.B, dtype=bool)
            dec_rows, marks, reward_rows, slot_rows = [], [], [], []
            summary = {}
            for g in self.groups:
                sl = slice(g.offset, g.offset + g.n)
                m_hat = np.where(g.pre["active"], m_hat_all[sl], 0.0)
                out = self._post_brain(g, conn, now_dt, m_hat, app_all[sl], av_all[sl], dpam_all[sl], dppl_all[sl],
                                       kc_frac_all[sl], brain_ok)
                d_hat_all[sl], learn_all[sl] = out["d_hat"], out["learn"]
                dec_rows += out["dec_rows"]; marks += out["marks"]; reward_rows += out["reward_rows"]; slot_rows += out["slot_rows"]
                summary[g.name] = out["summary"]
            # ---- phase 4: plasticity with last beat's RPE ----
            upd, jpath, jsha = None, None, None
            if self.opts.learn:
                upd = self.plast.step(torch.tensor(d_hat_all, device=self.dev, dtype=torch.float32),
                                      torch.tensor(learn_all, device=self.dev))
                if self.k_prev is not None:
                    self.plast.update_eligibility(self.k_prev)
                self.k_prev = ro.kc_rates.detach().clone()
                self.lif.set_W_KM(self.plast.W)
                jpath, jsha = self.plast.journal_beat(self.beat_no, now_dt, ro.kc_rates, d_hat_all, learn_all)
            # ---- phase 5: persist ----
            pop = ro.pop_rates.detach().cpu().numpy()
            activity = {"group_rates": pop.mean(axis=1).tolist(),
                        "kc_sparsity": float(np.mean(kc_frac_all[active_all]) if active_all.any() else 0.0),
                        "mbon_app_rate": float(np.mean(app_all[active_all]) if active_all.any() else 0.0),
                        "mbon_av_rate": float(np.mean(av_all[active_all]) if active_all.any() else 0.0),
                        "dan_rew_rate": float(np.mean(dpam_all[active_all]) if active_all.any() else 0.0),
                        "dan_pun_rate": float(np.mean(dppl_all[active_all]) if active_all.any() else 0.0),
                        "total_spikes": int(ro.total_spikes), "v_mean": float(ro.v_mean), "v_max": float(ro.v_max),
                        "nan_flag": bool(ro.nan_flag)}
            synapse = None
            if upd is not None:
                synapse = {"eta": upd.eta, "gamma": config.GAMMA, "n_slots": upd.n_slots, "reward_source": "mixed", "sum_abs": upd.sum_abs,
                           "max_abs": upd.max_abs, "frob": upd.frob, "frob_capped": upd.frob_capped, "n_pos": upd.n_pos,
                           "n_neg": upd.n_neg, "n_clipped": upd.n_clipped, "w_mean": upd.w_mean, "w_min": upd.w_min,
                           "w_max": upd.w_max, "delta_path": jpath, "delta_sha256": jsha}
            live_g = next((g for g in self.groups if not g.sim), None)
            replay_g = next((g for g in self.groups if g.sim), None)
            total_ms = int((time.perf_counter() - t_start) * 1000)
            feed_age = live_g.pre["feed_age"] if live_g else None
            beat = {"beat_no": self.beat_no,
                    "sim_ts": datetime.fromtimestamp(replay_g.clock, tz=timezone.utc) if (replay_g and not live_g) else None,
                    "tape_last_id": live_g.tape_last_id if live_g else None,
                    "n_slots_active": int(active_all.sum()),
                    "n_held_live": len(live_g.books["live"].held) if (live_g and "live" in live_g.books) else 0,
                    "n_held_paper": len(live_g.books[live_g.free_book].held) if (live_g and live_g.free_book) else 0,
                    "ticks": self.opts.ticks, "gpu_ms": gpu_ms, "total_ms": total_ms,
                    "feed_age_ms": int(feed_age * 1000) if feed_age is not None else None,
                    "notes": {"mean_mbon": mean_mbon, "brain_ok": brain_ok, **{k: v for k, v in summary.items()},
                              "replay_clock": datetime.fromtimestamp(replay_g.clock, tz=timezone.utc).isoformat() if replay_g else None}}
            beat_id, _ = self.writer.write(conn, run_id=self.run_id, ts=now_dt, beat=beat, slots=slot_rows if self.opts.persist_slots else [],
                                           activity=activity, decisions=dec_rows, rewards=reward_rows, marks=marks, synapse=synapse)
            if live_g:
                live_g.slots.persist(conn, now_dt)
            if replay_g:
                conn.execute("INSERT INTO ui_settings (key, value) VALUES (%s, %s) ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value, updated_at = now()",
                             (f"replay_clock:{replay_g.name}", json.dumps({"clock": replay_g.clock, "corpus": self.corpus})))
            if self.opts.learn and (self.beat_no - self.snapshot_beat >= config.SNAPSHOT_EVERY_BEATS):
                self.save_snapshot(conn, "auto", beat_id)
                self.snapshot_beat = self.beat_no
            if live_g:
                req = conn.execute("SELECT id, detail FROM events WHERE message = 'snapshot_request' AND (detail->>'done') IS NULL ORDER BY id LIMIT 1").fetchone()
                if req:
                    sid = self.save_snapshot(conn, (req["detail"] or {}).get("kind", "manual"), beat_id)
                    conn.execute("UPDATE events SET detail = detail || %s WHERE id = %s", (json.dumps({"done": True, "snapshot_id": sid}), req["id"]))
        blocks = live_g.pre.get("blocks", []) if live_g else []
        return {"beat_id": beat_id, "active": int(active_all.sum()), "gpu_ms": gpu_ms, "total_ms": total_ms, "mean_mbon": mean_mbon,
                "kc": activity["kc_sparsity"], "blocks": blocks, "groups": summary,
                "decisions": sum(s.get("decisions", 0) for s in summary.values()), "forced": sum(s.get("forced", 0) for s in summary.values())}

    # ---------- phase 1 ----------
    def _pre_brain(self, g: Group, conn, now: float, rows: list[dict]) -> None:
        now_dt = datetime.fromtimestamp(now, tz=timezone.utc)
        flows = {b: {} for b in (*g.books, "live")}
        if g.exec_worker is not None:
            for res in g.exec_worker.drain():
                m = res.request.mint
                if res.ok:
                    flows["live"][m] = flows["live"].get(m, 0.0) + res.lamports_delta / config.LAMPORTS_PER_SOL
                    if res.request.side == "sell" and res.realized_sol is not None and res.lamports_delta > 0:
                        proceeds = res.lamports_delta / config.LAMPORTS_PER_SOL
                        cost = proceeds - float(res.realized_sol)
                        if cost > 0:
                            g.realized_pulse[m] = config.REALIZED_GAIN * (proceeds / cost - 1.0)
                else:
                    record_event("warn", "runner", "live execution failed", {"mint": m, "error": res.error, "side": res.request.side})
        for r in rows:
            g.bank.ingest_tape_row(r)
            if r.get("id") is not None:
                g.tape_last_id = max(g.tape_last_id, int(r["id"]))
        g.ctx.observe(now, had_activity=len(rows) > 0)
        feed_age = g.ctx.feed_age_s(now)
        self.refresh_meta(g, conn, now)
        for b, bs in g.books.items():
            bs.positions = ledger.open_positions(conn, b)
            bs.held = {p["mint"] for p in bs.positions}
        pending = g.exec_worker.pending() if g.exec_worker else set()
        held_any = set().union(*[bs.held for bs in g.books.values()]) | pending
        prices = {m: g.bank.last_price(m) for m in held_any | set(g.meta)}
        last_swaps = {m: g.bank.last_swap_ts(m) for m in held_any}
        dt_s = (now - g.last_mark_ts) if getattr(g, "last_mark_ts", None) else 0.0
        g.last_mark_ts = now
        for b in g.books:
            ledger.mark_positions(conn, b, {m: p for m, p in prices.items() if p}, last_swaps, now_dt, dt_s=dt_s)
        for bs in g.books.values():           # re-read so peak/satiety are current for the reflex rails
            bs.positions = ledger.open_positions(conn, next(k for k, v in g.books.items() if v is bs))
        forced_decisions: list[dict] = []
        for b, bs in g.books.items():
            for p in bs.positions:
                if p["mint"] in pending:
                    continue
                kind = rails.forced_exit_kind(p, prices.get(p["mint"]), last_swaps.get(p["mint"]), now,
                                              rvol=g.bank.rvol(p["mint"], now, 900.0))
                if kind:
                    forced_decisions.append(self._exit(g, conn, b, p, kind, now_dt, flows, m_hat=None))
        candidates = {m: g.bank.activity_sol(m, now) for m in g.meta}
        for m in held_any:
            candidates.setdefault(m, float("inf"))
        slot_mints = g.slots.assign(now, held_any, candidates, dict(g.dstate.last_z))
        changed_cols = [i for i, m in enumerate(slot_mints) if m != g.prev_slot_mints[i]]
        g.prev_slot_mints = list(slot_mints)
        active = np.array([m is not None for m in slot_mints])
        n = g.n
        feats = np.zeros((n, D), dtype=np.float32)
        masks = np.zeros(n, dtype=np.int64)
        danger = np.zeros(n, dtype=np.float32)
        sweet = np.zeros(n, dtype=np.float32)
        bitter = np.zeros(n, dtype=np.float32)
        dan_rew = np.zeros(n, dtype=np.float32)
        dan_pun = np.zeros(n, dtype=np.float32)
        portfolio = np.zeros((n, 5), dtype=np.float32)
        deployable = max(config.CAPITAL_SOL - config.GAS_RESERVE_SOL, 1e-6)
        src_book = "live" if "live" in g.books else g.free_book
        deployed = sum(float(p.get("cost_sol") or 0.0) for p in g.books[src_book].positions) if src_book else 0.0
        g.dstate.exposure_frac = min(1.0, deployed / deployable)
        hunger = np.full(n, g.dstate.hunger(), dtype=np.float32)
        g.edge.update(now, lambda m, t: (g.bank.states[m].price_at(t) if m in g.bank.states else None))
        g.last_edge, g.last_edge_n = g.edge.edge()
        no_edge_bitter = config.NO_EDGE_BITTER * max(0.0, -(g.last_edge or 0.0)) if g.last_edge is not None else 0.0
        wealth_prev = g.rt.prev_wealth.get("live") or (g.rt.prev_wealth.get(g.free_book) if g.free_book else None) or config.CAPITAL_SOL
        for i, m in enumerate(slot_mints):
            if not m:
                continue
            meta = g.meta.get(m) or TokenMeta(m)
            f, mk = g.bank.state(m).features(now, meta)
            feats[i], masks[i] = f, mk
            danger[i] = danger_score(f, mk, self._age_h(g, m, now))
            src = "live" if ("live" in g.books and m in g.books["live"].held) else g.free_book
            pos = next((p for p in g.books[src].positions if p["mint"] == m), None) if src else None
            u = 0.0
            if pos and prices.get(m) and pos.get("entry_price"):
                u = prices[m] / float(pos["entry_price"]) - 1.0
                held_h = (now - pos["opened_at"].timestamp()) / 3600.0
                portfolio[i] = [1.0, u, math.log1p(max(0.0, held_h)), float(pos.get("cost_sol") or 0.0) / max(config.CAPITAL_SOL, 1e-9),
                                float(pos.get("satiety") or 0.0) / max(config.SATIETY_TARGET, 1e-9)]
            taste = u / config.REWARD_UNIT + g.realized_pulse.get(m, 0.0) / config.REWARD_UNIT
            sweet[i], bitter[i] = max(0.0, taste), max(0.0, -taste) + no_edge_bitter
            d = g.pending_delta.get(m)
            if d is not None:
                dan_rew[i], dan_pun[i] = max(0.0, d[0]), max(0.0, -d[0])
        g.pre = {"now": now, "now_dt": now_dt, "rows": len(rows), "flows": flows, "feed_age": feed_age, "pending": pending,
                 "held_any": held_any, "prices": prices, "forced_decisions": forced_decisions, "slot_mints": slot_mints,
                 "changed_cols": changed_cols, "active": active, "feats": feats, "masks": masks, "danger": danger, "sweet": sweet,
                 "bitter": bitter, "dan_rew": dan_rew, "dan_pun": dan_pun, "portfolio": portfolio, "hunger": hunger}

    # ---------- phase 3 ----------
    def _post_brain(self, g: Group, conn, now_dt_wall: datetime, m_hat, app, av, dpam, dppl, kc_frac, brain_ok) -> dict:
        pre = g.pre
        now, now_dt = pre["now"], pre["now_dt"]
        slot_mints, active, flows, pending, held_any, prices = pre["slot_mints"], pre["active"], pre["flows"], pre["pending"], pre["held_any"], pre["prices"]
        learn_mask = np.array([bool(m and m in g.pending_delta) for m in slot_mints])
        if self.dan_measured:
            d_hat = np.clip(((dpam - self.dan_base_pam) - (dppl - self.dan_base_ppl1)) / max(self.dan_rate_max, 1e-6) * config.DELTA_CLIP,
                            -config.DELTA_CLIP, config.DELTA_CLIP)
        else:
            d_hat = np.array([g.pending_delta[m][0] if (m and m in g.pending_delta) else 0.0 for m in slot_mints], dtype=np.float32)
        free_cash = ledger.paper_cash(conn, g.free_book) - config.GAS_RESERVE_SOL if g.free_book else config.CAPITAL_SOL
        innate = innate_score(pre["feats"], active)
        valence = np.where(active, m_hat + config.INNATE_GAIN * innate, 0.0)
        g.edge.record(now, slot_mints, valence, g.bank.last_price)
        if g.started_at is None:
            g.started_at = now
        warm = (now - g.started_at) >= config.WARMUP_S
        elig = eligible(pre["feats"], pre["masks"], active, [self._age_h(g, m, now) if m else None for m in slot_mints])
        decisions = decide(g.dstate, slot_mints, valence, held_any, free_cash, self.rng, exposure_frac=g.dstate.exposure_frac,
                           dwell=g.slots.dwell_of(), allow_entries=warm, eligible=elig)
        g.dstate.beats_since_entry += 1
        live_blocks: list[str] = []
        if g.apply_rails:
            if g.exec_worker is not None:
                self._refresh_live_balance(g)
            sol_free_live = g.sol_free_live if g.exec_worker is not None else ledger.paper_cash(conn, "paper_mirror" if "paper_mirror" in g.books else g.free_book)
            live_blocks = rails.entry_blocks(conn, feed_age_s=pre["feed_age"], sol_free=sol_free_live, context_ready=g.ctx.ready(now),
                                             brain_ok=brain_ok, edge=g.last_edge)
        pre["blocks"] = live_blocks
        dec_rows: list[dict] = list(pre["forced_decisions"])
        slot_dec: dict[int, tuple[str, float, float | None]] = {}
        entered = False
        for d in decisions:
            meta = g.meta.get(d.mint) or TokenMeta(d.mint)
            slot_dec[d.slot] = (d.kind, d.size_sol, d.softmax_p)
            targets = []
            row = {"slot": g.offset + d.slot, "mint": d.mint, "pool": meta.pool, "kind": d.kind, "m_hat": float(m_hat[d.slot]), "size_sol": d.size_sol,
                   "forced": False, "reason": d.reason, "softmax_p": d.softmax_p,
                   "detail": {"group": g.name, "valence": d.m_hat, "innate": float(innate[d.slot]), "z": d.z, "z_buy_eff": g.dstate.z_buy_eff(),
                              "hunger": g.dstate.hunger(), "mu": g.dstate.last_mu, "sigma": g.dstate.last_sigma, "edge": g.last_edge, "edge_n": g.last_edge_n}}
            if d.kind == "enter":
                if g.free_book and self._enter(g, conn, g.free_book, d.mint, d.size_sol, now_dt, flows):
                    targets.append(g.free_book)
                if "paper_mirror" in g.books and not live_blocks and getattr(meta, "tradable", True):
                    if self._enter(g, conn, "paper_mirror", d.mint, d.size_sol, now_dt, flows):
                        targets.append("paper_mirror")
                if g.exec_worker is not None and not live_blocks and getattr(meta, "tradable", True) and d.mint not in pending:
                    if self._enter_live(g, d, meta):
                        targets.append("live")
                if live_blocks:
                    row["rail"] = ",".join(live_blocks)
                entered = entered or bool(targets)
            else:
                for b, bs in g.books.items():
                    pos = next((p for p in bs.positions if p["mint"] == d.mint), None)
                    if pos is None or d.mint in pending:
                        continue
                    self._exit(g, conn, b, pos, None, now_dt, flows, m_hat=d.m_hat)
                    targets.append(b)
            row["book_targets"] = targets
            dec_rows.append(row)
        if entered:
            g.dstate.beats_since_entry = 0
        if live_blocks and any(d.kind == "enter" for d in decisions):
            dec_rows.append({"slot": None, "mint": None, "kind": "blocked", "rail": ",".join(live_blocks), "reason": "live entries blocked",
                             "book_targets": ["live"], "forced": False, "detail": {"group": g.name}})
        # marks
        marks: list[dict] = []
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
                peak = max(g.peak.get(b, wm.wealth), wm.wealth)   # running maximum since the run started
                g.peak[b] = peak
            dd = 1.0 - wm.wealth / peak if peak > 0 else 0.0
            marks.append({"book": b, "sol_free": wm.sol_free, "positions_value": wm.positions_value, "exit_cost": wm.exit_cost,
                          "wealth": wm.wealth, "peak": peak, "drawdown": dd, "exposure": wm.exposure, "n_open": wm.n_open})
        # rewards → RPE for next beat
        reward_rows: list[dict] = []
        new_pending: dict[str, tuple[float, str]] = {}
        r_global = {m["book"]: g.rt.global_reward(m["book"], m["wealth"]) for m in marks}
        # dopamine = the position's unrealized return (level, every beat) + the realized return of a trade closed
        # this beat; 1 unit = REWARD_UNIT; clipped at ±DELTA_CLIP. Simple and sharp (operator design).
        for i, m in enumerate(slot_mints):
            if not m:
                continue
            src = "live" if ("live" in g.books and m in g.books["live"].held) else g.free_book
            pos = next((p for p in g.books[src].positions if p["mint"] == m), None) if src else None
            pulse = g.realized_pulse.pop(m, 0.0)
            if pos is None and pulse == 0.0:
                g.rt.remember_m_hat(m, float(m_hat[i]))
                continue
            u = 0.0
            if pos and prices.get(m) and pos.get("entry_price"):
                u = prices[m] / float(pos["entry_price"]) - 1.0
            r = u + pulse
            r_t = max(-config.DELTA_CLIP, min(config.DELTA_CLIP, r / config.REWARD_UNIT))
            delta, m_prev = g.rt.rpe(m, r_t, float(m_hat[i]))
            g.rt.remember_m_hat(m, float(m_hat[i]))
            new_pending[m] = (delta, src)
            reward_rows.append({"slot": g.offset + i, "book": src, "mint": m, "r_slot": r, "r_global": r_global[src], "r_tilde": r_t,
                                "m_prev": m_prev, "m_now": float(m_hat[i]), "delta": delta, "source": src})
        for m in marks:
            g.rt.commit(m["book"], m["wealth"], values_by_book[m["book"]])
        for m, pulse in list(g.realized_pulse.items()):   # trade closed for a token that left the slots: still felt
            r_t = max(-config.DELTA_CLIP, min(config.DELTA_CLIP, pulse / config.REWARD_UNIT))
            delta, m_prev = g.rt.rpe(m, r_t, g.rt.prev_m_hat.get(m, 0.0))
            new_pending[m] = (delta, src if (src := ("live" if "live" in g.books else g.free_book)) else "paper_free")
            reward_rows.append({"slot": None, "book": new_pending[m][1], "mint": m, "r_slot": pulse, "r_global": 0.0, "r_tilde": r_t,
                                "m_prev": m_prev, "m_now": m_prev, "delta": delta, "source": "realized"})
        g.realized_pulse.clear()
        g.pending_delta = new_pending
        # slot rows
        slot_rows = []
        dwell = g.slots.dwell_of()
        for i, m in enumerate(slot_mints):
            if not m:
                continue
            dk = slot_dec.get(i)
            slot_rows.append({"slot": g.offset + i, "mint": m, "pool": g.meta[m].pool if m in g.meta else None, "dwell_beats": dwell[i],
                              "features": pre["feats"][i].tolist(), "feature_mask": int(pre["masks"][i]), "danger": float(pre["danger"][i]),
                              "portfolio": pre["portfolio"][i].tolist(), "glomeruli": pre["drive"][i].tolist(), "stim": pre["stim"][i].tolist() + [float(innate[i]), float(valence[i])],
                              "kc_active_frac": float(kc_frac[i]), "m_hat": float(m_hat[i]), "rho_app": float(app[i]), "rho_av": float(av[i]),
                              "dan_rew": float(dpam[i]), "dan_pun": float(dppl[i]), "delta_in": float(pre["dan_rew"][i] - pre["dan_pun"][i]),
                              "decision_kind": dk[0] if dk else None, "decision_size": dk[1] if dk else None, "softmax_p": dk[2] if dk else None})
        summary = {"active": int(active.sum()), "decisions": len(decisions), "forced": len(pre["forced_decisions"]), "rows": pre["rows"],
                   "blocks": live_blocks, "clock": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
                   "edge": g.last_edge, "edge_n": g.last_edge_n, "hunger": g.dstate.hunger(), "warm": warm, "eligible": int(elig.sum())}
        return {"d_hat": d_hat, "learn": learn_mask, "dec_rows": dec_rows, "marks": marks, "reward_rows": reward_rows,
                "slot_rows": slot_rows, "summary": summary}

    # ---------- execution helpers ----------
    def _refresh_live_balance(self, g: Group) -> None:
        try:
            g.sol_free_live = g.rpc.get_balance(g.pubkey) / config.LAMPORTS_PER_SOL
        except Exception as e:
            log.warning("balance refresh failed: %s", type(e).__name__)

    def _enter(self, g: Group, conn, book: str, mint: str, size_sol: float, now_dt: datetime, flows: dict) -> bool:
        meta = g.meta.get(mint) or TokenMeta(mint)
        bs = g.books[book]
        if mint in bs.held:
            return False
        cash = ledger.paper_cash(conn, book)
        if cash - size_sol < config.GAS_RESERVE_SOL:
            return False
        fr = bs.broker.buy(conn, decision_id=None, mint=mint, pool=meta.pool, size_sol=size_sol, price=g.bank.last_price(mint),
                           res_quote_sol=g.bank.res_quote_sol(mint), age_hours=self._age_h(g, mint, now_dt.timestamp()),
                           decimals=getattr(meta, "decimals", 6), program_label=meta.program_label, ts=now_dt)
        if fr.ok:
            flows[book][mint] = flows[book].get(mint, 0.0) + fr.sol_delta
            bs.held.add(mint)
        return fr.ok

    def _enter_live(self, g: Group, d, meta) -> bool:
        from ..execution.worker import ExecRequest
        if g.sol_free_live - d.size_sol < config.GAS_RESERVE_SOL:
            return False
        req = ExecRequest(decision_id=None, mint=d.mint, pool=meta.pool, side="buy", amount_in=int(d.size_sol * config.LAMPORTS_PER_SOL),
                          slippage_bps=config.SLIPPAGE_ENTRY_BPS, max_slippage_bps=config.MAX_SLIPPAGE_ENTRY_BPS,
                          decimals=getattr(meta, "decimals", 6), size_sol=d.size_sol)
        return g.exec_worker.submit(req)

    def _exit(self, g: Group, conn, book: str, pos: dict, forced_kind: str | None, now_dt: datetime, flows: dict, m_hat: float | None) -> dict:
        mint = pos["mint"]
        meta = g.meta.get(mint) or TokenMeta(mint)
        row = {"slot": None, "mint": mint, "pool": pos.get("pool"), "kind": forced_kind or "exit", "m_hat": m_hat,
               "forced": forced_kind is not None, "rail": forced_kind, "reason": f"{forced_kind or 'valence'} on {book}",
               "book_targets": [book], "size_sol": None, "detail": {"group": g.name}}
        if book == "live":
            from ..execution.worker import ExecRequest
            slip = config.SLIPPAGE_FORCED_BPS if forced_kind else config.SLIPPAGE_EXIT_BPS
            mx = config.MAX_SLIPPAGE_FORCED_BPS if forced_kind else config.MAX_SLIPPAGE_ENTRY_BPS
            req = ExecRequest(decision_id=None, mint=mint, pool=pos.get("pool"), side="sell", amount_in=int(pos["qty"]), slippage_bps=slip,
                              max_slippage_bps=mx, decimals=int(pos.get("decimals") or 6), forced_kind=forced_kind, position_id=int(pos["id"]))
            g.exec_worker.submit(req)
            return row
        fr = g.books[book].broker.sell(conn, position=pos, decision_id=None, price=g.bank.last_price(mint),
                                       res_quote_sol=g.bank.res_quote_sol(mint), age_hours=self._age_h(g, mint, now_dt.timestamp()),
                                       program_label=meta.program_label, forced_kind=forced_kind, ts=now_dt)
        if fr.ok:
            flows[book][mint] = flows[book].get(mint, 0.0) + fr.sol_delta
            g.books[book].held.discard(mint)
            cost = float(pos.get("cost_sol") or 0.0)
            if cost > 0 and fr.sol_delta > 0 and (book == g.free_book):
                g.realized_pulse[mint] = config.REALIZED_GAIN * (fr.sol_delta / cost - 1.0)
        row["reason"] += f" {fr.reason}"
        return row

    def finish(self, status: str = "done", metrics: dict | None = None) -> None:
        try:
            self.plast.journal.flush()
        except Exception:
            log.exception("journal flush failed")
        with transaction() as conn:
            if self.opts.learn and self.beat_no > 0:
                self.save_snapshot(conn, "stop", None, note=f"session end ({status})")
            conn.execute("UPDATE runs SET ended_at = now(), status = %s, metrics = %s WHERE run_id = %s",
                         (status, json.dumps(metrics or {}, default=str), self.run_id))
        for g in self.groups:
            if g.exec_worker is not None:
                g.exec_worker.stop()


def _git_sha() -> str | None:
    try:
        import subprocess
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=config.REPO_ROOT, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return None
