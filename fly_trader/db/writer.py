"""Persist one beat in one transaction: beats, beat_slots (COPY), brain_activity, decisions, rewards,
wealth_marks, synapse_updates. Returns the beat id and decision ids."""
from __future__ import annotations

import json
import logging
from datetime import datetime

from .schema import ensure_partitions_for

log = logging.getLogger(__name__)

SLOT_COLS = ["beat_id", "slot", "ts", "mint", "pool", "dwell_beats", "features", "feature_mask", "danger", "portfolio",
             "glomeruli", "stim", "kc_active_frac", "m_hat", "rho_app", "rho_av", "dan_rew", "dan_pun", "delta_in",
             "decision_kind", "decision_size", "softmax_p"]


class BeatWriter:
    def __init__(self):
        self._partition_keys: set[str] = set()

    def _ensure(self, conn, ts: datetime) -> None:
        key = ts.strftime("%Y%m%d%H")
        if key not in self._partition_keys:
            ensure_partitions_for(conn, ts)
            self._partition_keys.add(key)

    def write(self, conn, *, run_id: str | None, ts: datetime, beat: dict, slots: list[dict], activity: dict | None,
              decisions: list[dict], rewards: list[dict], marks: list[dict], synapse: dict | None) -> tuple[int, list[int]]:
        self._ensure(conn, ts)
        row = conn.execute(
            """INSERT INTO beats (run_id, ts, beat_no, sim_ts, tape_last_id, n_slots_active, n_held_live, n_held_paper,
                 ticks, gpu_ms, total_ms, feed_age_ms, notes) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
            (run_id, ts, beat.get("beat_no"), beat.get("sim_ts"), beat.get("tape_last_id"), beat.get("n_slots_active"),
             beat.get("n_held_live"), beat.get("n_held_paper"), beat.get("ticks"), beat.get("gpu_ms"), beat.get("total_ms"),
             beat.get("feed_age_ms"), json.dumps(beat.get("notes") or {}, default=str)),
        ).fetchone()
        beat_id = int(row["id"])
        if slots:
            with conn.cursor() as cur:
                with cur.copy(f"COPY beat_slots ({', '.join(SLOT_COLS)}) FROM STDIN") as copy:
                    for s in slots:
                        copy.write_row((beat_id, s["slot"], ts, s.get("mint"), s.get("pool"), s.get("dwell_beats"),
                                        s.get("features"), s.get("feature_mask"), s.get("danger"), s.get("portfolio"),
                                        s.get("glomeruli"), s.get("stim"), s.get("kc_active_frac"), s.get("m_hat"),
                                        s.get("rho_app"), s.get("rho_av"), s.get("dan_rew"), s.get("dan_pun"), s.get("delta_in"),
                                        s.get("decision_kind"), s.get("decision_size"), s.get("softmax_p")))
        if activity:
            conn.execute(
                """INSERT INTO brain_activity (beat_id, group_rates, kc_sparsity, mbon_app_rate, mbon_av_rate, dan_rew_rate,
                     dan_pun_rate, total_spikes, v_mean, v_max, nan_flag) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (beat_id, activity.get("group_rates"), activity.get("kc_sparsity"), activity.get("mbon_app_rate"),
                 activity.get("mbon_av_rate"), activity.get("dan_rew_rate"), activity.get("dan_pun_rate"),
                 activity.get("total_spikes"), activity.get("v_mean"), activity.get("v_max"), activity.get("nan_flag", False)),
            )
        dids = []
        for d in decisions:
            r = conn.execute(
                """INSERT INTO decisions (beat_id, run_id, ts, slot, mint, pool, kind, m_hat, size_sol, forced, rail, reason,
                     softmax_p, book_targets, detail) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) RETURNING id""",
                (beat_id, run_id, ts, d.get("slot"), d.get("mint"), d.get("pool"), d["kind"], d.get("m_hat"), d.get("size_sol"),
                 d.get("forced", False), d.get("rail"), d.get("reason"), d.get("softmax_p"), d.get("book_targets"),
                 json.dumps(d.get("detail") or {}, default=str)),
            ).fetchone()
            dids.append(int(r["id"]))
        if rewards:
            conn.cursor().executemany(
                """INSERT INTO rewards (beat_id, slot, book, mint, r_slot, r_global, r_tilde, m_prev, m_now, delta, source)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT DO NOTHING""",
                [(beat_id, r["slot"] if r.get("slot") is not None else 1000 + k, r.get("book"), r.get("mint"), r.get("r_slot"), r.get("r_global"), r.get("r_tilde"),
                  r.get("m_prev"), r.get("m_now"), r.get("delta"), r.get("source")) for k, r in enumerate(rewards)],
            )
        for m in marks:
            conn.execute(
                """INSERT INTO wealth_marks (beat_id, book, ts, sol_free, positions_value, exit_cost, wealth, peak, drawdown, exposure, n_open)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (beat_id, m["book"], ts, m.get("sol_free"), m.get("positions_value"), m.get("exit_cost"), m.get("wealth"),
                 m.get("peak"), m.get("drawdown"), m.get("exposure"), m.get("n_open")),
            )
        if synapse:
            conn.execute(
                """INSERT INTO synapse_updates (beat_id, ts, eta, gamma, n_slots, reward_source, sum_abs, max_abs, frob, frob_capped,
                     n_pos, n_neg, n_clipped, w_mean, w_min, w_max, delta_path, delta_sha256)
                   VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)""",
                (beat_id, ts, synapse.get("eta"), synapse.get("gamma"), synapse.get("n_slots"), synapse.get("reward_source"),
                 synapse.get("sum_abs"), synapse.get("max_abs"), synapse.get("frob"), synapse.get("frob_capped"),
                 synapse.get("n_pos"), synapse.get("n_neg"), synapse.get("n_clipped"), synapse.get("w_mean"),
                 synapse.get("w_min"), synapse.get("w_max"), synapse.get("delta_path"), synapse.get("delta_sha256")),
            )
        return beat_id, dids
