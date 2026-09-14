"""Pretraining metrics counted from rows of a run_id (never from status flags)."""
from __future__ import annotations

from ..db.connection import connect


def run_metrics(run_id: str, book: str) -> dict:
    with connect() as conn:
        w = conn.execute(
            """SELECT (SELECT wealth FROM wealth_marks wm JOIN beats b ON b.id = wm.beat_id WHERE b.run_id = %s AND wm.book = %s ORDER BY b.id LIMIT 1) AS w0,
                      (SELECT wealth FROM wealth_marks wm JOIN beats b ON b.id = wm.beat_id WHERE b.run_id = %s AND wm.book = %s ORDER BY b.id DESC LIMIT 1) AS w1,
                      (SELECT min(wealth) FROM wealth_marks wm JOIN beats b ON b.id = wm.beat_id WHERE b.run_id = %s AND wm.book = %s) AS wmin,
                      (SELECT max(wealth) FROM wealth_marks wm JOIN beats b ON b.id = wm.beat_id WHERE b.run_id = %s AND wm.book = %s) AS wmax""",
            (run_id, book, run_id, book, run_id, book, run_id, book)).fetchone()
        b = conn.execute("SELECT count(*) AS n, min(sim_ts) AS t0, max(sim_ts) AS t1, avg(gpu_ms) AS gpu_ms, avg(total_ms) AS total_ms, "
                         "sum(CASE WHEN (notes->>'brain_ok')='false' THEN 1 ELSE 0 END) AS brain_bad FROM beats WHERE run_id = %s", (run_id,)).fetchone()
        d = conn.execute("SELECT kind, count(*) AS n FROM decisions WHERE run_id = %s GROUP BY 1", (run_id,)).fetchall()
        a = conn.execute("""SELECT avg(kc_sparsity) AS kc, avg(mbon_app_rate) AS app, avg(mbon_av_rate) AS av, avg(dan_rew_rate) AS dr,
                                   avg(dan_pun_rate) AS dp, sum(CASE WHEN nan_flag THEN 1 ELSE 0 END) AS nan_beats
                            FROM brain_activity ba JOIN beats b ON b.id = ba.beat_id WHERE b.run_id = %s""", (run_id,)).fetchone()
        s = conn.execute("""SELECT avg(CASE WHEN frob_capped THEN 1.0 ELSE 0.0 END) AS capped, avg(frob) AS frob, count(*) AS n
                            FROM synapse_updates su JOIN beats b ON b.id = su.beat_id WHERE b.run_id = %s""", (run_id,)).fetchone()
        m = conn.execute("""SELECT stddev(m_hat) AS m_std, avg(m_hat) AS m_mean FROM beat_slots bs JOIN beats b ON b.id = bs.beat_id
                            WHERE b.run_id = %s AND bs.mint IS NOT NULL""", (run_id,)).fetchone()
        p = conn.execute("""SELECT count(*) AS n, COALESCE(sum(realized_sol),0) AS realized, avg(realized_sol) AS avg_realized,
                                   sum(CASE WHEN realized_sol > 0 THEN 1 ELSE 0 END) AS winners
                            FROM positions WHERE book = %s AND status = 'closed'""", (book,)).fetchone()
    days = ((b["t1"] - b["t0"]).total_seconds() / 86400.0) if (b and b["t0"] and b["t1"]) else 0.0
    import math
    return {
        "beats": int(b["n"] or 0), "days": days, "gpu_ms": float(b["gpu_ms"] or 0), "total_ms": float(b["total_ms"] or 0),
        "brain_bad_beats": int(b["brain_bad"] or 0),
        "log_wealth": (math.log(w["w1"] / w["w0"]) if (w and w["w0"] and w["w1"] and w["w0"] > 0 and w["w1"] > 0) else None),
        "wealth_start": w["w0"], "wealth_end": w["w1"], "max_drawdown": (1 - w["wmin"] / w["wmax"]) if (w and w["wmax"]) else None,
        "decisions": {r["kind"]: int(r["n"]) for r in d},
        "entries_per_day": (sum(int(r["n"]) for r in d if r["kind"] == "enter") / days) if days else None,
        "kc_sparsity": a["kc"], "mbon_app": a["app"], "mbon_av": a["av"], "dan_rew": a["dr"], "dan_pun": a["dp"], "nan_beats": int(a["nan_beats"] or 0),
        "frob_capped_frac": s["capped"], "frob_mean": s["frob"], "synapse_updates": int(s["n"] or 0),
        "m_hat_std": m["m_std"], "m_hat_mean": m["m_mean"],
        "closed_positions": int(p["n"] or 0), "realized_sol": float(p["realized"] or 0), "winners": int(p["winners"] or 0),
    }
