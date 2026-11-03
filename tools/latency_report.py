"""Per-minute timelines from a tools/latency_probe.py run, in seconds after each minute's boundary B (the minute ending
at B): the stream writes the minute -> Solana's live phase starts (its aggregate query) -> RH's live phase starts (its
aggregate) -> Solana's paper phase starts (the fly's scored rows stored) -> the fly's status (its paper minute done) ->
the selector's status; live orders; and the SQL the console was seen executing longest.

    .venv/bin/python tools/latency_report.py [logs/latency/probe_<UTC stamp>.jsonl]   (default: the newest probe)

Phase starts are recognised by their queries (agent/minute_engine.py ``_aggregate``, agent/fly_session.py
``_store_scored``): a query shorter than the 0.25 s sampling step can go unseen, so those columns have gaps.
"""
from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
PAT = {"sol_agg": re.compile(r"buy_sol, sell_sol, n_buys"), "rh_agg": re.compile(r"buy_eth, sell_eth, n_buys"),
       "fly_scored": re.compile(r"INSERT INTO fly_scored")}


def sig(q: str) -> str:
    return re.sub(r"\$\d+|'[^']*'|\b\d+\b", "?", q or "")[:90]


def iso(e: float) -> str:
    return datetime.fromtimestamp(e, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else max((REPO / "logs" / "latency").glob("probe_*.jsonl"))
    S = [json.loads(line) for line in open(path)]
    mins = list(range(int(S[0]["t"] // 60) * 60 + 60, int(S[-1]["t"] // 60) * 60 - 60 + 1, 60))
    first = defaultdict(dict); status = {}; orders = {}; cpu = defaultdict(list)
    sql_time = defaultdict(float); sql_max = defaultdict(float)
    for s in S:
        t = s["t"]
        for a in s["act"]:
            if a["state"] != "active" or not a.get("qs"):
                continue
            for k, p in PAT.items():
                if p.search(a["q"] or ""):
                    b = int(a["qs"] // 60) * 60
                    first[b][k] = min(first[b].get(k, 1e18), a["qs"])
            g = sig(a["q"]); sql_time[g] += 0.25; sql_max[g] = max(sql_max[g], t - a["qs"])
        for key, v in s["st"].items():
            if v.get("minute") and v.get("at"):
                status[(key, v["minute"])] = min(status.get((key, v["minute"]), 1e18), v["at"])
            if key in ("pumpstream_status", "rh_stream_status") and v.get("ft"):
                status[(key + ":ft", v["ft"])] = min(status.get((key + ":ft", v["ft"]), 1e18), v["at"] or t)
        for o in s["orders"]:
            orders[o["id"]] = o
        if s["cpu"] is not None:
            cpu[int(t // 60) * 60].append(s["cpu"])
    rows = []
    print(f"{path.name}\n{'minute':6s} {'stream':>6s} {'sol live':>8s} {'rh live':>7s} {'paper':>6s} {'fly done':>8s} {'sel done':>8s} {'rh sel':>6s} {'cpu%':>5s}  live orders")
    for B in mins:
        ft = status.get(("pumpstream_status:ft", iso(B - 60)))
        sol, rh, pap = first[B].get("sol_agg"), first[B].get("rh_agg"), first[B].get("fly_scored")
        fly, sel, rsel = status.get(("fly_status", iso(B))), status.get(("selector_status", iso(B))), status.get(("selector_status_rh", iso(B)))
        od = [f"{o['side']} +{o['t'] - B:.1f}s {o['status']}" for o in orders.values() if B <= o["t"] < B + 60]
        f = lambda x: f"{x - B:6.1f}" if x and x < 1e17 else "     -"         # noqa: E731
        rows.append({k: (v - B) if v and v < 1e17 else None for k, v in (("stream", ft), ("sol", sol), ("rh", rh), ("paper", pap), ("fly", fly), ("sel", sel))})
        print(f"{datetime.fromtimestamp(B, timezone.utc):%H:%M}  {f(ft)} {f(sol):>8s} {f(rh):>7s} {f(pap):>6s} {f(fly):>8s} {f(sel):>8s} {f(rsel):>6s} "
              f"{np.mean(cpu[B]) if cpu[B] else float('nan'):5.0f}  {'; '.join(od)}")
    print("\nseconds after the boundary (median / p90 / max):")
    for k, name in (("stream", "stream has written the minute"), ("sol", "Solana live phase starts"), ("rh", "RH live phase starts"),
                    ("paper", "Solana paper phase starts"), ("fly", "fly's paper minute done"), ("sel", "selector done (the console's latency)")):
        v = np.array([r[k] for r in rows if r[k] is not None])
        print(f"  {name:40s} " + (f"{np.median(v):5.1f} / {np.quantile(v, 0.9):5.1f} / {v.max():5.1f}  (n={len(v)})" if len(v) else "-"))
    v = np.array([r["sel"] - r["stream"] for r in rows if r["sel"] is not None and r["stream"] is not None])
    if len(v):
        print(f"  {'publish -> selector done':40s} {np.median(v):5.1f} / {np.quantile(v, 0.9):5.1f} / {v.max():5.1f}  (n={len(v)})")
    print("\nSQL the console was seen executing longest (sampled every 0.25 s):")
    for g, tt in sorted(sql_time.items(), key=lambda x: -x[1])[:12]:
        print(f"  {tt:7.1f}s seen, longest {sql_max[g]:5.1f}s | {g}")


if __name__ == "__main__":
    main()
