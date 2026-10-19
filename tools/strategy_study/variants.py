"""Strategy study, step 3: variants of the capitulation strategy on step 2's out-of-sample scores (logs/strategy_study),
one chain at a time -- every choice made on the selection half, judged on the evaluation half and the holdout:

1. the line chosen for each fitted hold;  2. the same trades sold after every hold from 10 to 480 minutes;
3. floors and caps on liquidity, volume, activity, age, market volume, exit cost and wash share;
4. hour of day;  5. the recent days against the spread of every 3-, 5- and 7-day window;
6. the compared selector (step 2's --compare) on the holdout days after its training, beside the fresh fit.

    .venv/bin/python tools/strategy_study/variants.py [--chain sol|rh]

Returns are the trainer's labels (net of costs, entry at the next minute's open). Robinhood Chain labels of July to
September 2026 hold impossible returns (up to +226,678 %): read rh results with that in mind.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from fly_trader.train.decisions import taken_idx  # noqa: E402

OUT = REPO / "logs" / "strategy_study"
FLOORS = ("log_liquidity_sol", "logvol_15m", "logvol_1h", "logn_1h", "traders_15m", "traders_1h", "log_age_h", "mkt_vol_1h", "exit_cost_0p1", "rvol_1h", "wash_share_15m")


def eday(iso: str) -> int:
    return int(datetime.fromisoformat(iso).replace(tzinfo=timezone.utc).timestamp() // 86400)


def stats(r: np.ndarray) -> str:
    r = r[np.isfinite(r)]
    if not len(r):
        return "n=0"
    pf = r[r > 0].sum() / -r[r < 0].sum() if (r < 0).any() else float("inf")
    return f"n={len(r):5d} mean {100 * r.mean():+6.2f}% win {100 * (r > 0).mean():3.0f}% PF {pf:4.2f} total {r.sum():+7.1f}"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0]); ap.add_argument("--chain", default="sol")
    chain = ap.parse_args().chain
    S = json.load(open(OUT / "summary.json")); C = dict(np.load(OUT / "cap.npz", allow_pickle=False))
    cap = next((f for f in S["fits"] if f["name"] == "capitulation" and f.get("hold_min")), None)
    day = (C["ts"] // 86400).astype(np.int64)
    s0, s1 = map(eday, S["selection_days"]); e0, e1 = map(eday, S["evaluation_days"]); h0, h1 = eday(S["holdout_days"][0]), eday(S["holdout_days"][-1])
    SEL, EV, HO = (day >= s0) & (day <= s1), (day >= e0) & (day <= e1), (day >= h0) & (day <= h1)
    print(f"{chain}: selection {S['selection_days']}, evaluation {S['evaluation_days']}, holdout {S['holdout_days'][0]}..{S['holdout_days'][-1]}")
    print("fresh capitulation fit:", {k: cap[k] for k in ("hold_min", "line", "thr")} if cap else "none kept")

    def book(pick, H):
        idx = np.flatnonzero(pick & np.isfinite(C[f"y_{H}"]))
        if not len(idx):
            return np.array([])
        return C[f"y_{H}"][taken_idx(C["ts"], C["mint"], H * 60.0, idx)]

    trig = np.ones(len(day), bool)
    for v, t in ((cap or {}).get("thr") or {}).items():
        if t is not None:
            x = -C["f_ret_1m"] if v == "neg_ret_1m" else C.get("f_" + v)
            if x is not None:
                trig &= x >= t
    base = trig & (C["chain"] == chain)

    print("\n1) THE LINE CHOSEN FOR EACH FITTED HOLD (most net profit on the selection half, PF >= 1.3, >= 100 trades)")
    for H in sorted(int(k[2:]) for k in C if k.startswith("s_")):
        s = C[f"s_{H}"]; best = None
        for q in (0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.975, 0.99, 0.995, 0.999):
            ok = base & SEL & np.isfinite(s)
            if not ok.any():
                break
            line = np.nanquantile(s[ok], q); r = book(base & SEL & (s >= line), H)
            if len(r) >= 100 and r[r > 0].sum() / max(1e-9, -r[r < 0].sum()) >= 1.3 and (best is None or r.sum() > best[1]):
                best = (line, r.sum())
        if best is None:
            print(f"  {H:4d} min: no line clears the bars on the selection half"); continue
        p = base & (s >= best[0])
        print(f"  {H:4d} min | evaluation {stats(book(p & EV, H))} | holdout {stats(book(p & HO, H))}")
    if not cap:
        return
    H0, L0 = cap["hold_min"], cap["line"]; P = base & (C[f"s_{H0}"] >= L0)
    print(f"\n2) THE SAME TRADES (hold-{H0} picks) SOLD AFTER EACH HOLD")
    for Hx in sorted(int(k[2:]) for k in C if k.startswith("y_")):
        print(f"  sold at {Hx:4d} min | evaluation {stats(book(P & EV, Hx))} | holdout {stats(book(P & HO, Hx))}")
    print(f"\n3) FLOORS AND CAPS (the best cut of each, chosen on the selection half)")
    print(f"  as fitted         | evaluation {stats(book(P & EV, H0))} | holdout {stats(book(P & HO, H0))}")
    for f in FLOORS:
        x = C.get("f_" + f)
        if x is None:
            continue
        best = None
        for side in (1, -1):
            for q in (0.1, 0.2, 0.3, 0.4, 0.5):
                cut = np.nanquantile(x[P & SEL], q if side > 0 else 1 - q); keep = (x >= cut) if side > 0 else (x <= cut)
                tot = np.nansum(book(P & SEL & keep, H0))
                if best is None or tot > best[0]:
                    best = (tot, side, q, cut, keep)
        _, side, q, cut, keep = best
        print(f"  {f:17s} {'>=' if side > 0 else '<='} {cut:8.3f} (drops {'bottom' if side > 0 else 'top'} {int(q * 100)}%) | "
              f"evaluation {stats(book(P & EV & keep, H0))} | holdout {stats(book(P & HO & keep, H0))}")
    print("\n4) HOUR OF DAY (UTC): evaluation + holdout, mean per trade (trades)")
    idx = np.flatnonzero(P & (EV | HO) & np.isfinite(C[f"y_{H0}"])); t = taken_idx(C["ts"], C["mint"], H0 * 60.0, idx)
    r = C[f"y_{H0}"][t]; hr = ((C["ts"][t] // 3600) % 24).astype(int)
    print("  " + " ".join(f"{h:02d}:{100 * r[hr == h].mean():+5.1f}%({(hr == h).sum()})" for h in range(24) if (hr == h).any()))
    print("\n5) THE RECENT DAYS against every window of the fresh book (out of sample)")
    B = dict(np.load(OUT / "book.npz", allow_pickle=False)); ok = (B["chain"] == chain) & np.isfinite(B["ret"])
    d = (B["ts"][ok] // 86400).astype(int); rr = B["ret"][ok]; ud = np.arange(d.min(), d.max() + 1)
    fmt = lambda e: datetime.fromtimestamp(e * 86400, timezone.utc).strftime("%m-%d")   # noqa: E731
    print(f"  {len(rr)} trades {fmt(ud[0])}..{fmt(ud[-1])}: mean {100 * rr.mean():+.2f}% per trade, median {100 * np.median(rr):+.2f}%")
    for w in (3, 5, 7):
        vals = [(ud[i + w - 1], rr[m].mean(), int(m.sum()), rr[m].sum()) for i in range(len(ud) - w + 1)
                for m in [(d >= ud[i]) & (d <= ud[i + w - 1])] if m.sum() >= 3]
        last = vals[-1]; means = np.array([v[1] for v in vals]); tots = np.array([v[3] for v in vals])
        print(f"  {w}-day windows ({len(vals)}): median {100 * np.median(means):+.2f}%; the last ({fmt(last[0])}, {last[2]} trades) {100 * last[1]:+.2f}% = "
              f"percentile {100 * np.mean(means <= last[1]):.0f}; losing windows {100 * np.mean(tots < 0):.0f}%")
    print("  per day (last 16): " + " ".join(f"{fmt(e)}:{100 * rr[d == e].mean():+.0f}%({(d == e).sum()})" if (d == e).any() else f"{fmt(e)}:-" for e in ud[-16:]))
    hc = OUT / "holdout_compare.json"
    if hc.exists():
        h = json.load(open(hc))
        print(f"\n6) SELECTOR #{h['snapshot']} ON THE HOLDOUT DAYS AFTER ITS TRAINING ({h['trained_through']}), its own decisions: {h.get(chain)}")
        t0 = eday(h["trained_through"]) + 1
        print(f"  fresh capitulation, same days: {stats(book(P & HO & (day >= t0), H0))}")


if __name__ == "__main__":
    main()
