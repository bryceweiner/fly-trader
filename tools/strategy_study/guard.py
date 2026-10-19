"""Guard for research runs on the machine that trades live: every 15 s it stops the run (``os._exit``) when it starts to
cost the live console. Measured on 2026-10-19: the console's minute did not slow while a research run worked, but a
selector search over every row (80 GB of build garbage, compressed) squeezed the console's own pages, and that is
latency in front of live orders. Stops when two consecutive Solana minutes finish more than ``SLOW_S`` after their
boundary, swap grows more than ``SWAP_GROWTH_GB``, the machine's free memory falls below ``MIN_FREE_PCT``, the console's
compressed memory stays ``CONSOLE_CMPRS_GROWTH_GB`` above its start for two readings, or this process passes
``MAX_FOOTPRINT_GB`` (top's MEM: compressed pages included, which RSS hides).
"""
from __future__ import annotations

import json
import logging
import os
import re
import subprocess
import threading
import time
import urllib.parse
import urllib.request

SLOW_S = 45.0                 # the old code's Monday baseline without any research run: 11-36 s at worst
SWAP_GROWTH_GB, MIN_FREE_PCT, CONSOLE_CMPRS_GROWTH_GB, MAX_FOOTPRINT_GB = 3.0, 40, 6.0, 60.0
CONSOLE_URL = "http://127.0.0.1:8501/api/topic/overview.minute?params=" + urllib.parse.quote(json.dumps({"chain": "sol"}))


def _gb(v: str) -> float | None:
    v = v.strip().rstrip("+-")
    try:
        n, unit = float(v[:-1]), v[-1]
    except (ValueError, IndexError):
        return None
    return {"K": n / 2**20, "M": n / 1024, "G": n, "T": n * 1024}.get(unit, n / 2**30)


def top_mem(pid: int | None) -> tuple[float | None, float | None]:
    """(footprint, compressed) of ``pid`` in GB, from top."""
    if pid is None:
        return None, None
    try:
        line = subprocess.run(["top", "-l", "1", "-pid", str(pid), "-stats", "mem,cmprs"], capture_output=True, text=True, timeout=30).stdout.strip().splitlines()[-1].split()
        return _gb(line[0]), _gb(line[1])
    except Exception:
        return None, None


def swap_gb() -> float | None:
    try:
        m = re.search(r"used = ([\d.]+)M", subprocess.run(["sysctl", "vm.swapusage"], capture_output=True, text=True, timeout=10).stdout)
        return float(m.group(1)) / 1024 if m else None
    except Exception:
        return None


def free_pct() -> int | None:
    try:
        m = re.search(r"free percentage:\s*(\d+)%", subprocess.run(["memory_pressure"], capture_output=True, text=True, timeout=30).stdout)
        return int(m.group(1)) if m else None
    except Exception:
        return None


def console_pid() -> int | None:
    try:
        out = subprocess.run(["pgrep", "-f", "fly_trader ui --port 8501"], capture_output=True, text=True, timeout=10).stdout.split()
        return int(out[0]) if out else None
    except Exception:
        return None


def sol_minute() -> tuple[str | None, float | None]:
    try:
        d = json.load(urllib.request.urlopen(CONSOLE_URL, timeout=10))
        return d.get("minute"), d.get("latency_s")
    except Exception:
        return None, None


def start(log_path: str) -> None:
    log = logging.getLogger("guard"); me = os.getpid(); cp = console_pid()
    base_c = top_mem(cp)[1]; base_s = swap_gb()

    def run():
        slow, seen, high = 0, None, 0
        with open(log_path, "a") as f:
            f.write(f"{time.strftime('%H:%M:%S')} start: console pid {cp}, compressed {base_c} GB, swap {base_s} GB\n"); f.flush()
            while True:
                fp, _ = top_mem(me); cc = top_mem(cp)[1]; sw = swap_gb(); fr = free_pct(); mn, lat = sol_minute()
                if mn is not None and mn != seen:
                    seen = mn; slow = slow + 1 if (lat is not None and lat > SLOW_S) else 0
                high = high + 1 if (cc is not None and base_c is not None and cc - base_c > CONSOLE_CMPRS_GROWTH_GB) else 0
                f.write(f"{time.strftime('%H:%M:%S')} free {fr}% swap {sw} GB console-compressed {cc} GB footprint {fp} GB | sol minute {mn} latency {lat}\n"); f.flush()
                why = []
                if slow >= 2:
                    why.append(f"two Solana minutes finished more than {SLOW_S:.0f} s late (last {lat:.1f} s)")
                if sw is not None and base_s is not None and sw - base_s > SWAP_GROWTH_GB:
                    why.append(f"swap grew {sw - base_s:.1f} GB")
                if fr is not None and fr < MIN_FREE_PCT:
                    why.append(f"machine free memory {fr}%")
                if high >= 2:
                    why.append(f"the live console's compressed memory stayed {cc - base_c:.1f} GB above its start")
                if fp is not None and fp > MAX_FOOTPRINT_GB:
                    why.append(f"footprint {fp:.0f} GB")
                if why:
                    log.error("STOPPING: %s -- the live console comes first", "; ".join(why)); f.write("STOP " + "; ".join(why) + "\n"); f.flush()
                    os._exit(3)
                time.sleep(15)
    threading.Thread(target=run, daemon=True).start()
