#!/bin/zsh
# Detached Kalshi selector run ($1 days or "all", $2 holdout days) with a memory guard: every 10 s it logs the process's true
# footprint (macOS `footprint`, which counts compressed pages; ps RSS does not) and system free memory, and stops the run if
# free memory falls below 15 % or the footprint exceeds 70 GB — losing a run beats a watchdog reboot (2026-10-06).
cd /Users/bryce/Documents/code/fly-trader
DAYS=${1:-120}; TAG=$(date -u +%Y%m%dT%H%M)
OUT=logs/kalshi_selector_$TAG.out; MEM=logs/kalshi_selector_$TAG.mem
nohup .venv/bin/python -m fly_trader kalshi-train-selector $( [ "$DAYS" = all ] || echo --days $DAYS ) --holdout-days ${2:-21} > $OUT 2>&1 &
PID=$!; echo "pid $PID days $DAYS out $OUT mem $MEM"
( while kill -0 $PID 2>/dev/null; do
    fp=$(/usr/bin/footprint -p $PID 2>/dev/null | awk '/phys_footprint:/ {v=$2; u=$3} END {if (u=="GB") print int(v); else if (u=="MB") print int(v/1024); else print 0}')
    free=$(memory_pressure | tail -1 | grep -oE '[0-9]+' | tail -1); swap=$(sysctl -n vm.swapusage | grep -oE 'used = [0-9.]+M')
    echo "$(date -u +%H:%M:%S) footprint_gb=${fp:-0} free=${free}% $swap"
    if [ "${free:-100}" -lt 15 ] || [ "${fp:-0}" -gt 70 ]; then
      echo "$(date -u +%H:%M:%S) GUARD: stopping the selector (free ${free}%, footprint ${fp} GB)"; kill -TERM $PID; sleep 30; kill -KILL $PID 2>/dev/null
    fi
    sleep 10
  done; echo "$(date -u +%H:%M:%S) exited" ) > $MEM 2>&1 &
