#!/bin/zsh
# After a Kalshi selector run newer than snapshot $1 (default 156): if that selector is deployable, bootstrap the visual fly
# from it on the full history, with the same memory guard as tools/kalshi_selector_run.sh (true footprint; stop below 15 %
# free or above 70 GB).
cd /Users/bryce/Documents/code/fly-trader
PY=.venv/bin/python; AFTER=${1:-156}
until $PY -c "
from fly_trader.db.connection import transaction
with transaction() as c:
    r = c.execute(\"SELECT max(id) AS m FROM brain_snapshots WHERE kind = 'kalshi_selector'\").fetchone()
raise SystemExit(0 if r['m'] and r['m'] > $AFTER else 1)" 2>/dev/null && ! pgrep -f "kalshi-train-selector" >/dev/null; do sleep 300; done
echo "$(date -u +%H:%M) new selector saved"
$PY -c "
import json
from fly_trader.db.connection import transaction
from fly_trader.kalshi import selector as S
with transaction() as c:
    r = c.execute(\"SELECT id, note FROM brain_snapshots WHERE kind = 'kalshi_selector' ORDER BY id DESC LIMIT 1\").fetchone()
m = json.loads(r['note']); print('selector', r['id'], 'deployable' if S.is_deployable(m) else 'NOT deployable', '-', m.get('deploy_reason'))
raise SystemExit(0 if S.is_deployable(m) else 3)" || { echo "selector not deployable: the fly is not trained from it"; exit 0; }
TAG=$(date -u +%Y%m%dT%H%M); OUT=logs/kalshi_fly_$TAG.out; MEM=logs/kalshi_fly_$TAG.mem
nohup $PY -m fly_trader kalshi-train-fly > $OUT 2>&1 &
PID=$!; echo "fly pid $PID out $OUT"
while kill -0 $PID 2>/dev/null; do
  fp=$(/usr/bin/footprint -p $PID 2>/dev/null | awk '/phys_footprint:/ {v=$2; u=$3} END {if (u=="GB") print int(v); else if (u=="MB") print int(v/1024); else print 0}')
  free=$(memory_pressure | tail -1 | grep -oE '[0-9]+' | tail -1)
  echo "$(date -u +%H:%M:%S) footprint_gb=${fp:-0} free=${free}%" >> $MEM
  if [ "${free:-100}" -lt 15 ] || [ "${fp:-0}" -gt 70 ]; then echo "GUARD: stopping the fly (free ${free}%, footprint ${fp} GB)" >> $MEM; kill -TERM $PID; sleep 30; kill -KILL $PID 2>/dev/null; fi
  sleep 15
done
echo "$(date -u +%H:%M) fly exited"; tail -2 $MEM
grep -E "teacher|bootstrap for|epoch|gates|Traceback|Error" $OUT | cut -c100-420 | tail -16
