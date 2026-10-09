#!/bin/zsh
# Run the Kalshi selector (tools/kalshi_selector_run.sh $1 $2) only when the shared Mac has room: free memory >= $3 % (default
# 45) for ten consecutive minutes. If the run ends without saving a new snapshot (the memory guard stopped it because another
# job filled the machine), wait for room again and retry, up to $4 attempts (default 6); each retry resumes from the
# walk-forward cache.
cd /Users/bryce/Documents/code/fly-trader
DAYS=${1:-all}; HOLD=${2:-90}; NEED=${3:-45}; TRIES=${4:-6}
last() { .venv/bin/python -c "
from fly_trader.db.connection import transaction
with transaction() as c:
    print(c.execute(\"SELECT COALESCE(max(id), 0) AS m FROM brain_snapshots WHERE kind = 'kalshi_selector'\").fetchone()['m'])"; }
START=$(last); echo "$(date -u +%H:%M) newest kalshi selector snapshot: $START"
for attempt in $(seq 1 $TRIES); do
  ok=0
  while [ $ok -lt 10 ]; do
    free=$(memory_pressure | tail -1 | grep -oE '[0-9]+' | tail -1)
    if [ "${free:-0}" -ge $NEED ]; then ok=$((ok + 1)); else ok=0; fi
    sleep 60
  done
  echo "$(date -u +%H:%M) attempt $attempt: ${free}% free for ten minutes, starting the selector"
  line=$(tools/kalshi_selector_run.sh $DAYS $HOLD); echo "$line"; PID=$(echo "$line" | awk '{print $2}'); MEM=$(echo "$line" | awk '{print $NF}')
  while kill -0 $PID 2>/dev/null; do sleep 60; done
  sleep 15; grep GUARD $MEM | head -1
  NOW=$(last)
  if [ "$NOW" -gt "$START" ]; then echo "$(date -u +%H:%M) selector snapshot $NOW saved"; exit 0; fi
  echo "$(date -u +%H:%M) attempt $attempt ended without a snapshot; waiting for room again"
done
echo "$(date -u +%H:%M) gave up after $TRIES attempts"; exit 1
