#!/bin/sh
# From the Mac: copy the relay's transfer day files (RELAY_TRANSFERS_DIR on the relay server) into data/corpus/transfers/.
# Read-only on the server; run it at least every RELAY_TRANSFERS_KEEP_DAYS days (the relay deletes older files).
#
#   RELAY_SSH=<login>@<relay host> sh stream_relay/pull_transfers.sh
set -eu
: "${RELAY_SSH:?set RELAY_SSH=<login>@<relay host>}"
HERE=$(cd "$(dirname "$0")/.." && pwd)
mkdir -p "$HERE/data/corpus/transfers"
rsync -a --include 'transfers-*.jsonl.gz' --exclude '*' "$RELAY_SSH:/var/lib/fly-stream-relay/transfers/" "$HERE/data/corpus/transfers/"
ls -la "$HERE/data/corpus/transfers" | tail -5
