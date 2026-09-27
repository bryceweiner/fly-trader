#!/bin/sh
# First boot: schema, the shipped models into the data volume, the seed rows that admit them. Every boot: the console,
# which autostarts the two trading workers (market feed + trading engine). Nothing here overwrites a running install.
set -e
cd /app

until pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; do
  echo "fly-trader: waiting for postgres"; sleep 2
done

fly-trader init-db

for sub in policies selectors connectome; do
  mkdir -p "data/brain/$sub"
  for f in models/$sub/*; do
    [ -e "$f" ] || continue
    [ -e "data/brain/$sub/$(basename "$f")" ] || cp "$f" "data/brain/$sub/"
  done
done

# The wallet-skill inputs: the fitted config and the newest table, both from the release (verified by the updater's
# hashes). Nothing is downloaded here: an unverified file from the network never reaches the engine.
mkdir -p data/corpus/wallet_skill
for f in models/wallet_skill/*; do
  [ -e "$f" ] || continue
  [ -e "data/corpus/wallet_skill/$(basename "$f")" ] || cp "$f" data/corpus/wallet_skill/
done
T=$(cat models/wallet_skill/TABLE 2>/dev/null || true)
if [ -n "$T" ] && [ ! -e "data/corpus/wallet_skill/$T" ] && [ -e "/releases" ]; then
  src=$(ls -d /releases/*/models/wallet_skill/"$T" 2>/dev/null | tail -1)
  [ -n "$src" ] && cp "$src" "data/corpus/wallet_skill/$T"
fi
[ -z "$T" ] || [ -e "data/corpus/wallet_skill/$T" ] || \
  echo "fly-trader: WARNING the wallet-skill table $T is not in this image or a verified release; the paper selector will not trade and the fly's skill inputs read as zero until it is present"

# The console's 3D brain view: each neuron's position, built once per connectome on the training machine and shipped
# (this image has no FlyWire annotations to build it from). The console serves it from its static directory.
mkdir -p fly_trader/ui/static/brain && cp models/geometry/* fly_trader/ui/static/brain/ 2>/dev/null || true

if [ -f seed/release_seed.json ]; then
  # a signed release (the vault server's updater): register this image's models with fresh ids, idempotently
  fly-trader apply-release /app --bootstrap
else
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f seed/seed.sql
fi

echo "fly-trader: device $(python -c 'from fly_trader.brain import device; print(device.describe(device.resolve()))' 2>/dev/null || echo unknown)"
echo "fly-trader: console at http://localhost:8501"
exec fly-trader ui --port "${PORT:-8501}"
