#!/bin/sh
# What every boot (and every self-update) does before the console starts: schema, the shipped models into the data
# volume, the wallet-skill table, the neuron geometry, the seed rows that admit the fly. Nothing here overwrites a
# running install: copies skip files that exist, the seed is ON CONFLICT DO NOTHING.
set -e
cd /app
until pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; do echo "fly-trader: waiting for postgres"; sleep 2; done
fly-trader init-db
for sub in policies selectors connectome; do
  mkdir -p "data/brain/$sub"
  for f in models/$sub/*; do [ -e "$f" ] || continue; [ -e "data/brain/$sub/$(basename "$f")" ] || cp "$f" "data/brain/$sub/"; done
done
mkdir -p data/corpus/wallet_skill
for f in models/wallet_skill/*; do [ -e "$f" ] || continue; [ -e "data/corpus/wallet_skill/$(basename "$f")" ] || cp "$f" data/corpus/wallet_skill/; done
T=$(cat models/wallet_skill/TABLE 2>/dev/null || true)
if [ -n "$T" ] && [ ! -e "data/corpus/wallet_skill/$T" ]; then
  echo "fly-trader: fetching the wallet-skill table $T (~300 MB, once)"
  curl -fsSL --retry 5 -o "data/corpus/wallet_skill/$T.part" "https://huggingface.co/${HF_REPO:-bryceweiner/fly-trader}/resolve/main/models/wallet_skill/$T" \
    && mv "data/corpus/wallet_skill/$T.part" "data/corpus/wallet_skill/$T" \
    || echo "fly-trader: WARNING could not fetch the wallet-skill table; the paper selector will not trade and the fly's skill inputs read as zero until it is present"
fi
mkdir -p fly_trader/ui/static/brain && cp models/geometry/* fly_trader/ui/static/brain/ 2>/dev/null || true
psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f seed/seed.sql
echo "fly-trader: device $(python -c 'from fly_trader.brain import device; print(device.describe(device.resolve()))' 2>/dev/null || echo unknown)"
