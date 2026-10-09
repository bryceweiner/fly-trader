#!/bin/sh
# What every boot (and every self-update) does before the console starts: schema, the shipped models into the data
# volume, the wallet-skill table, the neuron geometry, the rows that admit the fly. Nothing here overwrites a running
# install: copies skip files that exist, and apply-release reuses rows it already has.
set -e
cd /app
until pg_isready -d "$DATABASE_URL" >/dev/null 2>&1; do echo "fly-trader: waiting for postgres"; sleep 2; done
fly-trader init-db
# models/ holds exactly this release's fly and selector: an earlier release's (the updater before signed releases never
# deleted a file) would be copied below, and a pickled .joblib makes apply-release refuse the whole release
if [ -f seed/release_seed.json ]; then
  python - <<'PY'
import json, pathlib
keep = {s["file"] for s in json.loads(pathlib.Path("seed/release_seed.json").read_text())["snapshots"]}
for sub in ("policies", "selectors"):
    for f in pathlib.Path("models", sub).glob("*"):
        if f.is_file() and f.name not in keep:
            f.unlink(); print(f"fly-trader: removed {f} (not in this release)")
PY
fi
for sub in policies selectors connectome; do
  mkdir -p "data/brain/$sub"
  for f in models/$sub/*; do [ -e "$f" ] || continue; [ -e "data/brain/$sub/$(basename "$f")" ] || cp "$f" "data/brain/$sub/"; done
done
# The wallet-skill inputs: the fitted config and the newest table. The table (~300 MB) is too big for git; it comes from
# a verified release (the vault server's /releases), or is downloaded from Hugging Face and used only if its sha256 is
# the one pinned in models/wallet_skill/TABLE.sha256. Nothing unverified reaches the engine.
mkdir -p data/corpus/wallet_skill
for f in models/wallet_skill/*; do
  [ -e "$f" ] || continue
  case "$(basename "$f")" in TABLE|TABLE.sha256) continue;; esac
  [ -e "data/corpus/wallet_skill/$(basename "$f")" ] || cp "$f" data/corpus/wallet_skill/
done
T=$(cat models/wallet_skill/TABLE 2>/dev/null || true)
WANT=$(cat models/wallet_skill/TABLE.sha256 2>/dev/null || true)
if [ -n "$T" ] && [ ! -e "data/corpus/wallet_skill/$T" ]; then
  src=$(ls -d /releases/*/models/wallet_skill/"$T" 2>/dev/null | tail -1 || true)
  if [ -n "$src" ]; then
    cp "$src" "data/corpus/wallet_skill/$T"
  elif [ -n "$WANT" ]; then
    echo "fly-trader: fetching the wallet-skill table $T (~300 MB, once; checked against its pinned sha256)"
    if curl -fsSL --retry 5 -o "data/corpus/wallet_skill/$T.part" "https://huggingface.co/${HF_REPO:-bryceweiner/fly-trader}/resolve/main/models/wallet_skill/$T" \
       && [ "$(sha256sum "data/corpus/wallet_skill/$T.part" | cut -d' ' -f1)" = "$WANT" ]; then
      mv "data/corpus/wallet_skill/$T.part" "data/corpus/wallet_skill/$T"
    else
      rm -f "data/corpus/wallet_skill/$T.part"
    fi
  fi
fi
[ -z "$T" ] || [ -e "data/corpus/wallet_skill/$T" ] || \
  echo "fly-trader: WARNING the wallet-skill table $T is missing or did not match its sha256; the paper selector will not trade and the fly's skill inputs read as zero until it is present (data/corpus/wallet_skill/$T)"
mkdir -p fly_trader/ui/static/brain && cp models/geometry/* fly_trader/ui/static/brain/ 2>/dev/null || true
if [ -f seed/release_seed.json ]; then
  fly-trader apply-release /app --bootstrap        # the shipped models, registered by file hash (fresh ids), idempotently
else
  psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -q -f seed/seed.sql
fi
echo "fly-trader: device $(python -c 'from fly_trader.brain import device; print(device.describe(device.resolve()))' 2>/dev/null || echo unknown)"
