#!/usr/bin/env bash
# Builds the site and assembles <repo>/build/gandi/, the tree uploaded to Gandi Simple Hosting:
#   wsgi.py  site-headers.json  relay/ (without tests)  site/ (the Vite build incl. public files and local mp4s)
# Env: VITE_NETWORK, VITE_REOWN_PROJECT_ID, VITE_VAULT_ADDRESS, VITE_TIMELOCK_ADDRESS, VITE_FLY_ADDRESS, VITE_RELAY_BASE
set -euo pipefail

WEB="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO="$(cd "$WEB/.." && pwd)"
OUT="$REPO/build/gandi"

cd "$WEB"
if [[ -z "${VITE_REOWN_PROJECT_ID:-}" ]] && ! grep -qs '^VITE_REOWN_PROJECT_ID=.' .env .env.production .env.local 2>/dev/null; then
  echo "warning: VITE_REOWN_PROJECT_ID is not set; wallet connection will show as not configured" >&2
fi
npm ci
npm run build

rm -rf "$OUT"
mkdir -p "$OUT"
cp -R dist "$OUT/site"
for mp4 in public/assets/*.mp4; do
  [[ -e "$mp4" && ! -e "$OUT/site/assets/$(basename "$mp4")" ]] && cp "$mp4" "$OUT/site/assets/"
done
cp wsgi.py site-headers.json "$OUT/"
if [[ -d relay ]]; then
  mkdir -p "$OUT/relay"
  (cd relay && find . -type f ! -path './tests/*' ! -path '*/__pycache__/*' ! -name '*.pyc' -print0 | while IFS= read -r -d '' f; do
    mkdir -p "$OUT/relay/$(dirname "$f")"
    cp "$f" "$OUT/relay/$f"
  done)
else
  echo "warning: web/relay/ not found; the bundle has no /api" >&2
fi

echo
echo "== $OUT"
(cd "$OUT" && find . -path ./site/static -prune -o -print | sed 's|^\./||' | sort)
echo "site/static: $(find "$OUT/site/static" -type f | wc -l | tr -d ' ') hashed files, $(du -sh "$OUT/site/static" | cut -f1)"
echo "total: $(du -sh "$OUT" | cut -f1)"
cat <<MSG

Upload: copy the CONTENTS of build/gandi/ into the Gandi Simple Hosting instance's app directory
(where wsgi.py lives today) over the instance's SFTP or git access, replacing site/ entirely so stale
hashed bundles go away. Put relay.json next to wsgi.py (never inside site/), then restart the instance.
Check: https://fly-trader.app/ , /trading.html , /vault.html , /api/stats
MSG
