# fly-trader.app

Static site (Vite multi-page, vanilla TypeScript) plus the relay API under `relay/`, served together by
`wsgi.py` on Gandi Simple Hosting. Interfaces the site builds against: `docs/vault/SPEC.md`.

| Page | What it does |
|---|---|
| `index.html` | Project page. Loads only `src/site.ts` (~8 KB gzip). |
| `trading.html` | $FLY market strip + candles (GeckoTerminal), buy/sell with ETH or USDG through KyberSwap, balances. |
| `vault.html` | The fly's live stats, NAV chart, settlements/positions/trades/flows, your position (lock, request, cancel, withdraw), SOL claims. |
| `terms.html`, `privacy.html` | Legal pages. |

Header, footer and common `<head>` live in `partials/` and are included with `<!-- @include name -->`
(see `vite.config.ts`). Wallet support (Reown AppKit, EVM + Solana) is a dynamic import on the Trading and Vault
pages only.

## Develop

```sh
npm install
npm run dev          # http://localhost:5173 — /api is served from dev/fixtures, no backend needed
npm test             # vitest: claim-text vectors (tests/vectors/claim_v1.json), formatting, Kyber allowlist
npm run build        # tsc --noEmit, then vite build -> dist/
npm run preview      # serve dist/
```

`node dev/make-fixtures.mjs` regenerates the fixtures (deterministic). The dev middleware shifts their timestamps
to the present and walks a submitted claim through `received → verified → sending → paid` in ~10 s. It does not
verify signatures; the fly does.

To use a real relay in dev instead: `VITE_RELAY_PROXY=https://staging.example npm run dev` (proxies `/api`).

## Environment (build time, `.env` / `.env.production` or the shell)

| Variable | Default | Meaning |
|---|---|---|
| `VITE_NETWORK` | `mainnet` | `mainnet` (Robinhood Chain 4663 + Solana mainnet) or `testnet` (46630 + Solana devnet; swaps off, chart from fixtures). |
| `VITE_REOWN_PROJECT_ID` | — | Reown (WalletConnect) project id. Without it the pages work read-only and say wallet connection is not configured. |
| `VITE_VAULT_ADDRESS` | — | FlyVault proxy. Empty: the Vault page shows "not deployed yet" but still renders the fly's stats. |
| `VITE_TIMELOCK_ADDRESS` | — | TimelockController, linked from the Vault page. |
| `VITE_FLY_ADDRESS` | mainnet $FLY | Token override; required on testnet (MockFLY). |
| `VITE_RELAY_BASE` | `/api` | Relay base URL. |
| `VITE_RELAY_PROXY` | — | Dev only: proxy `/api` to this origin instead of the fixtures. |

The claim domain check: the Vault page refuses a claim challenge whose `domain` is not the host it is served
from, so a staging relay must be configured with the staging host (e.g. `localhost:5173`).

## Deploy (Gandi)

```sh
VITE_REOWN_PROJECT_ID=… VITE_VAULT_ADDRESS=0x… VITE_TIMELOCK_ADDRESS=0x… ./scripts/bundle_gandi.sh
```

This runs `npm ci && npm run build` and assembles `../build/gandi/`:

```
wsgi.py              static server + /api dispatch (owned by the relay work)
site-headers.json    headers wsgi.py adds: {"html": {"Content-Security-Policy": …}, "all": {}}
relay/               web/relay without tests/
site/                dist/ — pages, /static/ (hashed, cache forever), /assets/ (og.png, hero videos, …)
```

Upload the contents to the instance's app directory, replacing `site/` entirely, put `relay.json` next to
`wsgi.py` (never inside `site/`), restart the instance. The hero videos (`public/assets/*.mp4`) are not in git;
they are copied from the local checkout when present.

Operator checklist:
- Reown dashboard: create the project, allow the domain `fly-trader.app` (and any staging origin), put the id in
  `VITE_REOWN_PROJECT_ID`.
- After deploying the vault: set `VITE_VAULT_ADDRESS` and `VITE_TIMELOCK_ADDRESS` and rebuild.
- If a page starts calling a new host (a new AppKit version, say), add it to `site-headers.json` or the CSP will
  block it. The CSP deliberately leaves out `pulse.walletconnect.org` (WalletConnect telemetry).

## Layout

```
index.html trading.html vault.html terms.html privacy.html
partials/        head.html header.html footer.html
public/          copied as-is: favicon.svg robots.txt sitemap.xml assets/
src/site.ts      shared behaviour (links, CA copy, burger menu, hero headline/video, year)
src/config.ts    network constants (SPEC §5) + env overrides; no viem, it ships on every page
src/pages/       trading.ts vault.ts
src/lib/         api claim vault kyber chart chains appkit evm format ui (+ *.test.ts)
dev/             mock-api.ts (dev /api), make-fixtures.mjs, fixtures/*.json
scripts/         bundle_gandi.sh
```
