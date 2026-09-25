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

### Integration tests (real chains)

`e2e/` drives the page libraries (`src/lib/vault.ts`, `evm.ts`, `claim.ts`, `kyber.ts`) the way the pages do, with a
wagmi mock connector standing in for the wallet:

```sh
.venv/bin/python tools/vault_demo.py up      # from the repo root: anvil + validator + relay + vault fly + site
e2e/run.sh vault                              # lock/approve, request, cancel, withdraw, pause, every page error text,
                                              # then two claims paid by the fly on the local validator (~5 min)
anvil --fork-url https://rpc.mainnet.chain.robinhood.com --port 8546 &
e2e/run.sh trading                            # KyberSwap quotes + calldata, buy and sell simulated on mainnet (eth_call
                                              # with state overrides), approval and error paths on the fork
```

Known state on 2026-09-24: KyberSwap's quotes for *selling* $FLY claimed more USD out than in (a mispriced hop
through the Uniswap v4 pool), and its router then refused the swap at every slippage up to the page's 10 % cap.
The page marks such a quote as suspicious and its pre-flight `eth_call` reports the refusal before any wallet prompt.
The sell test walks the page's slippage ladder (0.2 %, then 0.5 % steps to 10 %) and prints the first step the real
pools honour; a step above the default is a finding worth reading.

They are not part of `npm test`. The vault suite is rerunnable (fresh keys each run; it funds a new settlement for
the claim section and waits up to 12 minutes for it). The trading suite talks to the public KyberSwap API, which
throttles bursts; it backs off for minutes rather than skip. The slippage ladder is measured with `eth_call` on
the real chain (state overrides give the test wallet its balance; nothing is sent). The swap is then also sent on
the fork at the measured step; start the fork right before the run, because one more than a few minutes old drifts
from the chain and KyberSwap's executor refuses it early (reported as "fork drift", not as a failure).

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
