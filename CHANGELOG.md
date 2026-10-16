# Changelog

Releases are identified by date. The Docker distribution (branch `distribution`, Hugging Face `bryceweiner/fly-trader`)
ships signed releases built from this development line.

## 2026-10-17: A new operator console

The Streamlit console is replaced by a web dashboard served by the same single process.

### Upgrading

- **Nothing to do for Docker installs:** the self-updater installs the new dependencies (FastAPI and uvicorn instead of
  Streamlit) from `uv.lock`, and the dashboard ships pre-built, so no Node is needed. Open <http://localhost:8501> as before.
- **Bare-metal installs:** `uv sync` (add your torch extra), then `uv run fly-trader ui`. The console now binds to
  `127.0.0.1` by default; Streamlit listened on every interface. Set `CONSOLE_HOST` to change it. Inside a container it
  binds the container's interface, so the published port still works.
- **A remote console** (the vault server) is still reached through an SSH tunnel. Requests that name a host other than
  `localhost`/`127.0.0.1` are refused; list extra names in `CONSOLE_ALLOWED_HOSTS` if you reach it another way.
- **New:** `fly-trader ui --observe --port 8599` serves the dashboard without hosting workers (a preview next to a running
  console; its controls are refused).

### Live money

- **The wallet can trade before the handover:** `LIVE_WITHOUT_HANDOVER=1` mirrors the fly's teacher-gated buys (the
  selector's buys and sizes, less what the fly's learning vetoes) onto the wallet as soon as signing is allowed, on
  Solana and Robinhood Chain, whoever holds the seat. The race still decides the seat. The default (0) keeps the old rule.
- **`fly-trader reset-paper`:** zeroes the paper race books' stats when live funds arrive, so they restart from
  `CAPITAL_SOL` / `RH_CAPITAL_ETH`. Their trades, fills, decisions and wealth marks are archived first. What the fly
  learned, its race clock, the live books and Kalshi are kept.
- **The console says live when it trades live:**
  - Overview's narrative, headline numbers and wealth chart follow the live book.
  - Trades opens on the live book.
  - The Safety page's money rows say what the wallet does.
  - The race card's "racing for" counts from the fly's own race clock, so a paper reset does not restart it.
- **Fixed:** `tools/publish_release.py` read the current release over TLS without root certificates (python.org's macOS
  Python) and counted the failure as "no release", so it would have published release 1 again. It now uses certifi and
  stops on anything but a 404.
- **Fixed:** a SIGTERM killed the worker threads mid-step: uvicorn re-raises the signal once the server stops. The
  console now stops them in the app's shutdown and waits for them to flush.

### The console

- **One process, as before.** `fly-trader ui` still starts the worker threads and stops them on exit; it now serves a React
  dashboard (Dashtreme's horizontal layout with the fly-trader colours) from FastAPI.
- **Live without reloading.** Every panel is pushed over one WebSocket the moment it changes. Each panel is computed once
  for all open tabs.
- **Every page of the old console, with nothing lost:** Overview, Trades, Model & training, Data pipelines, Robinhood
  Chain, the four Kalshi pages, Processes, Safety & wallet and the $FLY vault keep their routes (`/vault`, `/kalshi`, …).
- **States are colour widgets** at the top of every page (paper or live, trading, selector, fly, training, feed).
  Charts, gauges and datatables (paging, search, copy, CSV, print) replace the Streamlit tables.
- **New:**
  - A fly decision inspector: click any token for the fly's score against its teacher's and the buy line over time, what
    it bought and why, and the synapses it has learned most.
  - A live fly ticker, with a toast for every live-money buy, sell and fill.
  - Brain playback: replay the last 24 hours of the fly's activity, and click a neuron to follow its history.
  - Global search (token, mint, Kalshi ticker, transaction signature), a notifications bell, and a Logs page that tails
    every worker's log file.
- **Safer controls.**
  - Every state-changing button asks for confirmation and says what it will do.
  - The server enforces the rules the old console only showed: no reset while its engine runs, one withdrawal at a time
    per chain, the withdrawal re-planned and its blockers re-checked on the server.
  - The console refuses requests from other sites. A web page open in the same browser cannot press its buttons.
- **Fixed:**
  - The Kalshi model page no longer stops with an error after its first panel.
  - The vault worker has a card on Processes.
  - A dormant Kalshi (no `KALSHI_API_KEY_ID`) hides its status, worker cards and safety panels, and the console no longer
    shows part of the Kalshi key.
  - "Re-bootstrap" starts the trainer when it is not running.
  - A feed that runs as a CLI process no longer shows as stale.

## 2026-10-11: Robinhood Chain, and a fly that starts as its selector

Published to Hugging Face as signed release 1. Ships fly #157 (network) with selector #154, both trained on Solana and
Robinhood Chain rows together.

### Upgrading

- **Docker installs from before October 2026 must rebuild once:** `git pull && docker compose up -d --build`. The old
  image cannot verify signed releases, so it stops updating after this release. Data, the fly's learned state and
  open positions are kept.
- **Robinhood Chain is off by default** in the distribution. To turn it on, set `RH_ENABLED=1` and `ENVIO_API_TOKEN` (an
  Envio HyperSync token; the free tier works) in `.env`.
  - Indexing the full history from 2026-08-04 takes about 30 GB.
  - Setting `RH_START_DAY` about 40 days back is enough to trade.
- **New settings:**
  - `FLY_SCORE` (default `teacher`), `FLY_TEACHER_GATE` (default on), `FLY_LEARNED_VETO` (default 0.02) and `FLY_SIZING`
    (default `kelly`).
  - Robinhood Chain: `ENVIO_API_TOKEN`, `RH_*`, and optionally `QUICKNODE_RH_API_URL` for transactions.

### The fly

- **It scores as its selector plus what it has learned** (`FLY_SCORE=teacher`).
  - On the first day its score equals the selector's exactly, verified live to 3e-8 on every row.
  - Its learnable mushroom-body pathway (the plastic KC→MBON synapses) adds a correction, learned from realized returns,
    that starts at zero.
  - It therefore learns only where the selector is wrong.
  - The earlier fly was a distilled copy of the selector, accurate overall but not at the buy line: it ranked the
    selector's top 1 % of candidates only about 0.6-correlated.
- **It buys only what the selector buys** (`FLY_TEACHER_GATE`).
  - The selector's trigger, buy line, win filter, dump veto and fail-closed rules decide every row.
  - The fly's learning can drop a selector trade (a learned veto: its score more than 0.02 below its starting score) but
    can never add one.
  - Ungated, the fly had bought rows the selector rejected; those rows averaged −25.9 % per trade.
- **It sizes each buy exactly as the selector sizes it** (`FLY_SIZING=kelly`): the selector's score, buy line and Kelly
  table for that row.
- **It learns from the selector's trades,** live and in the replay, after a restart too. It no longer learns from rows
  above its own buy line.
- **It never trades without its teacher.** A fly whose selector cannot be loaded enters nothing. A release install now
  points the fly at the shipped selector's new database id. Before that fix, the Docker boot test found a fresh install
  running the fly ungated.
- **Replay verdict** (2026-09-24 to 10-09, the fly against the selector itself):
  - Solana +3.78 % per trade over 308 trades.
  - Robinhood Chain +2.66 % over 1,477 trades (selector +2.63 %).
  - Random picks lost 4.4–4.6 %.
- **Smaller changes:**
  - The fly calibrates on days its teacher never saw.
  - A NaN mean can no longer pass the deploy gate.
  - A second chain's teacher gets 28 days of its own data before the replay starts.
  - `fly-replay --chains` re-judges one chain and keeps the others' verdicts.

### Robinhood Chain (new)

- **Pons memecoins traded beside pump.fun.** It trades tokens launched on Pons V2 after they graduate into Uniswap v4 pools
  with the Pons hook.
  - Each chain has its own books in its own unit (Robinhood Chain in ETH), with its own bankroll, kill switch, handover
    and verdict.
  - One selector and one fly cover both chains. A chain trades only if its own replay verdict passed.
- **Its own indexer.** It reads launches, graduations, swaps, the base assets' prices and 1-minute rows.
  - Logs come from Envio HyperSync, in one query per pass.
  - Minutes land about 15–25 s after they close (they took 35–95 s at first).
  - If HyperSync fails, the official RPC carries the feed until it recovers.
  - RH minutes up to 240 s old are still traded (`RH_TRADE_LAG_S`).
- **Live trading** (`RH_LIVE_ENABLED=0` by default):
  - Swaps route through KyberSwap, with a Uniswap v4 fallback.
  - One transaction is in flight at a time, booked from measured balances.
  - It recovers after a crash and is reconciled against the chain.
  - It has its own circuit breaker (circuit 3).
  - `fly-trader rh-smoke` runs a tiny mainnet round trip.
  - A QuickNode endpoint (`QUICKNODE_RH_API_URL`), if set, carries transactions. Its free plan caps log queries at 5
    blocks, so logs stay on HyperSync.
- **Costs:** the Pons hook fee and creator tax (0.99 % per side at the median), price impact and gas.
- **Console:**
  - A chain picker switches the Overview and Trades pages.
  - A Robinhood Chain page shows the indexer, quote-asset prices and the ETH wallet.
  - Safety & wallet shows both chains' bot wallets in USD and withdraws only to your own wallets.

### $FLY vault

- Robinhood Chain profit is paid to lockers in ETH, beside the SOL pot.
- **Custody:** a Squads treasury with spending limits and a key-isolated signer container. Claims are paid from a
  separate payout wallet, and code releases are delayed.
- **Nothing published helps trade ahead of the fly.** The trading wallet's address, open positions and settlement
  timing are not shown.
- **Audits:** red-team tests, mutation testing, Echidna and Halmos proofs.
  - Fixed: a claim double-pay on resume.
  - Added: Kyber calldata and fee checks, and relay griefing limits.

### Docker distribution

- Runs the development line's code again, merged instead of hand-ported (it had fallen about 35 commits behind).
- **Signed releases only.**
  - The container installs a release only if its manifest is signed by a key in `release/allowed_signers`, its number
    is higher than the installed one, and every file matches its hash.
  - Releases carry no `.git*` file and no model they do not name.
  - Startup removes model files an earlier release left behind.
- **Models are accepted whether or not Robinhood Chain is on.** Before release, an install with RH off would have
  rejected the combined models and crash-looped at boot.
- The Robinhood Chain reference table ships in the seed.
- Kalshi ships dormant: no worker starts, and its pages appear only once `KALSHI_API_KEY_ID` is set.

### Kalshi prediction markets (development line)

- Under development, not traded in the distribution. Covers:
  - an outcome-blind event corpus
  - a selector judged by a weekly sign test
  - maker and taker arms
  - a visual fly

### Fixes

- **Schema changes no longer deadlock live workers.** An unchanged schema is skipped, and a changed one waits at most
  5 s for a lock, then retries.
- **The RH paper books had made no trades.** A day's wallet-skill table did not exist yet when its minutes were scored.
  Tables are now written ahead of their day, and the stream falls back to the newest one.
- **HyperSync could stall for 76 s.** A dead server address hung new connections; connections now time out in 5 s and
  are kept alive.
- **The RH price scan ran about 270 queries per pass.** It now reads every price cursor at once.

## 2026-09-24: Fly #107 taught by selector #100

- The selector was retrained on real fees, and the fly was bootstrapped from it.
- The container keeps itself current from Hugging Face.

## 2026-09-22: First distribution

- Fire-and-forget trading in Docker, with the fly, its teacher selector and their connectome.
