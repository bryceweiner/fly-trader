# fly-trader — distribution

**A fruit-fly connectome that trades memecoins with your money: on Solana, and on Robinhood Chain if you turn it on.**

> ## ⚠️ Read this first
>
> This software signs real transactions from a wallet you fund. It buys tokens that graduated from pump.fun (and, if
> you turn Robinhood Chain on, from Pons), holds them for two to four hours, and sells. **It can lose everything in that wallet, quickly.** The model was trained on a few
> months of data; the market it trades is adversarial and changes. Nothing here is a claim of profit, and nothing here is
> financial advice. Run it only with money you can lose entirely, on a wallet used for nothing else. MIT licence — no
> warranty of any kind. See [LICENSE](LICENSE).

This is the ready-to-run branch: the trading engine, its console, a trained fly, the selector that taught it, and the
connectome they run on. The training pipeline lives on [`master`](https://github.com/bryceweiner/fly-trader).

## What it does

- Every minute it scores every PumpSwap token that graduated from pump.fun at least six hours ago, from live 1-minute
  candles (a token needs that much history before its features mean anything).
- The fly — the FlyWire *Drosophila* central brain (41,756 neurons) as a rate network on its real signed synapses —
  starts as the gradient-boosted selector that taught it: it scores every token with the selector's own prediction of
  its net return. Its mushroom body's KC→MBON synapses then learn, from what the trades earn, a correction to that
  prediction, and can veto a selector buy they have learned to distrust.
- It buys only what the selector buys (the selector's strategy, buy line and filters), sizes the position exactly as the
  selector does (a quarter of the growth-optimal bet for its score band, capped), holds two to four hours (the
  strategy's hold), sells.
- **Every transaction is checked before it is signed** (`fly_trader/signer`): swaps go only through Jupiter's own
  routes, each one is simulated first and refused if it would spend more than ordered, deliver the token anywhere but
  your wallet, touch another holding, or return far less value than it costs; buys are capped per trade, per day and per
  minute.
- **It starts on paper.** After 20 closed paper trades it switches itself to the wallet and stays there. Before that,
  and whenever the wallet is unfunded or a key is missing, it trades paper and says so on the console.
- **Rails:** 20 % below the wallet's peak it stops entering and sells every open position (the kill switch); three
  failed transactions in a row trip a circuit breaker; a stale market feed halts entries; if live fills trail the paper
  mirror by more than 2 points per trade over 50 trades, entries pause and it tells you.
- **Watch it think.** The Overview page draws the fly's brain in three dimensions: every neuron lit by the minute's
  scoring, and the synapses it has changed by learning drawn as lines. Drag to orbit, hover a neuron for its cell type.

## Requirements

- Docker (Desktop on macOS/Windows, or Engine + Compose v2 on Linux; on Windows follow [INSTALL.md](INSTALL.md)). ~2 GB of disk for the image and database
  (Robinhood Chain, if you turn it on, adds ~30 GB for its full history; see below).
- A [Helius](https://dashboard.helius.dev) API key (free tier is enough) and a [Jupiter](https://portal.jup.ag) API key.
  Optional: a second Solana RPC from another provider as `SIGNER_RPC_URL`, so the checks above do not rely on Helius
  alone.
- 0.5 SOL you are prepared to lose. Not more: the caps below are sized for 0.5.
- Any 64-bit CPU: the fly scores about as fast on 16 CPU threads as on a GPU (its forward pass is a memory-bound
  scatter over a million synapses). An NVIDIA or AMD GPU is used when you start it with the CUDA or ROCm override (see
  below). Apple GPUs are not reachable from inside Docker.

> ### ⚠️ Robinhood Chain needs an Envio HyperSync API token
>
> Robinhood Chain is off by default. To trade it, create a token in the [Envio dashboard](https://envio.dev/app/api-tokens)
> (personal accounts get one free), put it in `.env` as `ENVIO_API_TOKEN`, and set `RH_ENABLED=1`. Its feed reads the
> chain's logs from [Envio HyperSync](https://envio.dev/pricing/hypersync); the public Robinhood Chain RPC is far too slow
> to index its history. Without a token, leave Robinhood Chain off: Solana trades either way.

## Start trading

**In one command:** after step 1, `./install.sh` does steps 2 to 4 and opens the console. It asks for the two API keys
(and checks them), offers an NVIDIA GPU when Docker can reach one, builds and starts the trader, waits until it is
healthy, creates the bot wallet and restarts the trader so it loads the key. Then back up `.env` and fund the wallet
(step 5; no restart needed). On Windows, [INSTALL.md](INSTALL.md) sets up WSL 2 and Docker Desktop first.

1. Download this branch and open a terminal in it:
   ```bash
   git clone -b distribution https://github.com/bryceweiner/fly-trader.git && cd fly-trader
   ```
2. Create your `.env` and paste the two API keys into it:
   ```bash
   cp .env.example .env
   ```
3. Start it:
   ```bash
   docker compose up -d --build
   ```
   The first start builds the image (several minutes), creates the database, downloads the wallet-skill table (~300 MB,
   checked against its pinned sha256), installs the models and starts the market feed and the trading engine. The
   Processes page also lists a `vault` worker that stops at once: it runs only on the hosted $FLY vault server.
4. Create the bot wallet — it prints the public key and writes the secret into your `.env`:
   ```bash
   docker compose exec fly fly-trader wallet new
   ```
   **Back up `.env` now.** It is the only copy of that key. (`wallet new` refuses if `.env` already has a
   `BOT_PRIVATE_KEY=` line, even an empty one.)
5. Send **0.5 SOL** to the printed address, then restart the engine so it picks up the key:
   ```bash
   docker compose restart fly
   ```
6. Open the console: <http://localhost:8501>. Watch the **Overview** page.

That is all. From here it runs on its own. The first ~24 hours are a warm-up (it needs a day of candles before it
scores anything); then it trades paper until 20 trades have closed; then it trades the wallet.

## Day to day

| Want to | Do |
|---|---|
| See what it is doing | <http://localhost:8501> — **Overview** (books, open positions, wealth), **Trades**, **Safety & wallet** (rails, wallet), **Processes** |
| Pause new entries | Safety & wallet → **Pause entries**, or `docker compose exec fly fly-trader pause-entries` |
| Clear a tripped kill switch | Safety & wallet → **Clear kill switch**, or `docker compose exec fly fly-trader reset-circuit --kill` (re-bases the peak) |
| Stop everything | `docker compose down` — **open positions stay open in the wallet**; start again and it sells them on schedule, or sell them yourself |
| See the logs | `docker compose logs -f fly` |
| Start again from nothing | `docker compose down -v` (deletes the database) and `sudo rm -rf data/` (the container writes it as root; it also holds the signer's ledger) |
| Open the console from another device | In `.env`, `CONSOLE_PUBLISH=0.0.0.0` and `CONSOLE_ALLOWED_HOSTS=` the address you will type (this machine's network address or name), then `docker compose up -d --force-recreate fly`. Only on a network you trust |

The console has no login: it can start and stop the engine and clear the rails. It listens on `127.0.0.1` only unless
you set `CONSOLE_PUBLISH`. Never expose it to the internet.

## What is in the box

| Path | What |
|---|---|
| `models/policies/fly_*.pt` | the fly (#604 on the training machine; a fresh install gives it its own id): taught by selector #595 on exactly the same data (a matched pair), one strategy (`capitulation`), 240-minute hold, its line matched to the selector's trade count. Loaded weights-only: the file cannot run code |
| `models/selectors/selector_*.skops` | the selector that taught it (#595), kept so the console can show the paper selector alongside. Saved with skops, not pickle |
| `models/connectome/` | the FlyWire FAFB v783 central brain as the fly runs on it (14 MB), with its calibrated scale |
| `models/wallet_skill/` | the fitted wallet-skill inputs: `config.json` (the horizon and lookback the model was trained with), the name of the newest per-day table (`TABLE`) and its sha256 (`TABLE.sha256`). The table itself (~300 MB) is on Hugging Face; the container downloads it once and uses it only if the hash matches |
| `seed/release_seed.json` (`seed/seed.sql` is the fallback) | the database rows that admit the fly to trade: the model records (by file hash), the passed replay verdict, autostart |
| `release/allowed_signers` | the release keys this install trusts for automatic updates (see Updates) |
| `.env.example` | every knob, set for 0.5 SOL |
| `docker-compose.yml`, `Dockerfile`, `docker/` | one Postgres, one engine + console (the Dockerfile's default stage; its `signer` stage is for the vault server) |
| `deploy/` | the hosted $FLY vault server: a separate signer container, host hardening, the signed-release updater ([docs/vault/RUNBOOK.md](docs/vault/RUNBOOK.md)) |

Out of sample, on the 21 days 2026-10-02..10-22 that neither model trained on, the selector's holdout-blind copy made
+6.65 % per trade after real costs over 485 trades (55 % winners, profit factor 1.61) against −4.06 % for random picks.
That is a backtest on one corpus, not a forecast.

## Tuning

Edit `.env`, then `docker compose up -d --force-recreate fly`. A plain `docker compose restart fly` keeps the values the
container was created with, so it does not apply a change. The ones that matter:

| Variable | Shipped | Meaning |
|---|---|---|
| `CAPITAL_SOL` | 0.5 | what you funded; also the paper warm-up bankroll |
| `MAX_POSITION_FRACTION` | 0.10 | the most of the bankroll one position may take |
| `MAX_POSITION_SOL` | 0.05 | the fixed size for a model without a sizing table (the shipped fly has one); must be above 0 to trade live |
| `GAS_RESERVE_SOL` | 0.05 | never spent |
| `KILL_SWITCH_DRAWDOWN` / `KILL_SWITCH_LIQUIDATE` | 0.20 / 1 | drawdown from peak that halts and sells everything |
| `HANDOVER_MIN_TRADES` | 20 | closed paper trades before it goes live |
| `LIVE_ENABLED` | 1 | set 0 to run paper only |

Scaling the wallet up scales the positions with it (`MAX_POSITION_FRACTION`). The signer has its own limits that do
not scale: at most `SIGNER_MAX_BUY_SOL` (1) per buy, `SIGNER_DAILY_BUY_SOL` (10) of buys a day and
`SIGNER_SWAPS_PER_MIN` (6) swaps a minute; raise them in `.env` deliberately.

## Robinhood Chain (Pons) memecoins

The same engine also trades Robinhood Chain: tokens launched on Pons that graduated into Uniswap v4 pools.

- **Off by default; turn it on with `RH_ENABLED=1` and `ENVIO_API_TOKEN` in `.env`** (an Envio HyperSync token; the
  free tier works). The `rh_stream` worker then starts with the others and indexes Pons launches, graduations and swaps
  into 1-minute rows: a backfill from `RH_START_DAY` (default 2026-08-04, ~30 GB; ~40 days back is enough to trade),
  then live, with RH minutes landing ~15-25 s after they close. If HyperSync fails, the official RPC carries the feed
  until it is back.
- **Paper, in ETH.** The Robinhood Chain books
  (`paper_rh_selector`, `paper_rh_fly`) keep their money in native ETH (`RH_CAPITAL_ETH`, 5 SOL's worth at build time)
  and pay Pons' hook fee, price impact and gas. They have their own kill switch and circuit (Safety & wallet →
  Robinhood Chain trading rails) and their own handover. The console's chain picker switches Overview and Trades
  between the two chains, and the **Robinhood Chain** page shows the indexer, the quote-asset prices and the ETH wallet.
- **One model for both chains.** The shipped selector and fly were trained on both chains' rows together, and the
  replay judged each chain on its own trades; a chain's books trade only where its verdict passed. With RH off the same
  models trade Solana alone.
- **Live** needs `RH_LIVE_ENABLED=1`, `RH_BOT_PRIVATE_KEY` (in `.env` only, never in the image) and the address it
  derives to as `RH_BOT_ADDRESS`, plus a funded wallet (ETH on Robinhood Chain, chain 4663). Then
  `docker compose up -d --force-recreate fly`. As on Solana, the fly trades the wallet only after it has won the RH paper race.
- `RH_ENABLED=0` turns all of it off.

RH training data does not ship: an install that turns RH on builds its own from the stream, as Solana does.

## NVIDIA GPU

Optional. The CPU image trades fine; a GPU makes the once-a-minute scoring and the fly's learning a little faster.
You need the NVIDIA driver (560 or newer) and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
on the host; then start with the CUDA override instead of the plain command in step 3:

```bash
docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
```

The image is built with the CUDA 12.6 torch wheel (`TORCH_INDEX: cu126` in `docker-compose.cuda.yml`; `cu130` for the
newest drivers) and the container is given every GPU. `DEVICE=auto` in `.env` then resolves to the GPU; the console's
Model & training page shows which device the fly is on. Batch sizes are derived from the card's free memory, so an 8 GB card works;
on a shared card set `DEVICE_MEMORY_GB` in `.env` to what the fly may use.

## AMD GPU

Optional, as above. Linux x86-64 with the `amdgpu` kernel driver is all the host needs: the ROCm 7.2 runtime is inside
the torch wheel. Start with the ROCm override instead of the plain command in step 3:

```bash
docker compose -f docker-compose.yml -f docker-compose.rocm.yml up -d --build
```

The image is built with the ROCm 7.2 torch wheel (`TORCH_INDEX: rocm72`, a much larger image than the CPU one), which
covers gfx1030 (RX 6800/6900), gfx1100-1102 (RX 7000), gfx1103 (Radeon 780M), gfx1150/1151 (Ryzen AI 300 and Ryzen
AI Max), gfx1200/1201 (RX 9000) and the Instinct cards. The container is given `/dev/kfd` and `/dev/dri`. `DEVICE=auto` resolves to the GPU
and the Model & training page shows it as `rocm:0` with its architecture. An integrated GPU (Ryzen AI Max) uses system
memory: set `DEVICE_MEMORY_GB` if the batch sizes derived from what the driver reports leave too little for the rest.

## Updates

The container keeps itself current, with **signed releases only**. Every hour it asks Hugging Face whether
[`bryceweiner/fly-trader`](https://huggingface.co/bryceweiner/fly-trader) has a new release. It installs one only if:
- its `release.json` is signed by a key in this install's `release/allowed_signers`
- its release number is higher than the installed one
- every file matches the size and sha256 the signed manifest lists

Then it restarts (about a minute; the paper books, the fly's learned state and any open positions are untouched --
open positions are sold on schedule by the restarted engine) and keeps the previous release beside the new one. If the
new release does not come up within five minutes, it goes back to the previous one and never tries that release again.
An unsigned or wrongly signed commit is refused and logged. While `release/allowed_signers` lists no key, nothing is
installed. `docker compose logs -f fly` shows what it did. `AUTO_UPDATE=0` in `.env` turns this off; then the
"Refreshing the model" steps below are the manual way.

**Installed before October 2026? Rebuild once.** The updater in those images installed whatever Hugging Face held,
unverified; it may install this release by itself, but only a rebuilt image can check signatures (it needs `ssh-keygen`),
so it stops at this release. From your install directory:

```bash
git pull && docker compose up -d --build
```

Your data, the fly's learned state and open positions are kept. Robinhood Chain stays off until you turn it on.

## Refreshing the model

The model is a snapshot of one training run. To ship a newer one, on a `master` checkout with its database, after a
passed replay:

```bash
.venv/bin/python tools/make_seed.py --out ../fly-trader-dist
```

Then publish it as a signed release with `tools/publish_release.py` (docs/vault/RUNBOOK.md, section 4.1): installs
with automatic updates pick it up within the hour. For a manual install, commit `models/` and `seed/` on this branch;
every start runs `fly-trader apply-release`, which reuses rows it already has (same file hash) and registers new
models, so pull the branch, `docker compose up -d --build`, and the new model is in place.

## Also on Hugging Face

- Model and this code: [`bryceweiner/fly-trader`](https://huggingface.co/bryceweiner/fly-trader)
- The training database (the full corpus, for retraining on `master`): [`bryceweiner/fly-trader-seed`](https://huggingface.co/datasets/bryceweiner/fly-trader-seed)

## Credits

FlyWire consortium, Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024; annotations Schlegel
et al. 2024; synapse signs Shiu et al. 2024. Connectome export: [snedea/flybrain](https://github.com/snedea/flybrain)
(MIT). Market data: [PumpAPI](https://pumpapi.io); routing: [Jupiter](https://jup.ag) (Metis routes only).

[MIT](LICENSE).
