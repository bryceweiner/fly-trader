# fly-trader — distribution

**A fruit-fly connectome that trades Solana memecoins with your money.**

> ## ⚠️ Read this first
>
> This software signs real transactions from a wallet you fund. It buys tokens that graduated from pump.fun, holds them
> for about two hours, and sells. **It can lose everything in that wallet, quickly.** The model was trained on a few
> months of data; the market it trades is adversarial and changes. Nothing here is a claim of profit, and nothing here is
> financial advice. Run it only with money you can lose entirely, on a wallet used for nothing else. MIT licence — no
> warranty of any kind. See [LICENSE](LICENSE).

This is the ready-to-run branch: the trading engine, its console, a trained fly, the selector that taught it, and the
connectome they run on. The training pipeline lives on [`master`](https://github.com/bryceweiner/fly-trader).

## What it does

- Every minute it scores every PumpSwap token that graduated from pump.fun at least six hours ago, from live 1-minute
  candles (a token needs that much history before its features mean anything).
- The fly — the FlyWire *Drosophila* central brain (41,756 neurons) as a rate network on its real signed synapses —
  predicts each token's net return over the next 120 minutes. It was taught by a gradient-boosted selector; it then keeps
  learning from what its own trades earn, through plasticity at its mushroom body's KC→MBON synapses.
- It buys when the predicted return clears its own buy line, sizes the position from measured certainty (a quarter
  of the growth-optimal bet, capped), holds 120 minutes, sells.
- **It starts on paper.** After 20 closed paper trades it switches itself to the wallet and stays there. Before that,
  and whenever the wallet is unfunded or a key is missing, it trades paper and says so on the console.
- **Rails:** 20 % below the wallet's peak it stops entering and sells every open position (the kill switch); three
  failed transactions in a row trip a circuit breaker; a stale market feed halts entries; if live fills trail the paper
  mirror by more than 2 points per trade over 50 trades, entries pause and it tells you.

## Requirements

- Docker (Desktop on macOS/Windows, or Engine + Compose v2 on Linux). ~2 GB of disk for the image and database.
- A [Helius](https://dashboard.helius.dev) API key (free tier is enough) and a [Jupiter](https://portal.jup.ag) API key.
- 0.5 SOL you are prepared to lose. Not more: the caps below are sized for 0.5.
- Any CPU. No GPU is used.

## Start trading

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
   The first start builds the image (several minutes), creates the database, installs the models and starts the market
   feed and the trading engine.
4. Create the bot wallet — it prints the public key and writes the secret into your `.env`:
   ```bash
   docker compose exec fly fly-trader wallet new
   ```
   **Back up `.env` now.** It is the only copy of that key.
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
| Start again from nothing | `docker compose down -v` (deletes the database) and delete `data/` |

The console listens on `127.0.0.1` only. Do not expose it: it can start and stop the engine and clear the rails.

## What is in the box

| Path | What |
|---|---|
| `models/policies/fly_*.pt` | the fly (`brain_snapshots` #62): taught by selector #59, two strategies (`ev`, `capitulation`), 120-minute hold, its own buy lines |
| `models/selectors/selector_*.joblib` | the selector that taught it (#59), kept so the console can show the paper selector alongside |
| `models/connectome/` | the FlyWire FAFB v783 central brain as the fly runs on it (14 MB), with its calibrated scale |
| `models/wallet_skill/` | the fitted wallet-skill inputs: `config.json` (the horizon and lookback the model was trained with) and the name of the newest per-day table. The table itself (~300 MB) is on Hugging Face; the container fetches it once at first start |
| `seed/seed.sql` | the database rows that admit the fly to trade: the model records, the passed replay verdict, autostart |
| `.env.example` | every knob, set for 0.5 SOL |
| `docker-compose.yml`, `Dockerfile`, `docker/entrypoint.sh` | one Postgres, one engine + console |

The fly's replay verdict shipped here: over the 28 evaluation days it made +1.52 % per trade after real costs (profit
factor 1.13) against −5.27 % for random picks of the same tokens. That is a backtest on one corpus, not a forecast.

## Tuning

Edit `.env`, then `docker compose restart fly`. The ones that matter:

| Variable | Shipped | Meaning |
|---|---|---|
| `CAPITAL_SOL` | 0.5 | what you funded; also the paper warm-up bankroll |
| `MAX_POSITION_SOL` / `MAX_POSITION_FRACTION` | 0.05 / 0.10 | per-position caps (the smaller wins) |
| `GAS_RESERVE_SOL` | 0.05 | never spent |
| `KILL_SWITCH_DRAWDOWN` / `KILL_SWITCH_LIQUIDATE` | 0.20 / 1 | drawdown from peak that halts and sells everything |
| `HANDOVER_MIN_TRADES` | 20 | closed paper trades before it goes live |
| `LIVE_ENABLED` | 1 | set 0 to run paper only |

Scaling the wallet up scales the position caps with it (`MAX_POSITION_FRACTION`), but `MAX_POSITION_SOL` is a hard
ceiling: raise it deliberately or not at all.

## Refreshing the model

The model is a snapshot of one training run. To ship a newer one, on a `master` checkout with its database, after a
passed replay:

```bash
.venv/bin/python tools/make_seed.py --out dist
cp -R dist/models/* models/ && cp -R dist/seed/* seed/
```

Commit `models/` and `seed/` on this branch. The seed is idempotent: an install that already has these rows is not
changed by it, so pull the branch, `docker compose up -d --build`, and the new model is in place.

## Also on Hugging Face

- Model and this code: [`bryceweiner/fly-trader`](https://huggingface.co/bryceweiner/fly-trader)
- The training database (the full corpus, for retraining on `master`): [`bryceweiner/fly-trader-seed`](https://huggingface.co/datasets/bryceweiner/fly-trader-seed)

## Credits

FlyWire consortium, Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024; annotations Schlegel
et al. 2024; synapse signs Shiu et al. 2024. Connectome export: [snedea/flybrain](https://github.com/snedea/flybrain)
(MIT). Market data: [PumpAPI](https://pumpapi.io); routing: [Jupiter](https://jup.ag).

[MIT](LICENSE).
