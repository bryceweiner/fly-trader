# fly-trader

**A fruit-fly connectome that trades Solana memecoins.**
Website: [fly-trader.app](https://fly-trader.app) · Token: $FLY (coming soon) · X community: coming soon

fly-trader trains the central brain of the [FlyWire](https://flywire.ai) adult *Drosophila* connectome (41,756 neurons, 1.04 M synapses; FAFB v783) to trade PumpSwap tokens that graduated from [pump.fun](https://pump.fun). A gradient-boosted selector learns which tokens to buy from 1-minute market data at real trading costs; the fly is then trained to make the same calls through its own wiring. Every decision is journaled to Postgres.

It is an experiment. No performance is claimed. The fly trades a paper book; once it has earned the selector's seat (the handover) and signing is allowed (`LIVE_ENABLED=1` and the live prerequisites), its decisions also trade the bot wallet. Live execution is default-deny.

> **Not financial advice.** This software can sign real transactions from a wallet you fund. It may lose all of that money. See the [Terms of Service](https://fly-trader.app/terms.html).

## How it works

```
PumpAPI hourly archive ─▶ 1-minute candles ─▶ features (live engine) ─▶ selector (walk-forward) ─▶ fly (distilled)
PumpAPI live stream ───▶ 1-minute candles ─▶ features ─▶ loaded model ─▶ buy / hold / sell ─▶ paper book (─▶ wallet after the handover)
Jupiter Tokens v2 ─────▶ token stats (history for future inputs)
```

| Piece | Where | What |
|---|---|---|
| Market data | `fly_trader/ingest/replay_pull.py`, `pumpstream.py` | Every PumpSwap trade of pump.fun tokens, from the free hourly archive (training) and the live stream (trading), aggregated to the same 1-minute candles |
| Features | `fly_trader/market/features.py`, `train/mature.py` | One engine for training and trading: returns, volatility, order flow, volume, liquidity, trade counts, age, drawdowns, launch facts and creator history |
| Costs | `fly_trader/market/exit_cost.py` | Measured, not assumed: the PumpSwap pool fee by market cap (1.25 % → 0.30 %), Jupiter's 10 bps, the network fee, and constant-product price impact |
| Selector | `fly_trader/train/selector.py`, `decisions.py` | Gradient-boosted model of each minute's net return over the strategy's hold. Walk-forward over the whole archive; the buy line is chosen on the first half of the held-out days and judged on the second; it trades only if that half made money after costs and beat random picks |
| Fly | `fly_trader/train/fly_selector.py`, `brain/connectome.py` | The FlyWire central brain as a rate network on its real, signed synapses: features enter the sensory neurons, a decoder reads the descending neurons; trained to reproduce the selector's predictions, then traded identically |
| Sizing | `fly_trader/agent/sizing.py` | Position size from the model's measured certainty: a fraction of the growth-optimal bet for its score band, capped by the bankroll and the pool's depth |
| Pipeline | `fly_trader/train/pipeline.py` | Every 7 days (or on request): retrain the selector, then the fly from it; the trading engine switches models without a restart. After the handover the weekly schedule stops; training runs only on request |
| Trading engine | `fly_trader/agent/selector_session.py`, `fly_session.py` | Scores every eligible token once a minute, buys, holds for the model's hold time, sells; kill switch, pause and reserve rails |
| Live mirror | `fly_trader/agent/fly_live.py`, `handover.py` | After the handover the fly's entries also trade the wallet; kill switch, circuit breaker, a pause when live fills trail the paper book |
| Signer | `fly_trader/signer` | The only code that signs. It simulates every swap and refuses one that would overspend, deliver elsewhere, touch another holding or return far less value; buys are capped per trade, per day and per minute; swaps go through Jupiter's Metis routes only. In-process here; on the vault server it is a separate container that alone holds the keys |
| Model files | `fly_trader/train/model_io.py` | Fly checkpoints load weights-only and selectors are skops files: a model file cannot run code |
| $FLY vault | `fly_trader/vault`, `web/`, `contracts/` | The hosted vault fly (only with `VAULT_ENABLED`): Squads treasury custody, weekly settlement, SOL claims ([docs/vault/SPEC.md](docs/vault/SPEC.md), [RUNBOOK.md](docs/vault/RUNBOOK.md)); its server files are on the `distribution` branch under `deploy/` |
| Kalshi | `fly_trader/kalshi` | A second trading type: prediction markets (`KALSHI_*` in `.env.example`) |
| Console | `fly_trader/ui/app.py` | One Streamlit app that runs every worker as an in-process thread and shows what the system is doing |

## Requirements

- Any of: **macOS on Apple Silicon** (the fly runs on the Metal GPU via PyTorch MPS), **Linux or Windows with an NVIDIA
  GPU** (CUDA), or **any 64-bit CPU**. The fly's forward pass is a memory-bound scatter over a million synapses, so a
  16-thread CPU scores as fast as the Apple GPU (~1,700 rows/s measured) and trains at about 0.8× its speed. Memory
  matters more than compute: batch sizes are derived from free memory (`fly_trader/brain/device.py`), so an 8 GB GPU
  works, in smaller batches; 64 GB of RAM is recommended for training on the full archive, 16 GB for trading.
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 15+ (`brew install postgresql@17`)
- A Jupiter API key (token stats; swaps); a Helius API key (RPC; also the signer's unless `SIGNER_RPC_URL` is set) and a funded wallet only for live execution

## Install

```bash
git clone https://github.com/bryceweiner/fly-trader.git
cd fly-trader
uv sync --extra dev           # macOS: the Apple-GPU torch from PyPI. Elsewhere add one: --extra cpu | --extra cu126 | --extra cu130
cp .env.example .env          # paste your keys; leave LIVE_ENABLED=0
createdb fly_trader
uv run fly-trader init-db
uv run fly-trader build-connectome   # ~100 MB download, verifies 139,255 neurons / 2,698,236 edges
uv run fly-trader ui                 # Streamlit console at http://localhost:8501
```

The console starts the market feed, the history archive (downloads the archive and builds the training features), the trainer, the trading engine and the token-stats worker. Training starts once every archived day has its features.

## Train

The trainer runs by itself every 7 days until the handover; "Retrain now" in the console queues a run. From the CLI:

```bash
uv run fly-trader train-selector        # walk-forward backtest over the whole archive, saves the model
uv run fly-trader train-fly-selector    # trains the fly to imitate the latest selector
uv run fly-trader fly-replay            # bootstraps the fly once, lets it learn over the corpus, judges it (the gate before it trades)
```

Selectors are saved as `.skops` and fly checkpoints load with `torch.load(weights_only=True)`; pickle formats are
refused. A checkout with older `.joblib` selectors converts them once: `uv run fly-trader models convert`.

## Trade

The trading engine trades the paper book as soon as a model qualifies. After the handover (`HANDOVER_*`) the fly's
entries also trade the wallet when `LIVE_ENABLED=1` and the live prerequisites are met (`CAPITAL_SOL`,
`MAX_POSITION_SOL`, `GAS_RESERVE_SOL`, a bot key, `JUPITER_API_KEY`, `HELIUS_API_KEY`, `SOLANA_CLUSTER=mainnet-beta`).
Every transaction is signed by `fly_trader/signer` (in-process here), which checks and simulates it first:

```bash
uv run fly-trader wallet new              # appends BOT_PRIVATE_KEY to .env (refuses if the name is already there), prints the pubkey
uv run fly-trader swap-smoke --sol 0.01   # one real round trip through the signer to prove signing and routing
```

Rails: kill switch at −30 % from peak (blocks entries; with `KILL_SWITCH_LIQUIDATE=1` also sells every open live
position), gas reserve, circuit breaker (3 failed transactions), stale-feed halt, pause, and a pause when live fills trail
the paper book; plus the signer's own caps (`SIGNER_MAX_BUY_SOL` 1 SOL per buy, `SIGNER_DAILY_BUY_SOL` 10 SOL a day, 6
swaps a minute). `pause-entries`, `resume-entries`, `reset-circuit` and `status` are available from the CLI and the
console.

## Configuration

Every knob is an environment variable read by `fly_trader/config.py`; `.env.example` lists the ones you are expected to touch. Notable ones:

| Variable | Default | Meaning |
|---|---|---|
| `CAPITAL_SOL` | — | starting paper bankroll |
| `GAS_RESERVE_SOL` | `0.30` | never deployed |
| `KELLY_FRACTION` | `0.25` | share of the growth-optimal bet per score band |
| `MAX_POSITION_FRACTION` / `MAX_POOL_SHARE` | `0.10` / `0.02` | largest position: of the bankroll / of the pool's SOL reserve |
| `MAX_POSITION_SOL` | `0` | fixed size for a model without a sizing table; must be above 0 to trade live |
| `KILL_SWITCH_DRAWDOWN` / `KILL_SWITCH_LIQUIDATE` | `0.30` / `0` | drawdown from peak that blocks new entries / also sell everything |
| `HANDOVER_DAYS` / `HANDOVER_MIN_TRADES` / `HANDOVER_BEAT_SELECTOR` | `14` / `30` / `1` | when the fly takes the selector's seat and trades the wallet |
| `SIGNER_RPC_URL` | Helius | the signer's own RPC; another provider is safer (it simulates every swap) |
| `SIGNER_MAX_BUY_SOL` / `SIGNER_DAILY_BUY_SOL` / `SIGNER_SWAPS_PER_MIN` | `1` / `10` / `6` | the signer's limits |
| `DEVICE` | `auto` | where the fly runs: `auto` (CUDA if present, else MPS, else CPU), `cpu`, `cuda`, `cuda:1`, `mps`; a missing backend falls back to the CPU with a warning |
| `DEVICE_MEMORY_GB` | measured | caps the memory batch sizes are derived from (a shared GPU, a container limit) |
| `LIVE_ENABLED` | `0` | signing on mainnet |

## CLI reference

```
init-db  wallet {new,show} [--key-file]  ui [--port]  run  status  health  ready-for-restart  worker {start,stop,status} [name]
pumpstream  replay-pull  refetch-pool-ids  discover [--once]  fit-wallet-skill [--days]
assemble-replay  build-corpus-meta  build-mature  build-corpus-features  backtest-corpus [--max-tokens --days --fees --only --universe]
train-selector [--days]  train-fly-selector [--days --epochs --device]  fly-replay [--days --device --start-day]
selector-eval  reset-training [--no-archive]  models convert  apply-release DIR [--bootstrap]
build-connectome [--annotations]  swap-smoke [--sol]  verify-fills  close-empty-atas
pause-entries  resume-entries  reset-circuit [--kill]
vault {status,scan,settle,reclassify,withdraw,custody,squads-setup,claims,resume}
kalshi-{subaccount {create,list},fund --usd,status,history,build,train-selector,train-fly,fly-replay,train,stream,run,
        pause-entries,resume-entries,reset-circuit [--kill]}
```

## Tests

```bash
createdb fly_trader_test
uv sync --extra dev
uv run pytest -q
(cd web && npm test)                  # the site (vitest)
(cd contracts && forge test)          # the vault contract
tools/custody_e2e.sh <mainnet RPC>    # the Squads custody against the real program on a local validator
```

`tests/test_custody_e2e.py` and `web/src/lib/squads.int.test.ts` are skipped unless `tools/custody_e2e.sh` runs them.

## Data and citations

- FlyWire consortium, Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024. Annotations: Schlegel et al., Nature 2024. Signs: Shiu et al., Nature 2024.
- Connectome export: [snedea/flybrain](https://github.com/snedea/flybrain) (MIT).
- Whole-connectome graph models: FlyGM, arXiv 2602.17997; ConnecTorch.
- Market data: [PumpAPI](https://pumpapi.io) archive and stream; [Jupiter](https://jup.ag) Tokens API and swaps.

## License

[MIT](LICENSE).
