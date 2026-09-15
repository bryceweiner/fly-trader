# fly-trader

**A fruit-fly connectome that trades Solana memecoins.**
Website: [fly-trader.app](https://fly-trader.app) · Token: $FLY (coming soon) · X community: coming soon

fly-trader trains the central brain of the [FlyWire](https://flywire.ai) adult *Drosophila* connectome (41,756 neurons, 1.04 M synapses; FAFB v783) to trade PumpSwap tokens that graduated from [pump.fun](https://pump.fun). A gradient-boosted selector learns which tokens to buy from 1-minute market data at real trading costs; the fly is then trained to make the same calls through its own wiring. Every decision is journaled to Postgres.

It is an experiment. No performance is claimed. The strategy trades a paper book; live execution is default-deny and not yet wired to it.

> **Not financial advice.** This software can sign real transactions from a wallet you fund. It may lose all of that money. See the [Terms of Service](https://fly-trader.app/terms.html).

## How it works

```
PumpAPI hourly archive ─▶ 1-minute candles ─▶ features (live engine) ─▶ selector (walk-forward) ─▶ fly (distilled)
PumpAPI live stream ───▶ 1-minute candles ─▶ features ─▶ loaded model ─▶ buy / hold 2 h / sell ─▶ paper book
Jupiter Tokens v2 ─────▶ token stats (history for future inputs)
```

| Piece | Where | What |
|---|---|---|
| Market data | `fly_trader/ingest/replay_pull.py`, `pumpstream.py` | Every PumpSwap trade of pump.fun tokens, from the free hourly archive (training) and the live stream (trading), aggregated to the same 1-minute candles |
| Features | `fly_trader/market/features.py`, `train/mature.py` | One engine for training and trading: returns, volatility, order flow, volume, liquidity, trade counts, age, drawdowns, launch facts and creator history |
| Costs | `fly_trader/market/exit_cost.py` | Measured, not assumed: the PumpSwap pool fee by market cap (1.25 % → 0.30 %), Jupiter's 10 bps, the network fee, and constant-product price impact |
| Selector | `fly_trader/train/selector.py`, `decisions.py` | Gradient-boosted model of each minute's net return over a 2-hour hold. Walk-forward over the whole archive; the buy line is chosen on the first half of the held-out days and judged on the second; it trades only if that half made money after costs and beat random picks |
| Fly | `fly_trader/train/fly_selector.py`, `brain/connectome.py` | The FlyWire central brain as a rate network on its real, signed synapses: features enter the sensory neurons, a decoder reads the descending neurons; trained to reproduce the selector's predictions, then traded identically |
| Sizing | `fly_trader/agent/sizing.py` | Position size from the model's measured certainty: a fraction of the growth-optimal bet for its score band, capped by the bankroll and the pool's depth |
| Pipeline | `fly_trader/train/pipeline.py` | Every 7 days (or on request): retrain the selector, then the fly from it; the trading engine switches models without a restart |
| Trading engine | `fly_trader/agent/selector_session.py` | Scores every eligible token once a minute, buys, holds for the model's hold time, sells; kill switch, pause and reserve rails |
| Console | `fly_trader/ui/app.py` | One Streamlit app that runs every worker as an in-process thread and shows what the system is doing |

## Requirements

- macOS on Apple Silicon (the fly trains on the Metal GPU via PyTorch MPS; 64 GB+ recommended for the full archive)
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 15+ (`brew install postgresql@17`)
- A Jupiter API key (token stats); a Helius API key and a funded wallet only for live execution

## Install

```bash
git clone https://github.com/bryceweiner/fly-trader.git
cd fly-trader
uv sync
cp .env.example .env          # paste your keys; leave LIVE_ENABLED=0
createdb fly_trader
uv run fly-trader init-db
uv run fly-trader build-connectome   # ~100 MB download, verifies 139,255 neurons / 2,698,236 edges
uv run fly-trader ui                 # Streamlit console at http://localhost:8501
```

The console starts the market feed, the history archive (downloads the archive and builds the training features), the trainer, the trading engine and the token-stats worker. Training starts once every archived day has its features.

## Train

The trainer runs by itself every 7 days; "Retrain now" in the console queues a run. From the CLI:

```bash
uv run fly-trader train-selector        # walk-forward backtest over the whole archive, saves the model
uv run fly-trader train-fly-selector    # trains the fly to imitate the latest selector
```

## Trade

The trading engine trades the paper book as soon as a model qualifies. Live execution is not wired to the selector yet; the wallet and signing path exist:

```bash
uv run fly-trader wallet new          # writes BOT_PRIVATE_KEY to .env, prints the pubkey
uv run fly-trader swap-smoke --sol 0.01   # one real round trip to prove signing and routing
```

Rails: kill switch at −30 % from peak (blocks entries), gas reserve, circuit breaker, stale-feed halt, pause. `pause-entries`, `resume-entries`, `reset-circuit` and `status` are available from the CLI and the console.

## Configuration

Every knob is an environment variable read by `fly_trader/config.py`; `.env.example` lists the ones you are expected to touch. Notable ones:

| Variable | Default | Meaning |
|---|---|---|
| `CAPITAL_SOL` | — | starting paper bankroll |
| `GAS_RESERVE_SOL` | `0.30` | never deployed |
| `KELLY_FRACTION` | `0.25` | share of the growth-optimal bet per score band |
| `MAX_POSITION_FRACTION` / `MAX_POOL_SHARE` | `0.10` / `0.02` | largest position: of the bankroll / of the pool's SOL reserve |
| `KILL_SWITCH_DRAWDOWN` | `0.30` | drawdown from peak that blocks new entries |
| `DEVICE` | `mps` | torch device for the fly |
| `LIVE_ENABLED` | `0` | signing on mainnet |

## CLI reference

```
init-db  wallet {new,show}  ui [--port]  run  status  worker {start,stop,status} [name]
pumpstream  replay-pull  discover [--once]
assemble-replay  build-corpus-meta  build-mature  build-corpus-features  backtest-corpus
train-selector [--days]  train-fly-selector [--days --test-days --line --epochs]  selector-eval  reset-training
build-connectome  swap-smoke [--sol]  verify-fills  close-empty-atas
pause-entries  resume-entries  reset-circuit [--kill]
```

## Tests

```bash
createdb fly_trader_test
uv run pytest -q
```

## Data and citations

- FlyWire consortium, Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024. Annotations: Schlegel et al., Nature 2024. Signs: Shiu et al., Nature 2024.
- Connectome export: [snedea/flybrain](https://github.com/snedea/flybrain) (MIT).
- Whole-connectome graph models: FlyGM, arXiv 2602.17997; ConnecTorch.
- Market data: [PumpAPI](https://pumpapi.io) archive and stream; [Jupiter](https://jup.ag) Tokens API.

## License

[MIT](LICENSE).
