# fly-trader

**A fruit-fly connectome that trades Solana memecoins.**
Website: [fly-trader.app](https://fly-trader.app) · Token: $FLY (Pump.fun, coming soon) · X community: coming soon

fly-trader runs the complete [FlyWire](https://flywire.ai) adult *Drosophila* connectome (FAFB v783: 139,255 neurons, 2.7 M connections, ≈33.9 M synapses) as a trained trading policy over recently graduated [pump.fun](https://pump.fun) tokens. Market observations are encoded into the fly's sensory neurons, activity propagates only along real synapses with neurotransmitter signs frozen, and a decoder on the descending neurons emits a target exposure per token. Execution goes through Jupiter; everything is journaled to Postgres and replayable.

It is an experiment. No performance is claimed. Live trading is default-deny.

> **Not financial advice.** This software can sign real transactions from a wallet you fund. It may lose all of that money. See the [Terms of Service](https://fly-trader.app/terms.html).

## How it works

```
Helius websocket ─┐                                            ┌─ Jupiter Swap v2
pump.fun swap API ─┼─▶ swap tape ─▶ 45 market features ─▶ encoder ─▶ CONNECTOME ─▶ decoder ─▶ target exposure ─▶ runner ─┤
PumpAPI replay ────┘      (Postgres / Parquet)          (afferent pops)  41,756 n   (descending)    per token    (rails)  └─ paper books
```

| Piece | Where | What |
|---|---|---|
| Connectome build | `fly_trader/brain/connectome_build.py`, `populations.py` | Downloads the flybrain export + FlyWire annotations, checks neuron/edge counts, writes `data/brain/connectome/*.npz`, labels 26 populations |
| Policy (default) | `fly_trader/brain/policy.py` | FlyGM-style rate model: `W = sign · softplus(θ)`, per-neuron gain/bias, learned encoder into sensory populations, decoder from descending neurons. Central brain (optic lobes excluded): 41,756 neurons, 1.04 M edges, 1.31 M params |
| Spiking mode | `fly_trader/brain/lif.py`, `plasticity.py`, `encoders.py` | Full-brain LIF with dopamine-gated KC→MBON plasticity. `BRAIN_MODE=lif` |
| Features | `fly_trader/market/features.py`, `danger.py`, `exit_cost.py` | 45 features per token per beat over 1 m–3 h windows |
| Ingest | `fly_trader/ingest/` | Discovery (Jupiter Tokens v2), Helius capture, pump.fun corpus pull, PumpAPI replay |
| Training | `fly_trader/train/` | Dataset from the tape → rule expert → imitation/DAgger → recurrent PPO on net P&L after fees and impact |
| Runner | `fly_trader/agent/` | 1 s beats, 128 slots, eligibility gates, reflex exits, risk rails, three books (`live`, `paper_free`, `paper_mirror`) |
| Execution | `fly_trader/exec/` | Jupiter swap order + execute, slippage ladder, fill verification |
| Console | `fly_trader/ui/app.py` | One Streamlit app that supervises every worker as an in-process thread. Nothing runs outside it |
| Storage | `fly_trader/db/schema.py` | Postgres (34 tables, partitioned tape) + Parquet archive |

## Requirements

- macOS on Apple Silicon (the brain runs on the Metal GPU via PyTorch MPS; 32 GB+ recommended)
- Python 3.13 and [uv](https://docs.astral.sh/uv/)
- PostgreSQL 15+ (`brew install postgresql@17`)
- A [Helius](https://helius.dev) API key
- Optional: a Jupiter API key, and a funded Solana wallet for live trading

## Install

```bash
git clone https://github.com/bryceweiner/fly-trader.git
cd fly-trader
uv sync
cp .env.example .env          # paste HELIUS_API_KEY; leave LIVE_ENABLED=0
createdb fly_trader
uv run fly-trader init-db
uv run fly-trader build-connectome   # ~100 MB download, verifies 139,255 neurons / 2,698,236 edges
uv run fly-trader ui                 # Streamlit console at http://localhost:8501
```

From the console start the `discover` and `capture` workers and let the tape fill for a few hours before training.

## Train

```bash
uv run fly-trader build-dataset                   # cached [T, tokens, 45] observations from the tape
uv run fly-trader train-policy --iterations 20    # imitation warm-start, then PPO; evals on the last 20 %
uv run fly-trader promote-checkpoint <id>         # make a snapshot the runner's policy
```

Optional corpus tooling: `corpus-pull`, `build-corpus-features`, `backtest-corpus`, `assemble-replay`.

## Trade

Paper books run as soon as the `runner` worker is started. To go live:

```bash
uv run fly-trader wallet new          # writes BOT_PRIVATE_KEY to .env, prints the pubkey
# fund the pubkey, then in .env:
#   LIVE_ENABLED=1  CAPITAL_SOL=5  MAX_POSITION_SOL=0.1  GAS_RESERVE_SOL=0.30
uv run fly-trader swap-smoke --sol 0.01   # one real round trip to prove signing and routing
```

Rails enforced in code regardless of the brain: max position, 24 h notional cap, kill switch at −30 % from peak (blocks entries), hard stop at −50 %, dead-bag sweep at 3 h, gas reserve, circuit breaker, stale-feed halt. `pause-entries`, `resume-entries`, `reset-circuit`, and `status` are available from the CLI and the console.

## Configuration

Every knob is an environment variable read by `fly_trader/config.py`; `.env.example` lists the ones you are expected to touch. Notable ones:

| Variable | Default | Meaning |
|---|---|---|
| `BRAIN_MODE` | `policy` | `policy` (trained connectome) or `lif` (spiking) |
| `DEVICE` | `mps` | torch device |
| `SLOTS` | `128` | tokens watched per beat |
| `LAUNCHPADS` | `pump.fun` | graduation source(s) |
| `LIVE_ENABLED` | `0` | signing on mainnet |
| `CAPITAL_SOL` / `MAX_POSITION_SOL` | `5` / `0.1` | capital and per-position cap |
| `KILL_SWITCH_DRAWDOWN` | `0.30` | drawdown from peak that blocks new entries |

## CLI reference

```
init-db  wallet {new,show}  discover [--once] [--probe]  capture  run  ui [--port]
corpus-pull  build-corpus-features  backtest-corpus  assemble-replay
build-connectome  calibrate-brain  brain-selftest  balance-brain  pretrain
build-dataset  train-policy [--iterations --window --eval-every --subgraph]  reset-training
snapshot  promote-checkpoint <id> [--hot]  swap-smoke [--sol]  status
pause-entries  resume-entries  reset-circuit [--kill]  verify-fills  close-empty-atas
archive-partitions  purge-archived --confirm  worker {start,stop,status} [name]
```

## Tests

```bash
createdb fly_trader_test
uv run pytest -q
```

## Data and citations

- FlyWire consortium, Dorkenwald et al., *Neuronal wiring diagram of an adult brain*, Nature 2024. Annotations: Schlegel et al., Nature 2024. Signs: Shiu et al., Nature 2024.
- Connectome export: [snedea/flybrain](https://github.com/snedea/flybrain) (MIT).
- Whole-connectome policy training: FlyGM, arXiv 2602.17997; ConnecTorch.
- Mushroom-body plasticity (spiking mode): Bennett, Philips & Nowotny, Nat Commun 2021; Aso et al., eLife 2014; Hige et al., 2015.

## License

[MIT](LICENSE).
