# Changelog

Releases are identified by date. The Docker distribution (branch `distribution`, Hugging Face `bryceweiner/fly-trader`)
ships signed releases built from this development line.

## Unreleased

### Live money

- **Minutes are processed 2–3× faster: a minute is ready a median 2.8 s after its end, down from 6.4 s.** Measured with
  `tools/latency_probe.py` against the 2026-10-19 baseline. Figures are seconds after the minute ends:

  | | Old median | New median | Old p90 | New p90 | Old max | New max |
  |---|---|---|---|---|---|---|
  | Stream publishes the minute | 4.3 | 2.3 | 16.3 | 9.9 | 17.6 | 41.9 |
  | Selector done (the console's "latency") | 6.4 | 2.8 | 16.9 | 10.3 | 18.4 | 42.5 |
  | Publish → selector done | 0.7 | 0.5 | 1.1 | 1.0 | 4.3 | 3.7 |

  - The worst minute waited 41.9 s for PumpAPI's stream to publish it. The feed is now the remaining delay, not the
    engine (see Platform).
  - The fly's status records each minute's timing: when its rows were ready, when the fly decided and when the live
    orders were queued (`seen_s`, `scored_s`, `orders_s`). The first minutes after the change: Solana 2.42, 2.47 and
    2.48 s; Robinhood Chain 4.79, 4.84 and 4.85 s. That is about 50 ms from a minute's data to its live orders.
- **The fly decides a live minute on one pass of its teacher.**
  - It takes the selector's decision once and scores only the rows the selector would buy, the input of its learned veto.
  - Every other row's score, its learning tag and the brain view wait until after the live orders.
  - Live scoring takes about 6 ms, down from a median 94 ms (150–400 ms when the GPU was idle), with the same decisions
    on 40 real minutes.
- **Solana swaps run in parallel.**
  - Up to four swaps are in flight at once (`EXEC_THREADS`): one per token, and each token's in order.
  - Each fill is read from its own transaction, since another swap may land between a swap's two balance reads.
  - If a transaction cannot be read while other swaps are in flight, the fill is recorded as unknown and entries pause
    until `verify-fills` reconciles it.
  - Robinhood Chain swaps stay one at a time.
- **Nothing in front of a live order waits that need not.**
  - The wallet's balance is read while the minute is scored.
  - The balances before a swap are read beside Jupiter's quote and the signing, in one round trip.
  - The signer's lookups run side by side, and a swap's confirmation is checked every slot instead of every second.
- **The engine wakes the moment a minute is written,** instead of looking every 250 ms.
  - The stream checks for complete minutes every 50 ms, on its own clock.
  - Both chains run their live phases and send their boundary sales at the same time.
  - Paper work steps aside when another chain's minute lands meanwhile.
- **A failed live entry is tried again only while the fly would still buy the token, at that minute's price.**
  - Before, every retry was checked against its decision's price. Once a token ran more than 3% (`MAX_ENTRY_CHASE`),
    every retry was refused as "price ran" until the two retry minutes were up: a runner missed once was missed for good.
  - Each minute's plan now says, for every token paper holds, whether it would buy it again and what paper would pay at
    that minute's price. A retry goes out on that minute's score and against that price, or waits, recorded as
    "no longer a buy" with the score against its line. Both chains.
- **Sell now: any open live position can be sold from the Overview's open positions, on both chains.**
  - After a confirmation, the wallet sells the whole position at once, at the exit's slippage of 3% stepping up to at most
    5%, and books it like any exit. The console waits up to 90 s and says what the sale brought, or why it failed.
  - The fly's paper book keeps its own position. An operator's sale counts neither in the live-vs-paper gap that pauses
    entries nor in what the fly learns about its fills.
- **Live money follows the seat: until the fly wins its race, the wallet trades the selector's decisions.**
  - Before, with `LIVE_WITHOUT_HANDOVER` on, the wallet traded the fly's decisions whoever held the seat. Those are the
    selector's picks less what the fly's learning vetoes, so an untested fly's learning reached live money before its race.
  - Now, until the handover, a pick the fly's learning vetoes still goes live as the selector decided it. It stays out of
    the fly's paper book, which is what the race compares with the selector's. Once the fly holds the seat, its vetoes
    bind the wallet too. A pick the selector itself blocks never goes live. Both chains.

### The universe

- **Solana: pump.fun graduates whose mint does not end in "pump" are traded too.**
  - They are 4–22 % of a day's graduations (40–60 a day this week), with the same launchpad, creator history, burned pool
    liquidity and token rules as the rest. A mint-suffix test dropped them in the stream, the corpus and the wallet skill.
  - A coin now counts as pump.fun's when its mint ends in "pump" or a pump.fun migration recorded it: the stream adds a
    graduate the moment it sees the migration and reloads `corpus_meta`'s every 5 minutes. Their custom pools stay blocked.
  - On 2026-10-21 they add 1,582 mints to the day's 10,754 and 5.5 % of its volume. No existing row changes.
  - The corpus is rebuilt to include them (aggregation version 5). The current selector and fly keep trading through it.

### The selector

- **A strategy's win filter chooses only its cut-off, at the strategy's own line, and must beat having no filter.**
  - Before, the filter searched the line too. In selector #154 it moved `ev`'s line from 0.038 to 0.115, which `ev`'s live
    scores never reached, so `ev` stayed deployed but never traded.
- **The book keeps the passed strategies, and the combination rule, that together do best on the selection half.**
  - Every subset of the strategies that passed is scored as one book under each rule. A strategy the book does better
    without is dropped in the open, as its own "combination" row with the numbers.
  - The combination step no longer says a rule "made more" when no book cleared the bars.

### Training

- **A corpus rebuild never retires a model.**
  - The corpus build versions (`mature.AGG_VERSION`, `rh/corpus.RH_AGG_VERSION`) say when the corpus files are rebuilt,
    not what a model's inputs, labels or costs mean. They are no longer part of a model's definitions.
  - Before, raising one retired every selector and fly and stopped all trading until a retrain and replay, even for data
    already in the corpus. Models, flies and replay verdicts that still carry the old keys stay current.
  - A changed or missing definition still retires a model, as before.
- **A rebuild of more than three days runs in a child process at the lowest CPU priority.**
  - Re-featuring the whole archive takes about 5 hours (~100 s a day). It no longer shares the console's interpreter with
    the live minute pipeline. A day's routine build stays in the console, as before.
- **The wallet skill counts SOL-quoted legs only.** A leg quoted in another token counted its amount as SOL.
- **The fly's training writes a checkpoint after every epoch, and resumes from it.**
  - Before, a bootstrap (10–20 hours) kept its progress in memory only. A crash or a stop lost every finished epoch,
    and training could not go on past its epoch cap.
  - Now each epoch saves the weights, the optimiser, the history, the best epoch so far and the random state. A stopped
    or crashed bootstrap resumes after its last finished epoch, and a finished one can train on past the cap it ran under.
  - A checkpoint written for other training data is set aside, never resumed. The live bootstrap deletes its checkpoint
    once the fly is saved.
- **A new fly may race once its calibration week made money over 30 trades, the race's own minimum (was 100).**
  - The wallet trades the selector's decisions until the fly wins its race, so the calibration week need not prove more
    than the race will. A quiet week (2026-10-15..21) gave the new fly 88 trades at +2.96%, which the old bar refused.
  - A losing calibration week still never deploys a fly. The Kalshi fly keeps 100.

### The console

- **The fly decision inspector shows the token's price and volume in the minute the fly last scored it.** Each has a
  green or red arrow for its move over the five minutes before: the price against the close five minutes earlier, the
  volume against the five minutes' average. Both are plain decimals, never scientific notation.

### Platform

- **The archive's hour files keep every venue's SOL-quoted trades, not only pump.fun's curve and PumpSwap.**
  - Meteora DLMM and DAMM, Raydium, Orca and other launchpads' curves add about a third to the rows (~90 MB an hour).
    The lifecycle events keep the quote mint. Both files are written atomically and the trades file says what it keeps.
  - `fly-trader refetch-archive` downloads the older hours again in its own process, 2 at a time (`REFETCH_PARALLEL`).
    It starts no download while the live stream is more than 15 s behind (`REFETCH_MAX_LAG_S`), leaves the newest 6 hours
    to the console, and rebuilds each finished day itself. About 4,500 hours, roughly 2.7 TB of downloads; run it under
    `nice -n 19` only after the console runs this code. Progress: `ui_settings['replay_refetch_status']`.
  - A day is rebuilt whenever one of its hour files is newer than its aggregate.
- **A stream relay is ready, but switched off.**
  - `stream_relay/` holds PumpAPI's stream on a server near its source. It forwards only the events and fields the
    minutes use: about 33 KB/s at the busiest minute, instead of the whole stream.
  - It serves them over TLS with a token, and resumes without loss after a dropped connection.
  - With `SOL_FEED=relay` it becomes the minutes' clock, and this machine's own PumpAPI connection stays as its backup.
  - Setup is in `stream_relay/README.md`. `fly-trader relay-check` tests a relay without switching anything.
  - New settings: `SOL_FEED` (default `pumpapi`), `STREAM_RELAY_URL`, `STREAM_RELAY_TOKEN` and `STREAM_RELAY_CA_FILE`.
- **A Helius decoder is built, but switched off** (`SOL_FEED=helius`).
  - It turns pump.fun and PumpSwap transactions into PumpAPI's events exactly as the archive has them: 312 of 312 events
    per sampled slot, and trades, creates and migrations field for field.
  - Helius's websockets deliver only about 1 MB/s in all to this machine, and drop trades when the market is busy.

### Tools

- **`tools/latency_probe.py` and `tools/latency_report.py` measure the minute pipeline from outside the console.**
  - They sample, four times a second, when the stream publishes each minute, what SQL the console is running, when each
    book's status lands, the console's CPU and every live order. The report turns that into per-minute timelines.
  - They're read-only and need no restart. Run them before and after a change to the minute pipeline.
  - The 2026-10-19 baseline of the old code is in `logs/latency_baseline_20261019/`. A minute was ready a median 6.4 s
    after its end (p90 16.9 s). Almost all of that was the wait for the stream to publish the minute; everything after
    publishing took 0.7 s.
- **`tools/strategy_study/` re-checks the trading strategy against fresh data without saving a model.**
  - It runs the trainer's own capitulation fit and judges it on the trainer's selection half, evaluation half and holdout.
  - Then it compares the live selector's decisions on days after its training, sells the same trades after every hold
    from 10 to 480 minutes, and tests floors on liquidity, volume, activity, age and wash share.
  - It is safe to run beside live trading. The dataset is built in a separate process and read from disk, and a guard
    stops the run if the live console's minutes slow down or its memory is squeezed.
  - The 2026-10-19 run kept selector #154's strategy and 240-minute hold. #154 made +8.6% per Solana trade on the 16 days
    after its training, against +8.9% in its backtest. Holds from 180 to 480 minutes did about equally well, and no
    floor beat the noise.

## 2026-10-18, release 9: The Overview page charts each chain's setups, and the money's book comes first

Published to Hugging Face as signed release 9. Fly #157 and selector #154 are unchanged, and so are the definitions they
were judged on: no model, seed or infrastructure file changes.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 9`.

### Console

- **The Overview page charts the selector's setups for the last 30 days, per chain.**
  - A setup is a buying chance: a token the selector picked while not holding it. Picks less than 30 minutes apart
    count once.
  - Each UTC day shows its setups beside the trades the selector took. A setup it did not trade was vetoed (dump veto,
    win filter).
  - Solana and Robinhood Chain sit side by side, so a quiet market on one chain shows against the other.
  - A day the selector book did not run, or ran only part of, is flagged. Few setups on such a day say nothing about
    the market.
- **The book's wealth and open positions sit near the top of the Overview page,** right below the summary and above the
  fly's brain.

## 2026-10-18, release 8: The live stats follow each live order, and empty token accounts close on their own

Published to Hugging Face as signed release 8. Fly #157 and selector #154 are unchanged, and so are the definitions they
were judged on: no model, seed or infrastructure file changes.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 8`.
- **Rent returned by an earlier `close-empty-atas` run** is in live wealth but not in realized P&L. Closes from this
  release on are booked.

### Live money

- **The live stats follow each live order.**
  - Once a live order is booked and no other waits, the wallet is marked at once and the console's panels update then.
  - Before, the mark waited for the engine's next idle moment, and the race card refreshed only every 30 seconds.
- **Empty token accounts close on their own, and their rent is booked.**
  - A sale that empties a token account no longer closes it on the spot, which held up the next live order.
  - The account now closes the same minute: mid-minute, on a thread of its own, while no order is waiting. The wallet
    is also checked for empty accounts at start and every hour, so none is left holding rent.
  - If an order arrives during a close, the order goes first. The close is booked once it lands. Until then, swaps
    read their fills from their own transactions, so the returning rent never counts as part of a trade.
  - Every close credits the books: the rent a first buy set aside comes back, and the close's fee goes to the sale
    that emptied the account. If the rent was paid before deposits were tracked, the whole refund goes to that sale.
  - Before, `close-empty-atas` and the vault's close only marked deposits as returned. The rent from accounts opened
    before deposits were tracked reached the wallet, but no trade was credited with it.
  - `fly-trader close-empty-atas` now leaves the live book's held tokens alone and prints what it booked.

## 2026-10-18, release 7: A live buy costs what its paper twin does, and the fly learns again

Published to Hugging Face as signed release 7. Fly #157 and selector #154 are unchanged, and so are the definitions they
were judged on: no model, seed or infrastructure file changes.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 7`.
- **New columns are added on start:** `deposit_sol` and `deposit_refunded` on `positions`.
- **If your fly's learning froze** (the console's model page says learning is frozen), lift it with Resume learning once
  this release runs.

### Live money

- **A live buy costs what its paper twin does.**
  - The ~0.0015 SOL that a first buy of a token pays to open its token account is a deposit. The account's close
    returns it.
  - It was booked as part of the buy, so live buys looked 1–3 % bigger than paper and their entry prices 1–3 % higher.
  - The deposit is now kept apart from the buy and counted in live wealth until the close returns it. Only the close's
    fee is the position's.

### The fly

- **It learns again, and one strategy's trouble no longer stops the other's learning.**
  - On 2026-10-13 the rollback checks froze all of the fly's learning after rolling ev back three times in three days.
  - ev cannot learn (its learning rate is 0), so there was nothing to roll back. The IC check had tripped only because
    the selector's own ev ranking had a bad day (IC −0.10 to −0.13).
  - Capitulation, which was learning, stopped too. Five days of decay then erased what it had learned, and the
    re-bootstrap the freeze asked for never ran.
  - The checks now judge only what learning did. A strategy that has learned nothing trips none. The IC check compares
    the fly with its frozen copy on the same rows instead of with zero, and the replays count it the same way.
  - Reaching the rollback limit freezes that strategy's learning only.
- **One fly's operator commands reach it.**
  - The Solana and Robinhood Chain books share one memory. Before, whichever book's minute ran first took a pause,
    resume or rollback, and a resume could reach only the book that did not need it.
  - The Robinhood Chain book also kept learning into the shared memory while the Solana book was frozen.
  - Only the Solana book takes these commands now, and the Robinhood Chain book's learning follows its pause and freezes.

## 2026-10-18, release 6: A restart never holds a sale back, and Kalshi orders go first

Published to Hugging Face as signed release 6. Fly #157 and selector #154 are unchanged, and so are the definitions they
were judged on: no model, seed or infrastructure file changes.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 6`.

### Live money

- **A restart no longer holds a sale back.**
  - The minute-boundary clock now starts with the trading engine. Before, it waited for the books to load and the
    engine to warm up, about four minutes after a restart or an update.
  - On start it sells at once anything that came due while the console was down.
  - Each chain has one live mirror per process, made by whichever comes first: the clock, or the fly's first live
    minute. A position without a hold of its own waits for the fly to load.
- **Kalshi orders go first too** (Kalshi live trading is off by default).
  - The fly's taker buys, maker posts and cancels, and a kill switch's liquidation go out the moment it has decided.
  - Everything else comes after: the paper books' new positions and orders, their marks, the exchange's settlements
    and fills, learning, calibration and state.
  - A paper order the plan cancels cancels its live twin in the same step.
  - The subaccount's cash is read before the orders only when there are taker buys.
  - A kill switch tripped by a minute's wealth mark liquidates with the next minute's orders.
  - The Kalshi feed announces a written minute at once, and the engine no longer waits 4 seconds for it.

## 2026-10-17, release 5: Live orders go first, and live trades what paper trades

Published to Hugging Face as signed release 5. Fly #157 and selector #154 are unchanged, and so are the definitions they
were judged on: no model or seed file changes.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds it until `fly_update.py approve 5`. The release changes
  infrastructure files: comments in `Dockerfile` and `docker-compose.yml`, the new `docker-compose.rocm.yml`, and the
  updater itself.
- **New columns are added on start:** `fee_v` on the Robinhood Chain swap and minute tables, `hook_fee_v` on its pools,
  and `outcome`, `fill` and `fill_score` on `fly_scored`.
- **Robinhood Chain installs:** TSLA-quoted coins were valued about 2,430× too high since 2026-09-05. TSLA's reference
  path ended at USDG, so its USD price was read as ETH.
  - The price scan now finds again any stored path that stops short of WETH, on its first pass, and prices these coins
    in ETH from then on.
  - Their minutes from before keep the USD scale. The engine treats the jump as a scale break and leaves each of these
    coins out until it restarts more than a day after the change.
- **New settings,** all with defaults: `MAX_ENTRY_CHASE` (0.03), `ENTRY_RETRY_MIN` (2), `ENTRY_MAX_LAG_S` (90),
  `CIRCUIT_COOLDOWN_S` (900), `FLY_OUTCOME_FROM_BOOK` (on), `FLY_FILL_PATHWAY` (on), `FLY_FILL_ALPHA` (0.02),
  `FLY_FILL_HALF_LIFE_DAYS` (14), `FLY_FILL_RANGE` (0.05), `RH_TX_FEE_MULTIHOP_ETH` and `RH_HOP_FEE`.

### Live money

- **Live orders go first, on both chains.**
  - The wallet's scheduled sales go out the moment a minute ends, from a thread of their own, before the feed has
    written the minute.
  - Its buys go out as soon as the fly has scored the minute.
  - Paper trades, the fly's learning, the selector's minute, the wealth mark and all other bookkeeping come after.
  - Before this release, Solana orders left 9–27 s after their minute ended and Robinhood Chain orders 5–39 s (once 207 s).
    A sale's booking also waited for the paper work to finish.
  - The Solana feed now announces a written minute at once instead of on its 5-second status, and the engine looks for
    it every 0.25 s.
- **Live trades what paper trades, when paper trades it.** These fixes come from a live-against-paper audit on
  2026-10-17:
  - A live position's hold now runs from its decision's minute, not from its fill. Live had sold 74–90 s after paper on
    every trade, once into a dump: +13 % on paper, −1.4 % live.
  - A Solana sale that empties its token account closes it, and the ~0.0015 SOL of rent comes back onto the position.
  - A buy is refused before signing when its quote is more than `MAX_ENTRY_CHASE` (3 %) above its paper fill.
  - A buy that failed without moving anything is retried, re-sized, for `ENTRY_RETRY_MIN` (2) minutes while paper still
    holds the token.
  - The circuit breaker re-arms `CIRCUIT_COOLDOWN_S` (15 minutes) after its last failure. Robinhood Chain failures
    count on its own circuit.
  - No book buys on a minute traded more than `ENTRY_MAX_LAG_S` (90 s) after it ended.
  - A sale or a mark in a quiet minute pays the coin's last known pool fee; Robinhood Chain coins quoted in another
    asset pay their multi-hop route.
  - Every paper entry the wallet did not take is recorded with the reason.
  - Solana live positions show their strategy and hold.
- **Robinhood Chain orders:**
  - A swap whose receipt timed out is settled by recovery, never failed and sent again, so it cannot buy twice.
  - Buys climb the slippage ladder like sales.
- **The kill switch** now trips from the wealth mark taken after the minute's orders. With `KILL_SWITCH_LIQUIDATE`, its
  sales start at the next minute boundary.
- **Fixed:** a test run could still reach Jupiter's `/execute`, which broadcasts a signed swap itself. It now refuses
  under a test run like every other send path. The test configuration also blanks the signer socket, and the database
  guard applies from its first line.

### The fly

- **It learns from outcomes and fills as mushroom-body pathways.**
  - Its outcome pathway learns what each traded row made in its own book; for a row it did not trade, it learns the
    market label.
  - A fill pathway per strategy learns what the wallet's real fills added (live minus paper) once the live position
    closes.
  - Learned bad fills can veto a teacher trade.
  - A fill pathway whose predictions run against the fills over 7 days is reset.
- **It scores a minute before it learns from it,** so what it learns in a minute applies from the next one.

### Robinhood Chain

- **Quote assets are priced in ETH again:** a reference path must reach WETH, and a route split across several hops is
  followed to the end.
- **Hook fees are stored as taken / gross:** the hook takes its fee out of the pool's gross output, so a 1 % fee had been
  stored as 0.0099. Rows stored the old way are converted when read.

### The console

- **Trades by hour of day** on the Overview: a colourful 24-hour wheel per chain (Solana left, Robinhood Chain right),
  Highcharts' variable-radius pie. Each hour is an equal slice and its radius is the trades entered in that hour, every
  trade ever (the positions table plus every reset's archive), all books or live only, in UTC or local time.
- **Bar sparklines stand on zero:** a losing day hangs below the line and a quiet day is a dash.

### Platform

- **AMD GPUs through ROCm:**
  - `DEVICE=rocm[:N]` drives the GPU through a ROCm build of torch; `uv sync --extra rocm72` installs torch
    2.14.1+rocm7.2.
  - `docker-compose.rocm.yml` builds that torch into the image and gives the container `/dev/kfd` and `/dev/dri`.
  - It has not yet run on AMD hardware.
- **Fixed:** the replay assembler failed on a missing dependency. `pytz` is now declared.

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
- **Fixed, after it burned 0.06 ETH:** a test run could spend real money. While `.env` had `RH_LIVE_ENABLED=1` and the
  real RH key, a vault test that expected signing to be off paid two test claims of 0.03 ETH from the live RH wallet to its
  placeholder address `0x1111…1111` (transactions `0x063cbd4f…` and `0x5fd55d3d…`). Now `fly_trader/netguard.py` refuses
  every Solana or EVM broadcast and every Kalshi order, cancel or transfer from a test run unless it goes to a local test
  chain. Tests never load the operator's signing keys or live switches.
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
