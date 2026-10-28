# Changelog

Releases are identified by date. The Docker distribution (branch `distribution`, Hugging Face `bryceweiner/fly-trader`)
ships signed releases built from this development line.

## 2026-10-29, release 15: A coin whose creator owns most of its supply is never traded

Release 15 changes which coins the models may trade. The models are unchanged: selector #640 and fly #641, as in releases
12 to 14. The rule has run on the development install since 2026-10-29 01:02 (UTC+8).

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 15`. The change reaches live trading: which coins are bought.
- **Positions open when the release installs are held as before:** the rule decides buys only.

### Trading

- **A coin whose creator owns most of its supply is never traded.** On the night of 2026-10-28 the live book lost about
  0.2 SOL on two coins named XTO, bought at 21:29 and 23:46 (UTC+8). Each was launched by a wallet that bought the whole
  bonding curve in the launch transaction (79 % of the supply), which graduated it the same second. Release 13 still
  traded such coins up to 8 h after graduation, at 0.1 SOL a buy. The first was drained 7.9 h after graduating, 111
  minutes after the buy; the second had already fallen 94 % when it was bought, and was drained 6 minutes later.
  - **The rule:** a coin is out of the models' universe when its creator's launch buy, or what the creator held when the
    coin graduated, was more than 70 % of the supply. It applies to every live and paper book on both chains, and to
    every future fit (`train/selector.in_universe`; `UNIVERSE_VERSION` creator-2 refits the walk-forward caches).
  - **Why creator history did not catch them:** XTO is a factory. Since 2026-10-14, 77 coins named XTO were launched
    exactly this way from 73 wallets, so the serial-rugger rule, which follows a wallet, saw almost none of them.
  - **Cost, measured on a close set:** on selector #640's out-of-sample book, leaving out every self-graduated coin
    (creator launch buy of at least half the supply, graduation within a minute) cost about 40 of 2,453 log points of
    growth (2026-10-28 study). This exact rule was not measured.
  - In the 14 days to 2026-10-29, 3,568 of 14,083 graduates (25 %) had a creator launch buy above 70 %, and 170 more a
    creator holding above 70 % at graduation.
  - The 0.1 SOL cap for these coins (`CREATOR_HEAVY_CAP_SOL`) no longer comes into play.

## 2026-10-28, release 14: A coin bought far below its high is sold once it doubles

Release 14 changes when some positions are sold. The models are unchanged: selector #640 and fly #641, as in releases 12
and 13. The rule has run on the development install since 2026-10-28 18:31 (UTC+8); it has not sold a position there yet.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 14`. The change reaches live trading: some positions are sold before their hold ends.
- **Positions open when the release installs keep their full hold:** a position gets its take-profit line when it is
  bought.

### Trading

- **A coin bought far below its high is sold once it doubles, instead of being held the full hold.** On 2026-10-28 the
  live book bought Darla 85 % below the high it had reached after graduation. It rose 146 % within the hour, then faded
  to +69 % while held.
  - **The rule:** a buy of a coin trading at least 50 % below its highest minute close since graduation gets a
    take-profit line at twice its entry price. The position is sold at the first minute that closes at or above the
    line; otherwise it is held its hold as before. Its token is not bought again until that hold would have ended.
  - **Why:** on selector #640's out-of-sample book, held to the same 240 minutes, the rule made more than holding in every
    period: on Solana 10 %, 31 % and 34 % more in the selection, evaluation and holdout periods, with every month from
    June on ahead and May behind; on Robinhood Chain 23 % and 9 % more in the evaluation and holdout periods.
  - **What did not pass:** the same take-profit at +50 %, a trailing sale 20 % below the peak after +50 %, and an exit
    model trained on every held minute.
  - **Where:** every paper and live book on both chains (`TAKE_PROFIT_X`, `TAKE_PROFIT_DIP`, `TAKE_PROFIT_CHAINS`). A live
    take-profit sale goes out the moment the engine reads the minute's closes, before the minute's features and the fly's
    scores (agent/live_first.chain_take_profits). The line is stored on the entry decision, so a live position carries
    its paper entry's line. Sales show as `take_profit` exits.
  - **Known limit:** the high is the highest close the install has recorded. The Solana stream keeps 7 days of minutes,
    so a coin older than that, or older than the install, is judged on its high since then.

## 2026-10-28, release 13: Coins whose creator bought most of the supply are traded briefly and small, serial ruggers' never

Release 13 changes which coins the models may trade and how much a risky one gets. The models are unchanged: selector #640
and fly #641, as in release 12. The rules have traded the development install since 2026-10-28 02:21 (UTC+8).

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 13`. The change reaches live trading: which coins are bought, and the size of some buys.

### Trading

- **Coins whose creator bought most of the supply at launch are traded only briefly and small, and a serial rugger's
  coins not at all.** On 2026-10-27 the live book lost 0.37 SOL on USDP. Its creator bought 79 % of the supply in the
  launch transaction, which graduated it the same second. It was drained 13.7 h later, 97 minutes after the buy.
  - **A creator-heavy coin** is one whose creator's launch buy was more than 70 % of the supply. Its position must be
    sold before the coin is 12 h old, so it is bought only up to 8 h after graduation, before a 4-hour hold. A buy stakes
    at most 0.1 SOL (on Robinhood Chain, 0.1 SOL's worth of ETH), never more than its Kelly size.
    - Why: of the coins whose creator bought at least half the supply at launch and that still stood 6 h after
      graduation, 82 % later fell 90 % within an hour, a median 24 h after graduation. Other graduates: 24 %.
  - **A serial rugger** has at least one earlier graduation with a known outcome, and at least half of those fell 90 %
    within an hour of graduating. Its coins are never traded.
    - Why: it was the only one of 16 rug signals, fixed before the test, whose exclusion raised selector #640's
      out-of-sample growth in every period.
  - Both rules belong to the models' universe (`train/selector.in_universe`): live, paper, both chains and every future
    fit. The cap is `CREATOR_HEAVY_CAP_SOL` (agent/sizing.py).
  - On #640's out-of-sample book (2026-05-10..10-22, quarter-Kelly sizing), the rules together give +39 log points of
    growth on +2,453, with 129 trades that lost 80 % or more instead of 158. But they cost growth from 05-10 to 07-21
    (−51) and from 10-02 to 10-22 (−61), and gain it from 07-21 to 10-01 (+152).
  - The walk-forward caches are keyed by the universe now, so a refit on the same corpus does not reuse walk-forwards
    fitted on the old one.

## 2026-10-26, release 12: `ev` trades beside `capitulation` again: selector #640 and fly #641

Release 12 ships selector #640 and fly #641 in place of release 11's #595 and #604. It is again a matched pair, trained on
the same days and rows (2026-04-19..10-22). The pair has traded the development install since 2026-10-26 19:25 (UTC+8):

- **Selector #640** trades `ev` and `capitulation`, both with 240-minute holds. On Solana its correctness bands size both.
  - #595 had dropped `ev`, because its book was chosen on the oldest half of the walk-forward (The selector, below).
- **Out of sample:** on the 21 days 2026-10-02..10-22 its holdout-blind copy made +7.69 % per trade over 704 trades (59 %
  winners, profit factor 1.78; random picks -3.41 %). #595 made +6.65 % over 485 trades on the same days: 54.1 in
  total against 32.2.
  - On Solana it made 353 trades at +7.90 % (#595: 215 at +4.96 %), on Robinhood Chain 351 at +7.48 % (#595: 270 at
    +7.99 %).
- **Fly #641** is taught by #640's full fit on exactly its fit rows. It ranks the rows as its teacher does (rank
  correlation 0.89), and its lines are matched to #640's own trade count.
  - Its evidence is the blind selector's own holdout picks (704 trades, +7.69 %).
  - Like every new fly, it races 14 days before it can take the seat. Until then the wallet trades #640's decisions.

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 12`. The only code change is in training (below); live trading runs as in release 11.

### The selector

- **The book is chosen on the walk-forward's last four weeks, not on its oldest half (memecoins).**
  - Each strategy must still pass both halves on its own. Which of the passing strategies trade together, and under which
    rule, is now decided on the walk-forward's last `RECENT_BOOK_DAYS` (28), because that choice moves with the market
    (Bryce, 2026-10-25).
  - On the 10-22 corpus the selection half (May–July) kept capitulation alone, and selector #595 traded it alone. ev
    passed on its own (+4.4 % and +5.0 % a trade on the two halves), but with one position per token it buys a token
    first and blocks capitulation's later entry: 1,331 of capitulation's trades, worth +136, on the selection half.
  - From mid-September the pair made as much as capitulation alone or more. On the 21 held-out days, from blind refits,
    the pair made 54.9 over 725 trades (+7.6 % a trade) against capitulation's 32.2 over 485 (+6.65 %).
  - Kalshi still chooses its book on the selection half.

## 2026-10-25, release 11: The matched pair, selector #595 and fly #604, and a one-command install

Release 11 ships selector #595 and fly #604 in place of release 10's #560 and #561. The two were trained as a matched pair,
on exactly the same data. #595 has traded the development install since 2026-10-25 00:07 (UTC+8):

- **Fly #604** was trained beside #595, on the same days and rows (2026-04-19..10-22), and is taught by #595's full fit.
  - Its line is matched to #595's own trade count: 322 Solana trades over the last 30 training days.
  - It ranks tokens as its teacher does (rank correlation 0.96).
  - Its evidence is the blind selector's own holdout picks (485 trades, +6.65 %). At bootstrap it trades exactly the
    selector's picks, and its live race judges its learning.
- **Selector #595** comes from the first matched retrain (`fly-trader matched-pair`, Training below) and is fitted through
  2026-10-22. It trades `capitulation` alone with 240-minute holds, and on Solana its correctness bands size it.
- **Out of sample:** on the 21 days 2026-10-02..10-22 its holdout-blind copy made +6.65 % per trade over 485 trades (55 %
  winners, profit factor 1.61; random picks -4.06 %). On Solana that was 215 trades at +4.96 %; on Robinhood Chain, 270
  at +7.99 %.
- **Over 10-18..10-22,** replayed with the live engine's real outages, stale minutes and entry failures, the blind copy
  made +1.69 SOL and +0.031 ETH. The same replay gives #154, which traded live then, +0.39 SOL and +0.0099 ETH. Live
  itself made +0.25 SOL and +0.0095 ETH.
- **It stakes more per trade:** its Solana buys in that replay averaged 0.42 SOL against #154's 0.14.
- **The wallet trades its decisions from the deploy,** because fly #561's gate now follows the deployed selector (Live
  money below).

### Live money

- **Buys that Jupiter routes through another token are signed again.**
  - Since 2026-10-23 16:29 Jupiter routes some buys through an intermediate token, for example SOL → USDC → token or
    SOL → PUMP → token.
  - Such a transaction opens the wallet's own account for the intermediate token, about 0.0015 SOL of rent per hop.
  - The signer left that account out of its simulation, so the rent counted as spending beyond the order. It refused
    every such buy as overspending ("the swap spends X lamports, more than the Y ordered").
  - The wallet missed 4 of the fly's picks this way, 3 of them on 10-24 (out of 12 picks that day). The best of them
    closed +56 % on paper.
  - Now the accounts the transaction's ATA instructions open for the wallet are in the simulation. Their rent stays the
    wallet's. Only token accounts the wallet owns count.
  - Re-simulated, the refused 10-23 and 10-24 routes pass; with the change switched off they are refused to the lamport.
  - The rent is not the trade's either. The fill, its price and the position's cost leave it out, as they leave out the
    traded token's account rent, and the mirror closes the empty account mid-minute to get it back.
- **A new fly never inherits the old fly's seat: it races first.**
  - A new bootstrap kept the old fly's race and its seat. On an install whose fly had won its race, a release that
    installed a new fly handed it live money at once (release 10's fly #561 where #157 held the seat).
  - Now a fly other than the one its chain's saved state was learned under starts its own race at the next minute, and
    that chain's seat goes back to the selector. A seat recorded for another fly is cleared when the fly starts (an
    install that ran release 10). A seat set by hand, naming no fly, stays.
  - An untested fly races `HANDOVER_DAYS` (14) before it trades live (Bryce, 2026-10-23).
- **Every corpus build runs in a child process at the lowest CPU priority, not only big rebuilds.**
  - Only more than three stale aggregates went to the child. A restart that killed one after its aggregates but before
    its features left 20 days to re-feature in the console's own interpreter: ~80 s a day, 16 GB and a whole core, for
    half an hour (2026-10-23).
  - A routine day's build (~80 s) no longer runs beside the live minute pipeline either, and neither does corpus_meta's
    rebuild, which the child now runs before its builds: 36-82 s every 30 minutes until now.
- **On Solana a strategy with a correctness model sizes by it: the EV line screens, P(clean) sets the Kelly band.**
  - Inside the EV line's picks the EV margin carried no information: its rank correlation with the outcome was +0.00 /
    +0.08 / +0.01 on the selection, evaluation and holdout periods, and #560's margin bands were flat to inverted
    (the 2026-10-23 study). P(clean) ranked the same picks at +0.14 to +0.24 on Solana.
  - A row that clears its strategy's line is sized by the band of its P(clean) above the bands' floor (``p0``, the lowest
    P(clean) among the selection half's trades), with the same Kelly per band, `KELLY_FRACTION` and caps as before.
  - Each band keeps its measured Kelly, and a band with no edge gets 0. A pick below the first band with an edge is blocked
    ("correctness N % below M % (no band with an edge below it)").
  - Robinhood Chain keeps its EV bands (neutral in the study); its rows still carry their P(clean).
  - The live mirrors size the wallet's buys, and its retries, at the same margin; the fly's gated buys are sized as the
    selector sizes them. A model without a correctness model (#560) trades exactly as before.
- **The fly's gate follows the deployed selector, not only the one that taught it.**
  - Every fly buy, the wallet's included, needs the selector's decision on the row. That decision came from the selector
    that taught the fly, so a newer selector traded only its own paper book until a fly trained on it was deployed.
  - Now the gate is the selector the selector book trades: its pin, else the newest deployable snapshot. The fly picks it
    up at its reload check, mid-minute, never in front of a minute's orders. Other chains' books follow the Solana book.
  - Weekly refits that win deploy themselves (Bryce, 2026-10-24), so the wallet trades a winning selector from its
    deploy, not from its matched fly's, which trains for about ten hours more.
  - A gated buy holds for the selector's own hold time. The fly's network, its learned correction, its race and its
    labels stay as they were.
  - A selector that fails to load, or wants an input live lacks, leaves the gate where it was. Strategies the fly has no
    network for, and holds that differ from the fly's, are logged.

### The selector

- **A per-strategy correctness model: P(clean), trained on all of the strategy's candidates.**
  - Clean: net ≥ +5 % over the strategy's hold, and no checkpoint of the hold grid up to it below -15 %.
  - Walk-forward on every candidate of the strategy (its base mask and fitted trigger in the universe, whatever its EV
    score), from the strategy's own inputs; cached like the strategies' walk-forwards.
  - Its Kelly bands are built on the selection half's trades at the strategy's line, each with its measured Kelly. It is
    kept when its bands grow the bankroll faster than the EV margin's bands on the strategy's evaluation-half trades
    (Kelly growth per trade on the same trades), and the book still passes there. Otherwise the strategy keeps its EV
    bands.
  - The first design forced the lowest band to Kelly 0 and judged that gate by its fixed-size total. In the 2026-10-24
    retrain that band's 787 selection-half trades had made +6.06 % each, so the gate failed, and with it the sizing,
    which grew the evaluation half +30.6 bp a trade against +17.1 bp with EV bands.
  - Memecoins only; Kalshi's stack is unchanged.
- **The random-removal test clips each trade to -100 % … +300 %.** Unclipped, one Robinhood Chain moonshot (+226,678 %)
  decided it: #560's capitulation win filter lost, filtered 377 against random 2,524.
- **An evaluation half confirms a component on 30 trades, the race's own minimum, not 100.** The selection half keeps
  100, and the profit-factor bar is unchanged. Held to 100, the wallet-skill inputs were dropped at 87 evaluation trades
  and a profit factor of 1.98, after 472 selection trades at 1.51. Kalshi's significance gate keeps its own minimum.

### The console

- **Correctness %: every decision records its strategy's P(clean)** (`correctness` in the decision's detail), shown in the
  decision inspector and the Overview's open positions.

### Training

- **A selector and a fly can be trained as a matched pair on exactly the same data (opt-in: `fly-trader matched-pair`).**
  - Until now they never saw the same days. The fly held out its own calibration week and a purge day, and learned from
    the deployed selector's holdout-blind refit: release 10's selector #560 was fit through 10-20, its blind refit
    through 09-29, and fly #561 through 10-13.
  - In matched mode the fly's teacher is the stack itself: its holdout-blind fit, to be scored on the holdout, or its full
    fit, to be deployed. The fly learns on exactly that fit's days and rows (each strategy only where its own label
    existed), with no calibration week or purge day of its own.
  - Its lines are matched to the teacher's own trade count on those same rows. The count needs no outcome, so the 21
    held-out days stay unseen by both. `--match-days N` matches on the last N training days only: scoring every training
    candidate costs about one pass of the network over them.
  - The command pins the corpus's last day and writes only to its scratch folder: the stack with its full and blind fits,
    the blind fly, an out-of-sample report, optionally the blind fly's learning replayed over the holdout (`--replay`),
    then the full fly. No registry row is written, and a rerun resumes. The report covers both books per strategy and per
    chain: trades, win rate, mean and total net return, and random picks, with the selector's own costs and hold rules.
  - A matched fly may race on its own out-of-sample picks (at least `HANDOVER_MIN_TRADES` trades, a positive mean)
    instead of a calibration week. At bootstrap the live fly (teacher score, teacher gate) trades the selector's own picks.
  - The default bootstrap, every model definition and the live fly and selector are unchanged.
  - `decisions.build` can pin its last day, and frees its per-day frames before the sort: a ~52 GB peak instead of ~78.
  - The report adds the selector book's Kelly growth per trade at its own bands, and each strategy's correctness verdict
    with its bands.
  - `--no-blind-fly` skips the test fly, about 19 hours of training. The report then scores the selector's book alone,
    and the full fly's evidence is the blind selector's holdout picks: at bootstrap the live fly trades exactly those, and
    its learning is judged by its live race. Its first run died after saving the full fly, before attaching that evidence
    (it read the freed test-fly variable); the evidence's source now comes from the report itself.

### Install

- **One command installs the Docker distribution: `./install.sh`** (Linux, macOS, Windows under WSL 2). It asks for the
  two API keys, checks each with its provider, and offers an NVIDIA GPU when Docker can reach one. It then builds and starts the containers, waits until
  the trader is healthy, creates the bot wallet, restarts the trader so it loads the key, and opens the dashboard. A
  rerun keeps the keys, the wallet and the data. On WSL it refuses a folder on a Windows drive.
- **A recreated container runs its installed release again, not the checkout its image was built from.**
  - The updater installs each release into the container, but records it on the data volume.
  - After `docker compose up --force-recreate`, or `down` then `up`, the container ran the code its image was built from.
    The updater still called that current and never installed the release again.
  - At boot it now compares the tree's signed `release.json` with the recorded manifest. When they differ, it installs
    the recorded release, or a newer one, before the console starts.
  - The installed release's verified copy stays on the data volume, so this fetches only the small signed manifest from
    Hugging Face.
  - Only that exact manifest may come again under the same release number; anything older is still refused.
  - With `AUTO_UPDATE=1`, a container rebuilt from a newer `git pull` also runs the latest signed release.
- **Windows has its own guide, [INSTALL.md](INSTALL.md):** WSL 2, Docker Desktop and its memory, then `install.sh`, the
  `.env` backup (not into OneDrive), sleep and sign-in after restarts, the NVIDIA GPU through Docker Desktop (the Windows
  driver only), and the dashboard from another device.
- **The dashboard can be opened from other devices on your network** (opt-in): `CONSOLE_PUBLISH=0.0.0.0` publishes it
  beyond 127.0.0.1, and `CONSOLE_ALLOWED_HOSTS` names the addresses the browser may use. It still has no login.
- **Line endings are LF on every OS** (`.gitattributes`): a Git for Windows clone gave `docker/*.sh` CRLF endings, and the
  container did not start.
- **The docs apply a changed setting with `docker compose up -d --force-recreate fly`.** `docker compose restart` keeps
  the values `.env` had when the container was created, because the process environment wins over `.env`. So "edit
  `.env`, then restart" changed nothing for any setting already in the file.

## 2026-10-23, release 10: A retrained selector trades `ev` again, and live money follows the seat

Published to Hugging Face as signed release 10. It ships selector #560 and fly #561 in place of selector #154 and fly
#157:

- **Selector #560** is fitted through 2026-10-20 on this release's rules and trades `ev` again besides `capitulation`. On
  the 18 days 2026-10-03..10-20, out of sample for both, its holdout-blind copy made +9.71 % per trade over 393 trades
  (62 % winners, profit factor 2.08); #154 made +9.64 % over 392 (58 %, 1.95).
- **Fly #561** is taught by it. Its own decisions reach the wallet only once it wins its 14-day race (Live money below).

### Upgrading

- **Nothing to do on a Docker install:** its updater installs this release within the hour, as usual.
- **A server on the host updater** (`deploy/fly_update.py`) holds this code release for 24 hours, or applies it at once
  with `fly_update.py approve 10`.
- **The corpus is rebuilt once on start,** because the Solana universe changed: about 5 hours in a child process at the
  lowest CPU priority. The rebuild retires no model.
- **New tables and columns are added on start:** `corpus_meta.grad_quote`; on Robinhood Chain, `rh_v4_keys`,
  `rh_v4_tokens` and new `rh_pools` and `rh_meta` columns.
- **Robinhood Chain installs:** run `fly-trader rh-v4-backfill` once, under `nice -n 19` and apart from the console, so
  v4 pools older than the console's watch are found.
- **New settings:** `EXEC_THREADS` (default 4), `SOL_FEED` (default `pumpapi`), `STREAM_RELAY_URL`, `STREAM_RELAY_TOKEN`,
  `STREAM_RELAY_CA_FILE`, `HELIUS_FEED_COMMITMENT`, `REFETCH_PARALLEL`, `REFETCH_MAX_LAG_S` and `REFETCH_SKIP_RECENT_H`.
- **On an install whose fly already won its race, fly #561 inherits that seat** and trades live without racing first.
  The next release makes every new fly race.

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
- **A Robinhood Chain minute is written in milliseconds at any hour, not up to 2.5 s late in the UTC day.**
  - Writing each minute re-read every swap since 00:00 UTC to warm the price filter (the median of a coin's previous 200
    prices that day): about 130,000 swaps and 2.1–2.5 s by 23:59, in front of that minute's live entries.
  - It now reads only each coin's last 200 prices that day: 0.03–0.07 s at 23:59. On 18 real ranges (single minutes at
    23:59, noon and now, 240 minutes, across midnight, a whole day, and four quote assets) the rows are identical.
- **The Robinhood Chain corpus is built in a child process, never in the stream's thread.**
  - At 00:00 UTC the stream built the previous day's training files itself: about 30 s on 2026-10-23, after that minute's
    write and before the next one's. The other v4 pools (The universe) add several times the swaps.
  - The stream now only looks at the files and starts `fly-trader rh-build-corpus` at the lowest CPU priority, one at a
    time.
- **Two reads the latency probe caught after the 2026-10-23 restart are cheap now, and off the live path.**
  - The Solana stream reloads the blocked custom pools every 5 minutes on the loop that flushes its minutes. Since the
    suffix-less graduates, that read probed `corpus_meta` once per pool row: up to 3.5 s. It now joins the two tables
    (the same 32,308 pools, 1.9 s → 65 ms on a copy) and runs on a thread.
  - `build-mature` read the same pools for every day of a rebuild; it now reads them once per round.
  - The Robinhood Chain corpus read every coin's median fee over all of `rh_minutes` (13 s) for every day it built. It now
    reads them once per run, in the middle of a minute, and a run builds up to 31 days.

### The universe

- **Solana: pump.fun graduates whose mint does not end in "pump" are traded too.**
  - They are 4–22 % of a day's graduations (40–60 a day this week), with the same launchpad, creator history, burned pool
    liquidity and token rules as the rest. A mint-suffix test dropped them in the stream, the corpus and the wallet skill.
  - A coin now counts as pump.fun's when its mint ends in "pump" or a pump.fun migration recorded it: the stream adds a
    graduate the moment it sees the migration and reloads `corpus_meta`'s every 5 minutes. Their custom pools stay blocked.
  - On 2026-10-21 they add 1,582 mints to the day's 10,754 and 5.5 % of its volume. No existing row changes.
  - The corpus is rebuilt to include them (aggregation version 5). The current selector and fly keep trading through it.
- **Solana: a pump.fun coin's Meteora DLMM pools make its minutes too.**
  - Each minute keeps the pool with the most legs, as before, now among the coin's SOL-quoted PumpSwap and DLMM pools.
    The stream and the corpus apply the same rule: on 2026-10-22 15:00 UTC both gave 24,552 minute rows for 3,078 coins.
  - In that hour 99 % of minutes kept their pool, 210 minutes of 39 coins moved to a DLMM pool, and 101 coins gained 937
    minutes. Coins graduated into a pool quoted in TTWO, PUMP, WBTC or USDC (Dacity, Macca, pill, baton, Au) have their
    only SOL market there.
  - A graduation's quote asset is now kept (`corpus_meta.grad_quote`): from the migration, or for an older coin from trades
    in its own pool. A coin graduated in another quote has its dev buy and seed reserve in that quote's units, so those two
    inputs read as unknown for it, live and in training. 17 % of graduations on 2026-10-20 were not in SOL.
  - Paper fills charge the minute's pool fee, so a DLMM minute costs that pool's dynamic fee.
- **Robinhood Chain: a Pons coin's own Uniswap v4 pools elsewhere, and plain v4 memecoins, are traded too.**
  - The stream already read every PoolManager log and kept only the Pons pools'. In one hour on 2026-10-23, 1,480 v4
    pools traded 40,960 times; the Pons pools made 3,375 of those swaps. 85 hookless pools pairing a coin with ETH, WETH or
    USDG cleared the bot's floors, with 17,051 swaps.
  - A pool is looked at once it trades: its last half hour must clear the RH book's floors (reserve and 15-minute
    volume, in ETH). Of 1.07 million v4 pools on the chain, only those few hundred a day are registered.
  - A new pool qualifies when it pairs a coin with ETH, WETH or USDG (`RH_V4_QUOTES`) and has no hook. A graduated Pons
    coin's other pool may use any priced quote.
  - Hooked pools stay out unless named in `RH_V4_HOOKS`. In a hookless pool the Swap event's amounts equal the
    PoolManager's token flows exactly (230 swaps measured). The dynamic-fee hook of ORBIO's and ZZZ's other USDG pools
    (`0x08e5…a080`) moved up to 6 times more than its Swap events show. The Pons coins' main outside pools are hookless,
    with fees of 0.8–3 %.
  - A plain coin's contract is checked first, busiest coins first, one a minute (`RH_V4_CHECKS_PER_SLOT`). It is refused
    when its owner or an admin role can mint, upgrade, blacklist, pause, set taxes or hooks, or switch trading; when it
    has a built-in tax; when it is not 18 decimals; and when it is a major, stablecoin or tokenized stock. On 15 known
    coins the check matched a manual review.
  - Its creator is the sender of its first pool's initialization, with the same history inputs as a Pons deployer. Its
    age counts from that pool. What only a Pons launch records reads as unknown.
  - Nothing of this runs in front of a minute: the stream keeps the pass's logs after the minute is written, and the rest
    runs mid-minute once that minute is written and no live order waits on either chain. Only the contract checks read
    the network: the public RPC, once per read, never after 40 s into the minute.
  - A pool goes live with its coin's meta in one commit, between two passes, so the stream indexes every swap from the
    next block on. A plain coin trades only after the engine has watched it for 3 hours, the longest feature window.
  - `fly-trader rh-v4-backfill` (its own process) imports every pool's key since block 0, so pools older than the
    console's watch can be found. It then backfills each live pool's history before it went live.
  - A coin with several live pools takes each minute from the one with the most swaps, as on Solana. A non-Pons pool's
    swaps carry its LP fee. WETH counts as ETH.

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
- **The Overview's open positions table fits its card without scrolling sideways, from 1200 px wide up.**
  - The card takes seven twelfths of the row and the wealth chart five, the other way round from before. The chart
    still shows its whole history.
  - The table's headers wrap onto two lines, its cells pad less, and a long token name ends in an ellipsis.

### Platform

- **The archive's hour files keep every venue's SOL-quoted trades, not only pump.fun's curve and PumpSwap.**
  - Meteora DLMM and DAMM, Raydium, Orca and other launchpads' curves add about a third to the rows (~90 MB an hour).
    The lifecycle events keep the quote mint. Both files are written atomically and the trades file says what it keeps.
  - `fly-trader refetch-archive` downloads the older hours again in its own process, 2 at a time (`REFETCH_PARALLEL`).
    It starts no download while the live stream is more than 15 s behind (`REFETCH_MAX_LAG_S`), leaves the newest 6 hours
    to the console, and rebuilds each finished day itself. About 4,500 hours, roughly 2.7 TB of downloads; run it under
    `nice -n 19` only after the console runs this code. Progress: `ui_settings['replay_refetch_status']`.
  - A day is rebuilt whenever one of its hour files is newer than its aggregate.
- **The stream relay forwards pump.fun graduates without the "pump" suffix and every SOL-quoted DLMM trade.**
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

- **`tools/make_seed.py` judges the stored replay verdict as the fly session and `apply-release` do,** with the corpus
  build versions set aside (`selector.defs_match`). It compared the verdict's definitions exactly, so the corpus rebuild
  that retires no model still blocked a release.
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
