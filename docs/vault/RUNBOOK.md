# $FLY vault: launch runbook

These are the steps only you can do, because they need your keys or accounts. They are in order. The agent never
handles your keys.

## 0. Accounts and keys (once)
- **Reown project id.** Create it at dashboard.reown.com and allowlist `fly-trader.app`, plus `localhost:5173` for the
  rehearsal.
- **Telegram bot.** Create one with @BotFather; you need its token and your chat id.
- **Release-signing key (Mac).** Run `ssh-keygen -t ed25519 -f ~/.ssh/fly_release`. The public key goes to the server's
  allowed signers in step 3.
- **Backup key.** This is the age or ssh-ed25519 public key the backups are encrypted to. You can use
  `~/.ssh/fly_release.pub` or an age key.
- **HF storage bucket.** Create a private bucket named `bryceweiner/fly-vault-backups`, then a fine-grained token that
  can write only to that bucket.
- **Hardware-wallet EVM account.** This account owns the vault.

## 1. Rehearsal (testnet 46630 + Solana devnet)
1. Deploy the contracts to the testnet:
   ```
   cd contracts
   forge script script/DeployTestnet.s.sol --rpc-url https://rpc.testnet.chain.robinhood.com --ledger --broadcast
   ```
   The addresses land in `contracts/deployments/46630.json`.
2. Build the site against the testnet:
   ```
   cd web
   VITE_NETWORK=testnet VITE_REOWN_PROJECT_ID=… VITE_VAULT_ADDRESS=… VITE_TIMELOCK_ADDRESS=… VITE_FLY_ADDRESS=<MockFLY> npm run dev
   ```
   Run a local relay (see `web/relay/README.md`) with `domain: localhost:5173`, `chain_id: 46630`, `sol_chain: devnet`.
3. Run the fly (Docker or the Mac) with these settings:
   - `VAULT_ENABLED=1`, `VAULT_CLUSTER=devnet`, `VAULT_SOLANA_RPC_URL=https://api.devnet.solana.com`
   - `VAULT_PERIOD_S=900`
   - `RH_CHAIN_ID=46630`, `RH_RPC_URL=https://rpc.testnet.chain.robinhood.com`
   - `VAULT_SITE_DOMAIN=localhost:5173`, `VAULT_SITE_URI=http://localhost:5173/vault.html`
   - `LIVE_ENABLED=0`
   - a throwaway devnet key, funded from the devnet faucet
4. Work through the checklist:
   - lock from two wallets at different times, then request and cancel a withdrawal
   - send devnet SOL from an address that is not a funding address (it counts as profit)
   - wait for a 15-minute settlement
   - claim from both wallets, then replay a used nonce
   - kill the fly in the middle of a claim and restart it
   - `fly-trader vault status`

## 2. Mainnet contracts
1. Deploy:
   ```
   cd contracts
   FLY_TOKEN=0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3 OWNER=<hardware account> \
     forge script script/Deploy.s.sol --rpc-url https://rpc.mainnet.chain.robinhood.com --ledger --broadcast
   ```
   The script refuses to run on chain 4663 unless the delays are 8 days (timelock) and 7 days (withdrawal).
2. Verify the contracts on Blockscout. `contracts/README.md` has a manual fallback if Cloudflare blocks forge.
3. Record the implementation code hash:
   ```
   cast keccak $(cast code <impl> --rpc-url https://rpc.mainnet.chain.robinhood.com)
   ```
   This value becomes `VAULT_IMPL_CODEHASHES` in step 3.

## 3. Server (Netcup RS 2000 G12, Ubuntu 24.04)
1. Put your SSH key on the server. Then copy the `deploy/` folder from the distribution branch and run:
   `sudo bash deploy/bootstrap-host.sh "$(cat ~/.ssh/fly_release.pub)"`
   It prints the **vault wallet address**. The key is created on the server and stays there.
2. Edit `/srv/fly/vault.env`. Fill in:
   - `HELIUS_API_KEY`, `JUPITER_API_KEY`
   - `FUNDING_ADDRESSES`
   - `VAULT_ADDRESS`, `VAULT_TIMELOCK`, `VAULT_START_BLOCK`, `VAULT_IMPL_CODEHASHES`
   - `RELAY_SECRET`
   - `TELEGRAM_*`
   - `VAULT_BACKUP_*`
3. Edit `/etc/fly/update.json`. Only the Telegram fields need filling.
4. Publish the first release from the Mac. Keep a distribution worktree at `../fly-trader-dist`, then run:
   `.venv/bin/python tools/publish_release.py --worktree ../fly-trader-dist --key ~/.ssh/fly_release --models`
5. On the server, start the updater:
   ```
   sudo systemctl enable --now fly-update.timer
   journalctl -u fly-update -f
   ```
   It builds the image and starts the fly, which warms up on paper.
6. To open the console:
   ```
   ssh -L 8501:127.0.0.1:8501 <server>
   ```
   Then browse to http://localhost:8501/vault.

## 4. Site (Gandi)
1. Build the upload bundle:
   ```
   cd web
   VITE_REOWN_PROJECT_ID=… VITE_VAULT_ADDRESS=… VITE_TIMELOCK_ADDRESS=… bash scripts/bundle_gandi.sh
   ```
2. Upload the contents of `build/gandi/` so they replace `site/` entirely.
3. Put `relay.json` next to `wsgi.py`, using the same secret as `RELAY_SECRET`, then restart the instance.
4. Probe the relay from the Mac:
   ```
   FLY_RELAY_SECRET=<secret> python web/relay/hmacauth.py https://fly-trader.app/api/fly/probe k1
   ```
   Check the output against the checklist in `web/relay/README.md`. The things that must hold: the database path is
   writable, WAL works, and nothing caches `/api`.

## 5. Go live
1. Send 5 SOL to the vault wallet from an address listed in `FUNDING_ADDRESSES`.
2. Check the deposit was counted: `fly-trader vault status` (inside the container) should show `deposits: 5000000000`.
   If it came from another address, run `fly-trader vault reclassify <sig> deposit`.
3. The fly trades live after the handover: at least 20 paper trades.
4. Settlements run every Monday at 00:00 UTC. Claims open from the first one.

## Day to day
- **New models or code:** run `publish_release.py`, with `--models` to refresh the models. The server applies the
  release within 5 minutes. A release that changes Docker or compose files waits until you run
  `sudo python3 /usr/local/lib/fly/fly_update.py approve <seq>` on the server.
- **Take principal out:** `docker compose -p fly -f /srv/fly/docker-compose.yml exec fly fly-trader vault withdraw --sol X --to <funding address>`.
- **A halt alert** (unknown transaction or anomaly):
  1. Find the cause on the console's `$FLY vault` page.
  2. Run `fly-trader vault resume`.
  3. Run `fly-trader resume-entries`.
- **Restore from backups:**
  - Key: `age -d -i ~/.ssh/fly_release keys/<pubkey>.age`
  - Ledger: `age -d … ledger/<day>.dump.age | pg_restore`
