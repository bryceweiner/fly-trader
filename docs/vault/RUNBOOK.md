# $FLY vault: launch runbook

The steps only you can do, because they need your keys, accounts or devices, in the order to do them. The agent never
handles your keys. How the system works is in [SPEC.md](SPEC.md); this file is what to type.

**Who holds what** (until the Ledger is back; the "Ledger" section moves everything to it without redeploying):

| Key | Where | Can do |
|---|---|---|
| Solana owner | Solflare on your phone | Everything with the treasury: create/remove its spending limits, take principal out |
| EVM owner | MetaMask | Pause new $FLY locks; propose upgrades (public for 8 days); hand the roles to the Ledger |
| SSH | Secretive on the Mac (Secure Enclave, Touch ID) | Log in to the server over Tailscale |
| Release signing | Secretive (Touch ID) + an offline recovery key | Sign releases; the server still holds code changes 24 h |
| Backup decryption | an age key, offline | Read the encrypted backups |
| Trading / payout | on the server, readable only by the signer container | Refill the float (L1, per day), pay claims (L2, per week) |

A full takeover of the server can lose at most the trading float plus what is left of this day's L1 and this week's
L2, and only until you revoke them from your phone ("Emergency").

Conventions below:
- `$` lines run on the **Mac** in the fly-trader checkout (the repo root). Commands for a subfolder run in a subshell,
  `$ (cd web && …)`, so you are always back at the root.
- `server$` lines run on the **server** as `fly-admin`.
- `sudo fly <command>` runs `fly-trader <command>` inside the running fly container; `sudo fly signer <command>` runs
  in the signer container.

## 0. Once: tools, repos, accounts, keys

### 0.1 Mac tools
```
$ brew install age node uv git
$ brew install --cask secretive tailscale
$ curl -L https://foundry.paradigm.xyz | bash && foundryup                  # forge, cast
$ sh -c "$(curl -sSfL https://release.anza.xyz/stable/install)"              # solana CLI + solana-test-validator
$ uv sync --extra dev                                                        # the Python environment (.venv)
$ (cd web && npm ci)
$ (cd contracts && forge soldeer install)
$ git worktree add ../fly-trader-dist distribution                            # the branch the server runs
```

### 0.2 Accounts
- **2FA everywhere.** Use a hardware key or TOTP on Netcup (both the SCP and CCP logins), Gandi, Hugging Face, GitHub,
  your Tailscale identity provider and your Apple ID. Set a Telegram two-step password.
- **Phone wallets.** Write the Solflare and MetaMask seed phrases on paper and keep them offline. The phone needs a
  passcode and biometrics. Keep a little SOL in Solflare (about 0.05 SOL pays the treasury's setup and any later
  change) and a little ETH on Robinhood Chain in MetaMask (for `pause()`).
- **Reown project id.** Create it at dashboard.reown.com. Allowlist `fly-trader.app`, and `localhost:5173` for local
  testing.
- **Telegram.**
  1. Create a bot with @BotFather and note its token.
  2. Send `/start` to your new bot; it cannot message you before that.
  3. Message @userinfobot to learn your user id. In a private chat with the bot, the chat id is that same number.
     Only that user's `/panic` and `/status` count.
- **Hugging Face.**
  1. Create a **public** model repo `bryceweiner/fly-trader`: the server and self-hosted installs read releases
     without logging in.
  2. Create a **private** storage bucket `bryceweiner/fly-vault-backups` for the encrypted backups.
  3. Create two tokens: one with write access to the model repo, for publishing from the Mac; one fine-grained,
     write-only to the bucket, for the server's backups.
  4. `$ uv run hf auth login` with the first token.
- **GitHub.** Push access to the fly-trader repo: publishing pushes the `distribution` branch.
- **Second Solana RPC.** Get an API URL from a provider other than Helius (Triton, QuickNode or Alchemy). It is used
  twice:
  - `SOLANA_CHECK_RPC_URL`: settlement balances and halts must agree with it.
  - `SIGNER_RPC_URL`: the signer's own reads and simulations.
- **healthchecks.io.** Create a check with period 5 minutes and grace 10 minutes, and connect its Telegram
  integration. Its ping URL is `HEALTHCHECK_URL`.

### 0.3 Tailscale
1. Install Tailscale on the Mac, log in, and turn on **MagicDNS** in the admin console, so `fly-vault` resolves.
2. In the admin console's access controls, add:
   ```
   "tagOwners": {"tag:fly": ["autogroup:admin"]},
   "acls": [{"action": "accept", "src": ["<your Tailscale login>"], "dst": ["tag:fly:22"]}]
   ```
   Add no rule whose source is `tag:fly`: the server needs to reach nothing on your tailnet.
3. Turn on **Tailnet Lock** from the Mac:
   ```
   $ tailscale lock status                                          # shows this Mac's tlpub:… key
   $ tailscale lock init --gen-disablements 3 --confirm tlpub:<that key>
   ```
   Store the printed disablement secrets offline, like a seed phrase.
4. Create a one-off, pre-authorised auth key tagged `tag:fly` in the admin console. Sign it so the server may join the
   locked tailnet, and keep the output (this is `TS_AUTHKEY` in section 4.2):
   ```
   $ tailscale lock sign tskey-auth-…
   ```

### 0.4 Your keys (Secretive, recovery, backup)
1. In Secretive create two keys, `fly-ssh` and `fly-release`, each requiring Touch ID. Save their public keys as
   `~/.ssh/fly-ssh.pub` and `~/.ssh/fly-release.pub`.
2. Put the path of Secretive's agent in your shell profile (`~/.zshrc`), since several steps use it:
   `export SECRETIVE=~/Library/Containers/com.maxgoedjen.Secretive.SecretAgent/Data/socket.ssh`.
   Then tell ssh to use Secretive for the server, in `~/.ssh/config`:
   ```
   Host fly-vault
     User fly-admin
     IdentityAgent ~/Library/Containers/com.maxgoedjen.Secretive.SecretAgent/Data/socket.ssh
   ```
3. The offline recovery release signer:
   ```
   $ mkdir -p ~/fly-keys
   $ ssh-keygen -t ed25519 -f ~/fly-keys/fly-release-recovery -C fly-release-recovery
   $ age -p ~/fly-keys/fly-release-recovery > ~/fly-keys/fly-release-recovery.age && rm ~/fly-keys/fly-release-recovery
   ```
   Put `fly-release-recovery.age` on a USB stick and in your password manager. Keep `fly-release-recovery.pub`.
4. The backup key: `$ age-keygen -o ~/fly-keys/fly-backup.key`. Store it like the recovery key; its public line
   (`age1…`, printed and at the top of the file) is `VAULT_BACKUP_RECIPIENT`. It is deliberately not the release key.
5. Trust both release keys in the distribution, so that self-hosted installs accept your releases:
   ```
   $ echo "bryce namespaces=\"fly-trader-release\" $(cat ~/.ssh/fly-release.pub)" >> ../fly-trader-dist/release/allowed_signers
   $ echo "bryce-recovery namespaces=\"fly-trader-release\" $(cat ~/fly-keys/fly-release-recovery.pub)" >> ../fly-trader-dist/release/allowed_signers
   $ git -C ../fly-trader-dist status                     # must be clean apart from this file (no merge in progress)
   $ git -C ../fly-trader-dist commit -m "Release keys" release/allowed_signers
   ```

## 1. Rehearse locally (no real money)
- **Custody against the real Squads program:**
  `$ tools/custody_e2e.sh <your Helius mainnet RPC URL>`. It dumps the program from mainnet (read-only) and runs it on
  a throwaway validator on ports 11xxx. It checks:
  - the owner page's code creates the treasury and both limits, changes L2 and revokes them
  - the signer refills through L1 and pays a claim through L2 up to its cap
  - a stolen key cannot send L1 anywhere but the trading wallet (refused on chain)
- **The whole vault on this Mac:** `$ .venv/bin/python tools/vault_demo.py up`. It runs anvil, the contracts, the relay,
  the vault worker, a treasury on a local validator and the site at http://localhost:5173. The file's docstring lists
  the wallet settings and commands (`fund-evm`, `profit`, `status`, `down`).
- **Optional: RH testnet + Solana devnet.**
  1. Deploy the contracts:
     ```
     $ cast wallet new ~/.foundry/keystores rehearsal     # fund it from faucet.testnet.chain.robinhood.com
     $ (cd contracts && forge script script/DeployTestnet.s.sol --rpc-url robinhood_testnet --account rehearsal \
         --sender $(cast wallet address --account rehearsal) --broadcast)
     $ cp contracts/deployments/46630.json ~/fly-keys/testnet-deployment.json
     ```
     The addresses land in `contracts/deployments/46630.json`: `vault`, `timelock`, `token` (MockFLY), `blockNumber`.
     Keep the copy: `vault_demo.py` deploys to anvil, which uses the same chain id, and overwrites that file.
  2. Run the site against it:
     Run a relay as in `web/relay/README.md` (e.g. `tools/vault_demo.py` shows how it starts one on port 8612), with
     `"domain": "localhost:5173"`, `"uri": "http://localhost:5173/vault.html"`, `"chain_id": 46630`,
     `"sol_chain": "devnet"`. Then:
     ```
     $ (cd web && VITE_NETWORK=testnet VITE_REOWN_PROJECT_ID=… VITE_VAULT_ADDRESS=<vault> VITE_TIMELOCK_ADDRESS=<timelock> \
         VITE_FLY_ADDRESS=<token> VITE_SOLANA_RPC=https://api.devnet.solana.com VITE_RELAY_PROXY=http://127.0.0.1:8612 npm run dev)
     ```
  3. Make two throwaway devnet keys (`$ solana-keygen new -o /tmp/devnet-trading.json`, the same for payout; fund
     the trading one at faucet.solana.com). Create the treasury: http://localhost:5173/owner.html with Solflare on
     devnet, as in section 4.3, with those two public keys.
  4. Run the vault worker on the Mac with these settings in the environment:
     - `VAULT_ENABLED=1`, `VAULT_CLUSTER=devnet`, `VAULT_SOLANA_RPC_URL=https://api.devnet.solana.com`,
       `SIGNER_RPC_URL=https://api.devnet.solana.com`, `VAULT_PERIOD_S=900`, `LIVE_ENABLED=0`
     - `BOT_PRIVATE_KEY_FILE=/tmp/devnet-trading.json`, `PAYOUT_PRIVATE_KEY_FILE=/tmp/devnet-payout.json`
     - `VAULT_MULTISIG`, `VAULT_LIMIT_TRADING`, `VAULT_LIMIT_PAYOUT` (from the owner page)
     - `RH_CHAIN_ID=46630`, `RH_RPC_URL=https://rpc.testnet.chain.robinhood.com`, `VAULT_ADDRESS`,
       `VAULT_TIMELOCK`, `VAULT_START_BLOCK`
     - `VAULT_SITE_DOMAIN=localhost:5173`, `VAULT_SITE_URI=http://localhost:5173/vault.html`
     - `RELAY_URL=http://127.0.0.1:8612`, `RELAY_SECRET` (the relay's), and a separate `DATABASE_URL`
       (`createdb fly_vault_rehearsal; DATABASE_URL=postgresql:///fly_vault_rehearsal uv run fly-trader init-db`)

     ```
     $ uv run python -c "from fly_trader.vault import worker; worker.main()"
     ```

## 2. Mainnet contracts (EVM owner = MetaMask)
1. Deploy from a throwaway keystore. It pays the gas and holds no role afterwards:
   ```
   $ cast wallet new ~/.foundry/keystores deployer                 # fund with a little ETH on Robinhood Chain
   $ (cd contracts && FLY_TOKEN=0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3 OWNER=<your MetaMask address> \
       forge script script/Deploy.s.sol --rpc-url robinhood --account deployer \
       --sender $(cast wallet address --account deployer) --broadcast \
       --verify --verifier blockscout --verifier-url https://robinhoodchain.blockscout.com/api/)
   ```
   The script refuses to run on chain 4663 unless the token is the real $FLY and the delays are 8 days (timelock) and
   7 days (withdrawal). It writes `contracts/deployments/4663.json`: `vault`, `implementation`, `timelock`,
   `blockNumber`. You need all four below.
2. Check that the deployer holds nothing. Each of these must print `false`:
   ```
   $ D=$(cast wallet address --account deployer)
   $ R=https://rpc.mainnet.chain.robinhood.com
   $ cast call <timelock> "hasRole(bytes32,address)(bool)" $(cast keccak PROPOSER_ROLE) $D --rpc-url $R
   $ cast call <timelock> "hasRole(bytes32,address)(bool)" 0x0000000000000000000000000000000000000000000000000000000000000000 $D --rpc-url $R
   $ cast call <vault> "hasRole(bytes32,address)(bool)" $(cast keccak PAUSER_ROLE) $D --rpc-url $R
   ```
   Then delete `~/.foundry/keystores/deployer`.
3. Check the contracts show as verified on Blockscout; `contracts/README.md` has a manual fallback if Cloudflare
   blocked forge's verification.
4. Record the implementation code hash; it becomes `VAULT_IMPL_CODEHASHES`:
   `$ cast keccak $(cast code <implementation> --rpc-url https://rpc.mainnet.chain.robinhood.com)`
5. From your phone (MetaMask), you use Blockscout's page for each contract, under "Write proxy" / "Write contract":
   - **Pause new locks:** FlyVault, `pause()`; later `unpause()`. Withdrawals always stay open.
   - **Cancel a timelock operation you did not schedule:** Timelock, `cancel(<id>)`. The Telegram alert gives the full
     id.

## 3. Site (Gandi)
The owner page is part of the site, and you need it for the treasury, so the site goes up before the server.
1. Build the upload bundle:
   ```
   $ (cd web && VITE_REOWN_PROJECT_ID=… VITE_VAULT_ADDRESS=<vault> VITE_TIMELOCK_ADDRESS=<timelock> bash scripts/bundle_gandi.sh)
   ```
   The bundle lands in `build/gandi/` at the repo root: `wsgi.py`, `site-headers.json`, `relay/` and `site/`.
   `owner.html` is included; it is not linked anywhere, it is noindex, and it can do nothing without your owner wallet.
2. Upload the contents of `build/gandi/` over Gandi SFTP (or git) into the instance's deploy directory (where
   `wsgi.py` lives), replacing `site/` entirely.
3. Make the relay secret: `$ python3 -c "import secrets;print(secrets.token_hex(32))"`. The fly gets the same value as
   `RELAY_SECRET`.
4. Upload `relay.json` next to `wsgi.py` (never into `site/`, never into git). The database goes on the instance's
   persistent disk, outside the deploy directory:
   ```json
   {"keys": {"k1": "<the secret>"}, "db_path": "/srv/data/home/relay/relay.sqlite3", "domain": "fly-trader.app",
    "uri": "https://fly-trader.app/vault.html", "chain_id": 4663, "sol_chain": "mainnet", "claim_ttl_s": 900,
    "min_lamports": 2000000, "client_ip_header": null}
   ```
   Restart the instance. If a later deploy replaces the directory, upload `relay.json` again.
5. Probe the relay and work through the 7-point checklist in `web/relay/README.md` (writable, WAL, client IP header,
   no caching, clock):
   `$ FLY_RELAY_SECRET=<the secret> python3 web/relay/hmacauth.py https://fly-trader.app/api/fly/probe k1`

## 4. Server (Netcup RS 2000 G12, Ubuntu 24.04)

### 4.1 Publish the first release
1. The distribution worktree must hold exactly what you want to ship. Code is ported there from master by hand (or with
   a merge); publishing does not port it. Check it is clean: `$ git -C ../fly-trader-dist status`.
2. `--models` refreshes the models from this Mac; use it for the first release. That needs:
   - a deployable fly (`fly-trader train-fly-selector`) that passed `fly-trader fly-replay`
   - a deployed selector (pinned, or the newest that qualifies)
   - fitted wallet skill (`fly-trader fit-wallet-skill`) with its tables, and a built connectome
   Without `--models` the release carries the models already in the worktree, and publishing refuses if the
   wallet-skill table named in `models/wallet_skill/TABLE` is not there.
3. Publish. `SSH_AUTH_SOCK` points `ssh-keygen` at Secretive, so Touch ID confirms the signature:
   ```
   $ SSH_AUTH_SOCK=$SECRETIVE .venv/bin/python tools/publish_release.py --worktree ../fly-trader-dist --key ~/.ssh/fly-release.pub --models
   ```
   It runs the tests, commits, signs, verifies with the server's own verifier, pushes the `distribution` branch and
   uploads one Hugging Face commit. It refuses unless the key is in the worktree's `release/allowed_signers` (0.4 step
   5). Note the printed **manifest sha256**; it pins this first release on the server.
4. **Do not publish again until the server has installed this release** (section 4.4). A new server accepts only the
   pinned manifest, and it permanently refuses any other head it sees first.

### 4.2 Bootstrap
1. Order the server with `~/.ssh/fly-ssh.pub` as root's SSH key. Then copy the scripts and the two release public
   keys to it:
   ```
   $ scp -o IdentityAgent=$SECRETIVE -r ../fly-trader-dist/deploy ~/.ssh/fly-release.pub ~/fly-keys/fly-release-recovery.pub root@<server IP>:
   ```
2. Log in as root (`$ ssh -o IdentityAgent=$SECRETIVE root@<server IP>`) and run the first pass:
   ```
   # TS_AUTHKEY=<signed key from 0.3 step 4> RECOVERY_PUBKEY="$(cat fly-release-recovery.pub)" \
       EXPECT_MANIFEST=<manifest sha256 from 4.1> ADMIN_USER=fly-admin bash deploy/bootstrap-host.sh "$(cat fly-release.pub)"
   ```
   It hardens the box (roughly CIS level 1):
   - **SSH:** only `fly-admin`, key only (ed25519, P-256, FIDO), root off. Root's key is copied to `fly-admin`.
   - **Firewall:** in and out; containers reach only HTTPS, HTTP (apt) and DNS.
   - **Kernel and Docker:** user namespaces, no new privileges.
   - **Monitoring:** fail2ban; an audit log of every read of the wallet keys, with a Telegram alert when anything but
     the signer reads them; a daily AIDE file-integrity check (05:30 UTC) that reports to Telegram when files changed;
     automatic security updates (reboot 04:00 UTC when needed).
   - **Keys:** it makes the trading and payout keys and prints them, **once**, as `bot_key: <address>` (the trading
     key) and `payout_key: <address>`. Copy both lines now. Later, once the containers run:
     `server$ sudo fly signer status` (as `"trading"` and `"payout"`).
3. **Keep the root session open.** From a second terminal on the Mac, log in over Tailscale:
   `$ ssh fly-vault`. Only when that works, run the second pass in the root session, which removes public SSH:
   ```
   # CLOSE_PUBLIC_SSH=1 bash deploy/bootstrap-host.sh "$(cat fly-release.pub)"
   ```
   The recovery key already installed is kept. After this the server has no open port on the internet.

### 4.3 The treasury (your phone)
1. Open `https://fly-trader.app/owner.html` in Solflare's browser and connect Solflare.
2. Paste the trading key (`bot_key`) and the payout key (`payout_key`), leave the multisig field empty, and tap
   "Read the treasury".
3. Tap "Create the treasury multisig": you are its only member, threshold 1, no time lock.
4. Tap "Create L1" (2 SOL a day, to the trading wallet only) and "Create L2" (1 SOL a week, claims).
5. Every check the page shows must pass. It prints the three lines for `custody.env`. **Save the multisig, L1 and L2
   addresses in your password manager.** The page remembers them only in that browser, and the Revoke button needs
   them.

### 4.4 Settings, start, check
1. Fill in the settings with `sudoedit`: `server$ sudoedit /srv/fly/custody.env`, and so on. The files are root-only
   (mode 600).
   - **`/srv/fly/custody.env`:** the three lines from the owner page (`VAULT_MULTISIG`, `VAULT_LIMIT_TRADING`,
     `VAULT_LIMIT_PAYOUT`) and `VAULT_BACKUP_RECIPIENT` (your `age1…`). The float and caps have working defaults.
   - **`/srv/fly/signer.env`:** `SIGNER_RPC_URL` (the second provider). Required: the signer does not start without it.
   - **`/srv/fly/vault.env`:**
     - `HELIUS_API_KEY`, `JUPITER_API_KEY`, `SOLANA_CHECK_RPC_URL`
     - `CAPITAL_SOL` (your principal, e.g. 5)
     - `FUNDING_ADDRESSES`: the addresses you will send principal from, comma-separated; at least your Solflare address
     - `VAULT_ADDRESS`, `VAULT_TIMELOCK`, `VAULT_START_BLOCK` (`blockNumber`), `VAULT_IMPL_CODEHASHES`, all from step 2
     - `RELAY_SECRET` (the relay's secret from step 3)
     - `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`, `TELEGRAM_ADMIN_USER_ID`, `HEALTHCHECK_URL`
     - `VAULT_BACKUP_BUCKET`, `VAULT_BACKUP_HF_TOKEN` (the bucket-only token)
   - **`/etc/fly/update.json`:** `telegram_bot_token` and `telegram_chat_id` (the host updater, the key-read watcher and
     AIDE report through these). `expect_manifest_sha256` was set by the bootstrap.

   Everything must be filled in before the next step: a first release that does not come up healthy is refused for
   good. If that happens: fix the cause, publish a new release, and put its manifest sha256 into
   `expect_manifest_sha256` in `/etc/fly/update.json`.
2. Start the updater. It installs the pinned release, builds the brain and signer images and starts them (the first
   build takes several minutes):
   ```
   server$ sudo systemctl enable --now fly-update.timer
   server$ sudo systemctl start fly-update.service          # the first round now, instead of in 2 minutes
   server$ sudo journalctl -u fly-update -f
   ```
3. Check everything:
   - `server$ sudo /usr/local/lib/fly/check-host.sh`: every control must pass (keys only in the signer, Tailnet Lock on,
     no public SSH…)
   - `server$ sudo fly vault custody`: no problems
   - `server$ sudo python3 /usr/local/lib/fly/fly_audit_watch.py test`: the Telegram test message arrives
4. The console: `$ ssh -L 8501:127.0.0.1:8501 fly-vault`, then http://localhost:8501/vault.

## 5. Go live
1. From a funding address, send **0.05 SOL to the trading wallet** (the `bot_key` address). It pays the fees of its own
   refills and of claims.
2. Send your principal to the **treasury** (the vault 0 address on the owner page) from a funding address. The fly
   refills its 2 SOL float from it through L1 on its own.
3. `server$ sudo fly vault status`: `"ledger": {"deposits": …}` must equal the SOL you sent from funding addresses. If
   some came from another address, it counts as profit: `server$ sudo fly vault reclassify <signature> deposit`.
4. The fly trades live after the handover (at least 20 paper trades). Settlements run every Monday 00:00 UTC; claims
   open from the first one.
5. When claims are larger than L2 allows in a week, Telegram says a claim is waiting. Raise L2 as in "Day to day".
   Waiting claims never fail.

## Emergency
- **Something is wrong (possible compromise):**
  1. On your phone, open the owner page and tap **Revoke L1 and L2 now**. From then on the server cannot move any SOL
     out of the treasury; trading keeps only its float.
  2. Send `/panic` to the bot. The fly halts; the signer refuses buys, top-ups and claims; open positions are sold; and
     the float goes back to the treasury, all but the 0.3 SOL gas reserve.
  3. If $FLY locking should stop too: MetaMask, Blockscout, FlyVault `pause()`.
- **A key-read, halt, AIDE or setup-change alert:** look before resuming. Resuming is SSH-only:
  ```
  server$ sudo fly vault resume
  server$ sudo fly reset-circuit --kill
  server$ sudo fly resume-entries
  server$ sudo fly signer clear-panic
  ```
- **Rebuild after losing the server:**
  1. On the owner page, revoke L1 and L2 of the old keys.
  2. Publish nothing; take the manifest sha256 of the current release:
     `$ curl -sL https://huggingface.co/bryceweiner/fly-trader/resolve/main/release.json | shasum -a 256`.
  3. New server: sections 4.2 (with that pin) to 4.4. On the owner page, clear the old L1/L2 fields and create new
     limits for the new keys; put the new addresses in `custody.env`.
  4. Restore the ledger. Download the newest `ledger/<day>.dump.age` from the backup bucket, then:
     ```
     $ age -d -i ~/fly-keys/fly-backup.key <day>.dump.age > ledger.dump
     $ scp ledger.dump fly-vault: && rm ledger.dump
     server$ sudo docker stop fly-fly-1                     # the brain must not write while the tables are replaced
     server$ sudo docker exec -i fly-db-1 pg_restore -U fly -d fly_trader --clean < ledger.dump && rm ledger.dump
     server$ sudo docker exec fly-db-1 psql -U fly -d fly_trader -c "DELETE FROM vault_kv WHERE key IN ('backup_key_done','custody','halt','panic')"
     server$ sudo python3 /usr/local/lib/fly/fly_update.py restart
     ```
  5. The old trading wallet's float and positions: `$ age -d -i ~/fly-keys/fly-backup.key keys/trading-<address>.age`
     gives its secret (base58, 64 bytes), which you can import into a wallet to move them to the treasury.

## Day to day
- **Changing a limit** (e.g. raising L2 after a big settlement): on the owner page, "Set L2 to this amount". A new
  limit has a **new address**, so put it in `/srv/fly/custody.env` and restart:
  `server$ sudo python3 /usr/local/lib/fly/fly_update.py restart`. Until then claims wait. The same applies to any L1
  change.
- **After editing any `/srv/fly/*.env`:** `server$ sudo python3 /usr/local/lib/fly/fly_update.py restart`.
- **New models or code:** publish as in 4.1 step 3 (`--models` to refresh the models).
  - Model-only releases apply within 5 minutes; they cannot carry code (weights-only torch files, skops selectors).
  - A release that changes **code** is held 24 h, and Telegram says so. To apply it now:
    `server$ sudo python3 /usr/local/lib/fly/fly_update.py approve <seq>`. To stop it:
    `server$ sudo python3 /usr/local/lib/fly/fly_update.py veto <seq>`. If you did not publish it, the release key is
    compromised: remove it from `/etc/fly/allowed_signers` and rotate.
  - A release that changes `deploy/`, `docker/`, the Dockerfile, `.dockerignore` or a compose file waits for
    `approve`. It replaces only the compose file. For changed host scripts (bootstrap, `fly_update.py`,
    `check-host.sh`, the watchers), copy them over and re-run the bootstrap **with `CLOSE_PUBLIC_SSH=1`** (without it
    the bootstrap opens public SSH again); your recovery key is kept:
    ```
    $ scp -r ../fly-trader-dist/deploy ~/.ssh/fly-release.pub fly-vault:
    server$ sudo CLOSE_PUBLIC_SSH=1 bash deploy/bootstrap-host.sh "$(cat fly-release.pub)"
    ```
  - To sign with the recovery key: decrypt it (`$ age -d ~/fly-keys/fly-release-recovery.age > /tmp/k && chmod 600
    /tmp/k`), pass `--key /tmp/k`, then `rm /tmp/k`.
- **Take principal out:** `server$ sudo fly vault withdraw` prints how much you may take and how. You send it from the
  treasury in the Squads app with your owner wallet; the fly books and reports it.
- **Watch it:** `/status` to the bot at any time. A daily report arrives at 09:00 UTC. If it does not, or
  healthchecks.io alerts, the server or the fly is down. `server$ sudo fly vault status` gives the full ledger.
- **After an AIDE alert** you expected (a release, apt upgrades): `server$ sudo aide --update` then
  `server$ sudo cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db`.
- **Backups:** hot keys once, ledger nightly, both age-encrypted in the bucket. Until `VAULT_BACKUP_*` and the recipient
  are set, Telegram reminds you daily. To read them:
  - Hot keys: `$ age -d -i ~/fly-keys/fly-backup.key keys/<name>-<address>.age`
  - Ledger: see "Rebuild", step 4.

## Ledger (when you're home): no redeploys
- **Solana:** in the Squads app, add the Ledger as a member, then remove Solflare (or keep both at threshold 2; the
  owner page then proposes and approves, and the other member executes in the Squads app). L1 and L2 stay as they are.
- **EVM:** move MetaMask's roles to the Ledger through the timelock (8 days):
  1. Print the calldata:
     ```
     $ (cd contracts && VAULT=<vault> TIMELOCK=<timelock> OLD_OWNER=<MetaMask> NEW_OWNER=<Ledger> MODE=calldata \
         forge script script/HandOver.s.sol --rpc-url robinhood)
     ```
  2. Send the printed "schedule" data from MetaMask to the timelock (Blockscout "Write contract", `scheduleBatch`, or
     any raw-data send).
  3. After 8 days, send the "execute" data from any wallet.
- **SSH and releases:** add the Ledger's SSH key (ledger-agent) to `~fly-admin/.ssh/authorized_keys`. For releases,
  make the Ledger's key the release key: re-run the bootstrap (as in "Day to day") with the Ledger's public key as the
  argument (the bootstrap rewrites `/etc/fly/allowed_signers` from its arguments; a line added by hand would be lost),
  and replace the `bryce` line in the distribution's `release/allowed_signers` (publish that change while the old key
  still signs). Then remove the Secretive keys.
