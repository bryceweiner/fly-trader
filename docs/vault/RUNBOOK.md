# $FLY vault: launch runbook

These are the steps only you can do, because they need your keys or accounts. They are in order. The agent never
handles your keys.

**Who holds what** (until the Ledger is back; section "Ledger" moves everything to it without redeploying):

| Key | Where | Can do |
|---|---|---|
| Solana owner | Solflare on your phone | Everything with the treasury: create/remove its spending limits, take principal out |
| EVM owner | MetaMask | Pause new $FLY locks; propose upgrades (public for 8 days); hand the roles to the Ledger |
| SSH | Secretive on the Mac (Secure Enclave, Touch ID) | Log in to the server over Tailscale |
| Release signing | Secretive (Touch ID) + an offline recovery key | Sign releases; code changes still wait 24 h on the server |
| Backup decryption | age key, offline | Read the encrypted backups |
| Trading / payout | on the server, readable only by the signer container | Refill the float (L1, per day), pay claims (L2, per week) |

A full takeover of the server can lose at most the trading float plus what is left of this day's L1 and this week's
L2, and only until you revoke them from your phone (section "Emergency").

## 0. Accounts and keys (once)
- **2FA everywhere.** Use a hardware key or TOTP on Netcup (both the SCP and CCP logins), Gandi, Hugging Face, GitHub,
  your Tailscale identity provider and your Apple ID. Also set a Telegram two-step password.
- **Phone wallets.** Write the Solflare and MetaMask seed phrases on paper and keep them offline. The phone needs a
  passcode and biometrics.
- **Secretive** (github.com/maxgoedjen/secretive).
  1. Create two keys, `fly-ssh` and `fly-release`, each requiring Touch ID.
  2. Export their public keys: `fly-ssh.pub` goes into the server's `authorized_keys`, `fly-release.pub` becomes the
     release signer.
  3. Point ssh at the agent: `export SSH_AUTH_SOCK=~/Library/Containers/com.maxgoedjen.Secretive.SecretAgent/Data/socket.ssh`.
- **Recovery release signer (offline).**
  1. Create it: `ssh-keygen -t ed25519 -f fly-release-recovery`.
  2. Encrypt it: `age -p fly-release-recovery > fly-release-recovery.age`.
  3. Store the `.age` file on a USB stick and in your password manager, then delete the plaintext.
  4. Keep `fly-release-recovery.pub`.
- **Backup key.** Run `age-keygen -o fly-backup.key`. Store it like the recovery key; its public line (`age1…`) is
  `VAULT_BACKUP_RECIPIENT`. It is deliberately not the release key.
- **Reown project id.** Create it at dashboard.reown.com and allowlist `fly-trader.app`, plus `localhost:5173` for the
  rehearsal.
- **Telegram.**
  1. Create a bot with @BotFather and note its token.
  2. Note your chat id and your own user id (message @userinfobot). Only that user's `/panic` and `/status` count.
- **Tailscale.**
  1. Sign up and add the Mac.
  2. In the admin console: create a tag `tag:fly` and an ACL that allows only your devices to reach `tag:fly:22`.
  3. Turn on **Tailnet Lock** (`tailscale lock init`, signed by the Mac).
  4. Make a one-off, pre-authorised auth key tagged `tag:fly` for the server.
- **Second Solana RPC.** Create an account with a provider other than Helius (Triton, QuickNode or Alchemy). Its URL
  is used twice:
  - `SOLANA_CHECK_RPC_URL`: settlement balances and halts must agree with it.
  - `SIGNER_RPC_URL`: the signer's own reads and simulations.
- **healthchecks.io.** Create a check with a 10-minute grace period and connect its Telegram integration. The check URL
  is `HEALTHCHECK_URL`.
- **HF storage bucket.** Create a private bucket `bryceweiner/fly-vault-backups` and a fine-grained token that can only
  write to it.

## 1. Rehearsal
- **Custody, end to end, locally (no accounts needed):** `tools/custody_e2e.sh <your Helius URL>`. It runs the real
  Squads program on a throwaway validator on ports 11xxx, so the vault demo on 8899 is untouched. It checks:
  - the owner page's code creates the treasury and both limits, changes L2 and revokes them
  - the signer refills through L1 and pays a claim through L2 up to its cap
  - a stolen key cannot send L1 anywhere but the trading wallet (refused on chain)
- **Testnet 46630 + Solana devnet** (optional, same as before):
  1. Deploy with `DeployTestnet.s.sol` and a throwaway keystore.
  2. Build the site with `VITE_NETWORK=testnet`.
  3. Run the fly with `VAULT_CLUSTER=devnet`, `VAULT_PERIOD_S=900`, `LIVE_ENABLED=0`.
  4. For the treasury, open `owner.html` on the testnet build with Solflare set to devnet.

## 2. Mainnet contracts (EVM owner = MetaMask)
1. Deploy from a throwaway keystore. It pays the gas and holds no role afterwards:
   ```
   cd contracts
   cast wallet new ~/.foundry/keystores deployer            # fund with a little ETH on Robinhood Chain
   FLY_TOKEN=0x2fC7f9E2911f20b2C4660d2AEf808aa91bDdb3D3 OWNER=<your MetaMask address> \
     forge script script/Deploy.s.sol --rpc-url https://rpc.mainnet.chain.robinhood.com --account deployer \
     --sender $(cast wallet address --account deployer) --broadcast
   ```
   The script refuses to run on chain 4663 unless the delays are 8 days (timelock) and 7 days (withdrawal).
2. Check that the deployer holds nothing. Each of these must print `false`:
   ```
   R=https://rpc.mainnet.chain.robinhood.com D=$(cast wallet address --account deployer)
   cast call <timelock> "hasRole(bytes32,address)(bool)" $(cast keccak PROPOSER_ROLE) $D --rpc-url $R
   cast call <timelock> "hasRole(bytes32,address)(bool)" 0x0000000000000000000000000000000000000000000000000000000000000000 $D --rpc-url $R
   cast call <vault> "hasRole(bytes32,address)(bool)" $(cast keccak PAUSER_ROLE) $D --rpc-url $R
   ```
   Then delete the deployer keystore.
3. Verify the contracts on Blockscout. `contracts/README.md` has a manual fallback.
4. Record the implementation code hash: `cast keccak $(cast code <impl> --rpc-url …)`. It becomes
   `VAULT_IMPL_CODEHASHES`.
5. **From your phone** (MetaMask), you use Blockscout's page for each contract, under "Write proxy" / "Write contract":
   - **Pause new locks:** FlyVault, `pause()`; later `unpause()`. Withdrawals always stay open.
   - **Cancel a scheduled upgrade you did not make:** Timelock, `cancel(<operation id from the Telegram alert>)`.

## 3. Server (Netcup RS 2000 G12, Ubuntu 24.04)
1. Order the server with `fly-ssh.pub` as root's key. Copy the distribution branch's `deploy/` folder to it.
2. Publish the first release from the Mac. `tools/publish_release.py` prints its manifest hash; you pin it in the
   next step:
   ```
   .venv/bin/python tools/publish_release.py --worktree ../fly-trader-dist --key fly-release.pub --models
   ```
3. First pass, as root:
   ```
   TS_AUTHKEY=tskey-auth-… RECOVERY_PUBKEY="$(cat fly-release-recovery.pub)" EXPECT_MANIFEST=<hash from step 2> \
     ADMIN_USER=fly-admin bash deploy/bootstrap-host.sh "$(cat fly-release.pub)"
   ```
   It hardens the box (roughly CIS level 1):
   - **SSH:** only `fly-admin`, key only (ed25519, P-256, FIDO), root off, over Tailscale.
   - **Firewall:** in and out; containers only reach HTTPS and DNS.
   - **Kernel and Docker:** user namespaces, no new privileges.
   - **Monitoring:** fail2ban; an audit log of every read of the wallet keys, with a Telegram alert when anything but
     the signer reads them; AIDE; automatic security updates.
   - **Keys:** it makes the trading and payout keys and prints their public keys (keep them private).
4. **Before closing the root session:** log in from the Mac over Tailscale, `ssh fly-admin@fly-vault`. Then run the
   second pass, which removes public SSH (the server then has no open port on the internet):
   `CLOSE_PUBLIC_SSH=1 bash deploy/bootstrap-host.sh "$(cat fly-release.pub)"`
5. **The treasury (your phone).** Open `https://fly-trader.app/owner.html` in Solflare's browser and connect.
   1. Paste the trading and payout keys that `fly-trader vault squads-setup` prints.
   2. "Create the treasury multisig": you are its only member, threshold 1, no time lock.
   3. "Create L1" (2 SOL a day, to the trading wallet only) and "Create L2" (1 SOL a week, claims).
   4. The page prints the three `custody.env` lines, and every check it shows must pass.
6. Fill in the settings (all mode 600):
   - `/srv/fly/custody.env`: the three lines from the owner page, plus `VAULT_BACKUP_RECIPIENT` (your `age1…`)
   - `/srv/fly/signer.env`: `SIGNER_RPC_URL`
   - `/srv/fly/vault.env`: `HELIUS_API_KEY`, `JUPITER_API_KEY`, `SOLANA_CHECK_RPC_URL`, `FUNDING_ADDRESSES`,
     `VAULT_ADDRESS`, `VAULT_TIMELOCK`, `VAULT_START_BLOCK`, `VAULT_IMPL_CODEHASHES`, `RELAY_SECRET`, `TELEGRAM_*`,
     `HEALTHCHECK_URL`, `VAULT_BACKUP_BUCKET`, `VAULT_BACKUP_HF_TOKEN`
   - `/etc/fly/update.json`: the Telegram fields
7. Start the updater: `sudo systemctl enable --now fly-update.timer; journalctl -u fly-update -f`. It installs only the
   release you pinned, builds the brain and signer images and starts them.
8. Check everything:
   - `sudo /usr/local/lib/fly/check-host.sh`: every control must pass (keys only in the signer, Tailnet Lock on, no
     public SSH…)
   - `sudo docker compose -p fly -f /srv/fly/docker-compose.yml exec fly fly-trader vault custody`: no problems
9. The console: `ssh -L 8501:127.0.0.1:8501 fly-admin@fly-vault`, then http://localhost:8501/vault.

## 4. Site (Gandi)
1. Build the upload bundle:
   ```
   cd web
   VITE_REOWN_PROJECT_ID=… VITE_VAULT_ADDRESS=… VITE_TIMELOCK_ADDRESS=… bash scripts/bundle_gandi.sh
   ```
   The bundle includes `owner.html`. It is not linked anywhere, it is noindex, and it can do nothing without your
   owner wallet.
2. Upload the contents of `build/gandi/` so they replace `site/` entirely.
3. Put `relay.json` next to `wsgi.py`, with the same secret as `RELAY_SECRET`, and restart the instance.
4. Probe the relay: `FLY_RELAY_SECRET=<secret> python web/relay/hmacauth.py https://fly-trader.app/api/fly/probe k1`.

## 5. Go live
1. Send your principal (5 SOL) to the **treasury** (vault 0, shown on the owner page) from an address in
   `FUNDING_ADDRESSES`. The fly refills its 2 SOL float from it through L1 on its own.
2. Check `fly-trader vault status` shows `deposits: 5000000000`. If the SOL came from another address:
   `fly-trader vault reclassify <sig> deposit`.
3. The fly trades live after the handover (at least 20 paper trades). Settlements run every Monday 00:00 UTC.
4. After each settlement, Telegram says what is owed. When it is more than L2 allows in a week, open the owner page
   and "Set L2 to this amount". Claims above the cap wait; they never fail.

## Emergency
- **Something is wrong (possible compromise):**
  1. On your phone, open the owner page and tap **Revoke L1 and L2 now**. From then on the server cannot move any SOL
     out of the treasury. Trading keeps only its float.
  2. Send `/panic` to the bot. The fly halts, the signer refuses buys, top-ups and claims, open positions are sold, and
     the float goes back to the treasury (L1 is not needed for that).
  3. If $FLY locking should stop too: MetaMask, Blockscout, FlyVault `pause()`.
- **A key-read, halt or setup-change alert:** look before resuming. Resuming is SSH-only:
  `fly-trader vault resume`, `fly-trader reset-circuit --kill`, `fly-trader resume-entries`, and in the signer
  container `python -m fly_trader.signer clear-panic`.
- **Rebuild after losing the server:**
  1. New server and new keys.
  2. On the owner page, add L1/L2 for the new keys (the old ones are revoked).
  3. Restore the ledger from backup. The treasury never moved.

## Day to day
- **New models or code:** run `publish_release.py` (`--models` to refresh the models). Model-only releases apply
  within 5 minutes; they cannot carry code (weights-only torch files, skops selectors). A release that changes **code**
  is held 24 h and Telegram tells you so:
  - to apply it now: `sudo python3 /usr/local/lib/fly/fly_update.py approve <seq>`
  - to stop it: `sudo python3 /usr/local/lib/fly/fly_update.py veto <seq>` (if you did not publish it, the release key
    is compromised: remove it from `/etc/fly/allowed_signers` and rotate)

  Docker or compose changes wait for `approve`.
- **Take principal out:** `fly-trader vault withdraw` prints how much you may take and how. You send it from the
  treasury in the Squads app with your owner wallet. The fly books and reports it.
- **`/status`** to the bot at any time. A daily report arrives at 09:00 UTC; if it does not, or healthchecks.io alerts,
  the server or the fly is down.
- **Restore from backups:**
  - Hot keys: `age -d -i fly-backup.key keys/<name>-<pubkey>.age`
  - Ledger: `age -d -i fly-backup.key ledger/<day>.dump.age | pg_restore …`

## Ledger (when you're home) — no redeploys
- **Solana:** in the Squads app, add the Ledger as a member, then remove Solflare (or keep both at threshold 2; the
  owner page then proposes and approves, and the other member executes). L1 and L2 stay as they are.
- **EVM:** move MetaMask's roles to the Ledger through the timelock (8 days):
  1. Print the calldata:
     ```
     cd contracts
     VAULT=<proxy> TIMELOCK=<timelock> OLD_OWNER=<MetaMask> NEW_OWNER=<Ledger> MODE=calldata \
       forge script script/HandOver.s.sol --rpc-url https://rpc.mainnet.chain.robinhood.com
     ```
  2. Send the printed "schedule" data from MetaMask to the timelock (Blockscout "Write contract", `scheduleBatch`, or
     any raw-data send).
  3. After 8 days, send the "execute" data from any wallet.
- **SSH and releases:** add the Ledger's SSH keys (ledger-agent) to `~fly-admin/.ssh/authorized_keys` and to
  `/etc/fly/allowed_signers` (principal `bryce`, `namespaces="fly-trader-release"`), then remove the Secretive keys.
