# Installing fly-trader on Windows (Docker)

> **This software signs real transactions from a wallet you fund. It can lose everything in that wallet, quickly.**
> Run it only with money you can lose entirely, on a wallet used for nothing else. Read
> [the distribution README](https://github.com/bryceweiner/fly-trader/blob/distribution/README.md) first: it says what
> the trader does and what its rails are.

You end up with two containers in Docker Desktop, the database and the trader, and the console at
<http://localhost:8501>. Every command after step 2 runs in an **Ubuntu terminal** (WSL 2), not in PowerShell. That is
Docker's recommended setup on Windows: the trader's data folder is much faster on Linux's file system, and the
container's start scripts need the Linux line endings that a Windows checkout would change.

## What you need

- Windows 11, or Windows 10 22H2, 64-bit ([Docker's current list](https://docs.docker.com/desktop/setup/install/windows-install/)),
  and an administrator account for the installs.
- Virtualization turned on: Task Manager → Performance → CPU shows **Virtualization: Enabled**. If it does not, turn on
  Intel VT-x or AMD SVM in the PC's UEFI/BIOS setup.
- 16 GB of memory recommended. ~2 GB of disk on drive C: (Robinhood Chain, if you turn it on, adds ~30 GB).
- A [Helius](https://dashboard.helius.dev) API key (the free tier is enough) and a [Jupiter](https://portal.jup.ag) API key.
- 0.5 SOL you are prepared to lose. The shipped caps are sized for 0.5.

Docker Desktop is free for personal use, education and small businesses; larger companies need a paid Docker
subscription.

## 1. Install WSL 2 and Ubuntu

Open **PowerShell as Administrator** (Start → type `PowerShell` → Run as administrator):

```powershell
wsl --install
```

Restart when it asks. Ubuntu opens after the restart (or start it from the Start menu) and asks you to choose a Linux
user name and password. Remember the password: `sudo` asks for it.

If WSL was already installed, run `wsl --update` and then `wsl --install -d Ubuntu` instead. `wsl -l -v` must show
Ubuntu with VERSION **2**.

## 2. Install Docker Desktop

Download it from [docker.com](https://www.docker.com/products/docker-desktop/), or in PowerShell:

```powershell
winget install -e --id Docker.DockerDesktop
```

Keep **Use WSL 2 instead of Hyper-V** ticked during the install. Start Docker Desktop, then in its Settings:

- **Resources → WSL integration**: turn on **Ubuntu**, then **Apply & restart**.
- **General**: keep **Start Docker Desktop when you sign in** on, so the trader comes back after a reboot.

Open Ubuntu and check that both commands print a version:

```bash
docker version
docker compose version
```

**Memory (only if the PC has less than 32 GB).** WSL 2 gives Docker half the PC's memory by default. To give it more,
in PowerShell open the WSL settings file:

```powershell
notepad "$env:USERPROFILE\.wslconfig"
```

Put this in it, leaving about 4 GB for Windows (12 GB on a 16 GB PC), and save:

```ini
[wsl2]
memory=12GB
```

Then run `wsl --shutdown` in PowerShell and start Docker Desktop again.

## 3. Download fly-trader

In **Ubuntu**:

```bash
cd ~
git clone -b distribution https://github.com/bryceweiner/fly-trader.git
cd fly-trader
```

Keep it in your Ubuntu home folder (`~`). Do not clone it under `/mnt/c/...`, and do not clone it with Git for Windows:
the first makes the trader's data folder slow, and the second converts line endings so the container does not start.
To see the folder in File Explorer, run `explorer.exe .`.

## 4. Add your API keys

```bash
cp .env.example .env
nano .env
```

Paste your keys after `HELIUS_API_KEY=` and `JUPITER_API_KEY=`, with no quotes and no spaces. Save with **Ctrl+O**,
**Enter**, and quit with **Ctrl+X**.

- Do not add a `BOT_PRIVATE_KEY=` line yourself; step 6 writes it.
- `LIVE_ENABLED=1` is the shipped setting: once the wallet is funded and 20 paper trades have closed, the trader trades
  the wallet. Set `LIVE_ENABLED=0` to run paper only.
- Keep each value bare: Docker keeps a `# comment` after a value as part of the value.

## 5. Start it

```bash
docker compose up -d --build
```

With an NVIDIA GPU, use the command in [NVIDIA GPU](#nvidia-gpu-optional) instead.

The first start takes several minutes. It builds the image, creates the database, downloads the wallet-skill table
(~300 MB, checked against its pinned sha256), installs the models and starts the market feed and the trading engine.
To watch it, run `docker compose logs -f fly`. **Ctrl+C** stops the log, not the trader. It is ready when
`docker compose ps` shows `fly` as `healthy`.

## 6. Create the bot wallet

```bash
docker compose exec fly fly-trader wallet new
```

It prints the wallet's public key and writes the secret key into `.env`.

**Back up `.env` now. It is the only copy of that key.** Run `explorer.exe .` and copy `.env` to a USB drive, or put
its `BOT_PRIVATE_KEY` line in a password manager. Do not copy it into a OneDrive folder (Desktop, Documents and Pictures
often are one): that uploads the key to the cloud.

## 7. Fund it

Send **0.5 SOL** to the printed address, then restart the trader so it picks up the key:

```bash
docker compose restart fly
```

## 8. Open the console

Open <http://localhost:8501> in any Windows browser and watch the **Overview** page.

From here it runs on its own. The first ~24 hours are a warm-up (it needs a day of candles before it scores anything).
Then it trades paper until 20 trades have closed, and then it trades the wallet.

The console listens on `127.0.0.1` only. Do not expose it (no port forwarding, no tunnels to the internet): it can start
and stop the engine and clear the rails.

## Keep it running

- **No sleep.** While Windows sleeps the trader is frozen: it buys nothing, and open positions are not sold on schedule
  until the PC wakes. In Settings → System → Power, set sleep to **Never** when plugged in.
- **After a restart, sign in.** The containers start by themselves, but only once Docker Desktop is running, and Docker
  Desktop starts when you sign in to Windows. That includes restarts for Windows updates.

## Day to day

Run these in Ubuntu, from the install folder (`cd ~/fly-trader`):

| Want to | Do |
|---|---|
| See what it is doing | <http://localhost:8501>: **Overview**, **Trades**, **Safety & wallet**, **Processes** |
| See the logs | `docker compose logs -f fly` |
| Pause new entries | Safety & wallet → **Pause entries**, or `docker compose exec fly fly-trader pause-entries` |
| Change a setting | `nano .env`, then `docker compose restart fly` |
| Stop everything | `docker compose down`. **Open positions stay open in the wallet**; start again and it sells them on schedule |
| Update by hand | `git pull && docker compose up -d --build` (signed releases also install themselves within the hour; `AUTO_UPDATE=0` in `.env` turns that off) |
| Start again from nothing | `docker compose down -v && sudo rm -rf data/` deletes the database, the fly's learned state and the signer's ledger; `.env` (your key) stays |

Tuning, Robinhood Chain and updates are described in
[the distribution README](https://github.com/bryceweiner/fly-trader/blob/distribution/README.md). Every command there
works as written in the Ubuntu terminal.

## NVIDIA GPU (optional)

The CPU image trades fine; a GPU makes the once-a-minute scoring and the fly's learning a little faster.

1. Install the current NVIDIA driver for Windows (Game Ready or Studio, 560 or newer). That is all: Docker Desktop
   passes the GPU through WSL 2. Install no NVIDIA driver inside Ubuntu, and no NVIDIA Container Toolkit.
2. In PowerShell: `wsl --update`, then restart Docker Desktop.
3. In step 5, start with the CUDA override instead:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose.cuda.yml up -d --build
   ```

   Any later `docker compose up` needs the same two `-f` files. To make the plain commands use them, add this line to `.env`:

   ```
   COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml
   ```

Check which device the fly is on:

```bash
docker compose logs fly | grep "fly-trader: device"
```

`cuda:0 (NVIDIA …)` means the GPU. `cpu (… threads)` means the container did not get it. The console's Model &
training page shows the same.

**AMD GPUs** are not reachable from Docker Desktop on Windows: `docker-compose.rocm.yml` needs Linux's `/dev/kfd`, so do
not use it here. The CPU image runs instead.

## Troubleshooting

| You see | Do |
|---|---|
| `0x80370102` while installing WSL | Virtualization is off: turn on Intel VT-x / AMD SVM in the UEFI/BIOS |
| `docker: command not found`, or "could not be found in this WSL 2 distro" | Start Docker Desktop; Settings → Resources → WSL integration → **Ubuntu** on → Apply & restart |
| `exec /app/docker/entrypoint.sh: no such file or directory`, or `set: Illegal option -` in the logs | Windows line endings: the folder was cloned by Git for Windows or under `/mnt/c`. Move `.env` somewhere safe, delete the folder, clone again in Ubuntu (step 3), put `.env` back, `docker compose up -d --build` |
| Everything is slow | The folder is under `/mnt/c`. `docker compose down`, clone again into `~` (step 3), copy `.env` and `data/` across (`sudo cp -a /mnt/c/…/fly-trader/data ~/fly-trader/`), start again. The database is kept |
| `could not select device driver "nvidia"` | Update the NVIDIA driver, run `wsl --update`, restart Docker Desktop; or start without the CUDA override |
| The console does not open | `docker compose ps`: `fly` must be `Up` (the first start takes minutes); `docker compose logs -f fly` says why |
