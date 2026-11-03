# Installing fly-trader on Windows (Docker)

> **This software signs real transactions from a wallet you fund. It can lose everything in that wallet, quickly.**
> Run it only with money you can lose entirely, on a wallet used for nothing else. Read
> [the distribution README](https://github.com/bryceweiner/fly-trader/blob/distribution/README.md) first: it says what
> the trader does and what its rails are.

You end up with two containers in Docker Desktop, the database and the trader, and the dashboard at
<http://localhost:8501>. Steps 1 and 2 install WSL 2 and Docker Desktop. After that, every command runs in an
**Ubuntu terminal**, not in PowerShell, and one script (`install.sh`) does the rest.

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

Open **Ubuntu** from the Start menu:

```bash
cd ~
git clone -b distribution https://github.com/bryceweiner/fly-trader.git
cd fly-trader
```

Keep it in your Ubuntu home folder (`~`), not under `/mnt/c/...`: the trader's data folder is much faster on Linux's
file system, and `install.sh` refuses to run from a Windows drive. To see the folder in File Explorer, run
`explorer.exe .`.

## 4. Run the installer

```bash
./install.sh
```

It:

1. checks that Docker answers and that the folder is on Linux's file system;
2. creates `.env` and asks for your Helius and Jupiter API keys (paste each and press Enter). It checks each key with
   its provider and asks again if one is refused;
3. offers the NVIDIA GPU when the PC has one and Docker can reach it;
4. builds the image and starts the database and the trader. The first build takes several minutes. On its first start
   the trader downloads the wallet-skill table (~300 MB, checked against its pinned sha256) and installs the models;
5. waits until the trader reports healthy;
6. creates the bot wallet, writes its key into `.env`, restarts the trader so it loads the key, and prints the
   wallet's address;
7. opens the dashboard in your Windows browser.

It is safe to run again: it keeps the keys, the wallet and the data it finds.

## 5. Back up `.env`

**`.env` holds the only copy of the wallet's key.** Run `explorer.exe .` and copy `.env` to a USB drive, or put its
`BOT_PRIVATE_KEY` line in a password manager. Do not copy it into a OneDrive folder (Desktop, Documents and Pictures
often are one): that uploads the key to the cloud.

## 6. Fund the wallet

Send **0.5 SOL** to the address the installer printed (`docker compose exec fly fly-trader wallet show` prints it
again).

From here it runs on its own. The first ~24 hours are a warm-up (it needs a day of candles before it scores anything).
Then it trades paper until 20 trades have closed, and then it trades the wallet. To run paper only, set
`LIVE_ENABLED=0` in `.env` and apply it (see [Change a setting](#day-to-day)).

## The dashboard

Open <http://localhost:8501> in any Windows browser, or click the `8501:8501` port of the `fly` container in Docker
Desktop. Watch the **Overview** page.

### From another device on your network

The dashboard has no login: whoever can open it can stop the engine and clear the rails. Open it to other devices
only on a network you trust, such as your home network. Never forward it through your router to the internet.

1. Find the PC's address: in PowerShell, `ipconfig`, then **IPv4 Address** under the Wi-Fi or Ethernet adapter, for
   example `192.168.1.20`.
2. Set these two lines in `.env` (`nano .env`; save with **Ctrl+O**, **Enter**, quit with **Ctrl+X**). A `.env` made
   before October 25, 2026 has no `CONSOLE_PUBLISH` line: add it.

   ```
   CONSOLE_PUBLISH=0.0.0.0
   CONSOLE_ALLOWED_HOSTS=192.168.1.20
   ```

   `CONSOLE_ALLOWED_HOSTS` lists every name you will type in the browser, comma-separated: add the PC's name or a
   Tailscale name the same way.
3. Apply it: `docker compose up -d --force-recreate fly`.
4. If Windows Firewall asks whether **Docker Desktop Backend** may accept connections, allow it on **Private** networks
   (and mark your home network as Private: Settings → Network & internet → your network → Private).

Then open `http://192.168.1.20:8501` on the other device. Ask your router to keep giving the PC that address (a DHCP
reservation), or the address will change.

## Keep it running

- **No sleep.** While Windows sleeps the trader is frozen: it buys nothing, and open positions are not sold on schedule
  until the PC wakes. In Settings → System → Power, set sleep to **Never** when plugged in.
- **After a restart, sign in.** The containers start by themselves, but only once Docker Desktop is running, and Docker
  Desktop starts when you sign in to Windows. That includes restarts for Windows updates.

## Day to day

Run these in Ubuntu, from the install folder (`cd ~/fly-trader`):

| Want to | Do |
|---|---|
| See what it is doing | The dashboard: **Overview**, **Trades**, **Safety & wallet**, **Processes** |
| See the logs | `docker compose logs -f fly` (**Ctrl+C** stops the log, not the trader) |
| Pause new entries | Safety & wallet → **Pause entries**, or `docker compose exec fly fly-trader pause-entries` |
| Change a setting | `nano .env`, then `docker compose up -d --force-recreate fly`. A plain `docker compose restart` keeps the old values |
| Stop everything | `docker compose down`. **Open positions stay open in the wallet**; start again (`docker compose up -d`) and it sells them on schedule |
| Update by hand | `git pull && docker compose up -d --build` (signed releases also install themselves within the hour; `AUTO_UPDATE=0` in `.env` turns that off) |
| Start again from nothing | `docker compose down -v && sudo rm -rf data/` deletes the database, the fly's learned state and the signer's ledger; `.env` (your key) stays |

Tuning, Robinhood Chain and updates are described in
[the distribution README](https://github.com/bryceweiner/fly-trader/blob/distribution/README.md). Every command there
works as written in the Ubuntu terminal.

## NVIDIA GPU (optional)

The CPU image trades fine; a GPU makes the once-a-minute scoring and the fly's learning a little faster.

Install the current NVIDIA driver for Windows (Game Ready or Studio, 560 or newer), run `wsl --update` in PowerShell
and restart Docker Desktop. That is all: Docker Desktop passes the GPU through WSL 2. Install no NVIDIA driver inside
Ubuntu, and no NVIDIA Container Toolkit.

`install.sh` then offers the GPU and, if you accept, adds this line to `.env`, so every `docker compose` command builds
and starts the CUDA image:

```
COMPOSE_FILE=docker-compose.yml:docker-compose.cuda.yml
```

To switch an existing install, add that line yourself and run `docker compose up -d --build`. To go back to the CPU,
delete it and run the same command. To see which device the fly is on:

```bash
docker compose logs fly | grep "fly-trader: device"
```

`cuda:0 (NVIDIA …)` means the GPU, `cpu (… threads)` the CPU. The dashboard's Model & training page shows the same.

**AMD GPUs** are not reachable from Docker Desktop on Windows: `docker-compose.rocm.yml` needs Linux's `/dev/kfd`, so do
not use it here. The CPU image runs instead.

## Troubleshooting

| You see | Do |
|---|---|
| `0x80370102` while installing WSL | Virtualization is off: turn on Intel VT-x / AMD SVM in the UEFI/BIOS |
| `docker: command not found`, "could not be found in this WSL 2 distro", or install.sh says Docker does not answer | Start Docker Desktop; Settings → Resources → WSL integration → **Ubuntu** on → Apply & restart |
| install.sh says the folder is on a Windows drive | Clone again inside Ubuntu (step 3). If the old folder already ran, `docker compose down` there first and copy its `.env` and `data/` across (`sudo cp -a /mnt/c/…/fly-trader/data ~/fly-trader/`); the database is kept |
| `exec /app/docker/entrypoint.sh: no such file or directory`, or `set: Illegal option -` in the logs | Windows line endings, from a copy made before October 25, 2026 or by a tool that converts them. Keep `.env`, clone again (step 3), `docker compose up -d --build` |
| `could not select device driver "nvidia"` | Update the NVIDIA driver, run `wsl --update`, restart Docker Desktop; or delete the `COMPOSE_FILE` line from `.env` to run on the CPU |
| The dashboard does not open | `docker compose ps`: `fly` must be `Up` (the first start takes minutes); `docker compose logs -f fly` says why |
| The dashboard says `host '…' is not allowed` | Add the name you typed to `CONSOLE_ALLOWED_HOSTS` in `.env`, then `docker compose up -d --force-recreate fly` |
