#!/usr/bin/env bash
# fly-trader in one command (Linux, macOS, and Windows under WSL 2; INSTALL.md walks Windows through WSL 2 and Docker
# Desktop first): the two API keys into .env, an NVIDIA GPU when Docker can reach one, the build and first start, the bot
# wallet, the dashboard. Run it again any time: it keeps the keys, the wallet and the data it finds.
set -euo pipefail
umask 077
cd "$(dirname "$0")"

URL="http://localhost:8501"
CUDA_FILES="docker-compose.yml:docker-compose.cuda.yml"
CLONE="cd ~ && git clone -b distribution https://github.com/bryceweiner/fly-trader.git && cd fly-trader && ./install.sh"

say() { printf '\n\033[1m==> %s\033[0m\n' "$*"; }
die() { printf '\n\033[1;31mfly-trader install: %s\033[0m\n' "$*" >&2; exit 1; }

# .env is bind-mounted into the running container as a single file, so it is rewritten in place (the same inode), never
# replaced by a rename: the container would keep the old file, and what it appends there (the wallet key) would be lost
env_get() { sed -n "s/^$1=//p" .env | tail -n 1; }
env_set() {
  local tmp; tmp=$(mktemp)
  awk -v k="$1" -v v="$2" 'index($0, k "=") == 1 { if (!done) print k "=" v; done = 1; next } { print } END { if (!done) print k "=" v }' .env >"$tmp"
  cat "$tmp" >.env; rm -f "$tmp"
}

# --------------------------------------------------------------------------------------------- where, and Docker
WSL=0
if grep -qi microsoft /proc/sys/kernel/osrelease 2>/dev/null; then
  WSL=1
  case "$PWD" in
    /mnt/[a-zA-Z]/*) die "this folder is on a Windows drive ($PWD), which Docker reads slowly. Clone it inside Ubuntu instead:
  $CLONE" ;;
  esac
fi
if ! command -v docker >/dev/null 2>&1; then
  [ "$WSL" = 1 ] && die "Docker is not reachable from Ubuntu: start Docker Desktop, then Settings > Resources > WSL integration > Ubuntu on > Apply & restart"
  die "Docker is not installed (https://docs.docker.com/get-docker/)"
fi
docker compose version >/dev/null 2>&1 || die "Docker Compose v2 is missing: 'docker compose version' must answer"
docker info >/dev/null 2>&1 || die "Docker does not answer: start Docker Desktop (Windows, macOS) or the docker service (Linux, where your user must be in the docker group)"
grep -q $'\r' docker/entrypoint.sh docker/bootstrap.sh && die "docker/*.sh have Windows line endings and the container would not start. Clone again inside Ubuntu (or with Linux's git):
  $CLONE"

# ----------------------------------------------------------------------------------------------- the API keys
[ -f .env ] || { cp .env.example .env; echo "created .env from .env.example"; }

# 200: accepted; 401/403: refused; anything else (offline, rate-limited, an outage): not judged
helius_status() { curl -s -o /dev/null -m 15 -w '%{http_code}' -H 'Content-Type: application/json' \
  -d '{"jsonrpc":"2.0","id":1,"method":"getHealth"}' "https://mainnet.helius-rpc.com/?api-key=$1" || true; }
jupiter_status() { curl -s -o /dev/null -m 15 -w '%{http_code}' -H "x-api-key: $1" \
  "https://api.jup.ag/tokens/v2/search?query=So11111111111111111111111111111111111111112" || true; }

ask_key() {  # NAME label signup-url check
  local name=$1 label=$2 where=$3 check=$4 v code tries=0
  v=$(env_get "$name")
  while :; do
    if [ -z "$v" ]; then
      printf '%s (from %s): ' "$label" "$where"
      read -r v
      v=$(printf '%s' "$v" | tr -d '[:space:]')
      [ -n "$v" ] || die "$name is needed; run ./install.sh again when you have it"
    fi
    code=$("$check" "$v")
    case "$code" in
      200) echo "$label: accepted"; break ;;
      401|403) echo "$label: refused ($code). Paste it again."; v=""; tries=$((tries + 1))
               [ "$tries" -lt 3 ] || die "$label refused three times" ;;
      *) echo "$label: could not be checked (HTTP ${code:-none}); keeping it"; break ;;
    esac
  done
  [ "$(env_get "$name")" = "$v" ] || env_set "$name" "$v"
}
say "API keys"
ask_key HELIUS_API_KEY "Helius API key" "https://dashboard.helius.dev" helius_status
ask_key JUPITER_API_KEY "Jupiter API key" "https://portal.jup.ag" jupiter_status

# ------------------------------------------------------------------------------------------ an NVIDIA GPU, if any
if [ -z "$(env_get COMPOSE_FILE)" ] && { command -v nvidia-smi >/dev/null 2>&1 || [ -x /usr/lib/wsl/lib/nvidia-smi ]; }; then
  say "NVIDIA GPU"
  if docker run --rm --gpus all --entrypoint true postgres:17 >/dev/null 2>&1; then
    printf 'Docker can reach an NVIDIA GPU. Run the fly on it? The CUDA image is several GB larger. [Y/n] '
    read -r a
    case "$a" in
      [nN]*) echo "staying on the CPU" ;;
      *) env_set COMPOSE_FILE "$CUDA_FILES"; echo "added COMPOSE_FILE=$CUDA_FILES to .env" ;;
    esac
  else
    echo "There is an NVIDIA GPU, but Docker cannot reach it; staying on the CPU."
    echo "(Windows: update the NVIDIA driver, run 'wsl --update', restart Docker Desktop. Linux: install the NVIDIA Container Toolkit.)"
  fi
fi

# ------------------------------------------------------------------------------------------- build and start
wait_healthy() {
  local id st health status restarts base n=0 last=""
  say "Waiting for the trader to report healthy (the first start downloads ~300 MB and installs the models)"
  id=$(docker compose ps -q fly)
  [ -n "$id" ] || die "the fly container was not created; see 'docker compose logs fly'"
  base=$(docker inspect -f '{{.RestartCount}}' "$id")
  while :; do
    st=$(docker inspect -f '{{.State.Status}} {{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}} {{.RestartCount}}' "$id" 2>/dev/null) \
      || die "the fly container is gone; see 'docker compose logs fly'"
    read -r status health restarts <<<"$st"
    [ "$health" = healthy ] && { printf '\n'; return 0; }
    case "$status" in
      exited|dead) printf '\n'; docker compose logs --tail 40 fly; die "the trader stopped (its last log lines are above)" ;;
    esac
    [ $((restarts - base)) -lt 3 ] || { printf '\n'; docker compose logs --tail 40 fly; die "the trader keeps restarting (its last log lines are above)"; }
    # unhealthy can pass: a first start that installs a newer release outlasts the health check's grace period
    [ "$health" = "$last" ] || { [ -z "$last" ] || printf '\n'; printf '%s ' "$health"; last=$health; }
    n=$((n + 1)); [ "$n" -lt 360 ] || { printf '\n'; die "not healthy after 30 minutes; 'docker compose logs -f fly' says why"; }
    printf '.'; sleep 5
  done
}

say "Building and starting (the first build takes several minutes)"
docker compose up -d --build
wait_healthy

# ----------------------------------------------------------------------------------------------- the bot wallet
pub=""
if grep -q '^BOT_PRIVATE_KEY=' .env; then
  [ -n "$(env_get BOT_PRIVATE_KEY)" ] || die ".env has an empty BOT_PRIVATE_KEY= line: delete that line and run ./install.sh again"
  say "Bot wallet (already in .env)"
  docker compose exec -T fly fly-trader wallet show || true
else
  say "Creating the bot wallet"
  pub=$(docker compose exec -T fly fly-trader wallet new | tail -n 1) || die "wallet new failed (its message is above)"
  [ -n "$(env_get BOT_PRIVATE_KEY)" ] || die "wallet new did not write the key into .env (it said: $pub)"
  echo "address: $pub"
  echo "restarting the trader so it loads the key"
  docker compose restart fly
  wait_healthy
fi

# ---------------------------------------------------------------------------------------------------- done
say "fly-trader is running"
echo "Dashboard: $URL"
[ -z "$pub" ] || echo "Bot wallet: $pub"
echo
echo "Two things only you can do:"
echo "  1. Back up .env now: it holds the only copy of the wallet's key."
[ "$WSL" = 0 ] || echo "     'explorer.exe .' opens this folder in File Explorer: copy .env to a USB drive, not into OneDrive."
echo "  2. Send 0.5 SOL to the bot wallet ('docker compose exec fly fly-trader wallet show' prints its address)."
echo
echo "After a ~24 h warm-up it trades paper until 20 trades have closed, then the wallet. For paper only, set"
echo "LIVE_ENABLED=0 in .env and run: docker compose up -d --force-recreate fly"

if [ "$WSL" = 1 ]; then explorer.exe "$URL" >/dev/null 2>&1 || true
elif command -v open >/dev/null 2>&1; then open "$URL" >/dev/null 2>&1 || true
elif command -v xdg-open >/dev/null 2>&1; then xdg-open "$URL" >/dev/null 2>&1 || true
fi
