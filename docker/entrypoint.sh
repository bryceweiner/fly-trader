#!/bin/sh
# pid 1 of the container. With AUTO_UPDATE=1 (the default) the self-updater runs the bootstrap and the console and keeps
# both current from Hugging Face (fly_trader/ops/autoupdate.py); otherwise the bootstrap runs once and the console is pid 1.
set -e
cd /app
chmod +x docker/bootstrap.sh 2>/dev/null || true
if [ "${AUTO_UPDATE:-1}" = "1" ]; then
  exec python -m fly_trader.ops.autoupdate
fi
sh docker/bootstrap.sh
echo "fly-trader: console at http://localhost:${PORT:-8501}"
exec fly-trader ui --port "${PORT:-8501}"
