# Stream relay

PumpAPI's firehose is held where it is fast, near its servers in Frankfurt. The relay forwards only what fly-trader's
aggregator uses to fly-trader over one connection: compressed, over TLS, protected by a token, and resumable. On a
dropped connection nothing is lost, within the relay's buffer.

PumpAPI's stream to fly-trader's machine falls behind and reconnects about once a minute, losing ~40 % of the trades.
The relay is the fix: it sends about 10 % of the events, with only the fields that are read, compressed.

- `relay.py`: the server. It needs only `websockets` and `orjson`: no fly-trader, no database, no keys.
- `../fly_trader/ingest/relay_feed.py`: the client in fly-trader, with `SOL_FEED=relay`.
  - With `SOL_FEED=relay`, the relay is the minutes' clock, and fly-trader's own PumpAPI connection stays as the backup. Each transaction is taken once.

## Server

**Size and place.** A small VPS in or near Frankfurt is enough: 1 vCPU, 1 GB RAM. Any provider works (Hetzner FSN, Netcup NBG and others).
- PumpAPI allows one connection per IP. The relay uses the server's IP, so fly-trader's own connection is unaffected.

**Name and certificate.** Point a DNS name (e.g. `relay.example.com`) at the server. Then, with port 80 open while the certificate is issued:

```sh
sudo useradd --system --home /opt/fly-stream-relay flyrelay
sudo mkdir -p /opt/fly-stream-relay /etc/fly-stream-relay
sudo certbot certonly --standalone -d relay.example.com \
  --deploy-hook "install -m 640 -g flyrelay /etc/letsencrypt/live/relay.example.com/fullchain.pem /etc/fly-stream-relay/ && \
                 install -m 640 -g flyrelay /etc/letsencrypt/live/relay.example.com/privkey.pem /etc/fly-stream-relay/ && \
                 systemctl restart fly-stream-relay"
```

The deploy hook copies the certificate to where the relay user can read it. It runs again on each renewal.

**Install.**

```sh
sudo cp relay.py requirements.txt /opt/fly-stream-relay/
sudo python3 -m venv /opt/fly-stream-relay/venv
sudo /opt/fly-stream-relay/venv/bin/pip install -r /opt/fly-stream-relay/requirements.txt
```

**Settings.** Write them to `/etc/fly-stream-relay.env` (mode 600, root). Make a token with `openssl rand -hex 32`.

```
RELAY_TOKEN=<64 hex characters>
RELAY_TLS_CERT=/etc/fly-stream-relay/fullchain.pem
RELAY_TLS_KEY=/etc/fly-stream-relay/privkey.pem
RELAY_PORT=8443
```

The optional settings:

| Setting | Default | What it does |
|---|---|---|
| `RELAY_BUFFER` | 300000 | events kept for a client that reconnects; about 30+ minutes, ~150 MB |
| `RELAY_BATCH_S` | 0.05 | how often batches go out, in seconds |
| `RELAY_UPSTREAM` | `wss://stream.pumpapi.io` | the stream the relay holds |

**Run.**

```sh
sudo cp fly-stream-relay.service /etc/systemd/system/
sudo systemctl daemon-reload && sudo systemctl enable --now fly-stream-relay
sudo ufw allow 8443/tcp            # and 80/tcp for certbot's renewals (or use its DNS challenge)
journalctl -u fly-stream-relay -f  # every minute: events in and out, clients, PumpAPI's state
```

**Docker, instead of the venv and systemd:**

```sh
docker build -t fly-stream-relay .
docker run -d --name fly-stream-relay --restart=always -p 8443:8443 --env-file /etc/fly-stream-relay.env \
  --user "$(id -u flyrelay):$(id -g flyrelay)" -v /etc/fly-stream-relay:/etc/fly-stream-relay:ro fly-stream-relay
```

**Check.** Run this from anywhere; `/health` needs no token:

```sh
curl https://relay.example.com:8443/health
```

`"ok": true` means PumpAPI is connected and events are flowing.

## fly-trader

Add to `.env`:

```
STREAM_RELAY_URL=wss://relay.example.com:8443/stream
STREAM_RELAY_TOKEN=<the same token>
```

Then test the relay without switching anything:

```sh
.venv/bin/python -m fly_trader relay-check --seconds 60
```

It prints the events per second and their kinds, the lag behind PumpAPI's stamps, and any gaps or reconnects.

**Switching over** is a separate step, taken only when Bryce says so:
1. Set `SOL_FEED=relay`.
2. Restart the console once no live position is open.
3. Check `pumpstream_status`: `"source": "relay"` means the relay is the minutes' clock.

To go back, set `SOL_FEED=pumpapi` (or remove it) and restart the console.
