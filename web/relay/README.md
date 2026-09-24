# Vault relay (`/api/*`)

Standard-library WSGI app that passes stats and claims between the vault fly and browsers
(`docs/vault/SPEC.md` §3). `web/wsgi.py` sends `/api/` here and serves `site/` for everything else.
Needs Python ≥ 3.8 and nothing from pip.

## Layout on the Gandi Python instance

```
vhosts/default/            deploy tree (what wsgi.py lives in)
  wsgi.py
  relay/                   this package (tests/ may be left out)
  site/                    the built static site
  site-headers.json        optional, owned by the site build (CSP etc.)
  relay.json               secret config, never inside site/, never committed
<data dir outside the deploy tree>/relay.sqlite3   (+ -wal and -shm files next to it)
```

- **relay.json** goes next to `wsgi.py`, or anywhere if `FLY_RELAY_CONFIG` points at it. It is re-read when
  its mtime changes, so rotating keys needs no restart. If your deploy replaces the whole vhost directory,
  re-upload it after each deploy (keep it in the untracked bundle, never in git).
- **The database** must be on a writable local disk outside the deploy tree, so deploys never wipe it: use the
  instance's persistent data area (on Simple Hosting, something like `/srv/data/home/relay/relay.sqlite3`,
  shown as `lamp0/home/...` over SFTP; confirm with the probe). The *directory* must be writable because SQLite
  creates `-wal`/`-shm` files beside the database. A relative `db_path` is resolved against relay.json's folder.

```json
{"keys": {"k1": "<64 hex>"}, "db_path": "/srv/data/home/relay/relay.sqlite3", "domain": "fly-trader.app",
 "uri": "https://fly-trader.app/vault.html", "chain_id": 4663, "sol_chain": "mainnet", "claim_ttl_s": 900,
 "min_lamports": 2000000, "client_ip_header": null}
```

- Generate a key: `python -c "import secrets;print(secrets.token_hex(32))"`. The HMAC key is that string's
  UTF-8 bytes (not the decoded 32 bytes). Put the same id and secret in the fly's environment.
- Rotation: add `"k2"`, switch the fly to k2, then remove k1.
- Keep `claim_ttl_s` at 900: the fly refuses claims whose expiry is not issue + 900 s.
- A missing or invalid relay.json makes `/api/*` answer 503 `relay not configured` (the reason goes to the uWSGI
  log without secrets); the static site keeps working.

## First deploy: the probe checklist

Call the HMAC-protected probe from any machine that has the secret:

```sh
FLY_RELAY_SECRET=<hex> python web/relay/hmacauth.py https://fly-trader.app/api/fly/probe k1
```

It prints the status, response headers and
`{"python", "db_path", "writable", "wal", "pid", "remote_addr", "headers", "sqlite", "time", "client_ip", ...}`.

1. **Python**: `python` is ≥ 3.8.
2. **Writable**: `writable` is `true`. If not, fix `db_path` or the directory permissions.
3. **WAL**: `wal` is `true`. If it is false the file is probably on a network filesystem; move it to local disk.
4. **Process model**: call it several times. A changing `pid` means several uWSGI workers. That is fine, because
   all shared state is in SQLite, but it confirms the setup.
5. **Client IP**: compare `remote_addr` with your own public IP. If it is a proxy address (e.g. 10.x or
   127.0.0.1), find the header in `headers` that carries your IP (`HTTP_X_FORWARDED_FOR`, `HTTP_X_REAL_IP`, ...)
   and set `client_ip_header` to it. Its first comma-separated entry is used. Then check it cannot be spoofed:
   send `-H 'X-Forwarded-For: 1.2.3.4'` (e.g. with curl on `/api/claim/challenge`, or by adding the header in
   hmacauth.py) and look at `client_ip` in the probe. If it becomes 1.2.3.4, the proxy appends rather than replaces,
   the first entry is client-controlled, and the per-IP limits can be dodged. Prefer a single-value header set by
   the proxy (X-Real-IP) if there is one. Without this, every visitor shares one set of rate limits.
6. **No proxy caching of /api**: the probe's `time` and `pid` must change between calls, and the response must not
   carry `Age > 0` or `X-Cache: HIT`/`X-Varnish` with two ids. Do the same with `curl -i .../api/stats` twice.
   Every `/api` response sends `Cache-Control: no-store`, which Varnish's default rules respect.
7. **Clock**: `time` is within a few seconds of real time. HMAC requests are refused outside ±300 s, and the 401
   message states the server time.

## Behaviour notes

- Rate limits (token buckets in SQLite): challenge 30/h/IP; claim 10/h/IP and 5/h/EVM; reads 600/h/IP. IPv6
  clients are bucketed per /64. The per-EVM bucket is only spent by claims that pass every other check.
- The relay checks only syntax, nonces, owed and limits. The fly re-renders the texts and verifies both signatures.
- Tests: `.venv/bin/python -m pytest web/relay/tests -q` and `uvx --python 3.8 --with pytest pytest web/relay/tests -q`.
