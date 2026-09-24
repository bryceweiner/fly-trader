#!/usr/bin/env bash
# One-time setup of the vault fly's server (Ubuntu 24.04, e.g. Netcup RS 2000 G12). Run as root:
#   bash bootstrap-host.sh "<your ssh-ed25519 RELEASE public key line>"
# It installs Docker, locks the box down (SSH keys only, firewall: SSH in, nothing else), installs the updater with
# YOUR release key as the only trusted signer, and lays out /srv/fly. It does not create the wallet or start the fly.
set -euo pipefail
RELEASE_PUBKEY="${1:?usage: bootstrap-host.sh \"ssh-ed25519 AAAA... release-key\"}"

apt-get update
apt-get install -y ca-certificates curl ufw unattended-upgrades python3 openssh-client
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt-get update
apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
dpkg-reconfigure -f noninteractive unattended-upgrades

# SSH: keys only (make sure your key is in ~/.ssh/authorized_keys BEFORE running this)
sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/; s/^#\?PermitRootLogin .*/PermitRootLogin prohibit-password/' /etc/ssh/sshd_config
systemctl reload ssh || systemctl reload sshd
ufw default deny incoming
ufw default allow outgoing
ufw allow OpenSSH
ufw --force enable

# layout
install -d -m 0750 /srv/fly /srv/fly/data /srv/fly/logs /srv/fly/releases
install -d -m 0700 /srv/fly/secrets
install -d -m 0755 /etc/fly /usr/local/lib/fly
echo "bryce ${RELEASE_PUBKEY}" > /etc/fly/allowed_signers
chmod 0644 /etc/fly/allowed_signers
# the trading wallet: an ed25519 seed made HERE, never copied off this box in plaintext (the fly's backup job writes an
# age-encrypted copy for you). fly_trader.chain.keys accepts a base58 32-byte seed.
if [ ! -e /srv/fly/secrets/bot_key ]; then
  umask 077
  openssl genpkey -algorithm ed25519 -out /srv/fly/secrets/bot_key.pem
  python3 - <<'PY'
import subprocess
A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58(b):
    n = int.from_bytes(b, "big"); s = ""
    while n: n, r = divmod(n, 58); s = A[r] + s
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + s
priv = subprocess.run(["openssl", "pkey", "-in", "/srv/fly/secrets/bot_key.pem", "-outform", "DER"], check=True, capture_output=True).stdout[-32:]
pub = subprocess.run(["openssl", "pkey", "-in", "/srv/fly/secrets/bot_key.pem", "-pubout", "-outform", "DER"], check=True, capture_output=True).stdout[-32:]
open("/srv/fly/secrets/bot_key", "x").write(b58(priv) + "\n")
print("vault wallet address:", b58(pub))
PY
  shred -u /srv/fly/secrets/bot_key.pem
  chmod 0400 /srv/fly/secrets/bot_key
fi
HERE="$(cd "$(dirname "$0")" && pwd)"
install -m 0755 "$HERE/fly_update.py" /usr/local/lib/fly/fly_update.py
[ -f /etc/fly/update.json ] || install -m 0600 "$HERE/update.json.example" /etc/fly/update.json
[ -f /srv/fly/vault.env ] || install -m 0600 "$HERE/vault.env.example" /srv/fly/vault.env
install -m 0644 "$HERE/docker-compose.vault.yml" /srv/fly/docker-compose.yml
install -m 0644 "$HERE/fly-update.service" /etc/systemd/system/fly-update.service
install -m 0644 "$HERE/fly-update.timer" /etc/systemd/system/fly-update.timer
systemctl daemon-reload
echo
echo "Next (see deploy/VAULT.md):"
echo "  1. edit /srv/fly/vault.env and /etc/fly/update.json (Telegram)"
echo "  2. systemctl enable --now fly-update.timer   # pulls and builds the first release"
echo "  3. create the wallet key, then restart:  see VAULT.md 'wallet'"
