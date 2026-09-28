#!/usr/bin/env bash
# One-time setup of the vault fly's server (Ubuntu 24.04, e.g. Netcup RS 2000 G12), hardened along the lines of the
# CIS Ubuntu 24.04 benchmark for a single-purpose host that holds a hot wallet. Run as root, in two passes:
#
#   1. TS_AUTHKEY=tskey-auth-... RECOVERY_PUBKEY="ssh-ed25519 AAAA... recovery" EXPECT_MANIFEST=<sha256 of release.json> \
#        ADMIN_USER=fly-admin bash bootstrap-host.sh "<release public key line: Secretive's ecdsa-sha2-nistp256 ...>"
#      then, from your Mac, log in over Tailscale:  ssh fly-admin@fly-vault   (MagicDNS name or the 100.x address)
#   2. CLOSE_PUBLIC_SSH=1 bash bootstrap-host.sh "<same release key>"
#      removes the public SSH rule: after it the server has no open port on the internet at all.
#
# ADMIN_USER (default fly-admin) becomes the only account that can log in over SSH (key only, sudo without a
# password; root login is switched off). If it has no ~/.ssh/authorized_keys yet, root's are copied to it first, and
# the script stops before touching SSH if neither exists. Keep this root session open until you have logged in as
# ADMIN_USER from a second terminal. deploy/check-host.sh verifies every setting afterwards.
#
# RELEASE_PUBKEY / RECOVERY_PUBKEY: the two keys whose signatures the updater accepts (principals bryce and
# bryce-recovery), valid for the release namespace only. EXPECT_MANIFEST pins the first release the updater may
# install (tools/publish_release.py prints it as the manifest sha256); until it is set the updater installs nothing.
# TS_AUTHKEY: a one-off, pre-authorised key tagged tag:fly from the Tailscale admin console; used once, not stored.
#
# Idempotent: safe to run again (keys and edited config files are never overwritten).
set -euo pipefail
RELEASE_PUBKEY="${1:?usage: ADMIN_USER=fly-admin bootstrap-host.sh \"ecdsa-sha2-nistp256 AAAA... release-key\"}"
RECOVERY_PUBKEY="${RECOVERY_PUBKEY:-}"
EXPECT_MANIFEST="${EXPECT_MANIFEST:-}"
TS_AUTHKEY="${TS_AUTHKEY:-}"
CLOSE_PUBLIC_SSH="${CLOSE_PUBLIC_SSH:-0}"
ADMIN_USER="${ADMIN_USER:-fly-admin}"
# container users (the images' USER lines); with userns-remap each is REMAP_UID + id on the host
BRAIN_UID=10001; SIGNER_UID=10002; FLY_GID=10000
HERE="$(cd "$(dirname "$0")" && pwd)"
export DEBIAN_FRONTEND=noninteractive
step() { printf '\n== %s\n' "$*"; }

# ---------------------------------------------------------------- 1. packages
step "packages"
apt-get update -q
apt-get install -y -q --no-install-recommends ca-certificates curl ufw unattended-upgrades python3 openssh-client openssl \
  fail2ban python3-systemd auditd audispd-plugins aide aide-common lynis apparmor-utils systemd-timesyncd needrestart \
  libpam-tmpdir debsums apt-show-versions apt-listchanges acct sysstat
install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg -o /etc/apt/keyrings/docker.asc
echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" > /etc/apt/sources.list.d/docker.list
apt-get update -q
apt-get install -y -q docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
# Tailscale: SSH reaches this box only over your tailnet (WireGuard); its repo key is pinned like Docker's
curl -fsSL "https://pkgs.tailscale.com/stable/ubuntu/$(. /etc/os-release && echo "$VERSION_CODENAME").noarmor.gpg" -o /etc/apt/keyrings/tailscale.gpg
echo "deb [signed-by=/etc/apt/keyrings/tailscale.gpg] https://pkgs.tailscale.com/stable/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") main" > /etc/apt/sources.list.d/tailscale.list
apt-get update -q
apt-get install -y -q tailscale
# nothing on this box needs snaps; fewer daemons, fewer holes
if dpkg -l snapd >/dev/null 2>&1; then apt-get purge -y -q snapd; fi
# no mail server: nothing here sends mail (alerts go to Telegram), and an MTA is one more listening daemon
if dpkg -l postfix 2>/dev/null | grep -q '^ii'; then apt-get purge -y -q postfix; fi

# ---------------------------------------------------------------- 2. patching
step "automatic security updates (reboot 04:00 UTC when a kernel needs it)"
cat > /etc/apt/apt.conf.d/20auto-upgrades <<'EOF'
APT::Periodic::Update-Package-Lists "1";
APT::Periodic::Unattended-Upgrade "1";
APT::Periodic::AutocleanInterval "7";
EOF
cat > /etc/apt/apt.conf.d/52fly-unattended <<'EOF'
Unattended-Upgrade::Automatic-Reboot "true";
Unattended-Upgrade::Automatic-Reboot-Time "04:00";
Unattended-Upgrade::Remove-Unused-Dependencies "true";
EOF
systemctl enable --now unattended-upgrades.service
timedatectl set-timezone UTC
systemctl enable --now systemd-timesyncd.service      # the relay refuses the fly's requests if the clock is > 5 min off

# ---------------------------------------------------------------- 3. admin account
step "admin account: $ADMIN_USER"
id "$ADMIN_USER" >/dev/null 2>&1 || useradd -m -s /bin/bash "$ADMIN_USER"
passwd -l "$ADMIN_USER" >/dev/null                    # no password at all: SSH key only
home="$(getent passwd "$ADMIN_USER" | cut -d: -f6)"
install -d -m 0700 -o "$ADMIN_USER" -g "$ADMIN_USER" "$home/.ssh"
if [ ! -s "$home/.ssh/authorized_keys" ]; then
  if [ -s /root/.ssh/authorized_keys ]; then
    install -m 0600 -o "$ADMIN_USER" -g "$ADMIN_USER" /root/.ssh/authorized_keys "$home/.ssh/authorized_keys"
  else
    echo "STOP: neither $ADMIN_USER nor root has ~/.ssh/authorized_keys; add your key first" >&2; exit 1
  fi
fi
echo "$ADMIN_USER ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/90-fly-admin
chmod 0440 /etc/sudoers.d/90-fly-admin
visudo -cf /etc/sudoers.d/90-fly-admin >/dev/null
# NOT in the docker group: that group is root without sudo's log line; docker runs through sudo
gpasswd -d "$ADMIN_USER" docker >/dev/null 2>&1 || true

# ---------------------------------------------------------------- 4. SSH
step "SSH: $ADMIN_USER only, key only, modern crypto"
cat > /etc/ssh/sshd_config.d/10-fly-hardening.conf <<EOF
PermitRootLogin no
PasswordAuthentication no
KbdInteractiveAuthentication no
PubkeyAuthentication yes
AuthenticationMethods publickey
AllowUsers $ADMIN_USER
MaxAuthTries 3
MaxSessions 4
LoginGraceTime 20
ClientAliveInterval 300
ClientAliveCountMax 2
PermitEmptyPasswords no
PermitUserEnvironment no
X11Forwarding no
AllowAgentForwarding no
# local forwarding stays on: it is how you reach the console (ssh -L 8501:127.0.0.1:8501)
AllowTcpForwarding local
GatewayPorts no
PermitTunnel no
Compression no
TCPKeepAlive no
MaxStartups 10:30:60
LogLevel VERBOSE
KexAlgorithms sntrup761x25519-sha512@openssh.com,curve25519-sha256,curve25519-sha256@libssh.org
Ciphers chacha20-poly1305@openssh.com,aes256-gcm@openssh.com,aes128-gcm@openssh.com
MACs hmac-sha2-512-etm@openssh.com,hmac-sha2-256-etm@openssh.com
HostKeyAlgorithms ssh-ed25519,rsa-sha2-512,rsa-sha2-256
# client keys: ed25519, FIDO/hardware-backed keys, and P-256 (Secure Enclave keys from Secretive; the Ledger later)
PubkeyAcceptedAlgorithms sk-ssh-ed25519@openssh.com,ssh-ed25519,sk-ecdsa-sha2-nistp256@openssh.com,ecdsa-sha2-nistp256
EOF
# older lines in the main file must not win: Include is at the top, and first match wins in sshd
sed -i 's/^\(PermitRootLogin\|PasswordAuthentication\|KbdInteractiveAuthentication\|X11Forwarding\)/# fly: see sshd_config.d\/10-fly-hardening.conf\n#&/' /etc/ssh/sshd_config
install -d -m 0755 /run/sshd                            # 24.04 starts sshd on demand; the check needs this dir
sshd -t
systemctl reload ssh 2>/dev/null || systemctl restart ssh

step "fail2ban: SSH brute force"
cat > /etc/fail2ban/jail.d/fly-sshd.local <<'EOF'
[sshd]
enabled = true
backend = systemd
maxretry = 4
findtime = 10m
bantime = 1h
bantime.increment = true
EOF
systemctl enable --now fail2ban
systemctl restart fail2ban

# ---------------------------------------------------------------- 5. Tailscale + firewall
step "Tailscale"
systemctl enable --now tailscaled
if [ -n "$TS_AUTHKEY" ] && ! tailscale status >/dev/null 2>&1; then
  tailscale up --auth-key="$TS_AUTHKEY" --hostname=fly-vault --advertise-tags=tag:fly --ssh=false --accept-dns=false
fi
unset TS_AUTHKEY
TS_UP=0; tailscale status >/dev/null 2>&1 && TS_UP=1

step "firewall: SSH over Tailscale only (plus public SSH until CLOSE_PUBLIC_SSH=1); out: HTTPS, HTTP (apt), DNS, NTP, WireGuard"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default deny outgoing
ufw default deny routed
ufw allow in on tailscale0 to any port 22 proto tcp comment 'ssh over the tailnet'
if [ "$CLOSE_PUBLIC_SSH" = 1 ]; then
  [ "$TS_UP" = 1 ] || { echo "STOP: Tailscale is not up; closing public SSH now would lock you out" >&2; exit 1; }
  echo "public SSH closed: log in with ssh $ADMIN_USER@fly-vault (Tailscale)"
else
  ufw limit in 22/tcp comment 'ssh (rate limited) -- removed by CLOSE_PUBLIC_SSH=1'
fi
ufw allow out 443/tcp comment 'https: RPCs, relay, HF, Telegram, apt, Tailscale control'
ufw allow out 80/tcp comment 'http: apt mirrors'
ufw allow out 53 comment 'dns'
ufw allow out 123/udp comment 'ntp'
ufw allow out 41641/udp comment 'tailscale wireguard'
ufw allow out 3478/udp comment 'tailscale stun'
ufw logging low
ufw --force enable

# Docker writes its own iptables rules and bypasses ufw; the DOCKER-USER chain is where the host gets a say. Containers
# may only talk to each other, reach DNS/HTTPS out, and nothing on the internet may open a connection to them.
cat > /usr/local/sbin/fly-docker-firewall <<'EOF'
#!/bin/sh
set -e
iptables -N DOCKER-USER 2>/dev/null || true
iptables -F DOCKER-USER
iptables -A DOCKER-USER -m conntrack --ctstate ESTABLISHED,RELATED -j RETURN
iptables -A DOCKER-USER -i docker0 -o docker0 -j RETURN
iptables -A DOCKER-USER -i br-+ -o br-+ -j RETURN
for p in 443 80 53; do                  # 80: apt mirrors during image builds (apt checks its own signatures)
  iptables -A DOCKER-USER -i br-+ -p tcp --dport $p -j RETURN
  iptables -A DOCKER-USER -i docker0 -p tcp --dport $p -j RETURN
done
iptables -A DOCKER-USER -i br-+ -p udp --dport 53 -j RETURN
iptables -A DOCKER-USER -i docker0 -p udp --dport 53 -j RETURN
iptables -A DOCKER-USER -i br-+ -j DROP
iptables -A DOCKER-USER -i docker0 -j DROP
iptables -A DOCKER-USER -m conntrack --ctstate NEW -j DROP
EOF
chmod 0750 /usr/local/sbin/fly-docker-firewall
cat > /etc/systemd/system/fly-docker-firewall.service <<'EOF'
[Unit]
Description=fly-trader: restrict container traffic (DOCKER-USER chain)
After=docker.service
Requires=docker.service
PartOf=docker.service
[Service]
Type=oneshot
RemainAfterExit=yes
ExecStart=/usr/local/sbin/fly-docker-firewall
[Install]
WantedBy=docker.service
EOF

# ---------------------------------------------------------------- 6. kernel
step "kernel and network settings"
cat > /etc/sysctl.d/60-fly-hardening.conf <<'EOF'
net.ipv4.conf.all.accept_redirects = 0
net.ipv4.conf.default.accept_redirects = 0
net.ipv4.conf.all.secure_redirects = 0
net.ipv4.conf.default.secure_redirects = 0
net.ipv4.conf.all.send_redirects = 0
net.ipv4.conf.default.send_redirects = 0
net.ipv4.conf.all.accept_source_route = 0
net.ipv4.conf.default.accept_source_route = 0
net.ipv4.conf.all.rp_filter = 1
net.ipv4.conf.default.rp_filter = 1
net.ipv4.conf.all.log_martians = 1
net.ipv4.conf.default.log_martians = 1
net.ipv4.icmp_echo_ignore_broadcasts = 1
net.ipv4.icmp_ignore_bogus_error_responses = 1
net.ipv4.tcp_syncookies = 1
net.ipv6.conf.all.accept_redirects = 0
net.ipv6.conf.default.accept_redirects = 0
net.ipv6.conf.all.accept_source_route = 0
net.ipv6.conf.default.accept_source_route = 0
kernel.kptr_restrict = 2
kernel.dmesg_restrict = 1
kernel.unprivileged_bpf_disabled = 1
net.core.bpf_jit_harden = 2
kernel.yama.ptrace_scope = 2
kernel.randomize_va_space = 2
kernel.sysrq = 0
fs.protected_hardlinks = 1
fs.protected_symlinks = 1
fs.protected_fifos = 2
fs.protected_regular = 2
fs.suid_dumpable = 0
EOF
sysctl -q --system
# core dumps could write a key to disk
cat > /etc/security/limits.d/60-fly-nocore.conf <<'EOF'
* hard core 0
EOF
mkdir -p /etc/systemd/coredump.conf.d
printf '[Coredump]\nStorage=none\nProcessSizeMax=0\n' > /etc/systemd/coredump.conf.d/60-fly.conf
aa-status --enabled                                   # AppArmor must be on (Docker's default profile uses it)
# protocols, filesystems and devices this server never uses: not loadable at all
for m in dccp sctp rds tipc usb-storage firewire-core cramfs freevxfs hfs hfsplus jffs2 udf; do
  echo "install $m /bin/false"
done > /etc/modprobe.d/60-fly-blacklist.conf
# new files readable by their group at most, never by everyone
sed -i 's/^UMASK.*/UMASK\t\t027/' /etc/login.defs
printf 'Authorized access only. Activity on this system is logged.\n' > /etc/issue
cp /etc/issue /etc/issue.net
sed -i 's/^ENABLED=.*/ENABLED="true"/' /etc/default/sysstat 2>/dev/null || true
systemctl enable --now acct sysstat >/dev/null 2>&1 || true

# ---------------------------------------------------------------- 7. Docker
step "Docker: user namespaces, no new privileges, bounded logs"
install -d -m 0755 /etc/docker
[ -f /etc/docker/daemon.json ] || cat > /etc/docker/daemon.json <<'EOF'
{
  "userns-remap": "default",
  "no-new-privileges": true,
  "icc": false,
  "live-restore": true,
  "userland-proxy": false,
  "log-driver": "json-file",
  "log-opts": {"max-size": "50m", "max-file": "5"}
}
EOF
systemctl restart docker
systemctl daemon-reload
systemctl enable --now fly-docker-firewall.service
# with user namespaces, root inside a container is this unprivileged uid on the host
REMAP_UID="$(awk -F: '$1=="dockremap"{print $2; exit}' /etc/subuid)"

# ---------------------------------------------------------------- 8. layout, keys, updater
step "fly layout and wallets"
install -d -m 0750 /srv/fly
install -d -m 0755 /srv/fly/releases                  # public release files (from HF); the remapped container root reads them
H_BRAIN=$((REMAP_UID + BRAIN_UID)); H_SIGNER=$((REMAP_UID + SIGNER_UID)); H_FLY=$((REMAP_UID + FLY_GID))
install -d -m 0750 -o "$H_BRAIN" -g "$H_FLY" /srv/fly/data /srv/fly/logs
install -d -m 0700 -o "$H_SIGNER" -g "$H_SIGNER" /srv/fly/signer        # the signer's own ledger (policy state)
install -d -m 0750 -o "$H_SIGNER" -g "$H_FLY" /srv/fly/run              # its socket: the brain may connect, nothing else
install -d -m 0700 /srv/fly/secrets
install -d -m 0755 /etc/fly /usr/local/lib/fly
# release signers: valid for the release namespace only, so these keys sign nothing else this box would accept. A re-run
# without RECOVERY_PUBKEY keeps the recovery signer already installed (so closing SSH or updating the scripts never drops it).
if [ -z "$RECOVERY_PUBKEY" ] && [ -f /etc/fly/allowed_signers ]; then
  RECOVERY_PUBKEY="$(awk '$1=="bryce-recovery"{ $1=""; $2=""; sub(/^  */, ""); print; exit }' /etc/fly/allowed_signers)"
fi
{
  echo "bryce namespaces=\"fly-trader-release\" ${RELEASE_PUBKEY}"
  if [ -n "$RECOVERY_PUBKEY" ]; then echo "bryce-recovery namespaces=\"fly-trader-release\" ${RECOVERY_PUBKEY}"; fi
} > /etc/fly/allowed_signers
chmod 0644 /etc/fly/allowed_signers
[ -n "$RECOVERY_PUBKEY" ] || echo "WARNING: no RECOVERY_PUBKEY: losing the release key would mean re-bootstrapping the updater"
install -m 0755 "$HERE/fly" /usr/local/bin/fly     # sudo fly <command>: fly-trader inside the running container
# the database password: made here, root-only, handed to both database and brain by compose
if [ ! -s /srv/fly/db.env ]; then
  pw="$(openssl rand -hex 24)"
  ( umask 077; printf 'POSTGRES_PASSWORD=%s\nDATABASE_URL=postgresql://fly:%s@db:5432/fly_trader\n' "$pw" "$pw" > /srv/fly/db.env )
  unset pw
fi
chmod 0600 /srv/fly/db.env
# two hot keys, each an ed25519 seed made HERE and never copied off this box in plaintext (the signer writes
# age-encrypted backups for you): the trading key (bot_key: the trading wallet; refills itself from the treasury under
# L1) and the payout key (payout_key: holds nothing; pays claims from the treasury under L2). Only the signer container
# mounts them. Their public keys are printed below, once; later: sudo fly signer status.
make_key() {
  local name="$1"
  if [ ! -e "/srv/fly/secrets/$name" ]; then
    umask 077
    openssl genpkey -algorithm ed25519 -out "/srv/fly/secrets/$name.pem"
    python3 - "$name" <<'PY'
import subprocess, sys
name = sys.argv[1]
A = "123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz"
def b58(b):
    n = int.from_bytes(b, "big"); s = ""
    while n: n, r = divmod(n, 58); s = A[r] + s
    return "1" * (len(b) - len(b.lstrip(b"\0"))) + s
pem = f"/srv/fly/secrets/{name}.pem"
priv = subprocess.run(["openssl", "pkey", "-in", pem, "-outform", "DER"], check=True, capture_output=True).stdout[-32:]
pub = subprocess.run(["openssl", "pkey", "-in", pem, "-pubout", "-outform", "DER"], check=True, capture_output=True).stdout[-32:]
open(f"/srv/fly/secrets/{name}", "x").write(b58(priv) + "\n")
print(f"{name}: {b58(pub)}")
PY
    shred -u "/srv/fly/secrets/$name.pem"
  fi
  # readable only by the signer container's user (remapped), not by the brain container or any host account
  chown "$H_SIGNER:$H_SIGNER" "/srv/fly/secrets/$name"
  chmod 0400 "/srv/fly/secrets/$name"
}
make_key bot_key        # the trading hot key: holds at most the float; refills itself from the Squads treasury (L1)
make_key payout_key     # the payout key: holds nothing; pays claims from the treasury within its weekly cap (L2)
install -m 0755 "$HERE/fly_update.py" /usr/local/lib/fly/fly_update.py
install -m 0755 "$HERE/check-host.sh" /usr/local/lib/fly/check-host.sh
[ -f /etc/fly/update.json ] || install -m 0600 "$HERE/update.json.example" /etc/fly/update.json
if [ -n "$EXPECT_MANIFEST" ]; then        # pin the first release: the updater refuses anything else until one is installed
  python3 - "$EXPECT_MANIFEST" <<'PY'
import json, re, sys
pin = sys.argv[1].strip().lower()
if not re.fullmatch(r"[0-9a-f]{64}", pin):
    raise SystemExit("EXPECT_MANIFEST must be a sha256 hex digest")
p = "/etc/fly/update.json"; c = json.load(open(p)); c["expect_manifest_sha256"] = pin
open(p, "w").write(json.dumps(c, indent=1) + "\n")
PY
fi
[ -f /srv/fly/vault.env ] || install -m 0600 "$HERE/vault.env.example" /srv/fly/vault.env
[ -f /srv/fly/custody.env ] || install -m 0600 "$HERE/custody.env.example" /srv/fly/custody.env
[ -f /srv/fly/signer.env ] || install -m 0600 "$HERE/signer.env.example" /srv/fly/signer.env
install -m 0644 "$HERE/docker-compose.vault.yml" /srv/fly/docker-compose.yml
install -m 0644 "$HERE/fly-update.service" /etc/systemd/system/fly-update.service
install -m 0644 "$HERE/fly-update.timer" /etc/systemd/system/fly-update.timer
install -m 0755 "$HERE/fly_audit_watch.py" /usr/local/lib/fly/fly_audit_watch.py
install -m 0644 "$HERE/fly-audit-watch.service" /etc/systemd/system/fly-audit-watch.service
install -m 0644 "$HERE/fly-audit-watch.timer" /etc/systemd/system/fly-audit-watch.timer
install -m 0644 "$HERE/fly-aide.service" /etc/systemd/system/fly-aide.service
install -m 0644 "$HERE/fly-aide.timer" /etc/systemd/system/fly-aide.timer
systemctl daemon-reload
systemctl enable --now fly-audit-watch.timer    # a key read by anything but the signer goes to Telegram within a minute
systemctl enable --now fly-aide.timer           # the daily file-integrity check reports to Telegram (there is no mail here)

# ---------------------------------------------------------------- 9. audit and integrity
step "audit log: key reads and config changes"
cat > /etc/audit/rules.d/60-fly.rules <<'EOF'
-w /srv/fly/secrets -p rwa -k fly_keys
-w /etc/fly -p wa -k fly_conf
-w /srv/fly/vault.env -p rwa -k fly_conf
-w /srv/fly/db.env -p rwa -k fly_conf
-w /srv/fly/custody.env -p wa -k fly_conf
-w /srv/fly/signer.env -p rwa -k fly_conf
-w /srv/fly/signer -p wa -k fly_signer
-w /usr/local/lib/fly -p wa -k fly_updater
-w /etc/ssh/sshd_config -p wa -k sshd
-w /etc/ssh/sshd_config.d -p wa -k sshd
-w /etc/sudoers -p wa -k sudoers
-w /etc/sudoers.d -p wa -k sudoers
-w /etc/docker -p wa -k docker_conf
-w /etc/systemd/system -p wa -k systemd_units
-w /etc/passwd -p wa -k identity
-w /etc/shadow -p wa -k identity
-w /etc/group -p wa -k identity
EOF
augenrules --load >/dev/null
systemctl enable --now auditd

step "file integrity baseline (AIDE, checked daily; takes a few minutes)"
echo 'Checksums = sha512' > /etc/aide/aide.conf.d/10_fly_checksums
cat > /etc/aide/aide.conf.d/90_fly_ignore <<'EOF'
!/srv/fly/data
!/srv/fly/logs
!/srv/fly/releases
!/var/lib/docker
!/var/lib/containerd
EOF
[ -f /var/lib/aide/aide.db ] || { aideinit -y -f >/dev/null 2>&1 && cp /var/lib/aide/aide.db.new /var/lib/aide/aide.db; }

# ---------------------------------------------------------------- 10. logs
step "persistent journal"
mkdir -p /etc/systemd/journald.conf.d
printf '[Journal]\nStorage=persistent\nSystemMaxUse=2G\n' > /etc/systemd/journald.conf.d/60-fly.conf
systemctl restart systemd-journald

step "done"
if [ "$CLOSE_PUBLIC_SSH" = 1 ]; then
  next="Verify:   sudo /usr/local/lib/fly/check-host.sh   (every control should pass now)"
else
  next="From your Mac, BEFORE closing this session:   ssh $ADMIN_USER@fly-vault   (over Tailscale; root login is now off)
Then run this script again with CLOSE_PUBLIC_SSH=1 to remove public SSH."
fi
cat <<EOF
$next
Next (docs/vault/RUNBOOK.md, sections 4.3 and 4.4):
  1. the treasury on https://fly-trader.app/owner.html with your owner wallet (the two keys printed above)
  2. sudoedit /srv/fly/{custody,signer,vault}.env and /etc/fly/update.json (Telegram)
  3. sudo systemctl enable --now fly-update.timer   # pulls, verifies and starts the pinned first release
EOF
