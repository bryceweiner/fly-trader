#!/usr/bin/env bash
# One-time setup of the vault fly's server (Ubuntu 24.04, e.g. Netcup RS 2000 G12), hardened along the lines of the
# CIS Ubuntu 24.04 benchmark for a single-purpose host that holds a hot wallet. Run as root:
#
#   ADMIN_USER=fly-admin bash bootstrap-host.sh "<your ssh-ed25519 RELEASE public key line>"
#
# ADMIN_USER (default fly-admin) becomes the only account that can log in over SSH (key only, sudo without a
# password; root login is switched off). If it has no ~/.ssh/authorized_keys yet, root's are copied to it first, and
# the script stops before touching SSH if neither exists. Keep this root session open until you have logged in as
# ADMIN_USER from a second terminal. deploy/check-host.sh verifies every setting afterwards.
#
# Idempotent: safe to run again (keys and edited config files are never overwritten).
set -euo pipefail
RELEASE_PUBKEY="${1:?usage: ADMIN_USER=fly-admin bootstrap-host.sh \"ssh-ed25519 AAAA... release-key\"}"
ADMIN_USER="${ADMIN_USER:-fly-admin}"
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
usermod -aG docker "$ADMIN_USER"

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

# ---------------------------------------------------------------- 5. firewall
step "firewall: SSH in; out only HTTPS, HTTP (apt), DNS, NTP"
ufw --force reset >/dev/null
ufw default deny incoming
ufw default deny outgoing
ufw default deny routed
ufw limit in 22/tcp comment 'ssh (rate limited)'
ufw allow out 443/tcp comment 'https: RPCs, relay, HF, Telegram, apt'
ufw allow out 80/tcp comment 'http: apt mirrors'
ufw allow out 53 comment 'dns'
ufw allow out 123/udp comment 'ntp'
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
for p in 443 53; do
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
install -d -m 0750 -o "$REMAP_UID" -g "$REMAP_UID" /srv/fly/data /srv/fly/logs
install -d -m 0700 /srv/fly/secrets
install -d -m 0755 /etc/fly /usr/local/lib/fly
echo "bryce ${RELEASE_PUBKEY}" > /etc/fly/allowed_signers
chmod 0644 /etc/fly/allowed_signers
# two wallets, each an ed25519 seed made HERE and never copied off this box in plaintext (the fly's backup job writes
# age-encrypted copies for you): the trading wallet, and the payout wallet claims are paid from, so holders never see
# the trading wallet. fly_trader.chain.keys accepts a base58 32-byte seed.
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
  # readable only by the container's root (the remapped uid), not by any other account on the host
  chown "$REMAP_UID:$REMAP_UID" "/srv/fly/secrets/$name"
  chmod 0400 "/srv/fly/secrets/$name"
}
make_key bot_key        # the trading wallet: fund it from a FUNDING_ADDRESSES address
make_key payout_key     # the payout wallet: never fund it yourself; the fly tops it up after each settlement
install -m 0755 "$HERE/fly_update.py" /usr/local/lib/fly/fly_update.py
install -m 0755 "$HERE/check-host.sh" /usr/local/lib/fly/check-host.sh
[ -f /etc/fly/update.json ] || install -m 0600 "$HERE/update.json.example" /etc/fly/update.json
[ -f /srv/fly/vault.env ] || install -m 0600 "$HERE/vault.env.example" /srv/fly/vault.env
install -m 0644 "$HERE/docker-compose.vault.yml" /srv/fly/docker-compose.yml
install -m 0644 "$HERE/fly-update.service" /etc/systemd/system/fly-update.service
install -m 0644 "$HERE/fly-update.timer" /etc/systemd/system/fly-update.timer
systemctl daemon-reload

# ---------------------------------------------------------------- 9. audit and integrity
step "audit log: key reads and config changes"
cat > /etc/audit/rules.d/60-fly.rules <<'EOF'
-w /srv/fly/secrets -p rwa -k fly_keys
-w /etc/fly -p wa -k fly_conf
-w /srv/fly/vault.env -p rwa -k fly_conf
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
cat <<EOF
Verify:   sudo /usr/local/lib/fly/check-host.sh
From your Mac, BEFORE closing this session:   ssh $ADMIN_USER@<server>   (root login is now off)
Next (docs/vault/RUNBOOK.md, section 3):
  1. edit /srv/fly/vault.env and /etc/fly/update.json (Telegram)
  2. sudo systemctl enable --now fly-update.timer   # pulls, verifies and starts the first signed release
EOF
