#!/usr/bin/env bash
# Verify the vault server's hardening (bootstrap-host.sh). Run as root: sudo /usr/local/lib/fly/check-host.sh
# Prints PASS/FAIL per control and a Lynis hardening index; exits 1 if any control fails.
set -uo pipefail
fail=0
pass() { printf '  PASS  %s\n' "$1"; }
bad()  { printf '  FAIL  %s\n' "$1"; fail=1; }
check() { local what="$1"; shift; if "$@" >/dev/null 2>&1; then pass "$what"; else bad "$what"; fi; }
sshd_is() { sshd -T 2>/dev/null | grep -qix "$1 $2"; }
sysctl_is() { [ "$(sysctl -n "$1" 2>/dev/null)" = "$2" ]; }

echo "SSH"
check "root login off"                  sshd_is permitrootlogin no
check "passwords off"                   sshd_is passwordauthentication no
check "keyboard-interactive off"        sshd_is kbdinteractiveauthentication no
check "only the admin account"          sh -c "sshd -T | grep -qi '^allowusers '"
check "max 3 auth tries"                sshd_is maxauthtries 3
check "X11 / agent forwarding off"      sh -c "sshd -T | grep -qix 'x11forwarding no' && sshd -T | grep -qix 'allowagentforwarding no'"
check "only local TCP forwarding"       sshd_is allowtcpforwarding local
check "no weak ciphers"                 sh -c "! sshd -T | grep -i '^ciphers' | grep -Eqi 'cbc|3des|arcfour'"
check "fail2ban guards sshd"            sh -c "fail2ban-client status sshd"
check "client keys: ed25519/P-256/FIDO only" sh -c "sshd -T | grep -i '^pubkeyacceptedalgorithms' | grep -vqi rsa"
check "admin not in the docker group"   sh -c "! id -nG \"\$(sshd -T | awk 'tolower(\$1)==\"allowusers\"{print \$2}')\" | grep -qw docker"

echo "Tailscale"
check "tailscaled up"                   sh -c "tailscale status >/dev/null"
check "Tailnet Lock enabled"            sh -c "tailscale lock status 2>/dev/null | grep -qi 'lock is enabled'"
check "no public SSH (tailnet only)"    sh -c "! ufw status | grep -E '^22/tcp' | grep -v tailscale0 | grep -q ."

echo "Firewall"
check "ufw active"                      sh -c "ufw status | grep -q 'Status: active'"
check "incoming denied by default"      sh -c "ufw status verbose | grep -q 'deny (incoming)'"
check "outgoing denied by default"      sh -c "ufw status verbose | grep -q 'deny (outgoing)'"
check "only SSH listens publicly"       sh -c "! ss -Hltn | awk '{print \$4}' | grep -vE '^(127\\.|\\[::1\\]|\\[::ffff:127)' | grep -vE ':22\$' | grep -q ."
check "container traffic restricted"    sh -c "iptables -S DOCKER-USER | grep -q -- '-j DROP'"
check "console bound to localhost"      sh -c "! ss -Hltn | grep -E ':8501 ' | grep -vqE '127\\.0\\.0\\.1:8501'"

echo "Kernel"
for kv in net.ipv4.conf.all.accept_redirects=0 net.ipv4.conf.all.send_redirects=0 net.ipv4.conf.all.accept_source_route=0 \
          net.ipv4.conf.all.rp_filter=1 net.ipv4.tcp_syncookies=1 kernel.kptr_restrict=2 kernel.dmesg_restrict=1 \
          kernel.unprivileged_bpf_disabled=1 kernel.yama.ptrace_scope=2 kernel.randomize_va_space=2 fs.suid_dumpable=0 \
          fs.protected_symlinks=1 fs.protected_hardlinks=1; do
  check "${kv%%=*} = ${kv#*=}"          sysctl_is "${kv%%=*}" "${kv#*=}"
done
check "core dumps off"                  sh -c "grep -qs 'Storage=none' /etc/systemd/coredump.conf.d/60-fly.conf"
check "AppArmor enforcing"              aa-status --enabled

echo "Docker"
check "user namespace remapping"        sh -c "docker info 2>/dev/null | grep -qi 'userns'"
check "no-new-privileges by default"    sh -c "grep -q '\"no-new-privileges\": true' /etc/docker/daemon.json"
check "log size bounded"                sh -c "grep -q 'max-size' /etc/docker/daemon.json"
check "live restore"                    sh -c "docker info 2>/dev/null | grep -qi 'Live Restore Enabled: true'"

echo "Secrets"
for k in bot_key payout_key; do
  f=/srv/fly/secrets/$k
  check "$k exists, mode 400"           sh -c "[ -f $f ] && [ \"\$(stat -c %a $f)\" = 400 ]"
  check "$k not owned by a login user"  sh -c "[ \"\$(stat -c %U $f)\" != root ] || true; ! getent passwd \"\$(stat -c %U $f)\" | grep -qE ':/bin/(ba)?sh\$'"
done
check "secrets dir only root"           sh -c "[ \"\$(stat -c %a /srv/fly/secrets)\" = 700 ]"
check "the two wallets differ"          sh -c "! cmp -s /srv/fly/secrets/bot_key /srv/fly/secrets/payout_key"
check "release signer pinned"           sh -c "[ -s /etc/fly/allowed_signers ] && [ \"\$(stat -c %U /etc/fly/allowed_signers)\" = root ]"
check "updater owned by root"           sh -c "[ \"\$(stat -c %U /usr/local/lib/fly/fly_update.py)\" = root ] && ! [ -w /usr/local/lib/fly/fly_update.py -a \"\$(stat -c %a /usr/local/lib/fly/fly_update.py)\" != 755 ]"
for e in vault custody signer db; do
  check "$e.env mode 600"                sh -c "[ \"\$(stat -c %a /srv/fly/$e.env)\" = 600 ]"
done
check "no default database password"    sh -c "! grep -qE 'POSTGRES_PASSWORD(:|=) *fly\$' /srv/fly/docker-compose.yml /srv/fly/db.env"
check "release signers namespaced"      sh -c "! grep -v 'namespaces=\"fly-trader-release\"' /etc/fly/allowed_signers | grep -q ."
check "first release pinned (no TOFU)"  sh -c "grep -q expect_manifest_sha256 /etc/fly/update.json || [ -s /srv/fly/state.json ]"

echo "Key isolation"
check "keys mounted into the signer only" sh -c "! docker inspect fly-fly-1 --format '{{range .Mounts}}{{.Source}} {{end}}' | grep -q /srv/fly/secrets"
check "signer has the keys"             sh -c "docker inspect fly-fly-signer-1 --format '{{range .Mounts}}{{.Source}} {{end}}' | grep -q /srv/fly/secrets/bot_key"
check "signer filesystem read-only"     sh -c "docker inspect fly-fly-signer-1 --format '{{.HostConfig.ReadonlyRootfs}}' | grep -q true"
check "brain runs as uid 10001"         sh -c "docker inspect fly-fly-1 --format '{{.Config.User}}' | grep -q '^10001'"
check "database has no route out"       sh -c "docker network inspect fly_internal --format '{{.Internal}}' | grep -q true"

echo "Monitoring"
check "auditd running"                  systemctl is-active auditd
check "key reads audited"               sh -c "auditctl -l | grep -q fly_keys"
check "key-read watcher running"        systemctl is-active fly-audit-watch.timer
check "AIDE baseline present"           test -f /var/lib/aide/aide.db
check "journal persistent"              test -d /var/log/journal
check "automatic security updates"      sh -c "systemctl is-enabled unattended-upgrades && grep -q 'Automatic-Reboot \"true\"' /etc/apt/apt.conf.d/52fly-unattended"
check "clock synchronised"              sh -c "timedatectl show -p NTPSynchronized --value | grep -q yes"
check "no snapd"                        sh -c "! dpkg -l snapd 2>/dev/null | grep -q '^ii'"

echo "Lynis"
if command -v lynis >/dev/null; then
  lynis audit system --quick --no-colors >/tmp/lynis.out 2>&1
  printf '  hardening index: %s\n' "$(grep -o 'Hardening index : [0-9]*' /tmp/lynis.out | grep -o '[0-9]*$' || echo '?')"
  grep -E '^\s+\* ' /var/log/lynis-report.dat >/dev/null 2>&1 || true
fi

echo
[ "$fail" = 0 ] && echo "all controls pass" || echo "SOME CONTROLS FAILED"
exit $fail
