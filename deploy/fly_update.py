#!/usr/bin/env python3
"""The vault fly's updater: pull signed releases from Hugging Face and apply them (plan phase 7).

Runs on the HOST (root, systemd timer, standard library only), never inside the container, because it rebuilds and
restarts the container. Everything it trusts comes from files on this host that no release can change:
``/etc/fly/allowed_signers`` (Bryce's release keys, restricted to the release namespace) and ``/etc/fly/update.json``
(which also pins the very first release by its manifest hash: no trust on first use).

One round:
 1. poll the HF model repo's commit sha; nothing to do if it is the installed one
 2. fetch ``release.json`` + ``release.json.sig`` from that immutable commit; verify the signature with
    ``ssh-keygen -Y verify``; require ``seq`` to grow and ``prev_sha256`` to be the installed manifest (no rollback,
    no forks)
 3. download exactly the listed paths (strict path check), verify size + sha256 of each, into a fresh directory
 4. compare file classes with the installed release:
      only model/seed  -> ``docker compose exec fly fly-trader apply-release /releases/<seq>`` (hot swap; models
                          cannot run code, fly_trader/train/model_io.py); unhealthy after it -> the previous release's
                          models are re-applied
      code             -> held for ``code_delay_s`` (24 h) with a Telegram notice, so a stolen signing key cannot ship
                          code faster than Bryce can ``veto`` it; ``approve`` applies it at once. Then: wait for
                          ``ready-for-restart``, build ``fly-trader:<seq>`` from that directory, recreate, wait for
                          healthy, else roll back to the previous tag and compose file
      infra            -> held until ``approve`` (compose/Dockerfile/entrypoint can grant host access, so a human
                          looks first)

    fly_update.py run | approve <seq> | veto <seq> | status | verify <dir> <allowed_signers> [principal ...]
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path

CONF = Path(os.environ.get("FLY_UPDATE_CONF", "/etc/fly/update.json"))
PATH_RE = re.compile(r"^(?!.*\.\.)(?!/)[A-Za-z0-9_.\-/]+$")
INFRA = ("Dockerfile", "docker-compose.yml", "docker-compose.cuda.yml", ".dockerignore", "docker/", "deploy/")
# what models/ and seed/ may hold: nothing a loader could execute (fly_trader/train/model_io.SAFE_SUFFIXES)
MODEL_SUFFIXES = {".pt", ".skops", ".npz", ".json", ".bin", ".parquet", ".txt", ""}
SEED_SUFFIXES = {".json", ".sql"}


def conf() -> dict:
    c = json.loads(CONF.read_text())
    c.setdefault("repo", "bryceweiner/fly-trader")
    c.setdefault("allowed_signers", "/etc/fly/allowed_signers")
    c.setdefault("principals", [c.get("principal") or "bryce", "bryce-recovery"])
    c.setdefault("code_delay_s", 86400)
    c.setdefault("namespace", "fly-trader-release")
    c.setdefault("root", "/srv/fly")
    c.setdefault("project", "fly")
    return c


def log(msg: str) -> None:
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)


def alert(c: dict, text: str) -> None:
    log("ALERT " + text)
    tok, chat = c.get("telegram_bot_token"), c.get("telegram_chat_id")
    if not tok or not chat:
        return
    try:
        body = json.dumps({"chat_id": chat, "text": "[fly updater] " + text}).encode()
        urllib.request.urlopen(urllib.request.Request(f"https://api.telegram.org/bot{tok}/sendMessage", data=body,
                                                      headers={"Content-Type": "application/json"}), timeout=10)
    except Exception as e:
        log(f"telegram failed: {type(e).__name__}")


def http(url: str, dest: Path | None = None, timeout: int = 120) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "fly-updater/1"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if dest is None:
            return r.read()
        with open(dest, "wb") as f:
            shutil.copyfileobj(r, f, 1 << 20)
    return b""


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def classify(path: str) -> str:
    if path.startswith("models/"):
        return "model"
    if path.startswith("seed/"):
        return "seed"
    if path in INFRA or any(path.startswith(p) for p in INFRA if p.endswith("/")):
        return "infra"
    return "code"


def verify_signature(c: dict, manifest: bytes, sig: bytes) -> str:
    """The principal whose key signed ``manifest`` (any of ``principals``), else SystemExit."""
    principals = c.get("principals") or [c["principal"]]
    err = ""
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "release.json.sig"; sp.write_bytes(sig)
        for who in principals:
            r = subprocess.run(["ssh-keygen", "-Y", "verify", "-f", c["allowed_signers"], "-I", who, "-n", c["namespace"], "-s", str(sp)],
                               input=manifest, capture_output=True)
            if r.returncode == 0:
                return who
            err = r.stderr.decode(errors="replace").strip()
    raise SystemExit(f"release signature does NOT verify: {err[:300]}")


def check_manifest(m: dict) -> None:
    if m.get("schema") != 1 or not isinstance(m.get("seq"), int) or not isinstance(m.get("files"), list):
        raise SystemExit("malformed release manifest")
    for f in m["files"]:
        p = f.get("path", "")
        if not PATH_RE.match(p) or p.startswith(".git") or "//" in p:
            raise SystemExit(f"refusing path {p!r}")
        if f.get("class") != classify(p):
            raise SystemExit(f"{p}: class {f.get('class')} does not match {classify(p)}")
        if f["class"] == "model" and Path(p).suffix not in MODEL_SUFFIXES or f["class"] == "seed" and Path(p).suffix not in SEED_SUFFIXES:
            raise SystemExit(f"{p}: file type not allowed in {f['class']}/ (a model or seed must not be able to run code)")
        if not re.fullmatch(r"[0-9a-f]{64}", f.get("sha256", "")) or not isinstance(f.get("size"), int):
            raise SystemExit(f"{p}: bad hash or size")


def verify_dir(d: Path, m: dict) -> None:
    """Exactly the listed files, each with its size and hash (no extra files may ride along into a build)."""
    listed = {f["path"] for f in m["files"]}
    for p in d.rglob("*"):
        if p.is_symlink():
            raise SystemExit(f"symlink in release: {p}")
        rel = p.relative_to(d).as_posix()
        if p.is_file() and rel not in listed and rel not in ("release.json", "release.json.sig"):
            raise SystemExit(f"unlisted file in release: {rel}")
    for f in m["files"]:
        p = d / f["path"]
        if not p.is_file() or p.stat().st_size != f["size"] or sha256_file(p) != f["sha256"]:
            raise SystemExit(f"{f['path']}: missing or hash/size mismatch")


def state_path(c: dict) -> Path:
    return Path(c["root"]) / "state.json"


def load_state(c: dict) -> dict:
    p = state_path(c)
    return json.loads(p.read_text()) if p.exists() else {"seq": 0, "manifest_sha256": None, "hf_sha": None, "image": None}


def save_state(c: dict, st: dict) -> None:
    p = state_path(c); tmp = p.with_suffix(".tmp"); tmp.write_text(json.dumps(st, indent=1)); tmp.replace(p)


def compose(c: dict, *args: str, check: bool = True, env: dict | None = None) -> subprocess.CompletedProcess:
    e = {**os.environ, **(env or {})}
    return subprocess.run(["docker", "compose", "-p", c["project"], "-f", str(Path(c["root"]) / "docker-compose.yml"), *args],
                          cwd=c["root"], check=check, capture_output=True, text=True, env=e)


def fetch(c: dict, sha: str) -> tuple[dict, Path, bytes, str]:
    base = f"https://huggingface.co/{c['repo']}/resolve/{sha}/"
    raw = http(base + "release.json"); sig = http(base + "release.json.sig")
    who = verify_signature(c, raw, sig)
    m = json.loads(raw)
    check_manifest(m)
    dest = Path(c["root"]) / "releases" / str(m["seq"])
    if dest.exists():
        try:                                     # already downloaded (a held or retried release): re-verify, reuse
            if (dest / "release.json").read_bytes() == raw:
                verify_dir(dest, m)
                return m, dest, raw, who
        except (SystemExit, OSError):
            pass
        shutil.rmtree(dest)
    part = dest.with_name(dest.name + ".part")
    if part.exists():
        shutil.rmtree(part)
    part.mkdir(parents=True)
    for f in m["files"]:
        out = part / f["path"]; out.parent.mkdir(parents=True, exist_ok=True)
        http(base + f["path"], out, timeout=1800)
    (part / "release.json").write_bytes(raw); (part / "release.json.sig").write_bytes(sig)
    verify_dir(part, m)
    part.rename(dest)
    return m, dest, raw, who


def installed_manifest(c: dict, st: dict) -> dict | None:
    if not st.get("seq"):
        return None
    p = Path(c["root"]) / "releases" / str(st["seq"]) / "release.json"
    return json.loads(p.read_text()) if p.exists() else None


def changed_classes(old: dict | None, new: dict) -> set[str]:
    if old is None:
        return {"code", "infra", "model", "seed"}
    o = {f["path"]: f["sha256"] for f in old["files"]}
    n = {f["path"]: f["sha256"] for f in new["files"]}
    paths = {p for p in set(o) | set(n) if o.get(p) != n.get(p)}
    return {classify(p) for p in paths}


def wait_ready(c: dict, limit_s: int = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        if compose(c, "exec", "-T", "fly", "fly-trader", "ready-for-restart", check=False).returncode == 0:
            return True
        time.sleep(15)
    return False


def healthy(c: dict, limit_s: int = 900) -> bool:
    t0 = time.time()
    while time.time() - t0 < limit_s:
        cid = compose(c, "ps", "-q", "fly", check=False).stdout.strip()
        if cid:
            st = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}", cid], capture_output=True, text=True).stdout.strip()
            if st == "healthy":
                return True
            if st == "unhealthy":
                return False
        time.sleep(15)
    return False


def deploy_code(c: dict, st: dict, m: dict, d: Path, infra: bool) -> str:
    """'ok', 'busy' (not ready for a restart: retried next round) or 'failed' (rolled back)."""
    tag = f"fly-trader:{m['seq']}"
    root = Path(c["root"]); live, backup = root / "docker-compose.yml", root / "docker-compose.yml.prev"
    try:                                        # the brain (default stage) and the key-holding signer, same release
        subprocess.run(["docker", "build", "-t", tag, str(d)], check=True, capture_output=True)
        subprocess.run(["docker", "build", "--target", "signer", "-t", f"fly-signer:{m['seq']}", str(d)], check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        alert(c, f"release {m['seq']}: image build failed: {(e.stderr or b'').decode(errors='replace')[-300:]}")
        return "failed"
    if not wait_ready(c):
        alert(c, f"release {m['seq']}: the fly never became ready for a restart (orders/claims/settlement in flight); will retry")
        return "busy"
    if infra:                                   # approved: the release's compose file becomes the running one
        shutil.copy2(live, backup)
        shutil.copy2(d / "deploy" / "docker-compose.vault.yml", live)
    prev = st.get("image")
    env = {"FLY_RELEASES": str(root / "releases")}
    compose(c, "up", "-d", check=False, env={**env, "FLY_RELEASE": str(m["seq"])})
    if healthy(c):
        return "ok"
    alert(c, f"release {m['seq']} did not become healthy; rolling back to {prev}")
    if infra and backup.exists():
        shutil.copy2(backup, live)
    if prev:
        compose(c, "up", "-d", check=False, env={**env, "FLY_RELEASE": str(prev)})
    return "failed"


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(ts))


def apply(c: dict, st: dict, m: dict, d: Path, raw: bytes, hf_sha: str, approved: bool = False, who: str = "?",
          now: float | None = None) -> None:
    now = time.time() if now is None else now
    msha = hashlib.sha256(raw).hexdigest()
    old = installed_manifest(c, st)
    if old is None:                              # the first release is pinned by hand at bootstrap (no TOFU)
        pin = c.get("expect_manifest_sha256")
        if not pin:
            raise SystemExit("no release installed and update.json pins none (expect_manifest_sha256): refusing to trust the first one seen")
        if msha != pin:
            raise SystemExit(f"first release's manifest {msha[:12]} is not the pinned {str(pin)[:12]}")
    else:
        if m["seq"] <= old["seq"]:
            raise SystemExit(f"release seq {m['seq']} is not newer than installed {old['seq']} (rollback refused)")
        pend = st.get("pending") or {}
        chain = {st.get("manifest_sha256"), pend.get("manifest_sha256"), *(st.get("vetoed") or [])} - {None}
        if m.get("prev_sha256") not in chain:
            raise SystemExit("release does not chain from the installed one (prev_sha256 mismatch)")
    classes = changed_classes(old, m)
    if old is not None and classes & {"code", "infra"}:
        pend = st.get("pending") or {}
        if pend.get("manifest_sha256") != msha:  # newly seen: the delay starts now (a replaced pending one restarts it)
            pend = {"seq": m["seq"], "hf_sha": hf_sha, "manifest_sha256": msha, "classes": sorted(classes), "signer": who,
                    "first_seen": now, "due_at": now + float(c.get("code_delay_s", 86400)), "approved": False}
            st["pending"] = pend
            what = ("changes Docker/compose/entrypoint files: review it, then `sudo python3 /usr/local/lib/fly/fly_update.py approve "
                    f"{m['seq']}`" if "infra" in classes else
                    f"changes code: it applies at {_when(pend['due_at'])} unless you run `sudo python3 /usr/local/lib/fly/fly_update.py veto "
                    f"{m['seq']}` (`approve {m['seq']}` applies it now)")
            alert(c, f"release {m['seq']} signed by {who} {what}")
        if approved:
            pend["approved"] = True
        save_state(c, st)
        if "infra" in classes and not pend["approved"]:
            return
        if not (pend["approved"] or now >= pend["due_at"]):
            return
    if classes & {"code", "infra"} or st.get("image") is None:
        res = deploy_code(c, st, m, d, "infra" in classes)
        if res == "busy":
            return
        if res == "failed":
            raise SystemExit("the new image did not come up healthy (rolled back)")
        st["image"] = m["seq"]
    elif classes & {"model", "seed"}:
        r = compose(c, "exec", "-T", "fly", "fly-trader", "apply-release", f"/releases/{m['seq']}", check=False)
        if r.returncode != 0:
            raise SystemExit(f"apply-release failed: {(r.stderr or r.stdout)[-300:]}")
        if not healthy(c, 300):
            if old is not None:
                compose(c, "exec", "-T", "fly", "fly-trader", "apply-release", f"/releases/{old['seq']}", check=False)
            raise SystemExit("unhealthy after the model release; the previous release's models were re-applied")
    st.update({"seq": m["seq"], "manifest_sha256": msha, "hf_sha": hf_sha, "pending": None})
    save_state(c, st)
    alert(c, f"release {m['seq']} applied ({', '.join(sorted(classes))}), signed by {who}")


def run(approve_seq: int | None = None, now: float | None = None) -> None:
    c = conf()
    now = time.time() if now is None else now
    Path(c["root"], "releases").mkdir(parents=True, exist_ok=True)
    st = load_state(c)
    pend = st.get("pending") or {}
    if approve_seq is not None:
        if pend.get("seq") != approve_seq:
            raise SystemExit(f"no pending release {approve_seq}")
        sha = pend["hf_sha"]
    else:
        head = json.loads(http(f"https://huggingface.co/api/models/{c['repo']}"))["sha"]
        if head not in (st.get("hf_sha"), st.get("hf_sha_rejected"), pend.get("hf_sha")):
            sha = head                           # a new release
        elif pend and (pend.get("approved") or now >= pend.get("due_at", float("inf"))):
            sha = pend["hf_sha"]                 # a held release whose time came (or a retry after 'busy')
        else:
            return                               # installed, rejected (alerted once), or still held
    try:
        m, d, raw, who = fetch(c, sha)
        apply(c, st, m, d, raw, sha, approved=approve_seq is not None, who=who, now=now)
    except SystemExit as e:
        st = load_state(c)
        if (st.get("pending") or {}).get("hf_sha") == sha:
            st["pending"] = None
        st["hf_sha_rejected"] = sha; save_state(c, st)
        alert(c, f"release at {sha[:10]} rejected: {e}")
        raise


def veto(seq: int) -> None:
    c = conf(); st = load_state(c); pend = st.get("pending") or {}
    if pend.get("seq") != seq:
        raise SystemExit(f"no pending release {seq}")
    st["vetoed"] = [*(st.get("vetoed") or []), pend["manifest_sha256"]][-20:]
    st["hf_sha_rejected"] = pend["hf_sha"]; st["pending"] = None
    save_state(c, st)
    alert(c, f"release {seq} VETOED: it will never apply. If you did not publish it, your release key is compromised: "
             "remove it from /etc/fly/allowed_signers and rotate.")


def main(argv: list[str]) -> None:
    cmd = argv[1] if len(argv) > 1 else "run"
    if cmd == "run":
        run()
    elif cmd == "approve":
        run(int(argv[2]))
    elif cmd == "veto":
        veto(int(argv[2]))
    elif cmd == "status":
        print(json.dumps(load_state(conf()), indent=1))
    elif cmd == "verify":                       # used by tools/publish_release.py on the Mac before uploading
        d = Path(argv[2]); c = {"allowed_signers": argv[3], "principals": argv[4:] or ["bryce"], "namespace": "fly-trader-release"}
        raw = (d / "release.json").read_bytes()
        verify_signature(c, raw, (d / "release.json.sig").read_bytes())
        m = json.loads(raw); check_manifest(m); verify_dir(d, m)
        print(f"release {m['seq']}: signature and {len(m['files'])} files verify")
    else:
        raise SystemExit(__doc__)


if __name__ == "__main__":
    main(sys.argv)
