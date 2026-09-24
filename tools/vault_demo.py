"""A local dry run of the whole $FLY vault on this Mac: chain, contracts, relay, the fly's vault worker, and the site.

    .venv/bin/python tools/vault_demo.py up        # start everything (about a minute)
    .venv/bin/python tools/vault_demo.py status
    .venv/bin/python tools/vault_demo.py down

What runs (state and logs in .vault-demo/, gitignored):
  anvil        chain id 46630 on :8545 (stands in for Robinhood testnet), Multicall3 at its canonical address
  contracts    MockFLY + FlyVault + timelock (withdraw delay 5 min); two test holders lock at different times
  relay        web/wsgi.py on :8612 (claim domain localhost:5173, devnet)
  vault fly    fly_trader.vault.worker against the chain above and Solana DEVNET with throwaway keys; its own database
               fly_vault_demo; settlement every 5 minutes; no trading (the Mac's paper fly is untouched)
  site         Vite dev server on :5173 (testnet build pointed at anvil, relay proxied)

SOL (on a local solana-test-validator standing in for devnet): a throwaway "funder" gets an airdrop and deposits 1 SOL; the fly wallet also gets a direct airdrop, which
counts as profit for the lockers.
"""
from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import httpx

REPO = Path(__file__).resolve().parents[1]
D = REPO / ".vault-demo"
PY = str(REPO / ".venv" / "bin" / "python")
ANVIL = "http://127.0.0.1:8545"
RELAY_PORT = 8612
DEVNET = "http://127.0.0.1:8899"          # a local solana-test-validator: unlimited airdrops
DB = "fly_vault_demo"
SECRET = "demo-relay-secret-not-for-production"
# anvil's public test accounts (well known; never fund them on a real chain)
ACCTS = ["0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
         "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d",
         "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"]


def sh(*a, env=None, cwd=None, check=True) -> str:
    r = subprocess.run(list(a), cwd=cwd, env={**os.environ, **(env or {})}, capture_output=True, text=True)
    if check and r.returncode:
        raise SystemExit(f"{' '.join(a[:3])}… failed:\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
    return r.stdout.strip()


def spawn(name: str, args: list, env=None, cwd=None) -> int:
    log = open(D / f"{name}.log", "a")
    p = subprocess.Popen(args, cwd=cwd, env={**os.environ, **(env or {})}, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    (D / f"{name}.pid").write_text(str(p.pid))
    return p.pid


def wait_http(url: str, what: str, tries: int = 60, method: str = "GET", body=None) -> None:
    for _ in range(tries):
        try:
            r = httpx.request(method, url, json=body, timeout=3)
            if r.status_code < 500:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise SystemExit(f"{what} did not come up ({url})")


def sol_keys() -> dict:
    from solders.keypair import Keypair
    import base58
    f = D / "sol_keys.json"
    if f.exists():
        return json.loads(f.read_text())
    keys = {}
    for n in ("fly", "funder"):
        kp = Keypair()
        keys[n] = {"secret": base58.b58encode(bytes(kp)).decode(), "pubkey": str(kp.pubkey())}
    f.write_text(json.dumps(keys)); f.chmod(0o600)
    (D / "fly_key").write_text(keys["fly"]["secret"] + "\n"); (D / "fly_key").chmod(0o600)
    return keys


def airdrop(pubkey: str, sol: float) -> bool:
    for attempt in range(4):
        try:
            r = httpx.post(DEVNET, json={"jsonrpc": "2.0", "id": 1, "method": "requestAirdrop", "params": [pubkey, int(sol * 1e9)]}, timeout=20).json()
            if r.get("result"):
                return True
        except httpx.HTTPError:
            pass
        time.sleep(5 * (attempt + 1))
    return False


def balance(pubkey: str) -> int:
    r = httpx.post(DEVNET, json={"jsonrpc": "2.0", "id": 1, "method": "getBalance", "params": [pubkey]}, timeout=20).json()
    return int(r["result"]["value"])


def transfer(secret: str, to: str, lamports: int) -> str:
    import base58, base64
    from solders.hash import Hash
    from solders.keypair import Keypair
    from solders.message import MessageV0
    from solders.pubkey import Pubkey
    from solders.system_program import TransferParams, transfer as tx_transfer
    from solders.transaction import VersionedTransaction
    kp = Keypair.from_bytes(base58.b58decode(secret))
    bh = httpx.post(DEVNET, json={"jsonrpc": "2.0", "id": 1, "method": "getLatestBlockhash", "params": []}, timeout=20).json()["result"]["value"]["blockhash"]
    msg = MessageV0.try_compile(kp.pubkey(), [tx_transfer(TransferParams(from_pubkey=kp.pubkey(), to_pubkey=Pubkey.from_string(to), lamports=lamports))], [], Hash.from_string(bh))
    tx = VersionedTransaction(msg, [kp])
    r = httpx.post(DEVNET, json={"jsonrpc": "2.0", "id": 1, "method": "sendTransaction",
                                 "params": [base64.b64encode(bytes(tx)).decode(), {"encoding": "base64"}]}, timeout=20).json()
    return r.get("result") or str(r.get("error"))


def fly_env(dep: dict, keys: dict) -> dict:
    return {"DATABASE_URL": f"postgresql:///{DB}", "VAULT_ENABLED": "1", "VAULT_CLUSTER": "devnet", "VAULT_SOLANA_RPC_URL": DEVNET,
            "SOLANA_CLUSTER": "devnet", "LIVE_ENABLED": "0", "BOT_PRIVATE_KEY": "", "BOT_PRIVATE_KEY_FILE": str(D / "fly_key"),
            "FUNDING_ADDRESSES": keys["funder"]["pubkey"], "RH_CHAIN_ID": "46630", "RH_RPC_URL": ANVIL,
            "VAULT_ADDRESS": dep["vault"], "VAULT_TIMELOCK": dep["timelock"], "VAULT_START_BLOCK": "0", "VAULT_PERIOD_S": "300",
            "RELAY_URL": f"http://127.0.0.1:{RELAY_PORT}", "RELAY_KEY_ID": "k1", "RELAY_SECRET": SECRET,
            "VAULT_SITE_DOMAIN": "localhost:5173", "VAULT_SITE_URI": "http://localhost:5173/vault.html",
            "GAS_RESERVE_SOL": "0.01", "TELEGRAM_BOT_TOKEN": "", "LOG_DIR": str(D / "fly-logs")}


def up() -> None:
    D.mkdir(exist_ok=True)
    # 1. chains: a local Solana validator (plays devnet) and anvil (plays Robinhood testnet)
    spawn("solana", [str(Path.home() / ".local/share/solana/install/active_release/bin/solana-test-validator"),
                     "--reset", "--quiet", "--ledger", str(D / "ledger"), "--rpc-port", "8899"])
    wait_http(DEVNET, "solana validator", method="POST", body={"jsonrpc": "2.0", "id": 1, "method": "getHealth"}, tries=90)
    spawn("anvil", ["anvil", "--chain-id", "46630", "--port", "8545", "--block-time", "1", "--silent"])
    wait_http(ANVIL, "anvil", method="POST", body={"jsonrpc": "2.0", "id": 1, "method": "eth_chainId", "params": []})
    mc = "0xcA11bde05977b3631167028862bE2a173976CA11"
    code = sh("cast", "code", mc, "--rpc-url", "https://rpc.mainnet.chain.robinhood.com")
    sh("cast", "rpc", "anvil_setCode", mc, code, "--rpc-url", ANVIL)
    # 2. contracts (withdraw delay 5 min, timelock 10 min)
    sh("forge", "script", "script/DeployTestnet.s.sol", "--rpc-url", ANVIL, "--private-key", ACCTS[0], "--broadcast",
       env={"WITHDRAW_DELAY": "300", "TIMELOCK_DELAY": "600"}, cwd=REPO / "contracts")
    dep = json.loads((REPO / "contracts" / "deployments" / "46630.json").read_text())
    dep = {"vault": dep["vault"], "timelock": dep["timelock"], "fly": dep.get("token") or dep.get("fly")}
    (D / "deployment.json").write_text(json.dumps(dep, indent=1))
    holders = []
    for i, (pk, amt) in enumerate(zip(ACCTS[1:], ("300000", "100000"))):
        a = sh("cast", "wallet", "address", pk); holders.append(a)
        sh("cast", "send", dep["fly"], "mint(address,uint256)", a, "1000000000000000000000000", "--rpc-url", ANVIL, "--private-key", pk)
        sh("cast", "send", dep["fly"], "approve(address,uint256)", dep["vault"], "1000000000000000000000000", "--rpc-url", ANVIL, "--private-key", pk)
        sh("cast", "send", dep["vault"], "lock(uint256)", amt + "000000000000000000", "--rpc-url", ANVIL, "--private-key", pk)
        time.sleep(3)
    # 3. database for the demo fly
    subprocess.run(["createdb", DB], capture_output=True)
    sh(PY, "-c", "from fly_trader.db import schema; schema.apply_schema()", env={"DATABASE_URL": f"postgresql:///{DB}"}, cwd=REPO)
    # 4. relay
    rd = D / "relay"; rd.mkdir(exist_ok=True)
    (rd / "relay.json").write_text(json.dumps({"keys": {"k1": SECRET}, "db_path": str(rd / "relay.sqlite3"), "domain": "localhost:5173",
                                               "uri": "http://localhost:5173/vault.html", "chain_id": 46630, "sol_chain": "devnet",
                                               "claim_ttl_s": 900, "min_lamports": 2000000, "client_ip_header": None}))
    spawn("relay", [PY, "-c", "import sys; sys.path.insert(0, 'web'); from wsgiref.simple_server import make_server; import wsgi; "
                              f"make_server('127.0.0.1', {RELAY_PORT}, wsgi.application).serve_forever()"],
          env={"FLY_RELAY_CONFIG": str(rd / "relay.json")}, cwd=REPO)
    wait_http(f"http://127.0.0.1:{RELAY_PORT}/api/account?evm=0x" + "00" * 20, "relay")
    # 5. devnet SOL: 1 SOL deposit from the funder, 0.25 SOL straight to the fly (= profit for lockers)
    keys = sol_keys()
    got = airdrop(keys["funder"]["pubkey"], 1.5)
    time.sleep(15)
    if got and balance(keys["funder"]["pubkey"]) > 1_100_000_000:
        print("deposit tx:", transfer(keys["funder"]["secret"], keys["fly"]["pubkey"], 1_000_000_000)[:20], "…")
    else:
        print("WARNING: devnet airdrop to the funder failed (rate limit); fund", keys["funder"]["pubkey"], "at faucet.solana.com")
    if not airdrop(keys["fly"]["pubkey"], 0.25):
        print("WARNING: devnet airdrop to the fly failed; send devnet SOL to", keys["fly"]["pubkey"], "to create profit")
    # 6. the vault fly (vault worker only, its own database)
    (D / "fly-logs").mkdir(exist_ok=True)
    env = fly_env(dep, keys)
    sh(PY, "-c", "from fly_trader.vault import state; import time; state.put('vault_started_at', int(time.time()) - 600)", env=env, cwd=REPO)
    spawn("fly", [PY, "-c", "from fly_trader.vault import worker; worker.main()"], env=env, cwd=REPO)
    # 7. the site
    web_env = {"VITE_NETWORK": "testnet", "VITE_EVM_RPC": ANVIL, "VITE_VAULT_ADDRESS": dep["vault"], "VITE_TIMELOCK_ADDRESS": dep["timelock"],
               "VITE_FLY_ADDRESS": dep["fly"], "VITE_RELAY_PROXY": f"http://127.0.0.1:{RELAY_PORT}",
               "VITE_REOWN_PROJECT_ID": os.environ.get("VITE_REOWN_PROJECT_ID", "")}
    spawn("site", ["npx", "vite", "--port", "5173", "--strictPort", "--host", "localhost"], env=web_env, cwd=REPO / "web")
    wait_http("http://localhost:5173/vault.html", "site")
    print(json.dumps({"site": "http://localhost:5173/vault.html", "trading": "http://localhost:5173/trading.html",
                      "chain": ANVIL + " (id 46630)", **dep, "holders": holders, "fly_wallet": keys["fly"]["pubkey"],
                      "funder": keys["funder"]["pubkey"], "logs": str(D)}, indent=1))


def down() -> None:
    for pid_file in D.glob("*.pid"):
        try:
            os.killpg(int(pid_file.read_text()), signal.SIGTERM)
        except (ProcessLookupError, ValueError, PermissionError):
            pass
        pid_file.unlink()
    subprocess.run(["dropdb", "--if-exists", DB], capture_output=True)
    for f in ("relay/relay.sqlite3", "relay/relay.sqlite3-wal", "relay/relay.sqlite3-shm", "sol_keys.json", "fly_key", "vault.log"):
        (D / f).unlink(missing_ok=True)
    (REPO / "contracts" / "deployments" / "46630.json").unlink(missing_ok=True)
    print("demo stopped and wiped")


def status() -> None:
    for pid_file in sorted(D.glob("*.pid")):
        pid = int(pid_file.read_text())
        alive = subprocess.run(["kill", "-0", str(pid)], capture_output=True).returncode == 0
        print(f"{pid_file.stem:6} {pid} {'running' if alive else 'DEAD'}")
    try:
        s = httpx.get(f"http://127.0.0.1:{RELAY_PORT}/api/stats", timeout=3).json()
        print(json.dumps({k: s.get(k) for k in ("fly", "wallet", "ledger", "vault", "settlement")}, indent=1)[:2500])
    except Exception as e:
        print("relay:", type(e).__name__)


if __name__ == "__main__":
    {"up": up, "down": down, "status": status}.get(sys.argv[1] if len(sys.argv) > 1 else "", lambda: print(__doc__))()
