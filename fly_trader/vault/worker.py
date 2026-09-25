"""The ``vault`` worker: one thread inside the console (ops/supervisor.py) that runs the vault's periodic jobs.

Every loop (~10 s): pull and pay claims. Every minute: scan the wallet's flows, index Robinhood Chain, mark NAV when
the live book is not doing it (paper warm-up), advance settlement, push the public stats. Hourly: close empty token
accounts. Each job is isolated: one failing never stops the others, and nothing here can raise into trading.
"""
from __future__ import annotations

import logging
import threading
import time
from datetime import datetime, timezone

from .. import config
from ..db.apilog import record_event
from ..db.connection import transaction
from ..logging_setup import setup
from . import alerts, backup, claims, flows, nav, publish, rh_index, settle, state, walletlock

log = logging.getLogger(__name__)
LOOP_S = 10.0
MINUTE_S = 60.0
HOUR_S = 3600.0


class Jobs:
    def __init__(self):
        from ..chain.keys import load_keypair
        from ..chain.rpc import HttpSolanaRpc
        from .evm import EvmRpc
        from .relay_client import RelayClient
        self.keypair = load_keypair()
        self.wallet = str(self.keypair.pubkey())
        self.rpc = HttpSolanaRpc(config.vault_solana_rpc_url())
        self.rh = EvmRpc(config.RH_RPC_URL, config.RH_CHAIN_ID)
        self.relay = RelayClient()
        self.last: dict[str, float] = {}
        self.fail: dict[str, int] = {}
        if not state.get("vault_started_at"):
            state.put("vault_started_at", int(time.time()))
            record_event("info", "vault", "vault started", {"wallet": self.wallet})

    def due(self, name: str, every: float) -> bool:
        now = time.monotonic()
        if now - self.last.get(name, -1e18) >= every:
            self.last[name] = now
            return True
        return False

    def run(self, name: str, fn) -> None:
        try:
            fn()
            if self.fail.pop(name, 0) >= 3:
                alerts.send(f"{name} recovered", key=f"{name}_ok", cooldown_s=60)
        except Exception as e:
            n = self.fail[name] = self.fail.get(name, 0) + 1
            log.exception("vault job %s failed (%d in a row)", name, n)
            if n in (3, 30, 300):
                alerts.send(f"{name} failing {n}x in a row: {type(e).__name__}", key=f"{name}_fail", cooldown_s=900)

    # ---- jobs ----
    def claims(self):
        claims.process(self.relay, self.rpc, self.keypair, self.rh)

    def scan(self):
        if not settle.paper():                                   # a paper book has no wallet to scan
            flows.scan(self.rpc, self.wallet)

    def index(self):
        self.index_state = rh_index.run_once(self.rh)

    def mark(self):
        """The live book marks NAV every trade minute; before the handover (or while the feed is stale) this does.
        A paper vault copies its book's own wealth marks: the running fly's real NAV, just not real SOL."""
        if settle.paper():
            with transaction() as conn:
                for r in conn.execute("SELECT ts, sol_free, positions_value, exit_cost FROM wealth_marks WHERE book = %s AND ts > "
                                      "COALESCE((SELECT max(ts) FROM vault_nav), to_timestamp(0)) ORDER BY ts LIMIT 5000", (config.VAULT_BOOK,)).fetchall():
                    L = config.LAMPORTS_PER_SOL
                    nav.mark(conn, r["ts"], int(round(float(r["sol_free"]) * L)), int(round((float(r["positions_value"]) + float(r["exit_cost"])) * L)),
                             int(round(float(r["exit_cost"]) * L)))
            return
        with transaction() as conn:
            r = conn.execute("SELECT max(ts) AS t FROM vault_nav").fetchone()
        if r["t"] and (datetime.now(timezone.utc) - r["t"]).total_seconds() < 150:
            return
        with walletlock.try_shared() as consistent:
            native = self.rpc.get_balance(self.wallet)
            with transaction() as conn:
                cost, _ = settle.open_positions(conn)
                m1 = datetime.fromtimestamp(int(time.time() // 60 * 60), timezone.utc)
                nav.mark(conn, m1, native, cost, 0, consistent=consistent)
        if native < int(config.GAS_RESERVE_SOL * config.LAMPORTS_PER_SOL):
            alerts.send(f"wallet {native / config.LAMPORTS_PER_SOL:.4f} SOL is below the gas reserve", key="low_gas", cooldown_s=6 * 3600)

    def settle(self):
        st = getattr(self, "index_state", None) or rh_index.index_state()
        settle.run_once(self.rpc, self.wallet, st)

    def publish(self):
        publish.push(self.relay, self.wallet)

    def backup(self):
        if settle.paper():
            return
        if not backup.configured():
            alerts.send("encrypted backups are not configured (VAULT_BACKUP_*): losing the server would lose the wallet",
                        key="backup_missing", cooldown_s=24 * 3600)
            return
        backup.key_once()
        backup.ledger_nightly()

    def close_atas(self):
        from ..chain.cluster_guard import assert_vault_signing_allowed
        from ..execution.broker_live import close_empty_atas
        if config.VAULT_CLUSTER != "mainnet-beta" or settle.paper():
            return                                               # the devnet rehearsal / a paper vault holds no token accounts worth closing
        assert_vault_signing_allowed()
        with walletlock.exclusive(timeout_s=120):
            with transaction() as conn:
                _, mints = settle.open_positions(conn)
            res = close_empty_atas(rpc=self.rpc, keypair=self.keypair, skip_mints=mints | {config.WSOL_MINT})
        if res.get("closed"):
            log.info("closed %d empty token accounts", res["closed"])


def main(stop_event: threading.Event | None = None) -> None:
    setup("vault")
    if not config.VAULT_ENABLED:
        log.error("vault worker: VAULT_ENABLED is not set; nothing to do")
        return
    stop_event = stop_event or threading.Event()
    jobs = Jobs()
    log.info("vault worker: wallet %s, relay %s, vault %s on chain %s", jobs.wallet, config.RELAY_URL, config.VAULT_ADDRESS, config.RH_CHAIN_ID)
    while not stop_event.is_set():
        jobs.run("claims", jobs.claims)
        if jobs.due("minute", MINUTE_S):
            for name in ("scan", "index", "mark", "settle", "publish"):
                if stop_event.is_set():
                    break
                jobs.run(name, getattr(jobs, name))
        if jobs.due("hour", HOUR_S):
            jobs.run("close_atas", jobs.close_atas)
            jobs.run("backup", jobs.backup)
        stop_event.wait(LOOP_S)
    log.info("vault worker stopped")
