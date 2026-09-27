"""The ``vault`` worker: one thread inside the console (ops/supervisor.py) that runs the vault's periodic jobs.

Every loop (~10 s): pull and pay claims, push the public stats. Every minute: check the treasury's on-chain setup,
scan the trading wallet's and the treasury's flows, index Robinhood Chain, keep the trading float (top-up / sweep),
mark NAV when the live book is not doing it (paper warm-up), advance settlement. Hourly: close empty token accounts,
backups. Each job is isolated: one failing never stops the others, and nothing here can raise into trading.

No key is loaded here: every signature comes from the signer (fly_trader/signer; its own container on the server).
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
from . import alerts, backup, claims, custody, flows, nav, payout, publish, rh_index, settle, state, telegram_cmd, walletlock

log = logging.getLogger(__name__)
LOOP_S = 10.0
MINUTE_S = 60.0
HOUR_S = 3600.0


class Jobs:
    def __init__(self, signer=None):
        from ..chain.rpc import HttpSolanaRpc
        from ..signer import client as signer_client
        from .evm import EvmRpc
        from .relay_client import RelayClient
        self.signer = signer or signer_client.get()
        keys = self.signer.call("pubkeys")
        self.wallet, self.payout_key = keys["trading"], keys.get("payout")
        if not self.payout_key or self.payout_key == self.wallet:
            raise RuntimeError("the payout key must exist and differ from the trading key")
        self.our_keys = {self.wallet, self.payout_key}
        self.treasury = custody.treasury_address()
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
        claims.process(self.relay, self.rpc, self.signer, self.rh)

    def custody(self):
        if not settle.paper() and self.treasury:
            custody.check(self.rpc, self.wallet, self.payout_key)

    def scan(self):
        if not settle.paper():                                   # a paper book has no wallet to scan
            flows.scan_all(self.rpc, self.wallet, self.our_keys)

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
        if r["t"] and (datetime.now(timezone.utc) - r["t"]).total_seconds() < 50:
            return
        with walletlock.try_shared() as consistent:
            native = self.rpc.get_balance(self.wallet) + payout.cached_balance()     # trading wallet + treasury: one book
            with transaction() as conn:
                cost, _ = settle.open_positions(conn)
                m1 = datetime.fromtimestamp(int(time.time() // 60 * 60), timezone.utc)
                nav.mark(conn, m1, native, cost, 0, consistent=consistent)
        trading = self.rpc.get_balance(self.wallet)
        if trading < int(config.GAS_RESERVE_SOL * config.LAMPORTS_PER_SOL):
            alerts.send(f"trading wallet {trading / config.LAMPORTS_PER_SOL:.4f} SOL is below the gas reserve", key="low_gas", cooldown_s=6 * 3600)

    def settle(self):
        st = getattr(self, "index_state", None) or rh_index.index_state()
        settle.run_once(self.rpc, self.wallet, st, treasury=None if settle.paper() else self.treasury, our_keys=self.our_keys)

    def rebalance(self):
        """Keep the trading float near TREASURY_FLOAT_SOL and the treasury able to pay everything owed."""
        if settle.paper() or not self.treasury:
            return
        payout.rebalance(self.rpc, self.signer, self.wallet, self.treasury)

    def publish(self):
        publish.push(self.relay, self.wallet)

    def telegram(self):
        telegram_cmd.poll(self.signer)

    def heartbeat(self):
        telegram_cmd.heartbeat(self.signer)

    def deadman(self):
        telegram_cmd.deadman_ping()

    def backup(self):
        if settle.paper():
            return
        if not backup.configured():
            alerts.send("encrypted backups are not configured (VAULT_BACKUP_*): losing the server would lose the ledger and the float's keys",
                        key="backup_missing", cooldown_s=24 * 3600)
            return
        backup.key_once(self.signer)
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
            res = close_empty_atas(rpc=self.rpc, signer=self.signer, skip_mints=mints | {config.WSOL_MINT})
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
        jobs.run("publish", jobs.publish)                        # every loop (10 s): the site is live
        jobs.run("telegram", jobs.telegram)                      # /panic answers within a loop
        if jobs.due("deadman", 300.0):
            jobs.run("deadman", jobs.deadman)
        if jobs.due("minute", MINUTE_S):
            for name in ("custody", "scan", "index", "rebalance", "mark", "settle", "rebalance", "heartbeat"):
                if stop_event.is_set():
                    break
                jobs.run(name, getattr(jobs, name))
        if jobs.due("hour", HOUR_S):
            jobs.run("close_atas", jobs.close_atas)
            jobs.run("backup", jobs.backup)
        stop_event.wait(LOOP_S)
    log.info("vault worker stopped")
