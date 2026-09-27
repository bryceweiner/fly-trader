"""``fly-trader vault …``: the operator's commands for the hosted vault fly."""
from __future__ import annotations

import json


def add(sub) -> None:
    v = sub.add_parser("vault", help="the $FLY vault: status, scan, settle, reclassify, withdraw, claims, resume, custody")
    vs = v.add_subparsers(dest="vault_cmd", required=True)
    vs.add_parser("status", help="ledger, index, halt state, last settlement")
    vs.add_parser("scan", help="classify new wallet transactions and index Robinhood Chain once")
    st = vs.add_parser("settle", help="advance settlement one step (snapshot / allocate)")
    st.add_argument("--dry-run", action="store_true", help="print what the current period would allocate, write nothing")
    rc = vs.add_parser("reclassify", help="relabel one transaction's flow")
    rc.add_argument("signature"); rc.add_argument("kind", choices=["deposit", "profit", "internal"]); rc.add_argument("--note")
    vs.add_parser("withdraw", help="how much principal you may take out of the treasury, and how (Squads app)")
    vs.add_parser("custody", help="read and check the treasury multisig and its spending limits L1/L2")
    vs.add_parser("squads-setup", help="the Squads app steps that create the treasury's spending limits for this server's keys")
    vs.add_parser("claims", help="recent claims")
    vs.add_parser("resume", help="clear a vault halt (after you checked why it halted)")
    v.set_defaults(fn=run)


def _rpc():
    from .. import config
    from ..chain.rpc import HttpSolanaRpc
    return HttpSolanaRpc(config.vault_solana_rpc_url())


def _keys() -> dict:
    """The server's public keys, from the signer (the private keys never enter this process)."""
    from ..signer import client
    return client.get().call("pubkeys")


def run(args) -> None:
    from .. import config
    from ..db.connection import transaction
    from . import custody, flows, rh_index, settle, state
    cmd = args.vault_cmd
    k = _keys() if cmd not in ("reclassify", "claims", "resume") else {}
    wallet, ours = k.get("trading"), {x for x in (k.get("trading"), k.get("payout")) if x}
    if cmd == "status":
        from . import publish
        with transaction() as conn:
            s = publish.stats(conn, wallet)
        print(json.dumps({k: s[k] for k in ("fly", "wallet", "ledger", "vault", "settlement", "index")}, indent=2, default=str))
        print("halt:", state.halted())
    elif cmd == "scan":
        print(flows.scan_all(_rpc(), wallet, ours))
        print(rh_index.run_once())
    elif cmd == "custody":
        print(json.dumps(custody.check(_rpc(), wallet, k.get("payout")), indent=2, default=str))
    elif cmd == "squads-setup":
        print(squads_setup(k))
    elif cmd == "settle":
        if args.dry_run:
            import time
            with transaction() as conn:
                t1 = settle.period_end_at_or_before(time.time()) + int(config.VAULT_PERIOD_S)
                start, evs = settle.event_inputs(conn, config.RH_CHAIN_ID, t1 - int(config.VAULT_PERIOD_S), int(time.time()))
                w = settle.weights(start, evs, t1 - int(config.VAULT_PERIOD_S), int(time.time()))
                t = custody.treasury_address()
                native = _rpc().get_balance(wallet) + (_rpc().get_balance(t) if t else 0)
                cost, _ = settle.open_positions(conn)
                f = settle.flow_totals(conn, 2**62)
                r = settle.realized(native, 0, cost, f["deposits"], f["withdrawals"], f["payouts"])
                p = settle.plan(r, settle.allocated_total(conn), f["payouts"], native, int(config.GAS_RESERVE_SOL * config.LAMPORTS_PER_SOL))
            print(json.dumps({"period_end": t1, "realized_so_far": r, "plan": p, "allocation_if_now": settle.allocate(p["amount"], w)}, indent=2))
        else:
            print(settle.run_once(_rpc(), wallet, rh_index.run_once(), treasury=custody.treasury_address(), our_keys=ours))
    elif cmd == "reclassify":
        print(flows.reclassify(args.signature, args.kind, args.note), "row(s) changed")
    elif cmd == "withdraw":
        from . import withdraw
        t = custody.treasury_address()
        if not t:
            raise SystemExit("no treasury configured (VAULT_MULTISIG)")
        rpc = _rpc()
        with transaction() as conn:
            print(withdraw.instructions(withdraw.cap(conn, rpc.get_balance(wallet), rpc.get_balance(t)), t))
    elif cmd == "claims":
        with transaction() as conn:
            for r in conn.execute("SELECT id, relay_id, evm, sol, status, reason, lamports, tx_signature, updated_at FROM vault_claims ORDER BY id DESC LIMIT 30"):
                print(dict(r))
    elif cmd == "resume":
        state.resume()
        print("vault halt cleared; entries stay paused until you resume them (fly-trader resume-entries)")


def squads_setup(k: dict) -> str:
    """Step by step, for the owner's wallet in the Squads app. Prints; signs nothing."""
    from .. import config
    ms = config.VAULT_MULTISIG
    lines = ["Treasury setup in the Squads app (app.squads.so), signed by your owner wallet (Solflare now, the Ledger later).",
             "The server's keys must NEVER be members of the multisig; they only get the two spending limits below.", ""]
    if not ms:
        lines += ["1. Create a multisig: members = your owner wallet only, threshold 1, time lock 0.",
                  "   Then set VAULT_MULTISIG=<its address> in /srv/fly/vault.env and run this again.", ""]
    else:
        from . import custody
        lines += [f"Multisig {ms}; treasury (vault 0) {custody.treasury_address()} -- fund the treasury, not the trading wallet.", ""]
    lines += ["2. Settings -> Spending limits -> Add (L1: the trading float's refill):",
              "     token SOL, vault 0, period Daily, amount "
              f"{config.TREASURY_FLOAT_SOL:g} SOL,",
              f"     member (who may use it): {k.get('trading')}   <- the trading key",
              f"     destination (ONLY this): {k.get('trading')}   <- the trading wallet itself",
              "   Put its address in VAULT_LIMIT_TRADING.",
              "3. Add another (L2: claims):",
              "     token SOL, vault 0, period Weekly, amount 1 SOL (raise it when a settlement alert says owed > cap),",
              f"     member: {k.get('payout')}   <- the payout key",
              "     destinations: none (any holder)",
              "   Put its address in VAULT_LIMIT_PAYOUT.",
              "4. Restart the fly, then: fly-trader vault custody   (every check must pass).",
              "",
              "Emergency: in the Squads app remove L1 and L2 (Settings -> Spending limits -> Remove). The server can then move",
              "nothing out of the treasury, whatever it runs; the trading wallet keeps only its float."]
    return "\n".join(lines)
