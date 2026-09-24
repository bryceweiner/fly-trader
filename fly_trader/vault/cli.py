"""``fly-trader vault …``: the operator's commands for the hosted vault fly."""
from __future__ import annotations

import json


def add(sub) -> None:
    v = sub.add_parser("vault", help="the $FLY vault: status, scan, settle, reclassify, withdraw, claims, resume")
    vs = v.add_subparsers(dest="vault_cmd", required=True)
    vs.add_parser("status", help="ledger, index, halt state, last settlement")
    vs.add_parser("scan", help="classify new wallet transactions and index Robinhood Chain once")
    st = vs.add_parser("settle", help="advance settlement one step (snapshot / allocate)")
    st.add_argument("--dry-run", action="store_true", help="print what the current period would allocate, write nothing")
    rc = vs.add_parser("reclassify", help="relabel one transaction's flow")
    rc.add_argument("signature"); rc.add_argument("kind", choices=["deposit", "profit", "internal"]); rc.add_argument("--note")
    wd = vs.add_parser("withdraw", help="send principal back to a funding address")
    wd.add_argument("--sol", type=float, required=True); wd.add_argument("--to", required=True)
    vs.add_parser("claims", help="recent claims")
    vs.add_parser("resume", help="clear a vault halt (after you checked why it halted)")
    v.set_defaults(fn=run)


def _rpc():
    from .. import config
    from ..chain.rpc import HttpSolanaRpc
    return HttpSolanaRpc(config.vault_solana_rpc_url())


def run(args) -> None:
    from .. import config
    from ..chain.keys import load_keypair
    from ..db.connection import transaction
    from . import flows, rh_index, settle, state
    cmd = args.vault_cmd
    if cmd == "status":
        from . import publish
        with transaction() as conn:
            s = publish.stats(conn, str(load_keypair().pubkey()))
        print(json.dumps({k: s[k] for k in ("fly", "wallet", "ledger", "vault", "settlement", "index")}, indent=2, default=str))
        print("halt:", state.halted())
    elif cmd == "scan":
        print(flows.scan(_rpc(), str(load_keypair().pubkey())))
        print(rh_index.run_once())
    elif cmd == "settle":
        if args.dry_run:
            import time
            with transaction() as conn:
                t1 = settle.period_end_at_or_before(time.time()) + int(config.VAULT_PERIOD_S)
                start, evs = settle.event_inputs(conn, config.RH_CHAIN_ID, t1 - int(config.VAULT_PERIOD_S), int(time.time()))
                w = settle.weights(start, evs, t1 - int(config.VAULT_PERIOD_S), int(time.time()))
                native = _rpc().get_balance(str(load_keypair().pubkey()))
                cost, _ = settle.open_positions(conn)
                f = settle.flow_totals(conn, 2**62)
                r = settle.realized(native, 0, cost, f["deposits"], f["withdrawals"], f["payouts"])
                p = settle.plan(r, settle.allocated_total(conn), f["payouts"], native, int(config.GAS_RESERVE_SOL * config.LAMPORTS_PER_SOL))
            print(json.dumps({"period_end": t1, "realized_so_far": r, "plan": p, "allocation_if_now": settle.allocate(p["amount"], w)}, indent=2))
        else:
            print(settle.run_once(_rpc(), str(load_keypair().pubkey()), rh_index.run_once()))
    elif cmd == "reclassify":
        print(flows.reclassify(args.signature, args.kind, args.note), "row(s) changed")
    elif cmd == "withdraw":
        from . import withdraw
        print(withdraw.withdraw(int(round(args.sol * config.LAMPORTS_PER_SOL)), args.to, _rpc(), load_keypair()))
    elif cmd == "claims":
        with transaction() as conn:
            for r in conn.execute("SELECT id, relay_id, evm, sol, status, reason, lamports, tx_signature, updated_at FROM vault_claims ORDER BY id DESC LIMIT 30"):
                print(dict(r))
    elif cmd == "resume":
        state.resume()
        print("vault halt cleared; entries stay paused until you resume them (fly-trader resume-entries)")
