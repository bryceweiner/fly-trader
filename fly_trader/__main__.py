"""fly-trader command line. Every command is a thin wrapper over library functions that the
Streamlit console also calls. Always run as ``.venv/bin/python -m fly_trader <command>``."""
from __future__ import annotations

import argparse
import json
import sys


def _cmd_init_db(args):
    from .db import schema
    from .db.connection import database_name
    created = schema.ensure_database()
    version = schema.apply_schema()
    tables = schema.table_names()
    print(f"database={database_name()} created={created} schema_version={version} tables={len(tables)}")
    for t in tables:
        print(f"  {t}")


def _cmd_wallet(args):
    from .chain import keys
    if args.action == "new":
        print(keys.create_wallet())
    else:
        print(keys.show_wallet())


def _cmd_discover(args):
    from .ingest import discovery
    if args.once:
        discovery.run_once()
    else:
        discovery.run_forever()


def _cmd_build_connectome(args):
    from .brain import connectome_build
    connectome_build.main(annotations=args.annotations)


def _cmd_run(args):
    from .agent import runner
    runner.main()


def _cmd_swap_smoke(args):
    from .execution import broker_live
    broker_live.swap_smoke(args.sol)


def _cmd_status(args):
    from .db import queries
    queries.print_status()


def _cmd_reset_circuit(args):
    from .agent import rails
    rails.reset_circuit(kill=args.kill)


def _cmd_pause(args):
    from .agent import rails
    rails.set_entries_paused(True)


def _cmd_resume(args):
    from .agent import rails
    rails.set_entries_paused(False)


def _cmd_reset_training(args):
    from .ops.reset import reset_training_state
    import json
    print(json.dumps(reset_training_state(reason="cli", archive=not args.no_archive), indent=1))


def _cmd_ui(args):
    from .ops.supervisor import run_console
    run_console(port=args.port)


def _cmd_verify_fills(args):
    from .execution import ledger
    ledger.verify_fills()


def _cmd_close_atas(args):
    from .execution import broker_live
    broker_live.close_empty_atas()


def _cmd_replay_pull(args):
    from .ingest import replay_pull
    replay_pull.main()


def _cmd_pumpstream(args):
    from .ingest import pumpstream
    pumpstream.main()


def _cmd_build_corpus_features(args):
    from .train import corpus_features
    corpus_features.main()


def _cmd_refetch_pool_ids(args):
    from .ingest import replay_pull
    print(f"{replay_pull.refetch_missing_pool_id()} archive hours marked for re-download (run the replay worker or fly-trader replay-pull)")


def _cmd_fit_wallet_skill(args):
    from .logging_setup import setup
    from .train import wallet_skill
    setup("train")
    out = wallet_skill.fit(days=args.days)
    print(json.dumps({k: out.get(k) for k in ("h", "L", "selection", "evaluation", "baseline", "beats_baseline", "candidates", "stopped")}, indent=1, default=str))


def _cmd_train_selector(args):
    from .logging_setup import setup
    from .train import selector
    setup("train")
    selector.main(days=args.days)


def _use_device(args) -> None:
    """--device overrides DEVICE from the environment for this command (brain/device.py resolves it)."""
    if getattr(args, "device", None):
        from . import config
        from .brain import device
        config.DEVICE = args.device; device.reset()


def _cmd_train_fly_selector(args):
    from .logging_setup import setup
    from .train import fly_selector
    setup("train"); _use_device(args)
    fly_selector.main(days=args.days, epochs=args.epochs)


def _cmd_fly_replay(args):
    from .logging_setup import setup
    from .train import fly_replay
    setup("train"); _use_device(args)
    out = fly_replay.main(days=args.days, start_day=args.start_day if args.start_day is not None else fly_replay.DEFAULT_START)
    print(json.dumps({k: out.get(k) for k in ("S", "passed", "reason", "alpha", "half_life_days", "evaluation", "random", "frozen", "plastic_beats_frozen")}, indent=1, default=str))


def _cmd_selector_eval(args):
    import runpy
    runpy.run_module("fly_trader.train.selector_eval", run_name="__main__")


def _cmd_build_mature(args):
    from .train import mature
    mature.main()


def _cmd_build_corpus_meta(args):
    from .train import corpus_meta
    corpus_meta.main()


def _cmd_assemble_replay(args):
    from .train import replay_assemble
    replay_assemble.main()


def _cmd_backtest_corpus(args):
    from .train import corpus_backtest
    argv = ["--fees", args.fees]
    if args.max_tokens: argv += ["--max-tokens", str(args.max_tokens)]
    if args.days: argv += ["--days", str(args.days)]
    if args.only: argv += ["--only", args.only]
    argv += ["--universe", args.universe]
    corpus_backtest.main(argv)


def _cmd_worker(args):
    from .ops import procs
    procs.worker_cli(args.action, args.name)


def _cmd_kalshi_history(args):
    from .kalshi import history
    history.main()


def _cmd_kalshi_build(args):
    from .kalshi import mature
    mature.main()


def _cmd_kalshi_train_selector(args):
    from .kalshi import selector
    from .logging_setup import setup
    setup("kalshi_train")
    out = selector.main(days=args.days)
    print(json.dumps({k: out.get(k) for k in ("snapshot_id", "deployable", "deploy_reason")}, indent=1, default=str))


def _cmd_kalshi_train_fly(args):
    from .kalshi import fly
    from .logging_setup import setup
    setup("kalshi_train"); _use_device(args)
    out = fly.main(days=args.days, epochs=args.epochs)
    print(json.dumps({k: out.get(k) for k in ("snapshot_id", "S", "lines", "calibration", "diagnostics", "gates_ok", "gate_failures", "teacher")}, indent=1, default=str))


def _cmd_kalshi_fly_replay(args):
    from .kalshi import fly_replay
    from .logging_setup import setup
    setup("kalshi_train"); _use_device(args)
    out = fly_replay.main(days=args.days, start_day=args.start_day if args.start_day is not None else fly_replay.DEFAULT_START)
    print(json.dumps({k: out.get(k) for k in ("S", "passed", "reason", "alpha", "half_life_days", "evaluation", "random", "frozen", "plastic_beats_frozen")}, indent=1, default=str))


def _cmd_kalshi_train(args):
    from .kalshi import pipeline
    pipeline.main()


def _cmd_kalshi_stream(args):
    from .kalshi import stream
    stream.main()


def _cmd_kalshi_run(args):
    from .kalshi import engine
    engine.main()


def _cmd_kalshi_subaccount(args):
    from .kalshi import client
    if args.action == "create":
        print(f"subaccount {client.subaccount_create()} created and written to .env")
    else:
        print(json.dumps(client.rest().subaccount_balances(), indent=1, default=str))


def _cmd_kalshi_fund(args):
    from .kalshi import client
    print(json.dumps(client.fund(args.usd), indent=1, default=str))


def _cmd_kalshi_status(args):
    from .kalshi import client
    print(json.dumps(client.status(), indent=1, default=str))


def _cmd_kalshi_pause(args):
    from .agent import rails
    rails.set_entries_paused(True, rails.KALSHI_CIRCUIT)


def _cmd_kalshi_resume(args):
    from .agent import rails
    rails.set_entries_paused(False, rails.KALSHI_CIRCUIT)


def _cmd_kalshi_reset_circuit(args):
    from .agent import rails
    rails.reset_circuit(kill=args.kill, circuit_id=rails.KALSHI_CIRCUIT)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fly-trader")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(fn=_cmd_init_db)
    w = sub.add_parser("wallet"); w.add_argument("action", choices=["new", "show"]); w.set_defaults(fn=_cmd_wallet)
    d = sub.add_parser("discover"); d.add_argument("--once", action="store_true"); d.set_defaults(fn=_cmd_discover)
    sub.add_parser("replay-pull").set_defaults(fn=_cmd_replay_pull)
    sub.add_parser("refetch-pool-ids", help="mark archive hours without pool_id for re-download").set_defaults(fn=_cmd_refetch_pool_ids)
    fw = sub.add_parser("fit-wallet-skill", help="fit the wallet-skill tables' hold and lookback by the selector's objective (one-off, many hours, resumable)")
    fw.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)"); fw.set_defaults(fn=_cmd_fit_wallet_skill)
    sub.add_parser("pumpstream").set_defaults(fn=_cmd_pumpstream)
    sub.add_parser("build-corpus-features").set_defaults(fn=_cmd_build_corpus_features)
    sub.add_parser("assemble-replay").set_defaults(fn=_cmd_assemble_replay)
    sub.add_parser("build-corpus-meta").set_defaults(fn=_cmd_build_corpus_meta)
    sub.add_parser("build-mature").set_defaults(fn=_cmd_build_mature)
    sub.add_parser("selector-eval").set_defaults(fn=_cmd_selector_eval)
    ts_ = sub.add_parser("train-selector"); ts_.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    ts_.set_defaults(fn=_cmd_train_selector)
    tf_ = sub.add_parser("train-fly-selector"); tf_.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    tf_.add_argument("--epochs", type=int, default=None, help="cap on full passes over the distillation rows (default: until the fit stops improving)")
    tf_.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N | mps (default: DEVICE from .env, auto)")
    tf_.set_defaults(fn=_cmd_train_fly_selector)
    fr = sub.add_parser("fly-replay", help="bootstrap the fly once, let it learn over the corpus, judge it (the gate before it trades)")
    fr.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    fr.add_argument("--device", default=None, help="auto | cpu | cuda | cuda:N | mps (default: DEVICE from .env, auto)")
    fr.add_argument("--start-day", type=int, default=None,
                    help="day index of the bootstrap (the selector teaches on the days before it); default: train/fly_replay.DEFAULT_START, "
                         "the earliest day that leaves the teacher's walk-forward enough days in each half")
    fr.set_defaults(fn=_cmd_fly_replay)
    bc = sub.add_parser("backtest-corpus"); bc.add_argument("--max-tokens", type=int, default=None); bc.add_argument("--days", type=int, default=None)
    bc.add_argument("--fees", default="0.003,0.0055"); bc.add_argument("--only", default=None); bc.add_argument("--universe", default="graduation", choices=["graduation", "mature"]); bc.set_defaults(fn=_cmd_backtest_corpus)
    b = sub.add_parser("build-connectome"); b.add_argument("--annotations", default=None); b.set_defaults(fn=_cmd_build_connectome)
    sub.add_parser("run").set_defaults(fn=_cmd_run)
    s = sub.add_parser("swap-smoke"); s.add_argument("--sol", type=float, default=0.01); s.set_defaults(fn=_cmd_swap_smoke)
    sub.add_parser("status").set_defaults(fn=_cmd_status)
    r = sub.add_parser("reset-circuit"); r.add_argument("--kill", action="store_true"); r.set_defaults(fn=_cmd_reset_circuit)
    sub.add_parser("pause-entries").set_defaults(fn=_cmd_pause)
    sub.add_parser("resume-entries").set_defaults(fn=_cmd_resume)
    u = sub.add_parser("ui"); u.add_argument("--port", type=int, default=8501); u.set_defaults(fn=_cmd_ui)
    rt = sub.add_parser("reset-training"); rt.add_argument("--no-archive", action="store_true"); rt.set_defaults(fn=_cmd_reset_training)
    sub.add_parser("verify-fills").set_defaults(fn=_cmd_verify_fills)
    sub.add_parser("close-empty-atas").set_defaults(fn=_cmd_close_atas)
    wk = sub.add_parser("worker"); wk.add_argument("action", choices=["start", "stop", "status"]); wk.add_argument("name", nargs="?"); wk.set_defaults(fn=_cmd_worker)
    # ---- Kalshi prediction markets (fly_trader/kalshi) ----
    ks = sub.add_parser("kalshi-subaccount", help="create the fly's dedicated Kalshi subaccount (writes KALSHI_SUBACCOUNT to .env) or list balances")
    ks.add_argument("action", choices=["create", "list"]); ks.set_defaults(fn=_cmd_kalshi_subaccount)
    kf = sub.add_parser("kalshi-fund", help="move dollars from the primary Kalshi account into the fly's subaccount"); kf.add_argument("--usd", type=float, required=True); kf.set_defaults(fn=_cmd_kalshi_fund)
    sub.add_parser("kalshi-status", help="exchange status, subaccount balance, resting orders, positions, settlements").set_defaults(fn=_cmd_kalshi_status)
    sub.add_parser("kalshi-history", help="the Kalshi corpus worker: dataset seed, settled markets, candles, trades").set_defaults(fn=_cmd_kalshi_history)
    sub.add_parser("kalshi-build", help="build the Kalshi feature parts from the corpus").set_defaults(fn=_cmd_kalshi_build)
    kts = sub.add_parser("kalshi-train-selector", help="fit the Kalshi strategy stack walk-forward and save it"); kts.add_argument("--days", type=int, default=None)
    kts.set_defaults(fn=_cmd_kalshi_train_selector)
    ktf = sub.add_parser("kalshi-train-fly", help="bootstrap the Kalshi fly (optic lobes) from the deployable Kalshi selector")
    ktf.add_argument("--days", type=int, default=None); ktf.add_argument("--epochs", type=int, default=None); ktf.add_argument("--device", default=None)
    ktf.set_defaults(fn=_cmd_kalshi_train_fly)
    kfr = sub.add_parser("kalshi-fly-replay", help="bootstrap the Kalshi fly once, let it learn from settlements over the corpus, judge it")
    kfr.add_argument("--days", type=int, default=None); kfr.add_argument("--device", default=None); kfr.add_argument("--start-day", type=int, default=None)
    kfr.set_defaults(fn=_cmd_kalshi_fly_replay)
    sub.add_parser("kalshi-train", help="the Kalshi training pipeline worker (selector weekly, fly bootstrap when needed)").set_defaults(fn=_cmd_kalshi_train)
    sub.add_parser("kalshi-stream", help="the Kalshi live feed worker (websocket → kalshi_minutes / kalshi_quotes)").set_defaults(fn=_cmd_kalshi_stream)
    sub.add_parser("kalshi-run", help="the Kalshi trading engine worker (the visual fly's paper arms, live mirror behind KALSHI_LIVE_ENABLED)").set_defaults(fn=_cmd_kalshi_run)
    sub.add_parser("kalshi-pause-entries").set_defaults(fn=_cmd_kalshi_pause)
    sub.add_parser("kalshi-resume-entries").set_defaults(fn=_cmd_kalshi_resume)
    kr = sub.add_parser("kalshi-reset-circuit"); kr.add_argument("--kill", action="store_true"); kr.set_defaults(fn=_cmd_kalshi_reset_circuit)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rc = args.fn(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
