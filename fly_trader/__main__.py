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


def _cmd_train_selector(args):
    from .logging_setup import setup
    from .train import selector
    setup("train")
    selector.main(days=args.days)


def _cmd_train_fly_selector(args):
    from .logging_setup import setup
    from .train import fly_selector
    setup("train")
    fly_selector.main(days=args.days, epochs=args.epochs)


def _cmd_fly_replay(args):
    from .logging_setup import setup
    from .train import fly_replay
    setup("train")
    out = fly_replay.main(days=args.days, start_day=args.start_day)
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


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="fly-trader")
    sub = p.add_subparsers(dest="command", required=True)
    sub.add_parser("init-db").set_defaults(fn=_cmd_init_db)
    w = sub.add_parser("wallet"); w.add_argument("action", choices=["new", "show"]); w.set_defaults(fn=_cmd_wallet)
    d = sub.add_parser("discover"); d.add_argument("--once", action="store_true"); d.set_defaults(fn=_cmd_discover)
    sub.add_parser("replay-pull").set_defaults(fn=_cmd_replay_pull)
    sub.add_parser("pumpstream").set_defaults(fn=_cmd_pumpstream)
    sub.add_parser("build-corpus-features").set_defaults(fn=_cmd_build_corpus_features)
    sub.add_parser("assemble-replay").set_defaults(fn=_cmd_assemble_replay)
    sub.add_parser("build-corpus-meta").set_defaults(fn=_cmd_build_corpus_meta)
    sub.add_parser("build-mature").set_defaults(fn=_cmd_build_mature)
    sub.add_parser("selector-eval").set_defaults(fn=_cmd_selector_eval)
    ts_ = sub.add_parser("train-selector"); ts_.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    ts_.set_defaults(fn=_cmd_train_selector)
    tf_ = sub.add_parser("train-fly-selector"); tf_.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    tf_.add_argument("--epochs", type=int, default=1, help="full passes over the distillation rows")
    tf_.set_defaults(fn=_cmd_train_fly_selector)
    fr = sub.add_parser("fly-replay", help="bootstrap the fly once, let it learn over the corpus, judge it (the gate before it trades)")
    fr.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    fr.add_argument("--start-day", type=int, default=30, help="day index of the bootstrap (the selector teaches on the days before it)")
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
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    rc = args.fn(args)
    return rc if isinstance(rc, int) else 0


if __name__ == "__main__":
    sys.exit(main())
