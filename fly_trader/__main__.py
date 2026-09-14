"""fly-trader command line. Every command is a thin wrapper over library functions that the
Streamlit console also calls. Always run as ``.venv/bin/python -m fly_trader <command>``."""
from __future__ import annotations

import argparse
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
    if args.probe:
        discovery.probe()
    elif args.once:
        discovery.run_once()
    else:
        discovery.run_forever()


def _cmd_capture(args):
    from .ingest import capture
    return capture.main()


def _cmd_build_connectome(args):
    from .brain import connectome_build
    connectome_build.main(annotations=args.annotations)


def _cmd_calibrate(args):
    from .brain import calibrate
    calibrate.main()


def _cmd_balance(args):
    from .brain import balance
    balance.main()


def _cmd_selftest(args):
    from .brain import calibrate
    calibrate.selftest()


def _cmd_pretrain(args):
    from .pretrain import trainer
    trainer.main(corpus=args.corpus, beat_s=args.beat_s, ticks=args.ticks, slots=args.slots,
                 max_beats=args.max_beats, days=args.days)


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


def _cmd_snapshot(args):
    from .brain import checkpoint
    print(checkpoint.snapshot_from_live(kind=args.kind))


def _cmd_promote(args):
    from .brain import checkpoint
    checkpoint.promote(args.id, hot=args.hot)


def _cmd_reset_training(args):
    from .ops.reset import reset_training_state
    import json
    print(json.dumps(reset_training_state(reason="cli", archive=not args.no_archive), indent=1))


def _cmd_build_dataset(args):
    from .train import dataset
    from .logging_setup import setup
    setup("dataset")
    print(dataset.build(hours=args.hours, min_swaps=args.min_swaps))


def _cmd_train_policy(args):
    from .train import ppo
    ppo.main(iterations=args.iterations, window=args.window, dataset=args.dataset, subgraph=args.subgraph, eval_every=args.eval_every,
             imitate_epochs=args.imitate, init_from=args.init_from)


def _cmd_ui(args):
    from .ops.supervisor import run_console
    run_console(port=args.port)


def _cmd_archive(args):
    from .ingest import tape
    tape.archive_partitions()


def _cmd_purge(args):
    from .ingest import tape
    tape.purge_archived(confirm=args.confirm)


def _cmd_verify_fills(args):
    from .execution import ledger
    ledger.verify_fills()


def _cmd_close_atas(args):
    from .execution import broker_live
    broker_live.close_empty_atas()


def _cmd_corpus_pull(args):
    from .ingest import corpus_pull
    corpus_pull.main()


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
    selector.main(days=args.days, top_frac=args.top_frac)


def _cmd_train_fly_selector(args):
    from .logging_setup import setup
    from .train import fly_selector
    setup("train")
    fly_selector.main(days=args.days, test_days=args.test_days, top_frac=args.top_frac, epochs=args.epochs, rows_per_epoch=args.rows)


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
    d = sub.add_parser("discover"); d.add_argument("--once", action="store_true"); d.add_argument("--probe", action="store_true"); d.set_defaults(fn=_cmd_discover)
    sub.add_parser("capture").set_defaults(fn=_cmd_capture)
    sub.add_parser("corpus-pull").set_defaults(fn=_cmd_corpus_pull)
    sub.add_parser("replay-pull").set_defaults(fn=_cmd_replay_pull)
    sub.add_parser("pumpstream").set_defaults(fn=_cmd_pumpstream)
    sub.add_parser("build-corpus-features").set_defaults(fn=_cmd_build_corpus_features)
    sub.add_parser("assemble-replay").set_defaults(fn=_cmd_assemble_replay)
    sub.add_parser("build-corpus-meta").set_defaults(fn=_cmd_build_corpus_meta)
    sub.add_parser("build-mature").set_defaults(fn=_cmd_build_mature)
    sub.add_parser("selector-eval").set_defaults(fn=_cmd_selector_eval)
    ts_ = sub.add_parser("train-selector"); ts_.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    ts_.add_argument("--top-frac", type=float, default=0.01); ts_.set_defaults(fn=_cmd_train_selector)
    tf_ = sub.add_parser("train-fly-selector"); tf_.add_argument("--days", type=int, default=None, help="most recent days only (default: the whole corpus)")
    tf_.add_argument("--test-days", type=int, default=21)
    tf_.add_argument("--top-frac", type=float, default=0.01); tf_.add_argument("--epochs", type=int, default=2); tf_.add_argument("--rows", type=int, default=600000)
    tf_.set_defaults(fn=_cmd_train_fly_selector)
    bc = sub.add_parser("backtest-corpus"); bc.add_argument("--max-tokens", type=int, default=None); bc.add_argument("--days", type=int, default=None)
    bc.add_argument("--fees", default="0.003,0.0055"); bc.add_argument("--only", default=None); bc.add_argument("--universe", default="graduation", choices=["graduation", "mature"]); bc.set_defaults(fn=_cmd_backtest_corpus)
    b = sub.add_parser("build-connectome"); b.add_argument("--annotations", default=None); b.set_defaults(fn=_cmd_build_connectome)
    sub.add_parser("calibrate-brain").set_defaults(fn=_cmd_calibrate)
    sub.add_parser("brain-selftest").set_defaults(fn=_cmd_selftest)
    sub.add_parser("balance-brain").set_defaults(fn=_cmd_balance)
    pt = sub.add_parser("pretrain"); pt.add_argument("--corpus", choices=["meteora", "capture"], default="meteora")
    pt.add_argument("--beat-s", type=float, default=None); pt.add_argument("--ticks", type=int, default=None)
    pt.add_argument("--slots", type=int, default=None); pt.add_argument("--max-beats", type=int, default=None)
    pt.add_argument("--days", type=float, default=None); pt.set_defaults(fn=_cmd_pretrain)
    sub.add_parser("run").set_defaults(fn=_cmd_run)
    s = sub.add_parser("swap-smoke"); s.add_argument("--sol", type=float, default=0.01); s.set_defaults(fn=_cmd_swap_smoke)
    sub.add_parser("status").set_defaults(fn=_cmd_status)
    r = sub.add_parser("reset-circuit"); r.add_argument("--kill", action="store_true"); r.set_defaults(fn=_cmd_reset_circuit)
    sub.add_parser("pause-entries").set_defaults(fn=_cmd_pause)
    sub.add_parser("resume-entries").set_defaults(fn=_cmd_resume)
    sn = sub.add_parser("snapshot"); sn.add_argument("--kind", default="manual"); sn.set_defaults(fn=_cmd_snapshot)
    pr = sub.add_parser("promote-checkpoint"); pr.add_argument("id", type=int); pr.add_argument("--hot", action="store_true"); pr.set_defaults(fn=_cmd_promote)
    u = sub.add_parser("ui"); u.add_argument("--port", type=int, default=8501); u.set_defaults(fn=_cmd_ui)
    bd = sub.add_parser("build-dataset"); bd.add_argument("--hours", type=float, default=None); bd.add_argument("--min-swaps", type=int, default=200); bd.set_defaults(fn=_cmd_build_dataset)
    tp = sub.add_parser("train-policy"); tp.add_argument("--iterations", type=int, default=None); tp.add_argument("--window", type=int, default=None)
    tp.add_argument("--dataset", default=None); tp.add_argument("--subgraph", choices=["central", "whole"], default=None); tp.add_argument("--eval-every", type=int, default=None)
    tp.add_argument("--imitate", type=int, default=0); tp.add_argument("--init-from", type=int, default=None); tp.set_defaults(fn=_cmd_train_policy)
    rt = sub.add_parser("reset-training"); rt.add_argument("--no-archive", action="store_true"); rt.set_defaults(fn=_cmd_reset_training)
    sub.add_parser("archive-partitions").set_defaults(fn=_cmd_archive)
    pg = sub.add_parser("purge-archived"); pg.add_argument("--confirm", action="store_true"); pg.set_defaults(fn=_cmd_purge)
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
