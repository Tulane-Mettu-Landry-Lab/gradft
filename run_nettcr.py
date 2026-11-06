#!/usr/bin/env python
"""The whole GradFT experiment on NetTCR-2.2: python run_nettcr.py [--smoke] [--jobs N]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import experiment as ex

from gradft import nettcr


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=HERE / "models" / "nettcr",
                   help="One directory per run, plus the result tables (default: %(default)s).")
    p.add_argument("--datasets", type=Path, default=HERE / "datasets")
    p.add_argument("--nettcr-dir", type=Path, default=HERE / "third_party" / "NetTCR-2.2",
                   help="A staged clone of the official NetTCR-2.2, or where to put one.")

    g = p.add_argument_group("the grid")
    g.add_argument("--peptides", nargs="+", default=ex.PEPTIDES, metavar="PEP")
    g.add_argument("--flips", nargs="+", type=int, default=ex.FLIPS, metavar="PCT",
                   help="Percent of training labels flipped (default: %(default)s).")
    g.add_argument("--seeds", nargs="+", type=int, default=ex.SEEDS, metavar="N")
    g.add_argument("--arms", nargs="+", default=ex.ARMS, choices=ex.ARMS)
    g.add_argument("--smoke", action="store_true",
                   help="One peptide, one flip rate, one seed, 12 epochs with warmup 5: "
                        "enough to prove both arms run end to end, not to measure anything.")

    r = p.add_argument_group("running")
    r.add_argument("--jobs", type=int, default=1,
                   help="Concurrent runs. Pair with --gpu-memory-limit (default: %(default)s).")
    r.add_argument("--gpu-memory-limit", type=int, default=None, metavar="MB",
                   help="Cap each run's GPU memory. Needed when --jobs > 1: TensorFlow's "
                        "memory growth is first-come-first-served, so one run that spikes "
                        "can starve the others. 1536 is ample for NetTCR-2.2.")
    r.add_argument("--rerun", action="store_true",
                   help="Retrain runs that already have a history.csv (default: keep them).")
    r.add_argument("--dry-run", action="store_true", help="Print the commands and stop.")
    r.add_argument("--report-only", action="store_true",
                   help="Train nothing; just collect and compare what is already on disk.")

    o = p.add_argument_group("overrides passed to every run")
    o.add_argument("--epochs", type=int)
    o.add_argument("--warmup", type=int, help="Default 50, from configs/nettcr_gradft.yaml.")
    o.add_argument("--exempt-rate", type=float, help="Default 0.40.")
    o.add_argument("--report-epoch", type=int, default=None,
                   help="Read every history.csv at this epoch, so both arms report the "
                        "same amount of training (default: each run's last epoch).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.peptides, args.flips, args.seeds = ["GILGFVFTL"], [20], [15]
        args.epochs, args.warmup = args.epochs or 12, args.warmup or 5
        args.report_epoch = args.report_epoch or args.epochs

    runs = ex.enumerate_runs(args.peptides, args.flips, args.seeds, args.arms)
    table_for = (lambda r: args.datasets / f"lopo_{r.peptide}_f{r.flip}.csv")

    print(f"NetTCR-2.2 + gradft | {len(args.peptides)} peptides x {len(args.flips)} flip "
          f"rates x {len(args.seeds)} seeds x {len(args.arms)} arms = {len(runs)} runs")
    print(f"  outdir : {args.outdir}")

    if not args.report_only:
        try:
            ex.ensure_tables(HERE, args.datasets, table_for, runs, mix=False)
        except FileNotFoundError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2
        repo = nettcr.ensure_nettcr(args.nettcr_dir)
        print(f"  nettcr : {repo}")

        def build(run):
            cmd = [sys.executable, str(HERE / "train_nettcr.py"),
                   "--table", str(table_for(run)),
                   "--outdir", str(args.outdir / run.name),
                   "--config", str(HERE / "configs" / f"nettcr_{run.arm}.yaml"),
                   "--nettcr-dir", str(args.nettcr_dir),
                   "--seed", str(run.seed)]
            if not args.rerun:
                cmd.append("--skip-existing")
            if args.gpu_memory_limit:
                cmd += ["--gpu-memory-limit", str(args.gpu_memory_limit)]
            if args.epochs:
                cmd += ["--epochs", str(args.epochs), "--patience", str(args.epochs)]
            if args.warmup is not None:
                cmd += ["--warmup", str(args.warmup)]
            if args.exempt_rate is not None:
                cmd += ["--exempt-rate", str(args.exempt_rate)]
            return cmd

        if args.jobs > 1 and not args.gpu_memory_limit:
            print(f"  NOTE   : --jobs {args.jobs} without --gpu-memory-limit. If these share "
                  "a GPU, one run can take all of it and the rest will fail with OOM.")
        ex.run_grid(runs, build, args.outdir, jobs=args.jobs, dry_run=args.dry_run)
        if args.dry_run:
            return 0

    frame = ex.collect(runs, args.outdir, args.report_epoch)
    if frame.empty:
        print(f"\nno finished runs under {args.outdir}; nothing to compare.")
        return 1
    ex.report(frame, args.outdir, "NetTCR-2.2: gradft vs vanilla")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
