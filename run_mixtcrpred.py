#!/usr/bin/env python
"""The whole GradFT experiment on MixTCRpred: python run_mixtcrpred.py [--smoke]"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import experiment as ex

from gradft import mixtcrpred


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outdir", type=Path, default=HERE / "models" / "mixtcrpred",
                   help="One directory per run, plus the result tables (default: %(default)s).")
    p.add_argument("--datasets", type=Path, default=HERE / "datasets")
    p.add_argument("--mixtcrpred-dir", type=Path, default=HERE / "third_party" / "MixTCRpred",
                   help="A staged clone of the official MixTCRpred, or where to put one.")
    p.add_argument("--device", default="cuda", help="Falls back to CPU if CUDA is absent.")

    g = p.add_argument_group("the grid")
    g.add_argument("--peptides", nargs="+", default=ex.PEPTIDES, metavar="PEP")
    g.add_argument("--flips", nargs="+", type=int, default=ex.FLIPS, metavar="PCT",
                   help="Percent of training labels flipped (default: %(default)s).")
    g.add_argument("--seeds", nargs="+", type=int, default=ex.SEEDS, metavar="N")
    g.add_argument("--arms", nargs="+", default=ex.ARMS, choices=ex.ARMS)
    g.add_argument("--smoke", action="store_true",
                   help="One peptide, one flip rate, one seed, 8 epochs with warmup 2: "
                        "enough to prove both arms run end to end, not to measure anything.")

    r = p.add_argument_group("running")
    r.add_argument("--jobs", type=int, default=1,
                   help="Concurrent runs. This model is large; 1-2 per GPU (default: %(default)s).")
    r.add_argument("--rerun", action="store_true",
                   help="Retrain runs that already have a history.csv (default: keep them).")
    r.add_argument("--dry-run", action="store_true", help="Print the commands and stop.")
    r.add_argument("--report-only", action="store_true",
                   help="Train nothing; just collect and compare what is already on disk.")

    o = p.add_argument_group("overrides passed to every run")
    o.add_argument("--epochs", type=int)
    o.add_argument("--warmup", type=int, help="Default 5, from configs/mixtcrpred_gradft.yaml.")
    o.add_argument("--exempt-rate", type=float, help="Default 0.40.")
    o.add_argument("--report-epoch", type=int, default=None,
                   help="Read every history.csv at this epoch, so both arms report the "
                        "same amount of training (default: each run's last epoch).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.smoke:
        args.peptides, args.flips, args.seeds = ["GILGFVFTL"], [20], [15]
        args.epochs, args.warmup = args.epochs or 8, args.warmup or 2
        args.report_epoch = args.report_epoch or args.epochs

    runs = ex.enumerate_runs(args.peptides, args.flips, args.seeds, args.arms)
    table_for = (lambda r: args.datasets / f"mix_lopo_{r.peptide}_f{r.flip}.csv")

    print(f"MixTCRpred + gradft | {len(args.peptides)} peptides x {len(args.flips)} flip "
          f"rates x {len(args.seeds)} seeds x {len(args.arms)} arms = {len(runs)} runs")
    print(f"  outdir : {args.outdir}")

    if not args.report_only:
        try:
            ex.ensure_tables(HERE, args.datasets, table_for, runs, mix=True)
        except FileNotFoundError as exc:
            print(f"\n{exc}", file=sys.stderr)
            return 2
        repo = mixtcrpred.ensure_mixtcrpred(args.mixtcrpred_dir)
        print(f"  mixtcr : {repo}")

        def build(run):
            cmd = [sys.executable, str(HERE / "train_mixtcrpred.py"),
                   "--table", str(table_for(run)),
                   "--outdir", str(args.outdir / run.name),
                   "--config", str(HERE / "configs" / f"mixtcrpred_{run.arm}.yaml"),
                   "--mixtcrpred-dir", str(args.mixtcrpred_dir),
                   "--device", args.device,
                   "--seed", str(run.seed)]
            if not args.rerun:
                cmd.append("--skip-existing")
            if args.epochs:
                cmd += ["--epochs", str(args.epochs), "--patience", str(args.epochs)]
            if args.warmup is not None:
                cmd += ["--warmup", str(args.warmup)]
            if args.exempt_rate is not None:
                cmd += ["--exempt-rate", str(args.exempt_rate)]
            return cmd

        ex.run_grid(runs, build, args.outdir, jobs=args.jobs, dry_run=args.dry_run)
        if args.dry_run:
            return 0

    frame = ex.collect(runs, args.outdir, args.report_epoch)
    if frame.empty:
        print(f"\nno finished runs under {args.outdir}; nothing to compare.")
        return 1
    ex.report(frame, args.outdir, "MixTCRpred: gradft vs vanilla")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
