#!/usr/bin/env python
"""Build the leave-one-peptide-out split tables: python make_datasets.py"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import pandas as pd

from gradft.data import PEPTIDES, build_table, load_tcrvdb

logger = logging.getLogger("make_datasets")
SOURCE = HERE / "datasets" / "source"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tcrvdb", type=Path, default=SOURCE / "tcrvdb_train.csv",
                   help="TCRvdb on NetTCR-2.2's schema (default: %(default)s).")
    p.add_argument("--raw-tcrvdb", type=Path, default=SOURCE / "TCRvdb.csv",
                   help="Raw TCRvdb, for the ambiguous=0 check (default: %(default)s).")
    p.add_argument("--no-require-raw", dest="require_raw", action="store_false",
                   help="Accept the table without verifying ambiguous=0. Not recommended.")
    p.add_argument("--outdir", type=Path, default=HERE / "datasets")
    p.add_argument("--peptides", nargs="+", default=PEPTIDES)
    p.add_argument("--fractions", nargs="+", type=float, default=[0.05, 0.10, 0.15, 0.20, 0.25, 0.30],
                   help="Share of each peptide's non-anchor training labels to flip (default: %(default)s).")
    p.add_argument("--test-frac", type=float, default=0.10, help="Seen-peptide test share, per peptide.")
    p.add_argument("--val-frac", type=float, default=0.10, help="Validation share of what remains.")
    p.add_argument("--anchors-per-class-per-group", type=int, default=25)
    p.add_argument("--seed", type=int, default=0, help="Split and noise seed (shared by every arm).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    df = load_tcrvdb(args.tcrvdb, args.raw_tcrvdb, require_raw=args.require_raw)
    args.outdir.mkdir(parents=True, exist_ok=True)

    rows = []
    for peptide in args.peptides:
        for frac in args.fractions:
            t = build_table(df, peptide, frac, test_frac=args.test_frac, val_frac=args.val_frac,
                            anchors_per_class_per_group=args.anchors_per_class_per_group, seed=args.seed)
            pct = int(round(frac * 100))
            path = args.outdir / f"lopo_{peptide}_f{pct}.csv"
            t.to_csv(path, index=False)
            tr = t[t.split == "train"]
            rows.append({"peptide": peptide, "pct_flipped": pct, "train": len(tr),
                         "flipped": int(tr.flipped.sum()), "anchors": int(t.anchor.sum()),
                         "val": int((t.split == "val").sum()), "test": int((t.split == "test").sum()),
                         "unseen": int((t.split == "unseen").sum()), "file": path.name})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"\nwrote {len(out)} tables to {args.outdir}")
    print("anchors keep their true labels; val, test and unseen are never flipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
