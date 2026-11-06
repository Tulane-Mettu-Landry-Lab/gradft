#!/usr/bin/env python
"""Re-express the split tables on MixTCRpred's schema: python make_mix_datasets.py"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import pandas as pd

FIELDS = {"epitope": "epitope_aa", "cdr3_TRA": "cdr3_alpha_aa", "cdr3_TRB": "cdr3_beta_aa",
          "TRAV": "TRAV_IMGT", "TRAJ": "TRAJ_IMGT", "TRBV": "TRBV_IMGT", "TRBJ": "TRBJ_IMGT"}
GENES = ("TRAV", "TRAJ", "TRBV", "TRBJ")
CARRY = ("tcr_id", "peptide", "split", "anchor", "flipped")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--datasets", type=Path, default=HERE / "datasets")
    p.add_argument("--raw", type=Path, default=HERE / "datasets" / "source" / "TCRvdb.csv",
                   help="The original TCRvdb.csv, which carries the gene names.")
    p.add_argument("--peptide", help="Only this held-out peptide (default: every table).")
    p.add_argument("--prefix", default="mix_", help="Prefix for the written tables.")
    return p.parse_args(argv)


def convert(table: Path, raw: pd.DataFrame, out: Path) -> dict:
    df = pd.read_csv(table)
    for col in CARRY + ("binder",):
        if col not in df.columns:
            raise ValueError(f"{table} has no {col!r} column; build it with make_datasets.py")

    merged = df.merge(raw, on="tcr_id", how="left", validate="one_to_one")
    missing = merged[FIELDS["cdr3_TRA"]].isna() | merged[FIELDS["cdr3_TRB"]].isna()

    res = pd.DataFrame({name: merged[src] for name, src in FIELDS.items()})
    for col in GENES:
        res[col] = res[col].astype("string").str.replace("_", "/", regex=False)
    res["tp"] = merged["binder"].to_numpy(int)
    for col in CARRY:
        res[col] = merged[col].to_numpy()

    res = res[~missing.to_numpy()].reset_index(drop=True)
    if int(res.loc[res.anchor == 1, "flipped"].sum()):
        raise AssertionError(f"{table}: an anchor carries a flipped label")
    bad = res.peptide.ne(res.epitope).sum()
    if bad:
        raise AssertionError(f"{table}: {bad} row(s) whose epitope does not match its peptide")

    res.to_csv(out, index=False)
    counts = res.split.value_counts().to_dict()
    return {"file": out.name, "rows": len(res), "dropped_no_cdr3": int(missing.sum()),
            "anchors": int((res.anchor == 1).sum()), "flipped": int(res.flipped.sum()),
            "binder_rate": round(float(res.tp.mean()), 3),
            **{f"n_{k}": int(v) for k, v in sorted(counts.items())}}


def main(argv=None) -> int:
    args = parse_args(argv)
    if not args.raw.is_file():
        raise FileNotFoundError(
            f"{args.raw} not found. The gene names live only in the raw TCRvdb, and "
            "MixTCRpred needs them to look up CDR1/CDR2; pass --raw.")
    raw = pd.read_csv(args.raw)
    keep = ["tcr_id"] + sorted(set(FIELDS.values()))
    raw = raw[keep].drop_duplicates("tcr_id")

    pattern = f"lopo_{args.peptide}_f*.csv" if args.peptide else "lopo_*_f*.csv"
    tables = sorted(p for p in args.datasets.glob(pattern) if not p.name.startswith(args.prefix))
    if not tables:
        raise FileNotFoundError(f"no {pattern} under {args.datasets}; run make_datasets.py first")

    report = [convert(t, raw, args.datasets / f"{args.prefix}{t.name}") for t in tables]
    summary = pd.DataFrame(report)
    print(summary.to_string(index=False))
    (args.datasets / f"{args.prefix}manifest.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {len(report)} table(s) to {args.datasets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
