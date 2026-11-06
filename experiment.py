#!/usr/bin/env python
"""The grid driver shared by run_nettcr.py and run_mixtcrpred.py."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PEPTIDES = ["GILGFVFTL", "GLCTLVAML", "NLVPMVATV", "TTDPSFLGRY", "YLQPRTFLL"]
FLIPS = [5, 10, 15, 20, 25, 30]
SEEDS = [15, 16, 17]
ARMS = ["gradft", "vanilla"]

METRICS = {
    "test_auc_01": "seen peptides, ROC-AUC@FPR<=0.1",
    "test_roc_auc": "seen peptides, ROC-AUC",
    "unseen_auc_01": "held-out peptide, ROC-AUC@FPR<=0.1",
    "unseen_roc_auc": "held-out peptide, ROC-AUC",
}


@dataclass(frozen=True)
class Run:
    arm: str
    peptide: str
    flip: int
    seed: int

    @property
    def name(self) -> str:
        return f"{self.arm}_{self.peptide}_f{self.flip}_s{self.seed}"


def enumerate_runs(peptides: Sequence[str], flips: Sequence[int], seeds: Sequence[int],
                   arms: Sequence[str]) -> List[Run]:
    return [Run(arm, pep, int(flip), int(seed))
            for pep in peptides for flip in flips for seed in seeds for arm in arms]



RAW_TCRVDB = "TCRvdb.csv"
PREPARED_TCRVDB = "tcrvdb_train.csv"

MISSING_DATA = """\
No split tables, and no TCRvdb to build them from.

  1. Download the TCRvdb release and save it as
       {raw}
     The README lists the columns it must carry.

  2. Harmonise and split it:
       python prepare_tcrvdb.py
       python make_datasets.py{mix}

Both steps are deterministic, so everyone who starts from the same TCRvdb release gets
the same splits, the same anchor set and the same flipped labels.
"""


def ensure_tables(here: Path, datasets: Path, table_for: Callable[[Run], Path],
                  runs: Sequence[Run], mix: bool) -> None:
    missing = sorted({table_for(r) for r in runs if not table_for(r).is_file()})
    if not missing:
        return
    source = datasets / "source"
    if not (source / PREPARED_TCRVDB).is_file():
        if not (source / RAW_TCRVDB).is_file():
            raise FileNotFoundError(MISSING_DATA.format(
                raw=source / RAW_TCRVDB,
                mix="\n       python make_mix_datasets.py" if mix else ""))
        print(f"{source / PREPARED_TCRVDB} missing -- harmonising {RAW_TCRVDB} first")
        subprocess.run([sys.executable, str(here / "prepare_tcrvdb.py"),
                        "--input", str(source / RAW_TCRVDB),
                        "--out", str(source / PREPARED_TCRVDB)], check=True, cwd=here)

    print(f"{len(missing)} split table(s) missing, e.g. {missing[0].name} -- building them")
    scripts = [[sys.executable, str(here / "make_datasets.py"),
                "--tcrvdb", str(source / PREPARED_TCRVDB),
                "--raw-tcrvdb", str(source / RAW_TCRVDB), "--outdir", str(datasets)]]
    if mix:
        scripts.append([sys.executable, str(here / "make_mix_datasets.py"),
                        "--datasets", str(datasets), "--raw", str(source / RAW_TCRVDB)])
    for cmd in scripts:
        print("  $", " ".join(cmd))
        subprocess.run(cmd, check=True, cwd=here)
    still = [p for p in missing if not p.is_file()]
    if still:
        raise FileNotFoundError(
            f"{len(still)} table(s) still missing after building, e.g. {still[0]}.")



def run_grid(runs: Sequence[Run], build_cmd: Callable[[Run], List[str]], outdir: Path,
             jobs: int = 1, dry_run: bool = False) -> Dict[str, int]:
    outdir.mkdir(parents=True, exist_ok=True)
    total, done, t_start = len(runs), 0, time.time()
    codes: Dict[str, int] = {}

    def one(run: Run) -> int:
        cmd = build_cmd(run)
        if dry_run:
            print("  $", " ".join(cmd))
            return 0
        log = outdir / f"{run.name}.log"
        with log.open("w") as fh:
            return subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT).returncode

    with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        for run, code in zip(runs, pool.map(one, runs)):
            done += 1
            codes[run.name] = code
            if dry_run:
                continue
            rate = (time.time() - t_start) / done
            left = rate * (total - done)
            print(f"[{done:>4}/{total}] {run.name:<40} "
                  f"{'ok' if code == 0 else f'FAILED (exit {code})'}"
                  f"  ~{left / 60:.0f} min left", flush=True)
    bad = {k: v for k, v in codes.items() if v}
    if bad:
        print(f"\n{len(bad)} run(s) failed; their logs are in {outdir}:")
        for name in sorted(bad)[:10]:
            print(f"  {outdir / (name + '.log')}")
    return codes



def collect(runs: Sequence[Run], outdir: Path, report_epoch: Optional[int]) -> pd.DataFrame:
    rows = []
    for run in runs:
        hist = outdir / run.name / "history.csv"
        if not hist.is_file():
            continue
        h = pd.read_csv(hist)
        if report_epoch is not None:
            h = h[h.epoch <= report_epoch]
        if h.empty:
            continue
        last = h.iloc[-1]
        row = {"arm": run.arm, "peptide": run.peptide, "flip": run.flip, "seed": run.seed,
               "epoch": int(last.epoch)}
        row.update({m: float(last[m]) for m in METRICS if m in h.columns})
        row["suppressed_rate"] = float(last.get("suppressed_rate", np.nan))
        summary = outdir / run.name / "summary.json"
        if summary.is_file():
            s = json.loads(summary.read_text())
            row["selected_epoch"] = s.get("selected_epoch")
            row["flipped_train"] = s.get("rows", {}).get("flipped_train")
        rows.append(row)
    return pd.DataFrame(rows)


def _paired(delta: np.ndarray) -> Dict[str, float]:
    from scipy import stats

    d = np.asarray(delta, float)
    d = d[np.isfinite(d)]
    out = {"n": len(d), "mean": float(d.mean()) if len(d) else np.nan,
           "se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else np.nan,
           "p_t": np.nan, "p_wilcoxon": np.nan}
    if len(d) > 1 and np.any(d != 0):
        out["p_t"] = float(stats.ttest_1samp(d, 0.0).pvalue)
        try:
            out["p_wilcoxon"] = float(stats.wilcoxon(d).pvalue)
        except ValueError:
            pass
    return out


def compare(frame: pd.DataFrame, metric: str) -> Dict[str, object]:
    if metric not in frame.columns:
        return {}
    keys = ["peptide", "flip", "seed"]
    wide = frame.pivot_table(index=keys, columns="arm", values=metric)
    if not {"gradft", "vanilla"} <= set(wide.columns):
        return {}
    wide = wide.dropna(subset=["gradft", "vanilla"])
    per_cell = _paired((wide.gradft - wide.vanilla).to_numpy())

    pep = wide.groupby(level=["peptide", "flip"]).mean()
    per_fold = _paired((pep.gradft - pep.vanilla).to_numpy())
    return {"metric": metric, "label": METRICS.get(metric, metric),
            "gradft": float(wide.gradft.mean()), "vanilla": float(wide.vanilla.mean()),
            "cell": per_cell, "fold": per_fold}


def report(frame: pd.DataFrame, outdir: Path, title: str) -> None:
    frame = frame.sort_values(["peptide", "flip", "seed", "arm"])
    frame.to_csv(outdir / "results_runs.csv", index=False)

    width = 78
    print("\n" + "=" * width)
    print(title)
    print("=" * width)
    arms = sorted(frame.arm.unique())
    print(f"runs collected: {len(frame)}  ({', '.join(f'{a}: {(frame.arm == a).sum()}' for a in arms)})")
    if "epoch" in frame.columns and len(frame):
        print(f"read at epoch  : {int(frame.epoch.min())}"
              + (f"-{int(frame.epoch.max())}" if frame.epoch.nunique() > 1 else ""))

    summary_rows = []
    for metric in METRICS:
        res = compare(frame, metric)
        if not res:
            continue
        c, f = res["cell"], res["fold"]
        print(f"\n{res['label']}")
        print(f"  vanilla {res['vanilla']:.4f}   gradft {res['gradft']:.4f}")
        print(f"  delta   {f['mean']:+.4f} +- {f['se']:.4f}   "
              f"p_pep {f['p_t']:.2g} (n={f['n']} folds, seeds averaged)")
        print(f"          {c['mean']:+.4f} +- {c['se']:.4f}   "
              f"p     {c['p_t']:.2g} (n={c['n']} cells)")
        summary_rows.append({
            "metric": metric, "description": res["label"],
            "vanilla": res["vanilla"], "gradft": res["gradft"],
            "delta": f["mean"], "se": f["se"], "n_folds": f["n"],
            "p_pep_t": f["p_t"], "p_pep_wilcoxon": f["p_wilcoxon"],
            "delta_per_cell": c["mean"], "se_per_cell": c["se"], "n_cells": c["n"],
            "p_t": c["p_t"], "p_wilcoxon": c["p_wilcoxon"]})

    if {"gradft", "vanilla"} <= set(frame.arm.unique()):
        print("\nby flip rate (seen peptides, ROC-AUC@FPR<=0.1)")
        print(f"  {'flipped':>8}  {'vanilla':>8}  {'gradft':>8}  {'delta':>8}")
        by = frame.pivot_table(index="flip", columns="arm", values="test_auc_01")
        for flip, row in by.iterrows():
            print(f"  {flip:>7}%  {row.vanilla:>8.4f}  {row.gradft:>8.4f}  "
                  f"{row.gradft - row.vanilla:>+8.4f}")

    if summary_rows:
        pd.DataFrame(summary_rows).to_csv(outdir / "results_summary.csv", index=False)
    print("\nwrote")
    print(f"  {outdir / 'results_runs.csv'}      one row per run")
    print(f"  {outdir / 'results_summary.csv'}   the comparison above")
    print("=" * width)
