#!/usr/bin/env python
"""Leave-one-peptide-out splits of TCRvdb with controlled label noise."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

FEATURES = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]
PEPTIDES = ["GILGFVFTL", "GLCTLVAML", "NLVPMVATV", "TTDPSFLGRY", "YLQPRTFLL"]
ALLELE_GROUPS = ["HLA-A*01", "HLA-A*02"]


def allele_group(allele: str) -> str:
    a = str(allele)
    for g in ALLELE_GROUPS:
        if a.startswith(g):
            return g
    return a



def load_tcrvdb(table: Path, raw: Optional[Path] = None, *,
                require_raw: bool = True) -> pd.DataFrame:
    df = pd.read_csv(table)
    missing = [c for c in FEATURES + ["binder", "allele"] if c not in df.columns]
    if missing:
        raise ValueError(f"{table} is missing column(s) {missing}")

    if "ambiguous" in df.columns:
        before = len(df)
        df = df[df.ambiguous == 0].reset_index(drop=True)
        logger.info("ambiguous=0 filter: %d -> %d rows (from the table's own column)", before, len(df))
        return df

    if raw is None or not Path(raw).is_file():
        if require_raw:
            raise FileNotFoundError(
                f"{table} has no 'ambiguous' column and the raw TCRvdb file "
                f"({raw}) was not found, so the ambiguous=0 requirement cannot be "
                "verified. Pass --raw-tcrvdb, or --no-require-raw to accept the table as given.")
        logger.warning("cannot verify ambiguous=0: no raw TCRvdb file. Proceeding on trust.")
        return df

    if "tcr_id" not in df.columns:
        raise ValueError(f"{table} needs a 'tcr_id' column to be matched against {raw}")
    rawdf = pd.read_csv(raw)
    for c in ("tcr_id", "ambiguous"):
        if c not in rawdf.columns:
            raise ValueError(f"{raw} is missing column {c!r}")
    ok = set(rawdf.loc[rawdf.ambiguous == 0, "tcr_id"])
    known = set(rawdf.tcr_id)
    unknown = set(df.tcr_id) - known
    if unknown:
        raise ValueError(
            f"{len(unknown)} tcr_id(s) in {table} do not appear in {raw}, so their "
            f"ambiguous status is unknown (e.g. {sorted(unknown)[:3]})")
    before = len(df)
    df = df[df.tcr_id.isin(ok)].reset_index(drop=True)
    logger.info("ambiguous=0 filter against %s: %d -> %d rows (%d dropped)",
                Path(raw).name, before, len(df), before - len(df))
    return df



def _clonotype(df: pd.DataFrame) -> pd.Series:
    return df.A3.astype(str) + "|" + df.B3.astype(str)


def _take_per_peptide(df: pd.DataFrame, frac: float, seed: int) -> pd.Index:
    rng = np.random.default_rng(seed)
    picked: List[np.ndarray] = []
    clone = _clonotype(df)
    for (_pep, _binder), block in df.groupby([df.peptide, df.binder], sort=True):
        want = int(round(frac * len(block)))
        if want <= 0:
            continue
        groups = [np.asarray(idx) for _, idx in
                  block.groupby(clone.loc[block.index], sort=True).groups.items()]
        order = rng.permutation(len(groups))
        taken: List[int] = []
        for gi in order:
            if len(taken) >= want:
                break
            taken.extend(groups[gi].tolist())
        picked.append(np.asarray(taken, dtype=int))
    return pd.Index(np.concatenate(picked)) if picked else pd.Index([], dtype=int)


def make_split(df: pd.DataFrame, held_out: str, *, test_frac: float = 0.10,
               val_frac: float = 0.10, seed: int = 0) -> pd.DataFrame:
    if held_out not in set(df.peptide):
        raise ValueError(f"peptide {held_out!r} is not in the table; have {sorted(set(df.peptide))}")
    out = df.copy().reset_index(drop=True)
    out["split"] = "train"
    out.loc[out.peptide == held_out, "split"] = "unseen"

    rest = out[out.split == "train"]
    test_idx = _take_per_peptide(rest, test_frac, seed)
    out.loc[test_idx, "split"] = "test"

    if val_frac > 0:
        rest2 = out[out.split == "train"]
        val_idx = _take_per_peptide(rest2, val_frac, seed + 1)
        out.loc[val_idx, "split"] = "val"

    rank = {"test": 0, "val": 1, "train": 2}
    inv = {v: k for k, v in rank.items()}
    seen = out.split != "unseen"
    sub = out[seen]
    strictest = sub.assign(_c=_clonotype(sub), _r=sub.split.map(rank)).groupby("_c")._r.transform("min")
    out.loc[seen, "split"] = strictest.map(inv)
    return out



def _spread(block: pd.DataFrame, n: int, seed: int) -> pd.Index:
    rng = np.random.default_rng(seed)
    by_pep = {p: rng.permutation(g.index.to_numpy()).tolist() for p, g in block.groupby("peptide", sort=True)}
    out: List[int] = []
    while len(out) < n and any(by_pep.values()):
        for p in sorted(by_pep):
            if by_pep[p] and len(out) < n:
                out.append(by_pep[p].pop())
    return pd.Index(out, dtype=int)


def choose_anchors(train: pd.DataFrame, per_class_per_group: int = 25, seed: int = 0) -> pd.Index:
    group = train.allele.map(allele_group)
    picked: List[pd.Index] = []
    for g in sorted(set(group)):
        for binder in (1, 0):
            block = train[(group == g) & (train.binder == binder)]
            if block.empty:
                logger.warning("no %s rows of class %d available as anchors", g, binder)
                continue
            idx = _spread(block, per_class_per_group, seed)
            if len(idx) < per_class_per_group:
                logger.warning("only %d of %d %s class-%d anchors available",
                               len(idx), per_class_per_group, g, binder)
            picked.append(idx)
    out = pd.Index(np.concatenate([i.to_numpy() for i in picked])) if picked else pd.Index([], dtype=int)
    logger.info("anchors: %d rows over peptides %s", len(out), sorted(train.loc[out].peptide.unique()))
    return out



def flip_labels(split: pd.DataFrame, frac: float, seed: int = 0) -> pd.DataFrame:
    out = split.copy()
    if "anchor" not in out.columns:
        raise ValueError("choose the anchors before flipping: no 'anchor' column")
    out["flipped"] = 0
    if frac <= 0:
        return out
    flippable = out[(out.split == "train") & (out.anchor == 0)]
    rng = np.random.default_rng(seed + int(round(frac * 1000)))
    chosen: List[int] = []
    for _pep, grp in flippable.groupby("peptide", sort=True):
        n = int(round(frac * len(grp)))
        if n:
            chosen.extend(rng.choice(grp.index.to_numpy(), n, replace=False).tolist())
    if chosen:
        out.loc[chosen, "binder"] = 1 - out.loc[chosen, "binder"]
        out.loc[chosen, "flipped"] = 1
    return out


def build_table(df: pd.DataFrame, held_out: str, frac: float, *, test_frac: float = 0.10,
                val_frac: float = 0.10, anchors_per_class_per_group: int = 25,
                seed: int = 0) -> pd.DataFrame:
    out = make_split(df, held_out, test_frac=test_frac, val_frac=val_frac, seed=seed)
    out["anchor"] = 0
    anchors = choose_anchors(out[out.split == "train"], anchors_per_class_per_group, seed)
    out.loc[anchors, "anchor"] = 1
    out = flip_labels(out, frac, seed)
    bad = out[(out.anchor == 1) & (out.flipped == 1)]
    if len(bad):
        raise AssertionError(f"{len(bad)} anchor rows were flipped")
    return out
