#!/usr/bin/env python
"""Raw TCRvdb onto NetTCR-2.2's schema: python prepare_tcrvdb.py [--input TCRvdb.csv]"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import pandas as pd

from gradft.nettcr import FEATURES, MAX_LEN, STANDARD_AA

logger = logging.getLogger("prepare_tcrvdb")

SOURCE = HERE / "datasets" / "source"

CDR_OVERRIDES: Dict[str, Tuple[Optional[str], Optional[str]]] = {
    "TRAV12-1": (None, "VYSSG"),
    "TRAV40": (None, "ETMEN"),
    "TRBV6-8": (None, "SAAAG"),
}

RAW_COLUMNS = ("tcr_id", "pos", "ambiguous", "cdr3_alpha_aa", "cdr3_beta_aa",
               "TRAV_IMGT", "TRBV_IMGT", "TRAV_vdjdb", "TRBV_vdjdb",
               "epitope_aa", "hla_long")


class VGeneCDR:
    def __init__(self, species: str = "homosapiens"):
        self.species = species
        self._cache: Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        self.unresolved: Dict[str, int] = {}
        try:
            import tidytcells
        except ImportError as exc:
            raise ImportError(
                "tidytcells is required to derive CDR1/CDR2 from V gene names. "
                "Install it with: pip install tidytcells") from exc

    @staticmethod
    def _base(symbol: str) -> str:
        return symbol.split("*")[0]

    def _resolve(self, gene: str) -> Tuple[Optional[str], Optional[str]]:
        import tidytcells as tt

        tt_log = logging.getLogger("tidytcells")
        prev = tt_log.level
        tt_log.setLevel(logging.ERROR)
        try:
            candidates = [gene]
            if "," in gene:
                candidates.insert(0, gene.split(",")[0].strip())
            symbol = None
            for cand in candidates:
                for precision in ("allele", "gene"):
                    try:
                        std = tt.tr.standardize(symbol=cand, species=self.species,
                                                precision=precision)
                    except Exception:
                        std = None
                    if std:
                        symbol = std if "*" in std else f"{std}*01"
                        break
                if symbol:
                    break
            if symbol is None:
                return (None, None)

            seqs = None
            for sym in (symbol, f"{self._base(symbol)}*01"):
                try:
                    seqs = tt.tr.get_aa_sequence(symbol=sym, species=self.species)
                except Exception:
                    seqs = None
                if seqs:
                    symbol = sym
                    break
            if not seqs:
                return (None, None)
            ov1, ov2 = CDR_OVERRIDES.get(self._base(symbol), (None, None))
            return (ov1 or seqs.get("CDR1-IMGT"), ov2 or seqs.get("CDR2-IMGT"))
        finally:
            tt_log.setLevel(prev)

    def __call__(self, gene: object) -> Tuple[Optional[str], Optional[str]]:
        if not isinstance(gene, str) or not gene.strip():
            return (None, None)
        key = gene.strip()
        if key not in self._cache:
            self._cache[key] = self._resolve(key)
        got = self._cache[key]
        if got[0] is None or got[1] is None:
            self.unresolved[key] = self.unresolved.get(key, 0) + 1
        return got


def strip_cdr3_anchors(seq: object) -> Optional[str]:
    if not isinstance(seq, str):
        return None
    s = seq.strip().upper()
    if len(s) < 3:
        return s or None
    if s[0] == "C" and s[-1] in ("F", "W"):
        return s[1:-1]
    return s


def normalise_allele(value: object, fields: int = 2) -> str:
    if value is None:
        return "unknown"
    s = re.sub(r"\s+", "", str(value))
    if not s or s.lower() in {"nan", "none", "na"}:
        return "unknown"
    body = (s[4:] if s.upper().startswith("HLA-") else s).upper()
    parts = body.split(":")
    if len(parts) > fields:
        body = ":".join(parts[:fields])
    return f"HLA-{body}"


def sanitise(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    work = df.copy()
    report: Dict[str, object] = {"rows_in": int(len(work)), "dropped": {}}
    for col in FEATURES:
        work[col] = work[col].apply(lambda v: v.strip().upper() if isinstance(v, str) else v)
    keep = pd.Series(True, index=work.index)
    for col in FEATURES:
        missing = work[col].isna() | (work[col].astype(str).str.len() == 0)
        bad_aa = ~missing & ~work[col].fillna("").map(lambda s: bool(s) and set(s) <= STANDARD_AA)
        too_long = ~missing & (work[col].fillna("").str.len() > MAX_LEN[col])
        counts = {"missing": int(missing.sum()), "non_standard_aa": int(bad_aa.sum()),
                  f"longer_than_{MAX_LEN[col]}": int(too_long.sum())}
        if any(counts.values()):
            report["dropped"][col] = {k: v for k, v in counts.items() if v}
        keep &= ~(missing | bad_aa | too_long)
    clean = work.loc[keep].reset_index(drop=True)
    report["rows_out"] = int(len(clean))
    report["rows_dropped_total"] = int(len(work) - len(clean))
    return clean, report


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("-i", "--input", type=Path, default=SOURCE / "TCRvdb.csv",
                   help="The raw TCRvdb release (default: %(default)s).")
    p.add_argument("-o", "--out", type=Path, default=SOURCE / "tcrvdb_train.csv",
                   help="Harmonised table for make_datasets.py (default: %(default)s).")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.input.is_file():
        raise FileNotFoundError(
            f"{args.input} not found. Download the TCRvdb release and save it there; "
            "the README says which columns it must carry.")
    raw = pd.read_csv(args.input)
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise ValueError(f"{args.input} is missing column(s) {missing}; this is not the "
                         "raw TCRvdb release the README describes")

    kept = raw[raw.ambiguous == 0].reset_index(drop=True)
    logger.info("%d rows -> %d with ambiguous=0 (pos=%d, neg=%d)", len(raw), len(kept),
                int(kept.pos.sum()), int((kept.pos == 0).sum()))

    lookup = VGeneCDR()

    def cdr12(imgt, vdjdb):
        got = lookup(imgt)
        return got if None not in got else lookup(vdjdb)

    a = [cdr12(i, v) for i, v in zip(kept.TRAV_IMGT, kept.TRAV_vdjdb)]
    b = [cdr12(i, v) for i, v in zip(kept.TRBV_IMGT, kept.TRBV_vdjdb)]
    rows = pd.DataFrame({
        "tcr_id": kept.tcr_id, "peptide": kept.epitope_aa,
        "A1": [x[0] for x in a], "A2": [x[1] for x in a],
        "A3": kept.cdr3_alpha_aa.map(strip_cdr3_anchors),
        "B1": [x[0] for x in b], "B2": [x[1] for x in b],
        "B3": kept.cdr3_beta_aa.map(strip_cdr3_anchors),
        "binder": kept.pos.to_numpy(int),
        "allele": kept.hla_long.map(normalise_allele),
    })
    if lookup.unresolved:
        logger.warning("%d V gene name(s) had no CDR1/CDR2 in the IMGT reference: %s",
                       len(lookup.unresolved),
                       ", ".join(sorted(lookup.unresolved)[:5]))

    clean, report = sanitise(rows)
    logger.info("encodable by NetTCR-2.2: %d of %d; dropped %s",
                report["rows_out"], report["rows_in"], report["dropped"] or "nothing")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    clean[["tcr_id"] + FEATURES + ["binder", "allele"]].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}: {len(clean)} rows, {int(clean.binder.sum())} binders, "
          f"{clean.peptide.nunique()} peptides, {clean.allele.nunique()} alleles")
    print(pd.crosstab(clean.peptide, clean.binder)
          .rename(columns={0: "neg", 1: "pos"}).to_string())
    print(pd.crosstab(clean.peptide, clean.allele).to_string())
    print("\nnext: python make_datasets.py && python make_mix_datasets.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
