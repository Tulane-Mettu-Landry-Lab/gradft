#!/usr/bin/env python
"""Adapter to the official NetTCR-2.2, imported as a dependency and never modified."""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

NETTCR_URL = "https://github.com/mnielLab/NetTCR-2.2"
MAX_LEN: Dict[str, int] = {"peptide": 12, "A1": 7, "A2": 8, "A3": 22, "B1": 6, "B2": 7, "B3": 23}
FEATURES = ["peptide", "A1", "A2", "A3", "B1", "B2", "B3"]
INPUT_ORDER = ["pep", "a1", "a2", "a3", "b1", "b2", "b3"]
_FEATURE_FOR_INPUT = dict(zip(INPUT_ORDER, FEATURES))
STANDARD_AA = frozenset("ACDEFGHIKLMNPQRSTVWY")



def ensure_nettcr(path: Path, url: str = NETTCR_URL, ref: Optional[str] = None) -> Path:
    path = Path(path)
    if (path / "src" / "nettcr_archs.py").is_file():
        return path
    if path.exists() and any(path.iterdir()):
        raise FileNotFoundError(
            f"{path} exists but holds no src/nettcr_archs.py; remove it or point --nettcr-dir elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else []) + [url, str(path)]
    logger.info("cloning NetTCR-2.2: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not (path / "src" / "nettcr_archs.py").is_file():
        raise FileNotFoundError(f"clone of {url} did not produce src/nettcr_archs.py under {path}")
    return path


def add_to_path(nettcr_dir: Path) -> None:
    src = str(Path(nettcr_dir).resolve() / "src")
    if src not in sys.path:
        sys.path.insert(0, src)



def check_encodable(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    bad_aa = pd.Series(False, index=df.index)
    too_long = pd.Series(False, index=df.index)
    for feat in FEATURES:
        s = df[feat].astype(str)
        bad_aa |= s.map(lambda v: bool(set(v) - STANDARD_AA))
        too_long |= s.str.len() > MAX_LEN[feat]
    keep = ~(bad_aa | too_long)
    report = {"non_standard_aa": int(bad_aa.sum()), "too_long": int(too_long.sum()),
              "kept": int(keep.sum()), "dropped": int((~keep).sum())}
    if report["dropped"]:
        logger.warning("dropped %d unencodable row(s): %s", report["dropped"], report)
    return df[keep].reset_index(drop=True), report


def encode_batch(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    import keras_utils

    return {inp: np.float32(keras_utils.enc_list_bl_max_len(
                df[feat].tolist(), keras_utils.blosum50_20aa, MAX_LEN[feat]) / 5)
            for inp, feat in _FEATURE_FOR_INPUT.items()}



def build_model(dropout_rate: float, seed: int):
    from nettcr_archs import CNN_CDR123_global_max

    return CNN_CDR123_global_max(dropout_rate, seed)


def hook_layer(model):
    from tensorflow import keras

    cats = [l for l in model.layers if isinstance(l, keras.layers.Concatenate)]
    if not cats:
        raise RuntimeError("no Concatenate layer in the model; upstream architecture changed?")
    return cats[-1]



def sample_weights(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    counts = train_df.peptide.value_counts()
    weight = np.log2(train_df.shape[0] / counts) / np.log2(len(train_df.peptide.unique()))
    weight = weight * (train_df.shape[0] / np.sum(weight * counts))
    return (train_df.peptide.map(weight).to_numpy(np.float32),
            np.nan_to_num(val_df.peptide.map(weight).to_numpy(np.float32)))


def keras_auc_01(y: np.ndarray, p: np.ndarray, batch_size: int) -> float:
    from sklearn.metrics import roc_auc_score

    vals = []
    for s in range(0, len(y), batch_size):
        try:
            vals.append(roc_auc_score(y[s:s + batch_size], p[s:s + batch_size], max_fpr=0.1))
        except ValueError:
            vals.append(0.0)
    return float(np.mean(vals)) if vals else float("nan")


def scores(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (float(roc_auc_score(y, p)), float(roc_auc_score(y, p, max_fpr=0.1)))


def bce(y: np.ndarray, p: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    p = np.clip(np.asarray(p, np.float64), 1e-7, 1 - 1e-7)
    ll = -(y * np.log(p) + (1 - y) * np.log(1 - p))
    if w is None:
        return float(ll.mean())
    w = np.asarray(w, np.float64)
    return float((ll * w).sum() / max(w.sum(), 1e-12))


def quiet_tensorflow() -> None:
    os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
    os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
    os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")


def configure_gpu(memory_limit_mb: Optional[int] = None) -> None:
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        logger.info("no GPU visible; training on CPU")
        return
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            if memory_limit_mb:
                tf.config.set_logical_device_configuration(
                    gpu, [tf.config.LogicalDeviceConfiguration(memory_limit=memory_limit_mb)])
        except RuntimeError as exc:
            logger.warning("could not configure %s (%s); concurrent runs may run out of memory",
                           gpu.name, exc)
    logger.info("GPU memory growth on for %d device(s)%s", len(gpus),
                f", capped at {memory_limit_mb} MB" if memory_limit_mb else "")
