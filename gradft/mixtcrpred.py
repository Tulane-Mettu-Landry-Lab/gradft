#!/usr/bin/env python
"""Adapter to the official MixTCRpred, imported as a dependency and never modified."""

from __future__ import annotations

import logging
import subprocess
import sys
from configparser import ConfigParser
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

MIXTCRPRED_URL = "https://github.com/GfellerLab/MixTCRpred"
COLUMNS = ("epitope", "cdr3_TRA", "cdr3_TRB", "TRAV", "TRAJ", "TRBV", "TRBJ", "tp")



def ensure_mixtcrpred(path: Path, url: str = MIXTCRPRED_URL,
                      ref: Optional[str] = None) -> Path:
    path = Path(path)
    if (path / "src" / "models.py").is_file():
        return path
    if path.exists() and any(path.iterdir()):
        raise FileNotFoundError(
            f"{path} exists but holds no src/models.py; remove it or point --mixtcrpred-dir elsewhere")
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["git", "clone", "--depth", "1"] + (["--branch", ref] if ref else []) + [url, str(path)]
    logger.info("cloning MixTCRpred: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)
    if not (path / "src" / "models.py").is_file():
        raise FileNotFoundError(f"clone of {url} did not produce src/models.py under {path}")
    return path


def add_to_path(mixtcrpred_dir: Path) -> None:
    root = str(Path(mixtcrpred_dir).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def read_config(path: Path) -> Tuple[ConfigParser, List[int]]:
    cfg = ConfigParser()
    if not cfg.read(path):
        raise FileNotFoundError(f"could not read MixTCRpred config at {path}")
    t = cfg["transformer"]
    padding = [t.getint("padding_epitope"), t.getint("padding_cdr3_TRA"),
               t.getint("padding_cdr3_TRB"), t.getint("padding_cdr12")]
    return cfg, padding



def _epitope_class():
    import torch
    from torch import nn

    import src.models

    class TransformerPredictorABEpitope(src.models.TransformerPredictor_AB_cdr123):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            emb, pad = self.embedding_dim, self.padding
            num_labels = self.output_net[-1].out_features
            self.out_dim = int(np.sum(pad[0:3]) * emb + 4 * pad[-1] * emb)
            self.output_net = nn.Sequential(
                nn.Linear(self.out_dim, emb), nn.BatchNorm1d(emb), nn.ReLU(inplace=True),
                nn.Linear(emb, num_labels))
            self.save_hyperparameters()

        def forward(self, inp_data, mask=False):
            if mask:
                m = [(t != self.padding_idx).unsqueeze(1).unsqueeze(2) for t in inp_data[:7]]
                mask_epi, mask_a3, mask_b3, mask_a1, mask_a2, mask_b1, mask_b2 = m
            else:
                mask_epi = mask_a3 = mask_b3 = None
                mask_a1 = mask_a2 = mask_b1 = mask_b2 = None

            x_TRA = torch.concat([inp_data[3], inp_data[4], inp_data[1]], dim=1)
            mask_TRA = torch.concat([mask_a1, mask_a2, mask_a3], dim=3) if mask else None
            x_TRA = (self.embedding_tok(x_TRA) * self.scale.to(inp_data[1].device)
                     ) + self.embedding_pos_TRA(x_TRA)
            x_TRA_out = self.transformer_encoder(x_TRA, mask=mask_TRA)

            x_TRB = torch.concat([inp_data[5], inp_data[6], inp_data[2]], dim=1)
            mask_TRB = torch.concat([mask_b1, mask_b2, mask_b3], dim=3) if mask else None
            x_TRB = (self.embedding_tok(x_TRB) * self.scale.to(inp_data[2].device)
                     ) + self.embedding_pos_TRB(x_TRB)
            x_TRB_out = self.transformer_encoder(x_TRB, mask=mask_TRB)

            x_epi = inp_data[0]
            x_epi = (self.embedding_tok(x_epi) * self.scale.to(x_epi.device)
                     ) + self.embedding_pos_epi(x_epi)
            x_epi_out = self.transformer_encoder(x_epi, mask=mask_epi)

            x = torch.concat([x_epi_out, x_TRA_out, x_TRB_out], dim=1)
            return self.output_net(x.flatten(start_dim=1))

    return TransformerPredictorABEpitope


def build_model(cfg: ConfigParser, padding: List[int], *, with_epitope: bool = True):
    import src.models
    import src.utils

    t = cfg["transformer"]
    cls = _epitope_class() if with_epitope else src.models.TransformerPredictor_AB_cdr123
    return cls(vocab_size=len(src.utils.valid_aa), embedding_dim=t.getint("embedding_dim"),
               hidden_dim=t.getint("hidden_dim"), num_heads=t.getint("num_heads"),
               num_layers=t.getint("num_layers"), dropout=t.getfloat("dropout"),
               lr=t.getfloat("lr"), num_labels=t.getint("num_labels"), padding=padding,
               warmup=t.getint("warmup"), max_iters=t.getint("max_iters"))


def hook_layer(model):
    return model.output_net, "output_net input = concat(epi, TRA, TRB).flatten()"



def make_dataset(df, padding: List[int]):
    import src.dataloaders

    ds = src.dataloaders.db_transformer(df.copy(), padding=padding, istrain=True)
    kept = ds.data.index.to_numpy()
    if len(kept) < len(df):
        logger.info("db_transformer dropped %d of %d rows", len(df) - len(kept), len(df))
    return ds, kept


def collate(items):
    import torch

    seqs = [i[0] for i in items]
    tens = [torch.stack([i[1][k] for i in items]) for k in range(len(items[0][1]))]
    y = torch.tensor([int(i[2]) for i in items])
    return seqs, tens, y


def to_device(tensors, device):
    return [t.to(device) for t in tensors]


def scores(y: np.ndarray, p: np.ndarray) -> Tuple[float, float]:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y)) < 2:
        return float("nan"), float("nan")
    return (float(roc_auc_score(y, p)), float(roc_auc_score(y, p, max_fpr=0.1)))


def evaluate(model, loader, device) -> Tuple[float, np.ndarray, np.ndarray]:
    import torch
    import torch.nn.functional as F

    model.eval()
    probs, ys, total, n = [], [], 0.0, 0
    with torch.no_grad():
        for _, tens, y in loader:
            tens, y = to_device(tens, device), y.to(device)
            out = model(tens, mask=True)
            total += float(F.cross_entropy(out, y, reduction="sum"))
            n += len(y)
            probs.append(torch.softmax(out, 1)[:, 1].cpu().numpy())
            ys.append(y.cpu().numpy())
    return (total / max(n, 1),
            np.concatenate(ys) if ys else np.zeros(0),
            np.concatenate(probs) if probs else np.zeros(0))
