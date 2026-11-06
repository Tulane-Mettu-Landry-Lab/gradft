"""The GradFT rule for PyTorch models."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

GROUP_BY = ("feature", "label")


class GradFTTorch:
    def __init__(self, model, layer, *, num_classes: int, n_train: int,
                 warmup: int = 50, lam: float = 10.0, group_by: str = "feature",
                 flip_top_frac: float = 1.0, exempt_rate: float = 0.0,
                 extra_anchor_samples: int = 0, reselect_epoch: int = 5,
                 extra_pool: Optional[Sequence[int]] = None, seed: int = 0):
        if group_by not in GROUP_BY:
            raise ValueError(f"group_by must be one of {GROUP_BY}, got {group_by!r}")
        if not 0.0 <= exempt_rate < 1.0:
            raise ValueError(f"exempt_rate must be in [0, 1), got {exempt_rate}")
        if not 0.0 < flip_top_frac <= 1.0:
            raise ValueError(f"flip_top_frac must be in (0, 1], got {flip_top_frac}")

        self.model, self.num_classes = model, num_classes
        self.warmup, self.lam, self.group_by = warmup, float(lam), group_by
        self.flip_top_frac, self.exempt_rate = float(flip_top_frac), float(exempt_rate)
        self.extra_anchor_samples = int(extra_anchor_samples)
        self.reselect_epoch = int(reselect_epoch)
        self.extra_pool = None if extra_pool is None else np.asarray(extra_pool, dtype=int)
        self.rng = np.random.default_rng(seed)

        self.epoch, self.active = 0, False
        self.extra = np.zeros(0, dtype=int)
        self.anchor_x = self.anchor_y = None
        self.ref = self.flip = self.centroids = self.has_ref = None
        self.rho = None
        self.flipped_rate_log: list = []
        self.units: Optional[int] = None
        n = int(n_train)
        self.rho_sum, self.rho_cnt = np.zeros(n), np.zeros(n)
        self.win_sum, self.win_cnt = np.zeros(n), np.zeros(n)
        self._h = None
        self._handle = layer.register_forward_pre_hook(self._pre_hook)

    def close(self) -> None:
        self._handle.remove()


    def _pre_hook(self, module, args):
        h = args[0]
        self._h = h
        if self.units is None:
            self.units = int(h.flatten(1).shape[1])
        if self.active and module.training and h.requires_grad:
            group = self._group(h.detach().flatten(1))
            h.register_hook(lambda g: self._reweight(g, group))

    def _group(self, h, labels=None):
        if self.group_by == "label" and labels is not None:
            return labels
        d = torch.cdist(h, self.centroids)
        d = d.masked_fill(~self.has_ref.unsqueeze(0), float("inf"))
        return d.argmin(1)

    def _flipped(self, g, group):
        flip, ref = self.flip[group], self.ref[group]
        nz = g != 0
        flipped = nz & (torch.sign(g) == -ref)
        rho = (flip * flipped).sum(1) / (flip * nz).sum(1).clamp(min=1e-12)
        return flipped, rho

    def _reweight(self, g, group):
        shape = g.shape
        g = g.flatten(1)
        flipped, rho = self._flipped(g, group)
        self.rho = rho.detach()
        w = torch.where(flipped, torch.exp(-self.lam * self.flip[group]), torch.ones_like(g))
        if self.flip_top_frac < 1.0:
            k = max(1, int(np.ceil(self.flip_top_frac * len(rho))))
            thresh = torch.sort(rho, descending=True).values[k - 1]
            w = torch.where((rho >= thresh).unsqueeze(1), w, torch.ones_like(w))
        if self.exempt_rate > 0.0:
            exempt = torch.rand(1, g.shape[1], device=g.device) < self.exempt_rate
            w = torch.where(exempt.expand_as(w), torch.ones_like(w), w)
        return (g * w).view(shape)


    def set_anchors(self, x, y) -> None:
        self.anchor_x = x
        self.anchor_y = torch.as_tensor(np.asarray(y).astype(int))

    def probe(self, forward) -> None:
        was_training = self.model.training
        self.model.eval()
        y = self.anchor_y.to(next(self.model.parameters()).device)
        onehot = F.one_hot(y, self.num_classes).float()
        wrong = (1.0 - onehot) / (self.num_classes - 1)
        out = forward(self.anchor_x)
        h = self._h
        logp = F.log_softmax(out, dim=1)
        g_true, = torch.autograd.grad(-(onehot * logp).sum(), h, retain_graph=True)
        g_wrong, = torch.autograd.grad(-(wrong * logp).sum(), h)
        avg = (onehot / onehot.sum(0).clamp(min=1.0)).T
        agree_orig = avg @ torch.sign(g_true.flatten(1))
        agree_wrong = avg @ torch.sign(g_wrong.flatten(1))
        self.ref = torch.sign(agree_orig).detach()
        self.flip = torch.relu(-agree_orig * agree_wrong).detach()
        self.centroids = (avg @ h.flatten(1)).detach()
        self.has_ref = (onehot.sum(0) > 0).detach()
        if was_training:
            self.model.train()


    def begin_epoch(self, epoch: int, n_train: int) -> np.ndarray:
        self.epoch, self.active = epoch, epoch > self.warmup
        rows = np.arange(n_train)
        return np.setdiff1d(rows, self.extra) if self.active else rows

    def record(self, idx: np.ndarray) -> None:
        if self.rho is None:
            return
        r = self.rho.detach().cpu().numpy()
        ok = np.isfinite(r)
        rows = np.asarray(idx)[ok]
        for s, c in ((self.rho_sum, self.rho_cnt), (self.win_sum, self.win_cnt)):
            np.add.at(s, rows, r[ok])
            np.add.at(c, rows, 1)
        self.flipped_rate_log.append(float(r[ok].mean()) if ok.any() else float("nan"))

    def end_epoch(self, epoch: int, y_train: np.ndarray):
        if not (self.active and self.extra_anchor_samples > 0
                and (epoch - self.warmup) % self.reselect_epoch == 0):
            return None
        score = np.where(self.win_cnt > 0, self.win_sum / np.maximum(self.win_cnt, 1), np.inf)
        eligible = np.ones(len(y_train), bool)
        if self.extra_pool is not None:
            eligible[:] = False
            eligible[self.extra_pool] = True
        new = []
        for c in range(self.num_classes):
            rows = np.flatnonzero((y_train == c) & np.isfinite(score) & eligible)
            new.append(rows[np.argsort(score[rows], kind="stable")[: self.extra_anchor_samples]])
        new = np.concatenate(new).astype(int) if new else np.zeros(0, int)
        returned, self.extra = self.extra, new
        self.win_sum[:], self.win_cnt[:] = 0, 0
        return returned, new

    def train_scores(self) -> np.ndarray:
        return np.where(self.rho_cnt > 0, self.rho_sum / np.maximum(self.rho_cnt, 1), np.nan)

    def epoch_flipped_rate(self) -> float:
        v = [r for r in self.flipped_rate_log if np.isfinite(r)]
        self.flipped_rate_log = []
        return float(np.mean(v)) if v else float("nan")
