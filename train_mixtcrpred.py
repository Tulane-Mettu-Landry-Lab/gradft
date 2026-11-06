#!/usr/bin/env python
"""One MixTCRpred run: python train_mixtcrpred.py --table T --outdir D --config C"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.dont_write_bytecode = True

import numpy as np
import pandas as pd

from gradft import mixtcrpred
from gradft.config import load_mix_config

logger = logging.getLogger("train_mixtcrpred")

SPLITS = ("train", "val", "test", "unseen")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, required=True,
                   help="mix_lopo_<PEPTIDE>_f<PCT>.csv from make_mix_datasets.py")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="YAML or JSON; defaults apply if omitted.")
    p.add_argument("--mixtcrpred-dir", type=Path, default=HERE / "third_party" / "MixTCRpred",
                   help="Where to find (or clone) the official MixTCRpred (default: %(default)s).")
    p.add_argument("--mixtcrpred-url", default=mixtcrpred.MIXTCRPRED_URL)
    p.add_argument("--model-config", type=Path, default=None,
                   help="Upstream src/config.ini (default: the one in --mixtcrpred-dir).")
    p.add_argument("--device", default="cuda")
    p.add_argument("--skip-existing", action="store_true", help="Do nothing if history.csv exists.")

    o = p.add_argument_group("config overrides (omitted values come from --config)")
    o.add_argument("--epochs", type=int)
    o.add_argument("--patience", type=int)
    o.add_argument("--batch-size", type=int)
    o.add_argument("--seed", type=int)
    o.add_argument("--warmup", type=int, help="Epochs of plain training before gradft engages.")
    o.add_argument("--exempt-rate", type=float, help="Share of hidden units exempted each step.")
    o.add_argument("--lam", type=float)
    o.add_argument("--group-by", choices=["feature", "label"])
    o.add_argument("--flip-top-frac", type=float)
    o.add_argument("--extra-anchor-samples", type=int)
    o.add_argument("--reselect-epoch", type=int)
    o.add_argument("--select", choices=["val_loss", "val_auc", "last"])
    o.add_argument("--enabled", dest="enabled", action="store_true", default=None,
                   help="Force gradft on.")
    o.add_argument("--disabled", dest="enabled", action="store_false",
                   help="Force gradft off (the vanilla control).")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.skip_existing and (args.outdir / "history.csv").is_file():
        logger.info("%s exists, nothing to do", args.outdir / "history.csv")
        return 0

    repo = mixtcrpred.ensure_mixtcrpred(args.mixtcrpred_dir, args.mixtcrpred_url)
    mixtcrpred.add_to_path(repo)
    cfg = load_mix_config(args.config, **{k: getattr(args, k) for k in (
        "epochs", "patience", "batch_size", "seed", "warmup", "exempt_rate", "lam",
        "group_by", "flip_top_frac", "extra_anchor_samples", "reselect_epoch",
        "select", "enabled")})
    cfg.save(args.outdir / "config.resolved.json")

    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Subset

    import src.models

    from gradft.reweight_torch import GradFTTorch

    torch.manual_seed(cfg.seed); np.random.seed(cfg.seed); random.seed(cfg.seed)
    device = args.device if torch.cuda.is_available() else "cpu"
    model_cfg, padding = mixtcrpred.read_config(
        args.model_config or repo / "src" / "config.ini")
    t = model_cfg["transformer"]

    df = pd.read_csv(args.table)
    for col in ("split", "anchor", "flipped", "tp"):
        if col not in df.columns:
            raise ValueError(f"{args.table} has no {col!r} column; build it with make_mix_datasets.py")
    ds, kept = mixtcrpred.make_dataset(df, padding)
    meta = df.loc[kept].reset_index(drop=True)
    pos = {s: np.flatnonzero((meta.split == s).to_numpy()) for s in SPLITS}
    anchor_pos = np.flatnonzero((meta.anchor == 1).to_numpy())
    y_all = meta.tp.to_numpy(int)
    if not len(anchor_pos) and cfg.enabled:
        raise ValueError(f"{args.table} marks no anchor rows, but gradft needs them")
    if int(meta.loc[anchor_pos, "flipped"].sum()):
        raise AssertionError("anchors must carry true labels, but some are flipped")

    def loader(idx, shuffle=False):
        return DataLoader(Subset(ds, list(idx)), batch_size=cfg.batch_size, shuffle=shuffle,
                          collate_fn=mixtcrpred.collate, num_workers=0)

    val_dl, test_dl, unseen_dl = (loader(pos[s]) for s in ("val", "test", "unseen"))
    model = mixtcrpred.build_model(model_cfg, padding, with_epitope=cfg.with_epitope).to(device)
    hooked, hook_label = mixtcrpred.hook_layer(model)
    opt = torch.optim.Adam(model.parameters(), lr=t.getfloat("lr"))
    sched = src.models.CosineWarmupScheduler(opt, warmup=t.getint("warmup"),
                                             max_iters=t.getint("max_iters"))

    train_pos = pos["train"]
    local = {g: i for i, g in enumerate(train_pos)}
    y_train = y_all[train_pos]
    rw = GradFTTorch(
        model, hooked, num_classes=t.getint("num_labels"), n_train=len(train_pos),
        warmup=cfg.epochs + 1 if not cfg.enabled else cfg.warmup,
        lam=cfg.lam, group_by=cfg.group_by, flip_top_frac=cfg.flip_top_frac,
        exempt_rate=cfg.exempt_rate,
        extra_anchor_samples=cfg.extra_anchor_samples if cfg.enabled else 0,
        reselect_epoch=cfg.reselect_epoch, seed=cfg.seed)
    _, anchor_tens, anchor_y = mixtcrpred.collate([ds[i] for i in anchor_pos])
    rw.set_anchors(mixtcrpred.to_device(anchor_tens, device), anchor_y.numpy())
    anchor_local = np.array([local[g] for g in anchor_pos if g in local], dtype=int)

    ckpt = args.outdir / "checkpoint"; ckpt.mkdir(parents=True, exist_ok=True)
    best_pt = ckpt / "best.pt"
    rng = np.random.default_rng(cfg.seed)
    history = []
    better = (lambda a, b: a > b) if cfg.select == "val_auc" else (lambda a, b: a < b)
    best = -np.inf if cfg.select == "val_auc" else np.inf
    best_epoch, wait = 0, 0

    logger.info("%s | train %d (flipped %d, anchors %d) val %d | test %d unseen %d (%s) "
                "| gradft=%s hook=%s | device %s",
                args.table.name, len(train_pos), int(meta.loc[train_pos, "flipped"].sum()),
                len(anchor_pos), len(pos["val"]), len(pos["test"]), len(pos["unseen"]),
                meta.loc[pos["unseen"], "peptide"].iloc[0] if len(pos["unseen"]) else "-",
                cfg.enabled, hook_label, device)

    def batches(rows):
        for s in range(0, len(rows), cfg.batch_size):
            sl = rows[s:s + cfg.batch_size]
            if len(sl) > 1:
                yield sl

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        pool = rw.begin_epoch(epoch, len(train_pos))
        order = pool[rng.permutation(len(pool))]
        model.train()
        losses = []
        for sl in batches(order):
            _, tens, y = mixtcrpred.collate([ds[int(train_pos[i])] for i in sl])
            tens, y = mixtcrpred.to_device(tens, device), y.to(device)
            if rw.active:
                rw.probe(lambda x: model(x, mask=True))
            opt.zero_grad()
            loss = F.cross_entropy(model(tens, mask=True), y)
            loss.backward()
            opt.step(); sched.step()
            losses.append(loss.detach().item())
            if rw.active:
                rw.record(sl)

        if cfg.anchor_batch_last and len(anchor_local) > 1:
            was, rw.active = rw.active, False
            model.train()
            for sl in batches(anchor_local):
                _, tens, y = mixtcrpred.collate([ds[int(train_pos[i])] for i in sl])
                opt.zero_grad()
                F.cross_entropy(model(mixtcrpred.to_device(tens, device), mask=True),
                                y.to(device)).backward()
                opt.step(); sched.step()
            rw.active = was
        rw.end_epoch(epoch, y_train)

        vl, vy, vp = mixtcrpred.evaluate(model, val_dl, device)
        tl, ty, tp_ = mixtcrpred.evaluate(model, test_dl, device)
        ul, uy, up = mixtcrpred.evaluate(model, unseen_dl, device)
        val_auc, val_auc01 = mixtcrpred.scores(vy, vp)
        test_auc, test_auc01 = mixtcrpred.scores(ty, tp_)
        unseen_auc, unseen_auc01 = mixtcrpred.scores(uy, up)
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
               "val_loss": vl, "val_auc": val_auc, "val_auc_01": val_auc01,
               "test_roc_auc": test_auc, "test_auc_01": test_auc01,
               "unseen_roc_auc": unseen_auc, "unseen_auc_01": unseen_auc01,
               "active": bool(rw.active), "train_rows": len(pool),
               "anchors": len(anchor_pos) + len(rw.extra),
               "suppressed_rate": rw.epoch_flipped_rate(),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)

        selectable = (not cfg.checkpoint_after_warmup) or (not cfg.enabled) or epoch > cfg.warmup
        score = row["val_auc"] if cfg.select == "val_auc" else row["val_loss"]
        if cfg.select == "last":
            torch.save(model.state_dict(), best_pt); best_epoch = epoch
        elif selectable and np.isfinite(score) and better(score, best):
            best, best_epoch, wait = score, epoch, 0
            torch.save(model.state_dict(), best_pt)
        elif selectable:
            wait += 1
        if epoch % 10 == 0 or epoch == 1:
            logger.info("epoch %3d/%d loss %.4f val_loss %.4f test_auc01 %.4f unseen_auc %.4f "
                        "suppressed %.3f (%.0fs)", epoch, cfg.epochs, row["loss"], vl,
                        test_auc01, unseen_auc, row["suppressed_rate"], row["seconds"])
        if cfg.select != "last" and wait >= cfg.patience:
            logger.info("early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    hist = pd.DataFrame(history)
    hist.to_csv(args.outdir / "history.csv", index=False)
    hist[["epoch", "test_roc_auc", "test_auc_01", "unseen_roc_auc", "unseen_auc_01"]].to_csv(
        args.outdir / "epoch_metrics.csv", index=False)

    if not best_pt.is_file():
        torch.save(model.state_dict(), best_pt); best_epoch = len(history)
    model.load_state_dict(torch.load(best_pt, map_location=device, weights_only=True))
    _, ty, tp_ = mixtcrpred.evaluate(model, test_dl, device)
    _, uy, up = mixtcrpred.evaluate(model, unseen_dl, device)
    test_auc, test_auc01 = mixtcrpred.scores(ty, tp_)
    unseen_auc, unseen_auc01 = mixtcrpred.scores(uy, up)
    pd.concat([
        meta.loc[pos["test"]].assign(split="test", score=tp_),
        meta.loc[pos["unseen"]].assign(split="unseen", score=up),
    ])[["peptide", "split", "tp", "score"]].rename(columns={"tp": "binder"}).to_csv(
        args.outdir / "predictions.csv", index=False)

    summary = {
        "table": str(args.table),
        "held_out_peptide": (meta.loc[pos["unseen"], "peptide"].iloc[0]
                             if len(pos["unseen"]) else None),
        "gradft": cfg.enabled, "selected_epoch": best_epoch, "epochs_run": len(history),
        "selection": cfg.select, "hooked_layer": hook_label,
        "hooked_units": int(rw.units) if rw.units else None,
        "rows": {s: int(len(pos[s])) for s in SPLITS} | {
            "anchors": int(len(anchor_pos)),
            "dropped_by_db_transformer": int(len(df) - len(meta)),
            "flipped_train": int(meta.loc[train_pos, "flipped"].sum())},
        "flip_rate_train": round(float(meta.loc[train_pos, "flipped"].mean()), 4),
        "selected": {"test_roc_auc": test_auc, "test_auc_01": test_auc01,
                     "unseen_roc_auc": unseen_auc, "unseen_auc_01": unseen_auc01},
        "final_epoch": {k: history[-1][k] for k in
                        ("test_roc_auc", "test_auc_01", "unseen_roc_auc", "unseen_auc_01")},
        "config": cfg.to_dict(),
        "mixtcrpred_config": dict(model_cfg.items("transformer")),
        "mixtcrpred_source": {"dir": str(repo), "url": args.mixtcrpred_url},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("done: epoch %d | test AUC %.4f AUC0.1 %.4f | unseen AUC %.4f AUC0.1 %.4f",
                best_epoch, test_auc, test_auc01, unseen_auc, unseen_auc01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
