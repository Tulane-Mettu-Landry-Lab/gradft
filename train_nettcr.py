#!/usr/bin/env python
"""One NetTCR-2.2 run: python train_nettcr.py --table T --outdir D --config C"""

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

from gradft import nettcr

nettcr.quiet_tensorflow()

import numpy as np
import pandas as pd

from gradft.config import load_config

logger = logging.getLogger("train_nettcr")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--table", type=Path, required=True,
                   help="Split table from make_datasets.py (columns split/anchor/flipped).")
    p.add_argument("--outdir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=None, help="YAML or JSON config; defaults apply if omitted.")
    p.add_argument("--nettcr-dir", type=Path, default=HERE / "third_party" / "NetTCR-2.2",
                   help="Where to find (or clone) the official NetTCR-2.2 (default: %(default)s).")
    p.add_argument("--nettcr-url", default=nettcr.NETTCR_URL)
    p.add_argument("--skip-existing", action="store_true", help="Do nothing if history.csv already exists.")
    p.add_argument("--gpu-memory-limit", type=int, default=None, metavar="MB",
                   help="Cap this process's GPU memory. Set it when packing several runs "
                        "onto one card; memory growth alone is first-come-first-served.")

    o = p.add_argument_group("config overrides (omitted values come from --config)")
    o.add_argument("--epochs", type=int)
    o.add_argument("--patience", type=int)
    o.add_argument("--batch-size", type=int)
    o.add_argument("--learning-rate", type=float)
    o.add_argument("--dropout-rate", type=float)
    o.add_argument("--seed", type=int)
    o.add_argument("--warmup", type=int, help="Epochs of plain training before gradft engages.")
    o.add_argument("--exempt-rate", type=float, help="Share of hidden units exempted each step.")
    o.add_argument("--lam", type=float)
    o.add_argument("--group-by", choices=["feature", "label"])
    o.add_argument("--flip-top-frac", type=float)
    o.add_argument("--extra-anchor-samples", type=int)
    o.add_argument("--reselect-epoch", type=int)
    o.add_argument("--select", choices=["val_auc_01", "val_loss", "last"])
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

    cfg = load_config(args.config, **{k: getattr(args, k) for k in (
        "epochs", "patience", "batch_size", "learning_rate", "dropout_rate", "seed",
        "warmup", "exempt_rate", "lam", "group_by", "flip_top_frac",
        "extra_anchor_samples", "reselect_epoch", "select", "enabled")})
    cfg.save(args.outdir / "config.resolved.json")

    repo = nettcr.ensure_nettcr(args.nettcr_dir, args.nettcr_url)
    nettcr.add_to_path(repo)

    import tensorflow as tf
    from tensorflow import keras

    nettcr.configure_gpu(args.gpu_memory_limit)

    from gradft.reweight import GradFT, take

    df = pd.read_csv(args.table)
    for col in ("split", "anchor", "flipped"):
        if col not in df.columns:
            raise ValueError(f"{args.table} has no {col!r} column; build it with make_datasets.py")
    df, enc_report = nettcr.check_encodable(df)

    train_df = df[df.split == "train"].reset_index(drop=True)
    val_df = df[df.split == "val"].reset_index(drop=True)
    test_df = df[df.split == "test"].reset_index(drop=True)
    unseen_df = df[df.split == "unseen"].reset_index(drop=True)
    anchor_df = train_df[train_df.anchor == 1].reset_index(drop=True)
    if anchor_df.empty and cfg.enabled:
        raise ValueError(f"{args.table} marks no anchor rows, but gradft needs them")
    if anchor_df.flipped.sum():
        raise AssertionError("anchors must carry true labels, but some are flipped")

    w_train, w_val = nettcr.sample_weights(train_df, val_df)
    x_train, x_val = nettcr.encode_batch(train_df), nettcr.encode_batch(val_df)
    x_test, x_unseen = nettcr.encode_batch(test_df), nettcr.encode_batch(unseen_df)
    x_anchor = nettcr.encode_batch(anchor_df)
    y_train = train_df.binder.to_numpy(int)
    y_val, y_test = val_df.binder.to_numpy(int), test_df.binder.to_numpy(int)
    y_unseen = unseen_df.binder.to_numpy(int)

    np.random.seed(cfg.seed); random.seed(cfg.seed); tf.random.set_seed(cfg.seed)
    model = nettcr.build_model(cfg.dropout_rate, cfg.seed)
    hooked = nettcr.hook_layer(model)

    rw = GradFT(
        model, hooked,
        num_classes=2, x_train=x_train, y_train=y_train,
        warmup=cfg.epochs + 1 if not cfg.enabled else cfg.warmup,
        anchor_samples=None,
        extra_anchor_samples=cfg.extra_anchor_samples if cfg.enabled else 0,
        reselect_epoch=cfg.reselect_epoch, lam=cfg.lam, group_by=cfg.group_by,
        flip_top_frac=cfg.flip_top_frac, exempt_rate=cfg.exempt_rate,
        optimizer=keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss_fn=keras.losses.BinaryCrossentropy(), output="sigmoid", seed=cfg.seed)
    rw.set_anchors(x_anchor, anchor_df.binder.to_numpy(int))

    ckpt = args.outdir / "checkpoint"; ckpt.mkdir(parents=True, exist_ok=True)
    best_h5 = ckpt / "best.h5"
    anchor_local = np.flatnonzero(train_df.anchor.to_numpy() == 1)
    rng = np.random.default_rng(cfg.seed)
    history = []
    better = (lambda a, b: a > b) if cfg.select == "val_auc_01" else (lambda a, b: a < b)
    best = -np.inf if cfg.select == "val_auc_01" else np.inf
    best_epoch, wait = 0, 0

    logger.info("%s | train %d (flipped %d, anchors %d) val %d | test %d unseen %d (%s) "
                "| gradft=%s hook=%s (%d units)",
                args.table.name, len(train_df), int(train_df.flipped.sum()), len(anchor_df),
                len(val_df), len(test_df), len(unseen_df),
                unseen_df.peptide.iloc[0] if len(unseen_df) else "-", cfg.enabled,
                hooked.name, rw.units)

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        pool = rw.begin_epoch(epoch)
        order = pool[rng.permutation(len(pool))]
        losses = [rw.train_step(take(x_train, idx), y_train[idx], w_train[idx], idx=idx)
                  for idx in np.array_split(order, np.arange(cfg.batch_size, len(order), cfg.batch_size))]

        if cfg.anchor_batch_last and len(anchor_local) > 1:
            was, rw.active = rw.active, False
            for s in range(0, len(anchor_local), cfg.batch_size):
                sl = anchor_local[s:s + cfg.batch_size]
                if len(sl) > 1:
                    rw.train_step(take(x_train, sl), y_train[sl], w_train[sl], idx=None)
            rw.active = was

        p_val = model.predict(x_val, batch_size=cfg.batch_size, verbose=0).ravel() if len(val_df) else np.zeros(0)
        p_test = model.predict(x_test, batch_size=cfg.batch_size, verbose=0).ravel()
        p_unseen = model.predict(x_unseen, batch_size=cfg.batch_size, verbose=0).ravel()
        test_auc, test_auc01 = nettcr.scores(y_test, p_test)
        unseen_auc, unseen_auc01 = nettcr.scores(y_unseen, p_unseen)
        rho = [r[2] for r in rw.batch_log if r[0] == epoch and not np.isnan(r[2])]
        row = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
               "val_loss": nettcr.bce(y_val, p_val, w_val) if len(val_df) else float("nan"),
               "val_auc_01": nettcr.keras_auc_01(y_val, p_val, cfg.batch_size) if len(val_df) else float("nan"),
               "test_roc_auc": test_auc, "test_auc_01": test_auc01,
               "unseen_roc_auc": unseen_auc, "unseen_auc_01": unseen_auc01,
               "active": bool(rw.active), "train_rows": len(pool),
               "anchors": len(rw.anchor_y) + len(rw.extra),
               "suppressed_rate": float(np.mean(rho)) if rho else float("nan"),
               "seconds": round(time.time() - t0, 1)}
        history.append(row)

        selectable = (not cfg.checkpoint_after_warmup) or (not cfg.enabled) or epoch > cfg.warmup
        score = row["val_auc_01"] if cfg.select == "val_auc_01" else row["val_loss"]
        if cfg.select == "last":
            model.save(best_h5); best_epoch = epoch
        elif selectable and np.isfinite(score) and better(score, best):
            best, best_epoch, wait = score, epoch, 0
            model.save(best_h5)
        elif selectable:
            wait += 1
        rw.end_epoch(epoch)
        if epoch % 10 == 0 or epoch == 1:
            logger.info("epoch %3d/%d loss %.4f val_auc_01 %.4f test_auc01 %.4f unseen_auc %.4f suppressed %.3f",
                        epoch, cfg.epochs, row["loss"], row["val_auc_01"], test_auc01, unseen_auc,
                        row["suppressed_rate"])
        if cfg.select != "last" and wait >= cfg.patience:
            logger.info("early stop at epoch %d (best %d)", epoch, best_epoch)
            break

    hist = pd.DataFrame(history)
    hist.to_csv(args.outdir / "history.csv", index=False)
    hist[["epoch", "test_roc_auc", "test_auc_01", "unseen_roc_auc", "unseen_auc_01"]].to_csv(
        args.outdir / "epoch_metrics.csv", index=False)

    if not best_h5.is_file():
        model.save(best_h5); best_epoch = len(history)
    best_model = keras.models.load_model(best_h5, compile=False)
    p_test = best_model.predict(x_test, batch_size=cfg.batch_size, verbose=0).ravel()
    p_unseen = best_model.predict(x_unseen, batch_size=cfg.batch_size, verbose=0).ravel()
    test_auc, test_auc01 = nettcr.scores(y_test, p_test)
    unseen_auc, unseen_auc01 = nettcr.scores(y_unseen, p_unseen)
    pd.concat([
        test_df.assign(split="test", score=p_test),
        unseen_df.assign(split="unseen", score=p_unseen),
    ])[["peptide", "split", "binder", "score"]].to_csv(args.outdir / "predictions.csv", index=False)

    summary = {
        "table": str(args.table), "held_out_peptide": (unseen_df.peptide.iloc[0] if len(unseen_df) else None),
        "gradft": cfg.enabled, "selected_epoch": best_epoch, "epochs_run": len(history),
        "selection": cfg.select, "hooked_layer": hooked.name, "hooked_units": int(rw.units),
        "rows": {"train": len(train_df), "val": len(val_df), "test": len(test_df),
                 "unseen": len(unseen_df), "anchors": len(anchor_df),
                 "flipped_train": int(train_df.flipped.sum())},
        "flip_rate_train": round(float(train_df.flipped.mean()), 4),
        "selected": {"test_roc_auc": test_auc, "test_auc_01": test_auc01,
                     "unseen_roc_auc": unseen_auc, "unseen_auc_01": unseen_auc01},
        "final_epoch": {k: history[-1][k] for k in
                        ("test_roc_auc", "test_auc_01", "unseen_roc_auc", "unseen_auc_01")},
        "encodability": enc_report,
        "config": cfg.to_dict(),
        "nettcr_source": {"dir": str(repo), "url": args.nettcr_url},
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join(sys.argv),
    }
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("done: epoch %d | test AUC %.4f AUC0.1 %.4f | unseen AUC %.4f AUC0.1 %.4f",
                best_epoch, test_auc, test_auc01, unseen_auc, unseen_auc01)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
