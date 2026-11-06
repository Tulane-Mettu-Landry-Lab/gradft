# GradFT

Anchor-guided gradient reweighting for noisy TCR–pMHC labels, on **NetTCR-2.2** and
**MixTCRpred**.

A small set of trusted anchors gives each class a reference direction at one hidden
layer. Gradient entries pointing against it are scaled down; the rest pass through. The
model, the loss and the optimiser are untouched.

Neither model's code is included here, and neither is the data. Steps 1–3 below fetch
them.

## 1. Install

Two separate environments. NetTCR-2.2 is TensorFlow, MixTCRpred is PyTorch, and the two
pull incompatible versions of the same CUDA wheels — one environment with both in it
will not work. GradFT imports its backends lazily, so neither side needs the other.

```bash
python -m venv .venv-nettcr && . .venv-nettcr/bin/activate
pip install -r requirements-nettcr.txt

python -m venv .venv-mix && . .venv-mix/bin/activate
pip install -r requirements-mixtcrpred.txt
```

Python 3.10. A GPU is not required but CPU training is roughly 10× slower.

## 2. Get the models

```bash
git clone --depth 1 https://github.com/mnielLab/NetTCR-2.2   third_party/NetTCR-2.2
git clone --depth 1 https://github.com/GfellerLab/MixTCRpred third_party/MixTCRpred
```

Skip this and the run scripts will clone them on first use. Neither clone is modified —
the adapters in `gradft/` put it on `sys.path` and call it. Note NetTCR-2.2 ships an
academic software license agreement; check it covers your use.

## 3. Get the data

Download the TCRvdb release and save it as `datasets/source/TCRvdb.csv`. It must carry
`tcr_id`, `pos`, `ambiguous`, `epitope_aa`, `cdr3_alpha_aa`, `cdr3_beta_aa`,
`TRAV_IMGT`, `TRAJ_IMGT`, `TRBV_IMGT`, `TRBJ_IMGT`, `TRAV_vdjdb`, `TRBV_vdjdb` and
`hla_long`; `prepare_tcrvdb.py` says which are missing if any are.

```bash
python prepare_tcrvdb.py       # -> datasets/source/tcrvdb_train.csv
python make_datasets.py        # -> 30 datasets/lopo_*.csv
python make_mix_datasets.py    # -> 30 datasets/mix_lopo_*.csv
```

The run scripts do all three for you if the tables are missing. Everything is seeded, so
the same TCRvdb release always gives the same splits, anchors and flipped labels.

## 4. Run

```bash
python run_nettcr.py        # in the TensorFlow environment
python run_mixtcrpred.py    # in the PyTorch environment
```

Each trains both arms — GradFT and the vanilla control — over 5 peptides × 6 flip rates
× 3 seeds, 180 runs, and prints the paired difference at the end.

Check the plumbing first:

```bash
python run_nettcr.py --smoke        # ~2 min
python run_mixtcrpred.py --smoke    # ~5 min
```

The full NetTCR-2.2 grid is ~12 h one run at a time. It is a small model, so pack a GPU:

```bash
python run_nettcr.py --jobs 6 --gpu-memory-limit 1536    # ~2 h
```

Set `--gpu-memory-limit` whenever `--jobs > 1`, or one run can take the whole card and
the rest die with OOM. MixTCRpred is a transformer at 15–25 min per run; one or two per
card at most.

Runs are independent and already-finished ones are skipped, so an interrupted grid
resumes if you re-run the same command. `--report-only` re-reads what is on disk without
training. `--peptides`, `--flips` and `--seeds` narrow the grid; `--help` lists the rest.

## Configuration

Both arms share everything except whether GradFT is on. `warmup` 50 / `exempt_rate` 0.40
on NetTCR-2.2, `warmup` 5 / `exempt_rate` 0.40 on MixTCRpred (it trains for 100 epochs
rather than 200 and converges earlier). Edit `configs/*.yaml` or override per run:

```bash
python run_nettcr.py --warmup 25 --exempt-rate 0.2 --epochs 100
```

Unknown keys in a config are rejected rather than ignored.

## Using GradFT elsewhere

`gradft/reweight.py` and `gradft/reweight_torch.py` know nothing about TCRs:

```python
from gradft.reweight import GradFT, take

rw = GradFT(model, layer, num_classes=2, x_train=x, y_train=y, optimizer=opt,
            warmup=50, exempt_rate=0.40, lam=10.0)
rw.set_anchors(x_anchor, y_anchor)

for epoch in range(1, epochs + 1):
    pool = rw.begin_epoch(epoch)
    for idx in batches(pool):
        rw.train_step(take(x, idx), y[idx], idx=idx)
    rw.end_epoch(epoch)
```

The torch version hooks a module's input, so `probe()` is called explicitly before each
batch; `train_mixtcrpred.py` is a worked example.
