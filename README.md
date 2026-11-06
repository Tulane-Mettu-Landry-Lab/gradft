# Gradient Filtering

Gradient filtering for noisy TCR–epitope labels training, on **NetTCR-2.2** and
**MixTCRpred**.


## 1. Install

NetTCR-2.2:TensorFlow
```bash
python -m venv .venv-nettcr && . .venv-nettcr/bin/activate
pip install -r requirements-nettcr.txt
```
MixTCRpred:PyTorch.

```bash
python -m venv .venv-mix && . .venv-mix/bin/activate
pip install -r requirements-mixtcrpred.txt
```

## 2. Get the models

```bash
git clone --depth 1 https://github.com/mnielLab/NetTCR-2.2   third_party/NetTCR-2.2
git clone --depth 1 https://github.com/GfellerLab/MixTCRpred third_party/MixTCRpred
```

## 3. Get the data

Download the TCRvdb release and save it as `datasets/source/TCRvdb.csv`.

```bash
python prepare_tcrvdb.py       # -> datasets/source/tcrvdb_train.csv
python make_datasets.py        # -> 30 datasets/lopo_*.csv
python make_mix_datasets.py    # -> 30 datasets/mix_lopo_*.csv
```

## 4. Run

```bash
python run_nettcr.py        # in the TensorFlow environment
python run_mixtcrpred.py    # in the PyTorch environment
```


## Configuration

Both arms share everything except whether GradFT is on. `warmup` 50 / `exempt_rate` 0.40
on NetTCR-2.2, `warmup` 5 / `exempt_rate` 0.40 on MixTCRpred (it trains for 100 epochs
rather than 200 and converges earlier). Edit `configs/*.yaml` or override per run:

```bash
python run_nettcr.py --warmup 25 --exempt-rate 0.2 --epochs 100
```

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
