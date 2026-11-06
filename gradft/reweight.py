"""The GradFT rule for Keras models."""

from __future__ import annotations

from typing import Optional, Union

import numpy as np
import tensorflow as tf
from tensorflow import keras

OUTPUTS = ("logits", "probs", "sigmoid")
GROUP_BY = ("feature", "label")


def take(x, idx):
    return tf.nest.map_structure(lambda a: a[idx], x)


def concat(*xs):
    return tf.nest.map_structure(lambda *a: np.concatenate(a), *xs)


def _flat(t):
    return tf.reshape(t, [tf.shape(t)[0], -1])


def _resolve_hooks(model, layer, where):
    items = list(layer) if isinstance(layer, (list, tuple)) else [layer]
    if not items:
        raise ValueError("no hook layer given")
    layers, tensors = [], []
    for item in items:
        if isinstance(item, str):
            item = model.get_layer(item)
        if isinstance(item, keras.layers.Layer):
            tensor, source = (item.output if where == "output" else item.input), item
        else:
            tensor = item
            source = getattr(getattr(item, "_keras_history", None), "layer", None)
        if isinstance(tensor, (list, tuple, dict)):
            raise ValueError(f"layer {getattr(source, 'name', '?')!r} has several {where} "
                             "tensors; hook a single-tensor one")
        layers.append(source)
        tensors.append(tensor)
    return layers, tensors


def _as_index(mask, total):
    m = np.asarray(mask)
    idx = np.flatnonzero(m) if m.dtype == bool else m.astype(np.int64).ravel()
    if idx.size == 0:
        raise ValueError("unit_mask selects no units")
    if idx.min() < 0 or idx.max() >= total:
        raise ValueError(f"unit_mask indexes outside the {total} hooked units")
    return tf.constant(np.sort(np.unique(idx)), tf.int32)


class GradFT:
    def __init__(self, model: keras.Model, layer: Union[str, keras.layers.Layer], *,
                 num_classes: int, x_train, y_train,
                 warmup: int = 20, anchor_samples: Optional[int] = None,
                 extra_anchor_samples: int = 5, reselect_epoch: int = 5, lam: float = 3.0,
                 optimizer: Optional[keras.optimizers.Optimizer] = None, loss_fn=None,
                 output: str = "logits", group_by: str = "feature", where: str = "output",
                 flip_top_frac: float = 1.0, exempt_rate: float = 0.0,
                 extra_pool=None, unit_mask=None, record_units: bool = False,
                 dynamic_exempt: bool = False, seed: int = 0):
        if output not in OUTPUTS:
            raise ValueError(f"output must be one of {OUTPUTS}, got {output!r}")
        if group_by not in GROUP_BY:
            raise ValueError(f"group_by must be one of {GROUP_BY}, got {group_by!r}")
        if output == "sigmoid" and num_classes != 2:
            raise ValueError("a sigmoid output is binary, so num_classes must be 2")
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if reselect_epoch < 1:
            raise ValueError("reselect_epoch must be at least 1")

        self.model, self.num_classes = model, num_classes
        self.warmup, self.anchor_samples = warmup, anchor_samples
        self.extra_anchor_samples, self.reselect_epoch, self.lam = extra_anchor_samples, reselect_epoch, lam
        self.output, self.group_by = output, group_by
        self.flip_top_frac, self.exempt_rate = float(flip_top_frac), float(exempt_rate)
        self._exempt = tf.Variable(self.exempt_rate, dtype=tf.float32, trainable=False)
        self._exempt_dynamic = bool(dynamic_exempt)
        self.extra_pool = None if extra_pool is None else np.asarray(extra_pool, dtype=int)
        self.optimizer = optimizer or keras.optimizers.Adam()
        if loss_fn is None:
            loss_fn = (keras.losses.BinaryCrossentropy() if output == "sigmoid"
                       else keras.losses.SparseCategoricalCrossentropy(from_logits=output == "logits"))
        self.loss_fn = loss_fn

        if where not in ("output", "input"):
            raise ValueError(f"where must be 'output' or 'input', got {where!r}")
        self.layers, hs = _resolve_hooks(model, layer, where)
        self.layer = self.layers[0]
        self._widths = [int(np.prod(t.shape[1:])) for t in hs]
        self._total = int(sum(self._widths))
        self._sel = None if unit_mask is None else _as_index(unit_mask, self._total)
        self.units = self._total if self._sel is None else int(self._sel.shape[0])
        self._aux = keras.Model(model.inputs, list(hs) + [model.outputs[0]])

        self.x_train = x_train
        self.y_train = np.asarray(y_train).astype(int).ravel()
        n = len(self.y_train)
        self.rho_sum, self.rho_cnt = np.zeros(n), np.zeros(n)
        self.win_sum, self.win_cnt = np.zeros(n), np.zeros(n)
        self.extra = np.zeros(0, dtype=int)
        self.swaps, self.batch_log = [], []
        self.flip_sum, self.n_probes = 0.0, 0
        self.epoch, self.step, self.active = 0, 0, False
        self._rng = np.random.default_rng(seed)
        self._anchors = None

        self.record_units = bool(record_units)
        if self.record_units:
            self._unit_flipped = tf.Variable(tf.zeros([self.units]), trainable=False)
            self._unit_rows = tf.Variable(0.0, trainable=False)

        self._plain_step = tf.function(self._plain_step_impl, reduce_retracing=True)
        self._ng_step = tf.function(self._ng_step_impl, reduce_retracing=True)
        self._probe = tf.function(self._probe_impl, reduce_retracing=True)
        self._score_step = tf.function(self._score_step_impl, reduce_retracing=True)


    def set_anchors(self, x, y) -> None:
        y = np.asarray(y).astype(int).ravel()
        keep = []
        for c in range(self.num_classes):
            rows = np.flatnonzero(y == c)
            if self.anchor_samples is not None and len(rows) > self.anchor_samples:
                rows = np.sort(self._rng.choice(rows, self.anchor_samples, replace=False))
            keep.append(rows)
        keep = np.concatenate(keep)
        if not len(keep):
            raise ValueError("the anchor pool is empty")
        self.anchor_rows = keep
        self.anchor_x, self.anchor_y = take(x, keep), y[keep]
        self.anchor_classes = np.unique(self.anchor_y)
        if self.group_by == "feature" and len(self.anchor_classes) < self.num_classes:
            missing = sorted(set(range(self.num_classes)) - set(self.anchor_classes.tolist()))
            raise ValueError(
                f"group_by='feature' needs anchors for every class, but classes {missing} have none: "
                "every sample would be compared with another class's reference. Use group_by='label'.")
        self._anchors = None

    def anchor_set(self):
        if self._anchors is None:
            x = concat(self.anchor_x, take(self.x_train, self.extra)) if len(self.extra) else self.anchor_x
            y = np.concatenate([self.anchor_y, self.y_train[self.extra]])
            self._anchors = (tf.nest.map_structure(tf.constant, x), tf.constant(y, tf.int32))
        return self._anchors


    def begin_epoch(self, epoch: int) -> np.ndarray:
        self.epoch, self.step = epoch, 0
        self.active = epoch > self.warmup
        rows = np.arange(len(self.y_train))
        return np.setdiff1d(rows, self.extra) if self.active else rows

    def train_step(self, x, y, sample_weight=None, idx=None) -> float:
        labels = np.asarray(y).astype(int).ravel()
        y_loss = tf.constant(labels[:, None] if self.output == "sigmoid" else labels,
                             tf.float32 if self.output == "sigmoid" else tf.int32)
        w = None if sample_weight is None else tf.constant(np.asarray(sample_weight, np.float32))
        x = tf.nest.map_structure(tf.constant, x)
        self.step += 1
        if not self.active:
            return float(self._plain_step(x, y_loss, w))

        ref, flip, has_ref, centroids = self._probe(*self.anchor_set())
        self.flip_sum, self.n_probes = self.flip_sum + flip.numpy(), self.n_probes + 1
        loss, rho = self._ng_step(x, y_loss, tf.constant(labels, tf.int32), w, ref, flip, has_ref, centroids)
        rho = rho.numpy()
        ok = ~np.isnan(rho)
        if idx is not None:
            rows = np.asarray(idx)[ok]
            for s, c in ((self.rho_sum, self.rho_cnt), (self.win_sum, self.win_cnt)):
                np.add.at(s, rows, rho[ok])
                np.add.at(c, rows, 1)
        self.batch_log.append((self.epoch, self.step, float(rho[ok].mean()) if ok.any() else float("nan"), int(ok.sum())))
        return float(loss)

    def end_epoch(self, epoch: int):
        if not (self.active and (epoch - self.warmup) % self.reselect_epoch == 0):
            return None
        score = np.where(self.win_cnt > 0, self.win_sum / np.maximum(self.win_cnt, 1), np.inf)
        eligible = np.ones(len(self.y_train), bool)
        if self.extra_pool is not None:
            eligible[:] = False
            eligible[self.extra_pool] = True
        new = []
        for c in self.anchor_classes:
            rows = np.flatnonzero((self.y_train == c) & np.isfinite(score) & eligible)
            new.append(rows[np.argsort(score[rows], kind="stable")[: self.extra_anchor_samples]])
        new = np.concatenate(new).astype(int)
        returned, self.extra = self.extra, new
        self.swaps.append((epoch, returned.tolist(), new.tolist()))
        self.win_sum[:], self.win_cnt[:] = 0, 0
        self._anchors = None
        return returned, new

    def score(self, x, y, batch_size: int = 1024) -> np.ndarray:
        ref, flip, has_ref, centroids = self._probe(*self.anchor_set())
        labels = np.asarray(y).astype(int).ravel()
        out = []
        for s in range(0, len(labels), batch_size):
            xb = tf.nest.map_structure(lambda a: tf.constant(a[s:s + batch_size]), x)
            out.append(self._score_step(xb, tf.constant(labels[s:s + batch_size], tf.int32),
                                        ref, flip, has_ref, centroids).numpy())
        return np.concatenate(out) if out else np.zeros(0)

    def set_exempt_rate(self, value: float) -> None:
        self.exempt_rate = float(value)
        self._exempt.assign(self.exempt_rate)

    def pop_unit_flipped(self):
        if not self.record_units:
            raise RuntimeError("pass record_units=True to accumulate per-unit statistics")
        rows = float(self._unit_rows.numpy())
        share = self._unit_flipped.numpy() / max(rows, 1.0)
        self._unit_flipped.assign(tf.zeros_like(self._unit_flipped))
        self._unit_rows.assign(0.0)
        return share, rows

    def train_scores(self) -> np.ndarray:
        return np.where(self.rho_cnt > 0, self.rho_sum / np.maximum(self.rho_cnt, 1), np.nan)

    @property
    def flip_mean(self):
        return self.flip_sum / max(self.n_probes, 1)


    def _call_aux(self, x, training):
        outs = self._aux(x, training=training)
        return list(outs[:-1]), outs[-1]

    def _join(self, ts):
        return _flat(ts[0]) if len(ts) == 1 else tf.concat([_flat(t) for t in ts], 1)

    def _pick(self, m):
        return m if self._sel is None else tf.gather(m, self._sel, axis=1)

    def _scatter(self, d, hs):
        if self._sel is not None:
            dt = tf.transpose(d)
            d = tf.transpose(tf.scatter_nd(self._sel[:, None], dt,
                                           tf.stack([self._total, tf.shape(dt)[1]])))
        if len(hs) == 1:
            return [tf.reshape(d, tf.shape(hs[0]))]
        return [tf.reshape(p, tf.shape(h)) for p, h in zip(tf.split(d, self._widths, 1), hs)]


    def _soft_ce(self, out, target):
        if self.output == "logits":
            return -tf.reduce_sum(target * tf.nn.log_softmax(out), axis=1)
        if self.output == "probs":
            return -tf.reduce_sum(target * tf.math.log(tf.clip_by_value(out, 1e-7, 1.0)), axis=1)
        p = tf.clip_by_value(tf.reshape(out, [-1]), 1e-7, 1 - 1e-7)
        return -(target[:, 1] * tf.math.log(p) + target[:, 0] * tf.math.log(1 - p))

    def _probe_impl(self, xa, ya):
        n = tf.shape(ya)[0]
        onehot = tf.one_hot(ya, self.num_classes)
        wrong = (1.0 - onehot) / (self.num_classes - 1)
        x2 = tf.nest.map_structure(lambda a: tf.concat([a, a], 0), xa)
        with tf.GradientTape() as tape:
            hs, out = self._call_aux(x2, training=False)
            loss = tf.reduce_sum(self._soft_ce(out, tf.concat([onehot, wrong], 0)))
        g, hf = self._pick(self._join(tape.gradient(loss, hs))), self._pick(self._join(hs))
        counts = tf.reduce_sum(onehot, 0)
        avg = tf.transpose(onehot) / tf.maximum(counts, 1.0)[:, None]
        agree_orig = avg @ tf.sign(g[:n])
        agree_wrong = avg @ tf.sign(g[n:])
        return (tf.sign(agree_orig), tf.nn.relu(-agree_orig * agree_wrong),
                counts > 0, avg @ hf[:n])

    def _group(self, hf, labels, has_ref, centroids):
        if self.group_by == "label":
            return labels
        d = (tf.reduce_sum(hf ** 2, 1, keepdims=True) - 2 * hf @ tf.transpose(centroids)
             + tf.reduce_sum(centroids ** 2, 1)[None, :])
        return tf.argmin(tf.where(has_ref[None, :], d, np.inf), axis=1, output_type=tf.int32)

    @staticmethod
    def _flipped(gf, group, ref, flip):
        f, r = tf.gather(flip, group), tf.gather(ref, group)
        nz = gf != 0
        flipped = nz & (tf.sign(gf) == -r)
        rho = (tf.reduce_sum(f * tf.cast(flipped, f.dtype), 1)
               / tf.maximum(tf.reduce_sum(f * tf.cast(nz, f.dtype), 1), 1e-12))
        return flipped, rho

    def _plain_step_impl(self, x, y_loss, w):
        with tf.GradientTape() as tape:
            loss = self.loss_fn(y_loss, self.model(x, training=True), sample_weight=w)
        variables = self.model.trainable_variables
        self.optimizer.apply_gradients(zip(tape.gradient(loss, variables), variables))
        return loss

    def ng_gradients(self, x, y_loss, labels, w, ref, flip, has_ref, centroids):
        with tf.GradientTape(persistent=True) as tape:
            hs, out = self._call_aux(x, training=True)
            loss = self.loss_fn(y_loss, out, sample_weight=w)
        gf = self._pick(self._join(tape.gradient(loss, hs)))
        group = self._group(self._pick(self._join(hs)), labels, has_ref, centroids)
        valid = tf.gather(has_ref, group)
        flipped, rho = self._flipped(gf, group, ref, flip)
        flipped &= valid[:, None]
        if self.record_units:
            rows = tf.cast(valid, gf.dtype)
            self._unit_flipped.assign_add(tf.reduce_sum(tf.cast(flipped, gf.dtype), 0))
            self._unit_rows.assign_add(tf.reduce_sum(rows))
        weight = tf.where(flipped, tf.exp(-self.lam * tf.gather(flip, group)), tf.ones_like(gf))
        if self.flip_top_frac < 1.0:
            r = tf.where(valid, rho, tf.fill(tf.shape(rho), -1.0))
            n = tf.shape(r)[0]
            k = tf.maximum(tf.cast(tf.math.ceil(self.flip_top_frac * tf.cast(n, tf.float32)), tf.int32), 1)
            thresh = tf.sort(r, direction="DESCENDING")[k - 1]
            weight = tf.where((r >= thresh)[:, None], weight, tf.ones_like(weight))
        if self.exempt_rate > 0.0 or self._exempt_dynamic:
            exempt = tf.random.uniform([1, tf.shape(gf)[1]]) < self._exempt
            weight = tf.where(tf.broadcast_to(exempt, tf.shape(gf)), tf.ones_like(weight), weight)
        deltas = self._scatter(gf * (weight - 1.0), hs)
        variables = self.model.trainable_variables
        grads = tape.gradient([loss] + hs, variables,
                              output_gradients=[tf.ones_like(loss)] + deltas,
                              unconnected_gradients=tf.UnconnectedGradients.ZERO)
        del tape
        return loss, grads, tf.where(valid, rho, np.nan)

    def _ng_step_impl(self, x, y_loss, labels, w, ref, flip, has_ref, centroids):
        loss, grads, rho = self.ng_gradients(x, y_loss, labels, w, ref, flip, has_ref, centroids)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss, rho

    def _score_step_impl(self, x, labels, ref, flip, has_ref, centroids):
        with tf.GradientTape() as tape:
            hs, out = self._call_aux(x, training=False)
            loss = tf.reduce_sum(self._soft_ce(out, tf.one_hot(labels, self.num_classes)))
        gf = self._pick(self._join(tape.gradient(loss, hs)))
        group = self._group(self._pick(self._join(hs)), labels, has_ref, centroids)
        _, rho = self._flipped(gf, group, ref, flip)
        return tf.where(tf.gather(has_ref, group), rho, np.nan)
