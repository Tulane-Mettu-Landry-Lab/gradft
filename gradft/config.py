#!/usr/bin/env python
"""Run configuration: defaults, YAML/JSON loading and validation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class Config:
    epochs: int = 200
    patience: int = 200
    batch_size: int = 64
    learning_rate: float = 1e-3
    dropout_rate: float = 0.6
    seed: int = 15

    enabled: bool = True
    warmup: int = 50
    exempt_rate: float = 0.40
    lam: float = 10.0
    group_by: str = "feature"
    flip_top_frac: float = 0.50
    extra_anchor_samples: int = 25
    reselect_epoch: int = 5
    anchor_batch_last: bool = True
    checkpoint_after_warmup: bool = True

    select: str = "val_auc_01"

    notes: str = ""

    def __post_init__(self):
        if self.group_by not in ("feature", "label"):
            raise ValueError(f"group_by must be 'feature' or 'label', got {self.group_by!r}")
        if self.select not in ("val_auc_01", "val_loss", "last"):
            raise ValueError(f"select must be 'val_auc_01', 'val_loss' or 'last', got {self.select!r}")
        if not 0.0 <= self.exempt_rate < 1.0:
            raise ValueError(f"exempt_rate must be in [0, 1), got {self.exempt_rate}")
        if not 0.0 < self.flip_top_frac <= 1.0:
            raise ValueError(f"flip_top_frac must be in (0, 1], got {self.flip_top_frac}")
        if self.reselect_epoch < 1:
            raise ValueError(f"reselect_epoch must be at least 1, got {self.reselect_epoch}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.enabled and self.warmup >= self.epochs:
            raise ValueError(
                f"warmup ({self.warmup}) >= epochs ({self.epochs}): the method would never "
                "activate. Raise epochs or lower warmup, or set enabled: false.")


    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True))

    def replace(self, **kw) -> "Config":
        d = self.to_dict()
        d.update({k: v for k, v in kw.items() if v is not None})
        return Config(**d)


@dataclass
class MixConfig:
    epochs: int = 100
    patience: int = 100
    batch_size: int = 64
    seed: int = 15

    enabled: bool = True
    warmup: int = 5
    exempt_rate: float = 0.40
    lam: float = 10.0
    group_by: str = "feature"
    flip_top_frac: float = 0.50
    extra_anchor_samples: int = 25
    reselect_epoch: int = 5
    anchor_batch_last: bool = True
    checkpoint_after_warmup: bool = True

    with_epitope: bool = True

    select: str = "val_loss"

    notes: str = ""

    def __post_init__(self):
        if self.group_by not in ("feature", "label"):
            raise ValueError(f"group_by must be 'feature' or 'label', got {self.group_by!r}")
        if self.select not in ("val_loss", "val_auc", "last"):
            raise ValueError(f"select must be 'val_loss', 'val_auc' or 'last', got {self.select!r}")
        if not 0.0 <= self.exempt_rate < 1.0:
            raise ValueError(f"exempt_rate must be in [0, 1), got {self.exempt_rate}")
        if not 0.0 < self.flip_top_frac <= 1.0:
            raise ValueError(f"flip_top_frac must be in (0, 1], got {self.flip_top_frac}")
        if self.reselect_epoch < 1:
            raise ValueError(f"reselect_epoch must be at least 1, got {self.reselect_epoch}")
        if self.epochs < 1:
            raise ValueError(f"epochs must be at least 1, got {self.epochs}")
        if self.enabled and self.warmup >= self.epochs:
            raise ValueError(
                f"warmup ({self.warmup}) >= epochs ({self.epochs}): the method would never "
                "activate. Raise epochs or lower warmup, or set enabled: false.")


    to_dict = Config.to_dict
    save = Config.save

    def replace(self, **kw) -> "MixConfig":
        d = self.to_dict()
        d.update({k: v for k, v in kw.items() if v is not None})
        return MixConfig(**d)


def _read(path: Optional[Path], cls):
    data: Dict[str, Any] = {}
    if path is not None:
        path = Path(path)
        text = path.read_text()
        if path.suffix in (".yaml", ".yml"):
            import yaml
            data = yaml.safe_load(text) or {}
        elif path.suffix == ".json":
            data = json.loads(text)
        else:
            raise ValueError(f"config must be .yaml, .yml or .json, got {path.suffix!r}")
        if not isinstance(data, dict):
            raise ValueError(f"{path} must hold a mapping, got {type(data).__name__}")

    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        import difflib
        hints = {u: difflib.get_close_matches(u, known, n=2) for u in sorted(unknown)}
        detail = "; ".join(f"{u!r}" + (f" (did you mean {' or '.join(h)}?)" if h else "")
                           for u, h in hints.items())
        raise ValueError(f"unknown key(s) in {path}: {detail}")
    return data


def load_config(path: Optional[Path] = None, **overrides) -> Config:
    return Config(**_read(path, Config)).replace(**overrides)


def load_mix_config(path: Optional[Path] = None, **overrides) -> MixConfig:
    return MixConfig(**_read(path, MixConfig)).replace(**overrides)
