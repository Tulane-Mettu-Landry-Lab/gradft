"""GradFT: anchor-guided gradient reweighting for noisy-label training."""

from gradft.config import Config, MixConfig, load_config, load_mix_config

__all__ = ["Config", "MixConfig", "GradFT", "GradFTTorch", "load_config",
           "load_mix_config", "concat", "take"]
__version__ = "1.0.0"

_LAZY = {"GradFT": "gradft.reweight", "concat": "gradft.reweight", "take": "gradft.reweight",
         "GradFTTorch": "gradft.reweight_torch"}


def __getattr__(name):
    if name in _LAZY:
        import importlib
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY))
