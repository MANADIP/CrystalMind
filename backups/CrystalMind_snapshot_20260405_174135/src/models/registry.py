"""
src/models/registry.py
Model factory — build any model variant from config.

Usage:
    from src.models.registry import build_model
    model = build_model(cfg)   # reads cfg["model"]["name"]
"""

from src.models.multitask    import CrystalMind
from src.models.baseline_cnn import BaselineCNN
from src.models.transformer  import XRDTransformer

# ── Registry ──────────────────────────────────────────────────────────

_MODELS = {
    "CrystalMind1DResNet": CrystalMind,
    "BaselineCNN":         BaselineCNN,
    "XRDTransformer":      XRDTransformer,
}


def list_models() -> list[str]:
    """Return all registered model names."""
    return list(_MODELS.keys())


def build_model(cfg: dict):
    """
    Instantiate a model from config.

    Parameters
    ----------
    cfg : dict
        Full config dict.  Must contain cfg["model"]["name"] matching
        one of: CrystalMind1DResNet, BaselineCNN, XRDTransformer

    Returns
    -------
    nn.Module with .forward(), .forward_with_features(),
    .predict_with_uncertainty(), .count_parameters()
    """
    name = cfg["model"]["name"]
    if name not in _MODELS:
        raise ValueError(
            f"Unknown model: '{name}'. "
            f"Available: {list_models()}"
        )
    return _MODELS[name](cfg)
