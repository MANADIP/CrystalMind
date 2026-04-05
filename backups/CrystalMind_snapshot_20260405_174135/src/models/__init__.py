"""
src/models — CrystalMind model zoo.

Available models:
    CrystalMind     — 1D ResNet + SE blocks (advanced)
    BaselineCNN     — Simple 3-layer 1D CNN (baseline)
    XRDTransformer  — 1D Transformer encoder (advanced)
    build_model     — Factory: instantiate any model from config
"""

from src.models.multitask    import CrystalMind
from src.models.baseline_cnn import BaselineCNN
from src.models.transformer  import XRDTransformer
from src.models.registry     import build_model, list_models

__all__ = [
    "CrystalMind",
    "BaselineCNN",
    "XRDTransformer",
    "build_model",
    "list_models",
]
