"""
src/data — XRD data loading, augmentation, and simulation.

Core exports (always available):
    XRDDataset, build_dataloaders, XRDAugmentor

Simulation exports (require pymatgen):
    simulate_xrd, simulate_batch
"""

from src.data.dataset import XRDDataset, build_dataloaders
from src.data.augment import XRDAugmentor

__all__ = [
    "XRDDataset",
    "build_dataloaders",
    "XRDAugmentor",
]


def __getattr__(name):
    """Lazy-import simulation functions that require pymatgen."""
    if name in ("simulate_xrd", "simulate_batch"):
        from src.data.xrd_simulator import simulate_xrd, simulate_batch
        globals()["simulate_xrd"]   = simulate_xrd
        globals()["simulate_batch"] = simulate_batch
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
