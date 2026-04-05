"""
tests/test_dataset.py
Tests for XRDDataset and build_dataloaders.
"""

import sys
import tempfile
import numpy as np
import torch
import pytest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.dataset import XRDDataset, build_dataloaders


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture
def synthetic_data():
    """Create small synthetic XRD data."""
    N = 100
    n_points = 2000
    rng = np.random.RandomState(42)
    patterns = rng.rand(N, n_points).astype(np.float64)
    labels = {
        "crystal_system": rng.randint(0, 7, size=N).astype(np.int64),
        "space_group":    rng.randint(1, 231, size=N).astype(np.int64),
    }
    return patterns, labels


@pytest.fixture
def synthetic_npz(synthetic_data, tmp_path):
    """Save synthetic data to a temporary .npz file."""
    patterns, labels = synthetic_data
    npz_path = tmp_path / "test_dataset.npz"
    np.savez(npz_path, xrd=patterns, **labels)
    return str(npz_path)


@pytest.fixture
def minimal_cfg(synthetic_npz):
    """Minimal config for testing."""
    return {
        "data": {
            "val_fraction": 0.15,
            "test_fraction": 0.15,
            "random_seed": 42,
            "xrd_n_points": 2000,
            "augment": {"enabled": False},
        },
        "tasks": {
            "crystal_system": {"type": "classification", "n_classes": 7, "weight": 1.0},
            "space_group":    {"type": "classification", "n_classes": 230, "weight": 0.5},
        },
        "training": {"batch_size": 16},
        "paths": {"processed": synthetic_npz},
    }


# ── XRDDataset tests ─────────────────────────────────────────────────

def test_dataset_length(synthetic_data):
    patterns, labels = synthetic_data
    ds = XRDDataset(patterns, labels)
    assert len(ds) == 100


def test_dataset_getitem_shapes(synthetic_data):
    patterns, labels = synthetic_data
    ds = XRDDataset(patterns, labels)
    x, y = ds[0]
    assert x.shape == (1, 2000)
    assert x.dtype == torch.float32
    assert "crystal_system" in y
    assert "space_group" in y
    assert y["crystal_system"].dtype == torch.long
    assert y["space_group"].dtype == torch.long


def test_dataset_normalisation(synthetic_data):
    """Patterns should be normalised to [0, 1]."""
    patterns, labels = synthetic_data
    # Scale patterns up to simulate unnormalised data
    patterns_scaled = patterns * 100
    ds = XRDDataset(patterns_scaled, labels)
    x, _ = ds[0]
    assert x.max() <= 1.0 + 1e-5
    assert x.min() >= 0.0 - 1e-5


def test_dataset_with_augmentor(synthetic_data):
    """Augmentor should not crash and should return same shape."""
    from src.data.augment import XRDAugmentor
    patterns, labels = synthetic_data
    aug_cfg = {
        "augment": {
            "enabled": True,
            "gaussian_noise_std": 0.01,
            "peak_shift_max": 3,
            "intensity_scale_range": [0.9, 1.1],
            "baseline_drift": False,
        }
    }
    augmentor = XRDAugmentor(aug_cfg)
    ds = XRDDataset(patterns, labels, augmentor=augmentor)
    x, y = ds[0]
    assert x.shape == (1, 2000)


# ── build_dataloaders tests ──────────────────────────────────────────

def test_build_dataloaders(minimal_cfg):
    train_dl, val_dl, test_dl = build_dataloaders(
        minimal_cfg["paths"]["processed"], minimal_cfg
    )
    # Check they return data
    xb, yb = next(iter(train_dl))
    assert xb.shape[1] == 1
    assert xb.shape[2] == 2000
    assert "crystal_system" in yb


def test_dataloaders_no_overlap(minimal_cfg):
    """Train/val/test sets should not share samples."""
    train_dl, val_dl, test_dl = build_dataloaders(
        minimal_cfg["paths"]["processed"], minimal_cfg
    )
    n_train = len(train_dl.dataset)
    n_val   = len(val_dl.dataset)
    n_test  = len(test_dl.dataset)
    assert n_train + n_val + n_test == 100
