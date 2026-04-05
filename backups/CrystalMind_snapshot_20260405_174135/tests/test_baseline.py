"""
tests/test_baseline.py
Sanity-check tests for the BaselineCNN model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from src.models.baseline_cnn import BaselineCNN

MINIMAL_CFG = {
    "data":  {"xrd_n_points": 2000},
    "model": {
        "name": "BaselineCNN",
        "dropout": 0.1,
    },
    "tasks": {
        "crystal_system":  {"type": "classification", "n_classes": 7,   "weight": 1.0},
        "space_group":     {"type": "classification", "n_classes": 230, "weight": 0.5},
    },
}


@pytest.fixture
def model():
    m = BaselineCNN(MINIMAL_CFG)
    m.eval()
    return m


def test_output_shapes(model):
    x = torch.randn(4, 1, 2000)
    with torch.no_grad():
        out = model(x)
    assert out["crystal_system"].shape == (4, 7)
    assert out["space_group"].shape    == (4, 230)


def test_forward_with_features(model):
    x = torch.randn(2, 1, 2000)
    with torch.no_grad():
        out, feat = model.forward_with_features(x)
    assert feat.shape[0] == 2
    assert feat.shape[1] == 128   # out channels


def test_parameter_count(model):
    n = model.count_parameters()
    print(f"BaselineCNN parameters: {n:,}")
    assert n > 10_000, f"Too few parameters: {n}"
    assert n < 1_000_000, f"Baseline should be small: {n}"


def test_mc_dropout(model):
    x = torch.randn(2, 1, 2000)
    unc = model.predict_with_uncertainty(x, n_passes=5)
    assert "mean_crystal_system" in unc
    assert "std_crystal_system"  in unc
    assert unc["std_crystal_system"].shape == (2, 7)


def test_batch_size_one(model):
    """Model should work with batch size 1."""
    x = torch.randn(1, 1, 2000)
    with torch.no_grad():
        out = model(x)
    assert out["crystal_system"].shape == (1, 7)
