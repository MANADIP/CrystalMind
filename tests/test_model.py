"""
tests/test_model.py
Quick sanity-check tests for the CrystalMind model.
"""

import copy
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.multitask import CrystalMind


MINIMAL_CFG = {
    "data": {"xrd_n_points": 2000},
    "model": {
        "channels": [32, 64, 128, 256, 256],
        "kernels": [15, 7, 7, 7, 7],
        "strides": [2, 2, 2, 2, 1],
        "dilations": [1, 1, 1, 1, 2],
        "residual_blocks_per_stage": 2,
        "dropout": 0.1,
    },
    "tasks": {
        "crystal_system": {"type": "classification", "n_classes": 7, "weight": 1.0},
        "space_group": {"type": "classification", "n_classes": 230, "weight": 0.5},
        "band_gap": {"type": "regression", "weight": 0.3},
        "formation_energy": {"type": "regression", "weight": 0.3},
        "magnetic_ordering": {"type": "classification", "n_classes": 3, "weight": 0.2},
    },
}


@pytest.fixture
def model():
    instance = CrystalMind(MINIMAL_CFG)
    instance.eval()
    return instance


def test_output_shapes(model):
    x = torch.randn(4, 1, 2000)
    with torch.no_grad():
        out = model(x)
    assert out["crystal_system"].shape == (4, 7)
    assert out["space_group"].shape == (4, 230)
    assert out["band_gap"].shape == (4,)
    assert out["formation_energy"].shape == (4,)
    assert out["magnetic_ordering"].shape == (4, 3)


def test_forward_with_features(model):
    x = torch.randn(2, 1, 2000)
    with torch.no_grad():
        out, feat = model.forward_with_features(x)
    assert out["crystal_system"].shape == (2, 7)
    assert feat.shape[0] == 2
    assert feat.shape[1] == 256


def test_parameter_count(model):
    n_params = model.count_parameters()
    assert n_params > 100_000
    assert n_params < 20_000_000


def test_mc_dropout(model):
    model.train()
    x = torch.randn(2, 1, 2000)
    unc = model.predict_with_uncertainty(x, n_passes=5)
    assert "mean_crystal_system" in unc
    assert "std_crystal_system" in unc
    assert unc["std_crystal_system"].shape == (2, 7)


def test_hybrid_backbone_shapes():
    cfg = copy.deepcopy(MINIMAL_CFG)
    cfg["model"]["transformer"] = {
        "enabled": True,
        "n_layers": 1,
        "n_heads": 4,
        "ff_multiplier": 2.0,
        "dropout": 0.1,
        "max_seq_len": 128,
    }

    model = CrystalMind(cfg)
    model.eval()
    x = torch.randn(2, 1, 2000)
    with torch.no_grad():
        out, feat = model.forward_with_features(x)

    assert model.backbone.sequence_encoder is not None
    assert out["crystal_system"].shape == (2, 7)
    assert out["space_group"].shape == (2, 230)
    assert feat.shape[1] == 256
