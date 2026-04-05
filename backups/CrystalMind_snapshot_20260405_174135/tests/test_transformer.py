"""
tests/test_transformer.py
Sanity-check tests for the XRDTransformer model.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import pytest
from src.models.transformer import XRDTransformer

MINIMAL_CFG = {
    "data":  {"xrd_n_points": 2000},
    "model": {
        "name": "XRDTransformer",
        "d_model": 64,       # smaller for fast tests
        "n_heads": 4,
        "n_layers": 2,
        "d_ff": 128,
        "patch_size": 16,
        "dropout": 0.1,
    },
    "tasks": {
        "crystal_system":  {"type": "classification", "n_classes": 7,   "weight": 1.0},
        "space_group":     {"type": "classification", "n_classes": 230, "weight": 0.5},
    },
}


@pytest.fixture
def model():
    m = XRDTransformer(MINIMAL_CFG)
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
    # feat: (B, d_model, n_patches+1)
    n_patches = 2000 // 16  # = 125
    assert feat.shape == (2, 64, n_patches + 1)


def test_cls_token_present(model):
    """Check that CLS token is the first position."""
    x = torch.randn(1, 1, 2000)
    with torch.no_grad():
        _, feat = model.forward_with_features(x)
    # feat[:, :, 0] is the CLS token output
    cls_feat = feat[:, :, 0]
    assert cls_feat.shape == (1, 64)


def test_parameter_count(model):
    n = model.count_parameters()
    print(f"XRDTransformer (test size) parameters: {n:,}")
    assert n > 50_000, f"Too few parameters: {n}"
    assert n < 5_000_000, f"Too many parameters for compact model: {n}"


def test_mc_dropout(model):
    x = torch.randn(2, 1, 2000)
    unc = model.predict_with_uncertainty(x, n_passes=5)
    assert "mean_crystal_system" in unc
    assert "std_crystal_system"  in unc
    assert unc["std_crystal_system"].shape == (2, 7)


def test_batch_size_one(model):
    x = torch.randn(1, 1, 2000)
    with torch.no_grad():
        out = model(x)
    assert out["crystal_system"].shape == (1, 7)


def test_different_patch_size():
    """Model should work with different patch sizes."""
    cfg = MINIMAL_CFG.copy()
    cfg["model"] = dict(cfg["model"])
    cfg["model"]["patch_size"] = 32  # 2000/32 = 62 patches

    m = XRDTransformer(cfg)
    m.eval()
    x = torch.randn(2, 1, 2000)
    with torch.no_grad():
        out = m(x)
    assert out["crystal_system"].shape == (2, 7)
