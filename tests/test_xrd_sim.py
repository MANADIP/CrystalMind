"""
tests/test_xrd_sim.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.xrd_simulator import augment_xrd


def test_augment_shape():
    pattern = np.random.rand(2000).astype(np.float32)
    out = augment_xrd(pattern, noise_std=0.01, shift_bins=5, scale=1.1)
    assert out.shape == (2000,)


def test_augment_clipped():
    pattern = np.ones(2000, dtype=np.float32)
    out = augment_xrd(pattern, noise_std=0.5, scale=2.0)
    assert out.max() <= 1.0 + 1e-5
    assert out.min() >= 0.0 - 1e-5


def test_augment_no_op():
    pattern = np.random.rand(2000).astype(np.float32)
    out = augment_xrd(
        pattern,
        noise_std=0.0,
        poisson_scale=0.0,
        shift_bins=0,
        scale=1.0,
        baseline_drift=False,
    )
    np.testing.assert_allclose(out, np.clip(pattern, 0, 1), atol=1e-6)


def test_poisson_noise_changes_signal():
    np.random.seed(0)
    pattern = np.full(2000, 0.5, dtype=np.float32)
    out = augment_xrd(
        pattern,
        noise_std=0.0,
        poisson_scale=50.0,
        shift_bins=0,
        scale=1.0,
        baseline_drift=False,
    )
    assert out.shape == (2000,)
    assert out.min() >= 0.0
    assert out.max() <= 1.0
    assert not np.allclose(out, pattern)

