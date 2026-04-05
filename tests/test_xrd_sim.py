"""
tests/test_xrd_sim.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from src.data.xrd_simulator import augment_xrd


def test_augment_shape():
    p = np.random.rand(2000).astype(np.float32)
    out = augment_xrd(p, noise_std=0.01, shift_bins=5, scale=1.1)
    assert out.shape == (2000,)


def test_augment_clipped():
    p = np.ones(2000, dtype=np.float32)
    out = augment_xrd(p, noise_std=0.5, scale=2.0)
    assert out.max() <= 1.0 + 1e-5
    assert out.min() >= 0.0 - 1e-5


def test_augment_no_op():
    p   = np.random.rand(2000).astype(np.float32)
    out = augment_xrd(p, noise_std=0.0, shift_bins=0, scale=1.0,
                      baseline_drift=False)
    np.testing.assert_allclose(out, np.clip(p, 0, 1), atol=1e-6)
