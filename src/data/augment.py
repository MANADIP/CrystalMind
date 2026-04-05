"""
src/data/augment.py
Wrap xrd_simulator.augment_xrd with config-driven random sampling.
Used by the Dataset class during training.
"""

import numpy as np

from src.data.xrd_simulator import augment_xrd


class XRDAugmentor:
    """
    Stateless augmentor. Call an instance on a pattern to get an augmented copy.

    Parameters come from configs/default.yaml -> data.augment
    """

    def __init__(self, cfg: dict):
        aug = cfg.get("augment", {})
        self.enabled = aug.get("enabled", True)
        self.noise_std = aug.get("gaussian_noise_std", 0.01)
        self.poisson_scale = aug.get("poisson_noise_scale", 0.0)
        self.shift_max = aug.get("peak_shift_max", 5)
        self.scale_range = aug.get("intensity_scale_range", [0.8, 1.2])
        self.baseline = aug.get("baseline_drift", True)

    def __call__(self, pattern: np.ndarray) -> np.ndarray:
        if not self.enabled:
            return pattern

        shift = int(np.random.uniform(-self.shift_max, self.shift_max))
        scale = float(np.random.uniform(*self.scale_range))

        return augment_xrd(
            pattern,
            noise_std=self.noise_std,
            poisson_scale=self.poisson_scale,
            shift_bins=shift,
            scale=scale,
            baseline_drift=self.baseline,
        )
