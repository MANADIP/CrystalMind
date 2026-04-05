"""
src/data/xrd_simulator.py
Simulate XRD patterns from pymatgen Structure objects and apply
physics-inspired augmentations.
"""

import numpy as np


def simulate_xrd(
    structure,
    wavelength: str = "CuKa",
    two_theta_min: float = 5.0,
    two_theta_max: float = 90.0,
    n_points: int = 2000,
    sigma_deg: float = 0.1,
) -> np.ndarray:
    """
    Simulate an XRD pattern for `structure`.

    Returns
    -------
    np.ndarray of shape (n_points,), normalized to [0, 1]
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure, two_theta_range=(two_theta_min, two_theta_max))

    two_theta_axis = np.linspace(two_theta_min, two_theta_max, n_points)
    signal = np.zeros(n_points, dtype=np.float32)

    for angle, intensity in zip(pattern.x, pattern.y):
        gauss = np.exp(-0.5 * ((two_theta_axis - angle) / sigma_deg) ** 2)
        signal += intensity * gauss

    if signal.max() > 0:
        signal /= signal.max()

    return signal


def simulate_batch(structures: list, **kwargs) -> tuple[np.ndarray, list[int]]:
    """
    Simulate XRD patterns for a list of structures.
    Skips None structures and returns a mask of valid indices.
    """
    patterns = []
    valid_indices = []

    for idx, struct in enumerate(structures):
        if struct is None:
            continue
        try:
            patterns.append(simulate_xrd(struct, **kwargs))
            valid_indices.append(idx)
        except Exception:
            pass

    if not patterns:
        raise RuntimeError("No valid XRD patterns could be simulated.")

    return np.stack(patterns, axis=0), valid_indices


def augment_xrd(
    pattern: np.ndarray,
    noise_std: float = 0.01,
    poisson_scale: float = 0.0,
    shift_bins: int = 0,
    scale: float = 1.0,
    baseline_drift: bool = False,
) -> np.ndarray:
    """
    Apply random augmentations to a single XRD pattern.
    Called at dataset __getitem__ time during training.
    """
    p = pattern.copy()

    if shift_bins != 0:
        p = np.roll(p, shift_bins)

    if scale != 1.0:
        p *= scale

    if baseline_drift:
        t = np.linspace(0, 1, len(p))
        drift = np.random.uniform(-0.05, 0.05) * t
        drift += np.random.uniform(-0.02, 0.02) * (t**2)
        p += drift.astype(np.float32)

    if poisson_scale > 0:
        counts = np.clip(p, 0, None) * float(poisson_scale)
        p = np.random.poisson(counts).astype(np.float32) / float(poisson_scale)

    if noise_std > 0:
        p += np.random.normal(0, noise_std, size=p.shape).astype(np.float32)

    return np.clip(p, 0, 1)
