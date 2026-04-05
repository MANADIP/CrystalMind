"""
src/data/xrd_simulator.py
Simulate XRD patterns from pymatgen Structure objects.

For each structure:
  1. Compute Bragg peaks using pymatgen XRDCalculator
  2. Discretise to a fixed 1D array of length `n_points`
  3. Apply Gaussian broadening to simulate instrument resolution
  4. Normalise to [0, 1]
"""

import numpy as np

# NOTE: pymatgen imports are deferred to function bodies so that
# augment_xrd (pure numpy) can be used without pymatgen installed.


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

    Parameters
    ----------
    structure     : pymatgen Structure
    wavelength    : X-ray source ('CuKa', 'MoKa', 'AgKa', ...)
    two_theta_min : minimum 2theta in degrees
    two_theta_max : maximum 2theta in degrees
    n_points      : number of discrete bins
    sigma_deg     : Gaussian peak broadening in degrees

    Returns
    -------
    pattern : np.ndarray of shape (n_points,), normalised to [0,1]
    """
    from pymatgen.analysis.diffraction.xrd import XRDCalculator

    calc = XRDCalculator(wavelength=wavelength)
    pattern = calc.get_pattern(structure,
                               two_theta_range=(two_theta_min, two_theta_max))

    # Bin positions and intensities
    two_theta_axis = np.linspace(two_theta_min, two_theta_max, n_points)
    signal = np.zeros(n_points, dtype=np.float32)

    for angle, intensity in zip(pattern.x, pattern.y):
        # Place Gaussian at each Bragg peak
        gauss = np.exp(-0.5 * ((two_theta_axis - angle) / sigma_deg) ** 2)
        signal += intensity * gauss

    # Normalise to [0, 1]
    if signal.max() > 0:
        signal /= signal.max()

    return signal


def simulate_batch(
    structures: list,          # list of pymatgen Structure or None
    **kwargs,
) -> tuple[np.ndarray, list[int]]:
    """
    Simulate XRD patterns for a list of structures.
    Skips None structures and returns a mask of valid indices.

    Returns
    -------
    patterns     : np.ndarray  (N_valid, n_points)
    valid_indices: list[int]   indices of structures that succeeded
    """
    patterns = []
    valid_indices = []

    for i, struct in enumerate(structures):
        if struct is None:
            continue
        try:
            p = simulate_xrd(struct, **kwargs)
            patterns.append(p)
            valid_indices.append(i)
        except Exception as e:
            # Some disordered structures cannot be simulated
            pass

    if not patterns:
        raise RuntimeError("No valid XRD patterns could be simulated.")

    return np.stack(patterns, axis=0), valid_indices


def augment_xrd(
    pattern: np.ndarray,
    noise_std: float = 0.01,
    shift_bins: int = 0,
    scale: float = 1.0,
    baseline_drift: bool = False,
) -> np.ndarray:
    """
    Apply random augmentations to a single XRD pattern.
    Called at dataset __getitem__ time during training.
    """
    p = pattern.copy()

    # 1. Gaussian noise
    if noise_std > 0:
        p += np.random.normal(0, noise_std, size=p.shape).astype(np.float32)

    # 2. Peak shift (roll along 2θ axis)
    if shift_bins != 0:
        p = np.roll(p, shift_bins)

    # 3. Intensity scale
    if scale != 1.0:
        p *= scale

    # 4. Random baseline drift (low-frequency polynomial)
    if baseline_drift:
        t = np.linspace(0, 1, len(p))
        drift = np.random.uniform(-0.05, 0.05) * t + \
                np.random.uniform(-0.02, 0.02) * t**2
        p += drift.astype(np.float32)

    # Re-clip to [0, 1]
    p = np.clip(p, 0, 1)
    return p
