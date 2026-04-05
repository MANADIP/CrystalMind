"""
scripts/02_build_dataset.py
Step 2 — Simulate XRD patterns from fetched structures and build the .npz dataset.

Usage:
    python scripts/02_build_dataset.py
    python scripts/02_build_dataset.py --config configs/default.yaml
"""

import sys
import json
import argparse
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymatgen.core import Structure
from src.data.fetch        import fetch_materials
from src.data.xrd_simulator import simulate_xrd
from src.utils.io           import load_config, ensure_dirs


def build_dataset(cfg: dict) -> None:
    data_cfg  = cfg["data"]
    out_path  = Path(cfg["paths"]["processed"])
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Load cached records ───────────────────────────────────────────
    n = data_cfg["n_materials"]
    print(f"Loading {n} materials from cache …")
    records = fetch_materials(
        n_materials=n,
        out_dir=cfg["paths"]["raw_data"],
        cache_dir=cfg["paths"]["cache"],
    )
    print(f"  Loaded {len(records)} records.")

    # ── Simulate XRD for each structure ──────────────────────────────
    xrd_patterns       = []
    crystal_system     = []
    space_group        = []
    band_gap           = []
    formation_energy   = []
    magnetic_ordering  = []
    material_ids       = []
    formulas           = []

    print("\nSimulating XRD patterns …")
    skipped = 0

    for rec in tqdm(records, desc="XRD Simulation"):
        struct_dict = rec.get("structure")
        if struct_dict is None:
            skipped += 1
            continue

        try:
            struct  = Structure.from_dict(struct_dict)
            pattern = simulate_xrd(
                struct,
                wavelength=data_cfg["xrd_wavelength"],
                two_theta_min=data_cfg["two_theta_min"],
                two_theta_max=data_cfg["two_theta_max"],
                n_points=data_cfg["xrd_n_points"],
                sigma_deg=data_cfg["two_theta_sigma"],
            )
        except Exception as e:
            skipped += 1
            continue

        xrd_patterns.append(pattern)
        crystal_system.append(rec["crystal_system"])
        space_group.append(rec["spacegroup_number"])
        band_gap.append(rec["band_gap"])
        formation_energy.append(rec["formation_energy"])
        magnetic_ordering.append(rec["magnetic_ordering"])
        material_ids.append(rec["material_id"])
        formulas.append(rec["formula"])

    n_valid = len(xrd_patterns)
    print(f"\n  Simulated : {n_valid}  |  Skipped : {skipped}")

    # ── Save .npz ────────────────────────────────────────────────────
    np.savez_compressed(
        out_path,
        xrd              = np.array(xrd_patterns,      dtype=np.float32),
        crystal_system   = np.array(crystal_system,    dtype=np.int32),
        space_group      = np.array(space_group,        dtype=np.int32),
        band_gap         = np.array(band_gap,           dtype=np.float32),
        formation_energy = np.array(formation_energy,   dtype=np.float32),
        magnetic_ordering= np.array(magnetic_ordering,  dtype=np.int32),
        material_ids     = np.array(material_ids,       dtype=object),
        formulas         = np.array(formulas,           dtype=object),
    )

    print(f"\n✓  Dataset saved → {out_path}")
    print(f"   Shape  : {n_valid} × {data_cfg['xrd_n_points']}")

    # ── Quick label stats ────────────────────────────────────────────
    cs_arr = np.array(crystal_system)
    names  = ["Cubic","Hexagonal","Trigonal","Tetragonal",
              "Orthorhombic","Monoclinic","Triclinic"]
    print("\n  Crystal system distribution:")
    for i, name in enumerate(names):
        count = (cs_arr == i).sum()
        print(f"    {name:14s} : {count:5d}  ({100*count/n_valid:.1f}%)")

    print(f"\n   Next step: python scripts/03_train.py")


def main():
    parser = argparse.ArgumentParser(
        description="Simulate XRD patterns and build dataset"
    )
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    print("=" * 55)
    print("  CrystalMind — Step 2: Build XRD Dataset")
    print("=" * 55)
    build_dataset(cfg)


if __name__ == "__main__":
    main()
