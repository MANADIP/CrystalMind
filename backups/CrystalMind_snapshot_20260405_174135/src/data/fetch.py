"""
src/data/fetch.py
Fetches crystal structure data and labels from the Materials Project API.

Usage (standalone):
    python -m src.data.fetch --n_materials 5000 --out data/raw

Requires:
    MP_API_KEY environment variable  (get free key at materialsproject.org)
"""

import os
import json
import argparse
import logging
from pathlib import Path
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)


FIELDS = [
    "material_id",
    "formula_pretty",
    "crystal_system",
    "spacegroup_number",
    "band_gap",
    "formation_energy_per_atom",
    "ordering",          # magnetic ordering: NM / FM / AFM / FiM / Unknown
    "structure",
    "nelements",
    "volume",
    "density",
]

# Map MP magnetic ordering strings → integer class
ORDERING_MAP = {"NM": 0, "FM": 1, "AFM": 2, "FiM": 2, "Unknown": 0}

# Map crystal system strings → integer class (7 classes)
CRYSTAL_SYSTEM_MAP = {
    "cubic":        0,
    "hexagonal":    1,
    "trigonal":     2,
    "tetragonal":   3,
    "orthorhombic": 4,
    "monoclinic":   5,
    "triclinic":    6,
}


def fetch_materials(n_materials: int = 5000,
                    out_dir: str = "data/raw",
                    cache_dir: str = "data/cache") -> list[dict]:
    """
    Pull up to `n_materials` entries from Materials Project.
    Returns a list of dicts with structure + labels.
    Caches raw API results so re-runs are fast.
    """
    from mp_api.client import MPRester

    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "MP_API_KEY not set.\n"
            "Get a free key at https://materialsproject.org/dashboard\n"
            "Then: export MP_API_KEY='your_key_here'"
        )

    cache_path = Path(cache_dir) / f"mp_raw_{n_materials}.json"
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        log.info(f"Loading from cache: {cache_path}")
        with open(cache_path) as f:
            return json.load(f)

    log.info(f"Fetching {n_materials} materials from Materials Project …")
    records = []

    with MPRester(api_key) as mpr:
        # Query summary endpoint — most fields live here
        docs = mpr.materials.summary.search(
            num_elements=(1, 6),           # not too complex
            energy_above_hull=(0, 0.1),    # near-stable materials only
            fields=FIELDS,
            chunk_size=500,
        )

        for doc in tqdm(docs, desc="Fetching", total=n_materials):
            if len(records) >= n_materials:
                break

            cs = getattr(doc, "crystal_system", None)
            if cs is None:
                continue

            # Serialise the pymatgen Structure to a plain dict
            structure_dict = None
            if doc.structure is not None:
                structure_dict = doc.structure.as_dict()

            record = {
                "material_id":             str(doc.material_id),
                "formula":                 doc.formula_pretty,
                "crystal_system":          CRYSTAL_SYSTEM_MAP.get(
                                               cs.value if hasattr(cs, "value") else str(cs),
                                               -1),
                "crystal_system_str":      str(cs),
                "spacegroup_number":       doc.spacegroup_number or 0,
                "band_gap":                doc.band_gap if doc.band_gap is not None else -1.0,
                "formation_energy":        (doc.formation_energy_per_atom
                                            if doc.formation_energy_per_atom is not None
                                            else 0.0),
                "magnetic_ordering":       ORDERING_MAP.get(
                                               str(getattr(doc, "ordering", "NM")), 0),
                "structure":               structure_dict,
            }
            records.append(record)

    # Save to cache
    with open(cache_path, "w") as f:
        json.dump(records, f)
    log.info(f"Fetched {len(records)} materials → cached at {cache_path}")
    return records


def save_raw_json(records: list[dict], out_dir: str = "data/raw"):
    """Save one JSON file per material (useful for debugging)."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for rec in records:
        out_path = Path(out_dir) / f"{rec['material_id']}.json"
        with open(out_path, "w") as f:
            json.dump(rec, f)
    log.info(f"Saved {len(records)} JSON files → {out_dir}/")


# ── CLI ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_materials", type=int, default=5000)
    parser.add_argument("--out", default="data/raw")
    parser.add_argument("--cache", default="data/cache")
    args = parser.parse_args()

    records = fetch_materials(args.n_materials, args.out, args.cache)
    log.info(f"Done. {len(records)} materials ready.")
