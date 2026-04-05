"""
scripts/01_fetch_data.py
Step 1 — Fetch crystal structures from the Materials Project API.

Usage:
    python scripts/01_fetch_data.py
    python scripts/01_fetch_data.py --n_materials 3000

Requires:
    export MP_API_KEY="your_key_here"
    (Get a free key at https://materialsproject.org/dashboard)
"""

import argparse
import sys
from pathlib import Path

# Make sure src/ is importable from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.fetch  import fetch_materials
from src.utils.io    import load_config, ensure_dirs


def main():
    parser = argparse.ArgumentParser(
        description="Download crystal structures from Materials Project"
    )
    parser.add_argument(
        "--n_materials", type=int, default=None,
        help="Number of materials to fetch (default: from config)"
    )
    parser.add_argument(
        "--config", default="configs/default.yaml",
        help="Path to config YAML"
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    n = args.n_materials or cfg["data"]["n_materials"]

    print("=" * 55)
    print("  CrystalMind — Step 1: Fetch Materials Project data")
    print("=" * 55)
    print(f"  Target : {n} materials")
    print(f"  Cache  : {cfg['paths']['cache']}/")
    print("=" * 55)

    records = fetch_materials(
        n_materials=n,
        out_dir=cfg["paths"]["raw_data"],
        cache_dir=cfg["paths"]["cache"],
    )

    print(f"\n✓  {len(records)} materials fetched and cached.")
    print(f"   Next step: python scripts/02_build_dataset.py")


if __name__ == "__main__":
    main()
