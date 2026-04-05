"""
scripts/05_explain.py
Step 5 — Generate Grad-CAM saliency maps for test samples.

Usage:
    python scripts/05_explain.py
    python scripts/05_explain.py --n_samples 20 --task crystal_system
"""

import sys
import argparse
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.multitask      import CrystalMind
from src.data.dataset          import build_dataloaders
from src.explainability.gradcam import GradCAM1D
from src.utils.io              import load_config, load_checkpoint


def main():
    parser = argparse.ArgumentParser(description="Grad-CAM XRD explainability")
    parser.add_argument("--config",    default="configs/default.yaml")
    parser.add_argument("--task",      default="crystal_system")
    parser.add_argument("--n_samples", type=int, default=10)
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    plot_dir = Path(cfg["paths"]["plots"]) / "gradcam"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print("  CrystalMind — Step 5: Grad-CAM Explainability")
    print("=" * 55)
    print(f"  Task     : {args.task}")
    print(f"  Samples  : {args.n_samples}")
    print(f"  Output   : {plot_dir}/")
    print("=" * 55)

    # ── Model ─────────────────────────────────────────────────────────
    model = CrystalMind(cfg).to(device)
    load_checkpoint(model, cfg["paths"]["checkpoint"], device=device)
    model.eval()

    # ── Data ─────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(cfg["paths"]["processed"], cfg)

    # Collect patterns and IDs from first batches
    patterns, labels_cs = [], []
    ds = np.load(cfg["paths"]["processed"], allow_pickle=True)
    mat_ids = ds["material_ids"] if "material_ids" in ds else None

    collected = 0
    for xb, yb in test_loader:
        for i in range(len(xb)):
            if collected >= args.n_samples:
                break
            patterns.append(xb[i, 0].numpy())         # (2000,)
            labels_cs.append(yb["crystal_system"][i].item())
            collected += 1
        if collected >= args.n_samples:
            break

    # ── Grad-CAM ─────────────────────────────────────────────────────
    gradcam = GradCAM1D(model, task=args.task)

    for i, (pattern, cs_label) in enumerate(zip(patterns, labels_cs)):
        x = torch.tensor(pattern).unsqueeze(0).unsqueeze(0).float().to(device)

        try:
            cam, pred_cls = gradcam.compute(x)
            mid = mat_ids[i] if mat_ids is not None else f"sample_{i:04d}"
            gradcam.plot(
                pattern, cam, pred_cls,
                material_id=str(mid),
                save_path=str(plot_dir / f"gradcam_{i:04d}_{mid}.png"),
                two_theta_range=(
                    cfg["data"]["two_theta_min"],
                    cfg["data"]["two_theta_max"],
                ),
            )
        except Exception as e:
            print(f"  [warn] Sample {i} failed: {e}")

    print(f"\n✓  {args.n_samples} Grad-CAM plots saved → {plot_dir}/")
    print(f"\n   Run the web demo: streamlit run app/streamlit_app.py")


if __name__ == "__main__":
    main()
