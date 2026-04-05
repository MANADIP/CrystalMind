"""
scripts/04_evaluate.py
Step 4 — Evaluate the trained model on the test set.

Produces:
  - Classification report for each classification task
  - Confusion matrix PNG (crystal system)
  - Saves text report to outputs/reports/
  - MC Dropout uncertainty summary

Usage:
    python scripts/04_evaluate.py
    python scripts/04_evaluate.py --config configs/baseline_cnn.yaml
    python scripts/04_evaluate.py --config configs/transformer.yaml
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")       # non-interactive backend for scripts
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry  import build_model
from src.data.dataset      import build_dataloaders
from src.utils.io          import load_config, load_checkpoint
from src.utils.metrics     import cls_metrics, reg_metrics, print_report


CRYSTAL_SYSTEM_NAMES = [
    "Cubic", "Hexagonal", "Trigonal", "Tetragonal",
    "Orthorhombic", "Monoclinic", "Triclinic",
]


def evaluate(model, loader, task_cfg, device):
    model.eval()
    all_preds   = {t: [] for t in task_cfg}
    all_targets = {t: [] for t in task_cfg}

    with torch.no_grad():
        for xb, yb in loader:
            xb   = xb.to(device)
            preds = model(xb)
            for t in task_cfg:
                if t not in preds:
                    continue
                p = preds[t]
                if p.dim() > 1:
                    all_preds[t].append(p.argmax(1).cpu().numpy())
                else:
                    all_preds[t].append(p.cpu().numpy())
                all_targets[t].append(yb[t].numpy())

    return (
        {t: np.concatenate(v) for t, v in all_preds.items()   if v},
        {t: np.concatenate(v) for t, v in all_targets.items() if v},
    )


def plot_confusion_matrix(y_true, y_pred, labels, save_path):
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 8))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues",
                xticklabels=labels, yticklabels=labels, ax=ax,
                linewidths=0.5, cbar_kws={"label": "Fraction"})
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Crystal system — normalised confusion matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"[eval] Confusion matrix -> {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluate CrystalMind")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--ckpt",   default=None,
                        help="Override checkpoint path")
    args = parser.parse_args()

    cfg    = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt   = args.ckpt or cfg["paths"]["checkpoint"]
    report_dir = Path(cfg["paths"]["reports"])
    plot_dir   = Path(cfg["paths"]["plots"])
    report_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    model_name = cfg["model"]["name"]
    print("=" * 55)
    print(f"  CrystalMind — Step 4: Evaluation [{model_name}]")
    print("=" * 55)

    # ── Load model (via registry) ─────────────────────────────────────
    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt, device=device)

    # ── Data ─────────────────────────────────────────────────────────
    _, _, test_loader = build_dataloaders(cfg["paths"]["processed"], cfg)

    # ── Inference ────────────────────────────────────────────────────
    preds, targets = evaluate(model, test_loader, cfg["tasks"], device)

    # ── Report ───────────────────────────────────────────────────────
    lines = [f"Model: {model_name}", "=" * 40]
    for task, tcfg in cfg["tasks"].items():
        if task not in preds:
            continue
        if tcfg["type"] == "classification":
            m = cls_metrics(targets[task], preds[task])
            line = f"{task:20s}  acc={m['accuracy']:.4f}  f1={m['f1']:.4f}"
            print(line)
            lines.append(line)
            print_report(targets[task], preds[task], task)
        else:
            m = reg_metrics(targets[task], preds[task])
            line = f"{task:20s}  MAE={m['mae']:.4f}  R²={m['r2']:.4f}"
            print(line)
            lines.append(line)

    # Save summary
    report_path = report_dir / f"test_metrics_{model_name}.txt"
    report_path.write_text("\n".join(lines))
    print(f"\n[eval] Report -> {report_path}")

    # ── Confusion matrix ─────────────────────────────────────────────
    if "crystal_system" in preds:
        plot_confusion_matrix(
            targets["crystal_system"],
            preds["crystal_system"],
            labels=CRYSTAL_SYSTEM_NAMES,
            save_path=str(plot_dir / f"confusion_{model_name}.png"),
        )

    # ── MC Dropout uncertainty (sample) ──────────────────────────────
    print("\nRunning MC Dropout uncertainty estimation (30 passes) …")
    sample_x = next(iter(test_loader))[0][:8].to(device)
    unc = model.predict_with_uncertainty(sample_x, n_passes=30)
    if "std_crystal_system" in unc:
        cs_std = unc["std_crystal_system"].mean(0).numpy()
        print(f"  Mean prediction uncertainty (crystal_system): "
              f"{cs_std.mean():.4f}  (lower = more confident)")

    print(f"\n[OK]  Evaluation complete.")
    print(f"   Next step: python scripts/05_explain.py --config {args.config}")


if __name__ == "__main__":
    main()
