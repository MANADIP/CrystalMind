"""
scripts/06_compare_models.py
Step 6 — Compare all trained models side-by-side on the test set.

Loads checkpoints for each model variant, evaluates on the same test split,
and outputs a comparison table (markdown + CSV).

Usage:
    python scripts/06_compare_models.py
"""

import sys
import argparse
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry  import build_model
from src.data.dataset      import build_dataloaders
from src.utils.io          import load_config, load_checkpoint
from src.utils.metrics     import cls_metrics


# ── Model configs to compare ─────────────────────────────────────────

MODEL_CONFIGS = [
    ("BaselineCNN",         "configs/baseline_cnn.yaml"),
    ("CrystalMind1DResNet", "configs/default.yaml"),
    ("XRDTransformer",      "configs/transformer.yaml"),
]


def evaluate_model(cfg, device):
    """Load model + checkpoint, run test set, return metrics dict."""
    model_name = cfg["model"]["name"]
    ckpt_path  = cfg["paths"]["checkpoint"]

    if not Path(ckpt_path).exists():
        print(f"  [skip] {model_name}: no checkpoint at {ckpt_path}")
        return None

    model = build_model(cfg).to(device)
    load_checkpoint(model, ckpt_path, device=device)
    model.eval()

    _, _, test_loader = build_dataloaders(cfg["paths"]["processed"], cfg)

    all_preds   = {t: [] for t in cfg["tasks"]}
    all_targets = {t: [] for t in cfg["tasks"]}

    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            preds = model(xb)
            for t in cfg["tasks"]:
                if t not in preds:
                    continue
                p = preds[t]
                if p.dim() > 1:
                    all_preds[t].append(p.argmax(1).cpu().numpy())
                else:
                    all_preds[t].append(p.cpu().numpy())
                all_targets[t].append(yb[t].numpy())

    metrics = {"model": model_name, "params": model.count_parameters()}
    for t, tcfg in cfg["tasks"].items():
        if t not in all_preds or not all_preds[t]:
            continue
        y_true = np.concatenate(all_targets[t])
        y_pred = np.concatenate(all_preds[t])
        if tcfg["type"] == "classification":
            m = cls_metrics(y_true, y_pred)
            metrics[f"acc_{t}"]  = m["accuracy"]
            metrics[f"f1_{t}"]   = m["f1"]

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Compare CrystalMind models")
    parser.add_argument("--output", default="outputs/reports/model_comparison.md")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  CrystalMind — Model Comparison")
    print("=" * 60)

    results = []
    for name, config_path in MODEL_CONFIGS:
        if not Path(config_path).exists():
            print(f"  [skip] Config not found: {config_path}")
            continue
        print(f"\n--- Evaluating: {name} ---")
        cfg = load_config(config_path)
        metrics = evaluate_model(cfg, device)
        if metrics:
            results.append(metrics)

    if not results:
        print("\n[WARNING]  No trained models found. Train models first:")
        print("   python scripts/03_train.py --config configs/baseline_cnn.yaml")
        print("   python scripts/03_train.py --config configs/default.yaml")
        print("   python scripts/03_train.py --config configs/transformer.yaml")
        return

    # ── Build comparison table ────────────────────────────────────────
    lines = [
        "# CrystalMind — Model Comparison",
        "",
        "| Model | Params | Crystal System Acc | Crystal System F1 | Space Group Acc | Space Group F1 |",
        "|---|---|---|---|---|---|",
    ]

    for r in results:
        row = (
            f"| {r['model']} "
            f"| {r['params']:,} "
            f"| {r.get('acc_crystal_system', 0):.4f} "
            f"| {r.get('f1_crystal_system', 0):.4f} "
            f"| {r.get('acc_space_group', 0):.4f} "
            f"| {r.get('f1_space_group', 0):.4f} |"
        )
        lines.append(row)

    # Print to console
    table_md = "\n".join(lines)
    print(f"\n{table_md}")

    # Save markdown
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(table_md)
    print(f"\n[compare] Saved -> {out_path}")

    # ── Bar chart ─────────────────────────────────────────────────────
    model_names = [r["model"] for r in results]
    cs_accs     = [r.get("acc_crystal_system", 0) for r in results]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors  = ["#4FC3F7", "#E8593C", "#66BB6A"][:len(model_names)]
    bars    = ax.bar(model_names, cs_accs, color=colors, edgecolor="0.3",
                     linewidth=0.8)

    for bar, acc in zip(bars, cs_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{acc:.2%}", ha="center", va="bottom", fontsize=12,
                fontweight="bold")

    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("Crystal System Classification — Model Comparison",
                 fontsize=14, pad=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plot_path = out_path.parent.parent / "plots" / "model_comparison.png"
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"[compare] Plot  -> {plot_path}")
    plt.close()


if __name__ == "__main__":
    main()
