"""
scripts/03_train.py
Step 3 — Train a CrystalMind model.

Supports all registered models — switch via config:
    python scripts/03_train.py                                     # default (ResNet)
    python scripts/03_train.py --config configs/baseline_cnn.yaml  # Baseline CNN
    python scripts/03_train.py --config configs/transformer.yaml   # Transformer

CLI overrides:
    python scripts/03_train.py --epochs 100 --lr 5e-4 --batch 32
"""

import sys
import argparse
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.registry   import build_model
from src.data.dataset      import build_dataloaders
from src.training.trainer  import Trainer
from src.utils.io          import load_config, ensure_dirs


def main():
    parser = argparse.ArgumentParser(description="Train CrystalMind")
    parser.add_argument("--config",  default="configs/default.yaml")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ensure_dirs(cfg)

    # CLI overrides
    if args.epochs: cfg["training"]["epochs"]        = args.epochs
    if args.lr:     cfg["training"]["learning_rate"] = args.lr
    if args.batch:  cfg["training"]["batch_size"]    = args.batch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg["model"]["name"]
    print("=" * 55)
    print(f"  CrystalMind — Step 3: Training [{model_name}]")
    print("=" * 55)
    print(f"  Device  : {device}")
    print(f"  Epochs  : {cfg['training']['epochs']}")
    print(f"  LR      : {cfg['training']['learning_rate']}")
    print(f"  Batch   : {cfg['training']['batch_size']}")
    print(f"  Tasks   : {list(cfg['tasks'].keys())}")
    print("=" * 55)

    # ── Data ──────────────────────────────────────────────────────────
    train_loader, val_loader, _ = build_dataloaders(
        cfg["paths"]["processed"], cfg
    )

    # ── Model (from registry) ────────────────────────────────────────
    model = build_model(cfg)
    print(f"  Model    : {model_name}")
    print(f"  Params   : {model.count_parameters():,}")

    # ── Train ─────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg, device)
    best_acc = trainer.fit(train_loader, val_loader)

    print(f"\n[OK]  Training complete.")
    print(f"   Best val acc (crystal_system): {best_acc:.4f}")
    print(f"   Checkpoint -> {cfg['paths']['checkpoint']}")
    print(f"\n   Next step: python scripts/04_evaluate.py --config {args.config}")


if __name__ == "__main__":
    main()
