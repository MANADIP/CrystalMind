"""
src/utils/io.py
Config loading, checkpoint save/load, path helpers.
"""

import os
import yaml
import torch
from pathlib import Path


def load_config(path: str = "configs/default.yaml") -> dict:
    """Load YAML config and return as nested dict."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


def save_checkpoint(
    model,
    optimiser,
    epoch: int,
    metrics: dict,
    path: str,
    criterion=None,
):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimiser": optimiser.state_dict(),
        "metrics": metrics,
    }
    if criterion is not None:
        payload["criterion"] = criterion.state_dict()
    torch.save(payload, path)
    print(f"[checkpoint] saved -> {path}  (epoch {epoch})")


def load_checkpoint(model, path: str, optimiser=None, criterion=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimiser is not None:
        optimiser.load_state_dict(ckpt["optimiser"])
    if criterion is not None and "criterion" in ckpt:
        criterion.load_state_dict(ckpt["criterion"])
    print(f"[checkpoint] loaded <- {path}  (epoch {ckpt['epoch']})")
    return ckpt["epoch"], ckpt.get("metrics", {})


def ensure_dirs(cfg: dict):
    """Create all output directories declared in config."""
    for key in ("raw_data", "cache", "plots", "reports"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["checkpoint"]).parent.mkdir(parents=True, exist_ok=True)
    Path(cfg["paths"]["processed"]).parent.mkdir(parents=True, exist_ok=True)
