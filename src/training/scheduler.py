"""
src/training/scheduler.py
Learning-rate schedule helpers.
Reads the 'training.scheduler' key from config.
"""

import torch


def build_scheduler(optimiser, cfg: dict):
    """
    Returns a (scheduler, use_warmup) tuple.

    Supported schedulers (set in config training.scheduler):
        cosine  — CosineAnnealingLR  (recommended)
        step    — StepLR  (every 20 epochs, gamma=0.5)
        plateau — ReduceLROnPlateau  (monitors val loss)
    """
    train_cfg = cfg["training"]
    name      = train_cfg.get("scheduler", "cosine")
    epochs    = train_cfg["epochs"]
    warmup    = train_cfg.get("warmup_epochs", 5)

    if name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=epochs - warmup,
            eta_min=1e-6,
        )
    elif name == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimiser, step_size=20, gamma=0.5
        )
    elif name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimiser, mode="min", factor=0.5, patience=5, verbose=True
        )
    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return scheduler, warmup


class LinearWarmup:
    """
    Linearly scales LR from lr/10 → lr over `warmup_epochs` epochs.
    Applied BEFORE handing off to the main scheduler.
    """

    def __init__(self, optimiser, base_lr: float, warmup_epochs: int):
        self.opt           = optimiser
        self.base_lr       = base_lr
        self.warmup_epochs = warmup_epochs

    def step(self, epoch: int):
        if epoch >= self.warmup_epochs:
            return
        scale = (epoch + 1) / self.warmup_epochs
        for pg in self.opt.param_groups:
            pg["lr"] = self.base_lr * scale * 0.1 + self.base_lr * scale * 0.9
