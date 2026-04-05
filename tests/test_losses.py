"""
tests/test_losses.py
Sanity checks for focal loss and uncertainty-weighted multi-task loss.
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.training.losses import FocalLoss, MultiTaskLoss


TASK_CFG = {
    "crystal_system": {
        "type": "classification",
        "n_classes": 7,
        "weight": 1.0,
        "loss": {"type": "cross_entropy"},
    },
    "space_group": {
        "type": "classification",
        "n_classes": 230,
        "weight": 1.0,
        "loss": {"type": "focal", "gamma": 2.0},
    },
}


def test_focal_loss_backward():
    loss_fn = FocalLoss(gamma=2.0, label_smoothing=0.05)
    logits = torch.randn(8, 230, requires_grad=True)
    target = torch.randint(0, 230, (8,))

    loss = loss_fn(logits, target)
    loss.backward()

    assert loss.item() > 0
    assert logits.grad is not None


def test_multitask_loss_static_mode():
    criterion = MultiTaskLoss(TASK_CFG, label_smoothing=0.05)
    preds = {
        "crystal_system": torch.randn(4, 7, requires_grad=True),
        "space_group": torch.randn(4, 230, requires_grad=True),
    }
    targets = {
        "crystal_system": torch.randint(0, 7, (4,)),
        "space_group": torch.randint(0, 230, (4,)),
    }

    total, per_task = criterion(preds, targets)
    total.backward()

    assert total.item() > 0
    assert set(per_task) == {"crystal_system", "space_group"}


def test_multitask_loss_uncertainty_mode_learns_log_vars():
    criterion = MultiTaskLoss(
        TASK_CFG,
        label_smoothing=0.05,
        multi_task_weighting="uncertainty",
    )
    preds = {
        "crystal_system": torch.randn(4, 7, requires_grad=True),
        "space_group": torch.randn(4, 230, requires_grad=True),
    }
    targets = {
        "crystal_system": torch.randint(0, 7, (4,)),
        "space_group": torch.randint(0, 230, (4,)),
    }

    total, _ = criterion(preds, targets)
    total.backward()
    metrics = criterion.get_weight_metrics()

    assert criterion.task_log_vars["crystal_system"].grad is not None
    assert criterion.task_log_vars["space_group"].grad is not None
    assert "log_var_crystal_system" in metrics
    assert "effective_weight_space_group" in metrics
