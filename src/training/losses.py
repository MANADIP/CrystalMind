"""
src/training/losses.py
Combined multi-task loss.

Each task contributes a weighted loss:
  Classification → CrossEntropyLoss (with label smoothing)
  Regression     → HuberLoss        (robust to outliers)

Total loss = Σ  weight_i * loss_i
"""

import torch
import torch.nn as nn


class MultiTaskLoss(nn.Module):
    """
    Parameters
    ----------
    task_cfg : dict  (from configs/default.yaml -> tasks)
        {
          "crystal_system":   {"type": "classification", "n_classes": 7,  "weight": 1.0},
          "band_gap":         {"type": "regression",                       "weight": 0.3},
          ...
        }
    label_smoothing : float  applied to all classification heads
    """

    def __init__(self, task_cfg: dict, label_smoothing: float = 0.05):
        super().__init__()
        self.task_cfg = task_cfg

        self.cls_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.reg_loss = nn.HuberLoss(delta=1.0)

        # Register weights as buffers so they move with .to(device)
        for name, tcfg in task_cfg.items():
            self.register_buffer(
                f"w_{name}",
                torch.tensor(tcfg.get("weight", 1.0), dtype=torch.float)
            )

    def forward(self, predictions: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        predictions : dict  task_name -> tensor
        targets     : dict  task_name -> tensor

        Returns
        -------
        total_loss   : scalar tensor
        per_task_loss: dict  task_name -> float  (for logging)
        """
        total      = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        per_task   = {}

        for name, tcfg in self.task_cfg.items():
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            tgt  = targets[name]
            w    = getattr(self, f"w_{name}")

            if tcfg["type"] == "classification":
                # Skip samples with invalid label (-1)
                valid = tgt >= 0
                if valid.sum() == 0:
                    continue
                loss = self.cls_loss(pred[valid], tgt[valid])
            else:
                # Regression: skip samples flagged with -1.0
                valid = tgt > -0.5
                if valid.sum() == 0:
                    continue
                loss = self.reg_loss(pred[valid].float(), tgt[valid].float())

            weighted = w * loss
            total    = total + weighted
            per_task[name] = loss.item()

        return total, per_task
