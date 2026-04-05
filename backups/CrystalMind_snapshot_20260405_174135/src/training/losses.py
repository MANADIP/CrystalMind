"""
src/training/losses.py
Combined multi-task loss with configurable classification losses and
optional uncertainty-based task weighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Multi-class focal loss for hard and imbalanced classification tasks."""

    def __init__(
        self,
        gamma: float = 2.0,
        alpha=None,
        reduction: str = "mean",
        label_smoothing: float = 0.0,
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        self.alpha_scalar = None
        self.register_buffer("alpha_vector", torch.tensor([], dtype=torch.float))

        if alpha is None:
            return
        if isinstance(alpha, (list, tuple)):
            self.alpha_vector = torch.tensor(alpha, dtype=torch.float)
        elif torch.is_tensor(alpha) and alpha.dim() > 0:
            self.alpha_vector = alpha.detach().float()
        else:
            self.alpha_scalar = float(alpha)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        ce = F.cross_entropy(
            logits,
            target,
            reduction="none",
            label_smoothing=self.label_smoothing,
        )
        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, target.unsqueeze(1)).squeeze(1).clamp_min(1e-8)
        loss = ((1.0 - pt) ** self.gamma) * ce

        if self.alpha_vector.numel() > 0:
            alpha = self.alpha_vector.to(logits.device)
            loss = loss * alpha[target]
        elif self.alpha_scalar is not None:
            loss = loss * self.alpha_scalar

        if self.reduction == "sum":
            return loss.sum()
        if self.reduction == "none":
            return loss
        return loss.mean()


class MultiTaskLoss(nn.Module):
    """
    Parameters
    ----------
    task_cfg : dict
        Mapping of task name to task metadata.
    label_smoothing : float
        Default label smoothing for classification heads.
    multi_task_weighting : str
        Either "static" or "uncertainty".
    uncertainty_init_log_var : float
        Initial log variance for Kendall-style uncertainty weighting.
    """

    def __init__(
        self,
        task_cfg: dict,
        label_smoothing: float = 0.05,
        multi_task_weighting: str = "static",
        uncertainty_init_log_var: float = 0.0,
    ):
        super().__init__()
        self.task_cfg = task_cfg
        self.multi_task_weighting = multi_task_weighting

        self.cls_losses = nn.ModuleDict()
        self.reg_loss = nn.HuberLoss(delta=1.0)

        for name, tcfg in task_cfg.items():
            self.register_buffer(
                f"w_{name}",
                torch.tensor(tcfg.get("weight", 1.0), dtype=torch.float),
            )

            if tcfg["type"] != "classification":
                continue

            loss_cfg = tcfg.get("loss", {})
            if isinstance(loss_cfg, str):
                loss_cfg = {"type": loss_cfg}
            loss_name = loss_cfg.get("type", "cross_entropy").lower()
            task_label_smoothing = float(loss_cfg.get("label_smoothing", label_smoothing))

            if loss_name == "focal":
                self.cls_losses[name] = FocalLoss(
                    gamma=float(loss_cfg.get("gamma", 2.0)),
                    alpha=loss_cfg.get("alpha"),
                    reduction="mean",
                    label_smoothing=task_label_smoothing,
                )
            elif loss_name == "cross_entropy":
                self.cls_losses[name] = nn.CrossEntropyLoss(
                    label_smoothing=task_label_smoothing
                )
            else:
                raise ValueError(
                    f"Unsupported classification loss '{loss_name}' for task '{name}'."
                )

        if self.multi_task_weighting == "uncertainty":
            self.task_log_vars = nn.ParameterDict(
                {
                    name: nn.Parameter(
                        torch.tensor(
                            float(
                                task_cfg[name].get(
                                    "log_var_init", uncertainty_init_log_var
                                )
                            )
                        )
                    )
                    for name in task_cfg
                }
            )
        elif self.multi_task_weighting == "static":
            self.task_log_vars = None
        else:
            raise ValueError(
                "multi_task_weighting must be either 'static' or 'uncertainty'."
            )

    def forward(self, predictions: dict, targets: dict) -> tuple[torch.Tensor, dict]:
        """
        Parameters
        ----------
        predictions : dict
            task_name -> tensor
        targets : dict
            task_name -> tensor

        Returns
        -------
        total_loss : scalar tensor
        per_task_loss : dict
            task_name -> raw loss value for logging
        """
        device = next(iter(predictions.values())).device
        total = None
        per_task = {}

        for name, tcfg in self.task_cfg.items():
            if name not in predictions or name not in targets:
                continue

            pred = predictions[name]
            tgt = targets[name]
            base_weight = getattr(self, f"w_{name}")

            if tcfg["type"] == "classification":
                valid = tgt >= 0
                if valid.sum().item() == 0:
                    continue
                loss = self.cls_losses[name](pred[valid], tgt[valid])
            else:
                valid = tgt > -0.5
                if valid.sum().item() == 0:
                    continue
                loss = self.reg_loss(pred[valid].float(), tgt[valid].float())

            weighted_loss = base_weight * loss
            if self.task_log_vars is not None:
                log_var = self.task_log_vars[name]
                weighted_loss = torch.exp(-log_var) * weighted_loss + log_var

            total = weighted_loss if total is None else total + weighted_loss
            per_task[name] = float(loss.detach().item())

        if total is None:
            total = torch.zeros((), device=device, requires_grad=True)

        return total, per_task

    def get_weight_metrics(self) -> dict:
        """Return scalar task-weight diagnostics for logging."""
        metrics = {}
        for name in self.task_cfg:
            base_weight = float(getattr(self, f"w_{name}").detach().cpu())
            metrics[f"base_weight_{name}"] = base_weight
            if self.task_log_vars is not None:
                log_var = self.task_log_vars[name].detach().cpu()
                metrics[f"log_var_{name}"] = float(log_var)
                metrics[f"effective_weight_{name}"] = float(
                    base_weight * torch.exp(-log_var)
                )
        return metrics
