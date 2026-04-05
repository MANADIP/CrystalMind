"""
src/models/multitask.py
CrystalMind multi-task model.

One shared backbone → multiple task-specific heads:
  • crystal_system     — classification (7 classes)
  • space_group        — classification (230 classes)
  • band_gap           — regression (eV)
  • formation_energy   — regression (eV/atom)
  • magnetic_ordering  — classification (3 classes)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.backbone import ResNet1DBackbone


def _cls_head(in_dim: int, n_classes: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 256),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(256, n_classes),
    )


def _reg_head(in_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, 128),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(128, 1),
    )


class CrystalMind(nn.Module):
    """
    Full multi-task model.

    forward() returns a dict:
        {
          "crystal_system":    (B, 7)   logits
          "space_group":       (B, 230) logits
          "band_gap":          (B,)     predicted eV
          "formation_energy":  (B,)     predicted eV/atom
          "magnetic_ordering": (B, 3)   logits
        }
    Only keys present in `task_cfg` are computed.
    """

    def __init__(self, cfg: dict):
        super().__init__()
        task_cfg = cfg["tasks"]
        dropout  = cfg["model"]["dropout"]

        self.backbone = ResNet1DBackbone(cfg)
        dim = self.backbone.out_dim

        self.heads = nn.ModuleDict()
        for name, tcfg in task_cfg.items():
            if tcfg["type"] == "classification":
                self.heads[name] = _cls_head(dim, tcfg["n_classes"], dropout)
            else:
                self.heads[name] = _reg_head(dim, dropout)

    def forward(self, x: torch.Tensor) -> dict:
        emb = self.backbone(x)
        out = {}
        for name, head in self.heads.items():
            pred = head(emb)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)   # regression → scalar
            out[name] = pred
        return out

    def forward_with_features(self, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """Return (predictions_dict, layer4_feature_maps) for Grad-CAM."""
        emb, feat = self.backbone.forward_with_features(x)
        out = {}
        for name, head in self.heads.items():
            pred = head(emb)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            out[name] = pred
        return out, feat

    def predict_with_uncertainty(
        self, x: torch.Tensor, n_passes: int = 30
    ) -> dict:
        """
        Monte Carlo Dropout uncertainty estimation.
        Enables dropout at inference and runs `n_passes` forward passes.

        Returns dict with:
            mean_<task>  : mean prediction
            std_<task>   : standard deviation (uncertainty)
        """
        self.train()   # activate dropout
        with torch.no_grad():
            all_preds = {name: [] for name in self.heads}
            for _ in range(n_passes):
                out = self.forward(x)
                for name, pred in out.items():
                    if pred.dim() > 1:
                        pred = pred.softmax(dim=-1)
                    all_preds[name].append(pred.cpu())

        self.eval()
        result = {}
        for name, preds in all_preds.items():
            stacked = torch.stack(preds, dim=0)  # (n_passes, B, ...)
            result[f"mean_{name}"] = stacked.mean(0)
            result[f"std_{name}"]  = stacked.std(0)
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
