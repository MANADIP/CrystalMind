"""
src/models/baseline_cnn.py
Simple 3-layer 1D CNN baseline for XRD crystal classification.

This is the ablation baseline — no residual connections, no SE blocks,
no dilation.  Establishes the accuracy floor that advanced models must beat.

Input : (B, 1, 2000)
Output: dict of task predictions  (same API as CrystalMind)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineCNN(nn.Module):
    """
    Minimal 1D CNN for XRD classification.

    Architecture
    ------------
    Conv1d(1->32, k=15, s=2) -> BN -> ReLU -> MaxPool(2)
    Conv1d(32->64, k=7, s=2)  -> BN -> ReLU -> MaxPool(2)
    Conv1d(64->128, k=7, s=1) -> BN -> ReLU
    AdaptiveAvgPool1d(1) -> Flatten -> 128-dim
    -> task-specific heads

    Parameters : ~150 K  (vs ~2 M for 1D-ResNet)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        task_cfg = cfg["tasks"]
        dropout  = cfg["model"].get("dropout", 0.3)

        # ── Convolutional feature extractor ───────────────────────────
        self.features = nn.Sequential(
            # Block 1
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            # Block 2
            nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Dropout(dropout),

            # Block 3
            nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

        self.gap     = nn.AdaptiveAvgPool1d(1)
        self.out_dim = 128

        # ── Task heads ────────────────────────────────────────────────
        self.heads = nn.ModuleDict()
        for name, tcfg in task_cfg.items():
            if tcfg["type"] == "classification":
                self.heads[name] = nn.Sequential(
                    nn.Linear(self.out_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, tcfg["n_classes"]),
                )
            else:
                self.heads[name] = nn.Sequential(
                    nn.Linear(self.out_dim, 64),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(64, 1),
                )

        self._init_weights()

    # ── Weight init ───────────────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> dict:
        """x: (B, 1, 2000) -> dict of predictions"""
        feat = self.features(x)              # (B, 128, L')
        emb  = self.gap(feat).squeeze(-1)    # (B, 128)

        out = {}
        for name, head in self.heads.items():
            pred = head(emb)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            out[name] = pred
        return out

    def forward_with_features(self, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """Return (predictions, last_conv_features) for Grad-CAM."""
        feat = self.features(x)              # (B, 128, L')
        emb  = self.gap(feat).squeeze(-1)    # (B, 128)

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
        """MC Dropout uncertainty estimation."""
        self.train()
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
            stacked = torch.stack(preds, dim=0)
            result[f"mean_{name}"] = stacked.mean(0)
            result[f"std_{name}"]  = stacked.std(0)
        return result

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
