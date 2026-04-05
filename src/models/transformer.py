"""
src/models/transformer.py
1D Transformer encoder for XRD crystal classification.

Designed to be lightweight enough for a 4 GB GTX GPU with 8 GB RAM.
Default config: d_model=128, 4 heads, 4 layers  ->  ~500 K params.

Architecture
------------
1. Patch embedding: Conv1d(1, d_model, patch_size, stride=patch_size)
   - Turns the 2000-point signal into ~125 tokens
2. Learnable positional encoding (captures 2theta position information)
3. Prepended [CLS] token
4. N x TransformerEncoderLayer (multi-head self-attention + FFN)
5. [CLS] output -> task-specific heads

Input : (B, 1, 2000)
Output: dict of task predictions  (same API as CrystalMind)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEmbedding1D(nn.Module):
    """Convert a 1D signal into a sequence of patch embeddings."""

    def __init__(self, in_channels: int = 1, d_model: int = 128,
                 patch_size: int = 16):
        super().__init__()
        self.proj = nn.Conv1d(in_channels, d_model,
                              kernel_size=patch_size, stride=patch_size,
                              bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 1, L) -> (B, n_patches, d_model)"""
        x = self.proj(x)           # (B, d_model, n_patches)
        return x.transpose(1, 2)   # (B, n_patches, d_model)


class XRDTransformer(nn.Module):
    """
    1D Vision-Transformer-style encoder for XRD patterns.

    Config keys (under model:):
        d_model       : int   (default 128)
        n_heads       : int   (default 4)
        n_layers      : int   (default 4)
        d_ff          : int   (default 256)
        patch_size    : int   (default 16)
        dropout       : float (default 0.3)
    """

    def __init__(self, cfg: dict):
        super().__init__()
        mcfg     = cfg["model"]
        task_cfg = cfg["tasks"]

        d_model    = mcfg.get("d_model", 128)
        n_heads    = mcfg.get("n_heads", 4)
        n_layers   = mcfg.get("n_layers", 4)
        d_ff       = mcfg.get("d_ff", 256)
        patch_size = mcfg.get("patch_size", 16)
        dropout    = mcfg.get("dropout", 0.3)
        n_points   = cfg["data"].get("xrd_n_points", 2000)

        n_patches = n_points // patch_size   # 2000 / 16 = 125

        # ── Patch embedding ───────────────────────────────────────────
        self.patch_embed = PatchEmbedding1D(1, d_model, patch_size)

        # ── CLS token + positional encoding ───────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.pos_embed = nn.Parameter(
            torch.randn(1, n_patches + 1, d_model) * 0.02
        )
        self.pos_drop = nn.Dropout(dropout)

        # ── Transformer encoder ───────────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,       # Pre-LN for training stability
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers
        )
        self.norm = nn.LayerNorm(d_model)

        # ── Task heads ────────────────────────────────────────────────
        self.out_dim = d_model
        self.heads = nn.ModuleDict()
        for name, tcfg in task_cfg.items():
            if tcfg["type"] == "classification":
                self.heads[name] = nn.Sequential(
                    nn.Linear(d_model, d_model),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model, tcfg["n_classes"]),
                )
            else:
                self.heads[name] = nn.Sequential(
                    nn.Linear(d_model, d_model // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(d_model // 2, 1),
                )

        self._init_weights()

    # ── Weight initialisation ─────────────────────────────────────────

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="linear")
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    # ── Forward ───────────────────────────────────────────────────────

    def _encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (cls_embedding, full_sequence) for head + explainability use.
        x: (B, 1, 2000)
        """
        B = x.size(0)

        # Patch embed
        tokens = self.patch_embed(x)                   # (B, n_patches, d)

        # Prepend CLS
        cls = self.cls_token.expand(B, -1, -1)         # (B, 1, d)
        tokens = torch.cat([cls, tokens], dim=1)       # (B, n_patches+1, d)

        # Add positional encoding
        tokens = self.pos_drop(tokens + self.pos_embed)

        # Transformer
        encoded = self.encoder(tokens)                 # (B, n_patches+1, d)
        encoded = self.norm(encoded)

        cls_out = encoded[:, 0]                        # (B, d)
        return cls_out, encoded

    def forward(self, x: torch.Tensor) -> dict:
        cls_out, _ = self._encode(x)

        out = {}
        for name, head in self.heads.items():
            pred = head(cls_out)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            out[name] = pred
        return out

    def forward_with_features(self, x: torch.Tensor) -> tuple[dict, torch.Tensor]:
        """
        Return (predictions, patch_features) for attention visualisation.
        patch_features: (B, n_patches+1, d_model)
        """
        cls_out, encoded = self._encode(x)

        out = {}
        for name, head in self.heads.items():
            pred = head(cls_out)
            if pred.shape[-1] == 1:
                pred = pred.squeeze(-1)
            out[name] = pred

        # Return the full encoded sequence (including CLS) as 'features'
        # Rearrange to (B, d_model, seq_len) to match CNN feature map convention
        feat = encoded.transpose(1, 2)     # (B, d, n_patches+1)
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
