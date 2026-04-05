"""
src/models/backbone.py
Shared 1D ResNet encoder with an optional transformer sequence block.
"""

import torch
import torch.nn as nn

from src.models.blocks import ResidualBlockSE1D, TransformerEncoder1D


class ResNet1DBackbone(nn.Module):
    """
    5-stage 1D ResNet with Squeeze-and-Excitation residual blocks.
    Optionally inserts a lightweight transformer encoder before pooling.

    Stage  | Out ch | Kernel | Stride | Dilation | Length (from 2000)
    -------+--------+--------+--------+----------+-------------------
    Stem   |     32 |     15 |      2 |        1 | 1000 -> 500 (MaxPool)
    Layer1 |     64 |      7 |      2 |        1 | 250
    Layer2 |    128 |      7 |      2 |        1 | 125
    Layer3 |    256 |      7 |      2 |        1 | 63
    Layer4 |    256 |      7 |      1 |        2 | 63
    Transf |    256 |      - |      - |        - | 63
    GAP    |    256 |      - |      - |        - | 1
    """

    def __init__(self, cfg: dict = None, dropout: float = 0.1):
        super().__init__()
        model_cfg = cfg.get("model", {}) if cfg is not None else {}

        if cfg is not None:
            ch = model_cfg["channels"]
            k = model_cfg["kernels"]
            s = model_cfg["strides"]
            d = model_cfg["dilations"]
            n = model_cfg["residual_blocks_per_stage"]
            drop = model_cfg["dropout"]
        else:
            ch = [32, 64, 128, 256, 256]
            k = [15, 7, 7, 7, 7]
            s = [2, 2, 2, 2, 1]
            d = [1, 1, 1, 1, 2]
            n = 2
            drop = dropout

        self.stem = nn.Sequential(
            nn.Conv1d(1, ch[0], k[0], stride=s[0], padding=k[0] // 2, bias=False),
            nn.BatchNorm1d(ch[0]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )

        self.layer1 = self._make_stage(ch[0], ch[1], k[1], s[1], d[1], n, drop)
        self.layer2 = self._make_stage(ch[1], ch[2], k[2], s[2], d[2], n, drop)
        self.layer3 = self._make_stage(ch[2], ch[3], k[3], s[3], d[3], n, drop)
        self.layer4 = self._make_stage(ch[3], ch[4], k[4], s[4], d[4], n, drop)

        self.sequence_encoder = None
        transformer_cfg = model_cfg.get("transformer", {})
        if transformer_cfg.get("enabled", False):
            self.sequence_encoder = TransformerEncoder1D(
                embed_dim=ch[4],
                num_heads=transformer_cfg.get("n_heads", 4),
                num_layers=transformer_cfg.get("n_layers", 1),
                ff_multiplier=transformer_cfg.get("ff_multiplier", 2.0),
                dropout=transformer_cfg.get("dropout", drop),
                max_seq_len=transformer_cfg.get("max_seq_len", 256),
            )

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.out_dim = ch[4]

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, kernel, stride, dilation, n_blocks, drop):
        blocks = [
            ResidualBlockSE1D(
                in_ch,
                out_ch,
                kernel=kernel,
                stride=stride,
                dilation=dilation,
                dropout=drop,
            )
        ]
        for _ in range(1, n_blocks):
            blocks.append(
                ResidualBlockSE1D(
                    out_ch,
                    out_ch,
                    kernel=kernel,
                    dilation=dilation,
                    dropout=drop,
                )
            )
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self.sequence_encoder is not None:
            x = self.sequence_encoder(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._encode_sequence(x)
        x = self.gap(x).squeeze(-1)
        return x

    def forward_with_features(self, x: torch.Tensor):
        """Return (embedding, final_feature_maps) for Grad-CAM / inspection."""
        feat = self._encode_sequence(x)
        emb = self.gap(feat).squeeze(-1)
        return emb, feat
