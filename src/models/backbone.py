"""
src/models/backbone.py
Shared 1D ResNet encoder.
Input : (B, 1, 2000)
Output: (B, 256)   — global-average-pooled feature vector
"""

import torch
import torch.nn as nn
from src.models.blocks import ResidualBlockSE1D


class ResNet1DBackbone(nn.Module):
    """
    5-stage 1D ResNet with Squeeze-and-Excitation residual blocks.

    Stage  | Out ch | Kernel | Stride | Dilation | Length (from 2000)
    -------+--------+--------+--------+----------+-------------------
    Stem   |     32 |     15 |      2 |        1 | 1000 → 500 (MaxPool)
    Layer1 |     64 |      7 |      2 |        1 | 250
    Layer2 |    128 |      7 |      2 |        1 | 125
    Layer3 |    256 |      7 |      2 |        1 |  63
    Layer4 |    256 |      7 |      1 |        2 |  63  (dilated — larger RF)
    GAP    |    256 |      –  |      – |        – |  1
    """

    def __init__(self, cfg: dict = None, dropout: float = 0.1):
        super().__init__()

        # Allow overriding via config
        if cfg is not None:
            ch   = cfg["model"]["channels"]     # e.g. [32, 64, 128, 256, 256]
            k    = cfg["model"]["kernels"]
            s    = cfg["model"]["strides"]
            d    = cfg["model"]["dilations"]
            n    = cfg["model"]["residual_blocks_per_stage"]
            drop = cfg["model"]["dropout"]
        else:
            ch   = [32, 64, 128, 256, 256]
            k    = [15,  7,   7,   7,   7]
            s    = [ 2,  2,   2,   2,   1]
            d    = [ 1,  1,   1,   1,   2]
            n    = 2
            drop = dropout

        self.stem = nn.Sequential(
            nn.Conv1d(1, ch[0], k[0], stride=s[0],
                      padding=k[0]//2, bias=False),
            nn.BatchNorm1d(ch[0]),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
        )

        self.layer1 = self._make_stage(ch[0], ch[1], k[1], s[1], d[1], n, drop)
        self.layer2 = self._make_stage(ch[1], ch[2], k[2], s[2], d[2], n, drop)
        self.layer3 = self._make_stage(ch[2], ch[3], k[3], s[3], d[3], n, drop)
        self.layer4 = self._make_stage(ch[3], ch[4], k[4], s[4], d[4], n, drop)

        self.gap        = nn.AdaptiveAvgPool1d(1)
        self.out_dim    = ch[4]

        self._init_weights()

    def _make_stage(self, in_ch, out_ch, kernel, stride, dilation, n_blocks, drop):
        blocks = [ResidualBlockSE1D(in_ch, out_ch,
                                    kernel=kernel, stride=stride,
                                    dilation=dilation, dropout=drop)]
        for _ in range(1, n_blocks):
            blocks.append(ResidualBlockSE1D(out_ch, out_ch,
                                            kernel=kernel, dilation=dilation,
                                            dropout=drop))
        return nn.Sequential(*blocks)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        # x: (B, 1, 2000)
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)          # (B, 256, ~63)
        x = self.gap(x).squeeze(-1) # (B, 256)
        return x

    def forward_with_features(self, x):
        """Return (embedding, layer4_feature_maps) for Grad-CAM."""
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat = self.layer4(x)                   # (B, 256, L)
        emb  = self.gap(feat).squeeze(-1)       # (B, 256)
        return emb, feat
