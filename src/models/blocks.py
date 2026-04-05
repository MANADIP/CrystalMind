"""
src/models/blocks.py
Building blocks for the 1D ResNet backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock1D(nn.Module):
    """
    Standard residual block for 1D signals.
        Conv1d → BN → ReLU → Dropout → Conv1d → BN  (+skip)
    """

    def __init__(self,
                 in_ch: int,
                 out_ch: int,
                 kernel: int = 7,
                 stride: int = 1,
                 dilation: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel,
                               stride=stride, padding=pad,
                               dilation=dilation, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel,
                               padding=pad, dilation=dilation, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.drop  = nn.Dropout(dropout)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class SEBlock1D(nn.Module):
    """
    Squeeze-and-Excitation block — channel attention.
    Learns to weight each channel by its global importance.
    Plug in after any conv block for a free accuracy boost.
    """

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc  = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, max(channels // reduction, 4)),
            nn.ReLU(),
            nn.Linear(max(channels // reduction, 4), channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, L)
        w = self.fc(self.gap(x))           # (B, C)
        return x * w.unsqueeze(-1)         # channel-wise scaling


class ResidualBlockSE1D(ResidualBlock1D):
    """ResidualBlock1D with a Squeeze-and-Excitation gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_ch = args[1] if len(args) > 1 else kwargs["out_ch"]
        self.se = SEBlock1D(out_ch)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.skip(x))
