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
    Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> skip connection.
    """

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel: int = 7,
        stride: int = 1,
        dilation: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        pad = (kernel // 2) * dilation
        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel,
            stride=stride,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel,
            padding=pad,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_ch)
        self.drop = nn.Dropout(dropout)

        self.skip = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.skip = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.skip(x))


class SEBlock1D(nn.Module):
    """Squeeze-and-excitation channel attention for 1D features."""

    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.gap(x))
        return x * weights.unsqueeze(-1)


class ResidualBlockSE1D(ResidualBlock1D):
    """ResidualBlock1D with a squeeze-and-excitation gate."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        out_ch = args[1] if len(args) > 1 else kwargs["out_ch"]
        self.se = SEBlock1D(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.drop(out)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + self.skip(x))


class LearnedPositionalEncoding1D(nn.Module):
    """Learned positional encoding with interpolation for variable lengths."""

    def __init__(self, embed_dim: int, max_seq_len: int = 256):
        super().__init__()
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        nn.init.normal_(self.pos_embed, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        if seq_len == self.pos_embed.size(1):
            return self.pos_embed

        pos = self.pos_embed.transpose(1, 2)
        pos = F.interpolate(pos, size=seq_len, mode="linear", align_corners=False)
        return pos.transpose(1, 2)


class TransformerEncoder1D(nn.Module):
    """
    Lightweight transformer encoder for convolutional feature sequences.

    Input:  (B, C, L)
    Output: (B, C, L)
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        num_layers: int = 1,
        ff_multiplier: float = 2.0,
        dropout: float = 0.1,
        max_seq_len: int = 256,
    ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})."
            )

        self.pos_encoding = LearnedPositionalEncoding1D(
            embed_dim=embed_dim,
            max_seq_len=max_seq_len,
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=max(int(embed_dim * ff_multiplier), embed_dim),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = x.transpose(1, 2)
        tokens = tokens + self.pos_encoding(tokens)
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        return tokens.transpose(1, 2)
