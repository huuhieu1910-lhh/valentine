"""Fig. 1 networks: Branch 1 (CNN), Branch 2 (Transformer) and the weighted ensemble.

Every branch maps an input window X of shape (batch, 24, 13) to a forecast of shape
(batch, horizon). Each activation is its own module instance, so SHAP DeepExplainer
can attach its DeepLIFT rules per layer.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from .config import ModelConfig


class CNNBranch(nn.Module):
    """Branch 1: local temporal motif extraction.

    [Conv1D(k=3, c=64) -> BN + ReLU -> MaxPool1D(s=2)] x 2 -> GAP -> Dense (z_CNN) -> y_CNN
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        c, k, s = cfg.conv_channels, cfg.conv_kernel, cfg.pool_stride
        self.features = nn.Sequential(
            nn.Conv1d(cfg.n_features, c, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=s, stride=s),
            nn.Conv1d(c, c, kernel_size=k, padding=k // 2),
            nn.BatchNorm1d(c),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=s, stride=s),
        )
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(nn.Linear(c, cfg.dense_dim), nn.ReLU())
        self.head = nn.Linear(cfg.dense_dim, cfg.horizon)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return z_CNN for x of shape (batch, seq_len, n_features)."""
        h = self.features(x.transpose(1, 2))      # (batch, 64, 24 / 4)
        return self.dense(self.gap(h).squeeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class PositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to the projected sequence."""

    def __init__(self, d_model: int, max_len: int):
        super().__init__()
        position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        freq = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * freq)
        pe[:, 1::2] = torch.cos(position * freq)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class MultiHeadSelfAttention(nn.Module):
    """Scaled dot-product self-attention with h heads of size d_k = d_model / h."""

    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} is not divisible by n_heads={n_heads}")
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def _split_heads(self, t: torch.Tensor) -> torch.Tensor:
        batch, length, _ = t.shape
        return t.view(batch, length, self.n_heads, self.d_k).transpose(1, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, d_model = x.shape
        q, k, v = (self._split_heads(proj(x)) for proj in (self.q_proj, self.k_proj, self.v_proj))
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)       # (batch, h, L, L)
        attn = self.attn_dropout(torch.softmax(scores, dim=-1))
        out = (attn @ v).transpose(1, 2).reshape(batch, length, d_model)
        return self.out_proj(out)


class FeedForward(nn.Module):
    """Position-wise feed-forward network d_model -> d_ff -> d_model."""

    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class EncoderBlock(nn.Module):
    """Layer Norm -> Multi-Head Self-Attention -> Layer Norm -> Feed-Forward, each with a residual."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.drop1 = nn.Dropout(cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.drop2 = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.norm1(x)))
        return x + self.drop2(self.ffn(self.norm2(x)))


class TransformerBranch(nn.Module):
    """Branch 2: long-range dependency extraction.

    Linear projection (d=64) -> positional encoding -> 2 encoder blocks -> GAP -> Dense (z_T) -> y_T
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.input_proj = nn.Linear(cfg.n_features, cfg.d_model)
        self.pos_enc = PositionalEncoding(cfg.d_model, cfg.seq_len)
        self.blocks = nn.Sequential(*(EncoderBlock(cfg) for _ in range(cfg.n_blocks)))
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dense = nn.Sequential(nn.Linear(cfg.d_model, cfg.dense_dim), nn.ReLU())
        self.head = nn.Linear(cfg.dense_dim, cfg.horizon)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return z_T for x of shape (batch, seq_len, n_features)."""
        h = self.blocks(self.pos_enc(self.input_proj(x)))           # (batch, 24, 64)
        return self.dense(self.gap(h.transpose(1, 2)).squeeze(-1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encode(x))


class WeightedEnsemble(nn.Module):
    """y = sum_i w_i * y_i over i in {CNN, T}, with w = softmax(alpha) so w_i >= 0 and sum_i w_i = 1."""

    def __init__(self, n_members: int = 2):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(n_members))

    @property
    def weights(self) -> torch.Tensor:
        return torch.softmax(self.logits, dim=0)

    def forward(self, member_preds: torch.Tensor) -> torch.Tensor:
        """member_preds: (..., n_members) -> (...)."""
        return (member_preds * self.weights).sum(dim=-1)


class PhysicsInformedEnsemble(nn.Module):
    """Both branches plus the weighted ensemble of Fig. 1."""

    members = ("cnn", "transformer")

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.cnn = CNNBranch(cfg)
        self.transformer = TransformerBranch(cfg)
        self.ensemble = WeightedEnsemble(len(self.members))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        y_cnn = self.cnn(x)
        y_t = self.transformer(x)
        y = self.ensemble(torch.stack([y_cnn, y_t], dim=-1))
        return {"cnn": y_cnn, "transformer": y_t, "ensemble": y}
