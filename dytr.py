# DyTR implementation derived from the paper "Residual Learning towards High-fidelity Vehicle Dynamics Modeling with Transformer".
# Implements equations (7)-(11) and training details from Section IV-A.

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Learnable positional embedding used for both encoder and decoder."""

    def __init__(self, max_len: int, dim: int):
        super().__init__()
        self.embedding = nn.Embedding(max_len, dim)

    def forward(self, length: int) -> torch.Tensor:
        indices = torch.arange(length, device=self.embedding.weight.device)
        return self.embedding(indices)


class DynamicsFeatureEncoder(nn.Module):
    """F_enc in Eq.(7): encode [s^r_i, u_i] with a small MLP."""

    def __init__(self, state_dim: int, control_dim: int, feature_dim: int, hidden: int | None = None):
        super().__init__()
        hidden = hidden or feature_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, feature_dim),
        )
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, s_hist: torch.Tensor, u_hist: torch.Tensor) -> torch.Tensor:
        # s_hist, u_hist: [B, T, ...]
        x = torch.cat([s_hist, u_hist], dim=-1)
        feat = self.mlp(x)
        feat = self.norm(feat)
        return feat.transpose(0, 1)  # [T, B, C]


class TemporalFusion(nn.Module):
    """Transformer encoder to aggregate temporal features (Eq.(8))."""

    def __init__(self, dim: int, layers: int = 2, nhead: int = 8, max_len: int = 50):
        super().__init__()
        self.pe = PositionalEncoding(max_len, dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=dim, nhead=nhead, batch_first=False, dim_feedforward=4 * dim)
        self.encoder = nn.TransformerEncoder(enc_layer, layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, B, _ = x.shape
        x = x + self.pe(T)[:, None, :]
        return self.encoder(x)


class ResidualDecoder(nn.Module):
    """Transformer decoder to estimate residuals (Eq.(9)-(11))."""

    def __init__(self, state_dim: int, cfg_dim: int, feature_dim: int, layers: int = 2, nhead: int = 8):
        super().__init__()
        self.query_proj = nn.Linear(state_dim + cfg_dim, feature_dim)
        self.pos_q = PositionalEncoding(1, feature_dim)
        dec_layer = nn.TransformerDecoderLayer(d_model=feature_dim, nhead=nhead, batch_first=False, dim_feedforward=4 * feature_dim)
        self.decoder = nn.TransformerDecoder(dec_layer, layers)
        self.out_proj = nn.Linear(feature_dim, state_dim)
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, s_pred_next: torch.Tensor, cfg: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        # s_pred_next: [B, sd]; cfg: [B, cfg_dim]; memory: [T,B,C]
        q = torch.cat([s_pred_next, cfg], dim=-1)
        q = self.query_proj(q)[None, :, :] + self.pos_q(1)[:, None, :]
        q = self.decoder(tgt=q, memory=memory)
        q = self.norm(q.squeeze(0))
        return self.out_proj(q)


class DyTR(nn.Module):
    """Main network composed of feature encoder, temporal fusion and residual decoder."""

    def __init__(self, state_dim: int = 3, control_dim: int = 8, cfg_dim: int = 1, feature_dim: int = 64, T: int = 15, layers: int = 2):
        super().__init__()
        self.T = T
        self.encoder = DynamicsFeatureEncoder(state_dim, control_dim, feature_dim)
        self.fusion = TemporalFusion(feature_dim, layers)
        self.decoder = ResidualDecoder(state_dim, cfg_dim, feature_dim, layers)

    def forward(self, s_hist: torch.Tensor, u_hist: torch.Tensor, s_pred_next: torch.Tensor, cfg: torch.Tensor, return_corrected: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        assert s_hist.shape[1] == self.T
        feat_seq = self.encoder(s_hist, u_hist)   # [T,B,C]
        feat_agg = self.fusion(feat_seq)          # [T,B,C]
        delta = self.decoder(s_pred_next, cfg, feat_agg)
        if return_corrected:
            return delta, s_pred_next + delta
        return delta, None


def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights
    return loss.mean()


if __name__ == "__main__":
    # minimal reproducible example using random data
    device = "cuda" if torch.cuda.is_available() else "cpu"
    B, T = 8, 15
    s_hist = torch.randn(B, T, 3, device=device)
    u_hist = torch.randn(B, T, 8, device=device)
    s_pred = torch.randn(B, 3, device=device)
    cfg = torch.randn(B, 1, device=device)
    target = torch.randn(B, 3, device=device)

    model = DyTR().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    weights = torch.tensor([1.0, 10.0, 1000.0], device=device)

    for _ in range(2):  # dummy training loop
        delta, s_corr = model(s_hist, u_hist, s_pred, cfg, return_corrected=True)
        loss = weighted_smooth_l1(s_corr, target, weights)
        opt.zero_grad()
        loss.backward()
        opt.step()
        print(f"loss: {loss.item():.4f}")

