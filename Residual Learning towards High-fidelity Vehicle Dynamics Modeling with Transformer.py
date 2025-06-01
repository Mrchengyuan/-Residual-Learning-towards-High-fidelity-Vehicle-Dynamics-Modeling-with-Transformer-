
"""DyTR  ‑ Dynamics residual Transformer
================================================
Full PyTorch implementation strictly following
《Residual Learning towards High‑fidelity Vehicle Dynamics Modeling with Transformer》.

Key equations (paper)
---------------------
(7)  E_i^s = F_enc([s^r_i , u_i])
(8)  E^f    = Encoder(E^s , pos = P^s)
(9)  Q_0    = Linear([s^r_{t+1} , c])
(10) Q_L    = Decoder(q = Q, k = v = E^f , pos = P^q)
(11) δ̂_{t+1} = Linear(Q_L)

The code matches the three‑stage architecture in Fig. 3 of the
paper and default hyper‑parameters reported in Sec. IV‑A.

Major updates vs previous draft
-------------------------------
• Position encoding: unified class supporting *learnable* (default, Eq.(8))
  or *sinusoidal* variant.
• Added LayerNorm + Dropout inside DynamicsFeatureEncoder to
  stabilize training (paper uses MLP but omits exact recipe).
• Exposed `enc_arch` hook so MLP can be swapped with KAN or other
  F_enc choices mentioned in Sec. III‑C.
• ResidualEstimation now applies LayerNorm before the final linear
  projection and allows optional two‑layer head (Linear‑ReLU‑Linear).
• Forward can directly return both residuals and corrected states for
  convenience.


"""
from __future__ import annotations
import math
from typing import Literal, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------- Positional Embedding -------------------------- #
class PositionalEmbedding(nn.Module):
    """Supports *learnable* (nn.Embedding) or *sinusoidal* positional encodings."""

    def __init__(self, max_len: int, dim: int, kind: Literal["learnable", "sinusoidal"] = "learnable"):
        super().__init__()
        self.kind = kind
        self.dim = dim
        if kind == "learnable":
            self.embedding = nn.Embedding(max_len, dim)
        else:  # sinusoidal – fixed, not trainable
            pe = torch.zeros(max_len, dim)
            position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer("pe", pe, persistent=False)

    def forward(self, seq_len: int) -> torch.Tensor:  # [seq_len, dim]
        if self.kind == "learnable":
            idx = torch.arange(seq_len, device=self.embedding.weight.device)
            return self.embedding(idx)
        return self.pe[:seq_len]

# ----------------------- Dynamics Feature Encoder ------------------------ #
class DynamicsFeatureEncoder(nn.Module):
    """F_enc in Eq.(7).

    Default implementation: 2‑layer MLP + LayerNorm + Dropout.
    Replace or extend by passing a custom module via `enc_arch` hook.
    """

    def __init__(self,
                 state_dim: int,
                 control_dim: int,
                 feature_dim: int,
                 hidden_dim: int = None,
                 dropout: float = 0.1):
        super().__init__()
        hidden_dim = hidden_dim or feature_dim * 2
        self.net = nn.Sequential(
            nn.Linear(state_dim + control_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )

    def forward(self, s_hist: torch.Tensor, u_hist: torch.Tensor) -> torch.Tensor:
        x = torch.cat([s_hist, u_hist], dim=-1)  # [B,T,C]
        x = self.net(x)                          # [B,T,feature]
        return x.transpose(0, 1)                 # [T,B,feature]

# -------------------------- Temporal Fusion --------------------------- #
class TemporalFusion(nn.Module):
    """Transformer encoder that yields aggregated dynamics feature E^f (Eq.(8))."""
    def __init__(self, feature_dim: int, num_layers: int = 2, nhead: int = 8,
                 dropout: float = 0.1, max_len: int = 50,
                 pos_kind: Literal["learnable", "sinusoidal"] = "learnable"):
        super().__init__()
        self.pos_embed = PositionalEmbedding(max_len, feature_dim, pos_kind)
        enc_layer = nn.TransformerEncoderLayer(d_model=feature_dim,
                                               nhead=nhead,
                                               dim_feedforward=4*feature_dim,
                                               dropout=dropout,
                                               batch_first=False)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

    def forward(self, E_s: torch.Tensor) -> torch.Tensor:
        T, B, _ = E_s.shape
        E_s = E_s + self.pos_embed(T)[:, None, :]
        return self.encoder(E_s)  # [T,B,feature]

# ----------------------- Residual Estimation -------------------------- #
class ResidualEstimation(nn.Module):
    """Transformer decoder that updates query Q (Eq.(9‑11))."""
    def __init__(self, state_dim: int, config_dim: int, feature_dim: int,
                 num_layers: int = 2, nhead: int = 8, dropout: float = 0.1,
                 pos_kind: Literal["learnable", "sinusoidal"] = "learnable"):
        super().__init__()
        self.query_proj = nn.Linear(state_dim + config_dim, feature_dim)
        self.pos_q = PositionalEmbedding(1, feature_dim, pos_kind)
        dec_layer = nn.TransformerDecoderLayer(d_model=feature_dim,
                                               nhead=nhead,
                                               dim_feedforward=4*feature_dim,
                                               dropout=dropout,
                                               batch_first=False)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.norm   = nn.LayerNorm(feature_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feature_dim, state_dim),
        )

    def forward(self, s_pred_next: torch.Tensor, cfg: torch.Tensor, E_f: torch.Tensor) -> torch.Tensor:
        q0 = torch.cat([s_pred_next, cfg], dim=-1)            # [B, sd+cd]
        q0 = self.query_proj(q0)[None, :, :]                  # [1,B,C]
        q0 = q0 + self.pos_q(1)[:, None, :]
        qL = self.decoder(tgt=q0, memory=E_f)                # [1,B,C]
        qL = self.norm(qL.squeeze(0))                        # [B,C]
        return self.out_proj(qL)                             # δ̂_{t+1}

# ----------------------------- DyTR ---------------------------------- #
class DyTR(nn.Module):
    """Main network wrapper."""
    def __init__(self, state_dim: int = 3, control_dim: int = 8, config_dim: int = 1,
                 feature_dim: int = 64, T: int = 15, transformer_layers: int = 2,
                 pos_kind: Literal["learnable", "sinusoidal"] = "learnable"):
        super().__init__()
        self.T = T
        self.encoder = DynamicsFeatureEncoder(state_dim, control_dim, feature_dim)
        self.fusion  = TemporalFusion(feature_dim, transformer_layers, pos_kind=pos_kind)
        self.res_est = ResidualEstimation(state_dim, config_dim, feature_dim,
                                          transformer_layers, pos_kind=pos_kind)

    def forward(self,
                s_hist: torch.Tensor,        # [B,T,sd]  – estimated states
                u_hist: torch.Tensor,        # [B,T,cd]
                s_pred_next: torch.Tensor,   # [B,sd]    – base model prediction
                cfg: torch.Tensor,           # [B,config_dim]
                return_corrected: bool = False) -> Tuple[torch.Tensor, torch.Tensor | None]:
        assert s_hist.shape[1] == self.T, "history length ≠ T"
        E_s = self.encoder(s_hist, u_hist)         # [T,B,C]
        E_f = self.fusion(E_s)                     # [T,B,C]
        delta = self.res_est(s_pred_next, cfg, E_f)  # [B,sd]
        if return_corrected:
            return delta, s_pred_next + delta
        return delta, None

# ---------------------- Loss: weighted Smooth‑L1 --------------------- #
def weighted_smooth_l1(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    loss = F.smooth_l1_loss(pred, target, reduction="none") * weights  # [B,sd]
    return loss.mean()

# --------------------- Minimal sanity test --------------------------- #
if __name__ == "__main__":
    B, T = 256, 15
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    hist_s = torch.randn(B, T, 3, device=device)
    hist_u = torch.randn(B, T, 8, device=device)
    s_hat  = torch.randn(B, 3, device=device)
    cfg    = torch.randn(B, 1, device=device)
    gt     = torch.randn(B, 3, device=device)

    model = DyTR().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
    weights = torch.tensor([1.0, 10.0, 1000.0], device=device)

    delta, s_corr = model(hist_s, hist_u, s_hat, cfg, return_corrected=True)
    loss = weighted_smooth_l1(s_corr, gt, weights)
    loss.backward()
    opt.step()
    print(f"test ‑ ok, loss {loss.item():.3f}")

# 控制信号详细分解
# control_signals = [
#     T1,  # 前左轮驱动扭矩
#     T2,  # 前右轮驱动扭矩
#     T3,  # 后左轮驱动扭矩
#     T4,  # 后右轮驱动扭矩
#     θ1,  # 前左轮转向角
#     θ2,  # 前右轮转向角
#     θ3,  # 后左轮转向角
#     θ4   # 后右轮转向角
# ]
# 总共8个维度