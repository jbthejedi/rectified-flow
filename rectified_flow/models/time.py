import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t.shape = (B,)
        """
        half_dim = self.dim // 2
        # Exponential frequency scale (log-spaced)
        freqs = torch.exp(torch.linspace(                               # (half_dim)
            0, math.log(10000), half_dim, device=t.device               
        ))
        args = t[:, None] * freqs[None, :]                              # (B, 1)x(1, half_dim) -> (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)     # (B, dim)
        return emb


class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.sin_emb = SinusoidalTimeEmbedding(dim) # (B, time_dim)
        self.mlp = nn.Sequential(                   # (B, time_dim)
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sin_emb(t) # (B, time_dim)
        out = self.mlp(emb)   # (B, time_dim)
        return out