import torch, math
import torch.nn as nn


class SinusoidalEmbedding(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.dim = dim

  def forward(self, t):
    """
    t.shape # (B)
    """
    half_dim = self.dim//2
    freqs = torch.exp(torch.linspace(0, math.log(10_000), half_dim, device=t.device)) # (half_dim)
    args = t[:, None] * freqs[None, :] # (B, half_dim)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


class TimeEmbedding(nn.Module):

  def __init__(self, dim):
    super().__init__()
    self.sin_emb = SinusoidalEmbedding(dim)
    self.mlp = nn.Sequential(
      nn.Linear(dim, dim),
      nn.SiLU(),
      nn.Linear(dim, dim),
    )

  def forward(self, t):
    h = self.sin_emb(t) # (B, time_dim)
    out = self.mlp(h)   # (B, time_dim)
    return out


def main():
  B, time_dim = 4, 16
  t = torch.rand(B) # (B)
  print(f"time t: {t}")
  model = SinusoidalEmbedding(dim=time_dim)
  out = model(t)
  print(f"out.shape {out.shape}")
  print(f"out {out}")

if __name__ == '__main__':
  main()