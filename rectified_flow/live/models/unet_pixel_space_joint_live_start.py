import torch
import torch.nn as nn
import torch.nn.functional as F
from rectified_flow.live.models.time import *


class Downsample(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

  def forward(self, x):
    return self.conv(x)


class Upsample(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode="nearest")
    self.conv = nn.Conv2d(channels, channels, 3, stride=1, padding=1)

  def forward(self, x):
    return self.conv(self.up(x))


class ResnetBlock(nn.Module):

  def __init__(self, in_ch, out_ch, time_dim, p_dropout=None):
    super().__init__()
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    self.time_proj = nn.Linear(time_dim, out_ch)

    self.norm1 = nn.GroupNorm(self.__get_num_groups(in_ch), in_ch)
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    self.norm2 = nn.GroupNorm(self.__get_num_groups(out_ch), out_ch)
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

    self.dropout = nn.Dropout(p=p_dropout) if p_dropout is not None else nn.Identity()

  def __get_num_groups(self, C):
    for g in [32, 16, 8, 4, 2, 1]:
      if C % g == 0:
        return g
    return 1

  def forward(self, xt, t_emb):
    """
    t_emb.shape = (B, time_dim)
    """
    h = self.conv1(F.silu(self.norm1(xt)))
    h = h + self.time_proj(t_emb)[:, :, None, None]
    h = self.conv2(F.silu(self.norm2(h)))
    h = self.dropout(h)
    return F.silu(h + self.skip(xt))


class UNetPixelSpace(nn.Module):

  def __init__(self, in_ch, time_dim, p_dropout=None):
    super().__init__()
    self.time_emb = TimeEmbedding(time_dim)

    # Encoder: 128 -> 256 -> 512 -> 1024
    self.enc1 = ResnetBlock(in_ch, 128, time_dim, p_dropout)
    self.enc2 = ResnetBlock(128, 256, time_dim, p_dropout)
    self.down2 = Downsample(256)
    self.enc3 = ResnetBlock(256, 512, time_dim, p_dropout)
    self.down3 = Downsample(512)
    self.enc4 = ResnetBlock(512, 1024, time_dim, p_dropout)
    self.down4 = Downsample(1024)

    # Middle: 1024 -> 1024
    self.mid = ResnetBlock(1024, 1024, time_dim, p_dropout)

    # Decoder: 1024 -> 512 -> 256 -> 128
    self.up4 = Upsample(1024)
    self.dec4 = ResnetBlock(1024 + 1024, 1024, time_dim, p_dropout)
    self.up3 = Upsample(1024)
    self.dec3 = ResnetBlock(1024 + 512, 512, time_dim, p_dropout)
    self.up2 = Upsample(512)
    self.dec2 = ResnetBlock(512 + 256, 256, time_dim, p_dropout)
    self.dec1 = ResnetBlock(256 + 128, 128, time_dim, p_dropout)

    self.image_out_proj = nn.Conv2d(128, in_ch, 1)

  def forward(self, xt, t):
    """
    xt.shape (B, C, H, W)
    t.shape (B)
    """
    t_emb = self.time_emb(t)
    d1 = self.enc1(xt, t_emb)                                   # (B, 128, 32, 32)
    d2 = self.enc2(d1, t_emb)                                   # (B, 256, 32, 32)
    d3 = self.enc3(self.down2(d2), t_emb)                       # (B, 512, 16, 16)
    d4 = self.enc4(self.down3(d3), t_emb)                       # (B, 1024, 8, 8)

    m = self.mid(self.down4(d4), t_emb)                         # (B, 1024, 4, 4)

    u4 = self.dec4(torch.cat([self.up4(m), d4], dim=1), t_emb)  # (B, 1024, 8, 8)
    u3 = self.dec3(torch.cat([self.up3(u4), d3], dim=1), t_emb) # (B, 512, 16, 16)
    u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1), t_emb) # (B, 256, 32, 32)
    u1 = self.dec1(torch.cat([u2, d1], dim=1), t_emb)           # (B, 128, 32, 32)
    return self.image_out_proj(u1)                                    # (B, 3, 32, 32)


def main():
  B, C, H, W = 4, 3, 32, 32
  xt = torch.rand(B, C, H, W, device='cpu')
  t = torch.rand(B, device='cpu')
  time_dim=16
  model = UNetPixelSpace(in_ch=C, time_dim=time_dim, p_dropout=None)
  out = model(xt, t)
  print(f"out.shape {out.shape}")


if __name__ == "__main__":
  main()