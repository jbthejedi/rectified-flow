import torch
import torch.nn as nn
from rectified_flow.models.time import *
import torch.nn.functional as F


class Downsample(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)

  def forward(self, x):
    return self.conv(x)


class Upsample(nn.Module):

  def __init__(self, channels):
    super().__init__()
    self.up = nn.Upsample(scale_factor=2, mode="nearest")
    self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    return self.conv(self.up(x))


class ResnetBlock(nn.Module):

  def __init__(self, in_ch, out_ch, time_dim, p_dropout=None):
    super().__init__()
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    # PreAct
    self.norm1 = nn.GroupNorm(self.__gn_groups(in_ch), in_ch)  # <- use in_ch here
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    self.time_proj = nn.Linear(time_dim, 2*out_ch)
    self.dropout = nn.Dropout(p_dropout) if p_dropout else nn.Identity()

    self.norm2 = nn.GroupNorm(self.__gn_groups(out_ch), out_ch)  # <- use out_ch here
    self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

  def __gn_groups(self, C: int, cap: int = 32) -> int:
    # pick the largest divisor of C from a safe set; fallback to 1
    for g in (32, 16, 8, 4, 2, 1):
      if C % g == 0 and g <= cap:
        return g
    return 1

  def forward(self, x, t_emb):
    norm1 = self.norm1(x)
    h = self.conv1(F.silu(norm1))

    # AdaGN
    scale, shift = self.time_proj(t_emb).chunk(2, dim=1)
    h = self.norm2(h)
    h = (1 + scale)[:, :, None, None] * h + shift[:, :, None, None]

    h = self.conv2(F.silu(h))
    h = self.dropout(h)
    return F.silu(h + self.skip(x))


class MHSA(nn.Module):

    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, c)
        self.qkv = nn.Conv2d(c, c*3, 1)
        self.proj = nn.Conv2d(c, c, 1)
        self.heads = heads

    def forward(self, x):
        b,c,h,w = x.shape
        h_ = self.norm(x)
        q,k,v = self.qkv(h_).view(b,3,self.heads,c//self.heads,h*w).unbind(1)
        q,k,v = [t.transpose(-2,-1) for t in (q,k,v)]  # (b,heads,HW,dim)
        attn = (q @ k.transpose(-2,-1)) * (1.0 / (q.shape[-1] ** 0.5))
        attn = attn.softmax(dim=-1)
        y = attn @ v                               # (b,heads,HW,dim)
        y = y.transpose(-2,-1).contiguous().view(b,c,h,w)
        return x + self.proj(y)


class AttnBlock(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__()
        self.norm = nn.GroupNorm(32, c)
        self.attn = MHSA(c, heads=heads)  # your MHSA from earlier
    def forward(self, x):
        return x + self.attn(self.norm(x))


class UnetLayer(nn.Module):

  def __init__(self, in_ch, out_ch, time_dim, p_dropout=None):
    super().__init__()
    self.block1 = ResnetBlock(in_ch, out_ch, time_dim, p_dropout)
    self.block2 = ResnetBlock(out_ch, out_ch, time_dim, p_dropout)
  
  def forward(self, xt, t):
    h = self.block1(xt, t)
    return self.block2(h, t)


class UNetPixelSpacePlus(nn.Module):

  def __init__(self, in_ch=3, time_dim=128, p_dropout=None):
    super().__init__()
    self.time_emb = TimeEmbedding(dim=time_dim)

    # Down path (keep blocks shape-preserving; downsample between levels)
    self.enc1 = UnetLayer(in_ch, 128, time_dim, p_dropout)
    self.ds1 = Downsample(128)

    self.enc2 = UnetLayer(128, 256, time_dim, p_dropout)
    self.ds2 = Downsample(256)
    self.enc_sa2 = AttnBlock(c=256, heads=8)

    self.enc3 = UnetLayer(256, 512, time_dim, p_dropout)
    self.ds3 = Downsample(512)
    self.enc_sa3 = AttnBlock(c=512, heads=8)

    self.enc4 = UnetLayer(512, 1024, time_dim, p_dropout)
    self.ds4 = Downsample(1024)

    self.mid = ResnetBlock(1024, 1024, time_dim, p_dropout)

    # Up path (upsample + concat + block)
    self.us4 = Upsample(1024)
    self.dec4 = UnetLayer(1024 + 1024, 1024, time_dim, p_dropout)
    self.dec_sa4 = AttnBlock(c=1024, heads=8)

    self.us3 = Upsample(1024)
    self.dec3 = UnetLayer(1024 + 512, 512, time_dim, p_dropout)
    self.dec_sa3 = AttnBlock(c=512, heads=8)

    self.us2 = Upsample(512)
    self.dec2 = UnetLayer(512 + 256, 256, time_dim, p_dropout)

    self.us1 = Upsample(256)
    self.dec1 = UnetLayer(256 + 128, 128, time_dim, p_dropout)

    self.image_out = nn.Conv2d(128, in_ch, 1)

  def forward(self, xt, t):
    B = xt.size(0)
    t_emb = self.time_emb(t)

    d1 = self.enc1(xt, t_emb)  # 128, 32, 32
    d2 = self.enc2(d1, t_emb)  # 256, 32, 32

    h = self.enc_sa2(self.ds2(d2))
    d3 = self.enc3(h, t_emb)  # 512, 16, 16

    h = self.enc_sa3(self.ds3(d3))
    d4 = self.enc4(h, t_emb)  # 1024, 8, 8

    m = self.mid(self.ds4(d4), t_emb)  # 1024, 4, 4

    u4 = self.dec4(torch.cat([self.us4(m), d4], dim=1), t_emb)  # 1024,  8, 8
    u4 = self.dec_sa4(u4)
    u3 = self.dec3(torch.cat([self.us3(u4), d3], dim=1), t_emb)  #  512, 16, 16
    u4 = self.dec_sa3(u3)
    u2 = self.dec2(torch.cat([self.us2(u3), d2], dim=1), t_emb)  #  256, 32, 32
    u1 = self.dec1(torch.cat([u2, d1], dim=1), t_emb)  #  128, 32, 32

    return self.image_out(u1)
