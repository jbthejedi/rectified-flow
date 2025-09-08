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
    self.conv = nn.Conv2d(channels, channels, 3, padding=1)

  def forward(self, x):
    u = self.up(x)
    out = self.conv(u)
    return out


class ResnetBlock(nn.Module):

  def __init__(self, in_ch, out_ch, time_dim, p_dropout=None):
    super().__init__()
    self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    # PreAct
    self.norm1 = nn.GroupNorm(self.__gn_groups(in_ch), in_ch)  # <- use in_ch here
    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

    self.time_proj = nn.Linear(time_dim, out_ch)
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
    h = h + self.time_proj(t_emb)[:, :, None, None]
    h = self.conv2(F.silu(self.norm2(h)))
    h = self.dropout(h)
    return F.silu(h + self.skip(x))


class FiLMAdapter(nn.Module):

  def __init__(self, txt_dim, target_dim):
    super().__init__()
    self.proj = nn.Linear(txt_dim, 2 * target_dim)

    #### ADDED #####
    # So FiLM contribution is 0 on first pass
    nn.init.zeros_(self.proj.weight)
    nn.init.zeros_(self.proj.bias)


  def forward(self, txt_emb):
    # txt_emb.shape                                    # (, txt_dim)
    gamma, beta = self.proj(txt_emb).chunk(2, dim=-1)  # (, target_dim), (, target_dim)
    return gamma, beta


class CrossAttention(nn.Module):

  def __init__(self, img_dim, txt_dim, n_heads=8):
    super().__init__()
    assert img_dim % n_heads == 0
    self.to_q = nn.Linear(img_dim, img_dim)
    self.to_k = nn.Linear(txt_dim, img_dim)
    self.to_v = nn.Linear(txt_dim, img_dim)

    self.d = img_dim // n_heads

    self.proj_out = nn.Linear(img_dim, img_dim)

    #### ADDED #####
    # So CrossAttention contribution is 0 on first pass
    nn.init.zeros_(self.proj_out.weight)
    nn.init.zeros_(self.proj_out.bias)
    self.ca_gate = nn.Parameter(torch.zeros(1))

    self.nh = n_heads
  
  def forward(self, img_toks, txt_toks):
    B, P, _ = img_toks.shape                            # (, P, img_dim), P = H*W
    _, L, _ = txt_toks.shape                            # (, L, txt_dim)
    # txt_emb.shape                                             (, 1, txt_dim)

    Q = self.to_q(img_toks).view(B, P, self.nh, self.d).transpose(2, 1)       # (, nh, P, D)
    K = self.to_k(txt_toks).view(B, L, self.nh, self.d).transpose(2, 1)       # (, nh, 1, D)
    V = self.to_v(txt_toks).view(B, L, self.nh, self.d).transpose(2, 1)       # (, nh, 1, D)

    attn = Q @ K.transpose(-2, -1) * (self.d ** -0.5)                   # (, nh, P, 1)
    attn = torch.softmax(attn, dim=-1)                                  # (, nh, P, 1)
    out = attn @ V                                                      # (, nh, P, D)
    out = out.transpose(1, 2).contiguous().view(B, P, self.d * self.nh) # (, P, img_dim), img_dim = D*n_heads
    return img_toks + self.ca_gate * self.proj_out(out)                                     # (, P, img_dim)


class UNetPixelSpace(nn.Module):

  def __init__(self, in_channels=3, time_dim=128, p_dropout=None):
    super().__init__()
    self.time_emb = TimeEmbedding(dim=time_dim)

    # Down path (keep blocks shape-preserving; downsample between levels)
    self.enc1 = ResnetBlock(in_channels, 128, time_dim, p_dropout)
    self.ds1 = Downsample(128)
    self.enc2 = ResnetBlock(128, 256, time_dim, p_dropout)
    self.ds2 = Downsample(256)
    self.enc3 = ResnetBlock(256, 512, time_dim, p_dropout)
    self.ds3 = Downsample(512)
    self.enc4 = ResnetBlock(512, 1024, time_dim, p_dropout)
    self.ds4 = Downsample(1024)

    self.mid = ResnetBlock(1024, 1024, time_dim, p_dropout)

    # Up path (upsample + concat + block)
    self.us4 = Upsample(1024)
    self.dec4 = ResnetBlock(1024 + 1024, 1024, time_dim, p_dropout)
    self.us3 = Upsample(1024)
    self.dec3 = ResnetBlock(1024 + 512, 512, time_dim, p_dropout)
    self.us2 = Upsample(512)
    self.dec2 = ResnetBlock(512 + 256, 256, time_dim, p_dropout)
    self.us1 = Upsample(256)
    self.dec1 = ResnetBlock(256 + 128, 128, time_dim, p_dropout)

    # self.v_pred = nn.Conv2d(128, in_channels, 1)
    # self.u_pred = nn.Conv2d(128, in_channels, 1)

    img_dim = 1024
    self.cross_attn = CrossAttention(img_dim=img_dim, txt_dim=128, n_heads=4)

    langvae_dim = 128
    self.film = FiLMAdapter(txt_dim=langvae_dim, target_dim=img_dim)

    self.out = nn.Conv2d(128, in_channels, 1)


  def forward(self, x_img_t, x_txt_pool_t, t_emb):
    B, C, H, W = x_img_t.shape
    _, TH = x_txt_pool_t.shape
    time_emb = self.time_emb(t_emb.reshape(B))

    # Image tokens
    # img_tokens = x_img_t.permute(0, 2, 3, 1).view(B, H*W, C)      # (H*W, C)
    d1 = self.enc1(x_img_t, time_emb)                               # (, 128, 32, 32)
    d2 = self.enc2(d1, time_emb)                                    # (, 256, 32, 32)
    d3 = self.enc3(self.ds2(d2), time_emb)                          # (, 512, 16, 16)
    d4 = self.enc4(self.ds3(d3), time_emb)                          # (, 1024, 8, 8)

    m = self.mid(self.ds4(d4), time_emb)                            # (, 1024, 4, 4)
    
    ##### FiLM #####
    _, C_m, H_m, W_m = m.shape
    γ, β = self.film(x_txt_pool_t)                                  # (, 1024), (, 1024)
    γ = γ[:, :, None, None] # (, 1024, 1, 1)
    β = β[:, :, None, None] # (, 1024, 1, 1)
    m = m * (1 + γ) + β # (, 1024, 4, 4)

    ##### Cross Attention #####
    img_toks = m.flatten(2).permute(0, 2, 1)    # (,  4*4, 1024)
    txt_toks = x_txt_pool_t.unsqueeze(1)        # (,    1, txt_dim)
    m = self.cross_attn(img_toks, txt_toks)     # (, 4*4, 102)
    m = m.permute(0, 2, 1).view(B, C_m, H_m, W_m)

    u4 = self.dec4(torch.cat([self.us4(m), d4], dim=1), time_emb)   # 1024,  8, 8
    u3 = self.dec3(torch.cat([self.us3(u4), d3], dim=1), time_emb)  #  512, 16, 16
    u2 = self.dec2(torch.cat([self.us2(u3), d2], dim=1), time_emb)  #  256, 32, 32
    u1 = self.dec1(torch.cat([u2, d1], dim=1), time_emb)            #  128, 32, 32

    return self.out(u1)
