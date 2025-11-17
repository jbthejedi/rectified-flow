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

# TODO class DoubleResnetBlock

class CrossAttention(nn.Module):
  def __init__(self, img_dim, txt_dim, n_heads=8, p_dropout=None):
    super().__init__()
    assert img_dim % n_heads == 0
    self.query = nn.Linear(img_dim, img_dim)
    self.key = nn.Linear(txt_dim, img_dim)
    self.value = nn.Linear(txt_dim, img_dim)
    self.proj = nn.Linear(img_dim, img_dim)
    self.dropout = nn.Dropout(p=p_dropout) if p_dropout is not None else nn.Identity()
    self.d = img_dim // n_heads
    self.nh = n_heads
  
  def forward(self, img_toks, txt_toks, attn_mask=None):
    """
    img_toks (B, P, img_dim)
    txt_toks (B, L, txt_dim)
    attn_mask (B, L)
    """
    B, P, _ = img_toks.shape
    _, L, _ = txt_toks.shape
    Q = self.query(img_toks).view(B, P, self.nh, self.d).transpose(1, 2) # (B, nh, P, d)
    K = self.key(txt_toks).view(B, L, self.nh, self.d).transpose(1, 2)   # (B, nh, L, d)
    V = self.value(txt_toks).view(B, L, self.nh, self.d).transpose(1, 2) # (B, nh, L, d)
    # K.transpose(2, 3).shape => (B, nh, d, L)
    attn = Q @ K.transpose(2, 3) * (self.d ** -0.5) # (B, nh, P, L)

    if attn_mask is not None:
      mask = (~attn_mask.bool()).unsqueeze(1).unsqueeze(2) # (B, 1, 1, L)
      attn = attn.masked_fill(mask, float('-inf'))

    attn = torch.softmax(attn, dim=-1)
    attn = self.dropout(attn @ V) # (B, nh, P, d)
    attn = attn.transpose(1, 2).contiguous().flatten(2) # (B, P, nh, d)

    return self.proj(attn)


class FiLM(nn.Module):

  def __init__(self, txt_dim, img_dim, time_dim, hidden_dim):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(txt_dim+time_dim, hidden_dim),
      nn.SiLU(),
      nn.Linear(hidden_dim, 2*img_dim),
    )

  def forward(self, pooled_txt_toks, img_emb, time_emb):
    """
    pooled_txt_toks (B, TH)
    img_emb (B, C, H, W)
    time_emb (B, time_dim)
    """
    h = torch.cat([pooled_txt_toks, time_emb], dim=-1) # (B, TH+time_dim)
    γ, β = self.proj(h).chunk(2, dim=-1) # {(B, C), (B, C)})
    return img_emb * (1 + γ[:, :, None, None]) + β[:, :, None, None] # (B, C, H, W)


class TxtVelocityHead(nn.Module):
  def __init__(self, txt_dim=512, img_dim=1024, hidden=1024):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(txt_dim + img_dim, hidden),
      nn.SiLU(),
      nn.Linear(hidden, txt_dim),
    )
  
  def forward(self, txt_toks, img_feats):
    """
    img_feat.shape => (B, C_m, H_m, W_m)
    txt_toks.shape => (B, L, P)
    """
    B, L, P = txt_toks.shape
    pooled_img_feats = img_feats.mean(dim=[2, 3]) # (B, C_m)
    h = pooled_img_feats[:, None, :].expand(B, L, pooled_img_feats.size(1)) # (B, L, C_m)
    return self.proj(torch.cat([h, txt_toks], dim=-1)) # (B, L, C_m + P)


class UNetJoint(nn.Module):
  
  def __init__(self, in_ch=3, time_dim=128, txt_dim=512, p_dropout=None):
    super().__init__()
    self.time_emb_img = TimeEmbedding(time_dim)
    self.time_emb_txt = TimeEmbedding(time_dim)
    time_hidden=512
    self.time_fuse = nn.Sequential(
      nn.Linear(2*time_dim, time_hidden),
      nn.SiLU(),
      nn.Linear(time_hidden, time_dim)
    )

    # Encoder: 128 -> 256 -> 512 -> 1024
    # TODO turn Rnb -> doubleRnb
    self.enc1 = ResnetBlock(in_ch, 128, time_dim, p_dropout)
    self.enc2 = ResnetBlock(128, 256, time_dim, p_dropout)
    self.down2 = Downsample(256)
    # TODO add MHSA after downsample
    self.enc3 = ResnetBlock(256, 512, time_dim, p_dropout)
    self.down3 = Downsample(512)
    self.enc4 = ResnetBlock(512, 1024, time_dim, p_dropout)
    self.down4 = Downsample(1024)

    # Middle: 1024 -> 1024
    self.mid = ResnetBlock(1024, 1024, time_dim, p_dropout)
    self.film = FiLM(txt_dim=txt_dim, img_dim=1024, time_dim=time_dim, hidden_dim=1024)
    self.cross_attn = CrossAttention(img_dim=1024, txt_dim=txt_dim, n_heads=8, p_dropout=0.1)

    # Decoder: 1024 -> 512 -> 256 -> 128
    self.up4 = Upsample(1024)
    self.dec4 = ResnetBlock(1024 + 1024, 1024, time_dim, p_dropout)
    self.up3 = Upsample(1024)
    self.dec3 = ResnetBlock(1024 + 512, 512, time_dim, p_dropout)
    self.up2 = Upsample(512)
    self.dec2 = ResnetBlock(512 + 256, 256, time_dim, p_dropout)
    self.dec1 = ResnetBlock(256 + 128, 128, time_dim, p_dropout)


    self.image_out_proj = nn.Conv2d(128, in_ch, 1)
    self.text_out_proj = TxtVelocityHead(txt_dim=txt_dim)

  def forward(self, x_img_t, x_txt_t, t_img, t_txt, attn_mask=None):
    """
    xt.shape (B, C, H, W)
    t.shape (B)
    """
    t_emb_img = self.time_emb_img(t_img) # (B, time_dim)
    t_emb_txt = self.time_emb_txt(t_txt) # (B, time_dim)
    t_emb = self.time_fuse(torch.cat([t_emb_img, t_emb_txt], dim=-1)) # (B, time_dim)
    d1 = self.enc1(x_img_t, t_emb)                              # (B, 128, 32, 32)
    d2 = self.enc2(d1, t_emb)                                   # (B, 256, 32, 32)
    # TODO add MHSA
    d3 = self.enc3(self.down2(d2), t_emb)                       # (B, 512, 16, 16)
    # TODO add MHSA
    d4 = self.enc4(self.down3(d3), t_emb)                       # (B, 1024, 8, 8)
    
    m = self.mid(self.down4(d4), t_emb)                         # (B, 1024, 4, 4)
    _, C_m, H_m, W_m = m.shape
    pooled_txt_toks = x_txt_t.mean(1) # (B, TH)
    m = self.film(pooled_txt_toks, m, t_emb)                    # (, 1024, 4, 4)
    m = m.flatten(2).transpose(1, 2)                            # (, 4*4, 1024)
    m = self.cross_attn(m, x_txt_t, attn_mask)                  # (, 4*4, 1024)
    m = m.transpose(1, 2).view(-1, C_m, H_m, W_m)

    u4 = self.dec4(torch.cat([self.up4(m), d4], dim=1), t_emb)  # (B, 1024, 8, 8)
    u3 = self.dec3(torch.cat([self.up3(u4), d3], dim=1), t_emb) # (B, 512, 16, 16)
    u2 = self.dec2(torch.cat([self.up2(u3), d2], dim=1), t_emb) # (B, 256, 32, 32)
    u1 = self.dec1(torch.cat([u2, d1], dim=1), t_emb)           # (B, 128, 32, 32)
    img_out = self.image_out_proj(u1)                           # (B, 3, 32, 32)
    txt_out = self.text_out_proj(x_txt_t, m)                    # (B, L, 512)
    return img_out, txt_out


def main():
  B, C, H, W = 4, 3, 32, 32
  L, TH = 7, 64
  x_img_t = torch.rand(B, C, H, W, device='cpu')
  t_img = torch.rand(B, device='cpu')
  x_txt_t = torch.rand(B, L, TH, device='cpu')
  t_txt = torch.rand(B, device='cpu')
  TD = 16

  model = UNetJoint(in_ch=C, time_dim=TD, txt_dim=TH, p_dropout=None)
  v_hat_img, v_hat_txt = model(x_img_t, x_txt_t, t_img, t_txt)
  print(f"v_hat_img.shape {v_hat_img.shape}")
  print(f"v_hat_txt.shape {v_hat_txt.shape}")

  B, C, H, W = 4, 1024, 8, 8
  L, TH = 7, 512
  TD = 16
  x_img_t = torch.rand(B, C, H, W, device='cpu')
  time_emb = torch.rand(B, TD, device='cpu')
  x_txt_t = torch.rand(B, L, TH, device='cpu') # (B, L, TH)

  model = FiLM(txt_dim=TH, img_dim=C, time_dim=TD, hidden_dim=1024)
  pooled_txt_toks = x_txt_t.mean(1) # (B, TH)
  out = model(pooled_txt_toks, x_img_t, time_emb)
  print(f"out.shape {out.shape}")


if __name__ == "__main__":
  main()