import torch
import torch.nn as nn
from rectified_flow.models.time import TimeEmbedding
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
    self.attn = MHSA(c, heads=heads)  # your MHSA from earlier

  def forward(self, x):
    return self.attn(x)


class DoubleResnetBlock(nn.Module):

  def __init__(self, in_ch, out_ch, time_dim, p_dropout=None):
    super().__init__()
    self.block1 = ResnetBlock(in_ch, out_ch, time_dim, p_dropout)
    self.block2 = ResnetBlock(out_ch, out_ch, time_dim, p_dropout)

  def forward(self, xt, t):
    h = self.block1(xt, t)
    return self.block2(h, t)


class FiLMAdapter(nn.Module):

  def __init__(self, txt_dim, time_dim, target_dim):
    super().__init__()
    self.proj = nn.Linear(txt_dim + time_dim, 2 * target_dim)
    nn.init.zeros_(self.proj.weight)
    nn.init.zeros_(self.proj.bias)

  def forward(self, txt_emb, t_emb):
    # txt_emb.shape                                    # (, txt_dim)
    h = torch.cat([txt_emb, t_emb], dim=-1)
    gamma, beta = self.proj(h).chunk(2, dim=-1)  # (, target_dim), (, target_dim)
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
    nn.init.zeros_(self.proj_out.bias)
    nn.init.xavier_uniform_(self.proj_out.weight, gain=1e-3)  # small, but nonzero
    self.ca_gate = nn.Parameter(torch.tensor(0.0))

    self.nh = n_heads

  def forward(self, img_toks, txt_toks, attn_mask=None):
    B, P, _ = img_toks.shape                            # (, P, img_dim), P = H*W
    _, L, _ = txt_toks.shape                            # (, L, txt_dim)
    # txt_emb.shape                                             (, 1, txt_dim)

    Q = self.to_q(img_toks).view(B, P, self.nh, self.d).transpose(2, 1)       # (, nh, P, D)
    K = self.to_k(txt_toks).view(B, L, self.nh, self.d).transpose(2, 1)       # (, nh, 1, D)
    V = self.to_v(txt_toks).view(B, L, self.nh, self.d).transpose(2, 1)       # (, nh, 1, D)

    attn = Q @ K.transpose(-2, -1) * (self.d ** -0.5)                   # (, nh, P, 1)
    if attn_mask is not None:
      # attn_mask: (B, L) with 1=keep, 0=pad
      mask = (~attn_mask.bool()).unsqueeze(1).unsqueeze(2)  # (B,1,1,L)
      attn = attn.masked_fill(mask, float("-inf"))

    attn = torch.softmax(attn, dim=-1)                                  # (, nh, P, 1)
    out = attn @ V                                                      # (, nh, P, D)
    out = out.transpose(1, 2).contiguous().view(B, P, self.d * self.nh) # (, P, img_dim), img_dim = D*n_heads
    return img_toks + self.ca_gate * self.proj_out(out)                                     # (, P, img_dim)


class TxtVelocityHead(nn.Module):

  def __init__(self, txt_dim, img_dim, hidden=512):
    super().__init__()
    self.proj = nn.Sequential(
      nn.Linear(txt_dim + img_dim, hidden),
      nn.SiLU(),
      nn.Linear(hidden, txt_dim)
    )

  def forward(self, txt_toks, img_ctx):
    # txt_toks: [B, L, Ctxt], img_ctx: [B, Cimg]
    B, L, Ctxt = txt_toks.shape
    h = torch.cat([txt_toks, img_ctx[:, None, :].expand(B, L, img_ctx.size(1))], dim=-1)
    return self.proj(h)  # [B, L, Ctxt]


class UNetPixelJointCAFiLM(nn.Module):

  def __init__(self, in_ch=3, time_dim=128, p_dropout=None):
    super().__init__()
    self.time_img = TimeEmbedding(dim=time_dim)
    self.time_txt = TimeEmbedding(dim=time_dim)
    self.time_fuse = nn.Sequential(
        nn.Linear(2*time_dim, time_dim*2),
        nn.SiLU(),
        nn.Linear(time_dim*2, time_dim)
    )

    # Down path (keep blocks shape-preserving; downsample between levels)
    self.enc1 = DoubleResnetBlock(in_ch, 128, time_dim, p_dropout)
    # self.ds1 = Downsample(128)

    self.enc2 = DoubleResnetBlock(128, 256, time_dim, p_dropout)
    self.ds2 = Downsample(256)
    self.enc_sa2 = AttnBlock(c=256, heads=8)

    self.enc3 = DoubleResnetBlock(256, 512, time_dim, p_dropout)
    self.ds3 = Downsample(512)
    self.enc_sa3 = AttnBlock(c=512, heads=8)

    self.enc4 = DoubleResnetBlock(512, 1024, time_dim, p_dropout)
    self.ds4 = Downsample(1024)

    self.mid = ResnetBlock(1024, 1024, time_dim, p_dropout)

    # Up path (upsample + concat + block)
    self.us4 = Upsample(1024)
    self.dec4 = DoubleResnetBlock(1024 + 1024, 1024, time_dim, p_dropout)
    self.dec_sa4 = AttnBlock(c=1024, heads=8)

    self.us3 = Upsample(1024)
    self.dec3 = DoubleResnetBlock(1024 + 512, 512, time_dim, p_dropout)
    self.dec_sa3 = AttnBlock(c=512, heads=8)

    self.us2 = Upsample(512)
    self.dec2 = DoubleResnetBlock(512 + 256, 256, time_dim, p_dropout)

    self.us1 = Upsample(256)
    self.dec1 = DoubleResnetBlock(256 + 128, 128, time_dim, p_dropout)

    # Text Conditioning
    img_dim = 1024
    # clip_dim = 512
    # langvae_dim = 768
    t5_dim = 512
    self.cross_attn = CrossAttention(img_dim=img_dim, txt_dim=t5_dim, n_heads=4)
    self.film = FiLMAdapter(txt_dim=t5_dim, time_dim=time_dim ,target_dim=img_dim)

    self.image_out = nn.Conv2d(128, in_ch, 1)
    self.txt_head = TxtVelocityHead(txt_dim=t5_dim, img_dim=img_dim, hidden=t5_dim*2)

    # Learnable unconditional (null) embedding for CFG
    self.null_token = nn.Parameter(torch.zeros(1, 1, t5_dim))
    nn.init.normal_(self.null_token, std=0.02)


  def _pool_text_tokens(self, txt_seq, attn_mask=None):
    # txt_seq: (B, L, clip_dim)
    if attn_mask is None:
      return txt_seq.mean(dim=1)                      # (B, clip_dim)
    # masked mean
    weights = attn_mask.float().unsqueeze(-1)          # (B, L, 1)
    s = (txt_seq * weights).sum(dim=1)                 # (B, clip_dim)
    z = weights.sum(dim=1).clamp_min(1.0)              # (B, 1)
    return s / z


  def forward(self, x_img_t, x_txt_t, t_img, t_txt, attn_mask: torch.Tensor=None, is_uncond: torch.Tensor=None):
    """
    txt_toks.shape (B, L, CL)
    """
    txt_for_img = x_txt_t
    if is_uncond is not None and is_uncond.any():
      B, L, C = x_txt_t.shape
      m = is_uncond.bool()
      txt_for_img = txt_for_img.clone()
      txt_for_img[m] = self.null_token.expand(m.sum(), L, C)

    B = x_img_t.size(0)
    t_emb_img = self.time_img(t_img)
    t_emb_txt = self.time_txt(t_txt)
    t_emb = self.time_fuse(torch.cat([t_emb_img, t_emb_txt], dim=1))  # [B, time_dim]

    d1 = self.enc1(x_img_t, t_emb)  # 128, 32, 32
    d2 = self.enc2(d1, t_emb)  # 256, 32, 32

    h = self.enc_sa2(self.ds2(d2))
    d3 = self.enc3(h, t_emb)  # 512, 16, 16

    h = self.enc_sa3(self.ds3(d3))
    d4 = self.enc4(h, t_emb)  # 1024, 8, 8

    m = self.mid(self.ds4(d4), t_emb)  # 1024, 4, 4

    ##### FiLM #####
    _, C_m, H_m, W_m = m.shape
    pooled_text = self._pool_text_tokens(txt_for_img, attn_mask)
    γ, β = self.film(pooled_text, t_emb)                               # (, 1024), (, 1024)
    γ = γ[:, :, None, None]                                     # (, 1024, 1, 1)
    β = β[:, :, None, None]                                     # (, 1024, 1, 1)
    m = m * (1 + γ) + β                                         # (, 1024, 4, 4)

    ##### Cross Attention #####
    img_toks = m.flatten(2).permute(0, 2, 1)                    # (, 4*4, 1024)
    m = self.cross_attn(img_toks, txt_for_img, attn_mask)          # (, 4*4, 1024)
    m = m.permute(0, 2, 1).view(B, C_m, H_m, W_m)


    u4 = self.dec4(torch.cat([self.us4(m), d4], dim=1), t_emb)  # 1024,  8, 8
    u4 = self.dec_sa4(u4)
    u3 = self.dec3(torch.cat([self.us3(u4), d3], dim=1), t_emb) #  512, 16, 16
    u3 = self.dec_sa3(u3)
    u2 = self.dec2(torch.cat([self.us2(u3), d2], dim=1), t_emb) #  256, 32, 32
    u1 = self.dec1(torch.cat([u2, d1], dim=1), t_emb)           #  128, 32, 32

    # Image velocity head
    v_img_pred = self.image_out(u1)

    # Text velocity head
    img_ctx = m.mean(dim=[2,3]) # (B, C)
    v_txt_pred = self.txt_head(x_txt_t, img_ctx)
    return v_img_pred, v_txt_pred
