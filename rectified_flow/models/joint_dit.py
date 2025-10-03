import math, torch, torch.nn as nn, torch.nn.functional as F

class AdaLN(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.norm = nn.LayerNorm(d)
        self.mod = nn.Linear(d, 2*d)  # produces scale, shift
    def forward(self, x, cond):       # cond: time embedding (and optionally pooled text)
        s, b = self.mod(cond).chunk(2, dim=-1)
        return self.norm(x) * (1 + s[:, None, :]) + b[:, None, :]

class MLP(nn.Module):
    def __init__(self, d, mlp_ratio=4):
        super().__init__()
        self.fc1 = nn.Linear(d, d*mlp_ratio)
        self.fc2 = nn.Linear(d*mlp_ratio, d)
    def forward(self, x): return self.fc2(F.gelu(self.fc1(x)))

class SelfAttn(nn.Module):
    def __init__(self, d, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nheads, batch_first=True)
    def forward(self, x):              # x: [B,N,D]
        y, _ = self.attn(x, x, x, need_weights=False)
        return y

class CrossAttn(nn.Module):
    def __init__(self, d, nheads):
        super().__init__()
        self.attn = nn.MultiheadAttention(d, nheads, batch_first=True)
    def forward(self, q, k, v, key_padding_mask=None):
        y, _ = self.attn(q, k, v, key_padding_mask=key_padding_mask, need_weights=False)
        return y

class DiTBlock(nn.Module):
    def __init__(self, d, nheads, cross=False, drop_path=0.0):
        super().__init__()
        self.cross = cross
        self.adaln1 = AdaLN(d); self.sa = SelfAttn(d, nheads)
        self.adaln2 = AdaLN(d); self.mlp = MLP(d)
        if cross:
            self.adaln_x = AdaLN(d); self.ca = CrossAttn(d, nheads)
        self.drop = nn.Dropout(drop_path)
    def forward(self, x, t_cond, tok=None, tok_mask=None):
        # Self-attn
        h = self.adaln1(x, t_cond); x = x + self.drop(self.sa(h))
        # Optional cross-attn
        if self.cross and tok is not None:
            h = self.adaln_x(x, t_cond)
            x = x + self.drop(self.ca(h, tok, tok, key_padding_mask=tok_mask))
        # MLP
        h = self.adaln2(x, t_cond); x = x + self.drop(self.mlp(h))
        return x

def sinusoidal_time(d, t):  # t in [0,1], returns [B,D]
    device = t.device
    half = d // 2
    freqs = torch.exp(torch.linspace(0, math.log(10000), half, device=device))
    args = t[:, None] * freqs[None, :]
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
    if d % 2: emb = F.pad(emb, (0,1))
    return emb

class TinyDiTLatent(nn.Module):

    def __init__(self, ih=4, h8=8, w8=8, d=256, nheads=8, depth=6, cross_at=(2,)):
        super().__init__()
        self.h8, self.w8, self.d = h8, w8, d
        self.img_in = nn.Conv2d(ih, d, 1)               # 1×1 patch embed
        self.pos = self.build_2d_sincos(d, h8, w8)      # [1,N,D]
        self.t_proj = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))
        self.tok_proj = nn.Linear(d, d)                 # project text tokens to D (set input dim later)
        self.blocks = nn.ModuleList([
            DiTBlock(d, nheads, cross=(i in set(cross_at))) for i in range(depth)
        ])
        self.head_img = nn.Linear(d, ih)
        self.txt_pool = nn.Linear(d, d)                 # attn/mean pool simplification
        self.txt_fuse = nn.Sequential(nn.Linear(d + d, d), nn.SiLU(), nn.Linear(d, 128))

    def build_2d_sincos(self, d, h, w):
        ys = torch.arange(h).float(); xs = torch.arange(w).float()
        y, x = torch.meshgrid(ys, xs, indexing='ij'); coords = torch.stack([y,x], -1)  # [H,W,2]
        freqs = torch.exp(torch.linspace(0, math.log(10000), d//4))
        def pe(a): return torch.stack([torch.sin(a[...,None]/freqs), torch.cos(a[...,None]/freqs)], -1).flatten(-2)
        pos = torch.cat([pe(coords[...,0]), pe(coords[...,1])], -1).view(1, h*w, d)
        return nn.Parameter(pos, requires_grad=False)

    def forward(self, x_img_t, t, pooled_txt_t, tok_embeds=None, tok_mask=None):
        B = x_img_t.size(0)
        # image tokens + pos
        x = self.img_in(x_img_t).permute(0,2,3,1).reshape(B, self.h8*self.w8, self.d) + self.pos.to(x_img_t.device)
        # time cond
        t_cond = self.t_proj(sinusoidal_time(self.d, t.view(B)))
        # optional text tokens
        tok = None; mask = None
        if tok_embeds is not None:
            tok = self.tok_proj(tok_embeds.detach())    # [B,L,D]
            mask = (tok_mask == 0) if tok_mask is not None else None
        # transformer
        for i, blk in enumerate(self.blocks):
            x = blk(x, t_cond, tok=tok, tok_mask=mask)
        # heads
        v_img = self.head_img(x).reshape(B, self.h8, self.w8, -1).permute(0,3,1,2)  # B,4,8,8
        img_vec = x.mean(dim=1)
        # fuse pooled text latent (flowed) with image summary to predict u_txt
        fuse = torch.cat([self.txt_pool(img_vec), pooled_txt_t], dim=-1)
        u_txt = self.txt_fuse(fuse)  # -> 128
        return v_img, u_txt
