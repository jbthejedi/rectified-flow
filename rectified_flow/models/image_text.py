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
    def __init__(self, time_dim: int = 128):
        super().__init__()
        self.sin_emb = SinusoidalTimeEmbedding(time_dim) # (B, time_dim)
        self.mlp = nn.Sequential(                   # (B, time_dim)
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        emb = self.sin_emb(t) # (B, time_dim)
        out = self.mlp(emb)   # (B, time_dim)
        return out


class ImageEncoder(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),   # 28x28 -> 14x14
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),  # 14x14 -> 7x7
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1,1))  # (B, 64, 1, 1)
        )
        self.fc = nn.Linear(64, out_dim)

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)  # (B, out_dim)


class TextEncoder(nn.Module):
    def __init__(self, vocab_size=10, out_dim=32):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, out_dim)
    def forward(self, x):
        return self.embed(x)  # (B, out_dim)


class JointRFModelMNIST(nn.Module):
    def __init__(self, hidden_dim=32):
        super().__init__()
        self.img_enc = ImageEncoder(out_dim=hidden_dim)
        self.txt_enc = TextEncoder(vocab_size=10, out_dim=hidden_dim)
        self.time_emb = TimeEmbedding(hidden_dim)
        
        # Fusion: concatenate (img, txt, time) -> 3*hidden_dim
        fusion_dim = 3 * hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, 128),
            nn.SiLU(),
            nn.Linear(128, 128),
            nn.SiLU()
        )
        
        # Prediction heads
        self.img_head = nn.Linear(128, hidden_dim)  # predicts v_img (displacement)
        self.txt_head = nn.Linear(128, hidden_dim)  # predicts u_txt (displacement)

    def forward(self, img_feat, txt_feat, t):
        """
        hidden_dim = 32
        img_feat.shape = (B, hidden_dim)
        txt_feat.shape = (B, hidden_dim)
        t.shape        = (B, 1)
        """
        t_feat   = self.time_emb(t)                          # (B, hidden_dim)
        # t_feat = t_feat[:, :, None, None]                  # (B, hidden_dim)
        h = torch.cat([img_feat, txt_feat, t_feat], dim=-1)  # (B, hidden_dim*3)
        h = self.fusion(h)                                   # (B, 128)
        
        v_img = self.img_head(h)                             # (B, hidden_dim)
        u_txt = self.txt_head(h)                             # (B, hidden_dim)
        return v_img, u_txt


class BaselineJointModel(nn.Module):
    def __init__(
            self, img_shape=(8, 8), txt_dim=768, img_dim=4, hidden=256, time_dim=128, txt_token_dim=77
        ):
        super().__init__()
        H, W = img_shape
        self.img_proj = nn.Linear(img_dim, 1)
        self.txt_proj = nn.Linear(txt_dim, 1)

        self.fusion = nn.Sequential(
            nn.Linear(H * W + txt_token_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.time_emb = TimeEmbedding(time_dim=128)
        self.t_proj_im = nn.Linear(time_dim, H*W)
        self.t_proj_txt = nn.Linear(time_dim, txt_token_dim)
    
    def forward(self, x_img_t : torch.Tensor, x_txt_t, t):
        """
        x_img_t.shape = (B, IH, 8, 8), IH = image_hidden
        x_txt_t.shape = (B, L, BD), L = cw_size, BD = bert_dim
        """
        B, IH, _, _ = x_img_t.shape
        _, L, _ = x_txt_t.shape

        #### Time Embedding ####
        t_emb = self.time_emb(t.view(B))                                 # (B, time_dim)
        t_emb_im = self.t_proj_im(t_emb)                                    # (B, H*W)
        t_emb_txt = self.t_proj_txt(t_emb)                                    # (B, L)

        #### Project Image ####
        img_tokens = x_img_t.permute(0, 2, 3, 1).view(B, -1, IH)         # (B, H*W, IH)
        img_tokens = self.img_proj(img_tokens).squeeze(2) + t_emb_im       # (B, H*W) 

        #### Project Text ####
        txt_tokens = self.txt_proj(x_txt_t).squeeze(2) + t_emb_txt          # (B, L)

        #### FUSE ####
        cat = torch.cat([img_tokens, txt_tokens], dim=1)                 # (B, H*W+L, hidden)
        fusion = self.fusion(cat)                                        # (B, H*W+L, hidden)

        out = None
        return out


class SimpleCrossAttentionModel(nn.Module):
    def __init__(
            self, img_shape=(8, 8), txt_dim=768, img_dim=4, hidden=256, time_dim=128
        ):
        super().__init__()
        H, W = img_shape
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)

        self.fusion = nn.Sequential(
            nn.Linear(H * W + txt_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.time_emb = TimeEmbedding(time_dim=128)
        self.time_proj = nn.Linear(time_dim, hidden)
    
    def forward(self, x_img_t : torch.Tensor, x_txt_t, t):
        """
        x_img_t.shape = (B, IH, 8, 8), IH = image_hidden
        x_txt_t.shape = (B, L, BD), L = cw_size, BD = bert_dim
        """
        B, IH, _, _ = x_img_t.shape

        #### TIME EMBEDDING ####
        t_emb = self.time_emb(t.view(B))                                 # (B, time_dim)

        # Project Time
        t_emb = self.time_proj(t_emb)                                    # (B, hidden)

        #### Project Image ####
        img_tokens = x_img_t.permute(0, 2, 3, 1).view(B, -1, IH)         # (B, H*W, IH)
        img_tokens = self.img_proj(img_tokens) + t_emb[:, None, :]       # (B, H*W, hidden) 

        #### Project Text ####
        txt_tokens = self.txt_proj(x_txt_t) + t_emb[:, None, :]          # (B, L, hidden)

        #### FUSE ####
        cat = torch.cat([img_tokens, txt_tokens], dim=1)                 # (B, H*W+L, hidden)
        # fusion = self.fusion(cat)                                        # (B, H*W+L, hidden)

        out = None
        return out