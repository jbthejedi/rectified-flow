import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rectified_flow.models.time import TimeEmbedding, SinusoidalTimeEmbedding


class JointEncDecMLPFusion(nn.Module):
    def __init__(
            self, img_shape=(8, 8), txt_dim=768, img_dim=4,
            hidden=256, time_dim=128, txt_token_dim=77,
            p_hidden=64,
        ):
        super().__init__()
        H, W = img_shape
        self.img_proj = nn.Linear(H*W*img_dim, p_hidden)
        self.txt_proj = nn.Sequential(
            nn.Linear(txt_dim, txt_dim // 2),
            nn.SiLU(),
            nn.Linear(txt_dim // 2, p_hidden)
        )

        self.fusion = nn.Sequential(
            nn.Linear(p_hidden*2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden)
        )

        self.time_emb = TimeEmbedding(time_dim=128)
        self.time_proj = nn.Linear(time_dim, p_hidden)

        self.img_head = nn.Linear(hidden, H*W*img_dim)
        self.txt_head = nn.Linear(hidden, txt_dim)
    
    def forward(self, x_img_t, x_txt_t, t):
        """
        x_img_t.shape = (B, IH, 8, 8), IH = image_hidden
        x_txt_t.shape = (B, L, BD), L = cw_size, BD = bert_dim
        """
        B, IH, H, W = x_img_t.shape
        # _, L, _ = x_txt_t.shape
        _, TH = x_txt_t.shape

        #### Time Embedding ####
        time_emb = self.time_emb(t.view(B))                                   # (B, time_dim)
        time_emb_im = self.time_proj(time_emb)                                      # (B, P)
        time_emb_txt = self.time_proj(time_emb)                                     # (B, P)

        #### Project Image ####
        img_tokens = x_img_t.reshape(B, -1)                                # (B, H*W*IH)
        img_tokens = self.img_proj(img_tokens) + time_emb_im                  # (B, P)

        #### Project Text ####
        txt_tokens = self.txt_proj(x_txt_t) + time_emb_txt                    # (B, P)

        #### FUSE ####
        cat = torch.cat([img_tokens, txt_tokens], dim=1)                   # (B, P+P)
        fusion = self.fusion(cat)                                          # (B, hidden)
        v_img = self.img_head(fusion).view(B, IH, H, W)                    # (B, IH*H*W)
        u_txt = self.txt_head(fusion)                                      # (B, L)

        return v_img, u_txt