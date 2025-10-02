import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rectified_flow.models.time import TimeEmbedding


class SimpleCrossAttentionModel(nn.Module):
    def __init__(
            self, img_shape=(8, 8), txt_dim=768, img_dim=4, hidden=256, time_dim=128
        ):
        super().__init__()
        H, W = img_shape
        self.img_proj = nn.Linear(img_dim, hidden)
        self.txt_proj = nn.Linear(txt_dim, hidden)


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
        t_emb = self.time_proj(t_emb)                                    # (B, hidden)

        #### Project Image ####
        img_tokens = x_img_t.permute(0, 2, 3, 1).view(B, -1, IH)         # (B, H*W, IH)
        img_tokens = self.img_proj(img_tokens)                           # (B, H*W, hidden) 

        #### Project Text ####
        txt_tokens = self.txt_proj(x_txt_t) + t_emb[:, None, :]          # (B, L, hidden)


        out = None
        return out