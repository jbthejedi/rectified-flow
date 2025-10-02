import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from rectified_flow.models.time import TimeEmbedding, SinusoidalTimeEmbedding

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


class JointRFSimpleFusion(nn.Module):
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