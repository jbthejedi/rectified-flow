import torch
import torch.nn as nn

import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        t: shape (B,) or (B,1) with values in [0,1]
        returns: (B, dim)
        """
        half_dim = self.dim // 2
        # exponential frequency scale (log-spaced)
        freqs = torch.exp(torch.linspace(0, math.log(10000), half_dim, device=t.device))
        # (B,1) * (half_dim,) → (B, half_dim)
        # args = t * freqs
        args = t[:, None] * freqs[None, :]                          # (B, half_dim)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1) # (B, dim)
        return emb

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int = 128):
        super().__init__()
        self.sin_emb = SinusoidalTimeEmbedding(dim) # (B, 128)
        self.mlp = nn.Sequential(                   # (B, 128)
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # dim = t.shape(1)
        if t.ndim > 1:
            t = t.view(t.shape[0])
        emb = self.sin_emb(t) # (B, dim)
        out = self.mlp(emb)   # (B, dim)
        return out


class SimpleConvModel(nn.Module):
    def __init__(self, img_shape):
        super().__init__()
        C, H, W = img_shape

        self.time_mlp1 = TimeEmbedding(128)

        self.to_bias1 = nn.Linear(128, 64)
        self.to_bias2 = nn.Linear(128, 28)
        self.to_bias3 = nn.Linear(128, 16)
        self.to_bias4 = nn.Linear(128, 1)


        self.conv1 = nn.Conv2d(C, 64, kernel_size=3, padding=1) 
        self.conv2 = nn.Conv2d(64, 28, kernel_size=3, padding=1) 
        self.conv3 = nn.Conv2d(28, 16, kernel_size=3, padding=1) 
        self.conv4 = nn.Conv2d(16, C, kernel_size=3, padding=1) 

        self.img_shape = img_shape

    def forward(self, x_t, t):
        # x_t.shape (B, 1, 28, 28)
        # t.shape (B, 1, 1, 1)
        
        B, C, H, W = x_t.shape
        t_embed = self.time_mlp1(t)                      # (B, 64) 

        x = self.conv1(x_t)                              # (B, 64, 28, 28)
        x = x + self.to_bias1(t_embed)[:, :, None, None] # (B, 64, 28, 28)
        x = torch.relu(x)                                # (B, 64, 28, 28)
        
        x = self.conv2(x)                                # (B, 28, 28, 28)
        x = x + self.to_bias2(t_embed)[:, :, None, None] # (B, 28, 1, 1)
        x = torch.relu(x)

        x = self.conv3(x)                                # (B, 16, 28, 28)
        x = x + self.to_bias3(t_embed)[:, :, None, None] # (B, 16, 28, 28)
        x = torch.relu(x)

        x = self.conv4(x)                                # (B, 1, 28, 28)
        x = x + self.to_bias4(t_embed)[:, :, None, None] # (B, 1, 28, 28)

        # out = x.view(B, *self.img_shape)
        return x


class TinyUNet(nn.Module):
    def __init__(self, img_shape, time_dim=128):
        super().__init__()
        C, H, W = img_shape
        self.time_emb = TimeEmbedding(time_dim)

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(out_ch, out_ch, 3, padding=1),
                nn.ReLU()
            )

        self.down1 = conv_block(C, 32)
        self.down2 = conv_block(32, 64)
        self.mid   = conv_block(64, 64)
        self.up1   = conv_block(64+32, 32)  # concat m↑ and d1
        self.up2   = conv_block(32, 16)     # just process 
        self.final = nn.Conv2d(16, C, 1)

        self.time_to_feat = nn.Linear(time_dim, 64)

    def forward(self, x, t):
        B, C, H, W = x.shape
        t_emb = self.time_emb(t.view(B))
        t_bias = self.time_to_feat(t_emb)[:, :, None, None]

        # Encoder
        d1 = self.down1(x)                        # (B, 32, 28, 28)
        d2 = self.down2(nn.MaxPool2d(2)(d1))      # (B, 64, 14, 14)

        # Bottleneck
        m  = self.mid(d2 + t_bias)                # (B, 64, 14, 14)

        # Decoder
        u1 = self.up1(torch.cat([nn.Upsample(scale_factor=2)(m), d1], dim=1))
        u2 = self.up2(u1)                         # no skip left
        out = self.final(u2)
        return out