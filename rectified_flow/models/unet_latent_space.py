import torch
import torch.nn as nn

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


class ResnetBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, time_dim, p_dropout=None
    ):
        super().__init__()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, 1)
        else:
            self.skip = nn.Identity()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1) # (B, C_out, H, W)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)

        self.dropout = nn.Dropout(p=p_dropout) if p_dropout is not None else nn.Identity()

        self.time_proj = nn.Linear(time_dim, out_channels)

        self.norm1 = nn.GroupNorm(num_groups=min(32, out_channels//2 or 1), num_channels=out_channels)
        self.norm2 = nn.GroupNorm(num_groups=min(32, out_channels//2 or 1), num_channels=out_channels)


    def forward(self, x, t_emb):
        identity = self.skip(x)
        h = self.norm1(self.conv1(x))
        t = self.time_proj(t_emb)
        t = t[:, :, None, None]
        h = h + t
        h = nn.SiLU(inplace=True)(h)
        h = self.norm2(self.conv2(h))
        h = self.dropout(h)
        out = nn.SiLU(inplace=True)(h + identity)
        return out


class UNetLatentSpace(nn.Module):
    def __init__(
            self,
            in_channels,
            time_dim=128,
            p_dropout=None
    ):
        super().__init__()
        self.down1 = ResnetBlock(in_channels, 128, time_dim, p_dropout)        # (64, 128, 128)
        self.pool1 = nn.MaxPool2d(2)                                              # (64, 64, 64)
        self.down2 = ResnetBlock(128, 256, time_dim, p_dropout)                # (128, 64, 64)
        self.pool2 = nn.MaxPool2d(2)                                              # (128, 32, 32)
        self.down3 = ResnetBlock(256, 512, time_dim, p_dropout)               # (256, 32, 32)
        self.pool3 = nn.MaxPool2d(2)                                              # (256, 16, 16)
        self.down4 = ResnetBlock(512, 1024, time_dim, p_dropout)               # (512, 16, 16)
        self.pool4 = nn.MaxPool2d(2)                                              # (512, 8, 8)

        self.middle = ResnetBlock(1024, 2048, time_dim, p_dropout)             # (1024, 8, 8)

        self.up4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)                     # (512, 16, 16)
        self.conv4 = ResnetBlock(2048, 1024, time_dim, p_dropout) # (512, 16, 16)
        self.up3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)                      # (256, 32, 32)
        self.conv3 = ResnetBlock(1024, 512, time_dim, p_dropout)  # (256, 32, 32)
        self.up2 = nn.ConvTranspose2d(512, 256, 2, stride=2)                      # (128, 64, 64)
        self.conv2 = ResnetBlock(512, 256, time_dim, p_dropout)               # (128, 64, 64)
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)                       # (64, 128, 128)
        self.conv1 = ResnetBlock(256, 128, time_dim, p_dropout)                # (64, 128, 128)

        self.out = nn.Conv2d(128, in_channels, 1)                                  # (3, 128, 128)
        self.time_emb = TimeEmbedding(dim=128)
    
    def forward(self, xt, t):
        B, C, H, W = xt.shape
        t_emb = self.time_emb(t.view(B))
        d1 = self.down1(xt, t_emb)                                             # (64, 128, 128)
        d2 = self.down2(self.pool1(d1), t_emb)                                 # (128, 64, 64)
        d3 = self.down3(self.pool2(d2), t_emb)                                 # (256, 32, 32)
        d4 = self.down4(self.pool3(d3), t_emb)                                 # (512, 16, 16)

        m = self.middle(self.pool4(d4), t_emb)                                 # (1024, 8, 8)

        u4 = self.conv4(torch.cat([self.up4(m), d4], dim=1), t_emb)            # (512, 16, 16)
        u3 = self.conv3(torch.cat([self.up3(u4), d3], dim=1), t_emb)           # (256, 32, 32)
        u2 = self.conv2(torch.cat([self.up2(u3), d2], dim=1), t_emb)           # (128, 64, 64)
        u1 = self.conv1(torch.cat([self.up1(u2), d1], dim=1), t_emb)           # (64, 128, 128)

        out = self.out(u1)                                                     # (3, 128, 128)
        return out