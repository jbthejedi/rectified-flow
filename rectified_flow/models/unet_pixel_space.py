import torch
import torch.nn as nn
from rectified_flow.models.time import *


class Downsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    def forward(self, x): return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1)
    def forward(self, x): return self.conv(self.up(x))

class ResnetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, p_dropout=None):
        super().__init__()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

        # PreAct
        self.norm1 = nn.GroupNorm(self.__gn_groups(in_ch),  in_ch)   # <- use in_ch here
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


class UNetPixelSpace(nn.Module):
    def __init__(self, in_channels=3, time_dim=128, p_dropout=None):
        super().__init__()
        self.time_emb = TimeEmbedding(dim=time_dim)

        # Down path (keep blocks shape-preserving; downsample between levels)
        self.enc1 = ResnetBlock(in_channels, 128, time_dim, p_dropout)
        self.ds1  = Downsample(128)
        self.enc2 = ResnetBlock(128, 256, time_dim, p_dropout)
        self.ds2  = Downsample(256)
        self.enc3 = ResnetBlock(256, 512, time_dim, p_dropout)
        self.ds3  = Downsample(512)
        self.enc4 = ResnetBlock(512, 1024, time_dim, p_dropout)
        self.ds4  = Downsample(1024)

        self.mid  = ResnetBlock(1024, 1024, time_dim, p_dropout)

        # Up path (upsample + concat + block)
        self.us4  = Upsample(1024)
        self.dec4 = ResnetBlock(1024+1024, 1024, time_dim, p_dropout)
        self.us3  = Upsample(1024)
        self.dec3 = ResnetBlock(1024+512, 512, time_dim, p_dropout)
        self.us2  = Upsample(512)
        self.dec2 = ResnetBlock(512+256, 256, time_dim, p_dropout)
        self.us1  = Upsample(256)
        self.dec1 = ResnetBlock(256+128, 128, time_dim, p_dropout)

        self.out  = nn.Conv2d(128, in_channels, 1)

    def forward(self, xt, t):
        B = xt.size(0)
        t_emb = self.time_emb(t.view(B))

        d1 = self.enc1(xt, t_emb)          # 128,128,128
        d2 = self.enc2(self.ds1(d1), t_emb) # 256,64,64
        d3 = self.enc3(self.ds2(d2), t_emb) # 512,32,32
        d4 = self.enc4(self.ds3(d3), t_emb) # 1024,16,16

        m  = self.mid(self.ds4(d4), t_emb)  # 1024,8,8

        u4 = self.dec4(torch.cat([self.us4(m), d4], dim=1), t_emb)  # 1024,16,16
        u3 = self.dec3(torch.cat([self.us3(u4), d3], dim=1), t_emb) # 512,32,32
        u2 = self.dec2(torch.cat([self.us2(u3), d2], dim=1), t_emb) # 256,64,64
        u1 = self.dec1(torch.cat([self.us1(u2), d1], dim=1), t_emb) # 128,128,128

        return self.out(u1)
