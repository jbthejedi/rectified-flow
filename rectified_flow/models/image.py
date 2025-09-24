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
                nn.ReLU(),
            )

        self.down1 = conv_block(C, 32)
        self.down2 = conv_block(32, 64)
        self.mid   = conv_block(64, 64)

        # Here the upsampling just reduces channels but doesn't increase H/W
        self.up_conv1   = conv_block(64+32, 32)  # concat m and d1
        self.up_conv2   = conv_block(32, 16)     # just process 

        self.final = nn.Conv2d(16, C, 1)

        self.time_to_feat = nn.Linear(time_dim, 64)

    def forward(self, x, t):
        B, C, H, W = x.shape                                 # (B, 1, 28, 28) MNIST
        t_emb = self.time_emb(t.view(B))                     # (B, time_dim)
        t_bias = self.time_to_feat(t_emb)[:, :, None, None]  # (B, 64, 1, 1)

        # Encoder
        d1 = self.down1(x)                                   # (B, 32, 28, 28)
        d2 = self.down2(nn.MaxPool2d(2)(d1))                 # (B, 64, 14, 14)

        # Bottleneck
        plus = d2 + t_bias                                   # (B, 64, 14, 14)
        m  = self.mid(plus)                                  # (B, 64, 14, 14)

        # Decoder
        up = nn.Upsample(scale_factor=2)(m)                  # (B, 64, 28, 28)
        cat = torch.cat([up, d1], dim=1)                     # (B, 64+32, 28, 28)
        uc1 = self.up_conv1(cat)                             # (B, 32, 28, 28)
        uc2 = self.up_conv2(uc1)                             # (B, 16, 28, 28)
        out = self.final(uc2)                                # (B, 1, 28, 28) MNIST
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
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout = nn.Dropout(p=p_dropout) if p_dropout is not None else nn.Identity()

        self.time_proj = nn.Linear(time_dim, out_channels)

    def forward(self, x, t_emb):
        identity = self.skip(x)
        h = self.bn1(self.conv1(x))
        t = self.time_proj(t_emb)
        t = t[:, :, None, None]
        h = h + t
        h = torch.relu(h)
        h = self.bn2(self.conv2(h))
        h = self.dropout(h)
        out = torch.relu(h + identity)
        return out


class UNet(nn.Module):
    def __init__(
            self,
            in_channels,
            config,
            time_dim=128,
    ):
        super().__init__()
        self.down1 = ResnetBlock(in_channels, 64, time_dim, p_dropout=0.1)        # (64, 128, 128)
        self.pool1 = nn.MaxPool2d(2)                                              # (64, 64, 64)
        self.down2 = ResnetBlock(64, 128, time_dim, p_dropout=0.1)                # (128, 64, 64)
        self.pool2 = nn.MaxPool2d(2)                                              # (128, 32, 32)
        self.down3 = ResnetBlock(128, 256, time_dim, p_dropout=0.1)               # (256, 32, 32)
        self.pool3 = nn.MaxPool2d(2)                                              # (256, 16, 16)
        self.down4 = ResnetBlock(256, 512, time_dim, p_dropout=0.1)               # (512, 16, 16)
        self.pool4 = nn.MaxPool2d(2)                                              # (512, 8, 8)

        self.middle = ResnetBlock(512, 1024, time_dim, p_dropout=0.1)             # (1024, 8, 8)

        self.up4 = nn.ConvTranspose2d(1024, 512, 2, stride=2)                     # (512, 16, 16)
        self.conv4 = ResnetBlock(1024, 512, time_dim, p_dropout=config.p_dropout) # (512, 16, 16)
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)                      # (256, 32, 32)
        self.conv3 = ResnetBlock(512, 256, time_dim, p_dropout=config.p_dropout)  # (256, 32, 32)
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)                      # (128, 64, 64)
        self.conv2 = ResnetBlock(256, 128, time_dim, p_dropout=0.1)               # (128, 64, 64)
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)                       # (64, 128, 128)
        self.conv1 = ResnetBlock(128, 64, time_dim, p_dropout=0.1)                # (64, 128, 128)

        self.out = nn.Conv2d(64, in_channels, 1)                                  # (3, 128, 128)
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
