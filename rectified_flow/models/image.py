import torch
import torch.nn as nn

import math


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
