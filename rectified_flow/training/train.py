import os
import random
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.nn as nn
import torch.optim as optim

import torchvision.utils as vutils
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Dataset, Subset
from tqdm import tqdm
from torchinfo import summary
import math


device = 'cuda' if torch.cuda.is_available() else 'cpu'

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


class RFModel(nn.Module):
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
 

def train_test_model(config):
    dataset = datasets.MNIST(
        root=config.data_root,
        download=False,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )

    if config.do_small_sample:
        indices = random.sample(range(len(dataset)), 1000)
        dataset = Subset(dataset, indices)
        print(len(dataset))
    train_len = int(len(dataset) * config.p_train_len)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dl = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_dl   = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    # Training loop
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = RFModel(img_shape=(1, 28, 28)).to(device)
    model = TinyUNet(img_shape=(1, 28, 28)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(config.n_epochs):
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            for batch_idx, (x0, _) in enumerate(pbar):    # (B, 1, 28, 28)
                x0 = x0.to(device)                        # (B, 1, 28)
                xT = torch.randn_like(x0)                 # (B, 1, 28)
                t = torch.rand(x0.size(0), device=device) # (B,)
                t = t.view(-1, 1, 1, 1)                   # (B, 1, 1, 1)
                xt = (1 - t) * x0 + t * xT                # (B, 1, 28, 28)
                v = xT - x0                               # (B, 1, 28, 28)
                v_pred = model(xt, t)
                loss = ((v_pred - v)**2).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            train_loss /= len(train_dl)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        with tqdm(val_dl, desc="Validation") as pbar:
            with torch.no_grad():
                model.eval()
                val_loss = 0.0
                for batch_idx, (x0, _) in enumerate(pbar):  # (batch, 1, 28, 28)
                    x0 = x0.to(device)
                    xT = torch.randn_like(x0)
                    t = torch.rand(x0.size(0), device=device)
                    t = t.view(-1, 1, 1, 1)
                    xt = (1 - t) * x0 + t * xT
                    v = xT - x0
                    v_pred = model(xt, t)
                    loss = ((v_pred - v)**2).mean()

                    val_loss += loss.item()
                val_loss /= len(val_dl)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
    
    with torch.no_grad():
        model.eval()
        imgs = sample_batch(model, batch_size=16, num_steps=50, img_shape=(1, 28, 28))

    # make a nice 4x4 grid
    grid = vutils.make_grid(imgs.cpu(), nrow=4, normalize=True, pad_value=1.0)

    plt.figure(figsize=(6,6))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.show()

def sample_batch(model, batch_size=16, num_steps=200, img_shape=(1, 28, 28)):
    device = next(model.parameters()).device
    x = torch.randn(batch_size, *img_shape, device=device)  # batch of noise
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    for i in range(len(t_vals)-1):
        t = t_vals[i].expand(batch_size, 1, 1, 1)  # Correct shape
        dt = t_vals[i+1] - t_vals[i]
        v_pred = model(x, t)               # (B, C, H, W)
        x = x + v_pred * dt
    return x.clamp(0, 1)  # keep images in [0,1]


def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")
    if config.device == 'cuda':
        torch.set_float32_matmul_precision('high')

    # if config.summary:
    #     print_summary(config)
    #     exit()

    elif config.train_model:
        train_test_model(config)
    elif config.inference:
        test_model(config)


def load_config(env="local"):
    base_config = OmegaConf.load("config/base.yaml")

    env_path = f"config/{env}.yaml"
    if os.path.exists(env_path):
        env_config = OmegaConf.load(env_path)
        # Merges env_config into base_config (env overrides base)
        config = OmegaConf.merge(base_config, env_config)
    else:
        config = base_config
    return config


if __name__ == '__main__':
    main()