import os
import random
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import torch
import torch.optim as optim

import torchvision.utils as vutils
from torchvision import transforms as T
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from rectified_flow.models.unet_pixel_space import *
from rectified_flow.data.datamodule_recover import ProjectData
from diffusers import AutoencoderKL


device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_test_model(config):
    dataset = ProjectData(config, device).dataset
    if config.do_small_sample:
        indices = random.sample(range(len(dataset)), config.sample_size_k)
        dataset = Subset(dataset, indices)
        print(len(dataset))
    train_len = int(len(dataset) * config.p_train_len)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dl = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_dl   = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    # Training loop
    img_shape = (config.num_channels, config.image_size, config.image_size)
    model = UNetPixelSpace(in_channels=config.num_channels, time_dim=128, p_dropout=None).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(config.n_epochs):
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            for x0, _ in pbar:                                                  # (B, 1, 28, 28)
                B, C, H, W = x0.shape
                x0 = x0.to(device)                                              # (B, 1, 28, 28)

                xT = torch.randn_like(x0)                                       # (B, 1, 28, 28)
                t = torch.rand(x0.size(0), 1, device=device)                    # (B, 1)
                xt = (1 - t[:, :, None, None]) * x0 + t[:, :, None, None] * xT  # (B, 1, 28, 28)
                v = xT - x0                                                     # (B, 1, 28, 28)

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
                for x0, _ in pbar:                                                         # (B, 1, 28, 28)
                    B, C, H, W = x0.shape
                    x0 = x0.to(device)                                                     # (B, 1, 28, 28)

                    xT = torch.randn_like(x0)                                              # (B, 1, 28, 28)
                    t = torch.rand(x0.size(0), 1, device=device)                           # (B, 1)
                    xt = (1 - t[:, :, None, None]) * x0 + t[:, :, None, None] * xT         # (B, 1, 28, 28)
                    v = xT - x0                                                            # (B, 1, 28, 28)

                    v_pred = model(xt, t)

                    loss = ((v_pred - v)**2).mean()

                    val_loss += loss.item()
                val_loss /= len(val_dl)
                print(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")
    
        tqdm.write("Showing inference samples...")
        with torch.no_grad():
            model.eval()
            imgs = sample_batch(model, batch_size=4, num_steps=50, img_shape=img_shape)

        # make a nice 4x4 grid
        grid = vutils.make_grid(imgs.cpu(), nrow=4, normalize=True, pad_value=1.0)

        plt.figure(figsize=(6,6))
        plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
        plt.axis("off")
        plt.show()


def sample_t(batch_size, device, schedule="uniform"):
    if schedule == "uniform":
        t = torch.rand(batch_size, device=device)

    elif schedule == "cosine":
        u = torch.rand(batch_size, device=device)
        t = torch.sin(0.5 * math.pi * u) ** 2

    elif schedule == "logit_normal":
        z = torch.randn(batch_size, device=device)
        t = torch.sigmoid(1.5 * z)  # 1.5 = sharpness factor

    return t[:, None, None, None]  # shape (B,1,1,1)


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
    env = os.environ.get("ENV", "pixel_space/local_overfit")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")
    if config.device == 'cuda':
        torch.set_float32_matmul_precision('high')
    if config.train_model:
        train_test_model(config)


def load_config(env="pixel_space/local_overfit"):
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