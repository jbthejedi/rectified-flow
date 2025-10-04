import os, copy, random, wandb, torch
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
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
    print("Model overfitting begin")
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
        settings=wandb.Settings(start_method="thread"),
    )
    wandb.define_metric("epoch")
    wandb.define_metric("*", step_metric="epoch")

    print("Downloading data")
    dataset = ProjectData(config, device).dataset
    if config.do_small_sample:
        indices = random.sample(range(len(dataset)), config.sample_size_k)
        dataset = Subset(dataset, indices)
        print(len(dataset))
    train_dl = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Model / EMA / Optimizer
    img_shape = (config.num_channels, config.image_size, config.image_size)
    model = UNetPixelSpace(
        in_channels=config.num_channels,
        time_dim=config.time_dim,
        p_dropout=None
    ).to(device)

    ema = EMA(model, beta=0.9995)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    log_dict = {}
    for epoch in range(config.n_epochs):
        log_dict["epoch"] = epoch
        model.train()
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            for images, _ in pbar:
                images = images.to(device)                      # x0 in model space (likely [-1,1])
                x0 = images
                x1 = torch.randn_like(x0)                       # noise
                t  = torch.rand(x0.size(0), 1, device=device)   # (B,1)
                xt = (1 - t[:, :, None, None]) * x0 + t[:, :, None, None] * x1
                v  = x1 - x0
                v_pred = model(xt, t)

                with torch.no_grad():
                    v_mag  = v.abs().mean().item()
                    vp_mag = v_pred.abs().mean().item()
                tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")

                loss = ((v_pred - v) ** 2).mean()

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ema.update(model)
                train_loss += loss.item()

            train_loss /= len(train_dl)
            tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        if (epoch % config.inference_peek_num) == 0:
            samples = sample_batch_pixels(
                ema, batch_size=4, num_steps=300, img_shape=img_shape, device=device
            )
            if config.local_visualization is True:
                show_samples(samples, nrow=4, title="RF pixel-space samples")
            if config.write_inference_samples is True:
                log_samples_wandb(samples, nrow=4, step=epoch)
                tqdm.write("Writing image grid")

        log_dict["train/loss"] = train_loss
        wandb.log(log_dict, step=epoch, commit=True)

    tqdm.write("Done Training")


def to_01(x):
    """
    x: (B,C,H,W) in either [-1,1] or already [0,1].
    We assume training used T.Normalize(mean=0.5, std=0.5) → model space is [-1,1].
    """
    # Map to [0,1] conservatively
    x = (x.clamp(-1, 1) + 1) / 2
    return x


def to_grid(imgs, nrow=4):
    # imgs expected in model space; convert for display
    imgs01 = to_01(imgs)
    g = vutils.make_grid(imgs01.detach().cpu(), nrow=nrow, padding=2)
    return g  # (3, GH, GW)


def log_samples_wandb(samples, nrow=4, step=None, prefix="samples/"):
    grid = to_grid(samples, nrow=nrow)
    payload = {f"{prefix}flow": wandb.Image(grid, caption="RF (pixel-space)")}
    wandb.log(payload, step=step)


def show_samples(samples, nrow=4, title="RF (pixel-space)"):
    grid = to_grid(samples, nrow=nrow)
    plt.figure(figsize=(6, 6))
    plt.imshow(grid.permute(1, 2, 0).numpy())
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def cosine_ts(num_steps, device):
    import math
    s = 0.008
    i = torch.arange(num_steps, device=device)
    f = lambda k: torch.cos(((k / num_steps + s) / (1 + s)) * math.pi / 2) ** 2
    t = f(i)                       # t[0] ~ 1.0  →  t[-1] ~ 0.0
    dt = f(i + 1) - f(i)           # negative steps (flow toward data)
    return t, dt


@torch.no_grad()
def sample_batch_pixels(ema, batch_size=16, num_steps=300, img_shape=(3, 128, 128), device="cpu"):
    """
    Integrate x' = v_theta(x, t) from t=1 → 0 in pixel space.
    Start at N(0, I) in model space (i.e., same normalization as training).
    Returns images in model space; use to_01(...) for display.
    """
    C, H, W = img_shape
    x = torch.randn(batch_size, C, H, W, device=device)  # pixel-space noise

    t_vals, dts = cosine_ts(num_steps=num_steps, device=device)
    for i in range(len(t_vals) - 1):
        t = t_vals[i].expand(batch_size, 1, 1, 1)  # (B,1,1,1); your UNet expects (B,) but broadcasts internally
        v = ema.m(x, t)                            # v_theta(x_t, t)
        x = x + v * dts[i]                         # integrate toward data
    return x


class EMA:
    def __init__(self, model, beta=0.999):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters(): p.requires_grad=False
        self.beta = beta

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.m.parameters(), model.parameters()):
            p_ema.mul_(self.beta).add_(p, alpha=1-self.beta)


def main():
    env = os.environ.get("ENV", "pixel_space/local_overfit_pixel")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")
    if config.device == 'cuda':
        torch.set_float32_matmul_precision('high')
    if config.train_model:
        train_test_model(config)


def load_config(env="pixel_space/local_overfit_pixel"):
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