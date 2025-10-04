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

    # Training loop
    img_shape = (config.num_channels, config.image_size, config.image_size)
    model = UNetPixelSpace(in_channels=config.num_channels, time_dim=128, p_dropout=None).to(device)
    ema = EMA(model, beta=0.9995)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #### DIFFUSERS/AUTOENCODERKL #######
    aekl = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    aekl.eval()
    for p in aekl.parameters():
        p.requires_grad = False

    if config.do_round_trip is True:
        round_trip(aekl, train_dl)
        print("Round Trip Finished")
    if config.do_decode_latent is True:
        decode_latent_sample(aekl, config)
        print("Decode Latent Finished")
    else:
        log_dict = {}
        for epoch in range(config.n_epochs):
            log_dict["epoch"] = epoch
            model.train()
            with tqdm(train_dl, desc="Training") as pbar:
                train_loss = 0.0
                for x0, _ in pbar:                                 # (B, 1, 28, 28)
                    B, C, H, W = x0.shape
                    x0 = x0.to(device)                                                     # (B, 1, 28, 28)
                    with torch.no_grad():
                        posterior = aekl.encode(x0).latent_dist
                        x0_latent = posterior.mean * aekl.config.scaling_factor             # (B, 4, H//8, W//8)

                    xT = torch.randn_like(x0_latent)                                       # (B, 1, 28, 28)
                    t = torch.rand(x0.size(0), 1, device=device)                           # (B, 1)
                    xt = (1 - t[:, :, None, None]) * x0_latent + t[:, :, None, None] * xT  # (B, 1, 28, 28)
                    v = xT - x0_latent                                                     # (B, 1, 28, 28)
                    v_pred = model(xt, t)

                    with torch.no_grad():
                        v_mag  = v.abs().mean().item()
                        vp_mag = v_pred.abs().mean().item()
                    tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")

                    loss = ((v_pred - v)**2).mean()

                    optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    ema.update(model)
                    train_loss += loss.item()

                train_loss /= len(train_dl)
                tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

            if (epoch % config.inference_peek_num == 0):
                flow, mush = sample_batch_vae(ema, aekl, batch_size=4, num_steps=300, img_shape=img_shape)

                if config.local_visualization is True:
                    show_flow_vs_mush(flow, mush)

                if config.write_inference_samples is True:
                    log_flow_vs_mush_wandb(mush, flow, originals=None, nrow=4, step=epoch)
                    tqdm.write("Writing image grid")

            log_dict["train/loss"] = train_loss
            wandb.log(log_dict, step=epoch, commit=True)
        tqdm.write("Done Training")


def to_grid(imgs, nrow=4):
    # imgs: (B, 3, H, W) in [0,1]
    g = vutils.make_grid(imgs.detach().cpu().clamp(0,1), nrow=nrow, padding=2)
    return g  # (3, GH, GW)


def log_flow_vs_mush_wandb(mush, flow, originals=None, nrow=4, step=None, prefix="samples/"):
    mush_grid = to_grid(mush, nrow)
    flow_grid = to_grid(flow, nrow)

    payload = {
        f"{prefix}mush": wandb.Image(mush_grid, caption="No flow (decode(randn))"),
        f"{prefix}flow": wandb.Image(flow_grid, caption="With flow"),
    }
    if originals is not None:
        orig_grid = to_grid(originals, nrow)
        payload[f"{prefix}originals"] = wandb.Image(orig_grid, caption="Round-trip originals")

    wandb.log(payload, step=step)


def decode_latent_sample(aekl : AutoencoderKL, config):
    B = 4
    z_unscaled = torch.randn(B, config.latent_channels,
                             config.image_size // 8, config.image_size // 8,
                             device=device)
                            
    with torch.no_grad():
        imgs = aekl.decode(z_unscaled / aekl.config.scaling_factor).sample
        imgs = (imgs.clamp(-1, 1) + 1) / 2

    grid = vutils.make_grid(imgs, nrow=B, normalize=True, pad_value=1.0)
    plt.figure(figsize=(4, 4))
    plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    plt.axis("off")
    plt.show()


def round_trip(aekl, train_dl):
    x_vis = next(iter(train_dl))[0][:8].to(device)
    with torch.no_grad():
        post = aekl.encode(x_vis).latent_dist
        z = post.mean * aekl.config.scaling_factor
        xr = aekl.decode(z / aekl.config.scaling_factor).sample
    viz = (xr.clamp(-1,1)+1)/2

    grid = vutils.make_grid(viz.cpu(), nrow=4, padding=2, normalize=False)

    plt.figure(figsize=(8, 8))
    plt.imshow(grid.permute(1, 2, 0).numpy())   # permute to HWC for matplotlib
    plt.axis("off")
    plt.show()


def show_flow_vs_mush(flow, mush, originals=None, nrow=4, title_left="No flow (mush)", title_right="With flow"):
    """
    mush, flow, originals: (B, 3, H, W) in [0,1] on any device
    """
    def to_grid(x):
        x = x.detach().cpu().clamp(0,1)
        return vutils.make_grid(x, nrow=nrow, padding=2)

    grids = [to_grid(mush), to_grid(flow)]
    titles = [title_left, title_right]

    if originals is not None:
        grids = [to_grid(originals)] + grids
        titles = ["Round-trip originals"] + titles

    cols = len(grids)
    plt.figure(figsize=(6*cols, 6))
    for i, (g, t) in enumerate(zip(grids, titles), 1):
        plt.subplot(1, cols, i)
        plt.imshow(g.permute(1, 2, 0).numpy())
        plt.title(t)
        plt.axis("off")
    plt.tight_layout()
    plt.show()



class EMA:
    def __init__(self, model, beta=0.999):
        self.m = copy.deepcopy(model).eval()
        for p in self.m.parameters(): p.requires_grad=False
        self.beta = beta

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.m.parameters(), model.parameters()):
            p_ema.mul_(self.beta).add_(p, alpha=1-self.beta)


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


def sample_batch_vae(ema, vae, batch_size=16, num_steps=300, img_shape=(3, 128, 128)):
    latent_shape = (4, img_shape[1] // 8, img_shape[2] // 8)   # 4 x H/8 x W/8

    # Start from Gaussian noise in latent space
    z0 = torch.randn(batch_size, *latent_shape, device=device)
    with torch.no_grad():
        mush = vae.decode(z0 / vae.config.scaling_factor).sample
        x = z0.clone()
        t_vals, dts = cosine_ts(num_steps=num_steps, device=device)
        for i in range(len(t_vals)-1):
            t = t_vals[i].expand(batch_size,1,1,1)
            x = x + ema.m(x, t) * dts[i]
        flow = vae.decode(x / vae.config.scaling_factor).sample
        flow = (flow.clamp(-1, 1) + 1) / 2  # map back to [0,1]

    return flow, mush


def cosine_ts(num_steps, device):
    s = 0.008
    i = torch.arange(num_steps, device=device)
    f = lambda k: torch.cos(( (k/num_steps + s)/(1+s) )*math.pi/2)**2
    t = f(i); dt = f(i+1)-f(i)
    return t, dt


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