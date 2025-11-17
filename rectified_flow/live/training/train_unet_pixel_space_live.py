import os, copy, random, wandb, torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as vutils

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from rectified_flow.data.datamodule_recover import ProjectData
from rectified_flow.live.models.unet_pixel_space_live import UNetJoint

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
  train_split = int(len(dataset) * config.p_train_len)
  train_set, val_set = random_split(dataset, [train_split, len(dataset) - train_split])
  train_dl = DataLoader(
      train_set,
      batch_size=config.batch_size,
      shuffle=True,
      num_workers=config.num_workers,
      pin_memory=config.pin_memory,
  )
  val_dl = DataLoader(
      val_set,
      batch_size=config.batch_size,
      shuffle=False,
      num_workers=config.num_workers,
      pin_memory=config.pin_memory,
  )
  print(f"train len {len(train_dl)}")
  print(f"val len {len(val_dl)}")

  # Model / Optimizer
  model = UNetJoint(in_ch=config.num_channels,
                         time_dim=config.time_dim, p_dropout=0.1).to(device)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

  log_dict = {}
  for epoch in range(1, config.n_epochs+1):
    tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
    model.train()
    train_epoch_loss = 0.0
    with tqdm(train_dl, desc="Training") as pbar:
      for images, _ in pbar:
        v_star, v_pred = compute_data(images, model)
        if config.debug is True: print_mag(v_star, v_pred)
        loss = ((v_pred - v_star)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
      train_epoch_loss /= len(train_dl)
      log_dict["train_loss"] = train_epoch_loss
      tqdm.write(f"Train Loss {train_epoch_loss}")

    model.eval()
    with torch.no_grad():
      val_epoch_loss = 0.0
      with tqdm(val_dl, desc="Val") as pbar:
        for images, _ in pbar:
          v_star, v_pred = compute_data(images, model)
          if config.debug is True: print_mag(v_star, v_pred)
          loss = ((v_pred - v_star)**2).mean()
          val_epoch_loss += loss.item()
        val_epoch_loss /= len(val_dl)
        log_dict["val_loss"] = val_epoch_loss
        tqdm.write(f"Val {val_epoch_loss}")
    
    if epoch % config.inference_peek_num == 0:
      img_shape = (config.num_channels, config.image_size, config.image_size)
      samples = sample_batch_pixels(model, batch_size=4, img_shape=img_shape, num_steps=config.num_steps)
      log_samples_wandb(samples, step=epoch)

      if config.local_visualization is True:
        tqdm.write("Show images")
        show_samples(samples)

    wandb.log(log_dict, step=epoch, commit=True)


def print_mag(v_star : torch.Tensor, v_pred):
  vs_mag = v_star.abs().mean().item()
  vp_mag = v_pred.abs().mean().item()
  tqdm.write(f"|v*| {vs_mag:.3f}\t|v^| {vp_mag:.3f}")


def compute_data(images, model):
  x0 = images.to(device)                   # (B, C, H, W)
  x1 = torch.randn_like(x0, device=device) # (B, C, H, W)
  B = x0.size(0)
  t = torch.rand(B, device=device)         # (B)

  # t[:, None, None, None].shape = (B, 1, 1, 1)
  xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1 # (B, C, H, W)
  v_star = x1 - x0
  v_pred = model(xt, t)
  return v_star, v_pred


@torch.no_grad()
def sample_batch_pixels(model, batch_size=1, num_steps=100, img_shape=(3, 128, 128)):
  """
  Inference
  """
  tqdm.write("Performing Inference")
  B, C, H, W = batch_size, *img_shape
  x = torch.randn(B, C, H, W, device=device) # (B, C, H, W)
  t_vals = torch.linspace(1, 0, steps=num_steps, device=device)
  # dt = 1.0 / num_steps
  dts = t_vals[1:] - t_vals[:-1]
  for i in range(len(t_vals) - 1):
    x = x + model(x, t_vals[i].repeat(B)) * dts[i]
  return x


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


def load_config(env="live/pixel_space/local_train_pixel"):
  base_config = OmegaConf.load("config/live/pixel_space/base.yaml")

  env_path = f"config/{env}.yaml"
  if os.path.exists(env_path):
    env_config = OmegaConf.load(env_path)
    # Merges env_config into base_config (env overrides base)
    config = OmegaConf.merge(base_config, env_config)
  else:
    config = base_config
  return config


def main():
  env = os.environ.get("ENV", "live/pixel_space/local_train_pixel")
  print(f"env={env}")
  config = load_config(env)
  print("Configuration loaded")
  config.device, config.env = device, env
  print(f"Seed {config.seed} Device {config.device}")
  if config.device == 'cuda':
    torch.set_float32_matmul_precision('high')
  if config.train_model:
    train_test_model(config)


if __name__ == '__main__':
  main()
