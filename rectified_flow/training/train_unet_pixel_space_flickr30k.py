import os, copy, random, wandb, torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
from torchvision import datasets, transforms as T
import torch.optim as optim
import torchvision.utils as vutils

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from rectified_flow.models.unet_pixel_space import *
from rectified_flow.data.flickr30k_tokenized import Flickr30kTokenized
from rectified_flow.data.datamodule_recover import ProjectData

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

  tok_path = "./langvae_tokenizer_ckpt"

  mean = [0.444, 0.421, 0.384]
  std = [0.275, 0.267, 0.276]
  train_tf = T.Compose([
       T.CenterCrop(224),
       T.Resize(config.image_size),
       T.ToTensor(),
       T.Normalize(mean, std)])
  # val_tf   = T.Compose([T.CenterCrop(224), T.Resize(config.image_size), T.ToTensor(), T.Normalize(mean, std)])

  images_root = f"{config.data_root}/flickr30k/Images"
  captions_file = f"{config.data_root}/flickr30k/captions.txt"
  print("Downloading data")
  dataset = Flickr30kTokenized(
      images_root=images_root,
      captions_file=captions_file,
      tokenizer_name_or_path=tok_path,
      transform=train_tf,
      max_length=77,
  )
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

  # Model / EMA / Optimizer
  img_shape = (config.num_channels, config.image_size, config.image_size)
  model = UNetPixelSpace(in_ch=config.num_channels, time_dim=config.time_dim,
                         p_dropout=None).to(device)
  if config.load_model is True:
    model = load_model(model, config)
  optimizer = optim.Adam(model.parameters(), lr=1e-4)

  log_dict = {}
  best_val_loss = float("inf")
  for epoch in range(config.n_epochs):
    log_dict["epoch"] = epoch

    model.train()
    with tqdm(train_dl, desc="Training") as pbar:
      train_epoch_loss = 0.0
      for images, input_ids, attn_mask in pbar:
        v, v_pred = compute_data(model, images, device, amp=True)
        if config.debug is True: print_mags(v, v_pred)
        loss += ((v_pred - v)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_epoch_loss += loss.item()
      train_epoch_loss /= len(train_dl)
    tqdm.write(f"Epoch {epoch}: Train Loss = {train_epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
      with tqdm(val_dl, desc="Validation") as pbar:
        val_epoch_loss = 0.0
        for images, input_ids, attn_mask in pbar:
          v, v_pred = compute_data(model, images, device)
          if config.debug is True: print_mags(v, v_pred)
          val_epoch_loss += ((v_pred - v)**2).mean()
        val_epoch_loss /= len(val_dl)
        tqdm.write(f"Epoch {epoch}: Val Loss = {val_epoch_loss:.4f}")

    if (epoch % config.inference_peek_num) == 0:
      samples = sample_batch_pixels(model, batch_size=4, num_steps=config.num_sample_steps,
                                    img_shape=img_shape, device=device)
      if config.local_visualization is True:
        show_samples(samples, nrow=4, title="RF pixel-space samples")
      if config.write_inference_samples is True:
        log_samples_wandb(samples, nrow=4, step=epoch)
        tqdm.write("Writing image grid")

    log_dict["train/loss"] = train_epoch_loss
    wandb.log(log_dict, step=epoch, commit=True)

    if config.save_model:
      tqdm.write("Saving model")
      best_val_loss = save_and_log_model(model, config, best_val_loss, val_epoch_loss)
  tqdm.write("Done Training")


def load_model(model, config):
  api = wandb.Api()
  artifact_name = config.artifact_name
  try:
    print("Setup artifact")
    artifact = api.artifact(artifact_name, type='model')
    print("Downloading model")
    artifact_dir = artifact.download()

    model = UNetPixelSpace(in_ch=config.num_channels,
                           time_dim=config.time_dim,
                           p_dropout=None).to(device)
    model.load_state_dict(torch.load(f"{artifact_dir}/best-model.pth", map_location="cpu"),
                          strict=False)
  except wandb.CommError as e:
    print(f"Artifact not found: {artifact_name}")
    print(f"Error: {e}")
  print("Model loaded successfully.")
  return model


def compute_data(model, images, device, amp=False):
  images = images.to(device, non_blocking=True)
  x0 = images
  x1 = torch.randn_like(x0)  # noise
  t = torch.rand(x0.size(0), 1, device=device)  # (B,1)
  xt = (1 - t[:, :, None, None]) * x0 + t[:, :, None, None] * x1
  v = x1 - x0
  v_pred = model(xt, t)
  return v, v_pred


@torch.no_grad()
@torch.inference_mode()
def sample_batch_pixels(model, batch_size=4, num_steps=300, img_shape=(3, 128, 128), device="cpu"):
  """
    Integrate x' = v_theta(x, t) from t=1 → 0 in pixel space.
    Start at N(0, I) in model space (i.e., same normalization as training).
    Returns images in model space; use to_01(...) for display.
    """
  C, H, W = img_shape
  x = torch.randn(batch_size, C, H, W, device=device)  # pixel-space noise

  t_vals = torch.linspace(1, 0, steps=num_steps, device=device)
  dts = t_vals[1:] - t_vals[:-1]
  model.eval()
  for i in range(len(t_vals) - 1):
    t = t_vals[i].repeat(batch_size, 1)
    v = model(x, t)
    x = x + v * dts[i]
  return x


def save_and_log_model(model, config, best_val_loss, val_loss, filename="best-model.pth"):
  if val_loss < best_val_loss:
    best_val_loss = val_loss
    tqdm.write(f"New best val loss: {best_val_loss:.4f} — uploading to wandb")

    # Save locally
    if config.compile:
      torch.save(model._orig_mod.state_dict(), filename)
    else:
      torch.save(model.state_dict(), filename)

    # Create or overwrite wandb artifact
    artifact = wandb.Artifact(name=f"{config.name}-best-model",
                              type="model",
                              description="Continuously updated best model")
    artifact.add_file(filename)
    wandb.log_artifact(artifact, aliases=["latest", "best"])

  return best_val_loss


@torch.no_grad()
def print_mags(v, v_pred):
  v_mag = v.abs().mean().item()
  vp_mag = v_pred.abs().mean().item()
  tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")


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


def denorm(x, mean, std):
  mean = torch.tensor(mean, device=x.device)[None, :, None, None]
  std = torch.tensor(std, device=x.device)[None, :, None, None]
  return (x * std + mean).clamp(0, 1)


def to_grid_std_norm(imgs, nrow=4, mean=[0.444, 0.421, 0.384], std=[0.275, 0.267, 0.276]):
  imgs01 = denorm(imgs, mean, std)
  return vutils.make_grid(imgs01.detach().cpu(), nrow=nrow, padding=2)


def log_samples_wandb(samples, nrow=4, step=None, prefix="samples/"):
  grid = to_grid_std_norm(samples, nrow=nrow)
  payload = {f"{prefix}flow": wandb.Image(grid, caption="RF (pixel-space)")}
  wandb.log(payload, step=step)


def show_samples(samples, nrow=4, title="RF (pixel-space)"):
  # grid = to_grid(samples, nrow=nrow)
  grid = to_grid_std_norm(samples, nrow=nrow)
  plt.figure(figsize=(6, 6))
  plt.imshow(grid.permute(1, 2, 0).numpy())
  plt.title(title)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


def load_and_test_model(config):
  # Setup W&B API
  # Replace with your actual project, run, and artifact names
  api = wandb.Api()
  artifact_name = config.artifact_name
  try:
    print("Setup artifact")
    artifact = api.artifact(artifact_name, type='model')
    print("Downloading model")
    artifact_dir = artifact.download()

    model = UNetPixelSpace(in_ch=config.num_channels,
                           time_dim=config.time_dim,
                           p_dropout=None).to(device)
    model.load_state_dict(torch.load(f"{artifact_dir}/best-model.pth", map_location="cpu"))
    print("Model loaded successfully.")
  except wandb.CommError as e:
    print(f"Artifact not found: {artifact_name}")
    print(f"Error: {e}")

  print("Performing inference")
  img_shape = (config.num_channels, config.image_size, config.image_size)
  samples = sample_batch_pixels(model,
                                batch_size=4,
                                num_steps=config.num_sample_steps,
                                img_shape=img_shape,
                                device=device)
  show_samples(samples, nrow=4, title="RF pixel-space samples")


def test_model(config):
  model = UNetPixelSpace(in_ch=config.num_channels, time_dim=config.time_dim,
                         p_dropout=None).to(device)

  fn = f"artifacts/server-unet-pixel-img32-flickr30k-best-model-v18/best-model.pth"
  model.load_state_dict(torch.load(fn, map_location="cpu"))
  print("Model loaded successfully.")

  print("Performing inference")
  img_shape = (config.num_channels, config.image_size, config.image_size)
  samples = sample_batch_pixels(model,
                                batch_size=1,
                                num_steps=config.num_sample_steps,
                                img_shape=img_shape,
                                device=device)
  show_samples(samples, nrow=4, title="RF pixel-space samples")


def load_config(env="pixel_space/local_train_pixel_flickr"):
  base_config = OmegaConf.load("config/base.yaml")

  env_path = f"config/{env}.yaml"
  if os.path.exists(env_path):
    env_config = OmegaConf.load(env_path)
    # Merges env_config into base_config (env overrides base)
    config = OmegaConf.merge(base_config, env_config)
  else:
    config = base_config
  return config


def main():
  env = os.environ.get("ENV", "pixel_space/local_train_pixel_flickr")
  print(f"env={env}")
  config = load_config(env)
  print("Configuration loaded")
  config.device, config.env = device, env
  print(f"Seed {config.seed} Device {config.device}")
  if config.device == 'cuda':
    torch.set_float32_matmul_precision('high')
  if config.load_and_test_model is True:
    # load_and_test_model(config)
    test_model(config)
  if config.train_model is True:
    train_test_model(config)


if __name__ == '__main__':
  main()
