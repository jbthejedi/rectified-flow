import os, copy, random, wandb, torch
from pathlib import Path
import shutil

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
import torchvision.utils as vutils

from omegaconf import OmegaConf
from tqdm import tqdm
from pprint import pp
from rectified_flow.models.unet_pixel_space_sa import *
from langvae import LangVAE
from transformers import CLIPTextModel, CLIPTokenizer
from transformers import AutoTokenizer
from rectified_flow.encdec.flant5 import FlanT5TextEncoderDecoder


def load_model(model, version: str, config):
  api = wandb.Api()
  artifact_name = config.artifact_name
  sanitized_name = artifact_name.split("/")[-1].split(":")[0] + ":" + version
  cache_dir = Path("artifacts")
  cache_dir.mkdir(parents=True, exist_ok=True)
  model_path = cache_dir / sanitized_name / 'best-model.pth'
  try:
    if model_path.exists():
      print(f"Found cached model at {model_path}")
    else:
      print("Setup artifact")
      artifact = api.artifact(artifact_name, type='model')
      print("Downloading model")
      artifact_dir = Path(artifact.download())
      # art
      src_model = artifact_dir / "best-model.pth"
      if not src_model.exists():
        raise FileNotFoundError(f"Model file not found in artifact: {src_model}")
      # shutil.copy2(src_model, model_path)
      print(f"Caching model to {model_path}")

    model.load_state_dict(torch.load(model_path, map_location="cpu"), strict=False)
  except wandb.CommError as e:
    print(f"Artifact not found: {artifact_name}")
    print(f"Error: {e}")
  print("Model loaded successfully.")
  return model


@torch.no_grad()
def print_mags(v, v_pred):
  v_mag = v.abs().mean().item()
  vp_mag = v_pred.abs().mean().item()
  tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")


def denorm(x, mean, std):
  mean = torch.tensor(mean, device=x.device)[None, :, None, None]
  std = torch.tensor(std, device=x.device)[None, :, None, None]
  return (x * std + mean).clamp(0, 1)


def to_grid_std_norm(imgs, nrow=4, mean=[0.444, 0.421, 0.384], std=[0.275, 0.267, 0.276]):
  imgs01 = denorm(imgs, mean, std)
  return vutils.make_grid(imgs01.detach().cpu(), nrow=nrow, padding=2)


def show_samples(samples, nrow=4, title="RF (pixel-space)"):
  # grid = to_grid(samples, nrow=nrow)
  grid = to_grid_std_norm(samples, nrow=nrow)
  plt.figure(figsize=(6, 6))
  plt.imshow(grid.permute(1, 2, 0).numpy())
  plt.title(title)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


def log_samples_wandb(samples, nrow=4, step=None, prefix="samples/"):
  grid = to_grid_std_norm(samples, nrow=nrow)
  payload = {f"{prefix}flow": wandb.Image(grid, caption="RF (pixel-space)")}
  wandb.log(payload)


def load_config_01(path, env="local"):
  base_config = OmegaConf.load(f"{path}/base.yaml")

  env_path = f"{path}/{env}.yaml"
  if os.path.exists(env_path):
    env_config = OmegaConf.load(env_path)
    # Merges env_config into base_config (env overrides base)
    config = OmegaConf.merge(base_config, env_config)
  else:
    config = base_config
  return config


def print_config_vars(config):
  pp(OmegaConf.to_container(config))

