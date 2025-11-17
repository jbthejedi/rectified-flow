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
from rectified_flow.live.encdec.flant5_live import FlanT5TextEncoderDecoder
from rectified_flow.live.models.unet_pixel_space_joint_live import UNetJoint


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


def log_samples_wandb(samples, nrow=4, prefix="samples/"):
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


def compute_data(images, model, device):
  x0 = images.to(device)  # (B, C, H, W)
  x1 = torch.randn_like(x0, device=device)  # (B, C, H, W)
  B = x0.size(0)
  t = torch.rand(B, device=device)  # (B)

  # t[:, None, None, None].shape = (B, 1, 1, 1)
  xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1  # (B, C, H, W)
  v_star = x1 - x0
  v_pred = model(xt, t)
  return v_star, v_pred


def compute_data_joint(model: UNetJoint, encdec: FlanT5TextEncoderDecoder, images: torch.Tensor,
                       input_ids: torch.Tensor, attention_mask: torch.Tensor, device: torch.device):
  images = images.to(device)
  input_ids = input_ids.to(device)
  attention_mask = attention_mask.to(device)

  ################
  ##### Image ####
  ################
  x_img_0 = images                                    # (B, C, H, W)
  x_img_1 = torch.randn_like(x_img_0, device=device)  # (B, C, H, W)
  t_img = torch.rand(x_img_0.size(0), device=device)  # (B)

  # t[:, None, None, None].shape = (B, 1, 1, 1)
  x_img_t = (1 - t_img[:, None, None, None]) * x_img_0 + t_img[:, None, None, None] * x_img_1  # (B, C, H, W)
  v_star_img = x_img_1 - x_img_0

  ################
  ##### Text ####
  ################
  ## Flan-T5 Encoding
  with torch.no_grad():
    x_txt_0 = encdec.encode(input_ids, attn_mask=attention_mask).to(device)     # (B, L, TH)

  x_txt_1 = torch.randn_like(x_txt_0, device=device)                            # (B, L, TH)
  t_txt = torch.rand(x_txt_0.size(0), device=device)                        # (B)

  # t[:, None, None].shape = (B, 1, 1)
  x_txt_t = (1 - t_txt[:, None, None]) * x_txt_0 + t_txt[:, None, None] * x_txt_1  # (B, L, TH)
  v_star_txt = x_txt_1 - x_txt_0

  v_pred_img, v_pred_txt = model(x_img_t, x_txt_t, t_img, t_txt)

  return v_star_img, v_pred_img, v_star_txt, v_pred_txt


@torch.no_grad()
def sample_batch_pixels(model, batch_size=1, num_steps=100, img_shape=(3, 128, 128), device="cpu"):
  """
  Inference
  """
  tqdm.write("Performing Inference")
  B, C, H, W = batch_size, *img_shape
  x = torch.randn(B, C, H, W, device=device)  # (B, C, H, W)
  t_vals = torch.linspace(1, 0, steps=num_steps, device=device)
  # dt = 1.0 / num_steps
  dts = t_vals[1:] - t_vals[:-1]
  for i in tqdm(range(len(t_vals) - 1)):
    x = x + model(x, t_vals[i].repeat(B)) * dts[i]
  return x


@torch.no_grad()
def inference_joint(model, encdec:FlanT5TextEncoderDecoder, batch_size=1,
                    seq_len=77, num_steps=100, img_shape=(3, 128, 128), device="cpu"):
  """
  Inference
  """
  tqdm.write("Performing Inference")
  B, C, H, W = batch_size, *img_shape
  L = seq_len
  D = encdec.hidden_size
  x_img = torch.randn(B, C, H, W, device=device)  # (B, C, H, W)
  x_txt = torch.randn(B, L, D, device=device)  # (B, C, H, W)
  t_vals = torch.linspace(1, 0, steps=num_steps, device=device)
  attn_mask = torch.ones(B, L, dtype=torch.long, device=device)
  # dt = 1.0 / num_steps
  dts = t_vals[1:] - t_vals[:-1]
  for i in tqdm(range(len(t_vals) - 1)):
    t = t_vals[i].repeat(B)
    t_img = t
    t_txt = t
    v_img, v_txt = model(x_img, x_txt, t_img, t_txt, attn_mask)
    x_img = x_img + v_img * dts[i]
    x_txt = x_txt + v_txt * dts[i]
  caption = encdec.decode(
    encoder_hidden_states=x_txt,
    attention_mask=attn_mask,
    max_new_tokens=32,
    num_beams=4,
  )
  return x_img, caption 
