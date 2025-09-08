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


def compute_data_tc_clip_cfg(model, text_enc,
                             images: torch.Tensor, token_ids: torch.Tensor, attn_mask: torch.Tensor,
                             device, p_uncond=0.15):
  images = images.to(device, non_blocking=True)
  attn_mask = attn_mask.to(device, non_blocking=True)
  token_ids = token_ids.to(device, non_blocking=True)
  # Encode text
  with torch.no_grad():
    text_tokens = text_enc(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state                   # (B, L, TH)
    text_tokens.to(device)

  x_img_0 = images                                                                     # (B, C, H, W)
  B = x_img_0.size(0)
  x_img_1 = torch.randn_like(x_img_0)                                                  # (B, C, H, W)
  t = torch.rand(B, device=device)                                                     # (B, 1)

  x_img_t = (1 - t[:, None, None, None]) * x_img_0 + t[:, None, None, None] * x_img_1  # (B, C, H, W)

  v_img_star = x_img_1 - x_img_0                                                       # (B, C, HW)
  is_uncond = (torch.rand(B, device=device) < p_uncond)
  v_pred = model(x_img_t, text_tokens, t, attn_mask, is_uncond)
  return v_img_star, v_pred


@torch.no_grad()
def print_mags(v, v_pred):
  v_mag = v.abs().mean().item()
  vp_mag = v_pred.abs().mean().item()
  tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")


def compute_data_tc_clip_cfg(model, text_enc,
                             images: torch.Tensor, token_ids: torch.Tensor, attn_mask: torch.Tensor,
                             device, p_uncond=0.15):
  images = images.to(device, non_blocking=True)
  attn_mask = attn_mask.to(device, non_blocking=True)
  token_ids = token_ids.to(device, non_blocking=True)
  # Encode text
  with torch.no_grad():
    text_tokens = text_enc(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state                   # (B, L, TH)
    text_tokens.to(device)

  x_img_0 = images                                                                     # (B, C, H, W)
  B = x_img_0.size(0)
  x_img_1 = torch.randn_like(x_img_0)                                                  # (B, C, H, W)
  t = torch.rand(B, device=device)                                                     # (B, 1)

  x_img_t = (1 - t[:, None, None, None]) * x_img_0 + t[:, None, None, None] * x_img_1  # (B, C, H, W)

  v_img_star = x_img_1 - x_img_0                                                       # (B, C, HW)
  is_uncond = (torch.rand(B, device=device) < p_uncond)
  v_pred = model(x_img_t, text_tokens, t, attn_mask, is_uncond)
  return v_img_star, v_pred


@torch.no_grad()
def text_triplet_metrics(model, text_enc, images, token_ids, attn_mask, device):
  model.eval()
  B = images.size(0)
  images      = images.to(device, non_blocking=True)
  token_ids   = token_ids.to(device, non_blocking=True)
  attn_mask   = attn_mask.to(device, non_blocking=True)

  # CLIP tokens
  txt = text_enc(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state  # (B,L,C)

  # wrong captions (shuffled)
  perm = torch.randperm(B, device=device)
  txt_wrong  = txt[perm]
  mask_wrong = attn_mask[perm]

  # single t in mid-range
  t = torch.empty(B, device=device).uniform_(0.1, 0.9)

  x0 = images
  x1 = torch.randn_like(x0)
  xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
  v_star = x1 - x0

  def mse_per_sample(v_hat):
    return ((v_hat - v_star) ** 2).mean(dim=(1,2,3))  # (B,)

  # conditional
  v_c = model(xt, txt, t, attn_mask=attn_mask, is_uncond=None)
  L_c = mse_per_sample(v_c)

  # unconditional via is_uncond=True (null-token path)
  v_u = model(xt, txt, t, attn_mask=attn_mask,
              is_uncond=torch.ones(B, dtype=torch.bool, device=device))
  L_u = mse_per_sample(v_u)

  # wrong-caption
  v_w = model(xt, txt_wrong, t, attn_mask=mask_wrong, is_uncond=None)
  L_w = mse_per_sample(v_w)

  return {
      "mse_c":  L_c.mean().item(),
      "mse_u":  L_u.mean().item(),
      "mse_w":  L_w.mean().item(),
      "delta_u": (L_u - L_c).mean().item(),
      "delta_w": (L_w - L_c).mean().item(),
  }


@torch.no_grad()
def null_proximity_stats(model, txt_toks, attn_mask):
  """
    txt_toks: (B, L, C) on same device as model
    attn_mask: (B, L) on same device
    Returns: dict(null_cos, null_l2, null_ratio)
    """
  B, L, C = txt_toks.shape
  # EOT pooling (last non-pad index from mask)
  lengths = attn_mask.sum(dim=1).clamp_min(1) - 1  # (B,)
  eot = txt_toks[torch.arange(B, device=txt_toks.device), lengths]  # (B, C)

  # Normalize BOTH sides for cosine/L2 to live in same space
  eot_n = F.normalize(eot, dim=-1)
  null = model.null_token[0, 0].expand_as(eot)  # (B, C)
  null_n = F.normalize(null, dim=-1)

  # Cosine similarity (1=identical)
  cos_mean = (eot_n * null_n).sum(dim=-1).mean().item()

  # L2 (in normalized space)
  l2_mean = (eot_n - null_n).norm(dim=-1).mean().item()

  # Robust scale-free denominator
  if B < 2:
    ratio = float('nan')  # can't measure with a single example
  else:
    # Use all pairwise distances excluding diagonal (stable even if B small)
    D = torch.cdist(eot_n, eot_n, p=2)  # (B,B)
    denom = D[~torch.eye(B, dtype=torch.bool, device=D.device)].mean()
    ratio = ((eot_n - null_n).norm(dim=-1).mean() / denom.clamp_min(1e-6)).item()

  return dict(null_cos=cos_mean, null_l2=l2_mean, null_ratio=ratio)


@torch.no_grad()
def encode_prompt_pool_clip(text_enc, prompt, device, max_length=77):
  tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
  enc = tok(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
  token_ids = enc["input_ids"].to(device)

  lhs = text_enc(token_ids).last_hidden_state.to(device) # (1, L, TH)

  return lhs


@torch.no_grad()
def v_with_cfg_euler(model, x, txt_pool, t, guidance_scale: float):
  """
    Returns v_cfg = v_u + s*(v_c - v_u)
    Runs a single forward on a packed batch: [uncond; cond].
    """
  B = x.size(0)
  # pack batch: 2B
  x_cat = torch.cat([x, x], dim=0)
  t_cat = torch.cat([t, t], dim=0)                   # (2B, 1)
  txt_cat = torch.cat([txt_pool, txt_pool], dim=0)     # (2B, TH)
  is_u = torch.cat([
    torch.ones(B, dtype=torch.bool, device=x.device),
    torch.zeros(B, dtype=torch.bool, device=x.device)], dim=0) # (2B,)

  v_cat = model(x_cat, txt_cat, t_cat, is_uncond=is_u) # (2B, C, H, W)
  v_u, v_c = v_cat.chunk(2, dim=0)
  return v_u + guidance_scale * (v_c - v_u)            # (B, C, H, W)


@torch.no_grad()
def inference_pixel_space_tc_prompt_list_cfg_clip(
    model, text_enc, prompts : list, batch_size=4,
    num_steps=300, img_shape=(3, 128, 128), guidance_scale=4.0, device=None
  ):
  """
    Integrate x' = v_theta(x, t, text) from t=1 → 0 in pixel space.
    Start from N(0, I). 'model' expects (x_img_t, x_txt_pool_t, t).
  """
  assert len(prompts) == batch_size, "Set batch_size == len(prompts) for 1:1 prompts."
  C, H, W = img_shape
  x = torch.randn(batch_size, C, H, W, device=device)

  # Get pooled text embedding once and tile to batch
  txt_pool = torch.cat([encode_prompt_pool_clip(text_enc, prompt, device) for prompt in prompts], dim=0) # (B, TH)

  # ODE schedule (linear in t)
  t_vals = torch.linspace(1.0, 0.0, steps=num_steps, device=device)
  dts = t_vals[1:] - t_vals[:-1]

  model.eval()
  for i in range(len(t_vals) - 1):
    t = t_vals[i].expand(batch_size)  # (B)
    v_cfg = v_with_cfg_euler(model, x, txt_pool, t, guidance_scale)
    x = x + v_cfg * dts[i]
  return x


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

