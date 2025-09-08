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


def load_config(env="local_train_pixel_flickr"):
  base_config = OmegaConf.load("config/pixel_space/base.yaml")

  env_path = f"config/pixel_space/{env}.yaml"
  if os.path.exists(env_path):
    env_config = OmegaConf.load(env_path)
    # Merges env_config into base_config (env overrides base)
    config = OmegaConf.merge(base_config, env_config)
  else:
    config = base_config
  return config


@torch.no_grad()
@torch.inference_mode()
def inference_pixel_space_linear(model, device, batch_size=4, num_steps=300, img_shape=(3, 128, 128)):
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
    t = t_vals[i].expand(batch_size)
    v = model(x, t)
    x = x + v * dts[i]
  return x


@torch.no_grad()
def encode_prompt_pool(langvae: LangVAE, prompt, device, max_length=77):
  """
    Returns a pooled text embedding of shape (1, TH) using your LangVAE.
    For determinism you can later swap to the encoder mean (mu) if the API exposes it.
    """
  tok = langvae.decoder.tokenizer(prompt, return_tensors="pt", truncation=True,
                                  padding="max_length", max_length=max_length)
  token_ids = tok["input_ids"].to(device)

  # If we want *deterministic* text,
  # consider modifying LangVAE.encode_z to return mu instead of a sample.
  z, _ = langvae.encode_z(token_ids, mean=True)  # (1, TH)
  return z


@torch.no_grad()
def encode_prompt_pool_clip(text_enc, prompt, device, max_length=77):
  tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
  enc = tok(prompt, return_tensors="pt", truncation=True, padding="max_length", max_length=max_length)
  token_ids = enc["input_ids"].to(device)

  lhs = text_enc(token_ids).last_hidden_state.to(device) # (1, L, TH)

  return lhs


@torch.no_grad()
def inference_pixel_space_tc_prompt_list(
    model, langvae, prompts : list, batch_size=4,
    num_steps=300, img_shape=(3, 128, 128), device=None
  ):
  """
    Integrate x' = v_theta(x, t, text) from t=1 → 0 in pixel space.
    Start from N(0, I). 'model' expects (x_img_t, x_txt_pool_t, t).
  """
  assert len(prompts) == batch_size, "Set batch_size == len(prompts) for 1:1 prompts."
  C, H, W = img_shape
  x = torch.randn(batch_size, C, H, W, device=device)

  # Get pooled text embedding once and tile to batch
  txt_pool = torch.cat([encode_prompt_pool(langvae, prompt, device) for prompt in prompts], dim=0) # (B, TH)

  # ODE schedule (linear in t)
  t_vals = torch.linspace(1.0, 0.0, steps=num_steps, device=device)
  dts = t_vals[1:] - t_vals[:-1]

  model.eval()
  for i in range(len(t_vals) - 1):
    t = t_vals[i].expand(batch_size)  # (B)
    v = model(x, txt_pool, t)  # conditioned step
    x = x + v * dts[i]
  return x


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
def inference_pixel_space_tc_prompt_list_cfg(
    model, langvae, prompts : list, num_steps=300, img_shape=(3, 128, 128), guidance_scale=4.0, device=None
  ):
  """
    Integrate x' = v_theta(x, t, text) from t=1 → 0 in pixel space.
    Start from N(0, I). 'model' expects (x_img_t, x_txt_pool_t, t).
  """
  C, H, W = img_shape
  batch_size = len(prompts)
  x = torch.randn(batch_size, C, H, W, device=device)

  # Get pooled text embedding once and tile to batch
  txt_pool = torch.cat([encode_prompt_pool(langvae, prompt, device) for prompt in prompts], dim=0) # (B, TH)

  # ODE schedule (linear in t)
  t_vals = torch.linspace(1.0, 0.0, steps=num_steps, device=device)
  dts = t_vals[1:] - t_vals[:-1]

  model.eval()
  for i in range(len(t_vals) - 1):
    t = t_vals[i].expand(batch_size)  # (B)
    v_cfg = v_with_cfg_euler(model, x, txt_pool, t, guidance_scale)
    x = x + v_cfg * dts[i]
  return x


@torch.no_grad()
def inference_joint_uncond_flan(
    model,
    text_model,          # FlanT5TextEncoderDecoder
    batch_size=4,
    num_steps=300,
    img_shape=(3, 128, 128),
    seq_len=77,
    device=None,
):
  """
    Unconditional joint sampling:
      - x_img: start at N(0, I) in pixel space
      - x_txt: start at N(0, I) in Flan-T5 encoder space
      - Both evolve from t=1 -> 0 with the SAME t schedule.

    model forward: (x_img_t, x_txt_t, t_img, t_txt, attn_mask, is_uncond)
    text_model.hidden_size must match model's txt_dim (e.g. 512).
    """
  if device is None:
    device = next(model.parameters()).device

  C, H, W = img_shape
  B = batch_size
  C_txt = text_model.hidden_size

  # Image: start at noise
  x_img = torch.randn(B, C, H, W, device=device)

  # Text: start at noise in encoder hidden space
  # We don't have real tokens here (unconditional), so we just choose a fixed seq_len
  x_txt = torch.randn(B, seq_len, C_txt, device=device)

  # All positions "valid" for attention
  attn_mask = torch.ones(B, seq_len, dtype=torch.long, device=device)

  # Shared time schedule for both modalities
  t_vals = torch.linspace(1.0, 0.0, steps=num_steps, device=device)
  dts = t_vals[1:] - t_vals[:-1]

  model.eval()
  for i in tqdm(range(num_steps - 1)):
    t = t_vals[i].expand(B)  # [B]

    # Same t for image and text in this version
    t_img = t
    t_txt = t

    v_img, v_txt = model(
        x_img, x_txt,
        t_img=t_img,
        t_txt=t_txt,
        attn_mask=attn_mask,
        is_uncond=None,           # fully "cond" on current text state
    )

    dt = dts[i]
    x_img = x_img + v_img * dt
    x_txt = x_txt + v_txt * dt

  # x_img is now approx x_img_0 ~ data
  # x_txt is now approx x_txt_0 = clean Flan-T5 encoder states
  # We can decode captions from x_txt
  captions = text_model.decode(
      encoder_hidden_states=x_txt,
      attention_mask=attn_mask,
      max_new_tokens=32,
      num_beams=4,
  )
  return x_img, captions


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


@torch.no_grad()
def inference_pixel_space_clip(
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


@torch.no_grad()
def inference_pixel_space_tc(
    model, langvae, prompt="a person", batch_size=4,
    num_steps=300, img_shape=(3, 128, 128), device=None
  ):
  """
    Integrate x' = v_theta(x, t, text) from t=1 → 0 in pixel space.
    Start from N(0, I). 'model' expects (x_img_t, x_txt_pool_t, t).
    """
  C, H, W = img_shape
  x = torch.randn(batch_size, C, H, W, device=device)

  # Get pooled text embedding once and tile to batch
  txt_pool_1 = encode_prompt_pool(langvae, prompt, device)  # (1, TH)
  txt_pool = txt_pool_1.expand(batch_size, -1).contiguous()  # (B, TH)

  # ODE schedule (linear in t)
  t_vals = torch.linspace(1, 0, steps=num_steps, device=device)
  dts = t_vals[1:] - t_vals[:-1]

  model.eval()
  for i in range(len(t_vals) - 1):
    t = t_vals[i].repeat(batch_size)  # (B)
    v = model(x, txt_pool, t)  # conditioned step
    x = x + v * dts[i]
  return x


@torch.no_grad()
def sample_img_from_text_joint(model, text_enc, prompts, steps=50, H=32,
                               W=32, device="cuda", guidance=4.0):
  tok = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
  enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=77).to(device)
  txt_clean = text_enc(**enc).last_hidden_state  # [B,L,C]
  attn = enc["attention_mask"]  # [B,L]
  B, C = len(prompts), txt_clean.size(-1)

  # Cond/Uncond copies for CFG
  txt_uncond = txt_clean.clone()
  null = model.null_token.expand(B, txt_clean.size(1), C)
  txt_uncond[:] = null

  x = torch.randn(B, 3, H, W, device=device)
  t_vals = torch.linspace(1.0, 0.0, steps=steps, device=device)
  for i in range(steps - 1):
    t = t_vals[i].expand(B)
    # cond
    v_img_c, _ = model(x, txt_clean, t, torch.zeros_like(t), attn, is_uncond=None)
    # uncond
    v_img_u, _ = model(x, txt_uncond, t, torch.zeros_like(t),
                       attn, is_uncond=torch.ones(B, dtype=torch.bool, device=device))
    v = v_img_u + guidance * (v_img_c - v_img_u)
    x = x + (t_vals[i + 1] - t_vals[i]) * v
  return x


@torch.no_grad()
def print_mags(v, v_pred):
  v_mag = v.abs().mean().item()
  vp_mag = v_pred.abs().mean().item()
  tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")


def save_and_log_model(model, config, filename="best-model.pth"):
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


def save_and_log_model_best_val(model, config, best_val_loss, val_loss, filename="best-model.pth"):
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


def load_and_test_model(config, device):
  # Setup W&B API
  # Replace with your actual project, run, and artifact names
  api = wandb.Api()
  artifact_name = config.artifact_name
  sanitized_name = artifact_name.replace("/", "_").replace(":", "_")
  cache_dir = Path("artifacts")
  cache_dir.mkdir(parents=True, exist_ok=True)
  model_path = cache_dir / f"{sanitized_name}.pth"
  try:
    if model_path.exists():
      print(f"Found cached model at {model_path}")
    else:
      print("Setup artifact")
      artifact = api.artifact(artifact_name, type='model')
      print("Downloading model")
      artifact_dir = Path(artifact.download())
      src_model = artifact_dir / "best-model.pth"
      if not src_model.exists():
        raise FileNotFoundError(f"Model file not found in artifact: {src_model}")
      shutil.copy2(src_model, model_path)
      print(f"Caching model to {model_path}")

    model = UNetPixelSpacePlus(in_ch=config.num_channels,
                               time_dim=config.time_dim,
                               p_dropout=None).to(device)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    print("Model loaded successfully.")
  except wandb.CommError as e:
    print(f"Artifact not found: {artifact_name}")
    print(f"Error: {e}")
  except FileNotFoundError as e:
    print(e)

  print("Performing inference")
  img_shape = (config.num_channels, config.image_size, config.image_size)
  samples = inference_pixel_space_linear(model, batch_size=1, num_steps=config.num_sample_steps,
                                         img_shape=img_shape, device=device)
  show_samples(samples, nrow=4, title="RF pixel-space samples")


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


def load_model_from_local(model, folder_name: str):
  ckpt_path = Path("artifacts") / folder_name / "best-model.pth"

  if not ckpt_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

  print(f"Loading checkpoint from: {ckpt_path}")
  state = torch.load(ckpt_path, map_location="cpu")
  model.load_state_dict(state, strict=False)
  print("Model loaded successfully.")
  return model


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
  wandb.log(payload)


def log_samples_wandb_captions(samples, nrow=4, step=None, prefix="samples/", captions=None):
  grid = to_grid_std_norm(samples, nrow=nrow)

  if captions is not None:
    # Put each caption on its own line, index them so you can match panel -> text
    cap_text = "\n".join([f"{i}: {c}" for i, c in enumerate(captions)])
  else:
    cap_text = "RF (pixel-space)"

  payload = {
      f"{prefix}flow": wandb.Image(grid, caption=cap_text)
  }
  wandb.log(payload)


def show_samples(samples, nrow=4, title="RF (pixel-space)"):
  # grid = to_grid(samples, nrow=nrow)
  grid = to_grid_std_norm(samples, nrow=nrow)
  plt.figure(figsize=(6, 6))
  plt.imshow(grid.permute(1, 2, 0).numpy())
  plt.title(title)
  plt.axis("off")
  plt.tight_layout()
  plt.show()


def test_model(config, device):
  model = UNetPixelSpacePlus(in_ch=config.num_channels, time_dim=config.time_dim,
                         p_dropout=None).to(device)

  fn = f"artifacts/server-unet-pixel-img32-flickr30k-best-model-v18/best-model.pth"
  model.load_state_dict(torch.load(fn, map_location="cpu"))
  print("Model loaded successfully.")

  print("Performing inference")
  img_shape = (config.num_channels, config.image_size, config.image_size)
  samples = inference_pixel_space_linear(model, batch_size=1, num_steps=config.num_sample_steps,
                                         img_shape=img_shape, device=device)
  show_samples(samples, nrow=4, title="RF pixel-space samples")


def compute_data(model, images, device):
  images = images.to(device, non_blocking=True)
  x0 = images
  x1 = torch.randn_like(x0)  # noise
  t = torch.rand(x0.size(0), device=device) # (B)
  xt = (1 - t[:, None, None, None]) * x0 + t[:, None, None, None] * x1
  v = x1 - x0
  v_pred = model(xt, t)
  return v, v_pred


def compute_data_tc_langvae(model, langvae: LangVAE, images: torch.Tensor, token_ids: torch.Tensor, device):
  images = images.to(device, non_blocking=True)

  # Encode text
  with torch.no_grad():
    # TODO z is a sample from the VAE's posterior.
    # Check to see if that's okay for training.
    # Might have to use mean instead
    pooled_txt, _ = langvae.encode_z(token_ids, mean=True)                                                 # (B, TH)
    pooled_txt.to(device)
  # x_txt_1 = torch.randn_like(z)

  x_img_0 = images                                                                     # (B, C, H, W)
  x_img_1 = torch.randn_like(x_img_0)                                                  # (B, C, H, W)

  t = torch.rand(x_img_0.size(0), device=device)                                       # (B, 1)

  x_img_t = (1 - t[:, None, None, None]) * x_img_0 + t[:, None, None, None] * x_img_1  # (B, C, H, W)

  v_img_star = x_img_1 - x_img_0                                                       # (B, C, HW)
  # v_txt_star = x_txt_1 - x_txt_0                                                     # (B, TH)
  v_pred = model(x_img_t, pooled_txt, t)
  # return v_img, v_txt_star, v_pred
  return v_img_star, v_pred


def compute_data_tc_langvae_cfg(model, langvae: LangVAE,
                                images: torch.Tensor, token_ids: torch.Tensor,
                                device, p_uncond=0.15):
  images = images.to(device, non_blocking=True)

  # Encode text
  with torch.no_grad():
    # TODO z is a sample from the VAE's posterior.
    # Check to see if that's okay for training.
    # Might have to use mean instead
    pooled_txt, _ = langvae.encode_z(token_ids, mean=True)                                      # (B, TH)
    pooled_txt.to(device)

  x_img_0 = images                                                                     # (B, C, H, W)
  B = x_img_0.size(0)
  x_img_1 = torch.randn_like(x_img_0)                                                  # (B, C, H, W)
  t = torch.rand(B, device=device)                                                     # (B, 1)

  x_img_t = (1 - t[:, None, None, None]) * x_img_0 + t[:, None, None, None] * x_img_1  # (B, C, H, W)

  v_img_star = x_img_1 - x_img_0                                                       # (B, C, HW)
  is_uncond = (torch.rand(B, device=device) < p_uncond)
  v_pred = model(x_img_t, pooled_txt, t, is_uncond)
  return v_img_star, v_pred


def compute_data_joint_flant5(model, encdec: FlanT5TextEncoderDecoder, images: torch.Tensor,
                       token_ids: torch.Tensor, attn_mask: torch.Tensor,
                       device, p_uncond=0.15):
  images = images.to(device, non_blocking=True)
  attn_mask = attn_mask.to(device, non_blocking=True)
  token_ids = token_ids.to(device, non_blocking=True)

  ### Image ###
  x_img_0 = images                                                                     # (B, C, H, W)
  B       = x_img_0.size(0)
  x_img_1 = torch.randn_like(x_img_0)                                                  # (B, C, H, W)
  t_img   = torch.rand(B, device=device)                                                     # (B, 1)

  x_img_t = (1 - t_img[:, None, None, None]) * x_img_0 + t_img[:, None, None, None] * x_img_1  # (B, C, H, W)
  v_img_star = x_img_1 - x_img_0                                                       # (B, C, H, W)

  ### Text ###
  with torch.no_grad(): # Encode text
    text_tokens = encdec.encode(token_ids, attn_mask).to(device)

  x_txt_0 = F.layer_norm(text_tokens, (text_tokens.size(-1),))
  B       = x_txt_0.size(0)
  x_txt_1 = torch.randn_like(x_txt_0)
  t_txt   = torch.rand(B, device=device)

  x_txt_t = (1 - t_txt[:, None, None]) * x_txt_0 + t_txt[:, None, None] * x_txt_1  # (B, L, TH)
  v_txt_star = x_txt_1 - x_txt_0                                                       # (B, L, TH)

  is_uncond = (torch.rand(B, device=device) < p_uncond)
  v_img_pred, v_txt_pred  = model(x_img_t, x_txt_t, t_img, t_txt, attn_mask, is_uncond)

  eps = 1e-4
  w_img = (t_img / (1 - t_img + eps)).clamp(max=5.0).view(-1,1,1,1)
  w_txt = (t_txt / (1 - t_txt + eps)).clamp(max=5.0).view(-1,1,1)

  return {
    "w_img": w_img,
    "w_txt": w_txt,
    "v_img_star": v_img_star,
    "v_txt_star": v_txt_star,
    "v_img_pred": v_img_pred,
    "v_txt_pred": v_txt_pred,
  }


@torch.no_grad()
def encode_prompt_clip_lhs(text_enc, tok, device):
  text = tok(["red car", "black dog"], padding=True, return_tensors="pt").to(device)
  txt_tokens = text_enc(**text).last_hidden_state  # (B, L, 768)
  txt_pooled = txt_tokens.mean(dim=1)
  return txt_pooled


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


def compute_data_joint_clip(model, text_enc, images: torch.Tensor,
                       token_ids: torch.Tensor, attn_mask: torch.Tensor,
                       device, p_uncond=0.15):
  images = images.to(device, non_blocking=True)
  attn_mask = attn_mask.to(device, non_blocking=True)
  token_ids = token_ids.to(device, non_blocking=True)

  # Encode text
  with torch.no_grad():
    text_tokens = text_enc(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state                   # (B, L, TH)
    text_tokens.to(device)

  ### Image ###
  x_img_0 = images                                                                     # (B, C, H, W)
  B = x_img_0.size(0)
  x_img_1 = torch.randn_like(x_img_0)                                                  # (B, C, H, W)
  t_img = torch.rand(B, device=device)                                                     # (B, 1)

  ### Text ###
  x_txt_0 = F.layer_norm(text_tokens, (text_tokens.size(-1),))
  x_txt_0 = text_tokens
  B = x_txt_0.size(0)
  x_txt_1 = torch.randn_like(x_txt_0)
  t_txt = torch.rand(B, device=device)

  x_img_t = (1 - t_img[:, None, None, None]) * x_img_0 + t_img[:, None, None, None] * x_img_1  # (B, C, H, W)
  x_txt_t = (1 - t_txt[:, None, None]) * x_txt_0 + t_txt[:, None, None] * x_txt_1  # (B, L, TH)

  v_img_star = x_img_1 - x_img_0                                                       # (B, C, H, W)
  v_txt_star = x_txt_1 - x_txt_0                                                       # (B, L, TH)

  is_uncond = (torch.rand(B, device=device) < p_uncond)
  v_img_pred, v_txt_pred  = model(x_img_t, x_txt_t, t_img, t_txt, attn_mask, is_uncond)

  eps = 1e-4
  w_img = (t_img / (1 - t_img + eps)).clamp(max=5.0).view(-1,1,1,1)
  w_txt = (t_txt / (1 - t_txt + eps)).clamp(max=5.0).view(-1,1,1)

  return {
    "w_img": w_img,
    "w_txt": w_txt,
    "v_img_star": v_img_star,
    "v_txt_star": v_txt_star,
    "v_img_pred": v_img_pred,
    "v_txt_pred": v_txt_pred,
  }


def get_langvae(device):
  langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128")
  langvae = langvae.to(device)
  langvae.eval()
  for p in langvae.parameters():
    p.requires_grad = False

  # Necessary otherwise won't be on GPU.
  langvae.encoder.to(device)
  langvae.decoder.to(device)

  tok_path = langvae.decoder.tokenizer.name_or_path
  if not os.path.isdir(tok_path):
    tok_path = "./langvae_tokenizer_ckpt"
    os.makedirs(tok_path, exist_ok=True)
    langvae.decoder.tokenizer.save_pretrained(tok_path)
  return langvae


def print_config_vars(config):
  pp(OmegaConf.to_container(config))


@torch.no_grad()
def caption_usage_metrics(
    model, text_enc, images, token_ids, attention_mask, device, n_t=8, t_min=0.05, t_max=0.95
):
  """
  Returns a list of dicts with per-t metrics:
    - delta_cu_mean: ||v_c - v_u|| / ||v*||
    - delta_cw_mean: ||v_c - v_w|| / ||v*||
    - loss_improve_mean: (MSE_uncond - MSE_cond)
  """
  model.eval()
  B = images.size(0)
  images = images.to(device, non_blocking=True)
  token_ids = token_ids.to(device, non_blocking=True)
  attention_mask = attention_mask.to(device, non_blocking=True)

  # Encode CLIP tokens (L, 512)
  txt = text_enc(input_ids=token_ids, attention_mask=attention_mask).last_hidden_state  # (B,L,C)

  # "Wrong" captions by shuffling other samples' tokens/masks
  perm = torch.randperm(B, device=device)
  txt_wrong = txt[perm]
  attn_wrong = attention_mask[perm]

  t_points = torch.linspace(t_min, t_max, steps=n_t, device=device)
  out = []

  def rms(e):  # root-mean-square over spatial dims
    return e.pow(2).mean(dim=(1,2,3)).sqrt()

  for t in t_points:
    t_vec = t.expand(B)

    # draw a fresh x1 per t so v* doesn't collapse
    x0 = images
    x1 = torch.randn_like(x0)
    xt = (1 - t_vec[:, None, None, None]) * x0 + t_vec[:, None, None, None] * x1
    v_star = x1 - x0  # ground-truth velocity

    # conditional
    v_c = model(xt, txt, t_vec, attn_mask=attention_mask, is_uncond=None)

    # unconditional (force null-token path for all items)
    v_u = model(xt, txt, t_vec, attn_mask=attention_mask,
                is_uncond=torch.ones(B, dtype=torch.bool, device=device))

    # uncond via zeroed text (sanity)
    zeros = torch.zeros_like(txt)
    zeros_mask = torch.zeros_like(attention_mask)
    zeros_mask[:, 0] = 1
    v_u0 = model(xt, zeros, t_vec, attn_mask=zeros_mask, is_uncond=None)

    # wrong caption (tokens from another sample)
    v_w = model(xt, txt_wrong, t_vec, attn_mask=attn_wrong, is_uncond=None)

    # normalized diffs: divide by ||v*|| to remove scale drift across t
    denom = rms(v_star) + 1e-8
    delta_cu = rms(v_c - v_u) / denom
    delta_cw = rms(v_c - v_w) / denom
    dcu0  = rms(v_c - v_u0) / denom

    # loss improvement from text
    L_c = F.mse_loss(v_c, v_star, reduction='none').mean(dim=(1, 2, 3))
    L_u = F.mse_loss(v_u, v_star, reduction='none').mean(dim=(1, 2, 3))

    out.append(
        dict(
            t=float(t),
            delta_cu_mean=float(delta_cu.mean().item()),
            delta_cw_mean=float(delta_cw.mean().item()),
            delta_cu0_mean=float(dcu0.mean().item()),
            loss_improve_mean=float((L_u - L_c).mean().item()),
        ))
  return out


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
def text_effect_eval(model, text_enc, val_dl, device, n_batches=50):
  model.eval()
  gains_c_vs_u = []
  gains_c_vs_w = []

  for i, (images, token_ids, attn_mask) in enumerate(val_dl):
    if i >= n_batches:
      break

    images = images.to(device)
    token_ids = token_ids.to(device)
    attn_mask = attn_mask.to(device)

    # Encode CLIP tokens
    txt = text_enc(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state

    # Wrong captions by shuffling
    perm = torch.randperm(images.size(0), device=device)
    txt_wrong = txt[perm]
    attn_wrong = attn_mask[perm]

    B = images.size(0)
    x0 = images
    x1 = torch.randn_like(x0)
    t  = torch.rand(B, device=device)

    xt = (1 - t[:,None,None,None]) * x0 + t[:,None,None,None] * x1
    v_star = x1 - x0

    test_use_text(model, token_ids, attn_mask, text_enc, xt, t, device)

    def mse(v):
      return ((v - v_star)**2).mean(dim=(1,2,3))

    # conditional
    v_c = model(xt, txt, t, attn_mask, is_uncond=None)
    L_c = mse(v_c)

    # unconditional via null token
    v_u = model(xt, txt, t, attn_mask,
                is_uncond=torch.ones(B, dtype=torch.bool, device=device))
    L_u = mse(v_u)

    # wrong caption
    v_w = model(xt, txt_wrong, t, attn_wrong, is_uncond=None)
    L_w = mse(v_w)
    # Sanity: how different are the *predictions*?

    delta = ((v_c - v_u) ** 2).mean().item()
    print("mean squared difference between v_c and v_u:", delta)

    # Optional: max absolute diff, just to be sure
    max_diff = (v_c - v_u).abs().max().item()
    print("max |v_c - v_u|:", max_diff)

    gains_c_vs_u.append((L_u - L_c).mean().item())
    gains_c_vs_w.append((L_w - L_c).mean().item())

  return float(sum(gains_c_vs_u)/len(gains_c_vs_u)), float(sum(gains_c_vs_w)/len(gains_c_vs_w))


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


def test_use_text(model, token_ids, attn_mask, text_enc, xt, t, device):
  with torch.no_grad():
    # same xt and t
    xt = xt.to(device)
    t  = t.to(device)

    # real captions
    txt_real = text_enc(input_ids=token_ids, attention_mask=attn_mask).last_hidden_state.to(device)

    # shuffled / wrong captions
    perm = torch.randperm(txt_real.size(0), device=device)
    txt_wrong = txt_real[perm]
    attn_wrong = attn_mask[perm]

    v_real  = model(xt, txt_real,  t, attn_mask, is_uncond=None)
    v_wrong = model(xt, txt_wrong, t, attn_wrong, is_uncond=None)

    delta_txt = ((v_real - v_wrong)**2).mean().item()
    max_txt   = (v_real - v_wrong).abs().max().item()
    print("delta_txt:", delta_txt)
    print("max_txt:", max_txt)
