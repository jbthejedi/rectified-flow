import os, copy, random, wandb, torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as vutils

from torchvision import datasets, transforms as T
from omegaconf import OmegaConf
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from tqdm import tqdm
from rectified_flow.models.unet_pixel_text_cond_film import *
from rectified_flow.data.flickr30k_tokenized import Flickr30kTokenized
from rectified_flow.data.datamodule_recover import ProjectData
from langvae import LangVAE

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

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

  ##### TEXT ENCODER #####
  print("Load LangVAE")
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

  ##### GET DATA ######
  train_dl, val_dl = get_data(tok_path, config)
  # train_dl, val_dl, max_steps = get_data(tok_path, config)
  print(f"train len {len(train_dl)}")
  print(f"val len {len(val_dl)}")

  ####### Model/Optimizer ######
  img_shape = (config.num_channels, config.image_size, config.image_size)
  model = UNetPixelSpace(in_channels=config.num_channels, time_dim=config.time_dim,
                         p_dropout=None).to(device)
  if config.load_model is True:
    model = load_model(model, config)
  freeze_backbone_keep_adapters(model)
  optimizer = make_optimizer(model, base_lr=config.adapter_lr)

  warmup_epochs  = config.adapter_warmup_epochs       # e.g., 3–5
  stage2_epoch   = warmup_epochs + config.stage2_len # unfreeze mid+dec
  stage3_epoch   = stage2_epoch + config.stage3_len  # unfreeze enc
  unfroze_middec = False
  unfroze_enc    = False
  unfroze_all    = False

  log_dict = {}
  best_val_loss = float("inf")
  tqdm.write("Training adapter-only intially")
  for epoch in range(1, config.n_epochs):
    #### UNFREEZE LOGIC ####
    if (not unfroze_middec) and epoch == warmup_epochs:
      unfreeze_patterns(model, optimizer, patterns=["mid", "dec"], lr_backbone=config.backbone_lr)
      unfroze_middec = True
      tqdm.write("Unfroze middec")
    if (not unfroze_enc) and epoch == stage2_epoch:
      unfreeze_patterns(model, optimizer, patterns=["enc"], lr_backbone=config.backbone_lr*0.5)
      unfroze_enc = True
      tqdm.write("Unfroze enc")
    if (not unfroze_all) and epoch == stage3_epoch:
      unfreeze_all(model, optimizer, config.backbone_lr*0.25)
      unfroze_all = True
      tqdm.write("Unfroze all")

    log_dict["epoch"] = epoch
    model.train()
    scaler = torch.amp.GradScaler()
    with tqdm(train_dl, desc="Training") as pbar:
      train_loss = 0.0
      model.train()
      for images, token_ids, _ in pbar:
        v_star, v_pred = compute_data(model, langvae, images, token_ids,
                                      device, amp=True)
        with torch.no_grad():
          v_mag = v_star.abs().mean().item()
          vp_mag = v_pred.abs().mean().item()
        tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")
        loss = ((v_pred - v_star)**2).mean()

        optimizer.zero_grad()
        scaler.scale(loss).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        # optimizer.step()
        scaler.update()
        train_loss += loss.item()

      train_loss /= len(train_dl)
    tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    model.eval()
    with torch.inference_mode(), tqdm(val_dl, desc="Validation") as pbar:
      val_loss = 0.0
      for images, token_ids, _ in pbar:
        v_star, v_pred = compute_data(model, langvae, images, token_ids,
                                      device, amp=True)
        v_mag = v_star.abs().mean().item()
        vp_mag = v_pred.abs().mean().item()
        tqdm.write(f"|v*|={v_mag:.3f} |v̂|={vp_mag:.3f}")

        loss = ((v_pred - v_star)**2).mean()
        val_loss += loss.item()
      val_loss /= len(val_dl)
    tqdm.write(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

    if (epoch % config.inference_peek_num) == 0:
      samples = sample_batch_pixels_cond(
          model, langvae, prompt="a person",
          batch_size=4, num_steps=config.num_sample_steps,
          img_shape=img_shape, device=device,
      )
      if config.local_visualization is True:
        show_samples(samples, nrow=4, title="RF pixel-space samples")
      if config.write_inference_samples is True:
        log_samples_wandb(samples, nrow=4, step=epoch)
        tqdm.write("Wrote image grid")

    log_dict["train/loss"] = train_loss
    log_dict["val/loss"] = val_loss
    wandb.log(log_dict, step=epoch, commit=True)

    if config.save_model:
      tqdm.write("Saving model")
      best_val_loss = save_and_log_model(model, config, best_val_loss, val_loss)

  tqdm.write("Done Training")


def get_data(tok_path, config):
  mean = [0.444, 0.421, 0.384]
  std = [0.275, 0.267, 0.276]
  train_tf = T.Compose(
      [T.CenterCrop(224),
       T.Resize(config.image_size),
       T.ToTensor(),
       T.Normalize(mean, std)])

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
  return train_dl, val_dl


def freeze_backbone_keep_adapters(model):
  for n, p in model.named_parameters():
    # train only adapters
    if ("film" in n) or ("cross_attn" in n):
      p.requires_grad = True
    else:
      p.requires_grad = False


def make_optimizer(model, base_lr=1e-4, wd=0.0):
  params = [p for p in model.parameters() if p.requires_grad]
  return optim.Adam(params, lr=base_lr, weight_decay=wd)


def unfreeze_patterns(model, optimizer, patterns, lr_backbone):
  new_params = []
  for n, p in model.named_parameters():
    if (not p.requires_grad) and any(pat in n for pat in patterns):
      p.requires_grad = True
      new_params.append(p)
  if new_params:
    optimizer.add_param_group({"params": new_params, "lr": lr_backbone})
    return True
  return False


def unfreeze_all(model, optimizer, lr):
  # Unfreeze everything
  for p in model.parameters():
    p.requires_grad = True

  # Collect ids of params already in optimizer
  existing = {id(p) for g in optimizer.param_groups for p in g['params']}

  # Add only params not yet tracked by optimizer (use identity via id)
  leftovers = [p for p in model.parameters() if p.requires_grad and id(p) not in existing]

  if leftovers:
    optimizer.add_param_group({"params": leftovers, "lr": lr})
  return True


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


def compute_data(model, langvae: LangVAE, images, token_ids, device, amp=False):
  images = images.to(device, non_blocking=True)

  amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

  # Encode text
  with torch.no_grad():
    with torch.autocast(device_type=device, dtype=(torch.bfloat16 if device=="cpu" else amp_dtype)):
      # TODO z is a sample from the VAE's posterior.
      # Check to see if that's okay for training.
      # Might have to use mean instead
      z, _ = langvae.encode_z(token_ids)                                            # (B, TH)
  # x_txt_1 = torch.randn_like(z)

  x_img_0 = images                                                                  # (B, C, H, W)
  x_txt_0 = z                                                                       # (B, TH)
  x_img_1 = torch.randn_like(x_img_0)                                               # (B, C, H, W)
  x_txt_1 = torch.randn_like(x_txt_0)                                               # (B, TH)

  t = torch.rand(x_img_0.size(0), 1, device=device)                                 # (B, 1)

  x_img_t = (1 - t[:, :, None, None]) * x_img_0 + t[:, :, None, None] * x_img_1     # (B, C, H, W)
  x_txt_t = (1 - t) * x_txt_0 + t * x_txt_1                                         # (B, TH)

  v_img_star = x_img_1 - x_img_0                                                    # (B, C, HW)
  # v_txt_star = x_txt_1 - x_txt_0                                                  # (B, TH)
  with torch.autocast(device_type=device, dtype=(torch.bfloat16 if device=="cpu" else amp_dtype)):
    v_pred = model(x_img_t, x_txt_t, t)
  # return v_img, v_txt_star, v_pred
  return v_img_star, v_pred


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


# replace to_01 with:
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


def load_model(model, config):
  api = wandb.Api()
  artifact_name = config.artifact_name
  try:
    print("Setup artifact")
    artifact = api.artifact(artifact_name, type='model')
    print("Downloading model")
    artifact_dir = artifact.download(root="artifacts")

    model = UNetPixelSpace(in_channels=config.num_channels,
                           time_dim=config.time_dim,
                           p_dropout=None).to(device)
    print("torch.load")
    load = torch.load("artifacts/best-model.pth", map_location="cpu")
    print("load_state_dict")
    model.load_state_dict(load, strict=False)
    print("Model loaded successfully.")
  except wandb.CommError as e:
    print(f"Artifact not found: {artifact_name}")
    print(f"Error: {e}")
    raise e
  return model


def test_model(config):
  model = UNetPixelSpace(in_channels=config.num_channels, time_dim=config.time_dim,
                         p_dropout=None).to(device)

  fn = f"artifacts/server-unet-pixel-img32-flickr30k-best-model-v18/best-model.pth"
  model.load_state_dict(torch.load(fn, map_location="cpu"))
  print("Model loaded successfully.")

  print("Performing inference")
  img_shape = (config.num_channels, config.image_size, config.image_size)
  ##### TEXT ENCODER #####
  print("Load LangVAE")
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
  samples = sample_batch_pixels_cond(model, langvae, prompt="a person",
                                     batch_size=4, num_steps=300,
                                     img_shape=(3, 128, 128), device="cpu")
  show_samples(samples, nrow=4, title="RF pixel-space samples")


@torch.no_grad()
def encode_prompt_pool(langvae: LangVAE, prompt, device, max_length=77):
  """
    Returns a pooled text embedding of shape (1, TH) using your LangVAE.
    For determinism you can later swap to the encoder mean (mu) if the API exposes it.
    """
  tok = langvae.decoder.tokenizer(
      prompt,
      return_tensors="pt",
      truncation=True,
      padding="max_length",
      max_length=max_length,
  )
  token_ids = tok["input_ids"].to(device)

  # If we want *deterministic* text,
  # consider modifying LangVAE.encode_z to return mu instead of a sample.
  z, _ = langvae.encode_z(token_ids, mean=True)  # (1, TH)
  return z


@torch.no_grad()
def sample_batch_pixels_cond(
    model, langvae,
    prompt="a person",
    batch_size=4,
    num_steps=300,
    img_shape=(3, 128, 128),
    device="cpu",
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
    t = t_vals[i].repeat(batch_size, 1)  # (B, 1)
    v = model(x, txt_pool, t)  # conditioned step
    x = x + v * dts[i]
  return x


def main():
  env = os.environ.get("ENV", "text_cond/local_train")
  print(f"env={env}")
  config = load_config(env)
  print("Configuration loaded")
  # config.device = device
  os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.env_name == "server" else "false"
  print(f"Seed {config.seed} Device {device}")
  if device == 'cuda':
    torch.set_float32_matmul_precision('high')
  if config.load_and_test_model is True:
    #load_model(model)
    test_model(config)
  if config.train_model is True:
    train_test_model(config)


def load_config(env="text_cond/local_train"):
  base_config = OmegaConf.load("config/text_cond/base.yaml")

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
