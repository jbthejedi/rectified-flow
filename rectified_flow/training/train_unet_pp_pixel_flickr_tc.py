import os, copy, random, wandb, torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from torchvision import transforms as T
import torch.optim as optim

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from rectified_flow.models.unet_pixel_space_sa_tc import *
import rectified_flow.training.train_utils as tu
from rectified_flow.data.flickr30k_tokenized import Flickr30kTokenized


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


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
      T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
      T.Resize(config.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
      T.ToTensor(),
      T.Normalize(mean, std),
  ])
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
  train_dl = DataLoader(train_set,
                        batch_size=config.batch_size,
                        shuffle=True,
                        num_workers=config.num_workers,
                        pin_memory=config.pin_memory)
  val_dl = DataLoader(val_set,
                      batch_size=config.batch_size,
                      shuffle=False,
                      num_workers=config.num_workers,
                      pin_memory=config.pin_memory)
  print(f"train len {len(train_dl)}")
  print(f"val len {len(val_dl)}")

  ##### VAE: LangVAE #####
  print("Load LangVAE")
  langvae = tu.get_langvae(DEVICE)

  # Model / EMA / Optimizer
  model = UNetPixelPlusTCPool(in_ch=config.num_channels, time_dim=config.time_dim,
                         p_dropout=config.p_dropout).to(DEVICE)
  if config.load_model is True:
    model = tu.load_model(model, config)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, betas=(0.9,0.999), weight_decay=0.01)

  log_dict = {}
  # best_val_loss = float("inf")
  for epoch in range(1, config.n_epochs+1):
    log_dict["epoch"] = epoch
    tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
    model.train()
    with tqdm(train_dl, desc="Training") as pbar:
      train_epoch_loss = 0.0
      for images, token_ids, attn_mask in pbar:
        v, v_pred = tu.compute_data_tc_langvae_cfg(model, langvae, images,
                                                   token_ids, DEVICE, p_uncond=config.p_uncond)
        if config.debug is True: tu.print_mags(v, v_pred)
        loss = ((v_pred - v)**2).mean()

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
        for images, token_ids, attn_mask in pbar:
          v, v_pred = tu.compute_data_tc_langvae(model, langvae, images, token_ids, DEVICE)
          if config.debug is True: tu.print_mags(v, v_pred)
          val_epoch_loss += ((v_pred - v)**2).mean().item()
        val_epoch_loss /= len(val_dl)
        tqdm.write(f"Epoch {epoch}: Val Loss = {val_epoch_loss:.4f}")

    if (epoch % config.inference_peek_num) == 0:
      img_shape = (config.num_channels, config.image_size, config.image_size)
      prompts = ["woman", "flower", "dog", "ball"]
      samples = tu.inference_pixel_space_tc_prompt_list_cfg(model, langvae, prompts,
                                                            num_steps=config.num_sample_steps,
                                                            img_shape=img_shape,
                                                            guidance_scale=config.guidance_scale,
                                                            device=DEVICE)
      if config.local_visualization is True:
        tu.show_samples(samples, nrow=4, title="RF pixel-space samples")
      if config.write_inference_samples is True:
        tu.log_samples_wandb(samples, nrow=4, step=epoch)
        tqdm.write("Writing image grid")
      if config.save_model:
        tqdm.write("Saving model")
        tu.save_and_log_model(model, config)

    log_dict["train/loss"] = train_epoch_loss
    log_dict["val/loss"] = val_epoch_loss
    wandb.log(log_dict, step=epoch, commit=True)

  tqdm.write("Done Training")


def load_and_test_model(config):
  model = UNetPixelPlusTCPool(in_ch=config.num_channels, time_dim=config.time_dim,
                          p_dropout=config.p_dropout).to(DEVICE)
  model = tu.load_model(model, "v98", config)

  ##### VAE: LangVAE #####
  print("Load LangVAE")
  langvae = tu.get_langvae(DEVICE)

  img_shape = (config.num_channels, config.image_size, config.image_size)
  prompts = ["person", "ball"]
  samples = tu.inference_pixel_space_tc_prompt_list_cfg(model, langvae, prompts,
                                                        num_steps=config.num_sample_steps,
                                                        img_shape=img_shape,
                                                        guidance_scale=config.guidance_scale,
                                                        device=DEVICE)
  tu.show_samples(samples, nrow=4, title="RF pixel-space samples")


def main():
  env = os.environ.get("ENV", "local")
  print(f"env={env}")
  config = tu.load_config_01(path="config/train_unet_pp_pixel_flickr_tc", env=env)
  tu.print_config_vars(config)
  print("Configuration loaded")
  os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.env_name == "server" else "false"
  config.env = env
  print(f"Seed {config.seed}")

  if DEVICE == 'cuda':
    torch.set_float32_matmul_precision('high')
  if config.load_and_test_model is True:
    load_and_test_model(config)
  if config.train_model is True:
    train_test_model(config)


if __name__ == '__main__':
  main()
