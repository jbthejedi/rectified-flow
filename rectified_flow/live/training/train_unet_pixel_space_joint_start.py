import os, copy, random, wandb, torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
# torch.backends.cudnn.benchmark = True

import matplotlib.pyplot as plt
import torch.optim as optim
import torchvision.utils as vutils
import rectified_flow.live.training.train_utils_live as tu

from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm

from rectified_flow.data.datamodule_recover import ProjectData
from rectified_flow.live.models.unet_pixel_space_live import UNetJoint

from transformers import CLIPModel
import rectified_flow.utils.gdrive_io as gdio

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

  ##### GDrive Service #####
  service = gdio.auth_drive("client_secret.json", "token.json")

  #########################
  ########## DATA #########
  #########################
  # TODO swap with COCO2017
  print("Downloading data")
  dataset = ProjectData(config, DEVICE).dataset
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

  #########################
  ########## FLAN-T5 #########
  #########################
  # TODO add FLAN-T5

  ################################
  ########## Mod/Opt ##########
  ################################
  # TODO swap new text-conditioning model
  model = UNetJoint(in_ch=config.num_channels, 
                         time_dim=config.time_dim, p_dropout=0.1).to(DEVICE)
  optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

  log_dict = {}
  for epoch in range(1, config.n_epochs+1):
    tqdm.write(f"Epoch {epoch}/{config.n_epochs}")

    ################################
    ########## Training ##########
    ################################
    model.train()
    train_epoch_loss = 0.0
    with tqdm(train_dl, desc="Training") as pbar:
      for images, _ in pbar:
        # TODO return images, captions, attention_mask
        v_star, v_pred = tu.compute_data(images, model, device=DEVICE)
        if config.debug is True: tu.print_mag(v_star, v_pred)
        loss = ((v_pred - v_star)**2).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_epoch_loss += loss.item()
      train_epoch_loss /= len(train_dl)
      log_dict["train_loss"] = train_epoch_loss
      tqdm.write(f"Train Loss {train_epoch_loss}")

    ################################
    ########## Validation ##########
    ################################
    model.eval()
    with torch.no_grad():
      val_epoch_loss = 0.0
      with tqdm(val_dl, desc="Val") as pbar:
        for images, _ in pbar:
          v_star, v_pred = tu.compute_data(images, model, device=DEVICE)
          if config.debug is True: tu.print_mag(v_star, v_pred)
          loss = ((v_pred - v_star)**2).mean()
          val_epoch_loss += loss.item()
        val_epoch_loss /= len(val_dl)
        log_dict["val_loss"] = val_epoch_loss
        tqdm.write(f"Val {val_epoch_loss}")

    
    ################################################
    ########## Text-conditioning metrics ###########
    ################################################
    # TODO implement text_triplet_metrics, null_proximity_stats


    ################################
    ########## Inference ##########
    ################################
    if (epoch % config.inference_peek_num) == 0:
      if config.do_inference is True:
        img_shape = (config.num_channels, config.image_size, config.image_size)
      # TODO change inference to use text conditioning
        samples = tu.sample_batch_pixels(model, batch_size=4,
                                        img_shape=img_shape,
                                        num_steps=config.num_sample_steps,
                                        device=DEVICE)
      if config.local_visualization is True:
        tu.show_samples(samples, nrow=4, title="RF pixel-space samples")
      if config.write_inference_samples is True:
        tu.log_samples_wandb(samples, nrow=4, step=epoch)
        tqdm.write("Wrote grid")
      if config.save_model:
        tqdm.write("Saving model")
        gdio.save_and_upload_model(service, model, config,
                                   drive_path=f"rf_ckpts/{config.name}",
                                   filename="best-model.pth")
        tqdm.write("Save Complete")

    wandb.log(log_dict, step=epoch, commit=True)


def main():
  env = os.environ.get("ENV", "local")
  print(f"env={env}")
  config = tu.load_config_01(path="config/live/train_unet_pp_pixel_coco_tc_clip", env=env)
  tu.print_config_vars(config)
  print("Configuration loaded")
  os.environ["TOKENIZERS_PARALLELISM"] = "true" if config.env_name == "server" else "false"
  config.env = env
  print(f"Seed {config.seed}")

  if DEVICE == 'cuda':
    torch.set_float32_matmul_precision('high')

  if config.train_model is True:
    train_test_model(config)


if __name__ == '__main__':
  main()
