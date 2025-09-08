import os, copy, random, wandb, torch

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.optim as optim
import rectified_flow.utils.gdrive_io as gdio
import rectified_flow.training.train_utils as tu

from torchvision import transforms as T
# from torch.optim.lr_scheduler import ConstantLR, CosineAnnealingLR, SequentialLR, LambdaLR
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from rectified_flow.models.unet_pixel_space_joint_sa_film import UNetPixelJointCAFiLM
from rectified_flow.data.coco_tokenized import CocoTokenized
from transformers import CLIPModel
from rectified_flow.encdec.flant5 import FlanT5TextEncoderDecoder


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_test_model(config):
  config_dict = OmegaConf.to_container(config)
  wandb.init(
      project=config.project,
      name=config.name,
      config=config_dict,
      mode=config.wandb_mode,
      settings=wandb.Settings(start_method="thread"),
  )
  wandb.define_metric("*", step_metric="epoch")
  # wandb.define_metric("text/*", step_metric="text_step")
  text_step = 0


  ##### GDrive #####
  service = gdio.auth_drive("client_secret.json", "token.json")

  tok_path = "google/flan-t5-small"

  mean = [0.444, 0.421, 0.384]
  std = [0.275, 0.267, 0.276]
  train_tf = T.Compose([
      T.RandomResizedCrop(224, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BICUBIC, antialias=True),
      T.Resize(config.image_size, interpolation=T.InterpolationMode.BICUBIC, antialias=True),
      T.ToTensor(),
      T.Normalize(mean, std),
  ])
  # val_tf   = T.Compose([T.CenterCrop(224), T.Resize(config.image_size), T.ToTensor(), T.Normalize(mean, std)])

  images_root   = f"{config.data_root}/coco/train2017"
  captions_json = f"{config.data_root}/coco/annotations/captions_train2017.json"
  print("Configuring Data")
  dataset = CocoTokenized(
      images_root=images_root,
      captions_file=captions_json,
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

  ##### Flan-T5 #####
  encdec = FlanT5TextEncoderDecoder(model_name=tok_path, device=DEVICE)

  ###### Model / EMA / Optimizer ######
  model = UNetPixelJointCAFiLM(in_ch=config.num_channels, time_dim=config.time_dim,
                               p_dropout=config.p_dropout).to(DEVICE)
  if config.load_model is True:
    model = tu.load_model(model, config)
  optimizer = torch.optim.AdamW(model.parameters(),lr=config.lr,
                                betas=(0.9,0.999), weight_decay=0.01)

  log_dict = {}
  for epoch in range(1, config.n_epochs+1):
    log_dict["epoch"] = epoch
    tqdm.write(f"Epoch {epoch}/{config.n_epochs}")

    ########################################    
    ########## TRAINING ##################  
    ########################################  
    model.train()
    with tqdm(train_dl, desc="Training") as pbar:
      train_epoch_loss = 0.0
      for images, token_ids, attn_mask, caption in pbar:
        # p_uncond = 0.3 if epoch < 5 else config.p_uncond
        result = tu.compute_data_joint_flant5(model, encdec, images,
                                              token_ids, attn_mask,
                                              DEVICE, p_uncond=config.p_uncond)
        if config.debug is True:
          tu.print_mags(result["v_img_star"], result["v_img_pred"])
          tu.print_mags(result["v_txt_star"], result["v_txt_pred"])
          tqdm.write("\n")

        img_loss = (result["w_img"] * (result["v_img_pred"] - result["v_img_star"]).pow(2)).mean()
        txt_loss = (result["w_txt"] * (result["v_txt_pred"] - result["v_txt_star"]).pow(2)).mean()
        loss = img_loss + txt_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        train_epoch_loss += loss.item()

      train_epoch_loss /= len(train_dl)
    tqdm.write(f"Epoch {epoch}: Train Loss = {train_epoch_loss:.4f}")

    ########################################    
    ########## VALIDATION ##################  
    ########################################  
    model.eval()
    with torch.no_grad():
      with tqdm(val_dl, desc="Validation") as pbar:
        val_epoch_loss = 0.0
        for images, token_ids, attn_mask, caption in pbar:
          result = tu.compute_data_joint_flant5(model, encdec, images,
                                                token_ids, attn_mask,
                                                DEVICE, p_uncond=config.p_uncond)
          if config.debug is True:
            tu.print_mags(result["v_img_star"], result["v_img_pred"])
            tu.print_mags(result["v_txt_star"], result["v_txt_pred"])
            tqdm.write("\n")

          img_loss = (result["w_img"] * (result["v_img_pred"] - result["v_img_star"]).pow(2)).mean()
          txt_loss = (result["w_txt"] * (result["v_txt_pred"] - result["v_txt_star"]).pow(2)).mean()
          loss = img_loss + txt_loss
          val_epoch_loss += loss.item()

        val_epoch_loss /= len(val_dl)
        tqdm.write(f"Epoch {epoch}: Val Loss = {val_epoch_loss:.4f}")

    ########################################    
    ########## INFERENCE ##################  
    ########################################  
    if (epoch % config.inference_peek_num) == 0:
      if config.do_inference is True:
        img_shape = (config.num_channels, config.image_size, config.image_size)
        samples, caps = tu.inference_joint_uncond_flan(
            model,
            encdec,
            batch_size=4,
            num_steps=config.num_sample_steps,
            img_shape=img_shape,
            seq_len=77,
            device=DEVICE,
        )
        tqdm.write(f"captions: {caps}\n")
      if config.local_visualization is True:
        tu.show_samples(samples, nrow=4, title="RF pixel-space samples")
      if config.write_inference_samples is True:
        tu.log_samples_wandb_captions(samples, nrow=4, step=epoch, captions=caps)
        tqdm.write("Wrote grid")
      if config.save_model:
        tqdm.write("Saving model")
        gdio.save_and_upload_model(service, model, config,
                                   drive_path=f"rf_ckpts/{config.name}",
                                   filename="best-model.pth")
        tqdm.write("Save Complete")

    log_dict["train/loss"] = train_epoch_loss
    log_dict["val/loss"] = val_epoch_loss
    wandb.log(log_dict)
    tqdm.write("\n\n")

  tqdm.write("Done Training")


def load_and_test_model(config):
  model = UNetPixelJointCAFiLM(in_ch=config.num_channels, time_dim=config.time_dim,
                        p_dropout=config.p_dropout).to(DEVICE)
  model = tu.load_model(model, config)

  ##### CLIP #####
  print("Load CLIP")
  clip = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
  text_enc = clip.text_model              # CLIPTextTransformer (the text encoder)
  text_enc.eval().to(DEVICE)

  img_shape = (config.num_channels, config.image_size, config.image_size)
  # prompts = ["person", "car", "dog", "ball"]
  prompts = ["car"]
  samples = tu.inference_pixel_space_tc_prompt_list_cfg_clip(model, text_enc, prompts,
                                                             batch_size=1,
                                                             num_steps=config.num_sample_steps,
                                                             img_shape=img_shape,
                                                             guidance_scale=config.guidance_scale,
                                                             device=DEVICE)
  tu.show_samples(samples, nrow=4, title="RF pixel-space samples")


def main():
  env = os.environ.get("ENV", "local")
  print(f"env={env}")
  config = tu.load_config_01(path="config/train_unet_pixel_space_joint", env=env)
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
