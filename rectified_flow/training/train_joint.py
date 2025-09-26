import os
import random
import torch
import torch.nn.functional as F
import torch.optim as optim

from omegaconf import OmegaConf
from tqdm import tqdm
from rectified_flow.models.image_text import *
from rectified_flow.models.image import *
from rectified_flow.data.datamodule import ProjectData
from diffusers import AutoencoderKL
from torch.utils.data import DataLoader
from transformers import BertModel

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# def collate(batch):
#     L = []
#     for item in batch:
#         x = item['pixel_values']
#         L.append(x)
#     pixel_values = torch.stack([item["pixel_values"] for item in batch])
#     captions = [item["caption"] for item in batch]
#     return {"pixel_values": pixel_values, "caption": captions}


def train_test_model(config):
    data = ProjectData(config, device)
    # train_dl = DataLoader(data.train_set, batch_size=config.batch_size, shuffle=True)
    # val_dl   = DataLoader(data.val_set, batch_size=config.batch_size)

    img_shape = (config.num_channels, config.image_size, config.image_size)
    # model = UNet(in_channels=config.num_channels, config=config).to(device)
    bert_dim = 768
    model = BaselineJointModel(
        txt_dim=768, img_dim=4, hidden=256
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #### DIFFUSERS/AUTOENCODERKL #######
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    #### BERT ENCODER ######
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float("inf")
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        with tqdm(data.train_dl, desc="Training") as pbar:
            model.train()
            train_loss = 0.0
            for batch_idx, (images, token_ids, attn_mask) in enumerate(pbar):
                images, token_ids = images.to(device), token_ids.to(device)

                # Image -> latent image
                with torch.no_grad():
                    # Posterior is DiagonalGaussianDistribution
                    posterior = vae.encode(images).latent_dist
                    x_img_1 = posterior.mean * vae.config.scaling_factor                    # (B, IH, 64//8, 64//8)
                
                outputs = bert(input_ids=token_ids, attention_mask=attn_mask)        
                x_txt_1 = outputs.pooler_output                                             # (B, TH)

                B = x_img_1.size(0)
                
                # Sample base noise
                x_img_0 = torch.randn_like(x_img_1)                                         # (B, IH, 8, 8)
                x_txt_0 = torch.randn_like(x_txt_1)                                         # (B, TH)

                # Sample timestep
                t = torch.rand(B, 1, device=device)                                         # (B, 1)

                # Interpolate
                x_img_t = (1 - t) * x_img_0 + t * x_img_1                                   # (B, IH, 8, 8)
                x_txt_t = (1 - t) * x_txt_0 + t * x_txt_1                                   # (B, TH)

                # Target velocity 
                v_star_img = x_img_t - x_img_0                                              # (B, IH, 8, 8)
                u_star_txt = x_txt_t - x_txt_0                                              # (B, L, TH)

                # Predict velocity
                v_pred, u_pred = model(x_img_t, x_txt_t, t)

                # Loss
                loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)

                # Update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

                if batch_idx == 2:
                    break

            train_loss /= len(data.train_dl)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        # with tqdm(data.val_dl, desc="Validation") as pbar:
        #     model.eval()
        #     eval_losses = []
        #     with torch.no_grad():
        #         for images, token_ids, _ in pbar:
        #             exit(0)


def train_test_model_ca(config):
    data = ProjectData(config, device)
    # train_dl = DataLoader(data.train_set, batch_size=config.batch_size, shuffle=True)
    # val_dl   = DataLoader(data.val_set, batch_size=config.batch_size)

    img_shape = (config.num_channels, config.image_size, config.image_size)
    # model = UNet(in_channels=config.num_channels, config=config).to(device)
    bert_dim = 768
    model = BaselineJointModel(
        txt_dim=768, img_dim=4, hidden=256
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    #### DIFFUSERS/AUTOENCODERKL #######
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False

    #### BERT ENCODER ######
    bert = BertModel.from_pretrained("bert-base-uncased").to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    best_val_loss = float("inf")
    for epoch in range(1, config.n_epochs + 1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs+1}")
        with tqdm(data.train_dl, desc="Training") as pbar:
            model.train()
            train_losses = []
            for images, token_ids, attn_mask in pbar:
                images, token_ids = images.to(device), token_ids.to(device)

                # Image -> latent image
                with torch.no_grad():
                    # Posterior is DiagonalGaussianDistribution
                    posterior = vae.encode(images).latent_dist
                    x_img_1 = posterior.mean * vae.config.scaling_factor                    # (B, IH, 64//8, 64//8)
                
                outputs = bert(input_ids=token_ids, attention_mask=attn_mask)        
                x_txt_1 = outputs.last_hidden_state                                         # (B, L, TH)
                # x_txt_1 = outputs.pooler_output                                             # (B, TH)

                B = x_img_1.size(0)
                
                # Sample base noise
                x_img_0 = torch.randn_like(x_img_1)                                         # (B, IH, 8, 8)
                x_txt_0 = torch.randn_like(x_txt_1)                                         # (B, L, TH)

                # Sample timestep
                t = torch.rand(B, 1, device=device)                                         # (B, 1)

                # Interpolate
                x_img_t = (1 - t) * x_img_0 + t * x_img_1                                   # (B, IH, 8, 8)
                x_txt_t = (1 - t.view(B, 1, 1)) * x_txt_0 + t.view(B, 1, 1) * x_txt_1       # (B, L, TH)

                # Target velocity 
                v_star_img = x_img_t - x_img_0                                              # (B, IH, 8, 8)
                u_star_txt = x_txt_t - x_txt_0                                              # (B, L, TH)

                # Predict velocity
                v_pred, u_pred = model(x_img_t, x_txt_t, t)

        #         # Loss
        #         loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #         train_loss += loss.item()
        #         exit(0)

        # with tqdm(data.val_dl, desc="Validation") as pbar:
        #     model.eval()
        #     eval_losses = []
        #     with torch.no_grad():
        #         for images, token_ids, _ in pbar:
        #             exit(0)


def sample_joint_batch(model, batch_size=16, num_steps=200, img_shape=(1, 28, 28), txt_dim=32):
    device = next(model.parameters()).device

    # --- Start from Gaussian noise for both modalities ---
    x_img = torch.randn(batch_size, *img_shape, device=device)   # noise image
    x_txt = torch.randn(batch_size, txt_dim, device=device)      # noise text embedding

    # Time discretization (from 0 → 1)
    t_vals = torch.linspace(0.0, 1.0, num_steps, device=device)

    for i in range(len(t_vals)-1):
        t = t_vals[i].expand(batch_size, 1)   # shape (B,1)
        dt = t_vals[i+1] - t_vals[i]

        # Predict velocities
        v_pred_img, u_pred_txt = model(x_img, x_txt, t)

        # Euler update
        x_img = x_img + v_pred_img * dt
        x_txt = x_txt + u_pred_txt * dt

    return x_img.clamp(0, 1), x_txt


def decode_text(model, x_txt):
    # x_txt: (B, hidden_dim)
    vocab_emb = model.txt_enc.weight.detach()   # (10, hidden_dim)
    # Compute cosine similarity
    sims = torch.matmul(F.normalize(x_txt, dim=-1),
                        F.normalize(vocab_emb.T, dim=0))
    preds = sims.argmax(dim=-1)  # predicted labels
    return preds


def main():
    env = os.environ.get("ENV", "local")
    print(f"env={env}")
    config = load_config(env)
    print("Configuration loaded")
    config.device, config.env = device, env
    print(f"Seed {config.seed} Device {config.device}")
    if config.device == 'cuda':
        torch.set_float32_matmul_precision('high')

    # if config.summary:
    #     print_summary(config)
    #     exit()

    elif config.train_model:
        train_test_model(config)
    elif config.inference:
        test_model(config)


def load_config(env="local"):
    base_config = OmegaConf.load("config/base.yaml")

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