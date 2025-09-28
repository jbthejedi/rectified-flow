import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
import torch.nn.functional as F
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from omegaconf import OmegaConf
from tqdm import tqdm
from rectified_flow.models.image_text import *
from rectified_flow.data.datamodule import ProjectData, LangVAECollator
from diffusers import AutoencoderKL

from langvae import LangVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_test_model(config):
    ##### INIT WANDB #####
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )

    ##### TEXT ENCODER #####
    langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128").to(device)

    # Build collator with its tokenizer
    collator = LangVAECollator(
        tokenizer=langvae.decoder.tokenizer,
        max_length=77,
        device=device
    )

    ##### DATA ######
    data = ProjectData(config, device, collator)
    print(f"Len Train: {len(data.train_dl)}")
    print(f"Len Val: {len(data.val_dl)}")

    #### DIFFUSERS/AUTOENCODERKL #######
    print("Load AEKL")
    aekl = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    aekl.eval()
    for p in aekl.parameters(): p.requires_grad = False

    #### BERT ENCODER ######
    print("Load LangVAE")
    # bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    # for p in bert.parameters(): p.requires_grad = False
    langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128")

    #### MODEL #####
    print("Load Model")
    langvae_proj_dim = 128
    model = BaselineJointModel(txt_dim=langvae_proj_dim, img_dim=4, hidden=256)
    if config.compile: model = torch.compile(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        with tqdm(data.train_dl, desc="Training") as pbar:
            model.train()
            train_loss = 0.0
            for images, token_ids, attn_mask in pbar:
                v_pred, u_pred, v_star_img, u_star_txt = compute_data(images, token_ids, attn_mask,
                                                                      aekl, langvae, model, device)

                loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(data.train_dl)
            tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        with tqdm(data.val_dl, desc="Validation") as pbar:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for images, token_ids, attn_mask in pbar:
                    v_pred, u_pred, v_star_img, u_star_txt = compute_data(images, token_ids, attn_mask,
                                                                          aekl, langvae, model, device)

                    loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)
                    val_loss += loss.item()
                    
            val_loss /= len(data.val_dl)
            tqdm.write(f"Epoch {epoch}: Val Loss = {val_loss:.4f}")

        with torch.no_grad():
            model.eval()
            imgs, sentences = sample_joint_batch_vae(model, aekl, langvae, batch_size=4,
                                                 num_steps=50, img_shape=(4, 8, 8))

        grid = vutils.make_grid(imgs.cpu(), nrow=4, normalize=True, pad_value=1.0)
        if config.local_visualization:
            plt.figure(figsize=(3, 3))
            plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
            plt.axis("off")
            plt.show()
        if (epoch % config.inference_peek_num == 0) and config.write_inference_samples:
            tqdm.write("Logging inference samples to wandb")
            images = wandb.Image(grid, caption=f"Epoch {epoch}")
            table = wandb.Table(
                    columns=["sample_id", "sentence"],
                    data=[[i, s] for i, s in enumerate(sentences)]
                )
            wandb.log({
                "epoch": epoch,
                "samples/images": images,
                "samples/texts": table,
            })
        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
        })
        if config.save_model:
            tqdm.write("Saving model")
            best_val_loss = save_and_log_model(model, config, best_val_loss, val_loss)

    print("Done Training")


def compute_data(images, token_ids, attn_mask, aekl, langvae : LangVAE, model, device):
    images, token_ids = images.to(device), token_ids.to(device)

    # Encode Image
    with torch.no_grad():
        # Posterior is DiagonalGaussianDistribution
        posterior = aekl.encode(images).latent_dist
        x_img_1 = posterior.mean * aekl.config.scaling_factor                        # (B, IH, 64//8, 64//8)
    
    # Encode text
    # TODO not using attn_mask might throw things off.
    z, _ = langvae.encode_z(token_ids)
    x_txt_1 = z                                                                      # (B, TH)

    # outputs = bert(input_ids=token_ids, attention_mask=attn_mask)        
    # x_txt_1 = outputs.pooler_output                                                # (B, TH)

    B = x_img_1.size(0)
    
    # Sample base noise
    x_img_0 = torch.randn_like(x_img_1)                                             # (B, IH, 8, 8)
    x_txt_0 = torch.randn_like(x_txt_1)                                             # (B, TH)

    # Sample timestep
    t = torch.rand(B, 1, device=device)                                             # (B, 1)

    # Interpolate
    x_img_t = (1 - t.view(B, 1, 1, 1)) * x_img_0 + t.view(B, 1, 1, 1) * x_img_1     # (B, IH, 8, 8)
    x_txt_t = (1 - t) * x_txt_0 + t * x_txt_1                                       # (B, TH)

    # Target velocity 
    v_star_img = x_img_t - x_img_0                                                  # (B, IH, 8, 8)
    u_star_txt = x_txt_t - x_txt_0                                                  # (B, L, TH)

    # Predict velocity
    v_pred, u_pred = model(x_img_t, x_txt_t, t)

    return v_pred, u_pred, v_star_img, u_star_txt


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
        artifact = wandb.Artifact(
            name=f"{config.name}-best-model",
            type="model",
            description="Continuously updated best model"
        )
        artifact.add_file(filename)
        wandb.log_artifact(artifact, aliases=["latest", "best"])

    return best_val_loss


@torch.no_grad()
def sample_joint_batch_vae(model, aekl, langvae, batch_size=4, num_steps=200, img_shape=(4, 8, 8)):
    device = next(model.parameters()).device
    txt_dim = langvae.latent_dim 

    # --- Start from Gaussian noise in latent space ---
    x_img = torch.randn(batch_size, *img_shape, device=device)   # (B, 4, 8, 8)
    x_txt = torch.randn(batch_size, txt_dim, device=device)      # (B, 128)

    # Time discretization (from 1 → 0, backward integration)
    t_vals = torch.linspace(1.0, 0.0, num_steps, device=device)

    for i in range(len(t_vals) - 1):
        t = t_vals[i].expand(batch_size, 1)  # (B,1)
        dt = t_vals[i + 1] - t_vals[i]

        # Predict velocities from joint model
        v_pred_img, u_pred_txt = model(x_img, x_txt, t)

        # Euler update
        x_img = x_img + v_pred_img * dt
        x_txt = x_txt + u_pred_txt * dt

    # --- Decode image latent → RGB image ---
    imgs = aekl.decode(x_img / aekl.config.scaling_factor).sample  # (B,3,H,W)
    imgs = (imgs.clamp(-1, 1) + 1) / 2  # map to [0,1]

    # --- Decode text latent → sentences ---
    sentences = langvae.decode_sentences(x_txt)

    return imgs, sentences


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