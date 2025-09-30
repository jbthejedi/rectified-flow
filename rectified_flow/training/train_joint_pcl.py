import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

import torch
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import time
import torch.nn.functional as F
import torch.optim as optim
import wandb
import matplotlib.pyplot as plt
import torchvision.utils as vutils

from omegaconf import OmegaConf
from tqdm import tqdm
from rectified_flow.models.image_text import *
from rectified_flow.data.datamodule import *
from torch.utils.data import DataLoader
from diffusers import AutoencoderKL

from langvae import LangVAE

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_test_model(config):
    ### PRE TESTING GPU ####
    if torch.cuda.is_available():
        print("Device count:", torch.cuda.device_count())
        print("Using:", torch.cuda.get_device_name(0))

    ##### INIT WANDB #####
    config_dict = OmegaConf.to_container(config)
    wandb.init(
        project=config.project,
        name=config.name,
        config=config_dict,
        mode=config.wandb_mode,
    )

    ##### TEXT ENCODER #####
    print("Load LangVAE")
    # bert = BertModel.from_pretrained("bert-base-uncased").to(device)
    # for p in bert.parameters(): p.requires_grad = False
    langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128")
    langvae = langvae.to(device)
    langvae.eval()
    for p in langvae.parameters(): p.requires_grad = False
    langvae.decoder.to(device)

    latents_dir = "./precomputed_latents"  # same as OUT_DIR in precompute.py

    train_ds = PrecomputedLatents(latents_dir)
    val_ds   = PrecomputedLatents(latents_dir)   # simple: same pool; or split indices if you want
    print(f"#train files: {len(train_ds)}  #val files: {len(val_ds)}")

    train_dl = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        # prefetch_factor=4,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
        persistent_workers=False,
        # prefetch_factor=4,
    )

    #### DIFFUSERS/AUTOENCODERKL #######
    print("Load AEKL")
    aekl = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse")
    aekl = aekl.to(device)
    aekl.eval()
    for p in aekl.parameters(): p.requires_grad = False


    #### MODEL #####
    print("Load Model")
    langvae_proj_dim = 128
    # model = BaselineJointModel(txt_dim=langvae_proj_dim, img_dim=4, hidden=256, p_hidden=config.image_size)
    model = BaselineJointModel(txt_dim=langvae_proj_dim, img_dim=4, hidden=256, p_hidden=64)
    if config.compile:
        print("Compile Mode = TRUE")
        model = torch.compile(model)
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # assert_cuda_ready(model, aekl, langvae, device)

    def timed_iter(dataloader):
        it = iter(dataloader)
        while True:
            t0 = time.perf_counter()
            try:
                batch = next(it)   # <-- includes worker time + transfers + collate
            except StopIteration:
                return
            t1 = time.perf_counter()
            yield batch, (t1 - t0)

    best_val_loss = float("inf")
    for epoch in range(1, config.n_epochs+1):
        tqdm.write(f"Epoch {epoch}/{config.n_epochs}")
        with tqdm(train_dl, desc="Training") as pbar:
            model.train()
            train_loss = 0.0
            tqdm.write("Pre Iter Fetch")
            for (x_img_1, x_txt_1, caps_), t_fetch in timed_iter(pbar):
                print(f"Time fetch Train {t_fetch}")
                v_pred, u_pred, v_star_img, u_star_txt = compute_data_from_latents(x_img_1, x_txt_1, model, device)
                loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_dl)
            tqdm.write(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

        with tqdm(val_dl, desc="Validation") as pbar:
            model.eval()
            with torch.no_grad():
                val_loss = 0.0
                for (x_img_1, x_txt_1, caps_), t_fetch in timed_iter(pbar):
                    print(f"Time fetch Val {t_fetch}")
                    v_pred, u_pred, v_star_img, u_star_txt = compute_data_from_latents(x_img_1, x_txt_1, model, device)
                    loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)
                
                    val_loss += loss.item()
                    
            val_loss /= len(val_dl)
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


def compute_data_from_latents(x_img_1, x_txt_1, model, device):
    """
    x_img_1: (B, 4, 8, 8) precomputed, already scaled
    x_txt_1: (B, D)       precomputed LangVAE z
    """
    B = x_img_1.size(0)

    # Move to GPU
    t1 = time.time()
    x_img_1 = x_img_1.to(device, non_blocking=True)
    x_txt_1 = x_txt_1.to(device, non_blocking=True)

    # Sample base noise
    x_img_0 = torch.randn_like(x_img_1)
    x_txt_0 = torch.randn_like(x_txt_1)

    # Sample timesteps
    t = torch.rand(B, 1, device=device)

    # Interpolate along straight line (RF)
    x_img_t = (1 - t).view(B,1,1,1) * x_img_0 + t.view(B,1,1,1) * x_img_1
    x_txt_t = (1 - t) * x_txt_0 + t * x_txt_1

    # Targets (velocities)
    v_star_img = x_img_t - x_img_0        # (B,4,8,8)
    u_star_txt = x_txt_t - x_txt_0        # (B,D)

    # Predict
    v_pred, u_pred = model(x_img_t, x_txt_t, t)  # same API as before
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
        print("Set float32_matmul_precision high")
        torch.set_float32_matmul_precision('high')

    if config.summary:
        raise NotImplementedError("Not Implemented")

    elif config.train_model:
        print("Training Model start...")
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
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass
    main()