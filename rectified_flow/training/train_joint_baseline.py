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
from rectified_flow.data.flickr30k_tokenized import *
from torch.utils.data import DataLoader, Subset
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
    langvae.encoder.to(device); langvae.decoder.to(device)

    # after loading langvae
    tok_path = langvae.decoder.tokenizer.name_or_path
    # If that’s not a real directory (e.g. came from HF hub cache), make one workers can read:
    if not os.path.isdir(tok_path):
        tok_path = "./langvae_tokenizer_ckpt"
        os.makedirs(tok_path, exist_ok=True)
        langvae.decoder.tokenizer.save_pretrained(tok_path)  # writes vocab + merges

    mean = [0.444, 0.421, 0.384]
    std  = [0.275, 0.267, 0.276]
    train_tf = T.Compose([T.CenterCrop(224), T.Resize(config.image_size), T.ToTensor(), T.Normalize(mean, std)])
    val_tf   = T.Compose([T.CenterCrop(224), T.Resize(config.image_size), T.ToTensor(), T.Normalize(mean, std)])

    images_root   = f"{config.data_root}/flickr30k/Images"
    captions_file = f"{config.data_root}/flickr30k/captions.txt"


    # --- build ONE base dataset ---
    base_ds = Flickr30kTokenized(
        images_root=images_root,
        captions_file=captions_file,
        tokenizer_name_or_path=tok_path,
        transform=train_tf,         # you can swap transforms per Subset later
        max_length=77,
    )

    # (optional) small sample BEFORE splitting
    if config.do_small_sample:
        import random
        random.seed(config.seed)
        idxs = random.sample(range(len(base_ds)), k=config.sample_size_k)
        base_ds = Subset(base_ds, idxs)
        print(f"Sampled base_ds: {len(base_ds)}")

    # --- deterministic split ---
    n = len(base_ds)
    n_train = int(config.p_train_len * n)
    g = torch.Generator().manual_seed(getattr(config, "split_seed", 1337))
    perm = torch.randperm(n, generator=g)

    train_idx = perm[:n_train].tolist()
    val_idx   = perm[n_train:].tolist()

    train_ds = Subset(base_ds, train_idx)
    val_ds   = Subset(base_ds, val_idx)

    print(f"train: {len(train_ds)} | val: {len(val_ds)}")

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
            # tqdm.write("Pre Iter Fetch")
            for (images, token_ids, attn_mask), t_fetch in timed_iter(pbar):
                v_pred, u_pred, v_star_img, u_star_txt = compute_data(images, token_ids, attn_mask, aekl, langvae, model, device)
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
                for (images, token_ids, attn_mask), t_fetch in timed_iter(pbar):
                    v_pred, u_pred, v_star_img, u_star_txt = compute_data(images, token_ids, attn_mask, aekl, langvae, model, device)
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


def compute_data(images, token_ids, attn_mask, aekl, langvae : LangVAE, model, device):
    images = images.to(device, non_blocking=True)
    token_ids = token_ids.to(device, non_blocking=True)
    attn_mask = attn_mask.to(device, non_blocking=True)
    # assert_batch_devices(images, token_ids, attn_mask, device)

    # Encode Image
    t1 = time.time()
    # tqdm.write("start img encode") 
    amp_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            posterior = aekl.encode(images).latent_dist
        x_img_1 = posterior.mean * aekl.config.scaling_factor          # (B,4,8,8)
    assert x_img_1.is_cuda, "AEKL.encode returned CPU tensor"
    # tqdm.write(f"encode img {time.time() - t1}") 
    
    # Encode text
    # TODO not using attn_mask might throw things off.
    t2 = time.time()
    with torch.no_grad():
        with torch.autocast(device_type="cuda", dtype=amp_dtype):
            z, _ = langvae.encode_z(token_ids)
            # tqdm.write(f"encode_z output device: {z.device}")
    x_txt_1 = z                                                                      # (B, TH)
    # tqdm.write(f"encode text {time.time() - t2}") 

    # outputs = bert(input_ids=token_ids, attention_mask=attn_mask)        
    # x_txt_1 = outputs.pooler_output                                                # (B, TH)

    B = x_img_1.size(0)
    
    t3 = time.time()
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
    # tqdm.write(f"tensor math {time.time() - t3}") 

    # assert_latents_shapes_devices(x_img_1, x_txt_1, x_img_t, x_txt_t, v_star_img, u_star_txt, device)

    # Predict velocity
    t4 = time.time()
    v_pred, u_pred = model(x_img_t, x_txt_t, t)
    # tqdm.write(f"prediction {time.time() - t4}") 
    # assert_model_outputs(v_pred, u_pred, v_star_img, u_star_txt, device)


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
    print(f"sentences {sentences}")

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