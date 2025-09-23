
import os
import random
from omegaconf import OmegaConf
import torch
from torchvision import datasets, transforms as T
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from rectified_flow.models.image_text import *
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_test_model(config):
    dataset = datasets.MNIST(
        root=config.data_root,
        download=False,
        transform=T.Compose([
            T.ToTensor(),
        ])
    )

    if config.do_small_sample:
        indices = random.sample(range(len(dataset)), 1000)
        dataset = Subset(dataset, indices)
        print(len(dataset))
    train_len = int(len(dataset) * config.p_train_len)
    train_set, val_set = random_split(dataset, [train_len, len(dataset) - train_len])
    train_dl = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_dl   = DataLoader(val_set, batch_size=config.batch_size, shuffle=False)

    model = JointRFModelMNIST().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    enc_dim = 32
    for epoch in range(config.n_epochs):
        with tqdm(train_dl, desc="Training") as pbar:
            train_loss = 0.0
            for x_img_1, labels in pbar:
                B = x_img_1.size(0)                                         # (8)
                x_img_1, labels = x_img_1.to(device), labels.to(device)
                
                # Sample base noise
                x_img_0 = torch.randn_like(model.img_enc(x_img_1))          # (B, 1, 28, 28)
                x_txt_0 = torch.randn(B, enc_dim, device=device)            # (B, 1, 28, 28)

                # Sample timestep
                t = torch.rand(B, 1, device=device)                         # (B, 1)

                # Interpolate
                img_feat_1 = model.img_enc(x_img_1)                         # (B, 32)
                txt_feat_1 = model.txt_enc(labels)                          # (B, 32)

                x_img_t = (1 - t) * x_img_0 + t * img_feat_1                # (B, 32)
                x_txt_t = (1 - t) * x_txt_0 + t * txt_feat_1                # (B, 32)

                # Target velocity 
                v_star_img = img_feat_1 - x_img_0
                u_star_txt = txt_feat_1 - x_txt_0

                # Predict velocity
                x_img_t_ = x_img_t                                          # (B, 32, 1)
                x_txt_t_ = x_txt_t                                          # (B, 32, 1)
                v_pred, u_pred = model(x_img_t_, x_txt_t_, t)

                # Loss
                loss = F.mse_loss(v_pred, v_star_img) + F.mse_loss(u_pred, u_star_txt)
                opt.zero_grad()
                loss.backward()
                opt.step()

                train_loss += loss.item()
            train_loss /= len(train_dl)
            print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}")

    
    # with torch.no_grad():
    #     model.eval()
    #     imgs = sample_joint_batch(model, batch_size=16, num_steps=50, img_shape=(1, 28, 28))

    # # make a nice 4x4 grid
    # grid = vutils.make_grid(imgs.cpu(), nrow=4, normalize=True, pad_value=1.0)

    # plt.figure(figsize=(6,6))
    # plt.imshow(grid.permute(1, 2, 0).numpy(), cmap="gray")
    # plt.axis("off")
    # plt.show()


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

def test_model(config):
    pass

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