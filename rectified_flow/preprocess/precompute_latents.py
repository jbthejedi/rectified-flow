# precompute_latents.py
import os
import torch
from tqdm import tqdm
from torchvision import transforms as T
from torch.utils.data import DataLoader

from diffusers import AutoencoderKL
from langvae import LangVAE
from rectified_flow.data.datamodule import Flickr30kDataset
from omegaconf import OmegaConf

########## TESTING ########## 
print(torch.__version__)
print(torch.version.cuda)
print(torch.backends.cudnn.version())
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))

x = torch.randn(32, 3, 64, 64, device="cuda")
for _ in tqdm(range(1_000_000_000)):
    y = torch.nn.functional.conv2d(x, torch.randn(16, 3, 3, 3, device="cuda"))
torch.cuda.synchronize()

##########

device = 'cuda' if torch.cuda.is_available() else 'cpu'

base_config = OmegaConf.load("config/base.yaml")
env = os.environ.get("ENV", "local")
print(f"env={env}")

env_path = f"config/{env}.yaml"
if os.path.exists(env_path):
    env_config = OmegaConf.load(env_path)
    # Merges env_config into base_config (env overrides base)
    config = OmegaConf.merge(base_config, env_config)
else:
    raise Exception("No config file exists")

print("Configuration loaded")

IMAGES_ROOT = f"{config.data_root}/flickr30k/Images"
CAPTIONS_FILE = f"{config.data_root}/flickr30k/captions.txt"
OUT_DIR = "./precomputed_latents"
os.makedirs(OUT_DIR, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# Load models
aekl = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
aekl.eval()
for p in aekl.parameters():
    p.requires_grad = False

langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128").to(device)
langvae.eval()
for p in langvae.parameters():
    p.requires_grad = False
    assert p.device.type == "cuda", "LangVAE param still on CPU"

# Dataset + transform
tf = T.Compose([
    T.CenterCrop(224),
    T.Resize(64),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])
dataset = Flickr30kDataset(IMAGES_ROOT, CAPTIONS_FILE, transform=tf)
loader = DataLoader(dataset, batch_size=64, num_workers=4, pin_memory=True)

print(f"Total samples: {len(dataset)}")

# Loop and precompute
for idx in tqdm(range(len(loader))):
    img, caption = dataset[idx]

    # ---- Image latent ----
    img = img.unsqueeze(0).to(device)  # (1,3,H,W)
    with torch.no_grad():
        posterior = aekl.encode(img).latent_dist
        img_latent = (posterior.mean * aekl.config.scaling_factor).cpu()

    # ---- Text latent ----
    token_ids = langvae.decoder.tokenizer(
        caption,
        padding="max_length",
        truncation=True,
        max_length=77,
        return_tensors="pt"
    )["input_ids"].to(device)

    with torch.no_grad():
        z, _ = langvae.encode_z(token_ids)
        txt_latent = z.cpu()

    # ---- Save ----
    torch.save(
        {
            "image_latent": img_latent,
            "text_latent": txt_latent,
            "caption": caption,
        },
        os.path.join(OUT_DIR, f"sample_{idx:06d}.pt")
    )

print(f"Done. Latents saved to {OUT_DIR}")
