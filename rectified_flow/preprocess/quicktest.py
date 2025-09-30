import os, json, math, torch, time
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusers import AutoencoderKL
from langvae import LangVAE

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# ---------- Config ----------
env = os.environ.get("ENV", "local")
cfg = OmegaConf.load("config/base.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.load(f"config/{env}.yaml"))

IMAGES_ROOT   = f"{cfg.data_root}/flickr30k/Images"
CAPTIONS_FILE = f"{cfg.data_root}/flickr30k/captions.txt"
OUT_DIR       = "./precomputed_latents"
BATCH_SIZE    = 16 # tune for your VRAM
NUM_WORKERS   = 8    # tune for your CPU
MAX_LEN       = 77

os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128").to(device)
langvae.encoder.to(device)
langvae.decoder.to(device)   
for n, p in langvae.named_parameters():
    print(n, p.device)
langvae.eval()

[setattr(p, "requires_grad", False) for p in langvae.parameters()]
ids = torch.randint(0, langvae.encoder.config.vocab_size, (2,77), device=device)
with torch.autocast("cuda", dtype=torch.float16):
    z, _ = langvae.encode_z(ids)
print("z device:", z.device)