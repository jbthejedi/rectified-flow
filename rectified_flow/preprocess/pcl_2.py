# precompute_latents_batched.py
import os, json, math
import torch
import time
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from omegaconf import OmegaConf

from diffusers import AutoencoderKL
from langvae import LangVAE

# ---------- Config ----------
env = os.environ.get("ENV", "local")
cfg = OmegaConf.load("config/base.yaml")
cfg = OmegaConf.merge(cfg, OmegaConf.load(f"config/{env}.yaml"))
IMAGES_ROOT   = f"{cfg.data_root}/flickr30k/Images"
CAPTIONS_FILE = f"{cfg.data_root}/flickr30k/captions.txt"
OUT_DIR       = "./precomputed_latents"
BATCH_SIZE    = 128                  # tune for your VRAM
NUM_WORKERS   = 8                    # tune for your CPU
MAX_LEN       = 77
os.makedirs(OUT_DIR, exist_ok=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

# ---------- Dataset that yields (index, image, caption) ----------
class Flickr30kSimple(Dataset):
    def __init__(self, images_root, captions_file, transform):
        self.images_root = images_root
        self.transform = transform
        self.captions = {}  # filename -> [captions]
        with open(captions_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2: 
                    continue
                img_id, cap = parts
                filename = img_id.split("#")[0].strip().strip('"').strip(',')
                self.captions.setdefault(filename, []).append(cap)
        self.filenames = sorted(self.captions.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fn = self.filenames[idx]
        path = os.path.join(self.images_root, fn)
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        # choose one caption at random (data-parallel friendly)
        cap_list = self.captions[fn]
        # local randomness is fine; PyTorch worker seeds differ
        caption = cap_list[torch.randint(len(cap_list), (1,)).item()]
        return idx, img, caption

tf = T.Compose([
    T.CenterCrop(224),
    T.Resize(64),
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3),
])

ds = Flickr30kSimple(IMAGES_ROOT, CAPTIONS_FILE, transform=tf)
dl = DataLoader(
    ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
    pin_memory=True, persistent_workers=True, prefetch_factor=4,
)

print(f"Device: {device}  |  Samples: {len(ds)}")

# ---------- Models on GPU, eval/frozen ----------
torch.backends.cudnn.benchmark = True
aekl = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
aekl.eval();  [setattr(p, "requires_grad", False) for p in aekl.parameters()]

langvae = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128").to(device)
langvae.eval(); [setattr(p, "requires_grad", False) for p in langvae.parameters()]

# Small sanity: all params really on CUDA
_ = next(aekl.parameters()).device, next(langvae.parameters()).device

t_load=t_tok=t_gpu=0.0
with torch.inference_mode():
    it = iter(dl)
    for _ in tqdm(range(min(20, len(dl))), desc="Profiling"):
        t0=time.time()
        imgs, caps = next(it)                  # dataloader fetch (I/O + decode + tfm)
        t1=time.time()

        tok = lang.decoder.tokenizer(list(caps), padding="max_length", truncation=True,
                                     max_length=MAX_LEN, return_tensors="pt")
        token_ids = tok["input_ids"].to(device, non_blocking=True)
        t2=time.time()

        imgs = imgs.to(device, non_blocking=True)
        post = aekl.encode(imgs).latent_dist
        _ = post.mean * aekl.config.scaling_factor
        _ = lang.encode_z(token_ids)[0]
        torch.cuda.synchronize()
        t3=time.time()

        t_load += (t1-t0); t_tok += (t2-t1); t_gpu += (t3-t2)

print(f"avg per-batch  load/transform: {t_load/20:.2f}s  tokenize: {t_tok/20:.2f}s  GPU encode: {t_gpu/20:.2f}s")


# # Optional: write an index file
# with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
#     json.dump({"count": len(ds)}, f)
# print(f"✅ Done. Latents in {OUT_DIR}")
