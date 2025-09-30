# debug_precompute_timing.py
import os, time, torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from PIL import Image
from tqdm import tqdm
from diffusers import AutoencoderKL
from langvae import LangVAE

IMAGES_ROOT   = "/workspace/data/flickr30k/Images"
CAPTIONS_FILE = "/workspace/data/flickr30k/captions.txt"
BATCH_SIZE    = 64
NUM_WORKERS   = 8
MAX_LEN       = 77
device = "cuda" if torch.cuda.is_available() else "cpu"
torch.backends.cudnn.benchmark = True

class Flickr(Dataset):
    def __init__(self, root, captions, tfm):
        self.root, self.tfm = root, tfm
        caps = {}
        with open(captions) as f:
            for line in f:
                line=line.strip()
                if not line or line.startswith("#"): continue
                fid, cap = line.split(None, 1)
                fn = fid.split("#")[0].strip().strip('"').strip(',')
                caps.setdefault(fn, []).append(cap)
        self.fns = sorted(caps.keys()); self.caps = caps
    def __len__(self): return len(self.fns)
    def __getitem__(self, i):
        fn = self.fns[i]; p = os.path.join(self.root, fn)
        img = Image.open(p).convert("RGB")
        img = self.tfm(img)
        cap = self.caps[fn][torch.randint(len(self.caps[fn]), (1,)).item()]
        return img, cap

tf = T.Compose([T.CenterCrop(224), T.Resize(64), T.ToTensor(), T.Normalize([0.5]*3,[0.5]*3)])
ds = Flickr(IMAGES_ROOT, CAPTIONS_FILE, tf)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
                pin_memory=True, persistent_workers=True, prefetch_factor=4)

aekl = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device).eval()
for p in aekl.parameters(): p.requires_grad=False
lang = LangVAE.load_from_hf_hub("neuro-symbolic-ai/eb-langvae-bert-base-cased-gpt2-l128").to(device).eval()
for p in lang.parameters(): p.requires_grad=False

print(f"Device={device}  batches={len(dl)}")

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
