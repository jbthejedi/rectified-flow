You’re right—those two lines make train and val identical. Do a deterministic split of the latent files and give each dataset its own file list.

### Option A (quick, in `train.py`)

```python
import glob, os, torch
latents_dir = "./precomputed_latents"

all_files = sorted(glob.glob(os.path.join(latents_dir, "sample_*.pt")))
n = len(all_files)
g = torch.Generator().manual_seed(config.seed)
perm = torch.randperm(n, generator=g).tolist()

n_train = int(config.p_train_len * n)   # e.g., 0.9 * n from your config
train_files = [all_files[i] for i in perm[:n_train]]
val_files   = [all_files[i] for i in perm[n_train:]]

train_ds = PrecomputedLatents(latents_dir, files=train_files)
val_ds   = PrecomputedLatents(latents_dir, files=val_files)
```

And tweak your dataset to accept a file subset:

```python
class PrecomputedLatents(Dataset):
    def __init__(self, latents_dir: str, files=None):
        self.latents_dir = latents_dir
        if files is None:
            files = sorted(glob.glob(os.path.join(latents_dir, "sample_*.pt")))
        if not files:
            raise FileNotFoundError(f"No sample_*.pt in {latents_dir}")
        self.files = files
    # ... __len__/__getitem__ unchanged ...
```

### Option B (cleaner, persist the split once)

At the end of `precompute.py`, save a split file so every run uses the same split:

```python
# After writing meta.json
import random, json
files = sorted(os.listdir(OUT_DIR))
files = [f for f in files if f.startswith("sample_") and f.endswith(".pt")]
random.Random(1337).shuffle(files)               # fixed seed
cut = int(0.9 * len(files))
splits = {"train": files[:cut], "val": files[cut:]}
with open(os.path.join(OUT_DIR, "splits.json"), "w") as f:
    json.dump(splits, f)
```

Then in `train.py`:

```python
with open(os.path.join(latents_dir, "splits.json")) as f:
    sp = json.load(f)
train_files = [os.path.join(latents_dir, f) for f in sp["train"]]
val_files   = [os.path.join(latents_dir, f) for f in sp["val"]]

train_ds = PrecomputedLatents(latents_dir, files=train_files)
val_ds   = PrecomputedLatents(latents_dir, files=val_files)
```

Either way, you’ll have non-overlapping train/val over the same latent directory.
