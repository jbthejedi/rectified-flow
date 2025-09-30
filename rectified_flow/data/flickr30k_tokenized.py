# rectified_flow/data/flickr30k_tokenized.py
import os, random
from pathlib import Path
from typing import Optional
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoTokenizer

# one tokenizer per process (worker); key by PID
_TOKENIZER_CACHE = {}

def _get_worker_tokenizer(name_or_path: str):
    pid = os.getpid()
    tok = _TOKENIZER_CACHE.get(pid)
    if tok is None:
        tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
        _TOKENIZER_CACHE[pid] = tok
    return tok

class Flickr30kTokenized(Dataset):
    def __init__(
        self,
        images_root: str,
        captions_file: str,
        tokenizer_name_or_path: str,
        transform: Optional[T.Compose] = None,
        max_length: int = 77,
    ):
        self.images_root = images_root
        self.transform = transform
        self.max_length = max_length
        self.tokenizer_name_or_path = tokenizer_name_or_path

        self.captions = {}
        with Path(captions_file).open("r") as f:
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
        if self.transform is not None:
            img = self.transform(img)

        caption = random.choice(self.captions[fn])

        # Lazily build tokenizer in *this* worker process
        tok = _get_worker_tokenizer(self.tokenizer_name_or_path)
        enc = tok(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids     = enc["input_ids"].squeeze(0)      # (L,)
        attention_mask= enc["attention_mask"].squeeze(0) # (L,)

        return img, input_ids, attention_mask
