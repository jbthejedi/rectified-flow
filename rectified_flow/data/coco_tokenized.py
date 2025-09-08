# rectified_flow/data/coco_tokenized.py
import os, json, random
from pathlib import Path
from typing import Optional, Dict, List
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from transformers import AutoTokenizer

# one tokenizer per process (worker); key by PID
_TOKENIZER_CACHE: Dict[int, AutoTokenizer] = {}

def _get_worker_tokenizer(name_or_path: str):
  pid = os.getpid()
  tok = _TOKENIZER_CACHE.get(pid)
  if tok is None:
    tok = AutoTokenizer.from_pretrained(name_or_path, use_fast=True)
    _TOKENIZER_CACHE[pid] = tok
  return tok


class CocoTokenized(Dataset):
  """
    images_root: path to COCO images (e.g., /data/coco/train2017 or /data/coco/val2017)
    captions_file: path to COCO captions json (e.g., .../annotations/captions_train2017.json)
    """
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

    # Load the COCO captions JSON (may be minified on one line; json.load handles it)
    data = json.loads(Path(captions_file).read_text(encoding="utf-8"))

    if not (isinstance(data, dict) and "images" in data and "annotations" in data):
      raise ValueError(
          f"{captions_file} doesn't look like a COCO captions json. "
          "Expected keys: 'images' and 'annotations'."
      )

    # Map image_id -> file_name
    id2file = {img["id"]: img["file_name"] for img in data["images"]}

    # Build filename -> [captions] (COCO provides ~5 per image)
    self.captions: Dict[str, List[str]] = {}
    missing = 0
    for ann in data["annotations"]:
      img_id = ann.get("image_id")
      cap = (ann.get("caption") or "").strip()
      if not cap:
        continue
      fname = id2file.get(img_id)
      if not fname:
        continue
      path = os.path.join(self.images_root, fname)
      if not os.path.exists(path):
        missing += 1
        continue
      self.captions.setdefault(fname, []).append(cap)

    # Keep only filenames that actually exist and have at least one caption
    self.filenames = sorted(self.captions.keys())
    if len(self.filenames) == 0:
      raise RuntimeError(
          f"No captioned images found under {self.images_root} using {captions_file}."
      )
    if missing:
      # Optional: surface how many annotations pointed to missing image files
      print(f"[CocoTokenized] Skipped {missing} annotations with missing image files.")

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
    input_ids = enc["input_ids"].squeeze(0)       # (L,)
    attention_mask = enc["attention_mask"].squeeze(0)  # (L,)

    # return img, input_ids, attention_mask, caption
    return img, input_ids, attention_mask
