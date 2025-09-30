import random
import os
from torchvision import datasets, transforms as T
from datasets import load_dataset
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from pathlib import Path
from transformers import BertTokenizer
import torch
from PIL import Image

from langvae.data_conversion.tokenization import TokenizedDataSet


class ProjectData:
    def __init__(self, config, collator=None):
        if config.dataset_type == 'cifar10':
            datasets.CIFAR10(
                root=config.data_root,
                download=False,
                transform=T.Compose([
                    T.Resize((config.image_size, config.image_size)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5], std=[0.5]),
                ])
            )
            if config.do_small_sample is not False:
                dataset = self.__sample_dataset(dataset)
            train_len = int(len(dataset) * config.p_train_len)
            self.train_set, self.val_set = random_split(dataset, [train_len, len(dataset) - train_len])

        elif config.dataset_type == 'celeba':
            self.data = datasets.CelebA(
                root=config.data_root,
                download=False,
                transform=T.Compose([
                    T.CenterCrop(178),
                    T.Resize((128, 128)),
                    T.ToTensor(),
                    T.Normalize(mean=[0.5]*3, std=[0.5]*3),
                ])
            )
            if config.do_small_sample is not False:
                dataset = self.__sample_dataset(dataset)
            train_len = int(len(dataset) * config.p_train_len)
            self.train_set, self.val_set = random_split(dataset, [train_len, len(dataset) - train_len])

        elif config.dataset_type == 'mscoco_mini':
            dataset = load_dataset("yerevann/coco-karpathy")
            ds = dataset
            print(ds)
            print(ds["train"].column_names)         # see actual names
            print(ds["train"].features)             # see dtypes (Image vs string path)
            print(ds["train"][0])                   # peek one row
            exit()
            def transform_batch(batch):
                image_transform = T.Compose([
                    T.Resize((64, 64)), # Resized from 128 -> 64 for toy example.
                    T.ToTensor(),
                    T.Normalize(mean=[0.5]*3, std=[0.5]*3)
                ])
                batch["pixel_values"] = [image_transform(img.convert("RGB")) for img in batch["image"]]
                return batch

            dataset.set_transform(transform_batch)
            self.train_set = dataset['train']
            self.val_set = dataset['validation']
        
        elif config.dataset_type == 'flickr30k':
                base_ds = Flickr30kDataset(
                    images_root=f"{config.data_root}/flickr30k/Images",
                    captions_file=f"{config.data_root}/flickr30k/captions.txt",
                    transform=None
                )
                if config.do_small_sample:
                    idxs = random.sample(range(len(base_ds)), k=config.sample_size_n)
                    base_ds = Subset(base_ds, idxs)
                n = len(base_ds)
                perm = torch.randperm(n).tolist()
                n_train = int(config.p_train_len * n)

                train_inds = perm[:n_train]
                val_inds   = perm[n_train:]

                mean = [0.444, 0.421, 0.384]
                std = [0.275, 0.267, 0.276]
                train_tf = T.Compose([
                    T.CenterCrop(224),
                    T.Resize(config.image_size),
                    T.ToTensor(),
                    T.Normalize(mean, std),
                ])
                val_tf = T.Compose([ T.CenterCrop(224), T.Resize(config.image_size), T.ToTensor(), T.Normalize(mean, std),
                ])

                train_ds = Subset(
                    Flickr30kDataset(
                        f"{config.data_root}/flickr30k/Images",
                        f"{config.data_root}/flickr30k/captions.txt",
                        transform=train_tf
                    ),
                    train_inds
                )
                val_ds = Subset(
                    Flickr30kDataset(
                        f"{config.data_root}/flickr30k/Images",
                        f"{config.data_root}/flickr30k/captions.txt",
                        transform=val_tf
                    ),
                    val_inds
                )
                if collator is not None:
                    self.train_dl = DataLoader(
                        train_ds, batch_size=config.batch_size, shuffle=True,
                        num_workers=config.num_workers,
                        collate_fn=collator,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4
                    )
                    self.val_dl = DataLoader(
                        val_ds, batch_size=config.batch_size, shuffle=False,
                        num_workers=config.num_workers,
                        collate_fn=collator,
                        pin_memory=True,
                        persistent_workers=True,
                        prefetch_factor=4
                    )
                else:
                    raise Exception("No collator")

        
class Flickr30kDataset(Dataset):
    def __init__(self, images_root, captions_file, transform=None):
        self.images_root = images_root
        self.transform = transform

        self.captions = {}
        with Path(captions_file).open('r') as f:
            for line in f:
                line = line.strip()
                if line is None or line.startswith('#'):
                    continue
                parts = line.split(None, 1)
                if len(parts) < 2:
                    continue
                img_id, caption = parts
                filename = img_id.split("#")[0].strip().strip('"').strip(',')
                self.captions.setdefault(filename, []).append(caption)
        self.filenames = sorted(self.captions.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        path = os.path.join(self.images_root, filename)
        img = Image.open(path).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        caption = random.choice(self.captions[filename])
        return img, caption


class BatchCollator:
    def __init__(self, max_length=77, device='cpu'):
        self.tokenizer : BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = max_length
        self.device = device

    def __call__(self, batch):
        images, captions = zip(*batch)
        images = torch.stack(images, dim=0)
        tokenized = self.tokenizer(
            list(captions),
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt',
        )
        attention_mask = tokenized.attention_mask
        input_ids = tokenized.input_ids
        return images, input_ids, attention_mask


class LangVAECollator:
    def __init__(self, tokenizer, max_length=32):
        self.tokenizer = tokenizer # use LangVAE’s tokenizer
        self.max_length = max_length

    def __call__(self, batch):
        images, captions = zip(*batch)       # unzip batch
        images = torch.stack(images, dim=0)  # (B,3,H,W)

        # Use LangVAE’s tokenizer
        tokenized = self.tokenizer(
            list(captions),
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Move tensors to correct device
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        return images, input_ids, attention_mask
