
import os
from torchvision import datasets, transforms as T
from datasets import load_dataset
from torch.utils.data import Dataset
from PIL import Image

class ProjectData:
  def __init__(self, config, device):
    if config.dataset_type == 'cifar10':
      print("Using Cifar10")
      self.dataset = datasets.CIFAR10(
        root=config.data_root,
        download=config.download_data,
        transform=T.Compose([
            T.Resize((config.image_size, config.image_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5]),
        ]))

    elif config.dataset_type == 'celeba':
      print("Using Celeba")
      transform = T.Compose([
          T.CenterCrop(178),
          T.Resize(config.image_size),
          T.ToTensor(),
          T.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
      ])
      self.dataset = CelebADataset(root_dir=os.path.join(config.data_root, "img_align_celeba"),
                                   transform=transform)


class CelebADataset(Dataset):

  def __init__(self, root_dir, transform=None):
    self.root_dir = root_dir
    self.img_names = sorted([f for f in os.listdir(root_dir) if f.endswith(".jpg")])
    self.transform = transform

  def __len__(self):
    return len(self.img_names)

  def __getitem__(self, idx):
    img_path = os.path.join(self.root_dir, self.img_names[idx])
    image = Image.open(img_path).convert("RGB")
    if self.transform:
      image = self.transform(image)
    return image, 0