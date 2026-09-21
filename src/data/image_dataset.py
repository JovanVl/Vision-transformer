from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

from PIL import Image

class ImageDataset(Dataset):
    def __init__(self, root: str, mean=None, std=None, augment: bool = False, max_per_class: int = None):
        self.root = Path(root)
        self.classes = sorted(p.name for p in self.root.iterdir() if p.is_dir())
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        self.weights = []
        self.max_per_class = max_per_class
        self.num_per_class = []
        
        self.samples = []
        for class_name in self.classes:
            class_images = sorted((self.root / class_name).glob("*.jpg"))
            if max_per_class is not None:
                max_per_class = len(class_images) if len(class_images) < max_per_class else max_per_class
                class_images = class_images[:max_per_class]
                class_weight = 1.0 / max_per_class
                self.weights.append(max_per_class)
            else:
                class_weight = 1.0 / len(class_images)
                self.weights.append(class_weight)
            for img_path in class_images:
                self.samples.append((img_path, self.class_to_idx[class_name]))

        self.weights = torch.Tensor([self.weights[i] / sum(self.weights) * len(self.weights) for i in range(len(self.weights))])
        norm = transforms.Normalize(mean or [0.5, 0.5, 0.5], std or [0.5, 0.5, 0.5])

        if augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                norm,
            ])
        else:
            self.transform = transforms.Compose([transforms.ToTensor(), norm])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        return img, label