import os
from pathlib import Path
from typing import Tuple, Dict, Optional

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import datasets
import albumentations as A


class PlantDiseaseDataset(Dataset):
    def __init__(self, root_dir: str, transform: A.Compose = None):
        self.root_dir = Path(root_dir)
        self.transform = transform

        # Koristi ImageFolder logiku za mapiranje klasa
        self.classes = sorted([
            d.name for d in self.root_dir.iterdir() if d.is_dir()
        ])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Sakupi sve slike
        self.samples = []
        valid_ext = {".jpg", ".jpeg", ".png", ".bmp", ".JPG"}
        for cls_name in self.classes:
            cls_dir = self.root_dir / cls_name
            cls_idx = self.class_to_idx[cls_name]
            for img_path in cls_dir.iterdir():
                if img_path.suffix in valid_ext:
                    self.samples.append((str(img_path), cls_idx))

        print(f"Loaded {len(self.samples)} pictures from {len(self.classes)} classes ({root_dir})")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_path, label = self.samples[idx]

        # Ucitavanje slike sa OpenCV
        image = cv2.imread(img_path)
        if image is None:
            raise ValueError(f"Can't load image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label

    def get_labels(self) -> list:
        return [label for _, label in self.samples]

    def get_class_weights(self) -> torch.Tensor:
        labels = self.get_labels()
        class_counts = np.bincount(labels, minlength=len(self.classes))
        total = len(labels)
        weights = total / (len(self.classes) * class_counts + 1e-6)
        return torch.FloatTensor(weights)


def create_weighted_sampler(dataset: PlantDiseaseDataset) -> WeightedRandomSampler:
    labels = dataset.get_labels()
    class_counts = np.bincount(labels, minlength=len(dataset.classes))
    sample_weights = 1.0 / (class_counts[labels] + 1e-6)
    sample_weights = torch.DoubleTensor(sample_weights)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )   # Ublazava efekat imbalance tako sto ce vise puta uzeti uzorke iz manjih klasa


def create_dataloaders(
    config: dict,
    train_transform: A.Compose,
    valid_transform: A.Compose,
    use_weighted_sampling: bool = True,
) -> Dict[str, DataLoader]:

    ds_cfg = config.get("dataset", {})
    dl_cfg = config.get("dataloader", {})

    root_dir = ds_cfg["root_dir"]
    train_dir = os.path.join(root_dir, ds_cfg.get("train_dir", "train"))
    valid_dir = os.path.join(root_dir, ds_cfg.get("valid_dir", "valid"))
    test_dir = os.path.join(root_dir, ds_cfg.get("test_dir", "test"))

    batch_size = dl_cfg.get("batch_size", 64)
    num_workers = dl_cfg.get("num_workers", 2)
    pin_memory = dl_cfg.get("pin_memory", True)
    prefetch_factor = dl_cfg.get("prefetch_factor", 2)
    persistent_workers = dl_cfg.get("persistent_workers", True) if num_workers > 0 else False
    drop_last = dl_cfg.get("drop_last", True)

    train_dataset = PlantDiseaseDataset(train_dir, transform=train_transform)
    valid_dataset = PlantDiseaseDataset(valid_dir, transform=valid_transform)

    # Sampler za balansiranje
    train_sampler = None
    train_shuffle = True
    if use_weighted_sampling:
        train_sampler = create_weighted_sampler(train_dataset)
        train_shuffle = False

    loaders = {
        "train": DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
            drop_last=drop_last,
        ),
        "valid": DataLoader(
            valid_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None,
            persistent_workers=persistent_workers,
        ),
    }

    if os.path.exists(test_dir) and len(os.listdir(test_dir)) > 0:
        test_dataset = PlantDiseaseDataset(test_dir, transform=valid_transform)
        loaders["test"] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )

    return loaders