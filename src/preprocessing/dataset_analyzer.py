import os
from pathlib import Path
from collections import Counter
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


class DatasetAnalyzer:
    def __init__(self, root_dir: str, train_dir: str = "train", valid_dir: str = "valid"):
        self.root_dir = Path(root_dir)
        self.train_path = self.root_dir / train_dir
        self.valid_path = self.root_dir / valid_dir

        if not self.train_path.exists():
            raise FileNotFoundError(f"Train directory doenst exist: {self.train_path}")

    def get_class_distribution(self) -> pd.DataFrame:
        records = []
        for split_name, split_path in [("train", self.train_path), ("valid", self.valid_path)]:
            if not split_path.exists():
                continue
            for class_dir in sorted(split_path.iterdir()):
                if class_dir.is_dir():
                    n_images = len([
                        f for f in class_dir.iterdir()
                        if f.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp")
                    ])
                    parts = class_dir.name.replace("___", "|").replace("__", "|").split("|")
                    plant = parts[0].replace("_", " ").strip() if len(parts) > 0 else class_dir.name
                    disease = parts[1].replace("_", " ").strip() if len(parts) > 1 else "unknown"

                    records.append({
                        "split": split_name,
                        "class_name": class_dir.name,
                        "plant": plant,
                        "disease": disease,
                        "is_healthy": "healthy" in class_dir.name.lower(),
                        "num_images": n_images,
                    })

        df = pd.DataFrame(records)
        return df

    def compute_dataset_stats(self, batch_size: int = 64, num_workers: int = 4) -> Dict[str, List[float]]:
        print("Calculating mean/std of dataset")

        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),  # [0, 1] range
        ])

        dataset = datasets.ImageFolder(str(self.train_path), transform=transform)
        loader = DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=False)

        channels_sum = torch.zeros(3)
        channels_sq_sum = torch.zeros(3)
        num_pixels = 0

        for images, _ in tqdm(loader, desc="Computing stats"):
            b, c, h, w = images.shape
            channels_sum += images.sum(dim=[0, 2, 3])
            channels_sq_sum += (images ** 2).sum(dim=[0, 2, 3])
            num_pixels += b * h * w

        mean = (channels_sum / num_pixels).tolist()
        std = (torch.sqrt(channels_sq_sum / num_pixels - torch.tensor(mean) ** 2)).tolist()

        print(f"Dataset Mean: [{mean[0]:.4f}, {mean[1]:.4f}, {mean[2]:.4f}]")
        print(f"Dataset Std:  [{std[0]:.4f}, {std[1]:.4f}, {std[2]:.4f}]")

        return {"mean": mean, "std": std}

    def print_summary(self) -> None:
        df = self.get_class_distribution()

        print("NEW PLANT DISEASES DATASET - ANALYSIS")
        
        for split in df["split"].unique():
            split_df = df[df["split"] == split]
            total = split_df["num_images"].sum()
            n_classes = len(split_df)
            print(f"\n{split.upper()} SET:")
            print(f"Classes: {n_classes}")
            print(f"Total images: {total:,}")
            print(f"Min per class: {split_df['num_images'].min():,}")
            print(f"Max per class: {split_df['num_images'].max():,}")
            print(f"Mean per class: {split_df['num_images'].mean():.0f}")
            print(f"Std per class: {split_df['num_images'].std():.0f}")

        train_df = df[df["split"] == "train"]
        print(f"PLANT SPECIES ({train_df['plant'].nunique()}):")
        for plant in sorted(train_df["plant"].unique()):
            plant_df = train_df[train_df["plant"] == plant]
            n_diseases = len(plant_df[~plant_df["is_healthy"]])
            total_imgs = plant_df["num_images"].sum()
            print(f"   {plant:<35} | {len(plant_df)} classes | {n_diseases} diseases | {total_imgs:,} images")

        ratio = train_df["num_images"].max() / train_df["num_images"].min()
        print(f"\nImbalance ratio (max/min): {ratio:.2f}x")
        if ratio > 2.0:
            print("Dataset is highly imbalanced")
        else:
            print("Relatively balanced dataset")

        print("=" * 70)

    def plot_distribution(self, save_path: str = None) -> None:
        df = self.get_class_distribution()
        train_df = df[df["split"] == "train"].sort_values("num_images", ascending=True)

        fig, axes = plt.subplots(1, 2, figsize=(20, 12))

        colors = ["#2ecc71" if h else "#e74c3c" for h in train_df["is_healthy"]]
        axes[0].barh(train_df["class_name"], train_df["num_images"], color=colors)
        axes[0].set_xlabel("Number of images")
        axes[0].set_title("Class distribution (Train set)\nHealthy  Diseased")
        axes[0].tick_params(axis="y", labelsize=7)

        plant_counts = train_df.groupby("plant")["num_images"].sum().sort_values(ascending=False)
        axes[1].pie(
            plant_counts.values,
            labels=plant_counts.index,
            autopct="%1.1f%%",
            textprops={"fontsize": 8},
        )
        axes[1].set_title("Distribution by plant species (Train set)")

        plt.tight_layout()

        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Graph saved: {save_path}")
        plt.show()