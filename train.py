import yaml
import torch
import json
import random
import numpy as np
from src.preprocessing.dataset_analyzer import DatasetAnalyzer
from src.augmentation.policies import get_train_transforms, get_valid_transforms
from src.preprocessing.dataloader import create_dataloaders
from src.models.classifier import create_model
from src.training.trainer import Trainer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("PLANT DISEASE CLASSIFICATION - TRAINING")
    print(f"SEED: {SEED}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    # Data analysis
    ds_cfg = config["dataset"]
    analyzer = DatasetAnalyzer(
        root_dir=ds_cfg["root_dir"],
        train_dir=ds_cfg.get("train_dir", "train"),
        valid_dir=ds_cfg.get("valid_dir", "valid"),
    )
    
    # Preprocessing
    if config["preprocessing"].get("use_dataset_stats", False):
        stats = analyzer.compute_dataset_stats()
        config["dataset"]["mean"] = stats["mean"]
        config["dataset"]["std"] = stats["std"]
    
    # Augmentation
    policy = config["augmentation"].get("policy", "medium")
    train_transform = get_train_transforms(config, policy=policy)
    valid_transform = get_valid_transforms(config)
    
    # Dataloaders
    print("\nCreating dataloaders...")
    loaders = create_dataloaders(
        config,
        train_transform=train_transform,
        valid_transform=valid_transform,
        use_weighted_sampling=True,
    )
    
    print("\nCreating model...")
    num_classes = ds_cfg.get("num_classes", 38)
    model_name = config.get("training", {}).get("model_name", "efficientnet_b0")
    model = create_model(num_classes=num_classes, model_name=model_name, pretrained=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        config=config,
        device=device
    )
    
    history = trainer.train(save_dir="outputs")
    
    with open("outputs/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    print("\nTraining completed and saved!")
    return history

history = main()