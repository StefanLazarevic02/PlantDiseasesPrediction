import yaml
import torch
import json
import random
import numpy as np
import os
import datetime
from src.preprocessing.dataset_analyzer import DatasetAnalyzer
from src.augmentation.policies import get_train_transforms, get_valid_transforms
from src.preprocessing.dataloader import create_dataloaders
from src.models.classifier import create_model
from src.training.trainer import Trainer

SEED = 42
MODELS = ["resnet18", "resnet50", "efficientnet_b0", "vgg16"]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_single_model(model_name, config, loaders, device):
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*60}")
    
    set_seed(SEED)
    
    # Creating model
    num_classes = config["dataset"].get("num_classes", 38)
    model = create_model(num_classes=num_classes, model_name=model_name, pretrained=True)
    
    save_dir = f"outputs/{model_name}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=loaders["train"],
        valid_loader=loaders["valid"],
        config=config,
        device=device
    )
    
    history = trainer.train(save_dir=save_dir)
    
    with open(f"{save_dir}/training_history.json", "w") as f:
        json.dump(history, f, indent=2)
    
    return {
        "model_name": model_name,
        "best_accuracy": max(history["valid_acc"]),
        "final_train_acc": history["train_acc"][-1],
        "final_valid_acc": history["valid_acc"][-1],
        "history": history
    }

def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    print("=" * 60)
    print("PLANT DISEASE CLASSIFICATION - TRAINING")
    print(f"Models: {', '.join(MODELS)}")
    print(f"SEED: {SEED}")
    print("=" * 60)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    
    set_seed(SEED)
    
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
    
    all_results = []
    for model_name in MODELS:
        result = train_single_model(model_name, config, loaders, device)
        all_results.append(result)
    
    print(f"\n{'='*60}")
    print("SUMMARY - ALL MODELS")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'Best Valid Acc':<15} {'Final Valid Acc':<15}")
    print("-" * 50)
    
    for result in all_results:
        print(f"{result['model_name']:<20} {result['best_accuracy']:.2f}%{'':<10} {result['final_valid_acc']:.2f}%")
    
    summary = {
        "timestamp": datetime.now().isoformat(),
        "seed": SEED,
        "models": MODELS,
        "results": [
            {
                "model_name": r["model_name"],
                "best_accuracy": r["best_accuracy"],
                "final_train_acc": r["final_train_acc"],
                "final_valid_acc": r["final_valid_acc"]
            }
            for r in all_results
        ]
    }
    
    with open("outputs/comparison_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nAll models trained and saved!")
    
    return all_results

history = main()