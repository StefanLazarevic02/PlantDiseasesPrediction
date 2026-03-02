import yaml
from src.preprocessing.dataset_analyzer import DatasetAnalyzer
from src.augmentation.policies import get_train_transforms, get_valid_transforms
from src.preprocessing.dataloader import create_dataloaders
from src.utils.visualization import show_augmentations, show_batch


def main():
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)

    print("PLANT DISEASE DETECTION PREPROCESSING & AUGMENTATION")

    ds_cfg = config["dataset"]
    analyzer = DatasetAnalyzer(
        root_dir=ds_cfg["root_dir"],
        train_dir=ds_cfg.get("train_dir", "train"),
        valid_dir=ds_cfg.get("valid_dir", "valid"),
    )

    # Statistike
    analyzer.print_summary()

    # Vizualizacija distribucije klasa
    analyzer.plot_distribution(save_path="outputs/class_distribution.png")

    # mean/std
    if config["preprocessing"].get("use_dataset_stats", False):
        stats = analyzer.compute_dataset_stats()
        config["dataset"]["mean"] = stats["mean"]
        config["dataset"]["std"] = stats["std"]
        print(f"Using dataset-specific mean/std")
    else:
        print(f"Using ImageNet mean/std")

    policy = config["augmentation"].get("policy", "medium")
    print(f"Augmentation policy: {policy.upper()}")

    train_transform = get_train_transforms(config, policy=policy)
    valid_transform = get_valid_transforms(config)

    print(f"Train transforms: {len(train_transform.transforms)} steps")
    print(f"Valid transforms: {len(valid_transform.transforms)} steps")

    # Vizualizacija augmentacije
    df = analyzer.get_class_distribution()
    train_df = df[df["split"] == "train"]
    sample_class = train_df.iloc[0]["class_name"]
    sample_dir = analyzer.train_path / sample_class
    sample_image = next(sample_dir.iterdir())

    print(f"\nDemo augmentation on: {sample_image.name} ({sample_class})")
    show_augmentations(str(sample_image), train_transform, n_examples=7)

    print("\nCreating DataLoaders")
    loaders = create_dataloaders(
        config,
        train_transform=train_transform,
        valid_transform=valid_transform,
        use_weighted_sampling=True,
    )

    for name, loader in loaders.items():
        print(f"{name}: {len(loader)} batchs * {loader.batch_size} = ~{len(loader) * loader.batch_size:,} pictures")

    class_names = loaders["train"].dataset.classes
    show_batch(loaders["train"], class_names, n_images=16)

    print("\nPreprocessing and augmentation ready! Next step: training.")

    return config, loaders, class_names


if __name__ == "__main__":
    config, loaders, class_names = main()