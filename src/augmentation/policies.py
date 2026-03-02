import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config: dict, policy: str = None) -> A.Compose:
    aug_config = config.get("augmentation", {})
    if policy is None:
        policy = aug_config.get("policy", "medium")

    p_cfg = aug_config.get(policy, {})
    ds_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})

    input_size = prep_cfg.get("resize", 224)
    mean = ds_cfg.get("mean", [0.485, 0.456, 0.406])
    std = ds_cfg.get("std", [0.229, 0.224, 0.225])

    if policy == "light":
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=p_cfg.get("horizontal_flip_p", 0.5)),
            A.Rotate(limit=p_cfg.get("rotation_limit", 15), p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=p_cfg.get("brightness_limit", 0.1),
                contrast_limit=p_cfg.get("contrast_limit", 0.1),
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif policy == "medium":
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=p_cfg.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=p_cfg.get("vertical_flip_p", 0.1)),
            A.Rotate(limit=p_cfg.get("rotation_limit", 30), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=p_cfg.get("brightness_limit", 0.2),
                contrast_limit=p_cfg.get("contrast_limit", 0.2),
                p=0.4,
            ),
            A.GaussianBlur(blur_limit=p_cfg.get("blur_limit", 3), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
            A.HueSaturationValue(
                hue_shift_limit=p_cfg.get("hue_shift_limit", 10),
                sat_shift_limit=p_cfg.get("sat_shift_limit", 20),
                val_shift_limit=10,
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    elif policy == "strong":
        return A.Compose([
            A.Resize(input_size, input_size),
            A.HorizontalFlip(p=p_cfg.get("horizontal_flip_p", 0.5)),
            A.VerticalFlip(p=p_cfg.get("vertical_flip_p", 0.2)),
            A.ShiftScaleRotate(
                shift_limit=p_cfg.get("shift_limit", 0.1),
                scale_limit=p_cfg.get("scale_limit", 0.2),
                rotate_limit=p_cfg.get("rotation_limit", 45),
                border_mode=0,
                p=0.5,
            ),
            A.RandomBrightnessContrast(
                brightness_limit=p_cfg.get("brightness_limit", 0.3),
                contrast_limit=p_cfg.get("contrast_limit", 0.3),
                p=0.5,
            ),
            A.GaussianBlur(blur_limit=p_cfg.get("blur_limit", 5), p=0.2),
            A.GaussNoise(std_range=(0.01, 0.1), p=0.2),  # ISPRAVLJENO!
            A.HueSaturationValue(
                hue_shift_limit=p_cfg.get("hue_shift_limit", 15),
                sat_shift_limit=p_cfg.get("sat_shift_limit", 30),
                val_shift_limit=15,
                p=0.3,
            ),
            A.Normalize(mean=mean, std=std),
            ToTensorV2(),
        ])

    else:
        # Default: medium
        return get_train_transforms(config, policy="medium")


def get_valid_transforms(config: dict) -> A.Compose:
    ds_cfg = config.get("dataset", {})
    prep_cfg = config.get("preprocessing", {})

    input_size = prep_cfg.get("resize", 224)
    mean = ds_cfg.get("mean", [0.485, 0.456, 0.406])
    std = ds_cfg.get("std", [0.229, 0.224, 0.225])

    return A.Compose([
        A.Resize(input_size, input_size),
        A.Normalize(mean=mean, std=std),
        ToTensorV2(),
    ])
