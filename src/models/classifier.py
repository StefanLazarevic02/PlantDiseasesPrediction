import torch
import torch.nn as nn
import torchvision.models as models


def create_model(num_classes: int, model_name: str = "resnet50", pretrained: bool = True):
    """
    Kreira model za klasifikaciju bolesti biljaka
    """
    print(f"Creating model: {model_name} (pretrained={pretrained})")
    
    if model_name == "efficientnet_b0":
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = models.efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    # Prebroj parametre
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    return model