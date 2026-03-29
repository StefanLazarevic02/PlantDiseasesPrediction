import torch
import yaml
import cv2
import os
import json
from pathlib import Path
import albumentations as A
from albumentations.pytorch import ToTensorV2
from src.models.classifier import create_model
import matplotlib.pyplot as plt


MODELS = ["resnet18", "resnet50", "efficientnet_b0", "vgg16"]


def load_model(model_name, config, device):
    model_path = f"outputs/{model_name}/best_model.pth"
    
    if not os.path.exists(model_path):
        print(f"Model did not exist: {model_path}")
        return None, None
    
    checkpoint = torch.load(model_path, map_location=device)
    print(f"{model_name} loaded (validation acc: {checkpoint['best_acc']:.2f}%)")
    
    num_classes = config["dataset"].get("num_classes", 38)
    model = create_model(num_classes=num_classes, model_name=model_name, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint['best_acc']


def predict_image(model, image_tensor, class_names):
    with torch.no_grad():
        output = model(image_tensor)
        probs = torch.softmax(output, dim=1)
        confidence, pred_idx = probs.max(1)
    
    return class_names[pred_idx.item()], confidence.item() * 100


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    test_transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(
            mean=config["dataset"].get("mean", [0.485, 0.456, 0.406]),
            std=config["dataset"].get("std", [0.229, 0.224, 0.225])
        ),
        ToTensorV2(),
    ])
    
    # Class names
    train_path = os.path.join(config["dataset"]["root_dir"], "train")
    class_names = sorted([d for d in os.listdir(train_path) if os.path.isdir(os.path.join(train_path, d))])
    print(f"Number of classes: {len(class_names)}")
    
    # Test image
    test_path = os.path.join(config["dataset"]["root_dir"], "test")
    test_images = [f for f in os.listdir(test_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    print(f"Test image: {len(test_images)}")
    
    models = {}
    val_accuracies = {}
    for model_name in MODELS:
        model, val_acc = load_model(model_name, config, device)
        if model is not None:
            models[model_name] = model
            val_accuracies[model_name] = val_acc
    
    if not models:
        print("Models not loaded!")
        return
    
    print(f"\n{'='*80}")
    print("PREDIKCIJE SVIH MODELA")
    print(f"{'='*80}\n")
    
    all_results = []
    
    for img_name in test_images:
        img_path = os.path.join(test_path, img_name)
        
        # Load picture
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        augmented = test_transform(image=image)
        input_tensor = augmented["image"].unsqueeze(0).to(device)
        
        predictions = {}
        print(f"{img_name}")
        
        for model_name, model in models.items():
            pred_class, conf = predict_image(model, input_tensor, class_names)
            predictions[model_name] = {"class": pred_class, "confidence": conf}
            print(f"   {model_name:<18} → {pred_class} ({conf:.1f}%)")
        
        unique_predictions = set(p["class"] for p in predictions.values())
        if len(unique_predictions) == 1:
            print(f"All models agree on class!")
        else:
            print(f"Models disagree!")
        
        print()
        
        all_results.append({
            "image": img_name,
            "predictions": predictions,
            "agreement": len(unique_predictions) == 1
        })
    
    print(f"{'='*80}")
    print("STATISTICS")
    print(f"{'='*80}")
    
    agreements = sum(1 for r in all_results if r["agreement"])
    print(f"\nModel agreement rate: {agreements}/{len(all_results)} ({100*agreements/len(all_results):.1f}%)")
    
    print(f"\nValidation accuracy:")
    for model_name, acc in val_accuracies.items():
        print(f"   {model_name:<18}: {acc:.2f}%")
    
    # Sačuvaj rezultate
    with open("outputs/test_all_models_results.json", "w") as f:
        json.dump({
            "models": list(models.keys()),
            "val_accuracies": val_accuracies,
            "results": all_results,
            "agreement_rate": agreements / len(all_results)
        }, f, indent=2)
    
    print(f"\nResults saved: outputs/test_all_models_results.json")
    
    return all_results


if __name__ == "__main__":
    results = main()