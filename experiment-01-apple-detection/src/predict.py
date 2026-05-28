#!/usr/bin/env python3
"""
Apple Quality Classification - Inference Script
Classify test images using trained model.
"""

import argparse
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# Class name mapping (Chinese -> English)
CLASS_LABELS = {
    "fresh": "合格 (Fresh)",
    "diseased": "病变 (Diseased)",
    "bruised": "碰伤 (Bruised)",
    "rotten": "腐烂 (Rotten)",
    "insect_damaged": "虫伤 (Insect Damaged)",
    "cracked": "裂果 (Cracked)",
    "wrinkled": "褶皱 (Wrinkled)",
    "black_spot": "黑斑 (Black Spot)",
}


def get_model(num_classes=8, model_path=None):
    """Load model with trained weights."""
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', list(CLASS_LABELS.keys()))
    else:
        print("Warning: No trained weights found. Using random initialization.")
        class_names = list(CLASS_LABELS.keys())

    return model, class_names


def predict_image(model, image_path, transform, class_names, device):
    """Predict single image."""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted = probabilities.max(1)

    pred_class = class_names[predicted.item()]
    pred_label = CLASS_LABELS.get(pred_class, pred_class)
    conf_value = confidence.item()

    # Get top-3 predictions
    top3_conf, top3_idx = probabilities.topk(3, dim=1)
    top3 = [
        (CLASS_LABELS.get(class_names[idx], class_names[idx]), conf)
        for idx, conf in zip(top3_idx[0].tolist(), top3_conf[0].tolist())
    ]

    return pred_label, conf_value, top3


def main():
    parser = argparse.ArgumentParser(description='Predict apple quality')
    parser.add_argument('image', type=str, help='Path to image or directory')
    parser.add_argument('--model', type=str, default='../models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default=None,
                        help='Output file for batch predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load model
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    model, class_names = get_model(num_classes=8, model_path=str(model_path))
    model = model.to(device)
    model.eval()

    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    input_path = Path(args.image)

    if input_path.is_file():
        # Single image
        print(f"\nImage: {input_path.name}")
        print("-" * 40)

        pred_label, confidence, top3 = predict_image(
            model, input_path, transform, class_names, device
        )

        print(f"Prediction: {pred_label}")
        print(f"Confidence: {confidence*100:.2f}%")
        print("\nTop-3:")
        for i, (label, conf) in enumerate(top3, 1):
            print(f"  {i}. {label}: {conf*100:.2f}%")
        print()

    elif input_path.is_dir():
        # Batch prediction
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_path.iterdir()
                      if f.suffix.lower() in image_extensions]

        print(f"Found {len(image_files)} images\n")

        results = []
        for img_path in sorted(image_files):
            pred_label, confidence, _ = predict_image(
                model, img_path, transform, class_names, device
            )
            results.append({
                'file': img_path.name,
                'prediction': pred_label,
                'confidence': f"{confidence*100:.2f}%"
            })
            print(f"{img_path.name:40s} -> {pred_label} ({confidence*100:.2f}%)")

        # Save results
        if args.output:
            output_path = Path(args.output)
            import json
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved to: {output_path}")

    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == '__main__':
    main()
