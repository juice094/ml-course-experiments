#!/usr/bin/env python3
"""
Ensemble Inference: Average predictions from K-Fold models
集成推理：加载 K-Fold 训练的 5 个模型，取 softmax 平均预测

用法：
    python ensemble_predict.py ../data/test --output ../outputs/ensemble_predictions.json
"""

import argparse
import json
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn
import numpy as np

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

NUM_CLASSES = 8


def get_model(model_path):
    """加载单个 fold 模型。"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )

    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    class_names = checkpoint.get('class_names', list(CLASS_LABELS.keys()))

    return model, class_names


def ensemble_predict(models, image_path, transform, class_names, device):
    """对单张图执行集成推理。"""
    image = Image.open(image_path).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device)

    all_probs = []
    with torch.no_grad():
        for model in models:
            model.eval()
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

    # 5 个模型概率取平均
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = avg_probs.argmax()
    pred_class = class_names[pred_idx]
    confidence = float(avg_probs[pred_idx])

    # 各模型的独立预测
    individual = []
    for i, probs in enumerate(all_probs):
        idx = probs.argmax()
        individual.append({
            'fold': i,
            'prediction': CLASS_LABELS.get(class_names[idx], class_names[idx]),
            'confidence': round(float(probs[idx]), 4),
        })

    return pred_class, confidence, avg_probs, individual


def main():
    parser = argparse.ArgumentParser(description='Ensemble inference with K-Fold models')
    parser.add_argument('image', type=str, help='Path to image or directory')
    parser.add_argument('--model-dir', type=str, default='../models',
                        help='Directory containing best_model_fold*.pth')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON for batch predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 加载所有 fold 模型
    # -------------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    fold_paths = sorted(model_dir.glob('best_model_fold*.pth'))

    if not fold_paths:
        print(f"Error: No fold models found in {model_dir}")
        print("Run: python train_kfold.py --n-splits 5 --epochs 50")
        return

    print(f"Loading {len(fold_paths)} ensemble models ...")
    models_list = []
    class_names = None
    for p in fold_paths:
        model, cls = get_model(p)
        model = model.to(device)
        models_list.append(model)
        class_names = cls
        print(f"  Loaded: {p.name}")

    print(f"Classes: {class_names}")

    # -------------------------------------------------------------------------
    # 推理
    # -------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    input_path = Path(args.image)

    if input_path.is_file():
        print(f"\nImage: {input_path.name}")
        print("-" * 50)

        pred_class, confidence, avg_probs, individual = ensemble_predict(
            models_list, input_path, transform, class_names, device
        )

        label = CLASS_LABELS.get(pred_class, pred_class)
        print(f"Ensemble Prediction: {label}")
        print(f"Ensemble Confidence: {confidence*100:.2f}%")
        print(f"\nPer-fold predictions:")
        for r in individual:
            print(f"  Fold {r['fold']}: {r['prediction']} ({r['confidence']*100:.2f}%)")

        # Top-3
        top3_idx = avg_probs.argsort()[-3:][::-1]
        print(f"\nTop-3 Ensemble:")
        for i, idx in enumerate(top3_idx, 1):
            cls = class_names[idx]
            conf = avg_probs[idx]
            print(f"  {i}. {CLASS_LABELS.get(cls, cls)}: {conf*100:.2f}%")
        print()

    elif input_path.is_dir():
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        image_files = [f for f in input_path.iterdir()
                      if f.suffix.lower() in image_extensions]

        print(f"Found {len(image_files)} images\n")

        results = []
        for img_path in sorted(image_files):
            pred_class, confidence, _, _ = ensemble_predict(
                models_list, img_path, transform, class_names, device
            )
            label = CLASS_LABELS.get(pred_class, pred_class)
            results.append({
                'file': img_path.name,
                'prediction': label,
                'confidence': f"{confidence*100:.2f}%"
            })
            print(f"{img_path.name:40s} -> {label} ({confidence*100:.2f}%)")

        if args.output:
            output_path = Path(args.output)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"\nResults saved: {output_path}")

    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == '__main__':
    main()
