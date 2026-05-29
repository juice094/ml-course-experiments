#!/usr/bin/env python3
"""
Test-Time Augmentation (TTA) Inference
测试时增强推理：对同一张图做多次增强取平均，提升预测稳定性

适用场景：
- 已有训练好的模型，不想重新训练
- 希望压榨现有模型的最后 1-3% 准确率
- 尤其对小数据集有效（降低单张图的随机裁剪偏差）
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


def get_model(num_classes=8, model_path=None):
    """加载模型（与 train.py 一致）。"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', list(CLASS_LABELS.keys()))
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    return model, class_names


def get_tta_transforms():
    """
    定义 TTA 用的多种增强策略。
    每种策略对同一张图产生不同的视角/光照变化，取平均可降低方差。
    """
    base_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                          std=[0.229, 0.224, 0.225])
    return [
        # 策略 0：标准中心裁剪（基线）
        transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # 策略 1：左侧裁剪
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=0),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # 策略 2：右侧裁剪
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224, padding=0),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # 策略 3：水平翻转 + 中心裁剪
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.RandomHorizontalFlip(p=1.0),  # 强制翻转
            transforms.ToTensor(),
            base_normalize,
        ]),
        # 策略 4：轻微旋转
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomRotation(degrees=10),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            base_normalize,
        ]),
        # 策略 5：ColorJitter（亮度变化）
        transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ColorJitter(brightness=0.15, contrast=0.15),
            transforms.ToTensor(),
            base_normalize,
        ]),
    ]


def tta_predict(model, image_path, tta_transforms, class_names, device):
    """
    对单张图执行 TTA 推理。

    返回:
        avg_probs: 各策略预测概率的平均值 (num_classes,)
        pred_class: 平均概率最高的类别
        confidence: 最高类别的平均概率
        individual: 各策略的独立预测结果（用于分析）
    """
    image = Image.open(image_path).convert('RGB')
    model.eval()

    all_probs = []
    individual = []

    with torch.no_grad():
        for i, transform in enumerate(tta_transforms):
            input_tensor = transform(image).unsqueeze(0).to(device)
            output = model(input_tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)

            pred_idx = probs.argmax()
            individual.append({
                'strategy': i,
                'prediction': class_names[pred_idx],
                'confidence': round(float(probs[pred_idx]), 4),
            })

    # 各策略概率取平均
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = avg_probs.argmax()
    pred_class = class_names[pred_idx]
    confidence = float(avg_probs[pred_idx])

    return avg_probs, pred_class, confidence, individual


def main():
    parser = argparse.ArgumentParser(description='TTA Inference for apple quality')
    parser.add_argument('image', type=str, help='Path to image or directory')
    parser.add_argument('--model', type=str, default='../models/best_model.pth')
    parser.add_argument('--tta-times', type=int, default=6,
                        help='TTA 策略数量（默认 6 种增强）')
    parser.add_argument('--output', type=str, default=None,
                        help='Output JSON for batch predictions')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    model, class_names = get_model(num_classes=8, model_path=str(model_path))
    model = model.to(device)

    tta_transforms = get_tta_transforms()[:args.tta_times]
    print(f"TTA strategies: {len(tta_transforms)}")

    input_path = Path(args.image)

    if input_path.is_file():
        print(f"\nImage: {input_path.name}")
        print("-" * 50)

        avg_probs, pred_class, confidence, individual = tta_predict(
            model, input_path, tta_transforms, class_names, device
        )

        print(f"TTA Prediction: {CLASS_LABELS.get(pred_class, pred_class)}")
        print(f"TTA Confidence: {confidence*100:.2f}%")
        print(f"\nIndividual strategies:")
        for r in individual:
            label = CLASS_LABELS.get(r['prediction'], r['prediction'])
            print(f"  Strategy {r['strategy']}: {label} ({r['confidence']*100:.2f}%)")

        # Top-3 平均概率
        top3_idx = avg_probs.argsort()[-3:][::-1]
        print(f"\nTTA Top-3:")
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
            avg_probs, pred_class, confidence, _ = tta_predict(
                model, img_path, tta_transforms, class_names, device
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
            print(f"\nResults saved to: {output_path}")

    else:
        print(f"Error: Path not found: {input_path}")


if __name__ == '__main__':
    main()
