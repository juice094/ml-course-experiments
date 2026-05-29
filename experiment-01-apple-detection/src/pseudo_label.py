#!/usr/bin/env python3
"""
Pseudo-Label Generation for Unlabeled Images
伪标签生成：用训练好的模型预测无标签图片，筛选高置信度样本

流程：
1. 加载已训练模型
2. 批量预测 data/raw/苹果图片汇总 中的 765 张图
3. 筛选 top-1 置信度 >= threshold 的样本
4. 输出统计、JSON 结果、可选按类别分目录复制
"""

import argparse
import json
import shutil
from pathlib import Path

import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
import torch.nn as nn

# 必须与 train.py 的模型架构一致
NUM_CLASSES = 8
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

UNLABELED_DIR = Path(__file__).parent.parent / "data" / "raw" / "苹果图片汇总"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def get_model(num_classes=NUM_CLASSES, model_path=None):
    """加载模型（与 train.py/evaluate.py 架构一致）。"""
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


def predict_batch(model, image_paths, transform, class_names, device, batch_size=32):
    """批量预测图片。"""
    model.eval()
    results = []

    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_tensors = []

        for p in batch_paths:
            img = Image.open(p).convert('RGB')
            batch_tensors.append(transform(img))

        batch_input = torch.stack(batch_tensors).to(device)

        with torch.no_grad():
            outputs = model(batch_input)
            probabilities = torch.softmax(outputs, dim=1)
            confidences, predicted = probabilities.max(1)

        for j, p in enumerate(batch_paths):
            pred_class = class_names[predicted[j].item()]
            conf = confidences[j].item()
            results.append({
                'file': p.name,
                'prediction': pred_class,
                'label_zh': CLASS_LABELS.get(pred_class, pred_class),
                'confidence': round(conf, 6),
                'top3': [
                    {
                        'class': class_names[idx],
                        'label_zh': CLASS_LABELS.get(class_names[idx], class_names[idx]),
                        'confidence': round(probabilities[j][idx].item(), 6)
                    }
                    for idx in probabilities[j].topk(3).indices.tolist()
                ]
            })

    return results


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo-labels for unlabeled images')
    parser.add_argument('--model', type=str, default='../models/best_model.pth',
                        help='Path to trained model')
    parser.add_argument('--threshold', type=float, default=0.95,
                        help='Minimum confidence for pseudo-label acceptance')
    parser.add_argument('--copy', action='store_true',
                        help='Copy high-confidence images to data/pseudo_train/')
    parser.add_argument('--batch-size', type=int, default=32, help='Inference batch size')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # -------------------------------------------------------------------------
    # 1. 加载模型
    # -------------------------------------------------------------------------
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    model, class_names = get_model(num_classes=NUM_CLASSES, model_path=str(model_path))
    model = model.to(device)
    model.eval()
    print(f"Loaded model: {model_path}")
    print(f"Classes: {class_names}")

    # -------------------------------------------------------------------------
    # 2. 准备数据
    # -------------------------------------------------------------------------
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([f for f in UNLABELED_DIR.iterdir()
                          if f.suffix.lower() in image_extensions])
    print(f"Found {len(image_files)} unlabeled images")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 3. 批量预测
    # -------------------------------------------------------------------------
    print("Running predictions ...")
    all_results = predict_batch(model, image_files, transform, class_names, device, args.batch_size)
    print(f"Predictions complete: {len(all_results)}")

    # -------------------------------------------------------------------------
    # 4. 筛选高置信度伪标签
    # -------------------------------------------------------------------------
    high_conf = [r for r in all_results if r['confidence'] >= args.threshold]
    low_conf = [r for r in all_results if r['confidence'] < args.threshold]

    print(f"\nHigh confidence (>= {args.threshold}): {len(high_conf)}")
    print(f"Low confidence (< {args.threshold}): {len(low_conf)}")

    # 按类别统计
    class_counts = {c: 0 for c in class_names}
    for r in high_conf:
        class_counts[r['prediction']] += 1

    print(f"\nPseudo-label distribution (threshold={args.threshold}):")
    for c, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {CLASS_LABELS.get(c, c):25s} : {count:4d}")

    # -------------------------------------------------------------------------
    # 5. 保存结果
    # -------------------------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary = {
        'model': str(model_path),
        'threshold': args.threshold,
        'total_unlabeled': len(all_results),
        'high_confidence_count': len(high_conf),
        'low_confidence_count': len(low_conf),
        'class_distribution': class_counts,
        'high_confidence_samples': high_conf,
    }

    out_path = OUTPUT_DIR / f'pseudo_labels_th{args.threshold}.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_path}")

    # -------------------------------------------------------------------------
    # 6. 可选：按类别复制到 data/pseudo_train/
    # -------------------------------------------------------------------------
    if args.copy:
        pseudo_dir = Path(__file__).parent.parent / "data" / "pseudo_train"
        pseudo_dir.mkdir(parents=True, exist_ok=True)

        for c in class_names:
            (pseudo_dir / c).mkdir(exist_ok=True)

        copied = 0
        for r in high_conf:
            src = UNLABELED_DIR / r['file']
            dst = pseudo_dir / r['prediction'] / r['file']
            if src.exists():
                shutil.copy2(src, dst)
                copied += 1

        print(f"Copied {copied} high-confidence images to: {pseudo_dir}")

        # 输出混合训练后的统计
        print(f"\nMixed training set preview:")
        for c in class_names:
            original = len(list((Path(__file__).parent.parent / "data" / "train" / c).iterdir()))
            pseudo = len(list((pseudo_dir / c).iterdir()))
            print(f"  {CLASS_LABELS.get(c, c):25s} : original={original:3d}  + pseudo={pseudo:3d}  = {original + pseudo:3d}")


if __name__ == '__main__':
    main()
