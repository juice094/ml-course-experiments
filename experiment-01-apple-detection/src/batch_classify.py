#!/usr/bin/env python3
"""
Batch Classify: AI sorts images into 8 class folders, human pulls outliers into rejected/
批量分类归档：AI 按 8 类预测归档到对应文件夹，用户从各文件夹挑出异常放入 rejected/

用法：
    python batch_classify.py --input "../data/raw/苹果图片汇总" --output ../outputs/classified

输出结构：
    ../outputs/classified/
    ├── fresh/              # AI 预测为合格的图
    ├── diseased/
    ├── bruised/
    ├── rotten/
    ├── insect_damaged/
    ├── cracked/
    ├── wrinkled/
    ├── black_spot/
    ├── rejected/           # 用户手动移入的异常（初始为空）
    └── classify_log.json   # 每张图的预测详情（含置信度）

审核流程：
    1. AI 归档后，用户逐文件夹浏览
    2. 发现某图类别明显错误，直接拖到 rejected/
    3. 全部审核完毕后，rejected/ 里的图可重新标注后补充训练
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

NUM_CLASSES = 8
CLASS_NAMES = [
    "fresh", "diseased", "bruised", "rotten",
    "insect_damaged", "cracked", "wrinkled", "black_spot"
]
CLASS_NAMES_CN = {
    "fresh": "合格", "diseased": "病变", "bruised": "碰伤",
    "rotten": "腐烂", "insect_damaged": "虫伤", "cracked": "裂果",
    "wrinkled": "褶皱", "black_spot": "黑斑",
}


def get_model(model_path):
    """加载单模型（EfficientNet-B0 + Dropout）。"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('class_names', CLASS_NAMES)


def load_ensemble_models(model_dir):
    """加载 K-Fold 集成模型。"""
    model_dir = Path(model_dir)
    fold_paths = sorted(model_dir.glob('best_model_fold*.pth'))
    if not fold_paths:
        raise FileNotFoundError(
            f"No fold models found in {model_dir}. "
            "Run: python train_kfold.py --n-splits 5 --epochs 50"
        )

    models_list = []
    class_names = None
    for p in fold_paths:
        model, cls = get_model(p)
        models_list.append(model)
        class_names = cls
    return models_list, class_names


def ensemble_predict(models_list, image_path, transform, class_names, device):
    """集成推理，返回预测类别和平均概率分布。"""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    all_probs = []
    all_preds = []
    with torch.no_grad():
        for model in models_list:
            model = model.to(device)
            model.eval()
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            all_preds.append(int(probs.argmax()))

    all_probs = np.array(all_probs)
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(avg_probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(avg_probs[pred_idx])

    sorted_probs = np.sort(avg_probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1])

    unique_preds, counts = np.unique(all_preds, return_counts=True)
    disagreement = 1.0 - (counts.max() / len(all_preds))

    return {
        'prediction': pred_class,
        'prediction_cn': CLASS_NAMES_CN.get(pred_class, pred_class),
        'confidence': round(confidence, 4),
        'margin': round(margin, 4),
        'disagreement': round(disagreement, 4),
        'top3': [
            {
                'class': CLASS_NAMES_CN.get(class_names[idx], class_names[idx]),
                'prob': round(float(avg_probs[idx]), 4),
            }
            for idx in np.argsort(avg_probs)[::-1][:3]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description='Batch classify into 8 folders')
    parser.add_argument('--input', type=str,
                        default='../data/raw/苹果图片汇总',
                        help='Input directory containing images to classify')
    parser.add_argument('--output', type=str,
                        default='../outputs/classified',
                        help='Output root directory for 8 class folders')
    parser.add_argument('--model-dir', type=str,
                        default='../models',
                        help='Directory containing K-Fold models')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 创建 8 个类别文件夹 + rejected
    for cls in CLASS_NAMES:
        (output_dir / cls).mkdir(exist_ok=True)
    (output_dir / 'rejected').mkdir(exist_ok=True)

    print(f"Loading ensemble models from: {args.model_dir}")
    models_list, class_names = load_ensemble_models(args.model_dir)
    print(f"Loaded {len(models_list)} models")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([f for f in input_dir.iterdir()
                         if f.suffix.lower() in exts])
    print(f"Found {len(image_files)} images")

    results = []
    for img_path in image_files:
        try:
            info = ensemble_predict(models_list, img_path, transform, class_names, device)
            pred_class = info['prediction']

            # 复制到对应类别文件夹
            dst = output_dir / pred_class / img_path.name
            shutil.copy2(str(img_path), str(dst))

            results.append({
                'file': img_path.name,
                'folder': pred_class,
                **info,
            })

            flag = "⚠️" if info['confidence'] < 0.5 else ""
            print(f"  {img_path.name:50s} -> {info['prediction_cn']:8s} "
                  f"({info['confidence']*100:5.1f}%) {flag}")

        except Exception as e:
            print(f"  ERROR: {img_path.name}: {e}")
            results.append({'file': img_path.name, 'error': str(e)})

    # 保存分类日志
    log_path = output_dir / 'classify_log.json'
    with open(log_path, 'w', encoding='utf-8') as f:
        json.dump({
            'total': len(image_files),
            'models_used': len(models_list),
            'results': results,
        }, f, ensure_ascii=False, indent=2)

    # 各类别统计
    counts = {cls: len(list((output_dir / cls).iterdir())) for cls in CLASS_NAMES}
    print(f"\n{'='*60}")
    print(f"Batch Classify Complete")
    print(f"{'='*60}")
    for cls in CLASS_NAMES:
        cn = CLASS_NAMES_CN[cls]
        print(f"  {cn:8s} ({cls:15s}): {counts[cls]:4d} 张")
    print(f"\nOutput: {output_dir}")
    print(f"Log:    {log_path}")
    print(f"\nNext: 打开各文件夹，把不符合的图拖到 rejected/ 中")


if __name__ == '__main__':
    main()
