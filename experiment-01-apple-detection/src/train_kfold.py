#!/usr/bin/env python3
"""
K-Fold Cross-Validation Training + Ensemble
K-Fold 交叉验证训练 + 模型集成

原理：
1. 将 150 张训练数据划分为 5 份（每份 30 张）
2. 每轮用 4 份训练、1 份验证，循环 5 次
3. 保存 5 个最优模型
4. 推理时取 5 个模型的 softmax 平均，降低单次划分的随机偏差

预期效果：
- 单次 random_split 的验证集仅 30 张，结果波动大（±3.33%）
- 5-Fold 平均后，评估更稳定，集成预测通常提升 2-4%
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from sklearn.model_selection import KFold
from tqdm import tqdm

DATA_DIR = Path(__file__).parent.parent / "data" / "train"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

CLASS_NAMES = [
    "fresh", "diseased", "bruised", "rotten",
    "insect_damaged", "cracked", "wrinkled", "black_spot"
]
NUM_CLASSES = len(CLASS_NAMES)


def get_transforms(train=True):
    if train:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])


def get_model(num_classes=NUM_CLASSES, pretrained=True):
    model = models.efficientnet_b0(
        weights='IMAGENET1K_V1' if pretrained else None
    )
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )
    return model


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc="Training")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(dataloader), 100. * correct / total


def validate(model, dataloader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return running_loss / len(dataloader), 100. * correct / total


def train_one_fold(fold, train_idx, val_idx, full_dataset, args, device):
    """训练单个 Fold。"""
    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/{args.n_splits}")
    print(f"Train: {len(train_idx)} samples, Val: {len(val_idx)} samples")
    print(f"{'='*60}")

    # 构建 Subset
    train_subset = Subset(full_dataset, train_idx)
    val_subset = Subset(full_dataset, val_idx)

    # DataLoader
    train_loader = DataLoader(train_subset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_subset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # 模型
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)

    # 类别权重
    class_weights = torch.tensor([1.0, 1.0, 1.25, 1.4, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    history = []

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        # 训练前恢复增强
        full_dataset.transform = get_transforms(train=True)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证前关闭增强
        full_dataset.transform = get_transforms(train=False)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        if is_best:
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'is_best': is_best,
        })

        if is_best:
            model_path = MODEL_DIR / f'best_model_fold{fold}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, model_path)

    return best_val_acc, history


def main():
    parser = argparse.ArgumentParser(description='K-Fold CV for apple quality')
    parser.add_argument('--n-splits', type=int, default=5, help='Number of folds')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 加载数据集
    # -------------------------------------------------------------------------
    print(f"Loading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=get_transforms(train=True))
    global CLASS_NAMES
    CLASS_NAMES = [full_dataset.classes[i] for i in range(len(full_dataset.classes))]
    print(f"Classes: {CLASS_NAMES}")
    print(f"Total samples: {len(full_dataset)}")

    # -------------------------------------------------------------------------
    # K-Fold 划分
    # -------------------------------------------------------------------------
    kfold = KFold(n_splits=args.n_splits, shuffle=True, random_state=args.seed)

    all_fold_results = []
    overall_best = 0.0

    for fold, (train_idx, val_idx) in enumerate(kfold.split(range(len(full_dataset)))):
        best_val_acc, history = train_one_fold(
            fold, train_idx, val_idx, full_dataset, args, device
        )
        all_fold_results.append({
            'fold': fold,
            'best_val_acc': best_val_acc,
            'history': history,
        })
        overall_best = max(overall_best, best_val_acc)

    # -------------------------------------------------------------------------
    # 汇总结果
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"K-Fold Cross-Validation Complete!")
    print(f"{'='*60}")

    fold_accs = [r['best_val_acc'] for r in all_fold_results]
    avg_acc = sum(fold_accs) / len(fold_accs)

    print(f"\nPer-fold Best Val Acc:")
    for i, acc in enumerate(fold_accs):
        print(f"  Fold {i}: {acc:.2f}%")
    print(f"\nAverage Val Acc: {avg_acc:.2f}%")
    print(f"Overall Best:    {overall_best:.2f}%")
    print(f"\nModels saved: {MODEL_DIR}/best_model_fold*.pth")

    # 保存汇总 JSON
    summary_file = OUTPUT_DIR / f'kfold_summary_{time.strftime("%Y%m%d_%H%M%S")}.json'
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            'n_splits': args.n_splits,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'seed': args.seed,
            'per_fold_best': fold_accs,
            'average_val_acc': avg_acc,
            'overall_best': overall_best,
            'fold_results': all_fold_results,
        }, f, ensure_ascii=False, indent=2)
    print(f"Summary saved: {summary_file}")


if __name__ == '__main__':
    main()
