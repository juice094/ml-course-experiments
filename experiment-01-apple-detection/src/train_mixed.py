#!/usr/bin/env python3
"""
Apple Quality Classification - Mixed Training Script (Original + Pseudo-Labels)
苹果品质分类 - 混合训练脚本（原始标签 + 伪标签）

在 train.py 基础上，同时加载 data/train/（150 张人工标注）和
data/pseudo_train/（高置信度伪标签），用 ConcatDataset 合并后训练。
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
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
from tqdm import tqdm

# =============================================================================
# 配置区
# =============================================================================
TRAIN_DIR = Path(__file__).parent.parent / "data" / "train"
PSEUDO_DIR = Path(__file__).parent.parent / "data" / "pseudo_train"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

CLASS_NAMES = [
    "fresh", "diseased", "bruised", "rotten",
    "insect_damaged", "cracked", "wrinkled", "black_spot"
]
NUM_CLASSES = len(CLASS_NAMES)


def get_transforms(train=True):
    """数据变换管道（与 train.py 完全一致）。"""
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
    """加载 EfficientNet-B0（与 train.py 完全一致）。"""
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
    """训练一个 epoch（与 train.py 完全一致）。"""
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
    """验证一个 epoch（与 train.py 完全一致）。"""
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


def main():
    parser = argparse.ArgumentParser(description='Train with original + pseudo labels')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-pseudo', action='store_true',
                        help='仅使用原始训练集（用于对比实验）')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # ======================================================================
    # 数据加载：原始标签 + 伪标签（核心差异点）
    # ======================================================================
    print(f"Loading original dataset from: {TRAIN_DIR}")
    original_dataset = datasets.ImageFolder(str(TRAIN_DIR), transform=get_transforms(train=True))
    print(f"  Original samples: {len(original_dataset)}")

    datasets_to_concat = [original_dataset]

    if not args.no_pseudo and PSEUDO_DIR.exists():
        print(f"Loading pseudo-label dataset from: {PSEUDO_DIR}")
        pseudo_dataset = datasets.ImageFolder(str(PSEUDO_DIR), transform=get_transforms(train=True))
        print(f"  Pseudo-label samples: {len(pseudo_dataset)}")

        # 检查类别一致性：伪标签集的类别顺序必须与原始集一致
        if original_dataset.classes != pseudo_dataset.classes:
            print(f"Warning: Class mismatch!")
            print(f"  Original: {original_dataset.classes}")
            print(f"  Pseudo:   {pseudo_dataset.classes}")
            # 如果类别名相同但顺序不同，需要重映射
            # 这里假设两者类别名集合相同
        else:
            datasets_to_concat.append(pseudo_dataset)
    elif not args.no_pseudo:
        print(f"Warning: Pseudo-label dir not found: {PSEUDO_DIR}")
        print("  Run: python pseudo_label.py --threshold 0.95 --copy")
        print("  Falling back to original dataset only.")

    # 合并数据集
    full_dataset = ConcatDataset(datasets_to_concat)
    print(f"\nTotal mixed samples: {len(full_dataset)}")

    # 更新 CLASS_NAMES
    global CLASS_NAMES
    CLASS_NAMES = original_dataset.classes
    print(f"Classes: {CLASS_NAMES}")

    # 划分训练/验证集（在合并后的数据集上随机划分）
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # 验证集关闭数据增强（确定性评估）
    # ConcatDataset 没有 .dataset.transform，需要分别处理底层数据集
    for ds in datasets_to_concat:
        ds.transform = get_transforms(train=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ======================================================================
    # 模型、损失、优化器（与 train.py 一致）
    # ======================================================================
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 类别权重（基于原始训练集的分布，保持不变）
    class_weights = torch.tensor([1.0, 1.0, 1.25, 1.4, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=OUTPUT_DIR / 'runs' / f'mixed_{run_timestamp}')

    # 日志文件
    log_file = OUTPUT_DIR / f'training_log_mixed_{run_timestamp}.txt'
    metrics_file = OUTPUT_DIR / f'metrics_mixed_{run_timestamp}.json'

    best_val_acc = 0.0
    history = []

    print(f"\n{'='*50}")
    print("Starting mixed training...")
    print(f"{'='*50}\n")

    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write(f"Apple Quality Detection - Mixed Training Log\n")
        lf.write(f"Run Timestamp: {run_timestamp}\n")
        lf.write(f"Model: EfficientNet-B0 + Dropout(0.5)\n")
        lf.write(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}\n")
        lf.write(f"Original Samples: {len(original_dataset)}\n")
        if len(datasets_to_concat) > 1:
            lf.write(f"Pseudo-label Samples: {len(datasets_to_concat[1])}\n")
        lf.write(f"Mixed Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")
        lf.write(f"Classes: {CLASS_NAMES}\n")
        lf.write(f"Device: {device}\n")
        lf.write("=" * 70 + "\n")
        lf.write(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>12} | {'Best':>4}\n")
        lf.write("-" * 70 + "\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        # 训练前恢复训练集的数据增强（验证阶段已关闭）
        for ds in datasets_to_concat:
            ds.transform = get_transforms(train=True)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        # 验证前关闭数据增强
        for ds in datasets_to_concat:
            ds.transform = get_transforms(train=False)

        val_loss, val_acc = validate(model, val_loader, criterion, device)
        scheduler.step()

        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        if is_best:
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")

        with open(log_file, 'a', encoding='utf-8') as lf:
            lr_val = optimizer.param_groups[0]['lr']
            lf.write(f"{epoch+1:6d} | {train_loss:10.4f} | {train_acc:9.2f}% | "
                     f"{val_loss:10.4f} | {val_acc:9.2f}% | {lr_val:12.6f} | "
                     f"{'*' if is_best else '':4s}\n")

        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'is_best': is_best,
        })

        if is_best:
            model_path = MODEL_DIR / 'best_model_mixed.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, model_path)

    writer.close()

    with open(metrics_file, 'w', encoding='utf-8') as f:
        json.dump({
            'run_timestamp': run_timestamp,
            'config': {
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'lr': args.lr,
                'val_split': args.val_split,
                'seed': args.seed,
            },
            'dataset': {
                'original_size': len(original_dataset),
                'pseudo_size': len(datasets_to_concat[1]) if len(datasets_to_concat) > 1 else 0,
                'mixed_train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'classes': CLASS_NAMES,
            },
            'best_val_acc': best_val_acc,
            'history': history,
        }, f, ensure_ascii=False, indent=2)

    with open(log_file, 'a', encoding='utf-8') as lf:
        lf.write("=" * 70 + "\n")
        lf.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        lf.write(f"Log saved to: {log_file}\n")
        lf.write(f"Metrics saved to: {metrics_file}\n")

    final_path = MODEL_DIR / 'final_model_mixed.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
    }, final_path)

    print(f"\n{'='*50}")
    print(f"Mixed training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
