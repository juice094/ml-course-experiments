#!/usr/bin/env python3
"""
Staged Fine-Tuning Training Script
冻层分阶段训练脚本

阶段 1（Epoch 1-10）：冻结 EfficientNet backbone，只训练 classifier
  - 目的：保护 ImageNet 预训练特征不被初期大梯度破坏
  - lr=0.001，训练新加的 Dropout + FC 层

阶段 2（Epoch 11-50）：解冻全部层，以更小学习率微调
  - 目的：让 backbone 适应苹果缺陷的特定特征
  - lr=0.0001（阶段 1 的 1/10），温和调整预训练权重

原理：小样本迁移学习中，backbone 的预训练特征极其珍贵。
如果一开始就全网络训练，初期的大梯度会迅速破坏这些通用特征，
导致模型在少量样本上过拟合。分阶段训练是防止此问题的标准做法。
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
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms
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


def set_requires_grad(model, requires_grad):
    """统一设置模型所有参数是否需要梯度。"""
    for param in model.parameters():
        param.requires_grad = requires_grad


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


def main():
    parser = argparse.ArgumentParser(description='Staged fine-tuning for apple quality')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--lr-stage1', type=float, default=0.001, help='Stage 1 learning rate')
    parser.add_argument('--lr-stage2', type=float, default=0.0001, help='Stage 2 learning rate')
    parser.add_argument('--freeze-epochs', type=int, default=10, help='Epochs to freeze backbone')
    parser.add_argument('--val-split', type=float, default=0.2)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 数据加载（与 train.py 一致）
    # -------------------------------------------------------------------------
    print(f"Loading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=get_transforms(train=True))
    global CLASS_NAMES
    CLASS_NAMES = [full_dataset.classes[i] for i in range(len(full_dataset.classes))]
    print(f"Classes: {CLASS_NAMES}")
    print(f"Total samples: {len(full_dataset)}")

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    val_dataset.dataset.transform = get_transforms(train=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # -------------------------------------------------------------------------
    # 模型初始化
    # -------------------------------------------------------------------------
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # -------------------------------------------------------------------------
    # 阶段 1：冻结 backbone，只训练 classifier
    # -------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"Stage 1: Freeze backbone, train classifier (Epoch 1-{args.freeze_epochs})")
    print(f"Learning rate: {args.lr_stage1}")
    print(f"{'='*60}\n")

    # 冻结 backbone（features 部分）
    set_requires_grad(model.features, False)
    # 确保 classifier 可训练
    set_requires_grad(model.classifier, True)

    # 只优化 classifier 的参数
    stage1_params = filter(lambda p: p.requires_grad, model.parameters())

    class_weights = torch.tensor([1.0, 1.0, 1.25, 1.4, 1.0, 1.0, 1.0, 1.0], dtype=torch.float32)
    class_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.AdamW(stage1_params, lr=args.lr_stage1, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.freeze_epochs)

    # -------------------------------------------------------------------------
    # 日志设置
    # -------------------------------------------------------------------------
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=OUTPUT_DIR / 'runs' / f'staged_{run_timestamp}')
    log_file = OUTPUT_DIR / f'training_log_staged_{run_timestamp}.txt'
    metrics_file = OUTPUT_DIR / f'metrics_staged_{run_timestamp}.json'

    best_val_acc = 0.0
    history = []

    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write(f"Apple Quality Detection - Staged Fine-Tuning Log\n")
        lf.write(f"Stage 1: Epoch 1-{args.freeze_epochs}, Freeze backbone, LR={args.lr_stage1}\n")
        lf.write(f"Stage 2: Epoch {args.freeze_epochs+1}-{args.epochs}, Unfreeze all, LR={args.lr_stage2}\n")
        lf.write(f"{'='*70}\n")
        lf.write(f"{'Epoch':>6} | {'Stage':>8} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'Best':>4}\n")
        lf.write("-" * 70 + "\n")

    # -------------------------------------------------------------------------
    # 训练主循环
    # -------------------------------------------------------------------------
    for epoch in range(args.epochs):
        current_stage = 1 if epoch < args.freeze_epochs else 2
        stage_name = "Stage1" if current_stage == 1 else "Stage2"

        # =====================================================================
        # 阶段切换：解冻 backbone（在 Stage 2 开始时执行一次）
        # =====================================================================
        if epoch == args.freeze_epochs:
            print(f"\n{'='*60}")
            print(f"Stage 2: Unfreeze all layers (Epoch {epoch+1}-{args.epochs})")
            print(f"Learning rate: {args.lr_stage2}")
            print(f"{'='*60}\n")

            # 解冻全部参数
            set_requires_grad(model.features, True)
            set_requires_grad(model.classifier, True)

            # 重新初始化优化器：现在优化全部参数，使用更小的学习率
            optimizer = optim.AdamW(model.parameters(), lr=args.lr_stage2, weight_decay=5e-4)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - args.freeze_epochs)

        print(f"\nEpoch {epoch+1}/{args.epochs} [{stage_name}]")
        print("-" * 30)

        full_dataset.transform = get_transforms(train=True)
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)

        full_dataset.transform = get_transforms(train=False)
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
            lf.write(f"{epoch+1:6d} | {stage_name:>8} | {train_loss:10.4f} | {train_acc:9.2f}% | "
                     f"{val_loss:10.4f} | {val_acc:9.2f}% | {lr_val:12.6f} | "
                     f"{'*' if is_best else '':4s}\n")

        history.append({
            'epoch': epoch + 1,
            'stage': current_stage,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'is_best': is_best,
        })

        if is_best:
            model_path = MODEL_DIR / 'best_model_staged.pth'
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
            'config': vars(args),
            'dataset': {
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'classes': CLASS_NAMES,
            },
            'best_val_acc': best_val_acc,
            'history': history,
        }, f, ensure_ascii=False, indent=2)

    with open(log_file, 'a', encoding='utf-8') as lf:
        lf.write("=" * 70 + "\n")
        lf.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")

    final_path = MODEL_DIR / 'final_model_staged.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
    }, final_path)

    print(f"\n{'='*50}")
    print(f"Staged training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
