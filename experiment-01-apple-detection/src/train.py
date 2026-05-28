#!/usr/bin/env python3
"""
Apple Quality Classification - Training Script
8-class classification: fresh, diseased, bruised, rotten, insect_damaged,
                        cracked, wrinkled, black_spot
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

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "train"
MODEL_DIR = Path(__file__).parent.parent / "models"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"

CLASS_NAMES = [
    "fresh", "diseased", "bruised", "rotten",
    "insect_damaged", "cracked", "wrinkled", "black_spot"
]

NUM_CLASSES = len(CLASS_NAMES)

# Data transforms
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
    """Load ResNet18 with custom classifier head."""
    model = models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
    # Replace final FC layer
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
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


def main():
    parser = argparse.ArgumentParser(description='Train apple quality classifier')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()

    # Set random seed
    torch.manual_seed(args.seed)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create directories
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    print(f"Loading dataset from: {DATA_DIR}")
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=get_transforms(train=True))

    # Fix class names mapping
    global CLASS_NAMES
    CLASS_NAMES = [full_dataset.classes[i] for i in range(len(full_dataset.classes))]
    print(f"Classes: {CLASS_NAMES}")
    print(f"Total samples: {len(full_dataset)}")

    # Train/val split
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    # Override val transform
    val_dataset.dataset.transform = get_transforms(train=False)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")

    # Dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # Model
    model = get_model(num_classes=NUM_CLASSES, pretrained=True)
    model = model.to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # TensorBoard
    run_timestamp = time.strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter(log_dir=OUTPUT_DIR / 'runs' / run_timestamp)

    # Text log file for course submission
    log_file = OUTPUT_DIR / f'training_log_{run_timestamp}.txt'
    metrics_file = OUTPUT_DIR / f'metrics_{run_timestamp}.json'

    # Training loop
    best_val_acc = 0.0
    history = []
    print(f"\n{'='*50}")
    print("Starting training...")
    print(f"{'='*50}\n")

    # Write log header
    with open(log_file, 'w', encoding='utf-8') as lf:
        lf.write(f"Apple Quality Detection - Training Log\n")
        lf.write(f"Run Timestamp: {run_timestamp}\n")
        lf.write(f"Model: ResNet18 (pretrained)\n")
        lf.write(f"Epochs: {args.epochs}, Batch Size: {args.batch_size}, LR: {args.lr}\n")
        lf.write(f"Train Samples: {len(train_dataset)}, Val Samples: {len(val_dataset)}\n")
        lf.write(f"Classes: {CLASS_NAMES}\n")
        lf.write(f"Device: {device}\n")
        lf.write("=" * 70 + "\n")
        lf.write(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'LR':>12} | {'Best':>4}\n")
        lf.write("-" * 70 + "\n")

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 30)

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        scheduler.step()

        # Log to TensorBoard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)

        is_best = val_acc > best_val_acc
        if is_best:
            best_val_acc = val_acc

        # Print to console
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        if is_best:
            print(f"Saved best model (val_acc: {val_acc:.2f}%)")

        # Write to text log
        with open(log_file, 'a', encoding='utf-8') as lf:
            lr_val = optimizer.param_groups[0]['lr']
            lf.write(f"{epoch+1:6d} | {train_loss:10.4f} | {train_acc:9.2f}% | "
                     f"{val_loss:10.4f} | {val_acc:9.2f}% | {lr_val:12.6f} | "
                     f"{'*' if is_best else '':4s}\n")

        # Append to history for JSON export
        history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': optimizer.param_groups[0]['lr'],
            'is_best': is_best,
        })

        # Save best model
        if is_best:
            model_path = MODEL_DIR / 'best_model.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'class_names': CLASS_NAMES,
            }, model_path)

    writer.close()

    # Save JSON metrics
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
                'train_size': len(train_dataset),
                'val_size': len(val_dataset),
                'classes': CLASS_NAMES,
            },
            'best_val_acc': best_val_acc,
            'history': history,
        }, f, ensure_ascii=False, indent=2)

    # Write log footer
    with open(log_file, 'a', encoding='utf-8') as lf:
        lf.write("=" * 70 + "\n")
        lf.write(f"Best Validation Accuracy: {best_val_acc:.2f}%\n")
        lf.write(f"Log saved to: {log_file}\n")
        lf.write(f"Metrics saved to: {metrics_file}\n")

    # Save final model
    final_path = MODEL_DIR / 'final_model.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': CLASS_NAMES,
    }, final_path)

    print(f"\n{'='*50}")
    print(f"Training complete! Best val accuracy: {best_val_acc:.2f}%")
    print(f"Models saved to: {MODEL_DIR}")
    print(f"{'='*50}")


if __name__ == '__main__':
    main()
