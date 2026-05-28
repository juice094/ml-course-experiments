#!/usr/bin/env python3
"""
Apple Quality Classification - Evaluation Script
Generate confusion matrix and per-class metrics on validation set.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "train"
MODEL_DIR = Path(__file__).parent.parent / "models"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


def get_transforms():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])


def get_model(num_classes=8, model_path=None):
    # 必须与 train.py 的模型架构一致：EfficientNet-B0 + Dropout
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, num_classes)
    )

    if model_path and Path(model_path).exists():
        checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
        model.load_state_dict(checkpoint['model_state_dict'])
        class_names = checkpoint.get('class_names', [f'class_{i}' for i in range(num_classes)])
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

    return model, class_names


def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def print_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    max_name_len = max(len(c) for c in class_names)

    print("\nConfusion Matrix:")
    print(" " * (max_name_len + 2), end="")
    for name in class_names:
        print(f"{name[:4]:>5}", end=" ")
    print()

    for i, name in enumerate(class_names):
        print(f"{name:>{max_name_len}} |", end="")
        for j in range(len(class_names)):
            print(f"{cm[i, j]:>5}", end=" ")
        print()

    return cm


def main():
    parser = argparse.ArgumentParser(description='Evaluate apple quality classifier')
    parser.add_argument('--model', type=str, default='../models/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--val-split', type=float, default=0.2, help='Validation split')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load dataset
    full_dataset = datasets.ImageFolder(str(DATA_DIR), transform=get_transforms())
    class_names = [full_dataset.classes[i] for i in range(len(full_dataset.classes))]

    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    _, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(args.seed)
    )

    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2)

    # Load model
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = Path(__file__).parent / model_path

    model, loaded_class_names = get_model(num_classes=len(class_names), model_path=str(model_path))
    model = model.to(device)

    print(f"Evaluating on {len(val_dataset)} validation samples...")
    print(f"Classes: {class_names}")
    print(f"Model: {model_path}")
    print("-" * 60)

    y_true, y_pred = evaluate(model, val_loader, device)

    # Confusion matrix
    cm = print_confusion_matrix(y_true, y_pred, class_names)

    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))

    # Overall accuracy
    acc = (y_true == y_pred).mean() * 100
    print(f"Overall Accuracy: {acc:.2f}%")

    # Save confusion matrix plot if matplotlib available
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(10, 8))
        im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=class_names, yticklabels=class_names,
               title='Confusion Matrix',
               ylabel='True label',
               xlabel='Predicted label')
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

        # Add text annotations
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], 'd'),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")

        fig.tight_layout()
        out_path = REPORTS_DIR / 'confusion_matrix.png'
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"\nConfusion matrix plot saved to: {out_path}")
    except Exception as e:
        print(f"\nCould not save confusion matrix plot: {e}")


if __name__ == '__main__':
    main()
