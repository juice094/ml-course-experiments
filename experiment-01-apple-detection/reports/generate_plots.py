#!/usr/bin/env python3
"""
Generate training curves from metrics JSON for experiment report.
Outputs PNG images to reports/figures/.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# For Chinese labels
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

REPORTS_DIR = Path(__file__).parent
FIGURES_DIR = REPORTS_DIR / 'figures'


def load_metrics(metrics_path: Path):
    with open(metrics_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_loss_curve(history, output_path: Path):
    epochs = [h['epoch'] for h in history]
    train_loss = [h['train_loss'] for h in history]
    val_loss = [h['val_loss'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_loss, label='Train Loss', color='steelblue', linewidth=2)
    ax.plot(epochs, val_loss, label='Validation Loss', color='coral', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Loss', fontsize=12)
    ax.set_title('Training & Validation Loss Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_accuracy_curve(history, output_path: Path):
    epochs = [h['epoch'] for h in history]
    train_acc = [h['train_acc'] for h in history]
    val_acc = [h['val_acc'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, train_acc, label='Train Accuracy', color='steelblue', linewidth=2)
    ax.plot(epochs, val_acc, label='Validation Accuracy', color='coral', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Training & Validation Accuracy Curve', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_lr_curve(history, output_path: Path):
    epochs = [h['epoch'] for h in history]
    lrs = [h['lr'] for h in history]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(epochs, lrs, color='green', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule (CosineAnnealingLR)', fontsize=14)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary_table(history, best_val_acc):
    print("\n" + "=" * 80)
    print("Training Summary Table (for pasting into report)")
    print("=" * 80)
    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>10} | {'Val Acc':>9} | {'Best':>4}")
    print("-" * 80)
    for h in history:
        mark = '*' if h['val_acc'] == best_val_acc else ''
        print(f"{h['epoch']:6d} | {h['train_loss']:10.4f} | {h['train_acc']:9.2f}% | "
              f"{h['val_loss']:10.4f} | {h['val_acc']:9.2f}% | {mark:>4}")
    print("=" * 80)
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")


def main():
    parser = argparse.ArgumentParser(description='Generate training plots from metrics JSON')
    parser.add_argument('metrics', type=str, nargs='?', default=None,
                        help='Path to metrics JSON file (default: latest in ../outputs/)')
    args = parser.parse_args()

    if args.metrics:
        metrics_path = Path(args.metrics)
    else:
        outputs_dir = Path(__file__).parent.parent / 'outputs'
        json_files = sorted(outputs_dir.glob('metrics_*.json'), key=lambda p: p.stat().st_mtime, reverse=True)
        if not json_files:
            print(f"Error: No metrics JSON found in {outputs_dir}")
            print("Please run training first, or specify --metrics path.")
            return
        metrics_path = json_files[0]
        print(f"Using latest metrics: {metrics_path}")

    data = load_metrics(metrics_path)
    history = data['history']
    best_val_acc = data['best_val_acc']

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    plot_loss_curve(history, FIGURES_DIR / 'loss_curve.png')
    plot_accuracy_curve(history, FIGURES_DIR / 'accuracy_curve.png')
    plot_lr_curve(history, FIGURES_DIR / 'lr_curve.png')

    print_summary_table(history, best_val_acc)

    print(f"\nAll figures saved to: {FIGURES_DIR}")


if __name__ == '__main__':
    main()
