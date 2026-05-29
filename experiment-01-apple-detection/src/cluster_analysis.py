#!/usr/bin/env python3
"""
Unsupervised Clustering Analysis
无监督聚类分析：用预训练 backbone 提取特征，K-Means 聚成 8 类，
与人工标签对比，验证类别定义的自然性。

输出：
- 聚类 vs 真实标签对比表
- Adjusted Rand Index (ARI)
- PCA 2D 可视化图
- 各类别混淆分析
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, confusion_matrix
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).parent.parent / "data" / "train"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"
REPORTS_DIR = Path(__file__).parent.parent / "reports" / "figures"


def get_backbone():
    """加载 EfficientNet-B0 的 backbone（去掉 classifier）。"""
    model = models.efficientnet_b0(weights='IMAGENET1K_V1')
    # 去掉 classifier，只保留特征提取部分
    backbone = model.features
    backbone.eval()
    return backbone


def extract_features(backbone, dataset, device):
    """提取所有图片的特征向量。"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 为 dataset 应用统一的 transform
    dataset.transform = transform
    loader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=2)

    all_features = []
    all_labels = []
    all_paths = []

    backbone = backbone.to(device)
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            features = backbone(images)  # (batch, 1280, 7, 7)
            features = torch.mean(features, dim=[2, 3])  # Global Average Pooling -> (batch, 1280)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())

    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    # 获取文件路径
    for i in range(len(dataset)):
        path, _ = dataset.samples[i]
        all_paths.append(Path(path).name)

    return all_features, all_labels, all_paths


def map_clusters_to_labels(cluster_ids, true_labels, class_names):
    """
    将聚类标签映射到真实标签。
    方法：对每个聚类，找出该簇中数量最多的真实类别，作为该簇的命名。
    """
    n_clusters = len(np.unique(cluster_ids))
    cluster_to_label = {}
    cluster_name = {}

    for c in range(n_clusters):
        mask = cluster_ids == c
        labels_in_cluster = true_labels[mask]
        if len(labels_in_cluster) == 0:
            cluster_name[c] = "empty"
            continue
        # 找出该簇中最常见的真实标签
        unique, counts = np.unique(labels_in_cluster, return_counts=True)
        majority_label = unique[np.argmax(counts)]
        cluster_to_label[c] = majority_label
        cluster_name[c] = class_names[majority_label]

    return cluster_to_label, cluster_name


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # 1. 加载数据和模型
    # -------------------------------------------------------------------------
    print("Loading dataset ...")
    dataset = ImageFolder(str(DATA_DIR))
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    print(f"Total samples: {len(dataset)}")

    print("\nLoading backbone ...")
    backbone = get_backbone()

    # -------------------------------------------------------------------------
    # 2. 提取特征
    # -------------------------------------------------------------------------
    print("\nExtracting features ...")
    features, true_labels, paths = extract_features(backbone, dataset, device)
    print(f"Feature shape: {features.shape}")  # (150, 1280)

    # -------------------------------------------------------------------------
    # 3. K-Means 聚类
    # -------------------------------------------------------------------------
    print("\nRunning K-Means (n_clusters=8) ...")
    kmeans = KMeans(n_clusters=8, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(features)

    # -------------------------------------------------------------------------
    # 4. 与真实标签对比
    # -------------------------------------------------------------------------
    ari = adjusted_rand_score(true_labels, cluster_ids)
    print(f"\nAdjusted Rand Index (ARI): {ari:.4f}")
    print("  ARI = 1.0: 聚类与人工标签完全一致")
    print("  ARI = 0.0: 随机聚类")
    print("  ARI < 0: 聚类比随机还差")

    # 聚类 → 真实标签映射
    cluster_to_label, cluster_name = map_clusters_to_labels(cluster_ids, true_labels, class_names)

    # 构建混淆矩阵（聚类 vs 真实标签）
    cm = confusion_matrix(true_labels, [cluster_to_label.get(c, -1) for c in cluster_ids])

    print(f"\nCluster-to-Label Mapping:")
    for c in sorted(cluster_name.keys()):
        print(f"  Cluster {c} -> {cluster_name[c]}")

    # -------------------------------------------------------------------------
    # 5. 每类聚类纯度分析
    # -------------------------------------------------------------------------
    print(f"\nPer-Class Cluster Purity:")
    purity_results = []
    for label_idx, cls_name in enumerate(class_names):
        mask = true_labels == label_idx
        n_total = mask.sum()
        clusters_for_class = cluster_ids[mask]
        unique, counts = np.unique(clusters_for_class, return_counts=True)
        majority_cluster = unique[np.argmax(counts)]
        majority_count = counts.max()
        purity = majority_count / n_total * 100

        purity_results.append({
            'class': cls_name,
            'total_samples': int(n_total),
            'dominant_cluster': int(majority_cluster),
            'dominant_count': int(majority_count),
            'purity': round(purity, 2),
        })
        print(f"  {cls_name:20s}: {purity:5.1f}% ({majority_count}/{n_total}) in Cluster {majority_cluster}")

    # -------------------------------------------------------------------------
    # 6. PCA 2D 可视化
    # -------------------------------------------------------------------------
    print("\nGenerating PCA visualization ...")
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features)
    print(f"  Explained variance ratio: {pca.explained_variance_ratio_}")

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # 左图：按真实标签着色
    colors_true = plt.cm.tab10(np.linspace(0, 1, 8))
    for i, cls in enumerate(class_names):
        mask = true_labels == i
        axes[0].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors_true[i]], label=cls, s=60, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[0].set_title('PCA by True Labels')
    axes[0].set_xlabel('PC1')
    axes[0].set_ylabel('PC2')
    axes[0].legend(loc='best', fontsize=8)
    axes[0].grid(True, alpha=0.3)

    # 右图：按聚类标签着色
    colors_cluster = plt.cm.Set2(np.linspace(0, 1, 8))
    for c in range(8):
        mask = cluster_ids == c
        axes[1].scatter(features_2d[mask, 0], features_2d[mask, 1],
                       c=[colors_cluster[c]], label=f'Cluster {c} ({cluster_name[c]})',
                       s=60, alpha=0.7, edgecolors='k', linewidth=0.5)
    axes[1].set_title('PCA by K-Means Clusters')
    axes[1].set_xlabel('PC1')
    axes[1].set_ylabel('PC2')
    axes[1].legend(loc='best', fontsize=8)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    viz_path = REPORTS_DIR / 'cluster_pca_visualization.png'
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {viz_path}")

    # -------------------------------------------------------------------------
    # 7. 保存结果
    # -------------------------------------------------------------------------
    results = {
        'ari': round(ari, 4),
        'n_samples': len(dataset),
        'feature_dim': features.shape[1],
        'pca_explained_variance': pca.explained_variance_ratio_.tolist(),
        'cluster_to_label': {int(k): class_names[v] for k, v in cluster_to_label.items()},
        'per_class_purity': purity_results,
    }

    out_path = OUTPUT_DIR / 'cluster_analysis_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_path}")

    # 保存混淆矩阵图
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=class_names, yticklabels=class_names,
           title='Cluster vs True Label Confusion Matrix',
           ylabel='True label',
           xlabel='Mapped Cluster label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    cm_path = REPORTS_DIR / 'cluster_confusion_matrix.png'
    fig.savefig(cm_path, dpi=150)
    plt.close()
    print(f"  Saved: {cm_path}")

    print(f"\n{'='*60}")
    print(f"Clustering Analysis Complete!")
    print(f"ARI: {ari:.4f}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
