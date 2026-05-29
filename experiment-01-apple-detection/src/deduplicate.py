#!/usr/bin/env python3
"""
Deduplication Check for Unlabeled Images
去重检查：检测 data/raw/苹果图片汇总 中的图片是否与 data/train 重复

方法：平均哈希 (aHash) + 汉明距离
"""

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

TRAIN_DIR = Path(__file__).parent.parent / "data" / "train"
UNLABELED_DIR = Path(__file__).parent.parent / "data" / "raw" / "苹果图片汇总"
OUTPUT_DIR = Path(__file__).parent.parent / "outputs"


def ahash(image_path, size=16):
    """计算平均哈希 (Average Hash)。"""
    img = Image.open(image_path).convert('L').resize((size, size), Image.LANCZOS)
    pixels = np.array(img, dtype=np.float32)
    mean = pixels.mean()
    diff = pixels > mean
    return diff.tobytes()


def hamming_distance(hash1, hash2):
    """计算两个哈希的汉明距离。"""
    return sum(bin(b1 ^ b2).count('1') for b1, b2 in zip(hash1, hash2))


def compute_hashes(image_dir):
    """遍历目录，计算所有图片的哈希。"""
    hashes = {}
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

    for img_path in sorted(image_dir.iterdir()):
        if img_path.suffix.lower() in image_extensions:
            try:
                h = ahash(img_path)
                hashes[img_path.name] = h
            except Exception as e:
                print(f"  Warning: failed to hash {img_path.name}: {e}")

    return hashes


def main():
    parser = argparse.ArgumentParser(description='Check duplicates between train and unlabeled sets')
    parser.add_argument('--threshold', type=int, default=5, help='Hamming distance threshold for duplicates')
    args = parser.parse_args()

    print(f"Train directory: {TRAIN_DIR}")
    print(f"Unlabeled directory: {UNLABELED_DIR}")
    print(f"Threshold: Hamming distance <= {args.threshold}")
    print("-" * 60)

    # -------------------------------------------------------------------------
    # 1. 计算训练集哈希
    # -------------------------------------------------------------------------
    print("\nComputing hashes for training set ...")
    train_hashes = {}
    for class_dir in sorted(TRAIN_DIR.iterdir()):
        if class_dir.is_dir():
            cls_hashes = compute_hashes(class_dir)
            for name, h in cls_hashes.items():
                train_hashes[f"{class_dir.name}/{name}"] = h
    print(f"  Training images: {len(train_hashes)}")

    # -------------------------------------------------------------------------
    # 2. 计算无标签集哈希
    # -------------------------------------------------------------------------
    print("\nComputing hashes for unlabeled set ...")
    unlabeled_hashes = compute_hashes(UNLABELED_DIR)
    print(f"  Unlabeled images: {len(unlabeled_hashes)}")

    # -------------------------------------------------------------------------
    # 3. 去重检测
    # -------------------------------------------------------------------------
    print(f"\nChecking duplicates (threshold={args.threshold}) ...")
    duplicates = []
    near_duplicates = []

    for u_name, u_hash in unlabeled_hashes.items():
        for t_name, t_hash in train_hashes.items():
            dist = hamming_distance(u_hash, t_hash)
            if dist == 0:
                duplicates.append({
                    'unlabeled': u_name,
                    'train': t_name,
                    'hamming_distance': dist,
                    'status': 'exact_duplicate'
                })
            elif dist <= args.threshold:
                near_duplicates.append({
                    'unlabeled': u_name,
                    'train': t_name,
                    'hamming_distance': dist,
                    'status': 'near_duplicate'
                })

    # -------------------------------------------------------------------------
    # 4. 输出结果
    # -------------------------------------------------------------------------
    print(f"\nExact duplicates (distance=0): {len(duplicates)}")
    print(f"Near duplicates (distance<={args.threshold}): {len(near_duplicates)}")
    print(f"Unique unlabeled images: {len(unlabeled_hashes) - len(duplicates)}")

    if duplicates:
        print("\nExact duplicates:")
        for d in duplicates[:10]:
            print(f"  {d['unlabeled']}  ->  {d['train']}  (dist={d['hamming_distance']})")
        if len(duplicates) > 10:
            print(f"  ... and {len(duplicates) - 10} more")

    if near_duplicates:
        print("\nNear duplicates (sample):")
        for d in near_duplicates[:10]:
            print(f"  {d['unlabeled']}  ->  {d['train']}  (dist={d['hamming_distance']})")
        if len(near_duplicates) > 10:
            print(f"  ... and {len(near_duplicates) - 10} more")

    # 保存结果
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = {
        'threshold': args.threshold,
        'train_images': len(train_hashes),
        'unlabeled_images': len(unlabeled_hashes),
        'exact_duplicates': len(duplicates),
        'near_duplicates': len(near_duplicates),
        'safe_to_use': len(unlabeled_hashes) - len(duplicates),
        'duplicates': duplicates,
        'near_duplicates': near_duplicates,
    }

    out_path = OUTPUT_DIR / 'deduplication_results.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
