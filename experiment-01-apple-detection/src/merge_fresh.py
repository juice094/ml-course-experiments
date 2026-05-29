#!/usr/bin/env python3
"""
Merge screened fresh images into training set
将审核通过的 fresh 样本合并到训练集中

用法：
    python merge_fresh.py --source ../outputs/classified/fresh --target ../data/train/fresh

逻辑：
    1. 扫描 target 目录，找到现有的 FreshApple 最大编号
    2. 将 source 中的文件按顺序重命名并入 target
    3. 生成合并清单
"""

import argparse
import json
import re
import shutil
from pathlib import Path


def find_max_index(target_dir):
    """从现有文件名中提取最大编号。"""
    pattern = re.compile(r'FreshApple\s*\((\d+)\)\.jpg', re.IGNORECASE)
    max_idx = 0
    for f in target_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            max_idx = max(max_idx, int(m.group(1)))
    return max_idx


def merge(source_dir, target_dir):
    source_dir = Path(source_dir)
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    max_idx = find_max_index(target_dir)
    print(f"Target existing max index: {max_idx}")

    source_files = sorted([f for f in source_dir.iterdir() if f.is_file()])
    print(f"Source files to merge: {len(source_files)}")

    merged = []
    for i, src in enumerate(source_files, start=1):
        new_idx = max_idx + i
        dst_name = f"FreshApple ({new_idx}).jpg"
        dst = target_dir / dst_name
        shutil.copy2(str(src), str(dst))
        merged.append({
            'original_name': src.name,
            'new_name': dst_name,
            'source': str(src),
            'destination': str(dst),
        })
        print(f"  {src.name:40s} -> {dst_name}")

    # 保存清单
    manifest = {
        'source_dir': str(source_dir),
        'target_dir': str(target_dir),
        'original_max_index': max_idx,
        'merged_count': len(merged),
        'merged_files': merged,
    }
    manifest_path = target_dir.parent / 'merge_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Merge Complete")
    print(f"{'='*60}")
    print(f"Added {len(merged)} fresh images to {target_dir}")
    print(f"New total in fresh/: {max_idx + len(merged)}")
    print(f"Manifest: {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description='Merge screened fresh images into training set')
    parser.add_argument('--source', type=str, default='../outputs/classified/fresh',
                        help='Source directory of screened fresh images')
    parser.add_argument('--target', type=str, default='../data/train/fresh',
                        help='Target training directory for fresh class')
    args = parser.parse_args()

    merge(args.source, args.target)


if __name__ == '__main__':
    main()
