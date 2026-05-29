#!/usr/bin/env python3
"""
Auto Sort Workflow: AI filters IN, human filters OUT
自动分拣工作流：AI 把高置信度样本筛入 passed，用户只负责从 passed 里挑出错误的。

用法：
    # 先运行 batch_screening.py 生成报告
    python batch_screening.py --screen --input "../data/raw/苹果图片汇总" --output ../outputs/screening

    # 然后自动分拣（基于已有报告）
    python auto_sort.py --report ../outputs/screening/screening_report.json \
                        --input "../data/raw/苹果图片汇总" \
                        --output ../outputs/screening

    # 用户打开 passed/ 文件夹，人工看图，把错的拖到 rejected/
    # 最后 finalize：把 passed/ 里剩下的作为最终训练数据
    python auto_sort.py --finalize --output ../outputs/screening

分拣逻辑：
    passed:   置信度 >= 85% 且 风险=LOW 且 分歧度=0  → AI 高度确信
    review:   置信度 50-85% 或 风险=MEDIUM           → 建议人工看
    rejected: 置信度 < 50% 或 风险=HIGH               → AI 基本在猜，直接丢弃
"""

import argparse
import json
import shutil
from pathlib import Path


def auto_sort(report_path, input_dir, output_dir):
    """基于 screening 报告自动分拣文件到 passed/review/rejected。"""
    report_path = Path(report_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    with open(report_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    passed_dir = output_dir / 'passed'
    review_dir = output_dir / 'review'
    rejected_dir = output_dir / 'rejected'
    for d in (passed_dir, review_dir, rejected_dir):
        d.mkdir(parents=True, exist_ok=True)

    passed_list, review_list, rejected_list = [], [], []

    for r in data.get('results', []):
        if 'error' in r:
            continue

        fname = r['file']
        src = input_dir / fname
        if not src.exists():
            print(f"  SKIP (not found): {fname}")
            continue

        conf = r['confidence']
        risk = r['risk']
        disagree = r['disagreement']

        # 分拣逻辑
        if conf >= 0.85 and risk == 'LOW' and disagree == 0.0:
            dst = passed_dir / fname
            passed_list.append(fname)
        elif conf < 0.50 or risk == 'HIGH':
            dst = rejected_dir / fname
            rejected_list.append(fname)
        else:
            dst = review_dir / fname
            review_list.append(fname)

        shutil.copy2(str(src), str(dst))

    # 保存清单
    manifest = {
        'passed': passed_list,
        'review': review_list,
        'rejected': rejected_list,
        'counts': {
            'passed': len(passed_list),
            'review': len(review_list),
            'rejected': len(rejected_list),
        },
    }
    manifest_path = output_dir / 'auto_sort_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Auto Sort Complete")
    print(f"{'='*60}")
    print(f"passed:   {len(passed_list):4d}  -> {passed_dir}")
    print(f"  └── 用户只需打开此文件夹，把错的拖到 rejected/")
    print(f"review:   {len(review_list):4d}  -> {review_dir}")
    print(f"  └── 建议也快速看一遍，对的拖 passed/，错的拖 rejected/")
    print(f"rejected: {len(rejected_list):4d}  -> {rejected_dir}")
    print(f"  └── AI 直接丢弃，通常无需再看")
    print(f"\nManifest: {manifest_path}")


def finalize(output_dir):
    """Finalize：把 review/ 里剩余的文件合并到 passed/，生成最终清单。"""
    output_dir = Path(output_dir)
    passed_dir = output_dir / 'passed'
    review_dir = output_dir / 'review'

    if not review_dir.exists():
        print("review/ directory not found. Nothing to finalize.")
        return

    moved = []
    for f in review_dir.iterdir():
        if f.is_file():
            dst = passed_dir / f.name
            shutil.move(str(f), str(dst))
            moved.append(f.name)

    # 更新清单
    manifest_path = output_dir / 'auto_sort_manifest.json'
    if manifest_path.exists():
        with open(manifest_path, 'r', encoding='utf-8') as f:
            manifest = json.load(f)
        manifest['passed'].extend(moved)
        manifest['review'] = []
        manifest['counts']['passed'] = len(manifest['passed'])
        manifest['counts']['review'] = 0
        with open(manifest_path, 'w', encoding='utf-8') as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Finalize Complete")
    print(f"{'='*60}")
    print(f"Moved {len(moved)} files from review/ -> passed/")
    print(f"Final passed count: {len(list(passed_dir.iterdir()))}")
    print(f"\nYou can now use {passed_dir} as clean training data.")


def main():
    parser = argparse.ArgumentParser(description='Auto Sort: AI filters IN, human filters OUT')
    parser.add_argument('--auto-sort', action='store_true',
                        help='Run auto-sort based on screening report')
    parser.add_argument('--finalize', action='store_true',
                        help='Move remaining review/ files to passed/')
    parser.add_argument('--report', type=str,
                        default='../outputs/screening/screening_report.json',
                        help='Path to screening_report.json')
    parser.add_argument('--input', type=str,
                        default='../data/raw/苹果图片汇总',
                        help='Source image directory')
    parser.add_argument('--output', type=str,
                        default='../outputs/screening',
                        help='Output directory')
    args = parser.parse_args()

    if args.auto_sort:
        auto_sort(args.report, args.input, args.output)
    elif args.finalize:
        finalize(args.output)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
