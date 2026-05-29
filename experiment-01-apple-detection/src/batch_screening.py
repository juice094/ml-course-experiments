#!/usr/bin/env python3
"""
Batch Screening Workflow: AI coarse filter + Human review
批量筛查工作流：AI 粗筛 → 人工审核 → 异常剔除

用法：
    # Stage 1: AI 粗筛（生成审核报告）
    python batch_screening.py --screen --input ../data/unlabeled --output ../outputs/screening

    # Stage 2: 人工审核后，按审核结果分拣
    python batch_screening.py --review --report ../outputs/screening/screening_report_reviewed.json --input ../data/unlabeled --output ../outputs/screening

输出：
    screening_report.json   — 结构化报告（含置信度、分歧度、风险等级）
    screening_report.csv    — 表格化报告（便于 Excel/人工审阅）
    passed/                 — 审核通过的样本（可复制或生成清单）
    rejected/               — 审核剔除的异常样本
"""

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models

NUM_CLASSES = 8
CLASS_LABELS = [
    "fresh", "diseased", "bruised", "rotten",
    "insect_damaged", "cracked", "wrinkled", "black_spot"
]
CLASS_LABELS_CN = {
    "fresh": "合格", "diseased": "病变", "bruised": "碰伤",
    "rotten": "腐烂", "insect_damaged": "虫伤", "cracked": "裂果",
    "wrinkled": "褶皱", "black_spot": "黑斑",
}

RISK_HIGH = 0.50
RISK_MEDIUM = 0.70


def get_model(model_path):
    """加载单模型（EfficientNet-B0 + Dropout）。"""
    model = models.efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )
    checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint.get('class_names', CLASS_LABELS)


def load_ensemble_models(model_dir):
    """加载 K-Fold 集成模型。"""
    model_dir = Path(model_dir)
    fold_paths = sorted(model_dir.glob('best_model_fold*.pth'))
    if not fold_paths:
        raise FileNotFoundError(
            f"No fold models found in {model_dir}. "
            "Run: python train_kfold.py --n-splits 5 --epochs 50"
        )

    models_list = []
    class_names = None
    for p in fold_paths:
        model, cls = get_model(p)
        models_list.append(model)
        class_names = cls
    return models_list, class_names


def ensemble_predict(models_list, image_path, transform, class_names, device):
    """对单张图执行集成推理，返回详细统计。"""
    image = Image.open(image_path).convert('RGB')
    tensor = transform(image).unsqueeze(0).to(device)

    all_probs = []
    all_preds = []
    with torch.no_grad():
        for model in models_list:
            model = model.to(device)
            model.eval()
            output = model(tensor)
            probs = torch.softmax(output, dim=1).cpu().numpy()[0]
            all_probs.append(probs)
            all_preds.append(int(probs.argmax()))

    all_probs = np.array(all_probs)  # (n_models, n_classes)
    avg_probs = np.mean(all_probs, axis=0)
    pred_idx = int(avg_probs.argmax())
    pred_class = class_names[pred_idx]
    confidence = float(avg_probs[pred_idx])

    # Top-2 margin: 第一名概率 - 第二名概率
    sorted_probs = np.sort(avg_probs)[::-1]
    margin = float(sorted_probs[0] - sorted_probs[1])

    # 分歧度：各模型预测不一致的比例
    unique_preds, counts = np.unique(all_preds, return_counts=True)
    disagreement = 1.0 - (counts.max() / len(all_preds))

    # 各 fold 的独立预测
    fold_details = []
    for i, (probs, pred) in enumerate(zip(all_probs, all_preds)):
        fold_details.append({
            'fold': i,
            'prediction': CLASS_LABELS_CN.get(class_names[pred], class_names[pred]),
            'confidence': round(float(probs[pred]), 4),
        })

    # 风险等级
    if confidence < RISK_HIGH or disagreement > 0.4:
        risk = 'HIGH'
    elif confidence < RISK_MEDIUM or disagreement > 0.0:
        risk = 'MEDIUM'
    else:
        risk = 'LOW'

    return {
        'file': Path(image_path).name,
        'prediction': CLASS_LABELS_CN.get(pred_class, pred_class),
        'prediction_en': pred_class,
        'confidence': round(confidence, 4),
        'margin': round(margin, 4),
        'disagreement': round(disagreement, 4),
        'risk': risk,
        'fold_details': fold_details,
        'top3': [
            {
                'class': CLASS_LABELS_CN.get(class_names[idx], class_names[idx]),
                'prob': round(float(avg_probs[idx]), 4),
            }
            for idx in np.argsort(avg_probs)[::-1][:3]
        ],
    }


def run_screening(input_dir, output_dir, model_dir, device):
    """Stage 1: AI 粗筛，生成审核报告。"""
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading ensemble models from: {model_dir}")
    models_list, class_names = load_ensemble_models(model_dir)
    print(f"Loaded {len(models_list)} models. Classes: {class_names}")

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])

    # 收集图片
    exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_files = sorted([f for f in input_dir.iterdir()
                         if f.suffix.lower() in exts])
    print(f"Found {len(image_files)} images in {input_dir}")

    results = []
    for img_path in image_files:
        try:
            info = ensemble_predict(models_list, img_path, transform, class_names, device)
            results.append(info)
            flag = "⚠️" if info['risk'] == 'HIGH' else ""
            print(f"  {info['file']:50s} -> {info['prediction']:8s} "
                  f"({info['confidence']*100:5.1f}%)  risk={info['risk']} {flag}")
        except Exception as e:
            print(f"  ERROR processing {img_path.name}: {e}")
            results.append({
                'file': img_path.name,
                'error': str(e),
            })

    # 按风险等级 + 置信度排序（HIGH 在前，便于人工优先审阅）
    def sort_key(r):
        if 'error' in r:
            return (3, 0)
        risk_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
        return (risk_order.get(r['risk'], 2), -r['confidence'])

    results.sort(key=sort_key)

    # 统计
    total = len(results)
    high_risk = sum(1 for r in results if r.get('risk') == 'HIGH')
    medium_risk = sum(1 for r in results if r.get('risk') == 'MEDIUM')
    low_risk = sum(1 for r in results if r.get('risk') == 'LOW')
    errors = sum(1 for r in results if 'error' in r)

    summary = {
        'total': total,
        'high_risk': high_risk,
        'medium_risk': medium_risk,
        'low_risk': low_risk,
        'errors': errors,
        'models_used': len(models_list),
        'threshold_high': RISK_HIGH,
        'threshold_medium': RISK_MEDIUM,
        'results': results,
    }

    # 保存 JSON 报告
    json_path = output_dir / 'screening_report.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\nReport saved: {json_path}")

    # 保存 CSV 报告（便于 Excel 审阅）
    csv_path = output_dir / 'screening_report.csv'
    with open(csv_path, 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.writer(f)
        writer.writerow([
            '文件名', '预测类别', '置信度', 'Top2差距', '分歧度',
            '风险等级', 'Top1', 'Top2', 'Top3', '审核结果', '备注'
        ])
        for r in results:
            if 'error' in r:
                writer.writerow([r['file'], 'ERROR', '', '', '', '', '', '', '', '', r['error']])
                continue
            top3_str = ' | '.join([f"{t['class']}({t['prob']*100:.1f}%)" for t in r['top3']])
            tops = r['top3']
            writer.writerow([
                r['file'],
                r['prediction'],
                f"{r['confidence']*100:.1f}%",
                f"{r['margin']:.3f}",
                f"{r['disagreement']:.2f}",
                r['risk'],
                tops[0]['class'] if len(tops) > 0 else '',
                tops[1]['class'] if len(tops) > 1 else '',
                tops[2]['class'] if len(tops) > 2 else '',
                '',  # 审核结果：人工填写 PASS / REJECT
                '',  # 备注
            ])
    print(f"CSV  report saved: {csv_path}")

    # 打印摘要
    print(f"\n{'='*50}")
    print(f"Screening Summary")
    print(f"{'='*50}")
    print(f"Total:    {total}")
    print(f"HIGH:     {high_risk}  ← 建议优先人工复核")
    print(f"MEDIUM:   {medium_risk}")
    print(f"LOW:      {low_risk}")
    print(f"Errors:   {errors}")
    print(f"\nNext: Open {csv_path.name} in Excel, fill '审核结果' column")
    print(f"      with PASS or REJECT, then run --review.")


def run_review(report_path, input_dir, output_dir):
    """Stage 2: 读取人工审核结果，分拣通过/剔除样本。"""
    report_path = Path(report_path)
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    # 支持 JSON 或 CSV 格式的审核结果
    if report_path.suffix.lower() == '.csv':
        reviewed = _load_reviewed_csv(report_path)
    else:
        with open(report_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        reviewed = {
            r['file']: r.get('review_result', '').strip().upper()
            for r in data.get('results', [])
            if 'error' not in r
        }

    passed_dir = output_dir / 'passed'
    rejected_dir = output_dir / 'rejected'
    passed_dir.mkdir(parents=True, exist_ok=True)
    rejected_dir.mkdir(parents=True, exist_ok=True)

    passed_list = []
    rejected_list = []
    unknown_list = []

    for fname, result in reviewed.items():
        src = input_dir / fname
        if not src.exists():
            print(f"  WARNING: Source file not found: {src}")
            unknown_list.append(fname)
            continue

        if result == 'PASS':
            dst = passed_dir / fname
            shutil.copy2(str(src), str(dst))
            passed_list.append(fname)
        elif result == 'REJECT':
            dst = rejected_dir / fname
            shutil.copy2(str(src), str(dst))
            rejected_list.append(fname)
        else:
            unknown_list.append(fname)

    # 保存清单
    manifest = {
        'passed': passed_list,
        'rejected': rejected_list,
        'unknown': unknown_list,
        'counts': {
            'passed': len(passed_list),
            'rejected': len(rejected_list),
            'unknown': len(unknown_list),
        },
    }
    manifest_path = output_dir / 'review_manifest.json'
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*50}")
    print(f"Review Complete")
    print(f"{'='*50}")
    print(f"Passed:   {len(passed_list):4d} -> {passed_dir}")
    print(f"Rejected: {len(rejected_list):4d} -> {rejected_dir}")
    print(f"Unknown:  {len(unknown_list):4d} (审核结果未标记或无效)")
    print(f"Manifest: {manifest_path}")


def _load_reviewed_csv(csv_path):
    """从 CSV 读取审核结果（支持 '审核结果' 或 'review_result' 列）。"""
    reviewed = {}
    with open(csv_path, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fname = row.get('文件名') or row.get('file')
            result = row.get('审核结果') or row.get('review_result') or ''
            if fname:
                reviewed[fname] = result.strip().upper()
    return reviewed


def main():
    parser = argparse.ArgumentParser(description='Batch Screening: AI coarse filter + Human review')
    parser.add_argument('--screen', action='store_true',
                        help='Stage 1: Run AI screening and generate reports')
    parser.add_argument('--review', action='store_true',
                        help='Stage 2: Apply human review results and sort files')
    parser.add_argument('--input', type=str, default='../data/unlabeled',
                        help='Input directory containing images to screen')
    parser.add_argument('--output', type=str, default='../outputs/screening',
                        help='Output directory for reports and sorted files')
    parser.add_argument('--model-dir', type=str, default='../models',
                        help='Directory containing K-Fold models (best_model_fold*.pth)')
    parser.add_argument('--report', type=str, default=None,
                        help='Path to reviewed report (JSON or CSV) for --review mode')
    args = parser.parse_args()

    if not args.screen and not args.review:
        parser.print_help()
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if args.screen:
        run_screening(args.input, args.output, args.model_dir, device)

    if args.review:
        if args.report is None:
            # 默认尝试读取 JSON，其次 CSV
            default_json = Path(args.output) / 'screening_report.json'
            default_csv = Path(args.output) / 'screening_report.csv'
            if default_json.exists():
                args.report = str(default_json)
            elif default_csv.exists():
                args.report = str(default_csv)
            else:
                print("Error: --review requires --report or a default report in --output")
                return
        run_review(args.report, args.input, args.output)


if __name__ == '__main__':
    main()
