import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import torch
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

from deepfake_detector import predict_deepfake_from_file
from model import AudioCNN

SUPPORTED_EXTS = {'.wav', '.mp3', '.ogg', '.flac', '.webm'}


def infer_bucket(path: Path, label: str) -> str:
    name = path.name.lower()
    if label == 'human':
        if 'whatsapp' in name:
            return 'human_whatsapp'
        if 'voice_preview' in name:
            return 'human_voice_preview'
        if 'live_scan' in name:
            return 'human_live_scan'
        if 'live_recording' in name:
            return 'human_live_recording'
        return 'human_other'

    if 'deepclone_hard_negative' in name:
        return 'ai_deepclone_hard_negative'
    if 'azure_clone' in name:
        return 'ai_azure_clone'
    if 'feedback_' in name:
        return 'ai_feedback_clone'
    return 'ai_other'


def group_files(files: list[Path], label: str) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in files:
        grouped[infer_bucket(path, label)].append(path)
    return dict(grouped)


def select_mixed_sample(grouped: dict[str, list[Path]], target_size: int, seed: int) -> list[Path]:
    rng = random.Random(seed)
    buckets = {name: items[:] for name, items in grouped.items() if items}
    for items in buckets.values():
        rng.shuffle(items)

    selected: list[Path] = []
    while len(selected) < target_size and buckets:
        empty = []
        for bucket_name in sorted(buckets.keys()):
            items = buckets[bucket_name]
            if not items:
                empty.append(bucket_name)
                continue
            selected.append(items.pop())
            if len(selected) >= target_size:
                break
        for bucket_name in empty:
            buckets.pop(bucket_name, None)
        for bucket_name in list(buckets.keys()):
            if not buckets[bucket_name]:
                buckets.pop(bucket_name, None)
    return selected


def evaluate_sample(model, device, threshold: float, files: list[Path], true_label: int) -> list[dict]:
    rows = []
    for path in files:
        result = predict_deepfake_from_file(str(path), model, device, threshold=threshold)
        pred_label = 1 if result['alert'] else 0
        rows.append(
            {
                'path': str(path),
                'bucket': infer_bucket(path, 'ai' if true_label == 1 else 'human'),
                'true_label': true_label,
                'pred_label': pred_label,
                'synthetic_probability': float(result['synthetic_probability']),
                'threshold': float(result['threshold']),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Run a mixed-format benchmark for Vacha-Shield.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--sample-size-per-class', type=int, default=30)
    args = parser.parse_args()

    base = Path(__file__).resolve().parent
    calibration_path = base / 'model_calibration.json'
    threshold = 0.5
    if calibration_path.exists():
        threshold = float(json.loads(calibration_path.read_text(encoding='utf-8')).get('threshold', 0.5))

    human_files = [
        p for p in sorted((base / 'continuous_learning_dataset' / 'human').glob('*'))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS
    ]
    ai_files = [
        p for p in sorted((base / 'continuous_learning_dataset' / 'ai').glob('*'))
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS and p.stat().st_size > 0
    ]

    human_grouped = group_files(human_files, 'human')
    ai_grouped = group_files(ai_files, 'ai')

    target = min(args.sample_size_per_class, len(human_files), len(ai_files))
    selected_humans = select_mixed_sample(human_grouped, target, seed=args.seed)
    selected_ai = select_mixed_sample(ai_grouped, target, seed=args.seed + 1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = AudioCNN(num_classes=1).to(device)
    model.load_state_dict(torch.load(base / 'model.pth', map_location=device))
    model.eval()

    rows = []
    rows.extend(evaluate_sample(model, device, threshold, selected_humans, true_label=0))
    rows.extend(evaluate_sample(model, device, threshold, selected_ai, true_label=1))

    y_true = [row['true_label'] for row in rows]
    y_pred = [row['pred_label'] for row in rows]
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='binary', zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    bucket_summary: dict[str, dict[str, int]] = {}
    for row in rows:
        bucket = row['bucket']
        bucket_summary.setdefault(bucket, {'count': 0, 'correct': 0})
        bucket_summary[bucket]['count'] += 1
        if row['true_label'] == row['pred_label']:
            bucket_summary[bucket]['correct'] += 1

    false_positives = [row for row in rows if row['true_label'] == 0 and row['pred_label'] == 1]
    false_negatives = [row for row in rows if row['true_label'] == 1 and row['pred_label'] == 0]

    summary = {
        'benchmark': 'mixed_all_conditions',
        'threshold': threshold,
        'sample_size_per_class': target,
        'total_samples': len(rows),
        'human_buckets': {name: len(paths) for name, paths in human_grouped.items()},
        'ai_buckets': {name: len(paths) for name, paths in ai_grouped.items()},
        'selected_bucket_counts': {name: sum(1 for row in rows if row['bucket'] == name) for name in sorted(bucket_summary)},
        'bucket_accuracy': {
            name: round(stats['correct'] / stats['count'], 4)
            for name, stats in sorted(bucket_summary.items())
        },
        'accuracy': round(float(acc), 4),
        'precision_ai': round(float(precision), 4),
        'recall_ai': round(float(recall), 4),
        'f1_ai': round(float(f1), 4),
        'confusion_matrix': {
            'true_human_pred_human': int(cm[0][0]),
            'true_human_pred_ai': int(cm[0][1]),
            'true_ai_pred_human': int(cm[1][0]),
            'true_ai_pred_ai': int(cm[1][1]),
        },
        'false_positive_count': len(false_positives),
        'false_negative_count': len(false_negatives),
        'false_positives': false_positives[:5],
        'false_negatives': false_negatives[:5],
    }

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
