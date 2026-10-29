"""Audit saved CLINC150 predictions using only the Python standard library.

Run from any working directory. This checks decisions and point estimates;
it does not rerun training, forward passes, calibration or bootstrap intervals.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
TEAM = ROOT / 'CLINC150_Team_20261019'
STUDY = ROOT / 'BERT_Rejection_Study_20261019'


def read_csv(path):
    """Read exported numeric text without pandas type inference."""
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def close(actual, expected, context):
    """Fail on a discrepancy beyond floating-point serialization error."""
    if not math.isclose(float(actual), float(expected), rel_tol=1e-9, abs_tol=1e-10):
        raise AssertionError(f'{context}: calculated {actual}, stored {expected}')


def boolean(value):
    """Require the experiment's explicit stored Boolean representation."""
    if value not in ('True', 'False'):
        raise ValueError(f'Invalid Boolean: {value!r}')
    return value == 'True'


def divide(numerator, denominator):
    """Use the recorded zero-division convention for precision and recall."""
    return numerator / denominator if denominator else 0.0


def macro_f1(truth, predictions, labels):
    """Compute the unweighted mean of per-class F1 from confusion counts."""
    true_counts, pred_counts = Counter(truth), Counter(predictions)
    correct = Counter(t for t, p in zip(truth, predictions) if t == p)
    return statistics.mean(divide(2 * correct[y], true_counts[y] + pred_counts[y]) for y in labels)


def ranking_metrics(truth, scores):
    """Compute tie-aware ROC AUC and non-interpolated average precision."""
    groups = defaultdict(lambda: [0, 0])
    for positive, score in zip(truth, scores):
        groups[score][0 if positive else 1] += 1
    positives, negatives = sum(truth), len(truth) - sum(truth)
    tp = fp = 0
    auc_wins = average_precision = 0.0
    for score in sorted(groups, reverse=True):
        pos, neg = groups[score]
        auc_wins += neg * (tp + 0.5 * pos)
        tp += pos
        fp += neg
        average_precision += divide(pos, positives) * divide(tp, tp + fp)
    return divide(auc_wins, positives * negatives), average_precision


def prediction_path(folder, row):
    """Resolve each historical comparison to the actual stored predictions."""
    seed = row['seed']
    if folder.name == STUDY.name:
        return folder / f"outputs/seed{seed}/{row['split']}_predictions.csv"
    arm, condition = row['arm'], row['condition']
    if folder.name == 'BERT_Typo_Augmentation_20261019' and arm == 'original':
        if condition == 'clean':
            return STUDY / f'outputs/seed{seed}/test_predictions.csv'
        return STUDY / f'typo_stress_test/outputs/seed{seed}/{condition}_predictions.csv'
    return folder / f'outputs/{arm}/seed{seed}/{condition}_predictions.csv'


def verify_frame(frame, raw, canonical, context):
    """Bind saved predictions to official labels, row order and shared inputs."""
    validation = len(frame) == 3100
    pairs = raw['val'] + raw['oos_val'] if validation else raw['test'] + raw['oos_test']
    assert len(frame) == len(pairs), context
    condition = frame[0].get('condition', 'clean')
    texts = [r[0] for r in pairs] if validation or condition == 'clean' else canonical[condition]
    for index, (row, pair, text) in enumerate(zip(frame, pairs, texts)):
        assert int(row['row_id']) == index, (context, index, 'row order')
        assert row['true_label'] == pair[1], (context, index, 'label')
        assert boolean(row['is_oos']) == (pair[1] == 'oos'), (context, index, 'scope')
        assert row['text'] == text, (context, index, 'text')


def verify_setting(frame, stored, context):
    """Recompute rejection, classification and OOS metrics for one setting."""
    score, policy = stored['score'], stored['policy']
    scores = [float(row[f'{score}_score']) for row in frame]
    rejects = [boolean(row[f'{score}_{policy}_reject']) for row in frame]
    threshold = float(stored['threshold'])
    assert all(reject == (value >= threshold) for reject, value in zip(rejects, scores)), (context, 'threshold')
    oos = [boolean(row['is_oos']) for row in frame]
    truth = [int(row['label_id']) for row in frame]
    predicted = [int(row['predicted_label_id']) for row in frame]
    n_oos, n_is = sum(oos), len(frame) - sum(oos)
    tp = sum(o and r for o, r in zip(oos, rejects))
    fp = sum(not o and r for o, r in zip(oos, rejects))
    accepted = n_is - fp
    correct_accepted = sum(not o and not r and t == p for o, r, t, p in zip(oos, rejects, truth, predicted))
    wrong_accepted = accepted - correct_accepted
    precision, recall = divide(tp, tp + fp), divide(tp, n_oos)
    auc, ap = ranking_metrics(oos, scores)
    in_truth = [t for t, o in zip(truth, oos) if not o]
    in_pred = [p for p, o in zip(predicted, oos) if not o]
    combined = [-1 if r else p for r, p in zip(rejects, predicted)]
    metrics = dict(oos_precision=precision, oos_recall=recall,
        oos_f1=divide(2 * precision * recall, precision + recall),
        oos_auroc=auc, oos_aupr_average_precision=ap,
        in_scope_acceptance=divide(accepted, n_is),
        in_scope_false_rejection_rate=divide(fp, n_is),
        in_scope_correct_and_accepted_rate=divide(correct_accepted, n_is),
        accepted_in_scope_accuracy=divide(correct_accepted, accepted),
        combined_151_accuracy=divide(correct_accepted + tp, len(frame)),
        combined_151_macro_f1=macro_f1(truth, combined, range(-1, 150)),
        oos_true_positive=tp, in_scope_false_rejections=fp, oos_missed=n_oos - tp,
        closed_set_accuracy=divide(sum(t == p for t, p in zip(in_truth, in_pred)), n_is),
        closed_set_macro_f1=macro_f1(in_truth, in_pred, range(150)),
        in_scope_wrong_and_accepted_rate=divide(wrong_accepted, n_is),
        oos_false_acceptance_rate=divide(n_oos - tp, n_oos))
    for name, value in metrics.items():
        if name in stored:
            close(value, stored[name], f'{context}/{name}')


def verify_aggregates(folder, metrics):
    """Check reported means and sample SD against per-seed point estimates."""
    summary = folder / 'outputs/summary_mean_sd.csv'
    if not summary.exists():
        return 0
    count = 0
    for row in read_csv(summary):
        keys = [k for k in ('arm', 'condition', 'split', 'score', 'policy', 'target_acceptance') if k in row and k in metrics[0]]
        group = [r for r in metrics if all(r[k] == row[k] for k in keys)]
        assert len(group) == 3, (summary, keys, len(group))
        if 'n_seeds' in row:
            assert int(row['n_seeds']) == 3
        for key, value in row.items():
            if key.endswith('_mean') and key[:-5] in group[0]:
                values = [float(r[key[:-5]]) for r in group]
                close(statistics.mean(values), value, f'{summary.name}/{key}')
                if key[:-5] + '_sd' in row:
                    close(statistics.stdev(values), row[key[:-5] + '_sd'], f'{summary.name}/{key[:-5]}_sd')
                count += 1
    return count


def verify_initial_classification():
    """Recompute historical BERT and TF-IDF accuracy/Macro-F1 from CSVs."""
    table = TEAM / 'outputs/multiseed/typo_classification_mean_sd.csv'
    n = 0
    for row in read_csv(table):
        if row['calibration'] != 'raw':
            continue
        model = row['model']
        seeds = (42, 43, 44) if model == 'bert' else (None,)
        accuracy, f1 = [], []
        for seed in seeds:
            label = f'seed{seed}' if seed else 'tfidf_lr'
            path = TEAM / f"outputs/robustness/multiseed/{label}/{row['condition']}_predictions.csv"
            frame = [r for r in read_csv(path) if not boolean(r['is_oos'])]
            assert len(frame) == 4500, path
            truth = [int(r['label_id']) for r in frame]
            pred = [int(r['predicted_label_id']) for r in frame]
            accuracy.append(sum(t == p for t, p in zip(truth, pred)) / 4500)
            f1.append(macro_f1(truth, pred, range(150)))
        close(statistics.mean(accuracy), row['accuracy_mean'], f'{model}/{row["condition"]}/accuracy')
        close(statistics.mean(f1), row['macro_f1_mean'], f'{model}/{row["condition"]}/f1')
        if len(seeds) > 1:
            close(statistics.stdev(accuracy), row['accuracy_sd'], 'accuracy_sd')
            close(statistics.stdev(f1), row['macro_f1_sd'], 'macro_f1_sd')
        n += 1
    return n


def main():
    """Audit all packaged principal point estimates and print a compact result."""
    raw = json.loads((TEAM / 'data/raw/data_full.json').read_text(encoding='utf-8'))
    expected = dict(train=15000, val=3000, test=4500, oos_train=100, oos_val=100, oos_test=1000)
    assert {k: len(v) for k, v in raw.items()} == expected
    assert len({r[1] for r in raw['train']}) == 150
    canonical = {c: [r['text'] for r in read_csv(TEAM / f'outputs/robustness/typo_data/{c}.csv')]
                 for c in ('typo_light', 'typo_medium')}
    settings = prediction_files = aggregate_values = 0
    for folder in sorted(ROOT.glob('BERT_*')):
        path = folder / 'outputs/metrics_by_seed.csv'
        if not path.exists():
            continue
        metrics = read_csv(path)
        grouped = defaultdict(list)
        for row in metrics:
            grouped[prediction_path(folder, row)].append(row)
        for prediction, group in grouped.items():
            frame = read_csv(prediction)
            verify_frame(frame, raw, canonical, str(prediction.relative_to(ROOT)))
            for row in group:
                verify_setting(frame, row, str(prediction.relative_to(ROOT)))
                settings += 1
            prediction_files += 1
        aggregate_values += verify_aggregates(folder, metrics)
        print(f'PASS {folder.name}: {len(metrics)} settings', flush=True)
    classification_groups = verify_initial_classification()
    print(json.dumps(dict(status='PASS', rejection_settings=settings,
        prediction_file_visits=prediction_files, aggregate_metric_checks=aggregate_values,
        historical_classification_groups=classification_groups,
        scope='Saved CSV decisions, point metrics and seed aggregates. No training/inference or uncertainty re-estimation.'), indent=2))


if __name__ == '__main__':
    main()
