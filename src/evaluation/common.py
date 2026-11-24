# Shared evaluation: calculate classification metrics and check saved evaluation evidence.

import argparse
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    average_precision_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits
from src.training.common import scores, write_json


# Evaluation stage: calculate class metrics and the confusion matrix at the saved threshold.
def binary_metrics(y, values, threshold):
    predicted = (values >= threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel().tolist()
    both = len(np.unique(y)) == 2
    return dict(
        samples=len(y),
        accuracy=float(accuracy_score(y, predicted)),
        phishing_precision=float(precision_score(y, predicted, zero_division=0)),
        phishing_recall=float(recall_score(y, predicted, zero_division=0)),
        phishing_f1=float(f1_score(y, predicted, zero_division=0)),
        average_precision=float(average_precision_score(y, values)) if both else None,
        roc_auc=float(roc_auc_score(y, values)) if both else None,
        confusion_matrix=dict(TN=tn, FP=fp, FN=fn, TP=tp),
        false_positive_rate=fp / (fp + tn) if fp + tn else None,
        false_negative_rate=fn / (fn + tp) if fn + tp else None,
        ranking_metrics_defined=both,
    )


# Validation stage: check nested results for nonfinite numbers.
def finite(value):
    if isinstance(value, dict):
        return all(finite(v) for v in value.values())
    if isinstance(value, list):
        return all(finite(v) for v in value)
    if isinstance(value, (int, float)):
        return bool(np.isfinite(value))
    return True


# Loading stage: check the selected model, saved threshold and training report agree.
def load_selected(root, kind):
    directory = root / 'models' / kind
    config = json.loads((directory / 'selected_threshold.json').read_text())
    comparison = json.loads(
        (root / 'reports/training' / f'{kind}_model_comparison.json').read_text()
    )
    assert config['model'] == comparison['selected_model']
    assert config['threshold'] == comparison['selected_threshold']
    assert (directory / 'selected_model.joblib').resolve() == (root / config['artifact']).resolve()
    assert config['positive_label'] == 1 and config['prediction_rule'] == 'score >= threshold'
    assert np.isfinite(config['threshold'])
    model = joblib.load(directory / 'selected_model.joblib')
    assert list(model.classes_) == [0, 1]
    expected = 'probability' if hasattr(model, 'predict_proba') else 'uncalibrated_margin'
    assert config['score_type'] == expected
    if expected == 'probability':
        assert hasattr(model, 'predict_proba')
    else:
        assert hasattr(model, 'decision_function')
    return model, config


# Reporting stage: save misclassified row positions and scores without raw content.
def save_errors(out, name, y, values, threshold):
    predicted = (values >= threshold).astype(np.int8)
    bad = predicted != y

    pd.DataFrame(
        dict(
            split_row=np.flatnonzero(bad),
            actual_label=y[bad],
            predicted_label=predicted[bad],
            score=values[bad],
            error_type=np.where(y[bad] == 0, 'false_positive', 'false_negative'),
        )
    ).to_parquet(out / f'{name}_errors.parquet', index=False)


# Reporting stage: draw and save the confusion matrix.
def plot_matrix(out, name, matrix):
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay

    values = np.array([[matrix['TN'], matrix['FP']], [matrix['FN'], matrix['TP']]])
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay(values, display_labels=['Legitimate', 'Phishing']).plot(
        ax=axis, cmap='Blues', values_format='d', colorbar=False
    )
    axis.set_title(f'{name.capitalize()} test — saved threshold')
    figure.tight_layout()
    figure.savefig(out / f'{name}_confusion_matrix.png', dpi=150)
    plt.close(figure)


# Leakage check: compare training and test identities for the supplied grouping columns.
def check_overlap(root, plural, rows, columns):
    train = pd.read_parquet(root / 'data/splits' / plural / 'train.parquet', columns=columns)
    overlaps = {}
    for column in columns:
        overlaps[f'train_test_{column}_overlap'] = len(set(train[column]) & set(rows[column]))
    return overlaps


# Verification stage: compare frozen input metadata with the saved evaluation summary.
def verify_summary(root, kind, current):
    out = root / 'reports/evaluation'
    path = out / f'{kind}_evaluation_summary.json'
    if not path.exists():
        path = out / 'evaluation_summary.json'
    summary = json.loads(path.read_text())
    assert summary['completed']
    assert all(
        summary['frozen_artifacts'].get(path) == metadata for path, metadata in current.items()
    )


# Verification stage: check saved metrics, confusion plots and error counts.
def verify_binary(root, kind):
    out = root / 'reports/evaluation'
    report = json.loads((out / f'{kind}_test_metrics.json').read_text())
    config = json.loads((root / 'models' / kind / 'selected_threshold.json').read_text())
    assert finite(report)
    assert (
        report['threshold'] == config['threshold'] and report['selected_model'] == config['model']
    )
    assert sum(report['confusion_matrix'].values()) == report['samples']
    assert (out / f'{kind}_confusion_matrix.png').is_file()
    errors = pd.read_parquet(out / f'{kind}_test_errors.parquet', columns=['error_type'])
    assert (errors.error_type == 'false_positive').sum() == report['confusion_matrix']['FP']
    assert (errors.error_type == 'false_negative').sum() == report['confusion_matrix']['FN']


# Preservation stage: refuse partial or existing outputs before starting evaluation.
def start(root, kind, before, outputs):
    out = root / 'reports/evaluation'
    out.mkdir(parents=True, exist_ok=True)
    marker = out / f'{kind}_evaluation_started.json'
    if marker.exists():
        raise RuntimeError(
            'Partial prior evaluation detected; inspect logs rather than automatically rerun'
        )
    if any((out / name).exists() for name in outputs):
        raise RuntimeError(
            'Existing evaluation outputs without the expected summary; refusing to overwrite evidence'
        )
    write_json(marker, {'frozen_artifacts': before})
    return out


# Reporting stage: save the completed evaluation summary and review flags.
def finish(root, kind, before, results, flags, files):
    summary_fields = ('samples', 'accuracy', 'phishing_recall', 'phishing_f1', 'average_precision')
    summaries = {}
    for name, result in results.items():
        summaries[name] = {}
        for field in summary_fields:
            if field in result:
                summaries[name][field] = result[field]
    write_json(
        root / 'reports/evaluation' / f'{kind}_evaluation_summary.json',
        dict(
            completed=True,
            frozen_artifacts=before,
            report_files=files,
            review_flags=flags,
            models_changed=False,
            thresholds_changed=False,
            source_analysis_note='Single-class source groups have undefined ranking metrics and rates set to null.',
            privacy='Error artifacts contain split row positions, labels and scores only; no raw content or URLs.',
            results=summaries,
        ),
    )


# Command stage: run held-out evaluation or verify saved evidence.
def evaluation_main(run, verify, description, argv=None):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args(argv)
    with threadpool_limits(limits=2):
        if args.verify:
            print(json.dumps(verify(args.root)))
        else:
            run(args.root)
