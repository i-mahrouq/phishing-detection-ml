# Shared training: fit candidates, compare validation scores and save models without overwriting evidence.

import argparse
import hashlib
import json
import logging
from pathlib import Path
import platform
import time
import warnings
import joblib
import numpy as np
import sklearn
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from threadpoolctl import threadpool_limits

PARTS = ('train', 'validation')


# Saving stage: write JSON through a temporary file before replacing the destination.
def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')
    temp.replace(path)


# Validation stage: require both legitimate and phishing classes.
def validate_labels(labels):
    classes = set(np.unique(labels))
    assert classes == {0, 1}, 'Both binary classes (0 legitimate, 1 phishing) required'


# Scoring stage: return class-1 probabilities or uncalibrated decision margins.
def scores(model, x):
    if hasattr(model, 'predict_proba'):
        index = list(model.classes_).index(1)
        result = model.predict_proba(x)[:, index]
        score_type = 'probability'
    else:
        assert list(model.classes_) == [0, 1]
        result = model.decision_function(x)
        score_type = 'uncalibrated_margin'
    assert np.isfinite(result).all()
    return np.asarray(result), score_type


# Evaluation stage: calculate classification and ranking metrics at a fixed threshold.
def metrics(y, values, threshold):
    predicted = (values >= threshold).astype(np.int8)
    result = dict(
        accuracy=float(accuracy_score(y, predicted)),
        phishing_precision=float(precision_score(y, predicted, zero_division=0)),
        phishing_recall=float(recall_score(y, predicted, zero_division=0)),
        phishing_f1=float(f1_score(y, predicted, zero_division=0)),
        average_precision=float(average_precision_score(y, values)),
        roc_auc=float(roc_auc_score(y, values)),
        confusion_matrix=confusion_matrix(y, predicted, labels=[0, 1]).tolist(),
    )
    assert all(np.isfinite(result[k]) for k in result if k != 'confusion_matrix')
    return result


# Selection stage: maximize validation F1 and prefer recall when scores tie.
def tune_threshold(y, values):
    precision, recall, thresholds = precision_recall_curve(y, values)
    numerator = 2 * precision[:-1] * recall[:-1]
    denominator = np.maximum(precision[:-1] + recall[:-1], np.finfo(float).tiny)
    f1 = numerator / denominator
    candidates = np.flatnonzero(f1 == f1.max())
    chosen = candidates[np.argmax(recall[candidates])]
    return float(thresholds[chosen])


# Selection stage: rank candidates by validation average precision, F1 and recall.
def selection_key(record):
    validation = record['validation_default_threshold']
    return validation['average_precision'], validation['phishing_f1'], validation['phishing_recall']


# Provenance stage: record input sizes and timestamps to detect changes.
def fingerprint(root, files):
    inputs = {}
    for path in files:
        details = path.stat()
        inputs[str(path.relative_to(root))] = {
            'size': details.st_size,
            'mtime_ns': details.st_mtime_ns,
        }
    return inputs


# Training stage: fit one candidate on training data, score validation data and save it.
def fit_candidate(root, kind, name, model, train_x, train_y, valid_x, valid_y):
    logging.info('%s/%s: fitting training only', kind, name)
    started = time.perf_counter()
    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - started
    started = time.perf_counter()
    values, score_type = scores(model, valid_x)
    inference_seconds = time.perf_counter() - started
    default = 0.0 if score_type == 'uncalibrated_margin' else 0.5
    record = dict(
        name=name,
        artifact=f'models/{kind}/{name}.joblib',
        score_type=score_type,
        default_threshold=default,
        training_seconds=fit_seconds,
        validation_inference_seconds=inference_seconds,
        validation_inference_ms_per_row=inference_seconds / len(valid_y) * 1000,
        validation_default_threshold=metrics(valid_y, values, default),
        warnings=[str(w.message) for w in captured],
    )
    joblib.dump(model, root / record['artifact'], compress=3)
    logging.info(
        '%s/%s: validation AP=%.6f',
        kind,
        name,
        record['validation_default_threshold']['average_precision'],
    )
    return record


# Training stage: compare candidates and save the validation-selected model and threshold.
def train_kind(root, kind, seed, configurations, *, preflight, load_partition, verify, fingerprint):
    report_path = root / 'reports/training' / f'{kind}_model_comparison.json'
    if report_path.exists():
        verify(root)
        logging.info('%s completed previously; skipping', kind)
        return
    provenance = preflight(root)
    directory = root / 'models' / kind

    marker = directory / 'training_started.json'
    if marker.exists():
        raise RuntimeError(
            f'Previous {kind} run started without a success report; inspect logs, no automatic restart'
        )
    protected = [directory / f'{name}.joblib' for name in configurations]
    protected += [directory / 'selected_model.joblib', directory / 'selected_threshold.json']
    if any(path.exists() or path.is_symlink() for path in protected):
        raise RuntimeError(
            'Existing model artifacts without a success report; use a fresh root, no automatic overwrite'
        )
    directory.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, {'seed': seed, 'inputs': provenance})
    train_x, train_y = load_partition(root, 'train')
    valid_x, valid_y = load_partition(root, 'validation')
    records = []
    for name, model in configurations.items():
        record = fit_candidate(root, kind, name, model, train_x, train_y, valid_x, valid_y)
        records.append(record)

    best = max(records, key=selection_key)
    selected = joblib.load(root / best['artifact'])
    values, score_type = scores(selected, valid_x)
    threshold = tune_threshold(valid_y, values)
    tuned = metrics(valid_y, values, threshold)
    alias = directory / 'selected_model.joblib'
    alias.symlink_to(Path(best['artifact']).name)
    write_json(
        directory / 'selected_threshold.json',
        dict(
            model=best['name'],
            artifact=best['artifact'],
            threshold=threshold,
            score_type=score_type,
            positive_label=1,
            prediction_rule='score >= threshold',
            tuned_on='validation only',
            objective='maximum phishing F1; ties prefer recall',
            validation_metrics=tuned,
        ),
    )
    assert fingerprint(root) == provenance, 'Inputs changed during training'
    report = dict(
        seed=seed,
        task=kind,
        configurations=records,
        selected_model=best['name'],
        selection='validation average precision, then default-threshold phishing F1, then recall',
        selected_threshold=threshold,
        selected_validation_metrics=tuned,
        inputs=provenance,
        classes={'0': 'legitimate', '1': 'phishing'},
        train_rows=len(train_y),
        validation_rows=len(valid_y),
        n_features=train_x.shape[1],
        test_evaluated=False,
        external_evaluated=False,
        feature_transformers_modified=False,
        inherited_limitations=json.loads(
            (root / 'data/features' / (kind + 's') / 'feature_report.json').read_text()
        ).get('input_limitations', []),
        versions={'python': platform.python_version(), 'sklearn': sklearn.__version__},
    )
    write_json(report_path, report)
    logging.info('%s complete; selected %s, threshold %.6g', kind, best['name'], threshold)


# Verification stage: check saved training reports and candidate predictions.
def verify_saved(root, kind, sample, inputs):
    report = json.loads((root / 'reports/training' / f'{kind}_model_comparison.json').read_text())
    assert report['inputs'] == inputs
    assert not report['test_evaluated'] and not report['external_evaluated']
    model = joblib.load(root / 'models' / kind / 'selected_model.joblib')
    threshold = json.loads((root / 'models' / kind / 'selected_threshold.json').read_text())
    assert threshold['model'] == report['selected_model'] and np.isfinite(threshold['threshold'])
    value, score_type = scores(model, sample)
    assert score_type == threshold['score_type'] and value.shape == (1,)
    for record in report['configurations']:
        restored = joblib.load(root / record['artifact'])
        assert scores(restored, sample)[0].shape == (1,)
        for metric in record['validation_default_threshold'].values():
            assert np.isfinite(np.asarray(metric)).all()
    return {
        'task': kind,
        'selected_model': report['selected_model'],
        'inference_verified': True,
        'test_evaluated': False,
    }


# Comparison stage: score saved candidates on validation data without fitting or tuning.
def compare_models(root, kind, *, verify, fingerprint, load_partition):
    verify(root)
    report = json.loads((root / 'reports/training' / f'{kind}_model_comparison.json').read_text())
    before = fingerprint(root)
    x, y = load_partition(root, 'validation')
    records = []
    for saved in report['configurations']:
        path = root / saved['artifact']
        with path.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        model = joblib.load(path)
        started = time.perf_counter()
        values, score_type = scores(model, x)
        elapsed = time.perf_counter() - started
        records.append(
            dict(
                name=saved['name'],
                artifact=saved['artifact'],
                artifact_sha256=digest,
                score_type=score_type,
                threshold=saved['default_threshold'],
                metrics=metrics(y, values, saved['default_threshold']),
                training_seconds=saved['training_seconds'],
                training_time_source='Historical training report for this artifact path; not remeasured',
                validation_inference_seconds=elapsed,
                validation_inference_ms_per_row=1000 * elapsed / len(y),
            )
        )
    assert before == fingerprint(root), 'Inputs changed during validation comparison'
    return dict(
        task=kind,
        partition='validation',
        samples=len(y),
        configurations=records,
        thresholds='Original default thresholds, unchanged; no tuning in this comparison',
        timing_scope='Model scoring only, excluding loading and feature extraction',
        models_fitted=False,
        test_evaluated=False,
        external_evaluated=False,
    )


# Command stage: train, verify or compare the requested task’s models.
def training_main(kind, train, verify, compare, argv=None):
    parser = argparse.ArgumentParser(
        description=f'Train/verify/compare V1 {kind} models independently'
    )
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=42)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        '--verify', action='store_true', help='Verify the existing task without fitting'
    )
    mode.add_argument(
        '--compare',
        action='store_true',
        help='Recompute saved candidate validation metrics without fitting',
    )
    parser.add_argument(
        '--output', type=Path, help='New comparison JSON only; refuses an existing file'
    )
    args = parser.parse_args(argv)
    if args.output and not args.compare:
        parser.error('--output requires --compare')
    root = args.root.resolve()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    try:
        with threadpool_limits(limits=2):
            if args.verify:
                print(json.dumps(verify(root)))
            elif args.compare:
                result = compare(root)
                payload = json.dumps(result, indent=2, allow_nan=False) + '\n'
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    with args.output.open('x') as stream:
                        stream.write(payload)
                else:
                    print(payload, end='')
            else:
                train(root, args.seed)
    except (OSError, ValueError, RuntimeError, AssertionError) as exc:
        parser.exit(2, f'Error: {exc}\n')
