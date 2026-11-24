# URL evaluation: score the frozen test split and measure recall on external phishing addresses.

import json
import logging
import time
import numpy as np
import pandas as pd
from src.features.url_features import URL_COLUMNS
from src.training.train_url import verify_kind as verify_training
from . import common


# External evaluation stage: measure detected and missed phishing URLs only.
def external_metrics(y, values, threshold):
    assert len(y) > 0 and np.all(y == 1), 'External dataset must be phishing-only'
    detected = int(np.count_nonzero(values >= threshold))
    return dict(
        samples=len(y),
        phishing_recall=detected / len(y),
        detected_phishing_urls=detected,
        missed_phishing_urls=len(y) - detected,
    )


# Provenance stage: record the model, threshold and test inputs that must remain unchanged.
def frozen_paths(root):
    paths = [
        root / 'models/url/selected_model.joblib',
        root / 'models/url/selected_threshold.json',
        root / 'reports/training/url_model_comparison.json',
        root / 'models/feature_engineering/urls_schema.json',
        root / 'data/features/urls/feature_report.json',
        root / 'data/features/urls/test.parquet',
    ]
    paths.append(root / 'data/features/urls/external.parquet')
    return {
        str(p.relative_to(root)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in paths
    }


# Loading stage: read permitted held-out features and verify their row alignment.
def load_data(root, part='test'):
    if part not in ('test', 'external'):
        raise ValueError('URL final evaluation uses test or external only')
    directory = root / 'data/features/urls'
    report = json.loads((directory / 'feature_report.json').read_text())
    schema = json.loads((root / 'models/feature_engineering/urls_schema.json').read_text())
    assert schema == report['schema']
    original = root / (
        'data/processed/urls/phishtank_external.parquet'
        if part == 'external'
        else 'data/splits/urls/test.parquet'
    )
    metadata = report['inputs'][part]
    assert (
        original.stat().st_size == metadata['size']
        and original.stat().st_mtime_ns == metadata['mtime_ns']
    )
    columns = ['url_key', 'label', 'source', 'hostname']
    original_rows = pd.read_parquet(original, columns=columns)
    frame = pd.read_parquet(directory / f'{part}.parquet')
    assert frame.columns.tolist() == list(URL_COLUMNS) + ['url_key', 'label']
    x = frame[list(URL_COLUMNS)]
    rows = frame[['url_key', 'label']]
    assert np.isfinite(x.to_numpy()).all()
    pd.testing.assert_frame_equal(
        rows.reset_index(drop=True), original_rows[['url_key', 'label']].reset_index(drop=True)
    )
    assert x.shape[0] == len(rows) == metadata['rows'] and rows['url_key'].is_unique
    y = rows.label.to_numpy()
    assert set(np.unique(y)) == ({1} if part == 'external' else {0, 1})
    return x, y, original_rows


# Leakage check: compare this task’s training and test groups.
def overlap(root, rows):
    return common.check_overlap(root, 'urls', rows, ['url_key', 'hostname'])


# Verification stage: check saved evaluation evidence without changing it.
def verify(root):
    common.verify_summary(root, 'url', frozen_paths(root))
    common.verify_binary(root, 'url')
    report = json.loads((root / 'reports/evaluation/phishtank_external_metrics.json').read_text())
    config = json.loads((root / 'models/url/selected_threshold.json').read_text())
    assert common.finite(report)
    assert (
        report['threshold'] == config['threshold'] and report['selected_model'] == config['model']
    )
    assert report['detected_phishing_urls'] + report['missed_phishing_urls'] == report['samples']
    assert (
        not {
            'accuracy',
            'phishing_precision',
            'roc_auc',
            'average_precision',
            'false_positive_rate',
        }
        & report.keys()
    )
    return dict(completed=True, models_and_thresholds_unchanged=True, reports_verified=True)


# Evaluation stage: score frozen held-out data or verify completed results.
def run(root):
    directory = root / 'reports/evaluation'
    if (directory / 'url_evaluation_summary.json').exists() or (
        directory / 'evaluation_summary.json'
    ).exists():
        logging.info('Existing evaluation verified: %s', verify(root))
        return
    verify_training(root)
    before = frozen_paths(root)
    outputs = ['url_test_metrics.json', 'url_test_errors.parquet', 'url_confusion_matrix.png']
    outputs.append('phishtank_external_metrics.json')
    out = common.start(root, 'url', before, outputs)
    model, config = common.load_selected(root, 'url')
    x, y, rows = load_data(root)
    assert model.n_features_in_ == x.shape[1]
    started = time.perf_counter()
    values, score_type = common.scores(model, x)
    elapsed = time.perf_counter() - started
    assert score_type == config['score_type']
    result = common.binary_metrics(y, values, config['threshold'])
    result.update(
        selected_model=config['model'],
        threshold=config['threshold'],
        score_type=score_type,
        inference_seconds=elapsed,
        inference_ms_per_sample=1000 * elapsed / len(y),
        timing_scope='model scoring only; excludes loading and feature extraction',
        overlap_checks=overlap(root, rows),
        label_mapping={'0': 'legitimate', '1': 'phishing'},
    )
    result['by_source'] = {
        str(source): common.binary_metrics(y[indices], values[indices], config['threshold'])
        for source, indices in rows.groupby('source').indices.items()
    }
    result['limitations'] = json.loads(
        (root / 'data/features/urls/feature_report.json').read_text()
    ).get('input_limitations', [])
    flags = []
    if any(result['overlap_checks'].values()):
        flags.append('url: detected training/test overlap; investigate')
    if result['average_precision'] >= 0.99 or result['roc_auc'] >= 0.99:
        flags.append(
            'url: very high ranking metrics warrant source/leakage review, not proof of leakage'
        )
    assert common.finite(result)
    common.save_errors(out, 'url_test', y, values, config['threshold'])
    common.plot_matrix(out, 'url', result['confusion_matrix'])
    common.write_json(out / 'url_test_metrics.json', result)
    results = {'url': result}
    del x, y, values, rows
    report_files = ['url_test_metrics.json']
    x, y, rows = load_data(root, 'external')
    assert model.n_features_in_ == x.shape[1]
    values, _ = common.scores(model, x)
    external = external_metrics(y, values, config['threshold'])
    external.update(
        selected_model=config['model'], threshold=config['threshold'], score_type=score_type
    )
    checks = {}
    for part in ('train', 'validation', 'test'):
        primary = pd.read_parquet(
            root / 'data/splits/urls' / f'{part}.parquet', columns=['url_key', 'hostname']
        )
        for column in ('url_key', 'hostname'):
            checks[f'{part}_external_{column}_overlap'] = len(
                set(primary[column]) & set(rows[column])
            )
    external['overlap_checks'] = checks
    if any(checks.values()):
        flags.append('External URLs overlap internal data; review independence')
    common.write_json(out / 'phishtank_external_metrics.json', external)
    results['external'] = external
    report_files.append('phishtank_external_metrics.json')
    assert before == frozen_paths(root), 'Frozen inputs changed during evaluation'
    common.finish(root, 'url', before, results, flags, report_files)
    verify(root)


# Command stage: read options and run this file’s operation.
def main(argv=None):
    common.evaluation_main(
        run,
        verify,
        'URL evaluation: score the frozen test split and measure recall on external phishing addresses.',
        argv,
    )


if __name__ == '__main__':
    main()
