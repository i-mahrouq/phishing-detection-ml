# Email evaluation: score the frozen test split using the saved model and threshold.

import json
import logging
import time
import numpy as np
import pandas as pd
from scipy import sparse
from src.training.train_email import verify_kind as verify_training
from . import common


# Provenance stage: record the model, threshold and test inputs that must remain unchanged.
def frozen_paths(root):
    paths = [
        root / 'models/email/selected_model.joblib',
        root / 'models/email/selected_threshold.json',
        root / 'reports/training/email_model_comparison.json',
        root / 'models/feature_engineering/emails_schema.json',
        root / 'data/features/emails/feature_report.json',
        root / 'data/features/emails/test.npz',
    ]
    paths += [
        root / 'models/feature_engineering/email_transformer.joblib',
        root / 'data/features/emails/test_rows.parquet',
    ]
    return {
        str(p.relative_to(root)): dict(size=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns)
        for p in paths
    }


# Loading stage: read permitted held-out features and verify their row alignment.
def load_data(root, part='test'):
    if part != 'test':
        raise ValueError('Email final evaluation uses test only')
    directory = root / 'data/features/emails'
    report = json.loads((directory / 'feature_report.json').read_text())
    schema = json.loads((root / 'models/feature_engineering/emails_schema.json').read_text())
    assert schema == report['schema']
    original = root / 'data/splits/emails/test.parquet'
    metadata = report['inputs'][part]
    assert (
        original.stat().st_size == metadata['size']
        and original.stat().st_mtime_ns == metadata['mtime_ns']
    )
    columns = ['record_id', 'label', 'source', 'exact_hash', 'duplicate_group']
    original_rows = pd.read_parquet(original, columns=columns)
    x = sparse.load_npz(directory / 'test.npz')
    rows = pd.read_parquet(directory / 'test_rows.parquet')
    assert sparse.isspmatrix_csr(x) and np.isfinite(x.data).all()
    assert x.shape[1] == schema['n_features']
    pd.testing.assert_frame_equal(
        rows.reset_index(drop=True), original_rows[['record_id', 'label']].reset_index(drop=True)
    )
    assert x.shape[0] == len(rows) == metadata['rows'] and rows['record_id'].is_unique
    y = rows.label.to_numpy()
    assert set(np.unique(y)) == {0, 1}
    return x, y, original_rows


# Leakage check: compare this task’s training and test groups.
def overlap(root, rows):
    return common.check_overlap(root, 'emails', rows, ['exact_hash', 'duplicate_group'])


# Verification stage: check saved evaluation evidence without changing it.
def verify(root):
    common.verify_summary(root, 'email', frozen_paths(root))
    common.verify_binary(root, 'email')
    return dict(completed=True, models_and_thresholds_unchanged=True, reports_verified=True)


# Evaluation stage: score frozen held-out data or verify completed results.
def run(root):
    directory = root / 'reports/evaluation'
    if (directory / 'email_evaluation_summary.json').exists() or (
        directory / 'evaluation_summary.json'
    ).exists():
        logging.info('Existing evaluation verified: %s', verify(root))
        return
    verify_training(root)
    before = frozen_paths(root)
    outputs = ['email_test_metrics.json', 'email_test_errors.parquet', 'email_confusion_matrix.png']
    out = common.start(root, 'email', before, outputs)
    model, config = common.load_selected(root, 'email')
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
        (root / 'data/features/emails/feature_report.json').read_text()
    ).get('input_limitations', [])
    flags = []
    if any(result['overlap_checks'].values()):
        flags.append('email: detected training/test overlap; investigate')
    if result['average_precision'] >= 0.99 or result['roc_auc'] >= 0.99:
        flags.append(
            'email: very high ranking metrics warrant source/leakage review, not proof of leakage'
        )
    assert common.finite(result)
    common.save_errors(out, 'email_test', y, values, config['threshold'])
    common.plot_matrix(out, 'email', result['confusion_matrix'])
    common.write_json(out / 'email_test_metrics.json', result)
    results = {'email': result}
    del x, y, values, rows
    report_files = ['email_test_metrics.json']
    assert before == frozen_paths(root), 'Frozen inputs changed during evaluation'
    common.finish(root, 'email', before, results, flags, report_files)
    verify(root)


# Command stage: read options and run this file’s operation.
def main(argv=None):
    common.evaluation_main(
        run,
        verify,
        'Email evaluation: score the frozen test split using the saved model and threshold.',
        argv,
    )


if __name__ == '__main__':
    main()
