# Email training: compare Logistic Regression and Linear SVM using the saved email splits.

import json
import joblib
import numpy as np
import pandas as pd
from . import common
from zipfile import ZipFile
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


# Configuration stage: create the retained models with their original parameters.
def model_configs(seed=42):
    configurations = {}
    for weight in (None, 'balanced'):
        suffix = weight or 'none'
        configurations[f'logistic_regression_{suffix}'] = LogisticRegression(
            solver='liblinear', max_iter=1000, random_state=seed, class_weight=weight
        )
    for weight in (None, 'balanced'):
        suffix = weight or 'none'
        configurations[f'linear_svm_{suffix}'] = LinearSVC(
            max_iter=10000, dual='auto', random_state=seed, class_weight=weight
        )
    return configurations


# Input stage: list the files needed for this task’s training and validation.
def input_files(root):
    files = [
        root / 'data/features/emails/feature_report.json',
        root / 'models/feature_engineering/emails_schema.json',
    ]
    files.append(root / 'models/feature_engineering/email_transformer.joblib')
    for part in common.PARTS:
        files.append(root / 'data/features/emails' / f'{part}.npz')
        files.append(root / 'data/features/emails' / f'{part}_rows.parquet')
        files.append(root / 'data/splits/emails' / f'{part}.parquet')
    return files


# Provenance stage: record input sizes and timestamps to detect changes.
def fingerprint(root):
    return common.fingerprint(root, input_files(root))


# Validation stage: check feature schemas, row alignment and input file identities.
def preflight(root):
    directory = root / 'data/features/emails'
    report = json.loads((directory / 'feature_report.json').read_text())
    assert report['validation']['row_alignment'] and report['validation']['finite_features']
    schema = json.loads((root / 'models/feature_engineering/emails_schema.json').read_text())
    assert report['schema'] == schema
    for part in common.PARTS:
        split = root / 'data/splits/emails' / f'{part}.parquet'
        saved = report['inputs'][part]
        assert (
            split.stat().st_size == saved['size'] and split.stat().st_mtime_ns == saved['mtime_ns']
        )
        expected = pd.read_parquet(split, columns=['record_id', 'label'])
        rows = pd.read_parquet(directory / f'{part}_rows.parquet')
        with ZipFile(directory / f'{part}.npz') as archive:
            shape = np.load(archive.open('shape.npy'), allow_pickle=False)
        assert tuple(shape) == (len(rows), schema['n_features'])
        pd.testing.assert_frame_equal(rows, expected)
        common.validate_labels(rows.label.to_numpy())
        assert len(rows) == saved['rows'] and rows['record_id'].is_unique
    return fingerprint(root)


# Loading stage: read training or validation features and labels only.
def load_partition(root, part):
    if part not in common.PARTS:
        raise ValueError('Only train and validation may be loaded')
    directory = root / 'data/features/emails'
    x = sparse.load_npz(directory / f'{part}.npz')
    assert sparse.isspmatrix_csr(x) and np.isfinite(x.data).all()
    y = pd.read_parquet(directory / f'{part}_rows.parquet', columns=['label']).label.to_numpy()
    common.validate_labels(y)
    assert len(y) == x.shape[0]
    return x, y


# Verification stage: check this task’s saved artifacts and reports.
def verify_kind(root):
    transformer = joblib.load(root / 'models/feature_engineering/email_transformer.joblib')
    sample = transformer.transform(
        pd.DataFrame(
            [
                {
                    'text': 'Verify your account https://example.test/login',
                    'subject': 'Verify',
                    'url_count': 1,
                }
            ]
        )
    )
    return common.verify_saved(root, 'email', sample, fingerprint(root))


# Comparison stage: evaluate saved candidates on this task’s validation split.
def compare(root):
    return common.compare_models(
        root, 'email', verify=verify_kind, fingerprint=fingerprint, load_partition=load_partition
    )


# Training stage: run this task’s models with its own readers and checks.
def train(root, seed=42):
    return common.train_kind(
        root,
        'email',
        seed,
        model_configs(seed),
        preflight=preflight,
        load_partition=load_partition,
        verify=verify_kind,
        fingerprint=fingerprint,
    )


# Command stage: read options and run this file’s operation.
def main(argv=None):
    common.training_main('email', train, verify_kind, compare, argv)


if __name__ == '__main__':
    main()
