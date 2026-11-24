# URL training: compare scaled Logistic Regression and Random Forest using the saved URL splits.

import json
import joblib
import numpy as np
import pandas as pd
from . import common
import pyarrow.parquet as pq
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.features.url_features import URLFeatures, URL_COLUMNS


# Configuration stage: create the retained models with their original parameters.
def model_configs(seed=42):
    return {
        'logistic_regression': Pipeline(
            [
                ('scaler', StandardScaler()),
                ('classifier', LogisticRegression(max_iter=1000, random_state=seed)),
            ]
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=100, min_samples_leaf=3, n_jobs=2, random_state=seed
        ),
    }


# Input stage: list the files needed for this task’s training and validation.
def input_files(root):
    files = [
        root / 'data/features/urls/feature_report.json',
        root / 'models/feature_engineering/urls_schema.json',
    ]
    for part in common.PARTS:
        files.append(root / 'data/features/urls' / f'{part}.parquet')
        files.append(root / 'data/splits/urls' / f'{part}.parquet')
    return files


# Provenance stage: record input sizes and timestamps to detect changes.
def fingerprint(root):
    return common.fingerprint(root, input_files(root))


# Validation stage: check feature schemas, row alignment and input file identities.
def preflight(root):
    directory = root / 'data/features/urls'
    report = json.loads((directory / 'feature_report.json').read_text())
    assert report['validation']['row_alignment'] and report['validation']['finite_features']
    schema = json.loads((root / 'models/feature_engineering/urls_schema.json').read_text())
    assert report['schema'] == schema
    for part in common.PARTS:
        split = root / 'data/splits/urls' / f'{part}.parquet'
        saved = report['inputs'][part]
        assert (
            split.stat().st_size == saved['size'] and split.stat().st_mtime_ns == saved['mtime_ns']
        )
        expected = pd.read_parquet(split, columns=['url_key', 'label'])
        path = directory / f'{part}.parquet'
        assert pq.read_schema(path).names == list(URL_COLUMNS) + ['url_key', 'label']
        rows = pd.read_parquet(path, columns=['url_key', 'label'])
        pd.testing.assert_frame_equal(rows, expected)
        common.validate_labels(rows.label.to_numpy())
        assert len(rows) == saved['rows'] and rows['url_key'].is_unique
    return fingerprint(root)


# Loading stage: read training or validation features and labels only.
def load_partition(root, part):
    if part not in common.PARTS:
        raise ValueError('Only train and validation may be loaded')
    directory = root / 'data/features/urls'
    frame = pd.read_parquet(directory / f'{part}.parquet')
    x = frame[list(URL_COLUMNS)]
    y = frame.label.to_numpy()
    assert np.isfinite(x.to_numpy()).all()
    common.validate_labels(y)
    assert len(y) == x.shape[0]
    return x, y


# Verification stage: check this task’s saved artifacts and reports.
def verify_kind(root):
    sample = URLFeatures().transform(['https://example.test/login?account=1'])
    return common.verify_saved(root, 'url', sample, fingerprint(root))


# Comparison stage: evaluate saved candidates on this task’s validation split.
def compare(root):
    return common.compare_models(
        root, 'url', verify=verify_kind, fingerprint=fingerprint, load_partition=load_partition
    )


# Training stage: run this task’s models with its own readers and checks.
def train(root, seed=42):
    return common.train_kind(
        root,
        'url',
        seed,
        model_configs(seed),
        preflight=preflight,
        load_partition=load_partition,
        verify=verify_kind,
        fingerprint=fingerprint,
    )


# Command stage: read options and run this file’s operation.
def main(argv=None):
    common.training_main('url', train, verify_kind, compare, argv)


if __name__ == '__main__':
    main()
