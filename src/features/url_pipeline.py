# URL feature export: calculate the same ordered indicators for internal and external data.

import json
import logging
import numpy as np
import pandas as pd
from .common import SPLITS, manifest, publish_report, pipeline_main
from .url_features import URLFeatures, URL_COLUMNS


# Input stage: list the task’s saved splits and external inputs when needed.
def inputs(root):
    paths = {name: root / 'data/splits/urls' / f'{name}.parquet' for name in SPLITS}
    paths['external'] = root / 'data/processed/urls/phishtank_external.parquet'
    return paths


# Verification stage: check feature columns, finite values and split row alignment.
def verify(root):
    directory = root / 'data/features/urls'
    report = json.loads((directory / 'feature_report.json').read_text())
    assert report['inputs'] == manifest(inputs(root)), 'Input splits changed'
    schema = json.loads((root / 'models/feature_engineering/urls_schema.json').read_text())
    for name, path in inputs(root).items():
        expected = pd.read_parquet(path, columns=['url_key', 'label'])
        features = pd.read_parquet(directory / f'{name}.parquet')
        rows = features[['url_key', 'label']]
        assert features.columns.tolist() == list(URL_COLUMNS) + ['url_key', 'label']
        assert np.isfinite(features[list(URL_COLUMNS)].to_numpy()).all()
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected.reset_index(drop=True))
        assert rows.label.isin([0, 1]).all()
        assert len(rows) == report['inputs'][name]['rows']
    return dict(
        dataset='urls',
        row_alignment=True,
        finite_features=True,
        consistent_columns=True,
        split_rows={n: v['rows'] for n, v in report['inputs'].items()},
    )


# Feature stage: export this task’s features or verify a completed run.
def run(root, seed=42):
    directory = root / 'data/features/urls'
    models = root / 'models/feature_engineering'
    report_path = directory / 'feature_report.json'
    if report_path.exists():
        verify(root)
        logging.info('urls already complete and valid; skipped')
        return
    paths = inputs(root)
    original = manifest(paths)
    preprocessing = json.loads((root / 'data/processed/urls/cleaning_report.json').read_text())
    expected = preprocessing['validation']['split_sizes']
    assert all(original[n]['rows'] == expected[n] for n in SPLITS), 'Incomplete input splits'
    directory.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    transformer = URLFeatures()
    for name, path in paths.items():
        frame = pd.read_parquet(path, columns=['url_original', 'url_key', 'label'])
        features = transformer.transform(frame.url_original)
        features['url_key'] = frame.url_key.to_numpy()
        features['label'] = frame.label.to_numpy()
        features.to_parquet(directory / f'{name}.parquet', index=False)
        del features
        del frame
    schema = transformer.schema()
    (models / 'urls_schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    assert manifest(paths) == original, 'Input files changed during feature extraction'
    report = dict(
        seed=seed,
        inputs=original,
        fitted_on="none (stateless)",
        predictive_identifiers=False,
        schema=schema,
        input_limitations=preprocessing.get('limitations', []),
    )
    publish_report(report_path, report, lambda: verify(root))


# Command stage: read options and run this file’s operation.
def main():
    pipeline_main(
        run,
        verify,
        'URL feature export: calculate the same ordered indicators for internal and external data.',
    )


if __name__ == '__main__':
    main()
