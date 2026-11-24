# Email feature export: fit on training data and reuse the transformer for the saved splits.

import json
import logging
import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from .common import SPLITS, manifest, publish_report, pipeline_main
from .email_features import EmailFeatures


# Input stage: list the task’s saved splits and external inputs when needed.
def inputs(root):
    paths = {name: root / 'data/splits/emails' / f'{name}.parquet' for name in SPLITS}
    return paths


# Verification stage: check feature columns, finite values and split row alignment.
def verify(root):
    directory = root / 'data/features/emails'
    report = json.loads((directory / 'feature_report.json').read_text())
    assert report['inputs'] == manifest(inputs(root)), 'Input splits changed'
    schema = json.loads((root / 'models/feature_engineering/emails_schema.json').read_text())
    for name, path in inputs(root).items():
        expected = pd.read_parquet(path, columns=['record_id', 'label'])
        rows = pd.read_parquet(directory / f'{name}_rows.parquet')
        matrix = sparse.load_npz(directory / f'{name}.npz')
        assert sparse.isspmatrix_csr(matrix)
        assert matrix.shape == (len(rows), schema['n_features'])
        assert matrix.dtype == np.float32 and np.isfinite(matrix.data).all()
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected.reset_index(drop=True))
        assert rows.label.isin([0, 1]).all()
        assert len(rows) == report['inputs'][name]['rows']
    fitted = joblib.load(root / 'models/feature_engineering/email_transformer.joblib')
    assert fitted.schema() == schema
    return dict(
        dataset='emails',
        row_alignment=True,
        finite_features=True,
        consistent_columns=True,
        split_rows={n: v['rows'] for n, v in report['inputs'].items()},
    )


# Feature stage: export this task’s features or verify a completed run.
def run(root, seed=42):
    directory = root / 'data/features/emails'
    models = root / 'models/feature_engineering'
    report_path = directory / 'feature_report.json'
    if report_path.exists():
        verify(root)
        logging.info('emails already complete and valid; skipped')
        return
    paths = inputs(root)
    original = manifest(paths)
    preprocessing = json.loads((root / 'data/processed/emails/cleaning_report.json').read_text())
    expected = preprocessing['split_sizes']
    assert all(original[n]['rows'] == expected[n] for n in SPLITS), 'Incomplete input splits'
    directory.mkdir(parents=True, exist_ok=True)
    models.mkdir(parents=True, exist_ok=True)
    transformer = EmailFeatures()
    for name, path in paths.items():
        frame = pd.read_parquet(
            path, columns=['record_id', 'label', 'text', 'subject', 'url_count']
        )
        matrix = (
            transformer.fit_transform(frame) if name == 'train' else transformer.transform(frame)
        )
        assert matrix.shape[0] == len(frame) and np.isfinite(matrix.data).all()
        sparse.save_npz(directory / f'{name}.npz', matrix, compressed=True)
        frame[['record_id', 'label']].to_parquet(directory / f'{name}_rows.parquet', index=False)
        del matrix
        del frame
    schema = transformer.schema()
    joblib.dump(transformer, models / 'email_transformer.joblib', compress=3)
    (models / 'emails_schema.json').write_text(json.dumps(schema, indent=2) + '\n')
    assert manifest(paths) == original, 'Input files changed during feature extraction'
    report = dict(
        seed=seed,
        inputs=original,
        fitted_on="train only",
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
        'Email feature export: fit on training data and reuse the transformer for the saved splits.',
    )


if __name__ == '__main__':
    main()
