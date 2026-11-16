"""Run once per dataset; verify outputs without fitting or changing input splits."""
import argparse
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
from .email_features import EmailFeatures
from .url_features import URLFeatures, URL_COLUMNS

SPLITS = ('train', 'validation', 'test')


def inputs(root, kind):
    paths = {name: root/'data/splits'/kind/f'{name}.parquet' for name in SPLITS}
    if kind == 'urls':
        paths['external'] = root/'data/processed/urls/phishtank_external.parquet'
    return paths


def manifest(paths):
    return {name: dict(path=str(path), size=path.stat().st_size,
                       mtime_ns=path.stat().st_mtime_ns, rows=pq.read_metadata(path).num_rows)
            for name, path in paths.items()}


def verify_kind(root, kind):
    directory = root/'data/features'/kind
    report = json.loads((directory/'feature_report.json').read_text())
    assert report['inputs'] == manifest(inputs(root, kind)), 'Input splits changed'
    schema = json.loads((root/'models/feature_engineering'/f'{kind}_schema.json').read_text())
    for name, path in inputs(root, kind).items():
        identity = 'record_id' if kind == 'emails' else 'url_key'
        expected = pd.read_parquet(path, columns=[identity, 'label'])
        if kind == 'emails':
            rows = pd.read_parquet(directory/f'{name}_rows.parquet')
            matrix = sparse.load_npz(directory/f'{name}.npz')
            assert sparse.isspmatrix_csr(matrix)
            assert matrix.shape == (len(rows), schema['n_features'])
            assert matrix.dtype == np.float32 and np.isfinite(matrix.data).all()
        else:
            features = pd.read_parquet(directory/f'{name}.parquet')
            rows = features[[identity, 'label']]
            assert features.columns.tolist() == list(URL_COLUMNS) + [identity, 'label']
            assert np.isfinite(features[list(URL_COLUMNS)].to_numpy()).all()
        pd.testing.assert_frame_equal(rows.reset_index(drop=True), expected.reset_index(drop=True))
        assert rows.label.isin([0, 1]).all()
        assert len(rows) == report['inputs'][name]['rows']
    if kind == 'emails':
        fitted = joblib.load(root/'models/feature_engineering/email_transformer.joblib')
        assert fitted.schema() == schema
    return dict(dataset=kind, row_alignment=True, finite_features=True,
                consistent_columns=True, split_rows={n: v['rows'] for n,v in report['inputs'].items()})


def run_kind(root, kind, seed=42):
    directory = root/'data/features'/kind
    models = root/'models/feature_engineering'
    report_path = directory/'feature_report.json'
    if report_path.exists():
        verify_kind(root, kind)
        logging.info('%s already complete and valid; skipped', kind)
        return
    paths = inputs(root, kind)
    original = manifest(paths)
    # Require preprocessing success markers and matching per-split row counts.
    preprocessing = json.loads((root/'data/processed'/kind/'cleaning_report.json').read_text())
    expected = preprocessing['split_sizes'] if kind == 'emails' else preprocessing['validation']['split_sizes']
    assert all(original[n]['rows'] == expected[n] for n in SPLITS), 'Incomplete input splits'
    directory.mkdir(parents=True, exist_ok=True); models.mkdir(parents=True, exist_ok=True)
    transformer = EmailFeatures() if kind == 'emails' else URLFeatures()
    for name, path in paths.items():
        logging.info('%s/%s: extracting features', kind, name)
        if kind == 'emails':
            frame = pd.read_parquet(path, columns=['record_id','label','text','subject','url_count'])
            # The only fitting branch in the full workflow is the training split.
            matrix = transformer.fit_transform(frame) if name == 'train' else transformer.transform(frame)
            assert matrix.shape[0] == len(frame) and np.isfinite(matrix.data).all()
            sparse.save_npz(directory/f'{name}.npz', matrix, compressed=True)
            frame[['record_id','label']].to_parquet(directory/f'{name}_rows.parquet', index=False)
            logging.info('%s/%s: sparse shape=%s nnz=%s', kind, name, matrix.shape, matrix.nnz)
            del matrix
        else:
            frame = pd.read_parquet(path, columns=['url_original','url_key','label'])
            features = transformer.transform(frame.url_original)
            features['url_key'] = frame.url_key.to_numpy()
            features['label'] = frame.label.to_numpy()
            features.to_parquet(directory/f'{name}.parquet', index=False)
            del features
        del frame
    schema = transformer.schema()
    if kind == 'emails':
        joblib.dump(transformer, models/'email_transformer.joblib', compress=3)
    (models/f'{kind}_schema.json').write_text(json.dumps(schema, indent=2)+'\n')
    assert manifest(paths) == original, 'Input files changed during feature extraction'
    report = dict(seed=seed, inputs=original, fitted_on='train only' if kind == 'emails' else 'none (stateless)',
                  predictive_identifiers=False, schema=schema,
                  input_limitations=preprocessing.get('limitations', []))
    # Publish the success marker only after output verification.
    temporary = directory/'feature_report.json'
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    try:
        validation = verify_kind(root, kind)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    report['validation'] = validation
    temporary.write_text(json.dumps(report, indent=2)+'\n')
    logging.info('%s: complete; %s', kind, validation)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    for kind in ('emails', 'urls'):
        if args.verify:
            print(json.dumps(verify_kind(args.root, kind)))
        else:
            run_kind(args.root, kind, args.seed)

if __name__ == '__main__':
    main()
