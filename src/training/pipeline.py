"""Train six fixed configurations once; never open test or external datasets."""
import argparse
import json
import logging
from pathlib import Path
import platform
import time
import warnings
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import sparse
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from threadpoolctl import threadpool_limits
from src.features.url_features import URL_COLUMNS

KINDS = ('email', 'url')
PARTS = ('train', 'validation')


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')
    temp.replace(path)


def model_configs(kind, seed=42):
    if kind == 'email':
        return {f'{family}_{weight or "none"}': (
            LogisticRegression(solver='liblinear', max_iter=1000, random_state=seed, class_weight=weight)
            if family == 'logistic_regression' else
            LinearSVC(max_iter=10000, dual='auto', random_state=seed, class_weight=weight))
            for family in ('logistic_regression', 'linear_svm') for weight in (None, 'balanced')}
    return {
        'logistic_regression': Pipeline([('scaler', StandardScaler()),
            ('classifier', LogisticRegression(max_iter=1000, random_state=seed))]),
        'random_forest': RandomForestClassifier(n_estimators=100, min_samples_leaf=3,
                                              n_jobs=2, random_state=seed),
    }


def input_files(root, kind):
    plural = kind + 's'
    files = [root/'data/features'/plural/'feature_report.json',
             root/'models/feature_engineering'/f'{plural}_schema.json']
    if kind == 'email':
        files.append(root/'models/feature_engineering/email_transformer.joblib')
    for part in PARTS:
        files.append(root/'data/features'/plural/f'{part}.{"npz" if kind == "email" else "parquet"}')
        if kind == 'email':
            files.append(root/'data/features'/plural/f'{part}_rows.parquet')
        files.append(root/'data/splits'/plural/f'{part}.parquet')
    return files


def fingerprint(root, kind):
    return {str(p.relative_to(root)): {'size':p.stat().st_size, 'mtime_ns':p.stat().st_mtime_ns}
            for p in input_files(root, kind)}


def validate_labels(labels):
    assert set(np.unique(labels)) == {0, 1}, 'Both binary classes (0 legitimate, 1 phishing) required'


def preflight(root, kind):
    """Inspect only training/validation labels, IDs, metadata, and array headers."""
    plural = kind + 's'
    directory = root/'data/features'/plural
    report = json.loads((directory/'feature_report.json').read_text())
    assert report['validation']['row_alignment'] and report['validation']['finite_features']
    schema = json.loads((root/'models/feature_engineering'/f'{plural}_schema.json').read_text())
    assert report['schema'] == schema
    for part in PARTS:
        identity = 'record_id' if kind == 'email' else 'url_key'
        split = root/'data/splits'/plural/f'{part}.parquet'
        saved = report['inputs'][part]
        assert split.stat().st_size == saved['size'] and split.stat().st_mtime_ns == saved['mtime_ns']
        expected = pd.read_parquet(split, columns=[identity, 'label'])
        if kind == 'email':
            rows = pd.read_parquet(directory/f'{part}_rows.parquet')
            with ZipFile(directory/f'{part}.npz') as archive:
                shape = np.load(archive.open('shape.npy'), allow_pickle=False)
            assert tuple(shape) == (len(rows), schema['n_features'])
        else:
            path = directory/f'{part}.parquet'
            assert pq.read_schema(path).names == list(URL_COLUMNS) + [identity, 'label']
            rows = pd.read_parquet(path, columns=[identity, 'label'])
        pd.testing.assert_frame_equal(rows, expected)
        validate_labels(rows.label.to_numpy())
        assert len(rows) == saved['rows'] and rows[identity].is_unique
    return fingerprint(root, kind)


def load_partition(root, kind, part):
    if part not in PARTS:
        raise ValueError('Only train and validation may be loaded')
    directory = root/'data/features'/(kind+'s')
    if kind == 'email':
        x = sparse.load_npz(directory/f'{part}.npz')
        assert sparse.isspmatrix_csr(x) and np.isfinite(x.data).all()
        y = pd.read_parquet(directory/f'{part}_rows.parquet', columns=['label']).label.to_numpy()
    else:
        frame = pd.read_parquet(directory/f'{part}.parquet')
        x = frame[list(URL_COLUMNS)]
        y = frame.label.to_numpy()
        assert np.isfinite(x.to_numpy()).all()
    validate_labels(y)
    assert len(y) == x.shape[0]
    return x, y


def scores(model, x):
    """Return continuous scores toward phishing (class 1), plus their native type.

    Linear SVM margins are uncalibrated; Random Forest class estimates are not
    guaranteed real-world probabilities. Thresholding remains a separate step.
    """
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


def metrics(y, values, threshold):
    predicted = (values >= threshold).astype(np.int8)
    result = dict(phishing_precision=float(precision_score(y,predicted,zero_division=0)),
                  phishing_recall=float(recall_score(y,predicted,zero_division=0)),
                  phishing_f1=float(f1_score(y,predicted,zero_division=0)),
                  average_precision=float(average_precision_score(y,values)),
                  roc_auc=float(roc_auc_score(y,values)),
                  confusion_matrix=confusion_matrix(y,predicted,labels=[0,1]).tolist())
    assert all(np.isfinite(result[k]) for k in result if k != 'confusion_matrix')
    return result


def tune_threshold(y, values):
    """Choose validation-only maximum phishing F1; prefer recall on ties.

    Training and test labels must never be supplied here: using held-out test
    outcomes for threshold selection would leak evaluation information.
    """
    precision, recall, thresholds = precision_recall_curve(y, values)
    f1 = 2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1], np.finfo(float).tiny)
    candidates = np.flatnonzero(f1 == f1.max())
    chosen = candidates[np.argmax(recall[candidates])]
    return float(thresholds[chosen])


def selection_key(record):
    m = record['validation_default_threshold']
    return m['average_precision'], m['phishing_f1'], m['phishing_recall']


def train_kind(root, kind, seed):
    report_path = root/'reports/training'/f'{kind}_model_comparison.json'
    if report_path.exists():
        verify_kind(root, kind)
        logging.info('%s completed previously; skipping', kind)
        return
    provenance = preflight(root, kind)
    directory = root/'models'/kind
    # A marker survives failures: never silently repeat a partially trained run.
    marker = directory/'training_started.json'
    if marker.exists():
        raise RuntimeError(f'Previous {kind} run started without a success report; inspect logs, no automatic restart')
    directory.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(marker, {'seed':seed, 'inputs':provenance})
    train_x, train_y = load_partition(root,kind,'train')
    valid_x, valid_y = load_partition(root,kind,'validation')
    records = []
    for name, model in model_configs(kind, seed).items():
        logging.info('%s/%s: fitting training only', kind, name)
        started = time.perf_counter()
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            model.fit(train_x, train_y)
        fit_seconds = time.perf_counter()-started
        started = time.perf_counter()
        values, score_type = scores(model,valid_x)
        inference_seconds = time.perf_counter()-started
        default = 0.0 if score_type == 'uncalibrated_margin' else 0.5
        record = dict(name=name, artifact=f'models/{kind}/{name}.joblib', score_type=score_type,
                      default_threshold=default, training_seconds=fit_seconds,
                      validation_inference_seconds=inference_seconds,
                      validation_inference_ms_per_row=inference_seconds/len(valid_y)*1000,
                      validation_default_threshold=metrics(valid_y,values,default),
                      warnings=[str(w.message) for w in captured])
        joblib.dump(model, root/record['artifact'], compress=3)
        records.append(record)
        logging.info('%s/%s: validation AP=%.6f', kind,name,record['validation_default_threshold']['average_precision'])
        del model, values
    best = max(records, key=selection_key)
    selected = joblib.load(root/best['artifact'])
    values, score_type = scores(selected,valid_x)
    threshold = tune_threshold(valid_y, values)
    tuned = metrics(valid_y,values,threshold)
    alias = directory/'selected_model.joblib'
    alias.symlink_to(Path(best['artifact']).name)  # No duplicate model bytes.
    write_json(directory/'selected_threshold.json', dict(model=best['name'], artifact=best['artifact'],
               threshold=threshold, score_type=score_type, positive_label=1,
               prediction_rule='score >= threshold', tuned_on='validation only',
               objective='maximum phishing F1; ties prefer recall', validation_metrics=tuned))
    assert fingerprint(root,kind) == provenance, 'Inputs changed during training'
    report = dict(seed=seed, task=kind, configurations=records, selected_model=best['name'],
                  selection='validation average precision, then default-threshold phishing F1, then recall',
                  selected_threshold=threshold, selected_validation_metrics=tuned,
                  inputs=provenance, classes={'0':'legitimate','1':'phishing'},
                  train_rows=len(train_y), validation_rows=len(valid_y), n_features=train_x.shape[1],
                  test_evaluated=False, external_evaluated=False,
                  feature_transformers_modified=False,
                  inherited_limitations=json.loads((root/'data/features'/(kind+'s')/'feature_report.json').read_text()).get('input_limitations', []),
                  versions={'python':platform.python_version(),'sklearn':sklearn.__version__})
    write_json(report_path,report)
    logging.info('%s complete; selected %s, threshold %.6g', kind,best['name'],threshold)


def verify_kind(root, kind):
    report = json.loads((root/'reports/training'/f'{kind}_model_comparison.json').read_text())
    assert report['inputs'] == fingerprint(root,kind)
    assert not report['test_evaluated'] and not report['external_evaluated']
    model = joblib.load(root/'models'/kind/'selected_model.joblib')
    threshold = json.loads((root/'models'/kind/'selected_threshold.json').read_text())
    assert threshold['model'] == report['selected_model'] and np.isfinite(threshold['threshold'])
    if kind == 'email':
        transformer = joblib.load(root/'models/feature_engineering/email_transformer.joblib')
        sample = transformer.transform(pd.DataFrame([{'text':'Verify your account https://example.test/login', 'subject':'Verify', 'url_count':1}]))
    else:
        from src.features.url_features import URLFeatures
        sample = URLFeatures().transform(['https://example.test/login?account=1'])
    value, score_type = scores(model,sample)
    assert score_type == threshold['score_type'] and value.shape == (1,)
    for record in report['configurations']:
        restored = joblib.load(root/record['artifact'])
        assert scores(restored,sample)[0].shape == (1,)
        for key, metric in record['validation_default_threshold'].items():
            assert np.isfinite(np.asarray(metric)).all()
    return {'task':kind,'selected_model':report['selected_model'],'inference_verified':True,'test_evaluated':False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--verify',action='store_true')
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
    with threadpool_limits(limits=2):
        if args.verify:
            summary=json.loads((args.root/'reports/training/training_summary.json').read_text())
            assert summary['completed']
            print(json.dumps([verify_kind(args.root,k) for k in KINDS]))
        else:
            for kind in KINDS:
                train_kind(args.root,kind,args.seed)
            checks=[verify_kind(args.root,k) for k in KINDS]
            write_json(args.root/'reports/training/training_summary.json',
                       dict(completed=True,seed=args.seed,validation=checks,test_evaluated=False,external_evaluated=False))

if __name__ == '__main__':
    main()
