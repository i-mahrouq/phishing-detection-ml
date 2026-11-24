# Tests: check email features behavior using saved artifacts or temporary inputs.

import json
import numpy as np
import pandas as pd
import pytest
from src.features.email_features import EmailFeatures
from src.features import email_pipeline as pipeline


# Test stage: email training only and inference.
def test_email_training_only_and_inference():
    training = pd.DataFrame(
        {
            'text': [
                'normal project meeting',
                'verify account urgent!',
                'project meeting tomorrow',
                'verify account password',
            ],
            'subject': ['meeting', 'urgent', 'tomorrow', 'verify'],
            'url_count': [0, 1, 0, 1],
        }
    )
    a, b = EmailFeatures(min_df=1), EmailFeatures(min_df=1)
    x, y = a.fit_transform(training), b.fit_transform(training)
    assert x.shape == y.shape and (x != y).nnz == 0
    vocabulary = dict(a.vectorizer.vocabulary_)
    scale = a.scaler.max_abs_.copy()
    new = pd.DataFrame(
        {'text': ['validationonlytoken https://evil.test/login !!!'], 'subject': ['new']}
    )
    z = a.transform(new)
    assert z.shape == (1, x.shape[1]) and np.isfinite(z.data).all()
    assert 'validationonlytoken' not in a.vectorizer.vocabulary_
    assert a.vectorizer.vocabulary_ == vocabulary
    np.testing.assert_array_equal(scale, a.scaler.max_abs_)
    assert any(' ' in token for token in vocabulary)
    with pytest.raises(RuntimeError):
        a.fit_transform(new)


# Test stage: saved alignment and skip.
def test_saved_alignment_and_skip(tmp_path):
    splits = tmp_path / 'data/splits/emails'
    processed = tmp_path / 'data/processed/emails'
    splits.mkdir(parents=True)
    processed.mkdir(parents=True)
    for name in ['train', 'validation', 'test']:
        frame = pd.DataFrame(
            {
                'record_id': [name + 'b', name + 'a'],
                'label': [1, 0],
                'text': ['verify account project', 'normal account project'],
                'subject': ['', ''],
                'url_count': [1, 0],
            }
        )
        frame.to_parquet(splits / f'{name}.parquet', index=False)
    report = {'split_sizes': {n: 2 for n in ['train', 'validation', 'test']}}
    (processed / 'cleaning_report.json').write_text(json.dumps(report))
    pipeline.run(tmp_path)
    checks = pipeline.verify(tmp_path)
    assert checks['row_alignment'] and checks['consistent_columns']
    marker = tmp_path / 'data/features/emails/feature_report.json'
    before = marker.stat().st_mtime_ns
    pipeline.run(tmp_path)
    assert marker.stat().st_mtime_ns == before
