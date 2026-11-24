# Tests: check url features behavior using saved artifacts or temporary inputs.

import json
import numpy as np
import pandas as pd
import pytest
from src.features.url_features import URLFeatures, URL_COLUMNS
from src.features import url_pipeline as pipeline


# Test stage: url inference columns and finiteness.
def test_url_inference_columns_and_finiteness():
    transformer = URLFeatures()
    values = [
        'https://login.example.test/a?x=1&y=2',
        'http://192.168.1.1/verify',
        'https://example.test/%zz',
    ]
    x = transformer.transform(values)
    assert x.shape == (3, 13) and x.columns.tolist() == list(URL_COLUMNS)
    assert np.isfinite(x.to_numpy()).all()
    assert x.loc[0, 'is_https'] == 1 and x.loc[1, 'hostname_is_ip'] == 1
    assert x.loc[0, 'query_parameter_count'] == 2
    pd.testing.assert_frame_equal(x, transformer.transform(values))
    assert not hasattr(transformer, 'fit')


# Test stage: saved alignment and skip.
def test_saved_alignment_and_skip(tmp_path):
    splits = tmp_path / 'data/splits/urls'
    processed = tmp_path / 'data/processed/urls'
    splits.mkdir(parents=True)
    processed.mkdir(parents=True)
    for name in ['train', 'validation', 'test']:
        frame = pd.DataFrame(
            {
                'url_key': [name + 'b', name + 'a'],
                'label': [1, 0],
                'url_original': ['https://login.evil.test/verify', 'https://normal.test/'],
            }
        )
        frame.to_parquet(splits / f'{name}.parquet', index=False)
    report = {'validation': {'split_sizes': {n: 2 for n in ['train', 'validation', 'test']}}}
    pd.DataFrame(
        {'url_key': ['external'], 'label': [1], 'url_original': ['https://outside.test/login']}
    ).to_parquet(processed / 'phishtank_external.parquet', index=False)
    (processed / 'cleaning_report.json').write_text(json.dumps(report))
    pipeline.run(tmp_path)
    checks = pipeline.verify(tmp_path)
    assert checks['row_alignment'] and checks['consistent_columns']
    marker = tmp_path / 'data/features/urls/feature_report.json'
    before = marker.stat().st_mtime_ns
    pipeline.run(tmp_path)
    assert marker.stat().st_mtime_ns == before
