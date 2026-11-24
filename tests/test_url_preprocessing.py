# Tests: check url preprocessing behavior using saved artifacts or temporary inputs.

import pandas as pd
import pytest
from src.preprocessing.urls import normalize_url, ingest, run, completed


# Test stage: normalization preserves lexical evidence.
def test_normalization_preserves_lexical_evidence():
    original = ' HTTPS://User:Pass@ExAmPlE.com:443/a%2Fb?b=2&a=1#Login '
    record = normalize_url(original)
    assert record['url_original'] == original
    assert record['url'] == 'https://User:Pass@example.com:443/a%2Fb?b=2&a=1#Login'
    assert (
        record['url_key']
        == normalize_url('https://User:Pass@example.com/a%2Fb?b=2&a=1#Other')['url_key']
    )
    assert normalize_url('https://example.com?')['url'].endswith('?')
    assert normalize_url('https://example.com/%zz')['has_invalid_percent_escape']
    for value in (
        '',
        'javascript:alert(1)',
        'https://bad host.test',
        'http://example.com:99999',
        'http://-bad.test',
    ):
        with pytest.raises(ValueError):
            normalize_url(value)


# Test stage: label conflicts and mapping.
def test_label_conflicts_and_mapping(tmp_path):
    path = tmp_path / 'input.csv'
    pd.DataFrame(
        {
            'URL': [
                'https://a.test',
                'https://a.test/#x',
                'https://b.test',
                'https://b.test/',
                'https://c.test',
                'bad',
            ],
            'label': [0, 1, 1, 1, 0, 0],
        }
    ).to_csv(path, index=False)
    frame, stats = ingest(path, 'phiusiil')
    assert frame.label.tolist() == [0, 1]
    assert stats['counts']['conflicting_keys'] == 1
    assert stats['removed']['exact_or_canonical_duplicate'] == 1


# Test stage: smoke and skip.
def test_smoke_and_skip(tmp_path):
    raw = tmp_path / 'data/raw_data/urls'
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            'URL': [f'https://host{i}.test/path?x={i}' for i in range(120)],
            'label': [i % 2 for i in range(120)],
            'URLLength': [30] * 120,
            'Domain': [f'host{i}.test' for i in range(120)],
        }
    ).to_csv(raw / 'primary.csv', index=False)
    pd.DataFrame(
        {
            'url': ['https://outside.test/login', 'https://host0.test/another'],
            'phish_id': ['1', '2'],
            'verified': ['yes', 'yes'],
            'online': ['yes', 'yes'],
            'submission_time': ['2026-01-01'] * 2,
        }
    ).to_csv(raw / 'external.csv', index=False)
    report = run(tmp_path)
    assert completed(tmp_path)
    assert report['validation']['verified_records'] == 120
    assert report['validation']['external_records'] == 1
    target = tmp_path / 'data/processed/urls/urls_cleaned.parquet'
    before = target.stat().st_mtime_ns
    assert run(tmp_path) == report
    assert target.stat().st_mtime_ns == before
