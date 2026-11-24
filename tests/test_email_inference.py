# Tests: check email inference behavior using saved artifacts or temporary inputs.

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler
from src.inference import email
from src.inference.predict import Predictor, main
from src.preprocessing.emails import clean, urls
from packaging_checks import block_network, check_bundle


# Test stage: input validation.
def test_input_validation():
    for subject, body in [
        ('', ''),
        ('', '  '),
        ('', '<html><script>alert(1)</script></html>'),
        ('', None),
    ]:
        with pytest.raises(ValueError):
            email.prepare(subject, body)


# Test stage: frozen parity no fit no network.
def test_frozen_parity_no_fit_no_network(monkeypatch):
    block_network(monkeypatch)

    # Test guard: fail if a forbidden training or data operation is called.
    def forbidden(*args, **kwargs):
        raise AssertionError('Inference must not fit')

    monkeypatch.setattr(TfidfVectorizer, 'fit', forbidden)
    monkeypatch.setattr(TfidfVectorizer, 'fit_transform', forbidden)
    monkeypatch.setattr(MaxAbsScaler, 'fit', forbidden)
    p = Predictor()
    paths = [
        p.root / 'models/email' / n for n in ['selected_model.joblib', 'selected_threshold.json']
    ]
    paths.append(p.root / 'models/feature_engineering/email_transformer.joblib')
    before = [f.stat().st_mtime_ns for f in paths]
    subject, body = (
        'Urgent verification',
        '<p>Verify account!!! <a href="https://example.test/login">click</a></p>',
    )
    expected = pd.DataFrame(
        [
            dict(
                text=clean(subject + '\n' + body),
                subject=clean(subject),
                url_count=len(urls(subject + '\n' + body)),
            )
        ]
    )
    pd.testing.assert_frame_equal(email.prepare(subject, body), expected)
    result = p.predict_email(subject, body)
    model, transformer, config = p._load('email')
    assert p._load('email')[0] is model
    assert result['score'] == float(model.decision_function(transformer.transform(expected))[0])
    assert result['score_type'] == 'uncalibrated_margin'
    assert result['threshold'] == config['threshold']
    assert result['prediction'] == (
        'PHISHING' if result['score'] >= config['threshold'] else 'LEGITIMATE'
    )
    assert before == [f.stat().st_mtime_ns for f in paths]


# Test stage: email cli.
def test_email_cli(capsys, monkeypatch):
    email.main(['--subject', 'Meeting', '--body', 'Project meeting tomorrow', '--json'])
    assert json.loads(capsys.readouterr().out)['score_type'] == 'uncalibrated_margin'
    answers = iter(['email', 'Meeting', 'Project meeting tomorrow', '.', 'quit'])
    monkeypatch.setattr('builtins.input', lambda *args: next(answers))
    main(['--interactive'])
    assert capsys.readouterr().out.count('Prediction:') == 1


# Test stage: independent artifact bundle.
def test_independent_artifact_bundle(tmp_path):
    check_bundle(
        tmp_path,
        'email',
        'linear_svm_none',
        ['email_transformer.joblib', 'emails_schema.json'],
        ['--body', 'Project meeting tomorrow'],
    )
