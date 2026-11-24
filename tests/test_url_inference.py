# Tests: check url inference behavior using saved artifacts or temporary inputs.

import json
import numpy as np
import pytest
from src.inference import url
from src.inference.predict import Predictor, main
from packaging_checks import block_network, check_bundle


# Test stage: input validation.
def test_input_validation():
    for value in ['', 'javascript:alert(1)', 'https://bad host.test', 'https://example.test:99999']:
        with pytest.raises(ValueError):
            url.prepare(value)


# Test stage: frozen parity no network.
def test_frozen_parity_no_network(monkeypatch):
    block_network(monkeypatch)
    p = Predictor()
    paths = [
        p.root / 'models/url' / n for n in ['selected_model.joblib', 'selected_threshold.json']
    ]
    before = [f.stat().st_mtime_ns for f in paths]
    result = p.predict_url('https://example.test/login?account=1')
    model, transformer, config = p._load('url')
    assert p._load('url')[0] is model
    assert np.isfinite(result['score']) and result['score_type'] == 'probability'
    assert result['threshold'] == config['threshold']
    assert result['prediction'] == (
        'PHISHING' if result['score'] >= config['threshold'] else 'LEGITIMATE'
    )
    assert before == [f.stat().st_mtime_ns for f in paths]


# Test stage: url cli.
def test_url_cli(capsys, monkeypatch):
    url.main(['--url', 'https://example.test/', '--json'])
    assert json.loads(capsys.readouterr().out)['score_type'] == 'probability'
    with pytest.raises(SystemExit) as error:
        url.main(['--url', 'bad'])
    assert error.value.code == 2 and 'Error:' in capsys.readouterr().err
    answers = iter(['url', 'https://example.test/', 'quit'])
    monkeypatch.setattr('builtins.input', lambda *args: next(answers))
    main(['--interactive'])
    assert capsys.readouterr().out.count('Prediction:') == 1


# Test stage: independent artifact bundle.
def test_independent_artifact_bundle(tmp_path):
    check_bundle(
        tmp_path, 'url', 'random_forest', ['urls_schema.json'], ['--url', 'https://example.com/']
    )
