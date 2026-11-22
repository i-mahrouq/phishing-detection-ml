"""The supported CLI works offline with just source and pretrained artifacts."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import urllib.request

from src.inference.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]


def test_offline_predictions(monkeypatch):
    def offline(*args, **kwargs):
        raise AssertionError('Prediction must not make network requests')

    monkeypatch.setattr(socket, 'create_connection', offline)
    monkeypatch.setattr(socket, 'getaddrinfo', offline)
    monkeypatch.setattr(socket.socket, 'connect', offline)
    monkeypatch.setattr(urllib.request, 'urlopen', offline)
    predictor = Predictor(ROOT)
    assert predictor.predict_email('Meeting', 'Project meeting tomorrow')['prediction'] in (
        'PHISHING', 'LEGITIMATE')
    assert predictor.predict_url('https://example.com/')['prediction'] in (
        'PHISHING', 'LEGITIMATE')


def test_pretrained_cli_needs_no_datasets(tmp_path):
    # Each bundle has only one task's artifacts: neither requires the other model.
    for kind, model, args in [
        ('email', 'linear_svm_none', ('email', '--body', 'Project meeting tomorrow')),
        ('url', 'random_forest', ('url', '--url', 'https://example.com/')),
    ]:
        bundle = tmp_path / kind
        shutil.copytree(ROOT / 'src', bundle / 'src', ignore=shutil.ignore_patterns('__pycache__'))
        files = {
            kind: ['selected_model.joblib', model + '.joblib', 'selected_threshold.json'],
            'feature_engineering': [kind + 's_schema.json'],
        }
        if kind == 'email':
            files['feature_engineering'].append('email_transformer.joblib')
        for folder, names in files.items():
            target = bundle / 'models' / folder
            target.mkdir(parents=True)
            for name in names:
                shutil.copy2(ROOT / 'models' / folder / name, target / name, follow_symlinks=False)
        assert not (bundle / 'data').exists()
        result = subprocess.run(
            [sys.executable, '-m', 'src.inference.predict', *args, '--json'],
            cwd=bundle, capture_output=True, text=True, check=True,
        )
        assert json.loads(result.stdout)['prediction'] in ('PHISHING', 'LEGITIMATE')
