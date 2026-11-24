# Packaging checks: verify inference without network access or the other task’s files.

import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]


# Test setup: reject external network calls during offline inference.
def block_network(monkeypatch):
    # Test guard: reject attempted network access.
    def offline(*args, **kwargs):
        raise AssertionError('Prediction must not make network requests')

    monkeypatch.setattr(socket, 'create_connection', offline)
    monkeypatch.setattr(socket, 'getaddrinfo', offline)
    monkeypatch.setattr(socket.socket, 'connect', offline)
    monkeypatch.setattr(urllib.request, 'urlopen', offline)


# Packaging check: run inference with only the requested task’s sources and artifacts.
def check_bundle(tmp_path, kind, model, schema_files, args):
    shutil.copytree(ROOT / 'src', tmp_path / 'src', ignore=shutil.ignore_patterns('__pycache__'))
    for folder, names in {
        kind: ['selected_model.joblib', model + '.joblib', 'selected_threshold.json'],
        'feature_engineering': schema_files,
    }.items():
        target = tmp_path / 'models' / folder
        target.mkdir(parents=True)
        for name in names:
            shutil.copy2(ROOT / 'models' / folder / name, target / name, follow_symlinks=False)
    assert not (tmp_path / 'data').exists()
    legacy = subprocess.run(
        [sys.executable, '-m', 'src.inference.predict', kind, *args, '--json'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(legacy.stdout)['prediction'] in ('PHISHING', 'LEGITIMATE')
    other = 'url' if kind == 'email' else 'email'

    for path in [
        f'inference/{other}.py',
        f'features/{other}_features.py',
        f'features/{other}_pipeline.py',
        f'training/train_{other}.py',
        f'evaluation/evaluate_{other}.py',
        f'preprocessing/{other}s.py',
    ]:
        (tmp_path / 'src' / path).unlink()
    command = subprocess.run(
        [sys.executable, '-m', 'src.inference.' + kind, *args, '--json'],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(command.stdout)['prediction'] in ('PHISHING', 'LEGITIMATE')
