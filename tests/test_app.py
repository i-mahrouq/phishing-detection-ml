"""Small UI and dataset-free packaging checks using the frozen V1 artifacts."""
import json
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import urllib.request

import pytest
from streamlit.testing.v1 import AppTest

from src.inference.predict import Predictor

ROOT = Path(__file__).resolve().parents[1]


def offline(*args, **kwargs):
    raise AssertionError('Prediction must not make network requests')


@pytest.fixture
def no_network(monkeypatch):
    monkeypatch.setattr(socket, 'create_connection', offline)
    monkeypatch.setattr(socket, 'getaddrinfo', offline)
    monkeypatch.setattr(socket.socket, 'connect', offline)
    monkeypatch.setattr(urllib.request, 'urlopen', offline)


def test_app_predictions_match_cli_and_no_network(no_network):
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    assert not app.exception
    subject, body, url = 'Meeting', 'Project meeting tomorrow at ten.', 'https://example.com/'
    app.text_input[0].set_value(subject)
    app.text_area[0].set_value(body)
    app.button[0].click().run()
    expected = Predictor(ROOT).predict_email(subject, body)
    labels = [x.value for x in app.success] + [x.value for x in app.error]
    assert f"Prediction: {expected['prediction'].title()}" in labels
    assert not app.exception
    app.text_input[1].set_value(url)
    app.button[1].click().run()
    expected = Predictor(ROOT).predict_url(url)
    labels = [x.value for x in app.success] + [x.value for x in app.error]
    assert f"Prediction: {expected['prediction'].title()}" in labels
    assert len(app.dataframe) == 1
    assert not app.exception
    # Same public entry point used by the installed CLI, in a fresh process.
    result = subprocess.run([sys.executable, '-m', 'src.inference.predict', 'url',
                             '--url', url, '--json'], cwd=ROOT, capture_output=True, text=True, check=True)
    assert json.loads(result.stdout) == expected


def test_empty_and_invalid_inputs(no_network):
    app = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=30).run()
    app.button[0].click().run()
    assert any('nonempty visible text' in x.value for x in app.error)
    app.button[1].click().run()
    assert any('nonempty string' in x.value for x in app.error)
    app.text_input[1].set_value('javascript:alert(1)')
    app.button[1].click().run()
    assert len(app.error) and not app.exception


def test_git_selected_copy_needs_no_datasets(tmp_path, no_network):
    """Copy only Git's index, preserving relative symlinks, then run both clients."""
    names = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    assert 'app.py' in names, 'Stage the reviewed application before checking clone packaging'
    for name in filter(None, names):
        source, target = ROOT / name, tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target, follow_symlinks=False)
    assert all(p.suffix == '.md' for p in (tmp_path / 'data').rglob('*') if p.is_file())
    for args in [('email', '--body', 'Project meeting tomorrow'), ('url', '--url', 'https://example.com/')]:
        result = subprocess.run([sys.executable, '-m', 'src.inference.predict', *args, '--json'],
                                cwd=tmp_path, capture_output=True, text=True, check=True)
        assert json.loads(result.stdout)['prediction'] in ('PHISHING', 'LEGITIMATE')
    app = AppTest.from_file(str(tmp_path / 'app.py'), default_timeout=30).run()
    app.text_area[0].set_value('Project meeting tomorrow')
    app.button[0].click().run()
    assert not app.exception and len(app.success) + len(app.error) == 1
    app.text_input[1].set_value('https://example.com/')
    app.button[1].click().run()
    assert not app.exception and len(app.success) + len(app.error) == 1
    assert all('Cannot analyze' not in x.value for x in app.error)
